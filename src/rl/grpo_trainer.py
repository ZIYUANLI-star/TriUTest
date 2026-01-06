import os, json
import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from src.losses.gkd import jsd_loss
from torch.nn.utils import clip_grad_norm_

@dataclass
class RolloutConfig:
    group_size: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512
import re

def truncate_after_last_assert(code: str) -> str:
    """
    在整段 code 里找到“最后一个以 assert 开头的语句”，
    并把该行及其之后的所有内容删除。
    如果不存在 assert，则原样返回。
    """
    last_pos = None
    # 匹配行首（可有缩进）的 assert
    pattern = re.compile(r'^[ \t]*assert\b', re.MULTILINE)
    for m in pattern.finditer(code):
        last_pos = m.start()

    if last_pos is None:
        # 没有 assert，直接返回原内容
        return code

    # 保留最后一个 assert 之前的所有内容
    return code[:last_pos].rstrip()

class GRPOTrainer:
    def __init__(self, policy, ref, tokenizer, optim, cfg, gkd_online=None, device="cuda"):
        self.policy = policy
        self.ref = ref
        self.tokenizer = tokenizer
        self.optim = optim
        self.cfg = cfg
        self.device = device
        self.beta = cfg["kl"]["beta"]                    # KL 系数
        self.gkd_online = gkd_online or {"enable": False}

        # 调试配置
        dbg = cfg.get("debug", {}) or {}
        self.dbg_print_candidates = bool(dbg.get("print_candidates", False))
        self.dbg_print_rewards    = bool(dbg.get("print_rewards", False))
        self.dbg_dump_jsonl       = bool(dbg.get("dump_jsonl", False))
        self.dbg_print_every      = int(dbg.get("print_every", 1))
        self.debug_path = os.path.join(cfg["output_dir"], "debug_generations.jsonl")

    # --------- 小工具：字符串截断用于打印 ---------
    def _short(self, s: str, n: int = 300) -> str:
        s = (s or "").replace("\r\n", "\n").strip()
        return (s[:n] + "…") if len(s) > n else s

    # ========== 生成候选 ==========
    @torch.no_grad()
    def generate_group(self, prompts: List[str], ro_cfg: RolloutConfig):
        outs = []

        # —— 记录与切换推理状态 ——
        was_training = self.policy.training
        gc_was_on = getattr(self.policy, "is_gradient_checkpointing", False)
        orig_use_cache = getattr(self.policy.config, "use_cache", True)

        if gc_was_on and hasattr(self.policy, "gradient_checkpointing_disable"):
            self.policy.gradient_checkpointing_disable()
        if hasattr(self.policy.config, "use_cache"):
            self.policy.config.use_cache = True
        self.policy.eval()

        try:
            # 采样防复读的可选项来自 cfg.rollout
            ro_cfg_yaml = self.cfg.get("rollout", {}) or {}
            no_repeat_ngram_size = ro_cfg_yaml.get("no_repeat_ngram_size", None)
            repetition_penalty = ro_cfg_yaml.get("repetition_penalty", None)
            eos_token_id = ro_cfg_yaml.get("eos_token_id", None)

            for p in prompts:
                group = []
                for _ in range(ro_cfg.group_size):
                    ip = self.tokenizer(
                        p,
                        return_tensors="pt",
                        truncation=True,
                        add_special_tokens=True
                    ).to(self.device)

                    gen_kwargs = dict(
                        **ip,
                        do_sample=True,
                        temperature=ro_cfg.temperature,
                        top_p=ro_cfg.top_p,
                        max_new_tokens=ro_cfg.max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    if no_repeat_ngram_size is not None:
                        gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
                    if repetition_penalty is not None:
                        gen_kwargs["repetition_penalty"] = float(repetition_penalty)
                    if eos_token_id is not None:
                        gen_kwargs["eos_token_id"] = int(eos_token_id)

                    gen = self.policy.generate(**gen_kwargs)

                    # 仅保留新增段
                    prompt_len = ip["input_ids"].size(1)
                    ans_ids = gen[0][prompt_len:]
                    ans = self.tokenizer.decode(ans_ids, skip_special_tokens=True).strip()
                    ans = truncate_after_last_assert(ans)
                    group.append(ans)
                outs.append(group)
        finally:
            # —— 恢复训练状态 ——
            if hasattr(self.policy.config, "use_cache"):
                self.policy.config.use_cache = orig_use_cache
            if gc_was_on and hasattr(self.policy, "gradient_checkpointing_enable"):
                self.policy.gradient_checkpointing_enable()
                if hasattr(self.policy.config, "use_cache"):
                    self.policy.config.use_cache = False
            if was_training:
                self.policy.train()

        return outs

    # ========== 计算在“生成路径”上的 per-sample 平均 logprob ==========

    def _mean_logprob_on_generated(self, model, prompt_ids: torch.Tensor, gen_ids: torch.Tensor) -> torch.Tensor:
        """
        返回形状 [B] 的张量：每个样本在“生成段（gen_ids）”上的平均 logprob。
        对齐方式：shift 对齐，logits[:, :-1] 对 labels[:, 1:].
        """
        ids = torch.cat([prompt_ids, gen_ids], dim=1)               # [B, Lp+Lg]
        # 优先使用 tokenizer 的 mask，如果没有就用全 1
        am_p = None
        am_g = None
        # 由于上游传进来的只是 ids，这里统一用全 1；如果你保留了各自的 attention_mask，可在调用处一起传入
        am = torch.ones_like(ids)

        out = model(input_ids=ids, attention_mask=am)
        logits = out.logits                                         # [B, Lp+Lg, V]

        # shift 对齐
        shift_logits = logits[:, :-1, :]                            # predict token t+1 at position t
        shift_labels = ids[:, 1:]                                   # gold labels

        B = ids.size(0)
        Lp = prompt_ids.size(1)
        Lg = gen_ids.size(1)

        # 仅取生成段对应的预测与标签
        gen_shift_logits = shift_logits[:, Lp-1: Lp+Lg-1, :]        # 对应标签区间 [Lp : Lp+Lg)
        gen_shift_labels = shift_labels[:, Lp:   Lp+Lg]
        # 断言长度匹配，防隐藏对齐错误
        assert gen_shift_logits.size(1) == Lg, f"gen len mismatch: logits_len={gen_shift_logits.size(1)} vs Lg={Lg}"

        logprobs = torch.log_softmax(gen_shift_logits, dim=-1)      # [B, Lg, V]
        lp_chosen = logprobs.gather(-1, gen_shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, Lg]
        return lp_chosen.mean(dim=1)                                # [B]

    # ========== 单步训练 ==========
    def step(self, batch_prompts: List[Dict[str,Any]], ro_cfg: RolloutConfig, reward_fn, step_idx: int = None):
        # 训练期：开 GC / 关 cache
        self.policy.train()
        if hasattr(self.policy, "gradient_checkpointing_enable"):
            self.policy.gradient_checkpointing_enable()
        if hasattr(self.policy.config, "use_cache"):
            self.policy.config.use_cache = False

        # 1) rollout
        prompts = [bp["prompt"] for bp in batch_prompts]
        groups = self.generate_group(prompts, ro_cfg)  # List[List[str]]

        # —— 可选打印候选 ——
        if self.dbg_print_candidates and (step_idx is None or (step_idx + 1) % self.dbg_print_every == 0):
            tag = f"[step {step_idx}]" if step_idx is not None else "[step ?]"
            for i, (p, cands) in enumerate(zip(prompts, groups)):
                print(f"{tag} prompt[{i}]: {p}", flush=True)
                for j, t in enumerate(cands):
                    print(f"{tag}   cand[{j}]: {t}", flush=True)

        # 2) 奖励与择优
        rewards_all = []
        chosen_texts = []
        for prompt, cands, bp in zip(prompts, groups, batch_prompts):
            cut_code = bp["cut_code"]
            group_rewards = []
            for t in cands:
                r = reward_fn(cut_code, t)             # 外部奖励：composite_reward
                # 防止 None/NaN
                if r is None or (isinstance(r, float) and (r != r)):  # NaN 检查
                    r = 0.0
                group_rewards.append(float(r))
            rewards_all.append(group_rewards)
            best_idx = max(range(len(group_rewards)), key=lambda i: group_rewards[i])
            chosen_texts.append(cands[best_idx])

        # —— 可选打印奖励 ——
        if self.dbg_print_rewards and (step_idx is None or (step_idx + 1) % self.dbg_print_every == 0):
            tag = f"[step {step_idx}]" if step_idx is not None else "[step ?]"
            for i, rs in enumerate(rewards_all):
                best = max(range(len(rs)), key=lambda k: rs[k])
                print(f"{tag} rewards[{i}]: " + ", ".join(f"{r:.3f}" for r in rs) + f"  (best=cand[{best}])", flush=True)

        # —— 可选 JSONL 落盘 ——
        if self.dbg_dump_jsonl:
            rec = {"step": step_idx, "items": []}
            for i, (bp, cands, rs) in enumerate(zip(batch_prompts, groups, rewards_all)):
                mean_r = sum(rs) / max(1, len(rs))
                advs = [r - mean_r for r in rs]
                best = max(range(len(rs)), key=lambda k: rs[k])
                rec["items"].append({
                    "idx": i,
                    "prompt": bp["prompt"],
                    "cut_code": bp.get("cut_code", ""),
                    "candidates": cands,
                    "rewards": rs,
                    "advantages": advs,
                    "best_idx": best
                })
            os.makedirs(os.path.dirname(self.debug_path), exist_ok=True)
            with open(self.debug_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 3) 计算组内优势
        advantages = []
        for g in rewards_all:
            mean = sum(g)/len(g)
            advantages.append([ri - mean for ri in g])

        # 4) 策略梯度 + KL + 可选 GKD
        losses = []
        self.optim.zero_grad()

        for p, cands, advs in zip(prompts, groups, advantages):
            for cand, a in zip(cands, advs):
                # 构造 (prompt, cand) 的 id
                ip = self.tokenizer(
                    p,
                    return_tensors="pt",
                    truncation=True,
                    add_special_tokens=True
                ).to(self.device)
                tg = self.tokenizer(
                    cand,
                    return_tensors="pt",
                    truncation=True,
                    add_special_tokens=False
                ).to(self.device)

                prompt_ids = ip["input_ids"]    # [1, Lp]
                gen_ids    = tg["input_ids"]    # [1, Lg]

                # ---- 策略梯度：只在生成段计算 logprob，并乘以 advantage ----
                a_t = torch.tensor(float(a), device=self.device).detach()
                pol_mean_logp = self._mean_logprob_on_generated(self.policy, prompt_ids, gen_ids)   # [1]
                pg_loss = - a_t * pol_mean_logp.mean()   # maximize advantage => minimize -adv*logp

                # ---- KL 正则：同一路径上 policy/ref 的 logprob 差（clamp 防止数值爆）----
                with torch.no_grad():
                    ref_mean_logp = self._mean_logprob_on_generated(self.ref, prompt_ids, gen_ids)  # [1]
                kl_term = (pol_mean_logp - ref_mean_logp).mean()
                kl_term = torch.clamp(kl_term, min=-1.0, max=1.0)
                loss = pg_loss + self.beta * kl_term

                # ---- 可选：在线蒸馏（与 ref 的 JSD）----
                if self.gkd_online.get("enable", False) and self.ref is not None:
                    ids = torch.cat([prompt_ids, gen_ids], dim=1)
                    # mask：优先拼接各自的 attention_mask
                    am_p = ip.get("attention_mask", None)
                    am_g = tg.get("attention_mask", None)
                    if am_p is not None and am_g is not None:
                        am = torch.cat([am_p, am_g], dim=1)
                    else:
                        am = torch.ones_like(ids)

                    pol_out = self.policy(input_ids=ids, attention_mask=am)
                    with torch.no_grad():
                        ref_out = self.ref(input_ids=ids, attention_mask=am)
                    pol_logits = pol_out.logits[:, :-1, :]
                    ref_logits = ref_out.logits[:, :-1, :]
                    loss = loss + self.gkd_online["lambda"] * jsd_loss(
                        pol_logits, ref_logits, alpha=self.gkd_online.get("jsd_alpha", 0.9)
                    )

                losses.append(loss)

        total = torch.stack(losses).mean()
        total.backward()
        clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optim.step()

        return float(total.detach().cpu().item()), rewards_all, chosen_texts
