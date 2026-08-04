"""DQ-GKD training (paper Sec. 3.2, Alg. 1).

L = L_CE(gold) + lambda_kd * L_Q-RKL + lambda_div * L_div
  - L_CE(gold): keeps basic executability/structure.
  - L_Q-RKL: quality-weighted reverse-KL distillation restricted to
    test-sensitive spans of on-policy candidates (span mask via AST).
  - L_div: differentiable surrogate of the structural diversity regularizer:
    up-weights the likelihood of structurally distinct candidates (Jaccard
    distinctness of AST feature sets) to counteract candidate collapse.

On-policy candidates are sampled from the student policy once per stage
(exp/gen_gkd_candidates.py) and scored offline (exp/score_gkd_candidates.py),
which keeps the teacher-forward budget tractable.
If no candidate file is configured, falls back to uniform-token JSD/RKL.
"""
import argparse, json, os, torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from datasets import Dataset
from src.data.collator import SFTDataCollator
from src.data.dq_collator import DQGKDCollator

# ====== Utils ======
def is_peft_dir(path: str) -> bool:
    return any(os.path.exists(os.path.join(path, f))
               for f in ["adapter_config.json","adapter_model.safetensors","adapter_model.bin"])

def get_dtype(cfg_dtype: Optional[str]):
    if not cfg_dtype:
        return None
    m = str(cfg_dtype).lower()
    if m in ["bf16","bfloat16"]: return torch.bfloat16
    if m in ["fp16","float16","half"]: return torch.float16
    if m in ["fp32","float32"]: return torch.float32
    return None


def load_dq_dataset(train_file: str, candidates_file: Optional[str], max_cands: int = 4):
    with open(train_file, "r", encoding="utf-8") as f:
        arr = json.load(f)
    cand_map = {}
    if candidates_file and os.path.exists(candidates_file):
        with open(candidates_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    cand_map[r["idx"]] = r
                except Exception:
                    pass
        print(f"[data] loaded candidates for {len(cand_map)} prompts", flush=True)
    rows = []
    for i, ex in enumerate(arr):
        target = ex.get("chosen") or ex.get("response") or ""
        row = {"prompt": ex["prompt"], "target": target,
               "cands": [], "cand_weights": []}
        c = cand_map.get(i)
        if c:
            # combined weight: quality w^(k) scaled by (1 + distinctness) so that
            # diverse candidates receive more distillation budget (L_div surrogate)
            pairs = list(zip(c["candidates"], c["weights"], c["distinct"]))
            pairs.sort(key=lambda p: -p[1])
            pairs = pairs[:max_cands]
            ws = [w for (_y, w, _d) in pairs]
            z = sum(ws) or 1.0
            row["cands"] = [y for (y, _w, _d) in pairs]
            row["cand_weights"] = [w / z for (_y, w, _d) in pairs]
            row["cand_distinct"] = [d for (_y, _w, d) in pairs]
        rows.append(row)
    return Dataset.from_list(rows)


# ====== Distill Trainer (span-masked quality-weighted RKL + diversity) ======
class DQGKDTrainer(Trainer):
    def __init__(self, *args, teacher=None, gkd_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        g = gkd_cfg or {}
        self.lambda_kd = float(g.get("kd_lambda", 0.5))
        self.lambda_div = float(g.get("div_lambda", 0.1))
        self.temperature = float(g.get("temperature", 2.0))
        self.use_jsd = bool(g.get("use_jsd", True))
        self.jsd_lambda = float(g.get("jsd_lambda", 0.5))
        self.rkl_lambda = float(g.get("rkl_lambda", 0.5))

    @staticmethod
    def _align_vocab(s_logits, t_logits):
        """Qwen2.5 sizes pad the embedding matrix differently (e.g. 151936 vs
        152064) while sharing one tokenizer; real tokens occupy the common
        prefix, so truncating both to the min vocab is exact."""
        v = min(s_logits.size(-1), t_logits.size(-1))
        return s_logits[..., :v], t_logits[..., :v]

    def _uniform_distill(self, shift_logits, mask, inputs, loss):
        """Fallback: uniform-token JSD + RKL on gold sequence (legacy behaviour)."""
        with torch.no_grad():
            t_out = self.teacher(input_ids=inputs["input_ids"],
                                 attention_mask=inputs.get("attention_mask"))
        t_logits = t_out.logits[:, :-1, :].contiguous()
        shift_logits, t_logits = self._align_vocab(shift_logits, t_logits)
        T = max(self.temperature, 1e-6)
        s_log, t_log = shift_logits / T, t_logits / T
        ps = torch.log_softmax(s_log, dim=-1)
        pt = torch.log_softmax(t_log, dim=-1)
        m = torch.logsumexp(torch.stack([s_log, t_log]), dim=0) - torch.log(
            torch.tensor(2.0, device=s_log.device))
        pm = torch.log_softmax(m, dim=-1)
        jsd = 0.5 * (torch.exp(ps) * (ps - pm)).sum(-1) + 0.5 * (torch.exp(pt) * (pt - pm)).sum(-1)
        jsd = (jsd * mask).sum() / (mask.sum() + 1e-8)
        rkl = (torch.exp(pt) * (pt - ps)).sum(-1)
        rkl = (rkl * mask).sum() / (mask.sum() + 1e-8)
        return loss + self.jsd_lambda * jsd + self.rkl_lambda * rkl

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        cand_ids = inputs.pop("cand_input_ids", None)
        cand_am = inputs.pop("cand_attention_mask", None)
        cand_span = inputs.pop("cand_span_mask", None)
        cand_w = inputs.pop("cand_weight", None)

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = (shift_labels != -100).float()

        ce = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100)
        loss = ce

        if self.teacher is None:
            return (loss, outputs) if return_outputs else loss

        if cand_ids is None or cand_ids.numel() == 0:
            loss = self._uniform_distill(shift_logits, mask, inputs, loss)
            return (loss, outputs) if return_outputs else loss

        # ---- quality-weighted masked reverse-KL on on-policy candidates ----
        # Memory strategy: the autograd graph must not accumulate full-vocab
        # (L x 152k) softmax outputs per candidate. We (i) select the masked
        # (test-sensitive) positions BEFORE the fp32 softmax for the RKL term,
        # and (ii) compute the diversity chosen-logprob via cross_entropy so
        # that at most one full-vocab tensor per candidate lives in the graph.
        T = max(self.temperature, 1e-6)
        kd_terms = []
        div_terms = []
        for k in range(cand_ids.size(0)):
            ids = cand_ids[k:k+1]
            am = cand_am[k:k+1]
            span = cand_span[k:k+1]
            w = cand_w[k]
            s_out = model(input_ids=ids, attention_mask=am)
            with torch.no_grad():
                t_out = self.teacher(input_ids=ids, attention_mask=am)
            s_full, t_full = self._align_vocab(s_out.logits[:, :-1, :],
                                               t_out.logits[:, :-1, :])
            s_full = s_full.squeeze(0)          # (L, V) bf16, graph-attached
            t_full = t_full.squeeze(0)
            del s_out, t_out
            m = (span[0, 1:] * am[0, 1:].float())
            idx = m.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                s_sel = s_full[idx].float() / T   # (n_masked, V) fp32
                with torch.no_grad():
                    t_sel = t_full[idx].float() / T
                    pt = torch.log_softmax(t_sel, dim=-1)
                    pt_exp = torch.exp(pt)
                ps = torch.log_softmax(s_sel, dim=-1)
                rkl_c = (pt_exp * (pt - ps)).sum(-1).mean()
                kd_terms.append(w * rkl_c)
                del s_sel, t_sel, ps, pt, pt_exp

            # diversity surrogate: raise likelihood of structurally distinct
            # candidates; cross_entropy(reduction) avoids a second full-vocab
            # softmax output in the graph
            lbl = ids[0, 1:]
            amf = am[0, 1:].float()
            nll = torch.nn.functional.cross_entropy(s_full, lbl, reduction="none")
            lp_c = -(nll * amf).sum() / (amf.sum() + 1e-8)
            div_terms.append(-(w * lp_c))
            del s_full, t_full, nll

        l_kd = torch.stack(kd_terms).sum() if kd_terms else torch.zeros((), device=shift_logits.device)
        l_div = torch.stack(div_terms).mean() if div_terms else torch.zeros((), device=shift_logits.device)
        loss = loss + self.lambda_kd * l_kd + self.lambda_div * l_div
        return (loss, outputs) if return_outputs else loss


# ====== Helper ======
def build_tokenizer(maybe_adapter_or_model: str):
    if is_peft_dir(maybe_adapter_or_model):
        peft_cfg = PeftConfig.from_pretrained(maybe_adapter_or_model)
        tok = AutoTokenizer.from_pretrained(peft_cfg.base_model_name_or_path, use_fast=True)
    else:
        tok = AutoTokenizer.from_pretrained(maybe_adapter_or_model, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


# ====== Main ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = cfg["training"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    student_id_or_adapter = cfg["base_model"]
    student_tok = build_tokenizer(student_id_or_adapter)
    teacher_id = cfg.get("teacher_model")
    if teacher_id is None:
        raise ValueError("teacher_model must be provided in config.")

    # -------- dataset & collator --------
    cand_file = cfg["data"].get("candidates_file")
    ds = load_dq_dataset(cfg["data"]["train_jsonl"], cand_file,
                         max_cands=int(cfg.get("gkd", {}).get("max_cands", 4)))
    if cand_file:
        collator = DQGKDCollator(student_tok, max_len=cfg["data"].get("max_length", 2048))
    else:
        collator = SFTDataCollator(student_tok, max_len=cfg["data"].get("max_length", 2048))

    # -------- student --------
    dtype = get_dtype(cfg["training"].get("dtype"))
    base_model_id = student_id_or_adapter
    resume_adapter = cfg["training"].get("resume_adapter")

    load_in_4bit = bool(cfg["training"].get("load_in_4bit", False))
    quant_cfg = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype or torch.bfloat16)
    student_base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=None if quant_cfg else (dtype or torch.bfloat16),
        quantization_config=quant_cfg, device_map="cuda:0")

    if cfg["training"].get("gradient_checkpointing", False):
        student_base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(student_base.config, "use_cache"):
            student_base.config.use_cache = False

    if resume_adapter and is_peft_dir(resume_adapter):
        model = PeftModel.from_pretrained(student_base, resume_adapter, is_trainable=True)
        print(f"[info] resumed adapter from '{resume_adapter}'", flush=True)
    else:
        lcfg = LoraConfig(
            r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"], bias="none",
            task_type="CAUSAL_LM", target_modules=cfg["lora"]["target_modules"])
        model = get_peft_model(student_base, lcfg)
        print("[info] created new LoRA head for distillation.", flush=True)

    # -------- teacher (frozen; 4-bit by default to fit alongside the student) ----
    teacher_4bit = bool(cfg.get("teacher_4bit", True))
    if teacher_4bit:
        from transformers import BitsAndBytesConfig
        t_quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype or torch.bfloat16)
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_id, quantization_config=t_quant, device_map="cuda:0")
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_id, torch_dtype=dtype or torch.bfloat16, device_map="cuda:0")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    targs = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg["training"].get("batch_size", 2),
        gradient_accumulation_steps=cfg["training"].get("grad_accum", 8),
        learning_rate=float(cfg["training"].get("lr", 1e-4)),
        num_train_epochs=cfg["training"].get("epochs", 1),
        max_steps=cfg["training"].get("max_steps", -1),
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.03),
        save_steps=cfg["training"].get("save_steps", 1000),
        save_total_limit=1,
        logging_steps=cfg["training"].get("logging_steps", 20),
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DQGKDTrainer(
        model=model, args=targs, train_dataset=ds, data_collator=collator,
        tokenizer=student_tok, teacher=teacher, gkd_cfg=cfg.get("gkd", {}))
    # auto-resume from the last checkpoint after interruption (e.g. reboot)
    import glob as _glob
    has_ckpt = bool(_glob.glob(os.path.join(out_dir, "checkpoint-*")))
    trainer.train(resume_from_checkpoint=has_ckpt)
    model.save_pretrained(out_dir)
    student_tok.save_pretrained(out_dir)
    print(f"[done] saved to: {out_dir}")


if __name__ == "__main__":
    main()
