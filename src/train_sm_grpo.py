# filename: src/train_grpo.py
import argparse, yaml, os, re, signal, json, random, sys
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    PeftModel, PeftConfig, LoraConfig, get_peft_model,
    prepare_model_for_kbit_training
)

from src.rl.grpo_trainer import GRPOTrainer, RolloutConfig
from src.reward.rewarders import composite_reward
from torch.optim import AdamW


# ---------------------- utils ----------------------
def set_env_and_seed(seed: int = 42):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def allow_tf32(enable: bool = True):
    torch.backends.cuda.matmul.allow_tf32 = enable
    torch.backends.cudnn.allow_tf32 = enable

# 简单通过文件名判断 path 是「LoRA 适配器目录」还是「完整模型目录」。
def is_peft_adapter(path: str) -> bool:
    return any(
        os.path.exists(os.path.join(path, f))
        for f in [
            "adapter_config.json", "adapter_model.bin", "adapter_model.safetensors",
            "adapter_config.cfg"
        ]
    )

# 若传入的是 adapter 目录，就从 adapter 的 base_model_name_or_path 取基座来加载 tokenizer；否则按普通模型 id 加载。
def load_tokenizer_from_base_or_adapter(model_or_adapter_path: str) -> AutoTokenizer:
    try:
        if is_peft_adapter(model_or_adapter_path):
            base_id = PeftConfig.from_pretrained(model_or_adapter_path).base_model_name_or_path
            tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
        else:
            tok = AutoTokenizer.from_pretrained(model_or_adapter_path, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_or_adapter_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def _single_device_map(device_str: str) -> Dict[str, Union[int, str, torch.device]]:
    # 让整个模型落在同一个 device（避免 auto 分片）
    return {"": device_str}

# 加载policy model
def load_peft_or_full_as_policy(
    path: str,
    quant_config: Optional[BitsAndBytesConfig],
    dtype: Optional[torch.dtype],
    device_map: Union[str, Dict[str, Union[int, str, torch.device]]],
    lora_cfg: Optional[LoraConfig] = None,
    enable_gc: bool = False,
) -> AutoModelForCausalLM:
    is_4bit = quant_config is not None and getattr(quant_config, "load_in_4bit", False)

    if is_peft_adapter(path):
        base_id = PeftConfig.from_pretrained(path).base_model_name_or_path
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            device_map=device_map,
            torch_dtype=dtype if not is_4bit else None,
            quantization_config=quant_config,
        )
        if is_4bit:
            base = prepare_model_for_kbit_training(base)
        model = PeftModel.from_pretrained(base, path, is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=device_map,
            torch_dtype=dtype if not is_4bit else None,
            quantization_config=quant_config,
        )
        if is_4bit:
            model = prepare_model_for_kbit_training(model)
        if lora_cfg is not None:
            model = get_peft_model(model, lora_cfg)
        else:
            # 4-bit 且未注入 LoRA -> 没有可训练参数
            num_trainable = sum(p.requires_grad for p in model.parameters())
            if num_trainable == 0:
                raise ValueError(
                    "Policy is 4-bit quantized without LoRA. No trainable params.\n"
                    "Please provide a `lora` config in YAML to inject LoRA, or set training.use_4bit=false."
                )

    if enable_gc:
        # 训练期必须关 cache
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        # ✅ 使用 non-reentrant 变体，避免 “None of the inputs have requires_grad=True” 警告
        #   兼容老版本 transformers/torch：不支持该参数时回退到旧行为
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("[gc] gradient checkpointing enabled (use_reentrant=False)", flush=True)
        except TypeError:
            # 老版本回退：可能继续看到警告，但不影响训练；建议后续升级
            model.gradient_checkpointing_enable()
            print("[gc] gradient checkpointing enabled (legacy reentrant=True); "
                  "consider upgrading torch/transformers to support non-reentrant", flush=True)

    try:
        model.print_trainable_parameters()
    except Exception:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[info] trainable params: {trainable} || all params: {total} || trainable%: {trainable/total*100:.4f}", flush=True)

    return model

# 加载ref model
def load_peft_or_full_as_ref(
    path: str,
    quant_config: Optional[BitsAndBytesConfig],
    dtype: Optional[torch.dtype],
    device_map: Union[str, Dict[str, Union[int, str, torch.device]]],
) -> AutoModelForCausalLM:
    if is_peft_adapter(path):
        base_id = PeftConfig.from_pretrained(path).base_model_name_or_path
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            device_map=device_map,
            torch_dtype=dtype if quant_config is None else None,
            quantization_config=quant_config,
        )
        ref = PeftModel.from_pretrained(base, path)
    else:
        ref = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=device_map,
            torch_dtype=dtype if quant_config is None else None,
            quantization_config=quant_config,
        )
    for p in ref.parameters():
        p.requires_grad_(False)
    if hasattr(ref.config, "use_cache"):
        ref.config.use_cache = True  # 参考模型前向可开 cache，不参与反传
    ref.eval()
    return ref

# 用来从 Prompt 里抓 “Code Under Test:” 后的代码
CUT_PATTERNS = [
    # 有围栏：吃到下一个 ```
    r"Code Under Test:\s*```(?:python)?\s*([\s\S]*?)```",
    # 无围栏：吃到下一个分节标题（Chosen/Rejected/Response…）或串尾
    r"Code Under Test:\s*([\s\S]*?)(?:\n{2,}(?:Chosen|Rejected|Response|Test|Ground\s*Truth|---|###)|\Z)",
    # 兜底：从下一行一路到串尾
    r"Code Under Test:\s*\n([\s\S]+)\Z",
]


def extract_cut(prompt: str, fallback: str = "") -> str:
    for pat in CUT_PATTERNS:
        m = re.search(pat, prompt, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if m:
            code = m.group(1).strip()
            # 保险：把后续“Function Description”误抓进来时切掉
            code = re.sub(r"\n\n*Function Description:.*", "", code, flags=re.S | re.I)
            return code
    return fallback.strip()

def build_bnB_4bit() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

def pick_target_device_from_cfg(cfg: Dict[str, Any]) -> str:
    # 优先读取 cfg["device"] 或 cfg["training"]["device"]，可写 "cuda:7" / "cuda:0" / "cpu"
    dev = cfg.get("device") or cfg.get("training", {}).get("device")
    if isinstance(dev, str) and dev:
        return dev

    # 其次读取 device_index
    dev_idx = cfg.get("device_index") or cfg.get("training", {}).get("device_index")
    if dev_idx is not None and torch.cuda.is_available():
        return f"cuda:{int(dev_idx)}"

    # 再次：若设置了 CUDA_VISIBLE_DEVICES -> 选逻辑 0（即第一张可见卡）
    if torch.cuda.is_available():
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            return "cuda:0"
        return "cuda:0"

    return "cpu"


# ---------------------- main ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_file", default="data/your_dataset_with_id.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    set_env_and_seed(cfg.get("seed", 42))
    allow_tf32(True)

    out_dir = cfg["output_dir"]; os.makedirs(out_dir, exist_ok=True)

    # ---- 选择一个统一的目标设备（避免 auto 分片）----
    target_device: str = pick_target_device_from_cfg(cfg)
    # 可选：把默认 CUDA 设备也指到这块卡（防止某些第三方库用默认设备）
    if target_device.startswith("cuda"):
        try:
            torch.cuda.set_device(int(target_device.split(":")[1]))
        except Exception:
            pass
    print(f"[device] using target_device = {target_device}", flush=True)
    single_device_map = _single_device_map(target_device)

    # ---- 精度/量化开关 ----
    use_4bit = bool(cfg.get("training", {}).get("use_4bit", True))
    use_bf16 = bool(cfg.get("training", {}).get("bf16", False))
    use_fp16 = bool(cfg.get("training", {}).get("fp16", False))
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
    quant_cfg = build_bnB_4bit() if use_4bit else None

    # ---- Tokenizer ----
    tokenizer = load_tokenizer_from_base_or_adapter(cfg["policy_model"])

    # ---- policy / ref 加载（都放同一设备） ----
    lora_cfg = LoraConfig(
        r=cfg.get("lora", {}).get("r", 64),
        lora_alpha=cfg.get("lora", {}).get("alpha", 128),
        lora_dropout=cfg.get("lora", {}).get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get("lora", {}).get("target_modules", None),
    ) if cfg.get("lora", {}) else None

    # 加载policy
    policy = load_peft_or_full_as_policy(
        path=cfg["policy_model"],
        quant_config=quant_cfg,
        dtype=dtype,
        device_map=single_device_map,  # 关键改动：不再用 "auto"
        lora_cfg=lora_cfg,
        enable_gc=bool(cfg.get("training", {}).get("gradient_checkpointing", False)),
    )
    policy.train()
    if hasattr(policy.config, "use_cache"):
        policy.config.use_cache = False  # 训练期关 cache

    # 加载ref
    ref_quant_cfg = quant_cfg
    ref = load_peft_or_full_as_ref(
        path=cfg["ref_model"],
        quant_config=ref_quant_cfg,
        dtype=dtype,
        device_map=single_device_map,  # 关键改动：与 policy 一致
    )

    # ---- 优化器：只训练可训练参数（一般就是 LoRA） ----
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check LoRA/4-bit settings.")
    optim = AdamW(trainable_params, lr=float(cfg["training"]["lr"]))

    # ---- 训练器 ----
    trainer = GRPOTrainer(
        policy=policy,
        ref=ref,
        tokenizer=tokenizer,
        optim=optim,
        cfg=cfg,
        gkd_online=cfg.get("gkd_online", {}),
        device=target_device,   # 关键改动：Trainer 的 device 与模型一致
    )

    # ---- 数据准备：prompt + cut_code ----
    with open(args.train_file, "r", encoding="utf-8") as f:
        arr = json.load(f)

    batch_data: List[Dict[str, str]] = []
    for ex in arr:
        p = ex.get("prompt", "")
        cut = ex.get("cut_code", "") or extract_cut(p, "")
        batch_data.append({"prompt": p, "cut_code": cut})

    ro = RolloutConfig(
        group_size=cfg["rollout"]["group_size"],
        temperature=cfg["rollout"]["temperature"],
        top_p=cfg["rollout"]["top_p"],
        max_new_tokens=cfg["rollout"]["max_new_tokens"],
    )

    def reward_fn(cut_code, test_code):
        return composite_reward(
            cut_code,
            test_code,
            weights=cfg["reward"]["weights"],
            time_budget_s=cfg["reward"]["time_budget_s"],
            repeat_runs=cfg["reward"]["repeat_runs"],
            invariance_aug=cfg["reward"]["invariance_aug"],
        )

    steps = int(cfg["training"]["steps"])
    bs = int(cfg["training"]["micro_batch_size"])

    # ---- 优雅中断：Ctrl+C 时保存一次 ----
    interrupted = {"flag": False}
    def _handle_sigint(signum, frame):
        interrupted["flag"] = True
        print("\n[GRPO] Caught KeyboardInterrupt, saving checkpoint...", flush=True)
        policy.save_pretrained(os.path.join(out_dir, "interrupted"))
        tokenizer.save_pretrained(os.path.join(out_dir, "interrupted"))
        sys.exit(0)
    signal.signal(signal.SIGINT, _handle_sigint)

    # ---- 训练主循环 ----
    for step in range(steps):
        batch_prompts = random.sample(batch_data, k=bs)
        loss, rewards, chosen = trainer.step(batch_prompts, ro, reward_fn)

        if (step + 1) % 5 == 0:
            r_flat = [r for grp in rewards for r in grp]
            r_avg = sum(r_flat) / max(1, len(r_flat))
            print(f"[GRPO] step {step+1}/{steps}  loss={loss:.4f}  avg_reward={r_avg:.4f}", flush=True)

        if (step + 1) % 20 == 0:
            ckpt_dir = os.path.join(out_dir, f"step_{step+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

        if interrupted["flag"]:
            break

    # ---- 最终保存 ----
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
