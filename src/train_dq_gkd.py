import argparse, yaml, os, torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from src.data.json_dataset import load_supervised_dataset
from src.data.collator import SFTDataCollator

# ====== Utils ======
def is_peft_dir(path: str) -> bool:
    return any(os.path.exists(os.path.join(path, f))
               for f in ["adapter_config.json","adapter_model.safetensors","adapter_model.bin"])

def maybe_enable_gc(model, enable: bool):
    if enable:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

def get_dtype(cfg_dtype: Optional[str]):
    if not cfg_dtype:
        return None
    m = str(cfg_dtype).lower()
    if m in ["bf16","bfloat16"]: return torch.bfloat16
    if m in ["fp16","float16","half"]: return torch.float16
    if m in ["fp32","float32"]: return torch.float32
    return None

# ====== Distill Trainer with vocab alignment (JSD + reverse-KL) ======
class DistillTrainer(Trainer):
    def __init__(self, *args, teacher=None, gkd_cfg=None,
                 student_tokenizer=None, teacher_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        self.gkd = gkd_cfg or {}
        self.use_jsd = bool(self.gkd.get("use_jsd", True))
        self.use_rkl = bool(self.gkd.get("use_rkl", True))
        self.jsd_lambda = float(self.gkd.get("jsd_lambda", 0.5))
        self.rkl_lambda = float(self.gkd.get("rkl_lambda", 0.5))
        self.jsd_alpha  = float(self.gkd.get("jsd_alpha", 0.9))
        self.temperature = float(self.gkd.get("temperature", 2.0))

        self._vmap_cached = False
        self._vmap = None  # student_vocab_size -> teacher_id ( -1 for OOV )

    @torch.no_grad()
    def _ensure_vocab_map(self, device):
        if self._vmap_cached:
            return
        if self.teacher is None or self.student_tokenizer is None or self.teacher_tokenizer is None:
            self._vmap = None
            self._vmap_cached = True
            return
        st_tok, te_tok = self.student_tokenizer, self.teacher_tokenizer
        st_vocab_size = st_tok.vocab_size
        vmap = torch.full((st_vocab_size,), -1, dtype=torch.long)
        st_tokens = st_tok.convert_ids_to_tokens(list(range(st_vocab_size)))
        for sid, tok in enumerate(st_tokens):
            tid = te_tok.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                vmap[sid] = tid
        self._vmap = vmap.to(device)
        self._vmap_cached = True

    def _align_vocab(self, s_logits: torch.Tensor, t_logits: torch.Tensor):
        if self.teacher is None or self._vmap is None:
            return s_logits, t_logits
        self._ensure_vocab_map(device=t_logits.device)
        idx = torch.clamp(self._vmap, min=0)
        aligned = t_logits.index_select(dim=2, index=idx)       # (B, T, V_s)
        oov_mask = (self._vmap < 0).unsqueeze(0).unsqueeze(0)   # (1,1,V_s)
        aligned = aligned.masked_fill(oov_mask, torch.finfo(aligned.dtype).min/2)
        return s_logits, aligned

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits                    # (B, L, V_s)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = (shift_labels != -100).float()

        # CE on gold labels
        ce = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        loss = ce

        # Distillation terms
        if self.teacher is not None and (self.use_jsd or self.use_rkl):
            with torch.no_grad():
                t_out = self.teacher(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None)
                )
            t_logits = t_out.logits[:, :-1, :].contiguous()
            s_log, t_log = self._align_vocab(shift_logits, t_logits)

            T = max(self.temperature, 1e-6)
            s_log = s_log / T
            t_log = t_log / T

            if self.use_jsd:
                # Jensen–Shannon divergence on logits (stable)
                loss_m = torch.logsumexp(torch.stack([s_log, t_log]), dim=0) - torch.log(torch.tensor(2.0, device=s_log.device))
                ps = torch.log_softmax(s_log, dim=-1)
                pt = torch.log_softmax(t_log, dim=-1)
                pm = torch.log_softmax(loss_m, dim=-1)
                jsd = 0.5 * (torch.exp(ps) * (ps - pm)).sum(-1) + 0.5 * (torch.exp(pt) * (pt - pm)).sum(-1)
                jsd = (jsd * mask).sum() / (mask.sum() + 1e-8)
                loss = loss + self.jsd_lambda * jsd

            if self.use_rkl:
                # reverse KL: KL( teacher || student )
                ps = torch.log_softmax(s_log, dim=-1)
                pt = torch.log_softmax(t_log, dim=-1)
                rkl = (torch.exp(pt) * (pt - ps)).sum(-1)
                rkl = (rkl * mask).sum() / (mask.sum() + 1e-8)
                loss = loss + self.rkl_lambda * rkl

        return (loss, outputs) if return_outputs else loss

# ====== Helper ======
def build_tokenizer(maybe_adapter_or_model: str):
    if is_peft_dir(maybe_adapter_or_model):
        peft_cfg = PeftConfig.from_pretrained(maybe_adapter_or_model)
        base_name = peft_cfg.base_model_name_or_path
        tok = AutoTokenizer.from_pretrained(base_name, use_fast=True)
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
        cfg = yaml.safe_load(f)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # -------- output dir --------
    out_dir = cfg["training"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # -------- tokenizers --------
    student_id_or_adapter = cfg["base_model"]                         # may be LoRA dir or model id
    student_tok = build_tokenizer(student_id_or_adapter)
    teacher_id = cfg.get("teacher_model", None)
    if teacher_id is None:
        raise ValueError("teacher_model must be provided in config.")
    teacher_tok = AutoTokenizer.from_pretrained(teacher_id, use_fast=True)

    # -------- dataset & collator --------
    # keep exactly same pipeline as SFT: dataset -> collator(tokenize+pad/trunc)
    expansion_files = cfg.get("data", {}).get("domain_expansion", None)
    ds = load_supervised_dataset(
        cfg["data"]["train_jsonl"],
        add_domain_expansion=bool(expansion_files),
        expansion_files=expansion_files or []
    )
    collator = SFTDataCollator(student_tok, max_len=cfg["data"].get("max_length", 2048))

    # -------- student model (may resume from SFT LoRA dir) --------
    dtype = get_dtype(cfg["training"].get("dtype"))
    base_model_id = student_id_or_adapter
    resume_adapter = cfg["training"].get("resume_adapter", None)

    if is_peft_dir(base_model_id):
        # base + load adapter from given LoRA dir (SFT output)
        peft_cfg = PeftConfig.from_pretrained(base_model_id)
        base_model_id = peft_cfg.base_model_name_or_path
        student_base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype or torch.bfloat16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(student_base, student_id_or_adapter)
        print(f"[info] loaded base '{base_model_id}' + adapter '{student_id_or_adapter}'", flush=True)
    else:
        # from plain base id, optionally 4-bit + new LoRA head
        load_in_4bit = bool(cfg["training"].get("load_in_4bit", True))
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype or torch.bfloat16,
            )
            student_base = AutoModelForCausalLM.from_pretrained(
                base_model_id, quantization_config=quant_cfg, device_map="auto"
            )
        else:
            student_base = AutoModelForCausalLM.from_pretrained(
                base_model_id, torch_dtype=dtype or torch.bfloat16, device_map="auto"
            )
        maybe_enable_gc(student_base, enable=bool(cfg["training"].get("gradient_checkpointing", False)))

        if resume_adapter and is_peft_dir(resume_adapter):
            model = PeftModel.from_pretrained(student_base, resume_adapter)
            print(f"[info] resumed adapter from '{resume_adapter}'", flush=True)
        else:
            lcfg = LoraConfig(
                r=cfg["lora"]["r"],
                lora_alpha=cfg["lora"]["alpha"],
                lora_dropout=cfg["lora"]["dropout"],
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=cfg["lora"]["target_modules"]
            )
            model = get_peft_model(student_base, lcfg)
            print("[info] created new LoRA head for distillation.", flush=True)

    # -------- teacher model (frozen) --------
    teacher_8bit = bool(cfg.get("teacher_8bit", False))
    if teacher_8bit:
        teacher = AutoModelForCausalLM.from_pretrained(teacher_id, load_in_8bit=True, device_map="auto")
    else:
        teacher = AutoModelForCausalLM.from_pretrained(teacher_id, torch_dtype=dtype or torch.bfloat16, device_map="auto")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # -------- training args --------
    targs = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg["training"].get("batch_size", 2),
        gradient_accumulation_steps=cfg["training"].get("grad_accum", 1),
        learning_rate=cfg["training"].get("lr", 2e-4),
        num_train_epochs=cfg["training"].get("epochs", 1),
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.03),
        save_steps=cfg["training"].get("save_steps", 500),
        logging_steps=cfg["training"].get("logging_steps", 10),
        bf16=(get_dtype(cfg["training"].get("dtype")) == torch.bfloat16),
        fp16=(get_dtype(cfg["training"].get("dtype")) == torch.float16),
        report_to=cfg["training"].get("report_to", "none"),
        remove_unused_columns=False,
    )

    trainer = DistillTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=student_tok,
        teacher=teacher,
        gkd_cfg=cfg.get("gkd", {}),
        student_tokenizer=student_tok,
        teacher_tokenizer=teacher_tok,
    )

    trainer.train()
    model.save_pretrained(out_dir)
    student_tok.save_pretrained(out_dir)
    print(f"[done] saved to: {out_dir}")

if __name__ == "__main__":
    main()
