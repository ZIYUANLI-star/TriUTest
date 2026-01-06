# ==== 1) 配置这里 ====
# "C:\\Users\\28772\\.cache\\huggingface\\hub\\models--Qwen--Qwen2.5-3B-Instruct\\snapshots\\Qwen2.5-3B"
ADAPTER_DIR = "D:\postgradute_Learning\paper\exam\GKD\\runs\grpo-mutant\step_120"  # ← 换成你的 LoRA 目录
PROMPT = """
Based on the function description and the code snippet below, please generate a comprehensive set of detailed test cases that cover typical usage, edge cases, and potential error conditions.\n\n
Function Description:
Traverse around body and its simple definition scope.
Code Under Test:
from collections.abc import Sequence, Iterable, Iterator
def walk_body(body: Sequence[stmt]) -> Iterator[stmt]:
    
    for node in body:
        if isinstance(node, If):
            yield from walk_body(node.body)
            yield from walk_body(node.orelse)
        elif isinstance(node, Try):
            yield from walk_body(node.body)
            for h in node.handlers:
                yield from walk_body(h.body)
            yield from walk_body(node.orelse)
            yield from walk_body(node.finalbody)
        else:
            yield node






"""
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# 可选：把 LoRA 合并并另存为完整模型（不想合并就设为 None）
MERGE_AND_SAVE_TO = None  # 例如 "/abs/path/to/merged-qwen3b-sft"

# 是否尝试 4-bit 量化加载基座（省显存）
USE_4BIT = True
DTYPE = "bf16"  # 可选: "bf16" | "fp16" | "fp32"

# ==== 2) 代码区：直接运行 ====
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, None: None}
dtype = DTYPE_MAP.get(DTYPE, torch.bfloat16)

try:
    from transformers import BitsAndBytesConfig
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False

def load_tokenizer(adapter_dir: str, base_model_name: str):
    has_tok = any(os.path.exists(os.path.join(adapter_dir, f))
                  for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json"])
    src = adapter_dir if has_tok else base_model_name
    tok = AutoTokenizer.from_pretrained(src, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok

def load_base(base_model_name: str):
    if USE_4BIT and HAVE_BNB:
        print("[info] loading base in 4-bit (NF4 + double quant) …")
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        return AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="auto", quantization_config=qcfg, trust_remote_code=True
        )
    else:
        print("[info] loading base in", DTYPE, "…")
        return AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )

@torch.no_grad()
def chat(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)

# ---- 加载 LoRA + 基座 ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

peft_cfg = PeftConfig.from_pretrained(ADAPTER_DIR)
base_name = peft_cfg.base_model_name_or_path
if not base_name:
    raise ValueError("adapter_config.json 缺少 base_model_name_or_path，请补齐或手动指定。")

tokenizer = load_tokenizer(ADAPTER_DIR, base_name)
base = load_base(base_name)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

print(f"[ok] base: {base_name}")
print(f"[ok] adapter: {ADAPTER_DIR}")

# ---- 测试一次 ----
print("\n[Prompt]\n", PROMPT)
reply = chat(model, tokenizer, PROMPT, MAX_NEW_TOKENS, TEMPERATURE, TOP_P)
print("\n[Model Output]\n", reply)

# ---- 可选：合并 LoRA 并保存 ----
if MERGE_AND_SAVE_TO:
    print(f"[info] merging LoRA into base and saving to: {MERGE_AND_SAVE_TO}")
    merged = model.merge_and_unload()
    os.makedirs(MERGE_AND_SAVE_TO, exist_ok=True)
    merged.save_pretrained(MERGE_AND_SAVE_TO)
    tokenizer.save_pretrained(MERGE_AND_SAVE_TO)
    print("[done] merged model saved.")
