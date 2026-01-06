import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype="auto")

    prompt = open(args.prompt_file, "r", encoding="utf-8").read()
    ip = tok(prompt, return_tensors="pt").to(mdl.device)
    gen = mdl.generate(**ip, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tok.eos_token_id)
    out = tok.decode(gen[0], skip_special_tokens=True)
    print(out)

if __name__ == "__main__":
    main()
