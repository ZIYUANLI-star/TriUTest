import json, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.reward.rewarders import reward_speed_stability

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/your_dataset.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype="auto")

    arr = json.load(open(args.data,"r",encoding="utf-8"))
    total, score_sum = 0, 0.0
    for ex in arr[:50]:
        prompt = ex["prompt"]
        cut = prompt.split("Code Under Test:")[1] if "Code Under Test:" in prompt else ""
        ip = tok(prompt, return_tensors="pt").to(mdl.device)
        gen = mdl.generate(**ip, do_sample=False, max_new_tokens=256, pad_token_id=tok.eos_token_id)
        out = tok.decode(gen[0], skip_special_tokens=True)
        test_code = out.split(prompt)[-1]
        s = reward_speed_stability(cut, test_code, time_budget_s=10, repeat=2)
        score_sum += s; total += 1
    print(f"Speed/Stability avg: {score_sum/max(1,total):.3f} over {total} items")

if __name__=="__main__":
    main()
