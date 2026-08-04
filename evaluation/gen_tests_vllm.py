"""Fast batch test generation with vLLM (run in the `vllm` conda env).

Usage:
  python exp/gen_tests_vllm.py --model <merged_dir_or_hf_id> --method sft-3b \
      --subjects humaneval quixbugs codetiming apimd --seeds 40 41 42 43 44 \
      [--format raw|chat]

Writes results/gen/<method>/<subject>_seed<k>.jsonl (same schema as gen_tests.py).
Wall-clock cost is measured per UUT as total batch time split proportionally by
generated tokens (all methods measured identically, so comparisons stay fair).
"""
import argparse, json, os, time

ROOT = "/root/autodl-tmp/TriUTest"
UUT_DIR = f"{ROOT}/exp/uuts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--n-candidates", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--format", choices=["raw", "chat"], default="raw")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--lora", default=None,
                    help="LoRA adapter dir; serve base model + adapter without "
                         "merging (saves the disk for a merged copy)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    lora_req = None
    tok = AutoTokenizer.from_pretrained(args.lora or args.model, use_fast=True)
    if args.lora:
        from vllm.lora.request import LoRARequest
        llm = LLM(model=args.model, dtype="bfloat16",
                  gpu_memory_utilization=args.gpu_mem, max_model_len=4096,
                  enforce_eager=False, enable_lora=True, max_lora_rank=64)
        lora_req = LoRARequest("adapter", 1, args.lora)
    else:
        llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
                  max_model_len=4096, enforce_eager=False)

    for subject in args.subjects:
        with open(os.path.join(UUT_DIR, f"{subject}.json")) as f:
            uuts = json.load(f)
        for seed in args.seeds:
            out_path = f"{ROOT}/exp/results/gen/{args.method}/{subject}_seed{seed}.jsonl"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            done = set()
            if os.path.exists(out_path):
                for line in open(out_path):
                    try:
                        done.add(json.loads(line)["uut_id"])
                    except Exception:
                        pass
            todo = [u for u in uuts if u["uut_id"] not in done]
            if not todo:
                print(f"[genv] skip {subject} seed{seed}", flush=True)
                continue
            prompts = []
            for u in todo:
                if args.format == "chat":
                    msgs = [
                        {"role": "system", "content": "You are a Python testing assistant. Output only executable Python test code."},
                        {"role": "user", "content": u["prompt"]},
                    ]
                    prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                           add_generation_prompt=True))
                else:
                    prompts.append(u["prompt"])
            sp = SamplingParams(n=args.n_candidates, temperature=args.temperature,
                                top_p=args.top_p, max_tokens=args.max_new_tokens,
                                seed=seed)
            t0 = time.time()
            outs = llm.generate(prompts, sp, lora_request=lora_req) if lora_req \
                else llm.generate(prompts, sp)
            batch_wall = time.time() - t0
            total_tokens = sum(len(o.token_ids) for r in outs for o in r.outputs) or 1
            with open(out_path, "a", encoding="utf-8") as fout:
                for u, r in zip(todo, outs):
                    cands = [o.text.strip() for o in r.outputs]
                    ntok = sum(len(o.token_ids) for o in r.outputs)
                    rec = {"uut_id": u["uut_id"], "subject": subject, "seed": seed,
                           "candidates": cands,
                           "gen_wall_s": round(batch_wall * ntok / total_tokens, 3),
                           "gen_tokens": ntok,
                           "prompt_tokens": len(r.prompt_token_ids)}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[genv] {args.method}/{subject}/seed{seed}: {len(todo)} UUTs "
                  f"in {batch_wall:.0f}s", flush=True)
    print("[genv] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
