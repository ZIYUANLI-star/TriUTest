"""Stage-2 (DQ-GKD) on-policy candidate sampling.

Samples K candidates per GKD training prompt from the *current student policy*
(the SFT checkpoint), which are then scored by score_gkd_candidates.py and used
for quality-weighted masked distillation. This realizes the on-policy sampling
of paper Alg. 1 at dataset granularity (one refresh per stage), a standard
engineering approximation that keeps teacher-forward cost tractable.

Run in vllm-enabled env:
  python exp/gen_gkd_candidates.py --model runs/sft-3b-merged --out data/gkd_candidates_raw.jsonl
"""
import argparse, json, os

ROOT = "/root/autodl-tmp/TriUTest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged student checkpoint")
    ap.add_argument("--train-file", default=f"{ROOT}/data/gkd_train.json")
    ap.add_argument("--out", default=f"{ROOT}/data/gkd_candidates_raw.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    with open(args.train_file, encoding="utf-8") as f:
        arr = json.load(f)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass

    todo = [(i, ex) for i, ex in enumerate(arr) if i not in done]
    print(f"[gkdgen] total={len(arr)} todo={len(todo)}", flush=True)
    if not todo:
        return

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=3072)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_new_tokens, seed=42)

    B = 2000
    with open(args.out, "a", encoding="utf-8") as fout:
        for s in range(0, len(todo), B):
            chunk = todo[s:s + B]
            prompts = [ex["prompt"] for _, ex in chunk]
            outs = llm.generate(prompts, sp)
            for (i, ex), r in zip(chunk, outs):
                rec = {"idx": i, "candidates": [o.text.strip() for o in r.outputs]}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[gkdgen] {min(s+B, len(todo))}/{len(todo)}", flush=True)
    print("[gkdgen] DONE", flush=True)


if __name__ == "__main__":
    main()
