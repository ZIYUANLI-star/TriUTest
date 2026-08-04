"""Score stage-2 on-policy candidates with DQ-GKD quality signals (paper Eqs. 4-6)
and structural diversity distinctness (Eqs. 7-8).

Produces data/gkd_candidates.jsonl:
  {idx, candidates: [...], weights: [...], distinct: [...], categories: [...]}
Run in the triu env (CPU-parallel):
  python exp/score_gkd_candidates.py
"""
import argparse, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = "/root/autodl-tmp/TriUTest"
sys.path.insert(0, ROOT)

from src.reward.quality import (s_exec, s_ass_batch, quality_weights,
                                diversity_scores, NON_SEMANTIC, SEMANTIC)

CUT_PAT = re.compile(r"Code Under Test:\s*\n?([\s\S]+?)\s*$", re.I)


def extract_cut(prompt):
    m = CUT_PAT.search(prompt)
    return m.group(1).strip() if m else ""


def score_one(job):
    idx, prompt, cands = job
    cut = extract_cut(prompt)
    execs, fails, cats = [], [], []
    for c in cands:
        if not cut or not c.strip():
            execs.append(0.0)
            fails.append(1.0)
            cats.append(NON_SEMANTIC)
            continue
        e, cat = s_exec(cut, c, timeout_s=10)
        execs.append(e)
        fails.append(1.0 if cat == NON_SEMANTIC else 0.0)
        cats.append(cat)
    asses = s_ass_batch(cands)
    w = quality_weights(execs, asses, fails)
    _mean, distinct = diversity_scores(cands)
    return {"idx": idx, "candidates": cands, "weights": w,
            "distinct": distinct, "categories": cats,
            "s_exec": execs, "s_ass": asses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default=f"{ROOT}/data/gkd_train.json")
    ap.add_argument("--raw", default=f"{ROOT}/data/gkd_candidates_raw.jsonl")
    ap.add_argument("--out", default=f"{ROOT}/data/gkd_candidates.jsonl")
    ap.add_argument("--jobs", type=int, default=48)
    args = ap.parse_args()

    with open(args.train_file, encoding="utf-8") as f:
        arr = json.load(f)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass

    jobs = []
    for line in open(args.raw, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["idx"] in done:
            continue
        jobs.append((r["idx"], arr[r["idx"]]["prompt"], r["candidates"]))
    print(f"[gkdscore] todo={len(jobs)}", flush=True)

    with open(args.out, "a", encoding="utf-8") as fout:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for n, rec in enumerate(ex.map(score_one, jobs, chunksize=4)):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (n + 1) % 200 == 0:
                    fout.flush()
                    print(f"[gkdscore] {n+1}/{len(jobs)}", flush=True)
    print("[gkdscore] DONE", flush=True)


if __name__ == "__main__":
    main()
