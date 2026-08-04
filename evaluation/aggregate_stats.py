"""Aggregate all results: coverage/FTR/MKR with mean±std over seeds,
Wilcoxon signed-rank tests and Vargha-Delaney A12 vs. TriUTest, plus cost table.

Usage: python exp/aggregate_stats.py [--mkr-tag mkr]
Writes exp/results/summary.json and prints markdown tables.
"""
import argparse, glob, json, os, itertools
from collections import defaultdict

import numpy as np
try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

ROOT = "/root/autodl-tmp/TriUTest"
RES = f"{ROOT}/exp/results"
SUBJECTS = ["humaneval", "quixbugs", "codetiming", "apimd"]
MUT_SUBJECTS = ["humaneval", "codetiming", "apimd"]


def a12(x, y):
    """Vargha-Delaney A12: P(x > y) + 0.5 P(x == y)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return float("nan")
    gt = eq = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                gt += 1
            elif xi == yi:
                eq += 1
    return (gt + 0.5 * eq) / (m * n)


def load_eval():
    data = defaultdict(dict)   # method -> (subject, seed) -> record
    for path in glob.glob(f"{RES}/eval/*/*.json"):
        method = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            r = json.load(f)
        data[method][(r["subject"], r["seed"])] = r
    return data


def load_mkr(tag):
    data = defaultdict(dict)
    for path in glob.glob(f"{RES}/{tag}/*/*.json"):
        method = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            r = json.load(f)
        data[method][(r["subject"], r["seed"])] = r
    return data


def pooled_mkr(mkr_data):
    """Pool-based equivalent-mutant filtering across all methods and seeds.

    Returns: mkr[method][(subject, seed)] = percentage,
             filter_stats[subject] = dict(N_all, N_ne, N_eq)
    """
    # union of killed mutant ids per subject/target across every method & seed
    union = defaultdict(lambda: defaultdict(set))   # subject -> target -> killed ids
    totals = defaultdict(dict)                      # subject -> target -> total
    for method, d in mkr_data.items():
        for (subj, seed), rec in d.items():
            for tid, t in rec["targets"].items():
                union[subj][tid].update(t["killed"])
                if t["total"]:
                    totals[subj][tid] = max(totals[subj].get(tid, 0), t["total"])
    filter_stats = {}
    for subj in union:
        n_all = sum(totals[subj].values())
        n_ne = sum(len(union[subj][tid]) for tid in union[subj])
        filter_stats[subj] = {"N_all": n_all, "N_ne": n_ne, "N_eq": n_all - n_ne,
                              "r_eq": round((n_all - n_ne) / n_all, 4) if n_all else 0}
    out = defaultdict(dict)
    for method, d in mkr_data.items():
        for (subj, seed), rec in d.items():
            n_ne = filter_stats.get(subj, {}).get("N_ne", 0)
            killed = 0
            for tid, t in rec["targets"].items():
                killed += len(set(t["killed"]) & union[subj][tid])
            out[method][(subj, seed)] = 100.0 * killed / n_ne if n_ne else 0.0
    return out, filter_stats


def load_ftr_v2():
    """Authoritative FTR: pruning-decoupled, redef-stripped recomputation.
    Returns method -> seed -> {uut_id: exposed}."""
    data = defaultdict(dict)
    for path in glob.glob(f"{RES}/ftr_v2/*/quixbugs_seed*.json"):
        method = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            r = json.load(f)
        data[method][r["seed"]] = {p["uut_id"]: p["exposed"]
                                   for p in r.get("per_program", [])}
    return data


def mstd(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mkr-tag", default="mkr")
    ap.add_argument("--ref-method", default="triutest-3b")
    args = ap.parse_args()

    ev = load_eval()
    mk = load_mkr(args.mkr_tag)
    fv = load_ftr_v2()
    mkr, fstats = pooled_mkr(mk) if mk else ({}, {})

    summary = {"filter_stats": fstats, "methods": {}}
    for method in sorted(ev.keys()):
        m = {"coverage": {}, "ftr": {}, "mkr": {}, "cost": {}}
        for subj in SUBJECTS:
            seeds = sorted(s for (sj, s) in ev[method] if sj == subj)
            if not seeds:
                continue
            recs = [ev[method][(subj, s)] for s in seeds]
            st_m, st_s = mstd([r["stmt"] for r in recs])
            br_m, br_s = mstd([r["branch"] for r in recs])
            m["coverage"][subj] = {"stmt_mean": st_m, "stmt_std": st_s,
                                   "branch_mean": br_m, "branch_std": br_s,
                                   "seeds": seeds}
            if subj == "quixbugs":
                if method in fv:
                    per_seed = [100.0 * np.mean([1.0 if e else 0.0
                                                 for e in fv[method][s].values()])
                                for s in sorted(fv[method])]
                    f_m, f_s = mstd(per_seed)
                    m["ftr"] = {"mean": f_m, "std": f_s, "source": "ftr_v2"}
                else:
                    f_m, f_s = mstd([r.get("ftr") for r in recs])
                    m["ftr"] = {"mean": f_m, "std": f_s, "source": "eval"}
            m["cost"][subj] = {
                "gen_wall_s": mstd([r.get("gen_wall_s", 0) for r in recs])[0],
                "gen_tokens": mstd([r.get("gen_tokens", 0) for r in recs])[0],
                "exec_wall_s": mstd([r.get("exec_wall_s", 0) for r in recs])[0],
                "n_pytest_runs": mstd([r.get("n_pytest_runs", 0) for r in recs])[0],
            }
        if method in mkr:
            for subj in MUT_SUBJECTS:
                vals = [v for (sj, s), v in mkr[method].items() if sj == subj]
                if vals:
                    mm, ss = mstd(vals)
                    m["mkr"][subj] = {"mean": mm, "std": ss}
            if all(s in m["mkr"] for s in MUT_SUBJECTS):
                per_seed_avg = defaultdict(list)
                for (sj, s), v in mkr[method].items():
                    if sj in MUT_SUBJECTS:
                        per_seed_avg[s].append(v)
                avgs = [np.mean(v) for s, v in per_seed_avg.items() if len(v) == 3]
                if avgs:
                    mm, ss = mstd(avgs)
                    m["mkr"]["average"] = {"mean": mm, "std": ss}
        summary["methods"][method] = m

    # ---- significance tests vs ref method (per-UUT paired where possible) ----
    ref = args.ref_method
    tests = {}

    def _wilcoxon_pair(x, y):
        if len(x) < 8 or all(a == b for a, b in zip(x, y)):
            return None
        try:
            _stat, p = wilcoxon(x, y)
        except ValueError:
            return None
        return {"p": float(p), "a12": a12(x, y), "n": len(x),
                "mean_ref": float(np.mean(x)), "mean_other": float(np.mean(y))}

    if ref in ev and wilcoxon is not None:
        # per-subject union of killed ids for MKR per-target rates
        union = defaultdict(lambda: defaultdict(set))
        for method, d in mk.items():
            for (subj, seed), rec in d.items():
                for tid, t in rec["targets"].items():
                    union[subj][tid].update(t["killed"])

        for method in ev:
            if method == ref:
                continue
            tests[method] = {}
            # --- coverage: paired per-UUT statement/branch, pooled over seeds ---
            for subj in ["humaneval", "quixbugs"]:
                for metric in ["stmt", "branch"]:
                    x, y = [], []
                    for (sj, s), r in ev[ref].items():
                        if sj != subj or (subj, s) not in ev[method]:
                            continue
                        ru = {p["uut_id"]: p for p in r.get("per_uut", [])}
                        mu = {p["uut_id"]: p for p in ev[method][(subj, s)].get("per_uut", [])}
                        for uid in ru:
                            if uid in mu and metric in ru[uid]:
                                x.append(ru[uid][metric])
                                y.append(mu[uid].get(metric, 0.0))
                    t = _wilcoxon_pair(x, y)
                    if t:
                        tests[method][f"{subj}_{metric}"] = t
            # --- MKR: paired per-target pooled kill rates, pooled over seeds ---
            if ref in mk and method in mk:
                for subj in MUT_SUBJECTS:
                    x, y = [], []
                    for (sj, s), rr in mk[ref].items():
                        if sj != subj or (subj, s) not in mk[method]:
                            continue
                        mr = mk[method][(subj, s)]
                        for tid, t in rr["targets"].items():
                            ne = union[subj][tid]
                            if not ne or tid not in mr["targets"]:
                                continue
                            x.append(len(set(t["killed"]) & ne) / len(ne))
                            y.append(len(set(mr["targets"][tid]["killed"]) & ne) / len(ne))
                    t = _wilcoxon_pair(x, y)
                    if t:
                        tests[method][f"{subj}_mkr"] = t
            # --- FTR: exact McNemar on paired per-program exposure, pooled seeds ---
            try:
                from scipy.stats import binomtest
                b = c = 0
                x, y = [], []
                if ref in fv and method in fv:
                    # authoritative per-program exposure from ftr_v2
                    pairs = []
                    for s in fv[ref]:
                        if s in fv[method]:
                            ru, mu = fv[ref][s], fv[method][s]
                            pairs += [(ru[u], mu[u]) for u in ru if u in mu]
                    for re_, me_ in pairs:
                        x.append(1.0 if re_ else 0.0)
                        y.append(1.0 if me_ else 0.0)
                        if re_ and not me_:
                            b += 1
                        elif me_ and not re_:
                            c += 1
                else:
                    for (sj, s), r in ev[ref].items():
                        if sj != "quixbugs" or ("quixbugs", s) not in ev[method]:
                            continue
                        ru = {p["uut_id"]: p.get("exposed", False) for p in r.get("per_uut", [])}
                        mu = {p["uut_id"]: p.get("exposed", False)
                              for p in ev[method][("quixbugs", s)].get("per_uut", [])}
                        for uid in ru:
                            if uid in mu:
                                x.append(1.0 if ru[uid] else 0.0)
                                y.append(1.0 if mu[uid] else 0.0)
                                if ru[uid] and not mu[uid]:
                                    b += 1
                                elif mu[uid] and not ru[uid]:
                                    c += 1
                if b + c > 0:
                    p = binomtest(b, b + c, 0.5).pvalue
                    tests[method]["quixbugs_ftr_mcnemar"] = {
                        "p": float(p), "ref_only": b, "other_only": c,
                        "a12": a12(x, y), "n": len(x)}
            except ImportError:
                pass
    summary["wilcoxon_vs_" + ref] = tests

    out = f"{RES}/summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1)[:6000])
    print("saved to", out)


if __name__ == "__main__":
    main()
