"""Mutation Kill Rate evaluation with MutPy (run inside the mut39 / py3.9 env,
or inside the unified py3.10 env for the R1.5 consistency check).

Usage:
  python exp/run_mkr.py --method <m> --subject humaneval|codetiming|apimd --seed 40

Reads sanitized passing suites from results/suites/<method>/<subject>_seed<k>/
Writes results/mkr/<method>/<subject>_seed<k>.json:
  {targets: {tid: {killed: [ids], total: N}}, mut_wall_s, n_mut_runs}
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor

ROOT = "/root/autodl-tmp/TriUTest"
BENCH = "/root/autodl-tmp/benchmarks"
UUT_DIR = f"{ROOT}/exp/uuts"

OPERATORS = ["AOD", "AOR", "ASR", "BCR", "COD", "COI", "EHD", "EXS", "LCR", "ROR", "SIR"]
# generous: large project modules (e.g. apimd.compiler, 400+ mutants x pytest
# runner) need ~10 min; killing MutPy early silently loses the whole report
MUT_TIMEOUT = 2400


def mutpy_script():
    """Console script mut.py living next to the current interpreter."""
    cand = os.path.join(os.path.dirname(sys.executable), "mut.py")
    if os.path.exists(cand):
        return [sys.executable, cand]
    from shutil import which
    m = which("mut.py")
    if m:
        return [sys.executable, m]
    raise RuntimeError("mut.py not found for interpreter " + sys.executable)


def parse_report(path):
    """Parse MutPy YAML report -> (killed_ids, total_non_incompetent)."""
    import re, yaml
    with open(path) as f:
        text = f.read()
    # strip python object tags (e.g. "!!python/module:cut ''") so safe_load works
    text = re.sub(r"!!python\S*", "", text)
    rep = yaml.safe_load(text)
    killed, total = [], 0
    for m in (rep or {}).get("mutations", []):
        status = str(m.get("status", "")).lower()
        if status == "incompetent":
            continue
        total += 1
        sub = m.get("mutations") or [{}]
        op = sub[0].get("operator", "?")
        ln = sub[0].get("lineno", "?")
        mid = f"{op}_{ln}_{m.get('number', total)}"
        if status in ("killed", "timeout"):
            killed.append(mid)
    return killed, total


def run_mutpy_target(job):
    """One MutPy invocation: (tid, src_files, test_files, target_mod)."""
    tid, src_files, test_files, target_mod = job
    if not test_files:
        return tid, {"killed": [], "total": 0, "wall_s": 0.0, "skipped": "no_tests"}
    tmp = tempfile.mkdtemp(prefix="mkr_")
    t0 = time.time()
    try:
        for rel, content in src_files.items():
            dst = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as f:
                f.write(content)
        unit_mods = []
        for i, tf in enumerate(test_files):
            name = f"test_m{i}"
            dst = os.path.join(tmp, name + ".py")
            shutil.copy(tf, dst)
            # MutPy precondition: the suite must pass on the original program
            # UNDER THIS interpreter. Suites were filtered in the primary env;
            # environment-sensitive files (py3.13-only syntax/behaviour) must
            # be dropped here or MutPy aborts and the whole report is lost.
            try:
                pre = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", name + ".py",
                     "-p", "no:cacheprovider"],
                    cwd=tmp, capture_output=True, text=True, timeout=120)
                ok = pre.returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
            if not ok:
                os.remove(dst)
                continue
            unit_mods.append(name)
        if not unit_mods:
            return tid, {"killed": [], "total": 0, "wall_s": 0.0,
                         "skipped": "no_env_compatible_tests"}
        cmd = mutpy_script() + [
               "--target", target_mod,
               "--unit-test"] + unit_mods + [
               "--runner", "pytest",
               "--report", "rep.yaml",
               "--timeout-factor", "3",
               "--operator"] + OPERATORS
        try:
            subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                           timeout=MUT_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass
        rep = os.path.join(tmp, "rep.yaml")
        killed, total = [], 0
        if os.path.exists(rep):
            try:
                killed, total = parse_report(rep)
            except Exception as e:
                return tid, {"killed": [], "total": 0, "wall_s": round(time.time() - t0, 2),
                             "error": f"parse: {e}"}
        return tid, {"killed": killed, "total": total,
                     "wall_s": round(time.time() - t0, 2)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_jobs(method, subject, seed):
    suite_root = f"{ROOT}/exp/results/suites/{method}/{subject}_seed{seed}"
    with open(os.path.join(UUT_DIR, f"{subject}.json")) as f:
        uuts = json.load(f)
    jobs = []
    if uuts and uuts[0].get("kind") == "standalone":
        for u in uuts:
            tests = sorted(glob.glob(os.path.join(suite_root, u["uut_id"], "*.py")))
            jobs.append((u["uut_id"], {"cut.py": u["cut_code"]}, tests, "cut"))
    else:
        # project: one MutPy target per module; tests = union of that module's UUT suites
        proj = uuts[0]["project_root"]
        pkg = uuts[0]["package"]
        mod_uuts = {}
        for u in uuts:
            mod_uuts.setdefault(u["module"], []).append(u)
        for mod, us in mod_uuts.items():
            rel = mod.replace(".", "/") + ".py"
            fpath = os.path.join(proj, rel)
            if not os.path.exists(fpath):
                fpath = os.path.join(proj, mod.replace(".", "/"), "__init__.py")
                rel = mod.replace(".", "/") + "/__init__.py"
                if not os.path.exists(fpath):
                    continue
            src_files = {}
            # copy whole package (imports inside module need siblings)
            for dirpath, _d, files in os.walk(os.path.join(proj, pkg)):
                for fn in files:
                    if fn.endswith(".py"):
                        fp = os.path.join(dirpath, fn)
                        src_files[os.path.relpath(fp, proj)] = open(fp, encoding="utf-8").read()
            tests = []
            for u in us:
                tests += sorted(glob.glob(os.path.join(suite_root, u["uut_id"], "*.py")))
            jobs.append((mod, src_files, tests, mod))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--out-tag", default="mkr", help="mkr | mkr_unified (R1.5 check)")
    args = ap.parse_args()

    jobs = build_jobs(args.method, args.subject, args.seed)
    out_path = f"{ROOT}/exp/results/{args.out_tag}/{args.method}/{args.subject}_seed{args.seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    targets = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for tid, res in ex.map(run_mutpy_target, jobs):
            targets[tid] = res
    result = {
        "method": args.method, "subject": args.subject, "seed": args.seed,
        "python": sys.version.split()[0],
        "targets": targets,
        "mut_wall_s": round(sum(t.get("wall_s", 0) for t in targets.values()), 2),
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    tot = sum(t["total"] for t in targets.values())
    kil = sum(len(t["killed"]) for t in targets.values())
    print(f"[mkr] {args.method}/{args.subject}/seed{args.seed}: raw {kil}/{tot}")


if __name__ == "__main__":
    main()
