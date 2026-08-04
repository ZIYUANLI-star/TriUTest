"""Coverage / executability / FTR evaluation for generated test suites.

Usage:
  python exp/run_eval.py --gen results/gen/<method>/<subject>_seed<k>.jsonl \
      --method <method> --subject <subject> --seed <k>

Outputs:
  results/eval/<method>/<subject>_seed<k>.json   (per-UUT + aggregate metrics, costs)
  results/suites/<method>/<subject>_seed<k>/<uut_id>/test_<i>.py  (sanitized suites for MKR)
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re as _re

from extract_util import sanitize_test, prune_failing_tests, strip_cut_redefs

FAILED_PAT = _re.compile(r"(?:FAILED|ERROR) [^:]+::(\w+)")


def run_and_prune(tmp, fn, code, timeout):
    """Run one test file; if it fails, prune failing test functions and retry
    once. Returns (final_rc, final_code_or_None, wall_s, n_runs, first_failed).

    first_failed is the set of test-function names that failed in the initial
    (unpruned) run - the FTR check needs it for function-level exposure.
    Test-level filtering is applied identically to every method.
    """
    rc, out, dt = run([sys.executable, "-m", "pytest", "-q", fn,
                       "-p", "no:cacheprovider", "--timeout", str(timeout)],
                      cwd=tmp, timeout=timeout + 30)
    if rc == 0:
        return 0, code, dt, 1, set()
    failing = set(FAILED_PAT.findall(out or ""))
    if not failing:
        return rc, None, dt, 1, failing
    pruned = prune_failing_tests(code, failing)
    if not pruned:
        return rc, None, dt, 1, failing
    with open(os.path.join(tmp, fn), "w") as f:
        f.write(pruned)
    rc2, out2, dt2 = run([sys.executable, "-m", "pytest", "-q", fn,
                          "-p", "no:cacheprovider", "--timeout", str(timeout)],
                         cwd=tmp, timeout=timeout + 30)
    if rc2 == 0:
        # IMPORTANT: return the ORIGINAL rc, not 0. The pruned suite only
        # feeds MKR; FTR must see that the unpruned file failed on the CUT,
        # otherwise defect-exposing candidates are silently skipped.
        return rc, pruned, dt + dt2, 2, failing
    return rc, None, dt + dt2, 2, failing

ROOT = "/root/autodl-tmp/TriUTest"
UUT_DIR = f"{ROOT}/exp/uuts"
BENCH = "/root/autodl-tmp/benchmarks"
PYTEST_TIMEOUT = 60          # per test file wall limit (paper budget: 300s)
MAX_CANDS = 15               # unified budget: first 15 generated test files


def run(cmd, cwd, timeout, env=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return p.returncode, p.stdout + p.stderr, time.time() - t0
    except subprocess.TimeoutExpired:
        return -9, "TIMEOUT", time.time() - t0


def coverage_json(tmpdir, source_arg, test_files, extra_env=None, total_timeout=None):
    """Run pytest under coverage; return (stmt%, branch%, exec_s)."""
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    cov = [sys.executable, "-m", "coverage"]
    exec_s = 0.0
    if total_timeout is None:
        total_timeout = max(PYTEST_TIMEOUT * 8, 5 * len(test_files))
    rc, out, dt = run(cov + ["run", "--branch", f"--source={source_arg}",
                             "-m", "pytest", "-q", "--continue-on-collection-errors",
                             "--timeout", str(PYTEST_TIMEOUT),
                             "-p", "no:cacheprovider"] + test_files,
                      cwd=tmpdir, timeout=total_timeout, env=env)
    exec_s += dt
    rc2, out2, dt2 = run(cov + ["json", "-o", "cov.json"], cwd=tmpdir, timeout=60, env=env)
    exec_s += dt2
    stmt = branch = 0.0
    covp = os.path.join(tmpdir, "cov.json")
    if os.path.exists(covp):
        with open(covp) as f:
            data = json.load(f)
        tot = data.get("totals", {})
        n_st = tot.get("num_statements", 0)
        cov_st = tot.get("covered_lines", 0)
        n_br = tot.get("num_branches", 0)
        cov_br = tot.get("covered_branches", 0)
        stmt = 100.0 * cov_st / n_st if n_st else 0.0
        branch = 100.0 * cov_br / n_br if n_br else 0.0
    return stmt, branch, exec_s


def eval_standalone_uut(args):
    """HumanEval / QuixBugs: per-UUT coverage on cut.py; QuixBugs adds FTR."""
    u, cands, suite_dir, is_quixbugs, node_src = args
    tmp = tempfile.mkdtemp(prefix="ev_")
    rec = {"uut_id": u["uut_id"], "n_cands": 0, "n_exec": 0,
           "stmt": 0.0, "branch": 0.0, "exposed": False,
           "exec_wall_s": 0.0, "n_pytest_runs": 0}
    try:
        with open(os.path.join(tmp, "cut.py"), "w") as f:
            f.write(u["cut_code"])
        if node_src:
            with open(os.path.join(tmp, "node.py"), "w") as f:
                f.write(node_src)

        os.makedirs(suite_dir, exist_ok=True)
        kept, results = [], []
        for i, c in enumerate(cands[:MAX_CANDS]):
            code = sanitize_test(c, "from cut import *")
            if code:
                code = strip_cut_redefs(code, u["cut_code"])
            if not code:
                continue
            fn = f"test_g{i}.py"
            with open(os.path.join(tmp, fn), "w") as f:
                f.write(code)
            rc, passing_code, dt, nruns, first_failed = run_and_prune(
                tmp, fn, code, PYTEST_TIMEOUT)
            rec["exec_wall_s"] += dt
            rec["n_pytest_runs"] += nruns
            # coverage keeps the ORIGINAL file (failing tests still execute code)
            with open(os.path.join(tmp, fn), "w") as f:
                f.write(code)
            kept.append(fn)
            results.append((fn, rc, first_failed))
            # only tests that pass on the original program feed the MutPy stage;
            # test-level pruning salvages passing tests from partially-failing
            # files (applied identically to every method)
            if passing_code is not None:
                with open(os.path.join(suite_dir, fn), "w") as f:
                    f.write(passing_code)
        rec["n_cands"] = len(cands[:MAX_CANDS])
        rec["n_exec"] = sum(1 for fn, _rc, _ff in results
                            if os.path.exists(os.path.join(suite_dir, fn)))
        if kept:
            stmt, br, dt = coverage_json(tmp, "cut", kept)
            rec["stmt"], rec["branch"] = stmt, br
            rec["exec_wall_s"] += dt
            rec["n_pytest_runs"] += 1

        if is_quixbugs and kept:
            # FTR at test-function granularity: exposed iff some test function
            # fails on the buggy version AND passes on the fixed version.
            # (File-level pass/fail would let unrelated wrong assertions mask
            # genuine defect-triggering tests, and pruning-against-buggy would
            # silently delete them.)
            tmp2 = tempfile.mkdtemp(prefix="evf_")
            try:
                with open(os.path.join(tmp2, "cut.py"), "w") as f:
                    f.write(u["cut_code_fixed"])
                if node_src:
                    with open(os.path.join(tmp2, "node.py"), "w") as f:
                        f.write(node_src)
                for fn, rc_buggy, buggy_failed in results:
                    if rc_buggy == 0:
                        continue  # everything passes on buggy -> cannot expose
                    shutil.copy(os.path.join(tmp, fn), os.path.join(tmp2, fn))
                    rc_fixed, out_fixed, dt = run(
                        [sys.executable, "-m", "pytest", "-q", fn,
                         "-p", "no:cacheprovider", "--timeout", str(PYTEST_TIMEOUT)],
                        cwd=tmp2, timeout=PYTEST_TIMEOUT + 30)
                    rec["exec_wall_s"] += dt
                    rec["n_pytest_runs"] += 1
                    if rc_fixed == 0:
                        rec["exposed"] = True     # whole file passes on fixed
                        break
                    if rc_fixed > 0 and buggy_failed:
                        fixed_failed = set(FAILED_PAT.findall(out_fixed or ""))
                        if buggy_failed - fixed_failed:
                            rec["exposed"] = True  # some test fails only on buggy
                            break
            finally:
                shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rec


def eval_project(uuts, gen_map, suite_root, subject):
    """codetiming / apimd: merge all UUT suites, project-level coverage."""
    proj = uuts[0]["project_root"]
    pkg = uuts[0]["package"]
    tmp = tempfile.mkdtemp(prefix="evp_")
    agg = {"n_cands": 0, "n_exec": 0, "exec_wall_s": 0.0, "n_pytest_runs": 0}
    per_uut = []
    try:
        # copy package source into tmp so coverage paths are local
        shutil.copytree(os.path.join(proj, pkg), os.path.join(tmp, pkg))
        kept = []
        mod_src_cache = {}

        def _module_source(module):
            if module in mod_src_cache:
                return mod_src_cache[module]
            rel = module.replace(".", "/")
            src = ""
            for p in (os.path.join(proj, rel + ".py"),
                      os.path.join(proj, rel, "__init__.py")):
                if os.path.exists(p):
                    try:
                        src = open(p, encoding="utf-8", errors="replace").read()
                    except OSError:
                        src = ""
                    break
            mod_src_cache[module] = src
            return src

        for u in uuts:
            cands = gen_map.get(u["uut_id"], [])[:MAX_CANDS]
            urec = {"uut_id": u["uut_id"], "n_cands": len(cands), "n_exec": 0,
                    "exec_wall_s": 0.0, "n_pytest_runs": 0}
            sd = os.path.join(suite_root, u["uut_id"])
            os.makedirs(sd, exist_ok=True)
            for i, c in enumerate(cands):
                code = sanitize_test(c, f"from {u['module']} import *")
                if code:
                    msrc = _module_source(u["module"])
                    if msrc:
                        code = strip_cut_redefs(code, msrc)
                if not code:
                    continue
                fn = f"test_{u['uut_id']}_{i}.py"
                with open(os.path.join(tmp, fn), "w") as f:
                    f.write(code)
                rc, passing_code, dt, nruns, _ff = run_and_prune(
                    tmp, fn, code, PYTEST_TIMEOUT)
                urec["exec_wall_s"] += dt
                urec["n_pytest_runs"] += nruns
                with open(os.path.join(tmp, fn), "w") as f:
                    f.write(code)
                if passing_code is not None:
                    urec["n_exec"] += 1
                    with open(os.path.join(sd, fn), "w") as f:
                        f.write(passing_code)
                kept.append(fn)
            per_uut.append(urec)
            agg["n_cands"] += urec["n_cands"]
            agg["n_exec"] += urec["n_exec"]
            agg["exec_wall_s"] += urec["exec_wall_s"]
            agg["n_pytest_runs"] += urec["n_pytest_runs"]
        stmt = br = 0.0
        if kept:
            stmt, br, dt = coverage_json(tmp, pkg, kept)
            agg["exec_wall_s"] += dt
            agg["n_pytest_runs"] += 1
        agg["stmt"], agg["branch"] = stmt, br
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return agg, per_uut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--out-tag", default="eval", help="eval | eval_unified (R1.5 check)")
    args = ap.parse_args()

    with open(os.path.join(UUT_DIR, f"{args.subject}.json")) as f:
        uuts = json.load(f)
    gen_map, gen_stats = {}, []
    with open(args.gen) as f:
        for line in f:
            r = json.loads(line)
            gen_map[r["uut_id"]] = r["candidates"]
            gen_stats.append({"uut_id": r["uut_id"],
                              "gen_wall_s": r.get("gen_wall_s", 0.0),
                              "gen_tokens": r.get("gen_tokens", 0)})

    suite_root = f"{ROOT}/exp/results/suites/{args.method}/{args.subject}_seed{args.seed}"
    if args.out_tag != "eval":
        suite_root += "_" + args.out_tag
    out_path = f"{ROOT}/exp/results/{args.out_tag}/{args.method}/{args.subject}_seed{args.seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(suite_root, exist_ok=True)

    result = {"method": args.method, "subject": args.subject, "seed": args.seed,
              "gen_wall_s": sum(g["gen_wall_s"] for g in gen_stats),
              "gen_tokens": sum(g["gen_tokens"] for g in gen_stats)}

    if uuts and uuts[0].get("kind") == "standalone":
        node_src = None
        if args.subject == "quixbugs":
            npth = os.path.join(BENCH, "QuixBugs", "python_programs", "node.py")
            node_src = open(npth).read() if os.path.exists(npth) else None
        tasks = []
        for u in uuts:
            cands = gen_map.get(u["uut_id"], [])
            sd = os.path.join(suite_root, u["uut_id"])
            tasks.append((u, cands, sd, args.subject == "quixbugs",
                          node_src if u.get("needs_node") else None))
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            recs = list(ex.map(eval_standalone_uut, tasks))
        n = len(recs)
        result["per_uut"] = recs
        result["stmt"] = sum(r["stmt"] for r in recs) / n
        result["branch"] = sum(r["branch"] for r in recs) / n
        result["exec_wall_s"] = sum(r["exec_wall_s"] for r in recs)
        result["n_pytest_runs"] = sum(r["n_pytest_runs"] for r in recs)
        result["exec_rate"] = (sum(r["n_exec"] for r in recs) /
                               max(1, sum(r["n_cands"] for r in recs)))
        if args.subject == "quixbugs":
            result["ftr"] = 100.0 * sum(1 for r in recs if r["exposed"]) / n
    else:
        agg, per_uut = eval_project(uuts, gen_map, suite_root, args.subject)
        result.update(agg)
        result["per_uut"] = per_uut
        result["exec_rate"] = agg["n_exec"] / max(1, agg["n_cands"])

    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    brief = {k: v for k, v in result.items() if k != "per_uut"}
    print("[eval]", json.dumps(brief))


if __name__ == "__main__":
    main()
