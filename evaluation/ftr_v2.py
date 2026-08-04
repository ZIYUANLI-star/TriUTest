"""FTR v2: fully decoupled from pruning (fixes the undercount found in review).

For every candidate file (sanitized, UNPRUNED):
  run on buggy  -> (rc_b, failed_b: set of test ids)
  run on fixed  -> (rc_f, failed_f)
  exposed iff  (rc_b != 0 and rc_f == 0)                      # file-level
           or  (failed_b - failed_f != empty and rc_f >= 0)   # test-level

Reads existing generations; writes results/ftr_v2/<method>/quixbugs_seed<k>.json.
Never touches the original eval outputs.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_util import sanitize_test, strip_cut_redefs

ROOT = "/root/autodl-tmp/TriUTest"
UUT_DIR = f"{ROOT}/exp/uuts"
BENCH = "/root/autodl-tmp/benchmarks"
# 30s 足够暴露 QuixBugs 的死循环缺陷；过长会让"买错版超时"的候选串行拖满整个任务
TIMEOUT = 30
FAILED_PAT = re.compile(r"(?:FAILED|ERROR) [^:]+::(\w+)")


def run_pytest(cwd, fn):
    # Popen + 进程组：超时后 killpg 整组，避免残留子进程持有管道导致
    # capture_output 永久等待 EOF（曾使 ProcessPool 整体悬挂）。
    p = subprocess.Popen([sys.executable, "-m", "pytest", "-q", fn,
                          "-p", "no:cacheprovider", "--timeout", str(TIMEOUT)],
                         cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        out, _ = p.communicate(timeout=TIMEOUT + 30)
        return p.returncode, set(FAILED_PAT.findall(out or ""))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), 9)
        except Exception:
            p.kill()
        try:
            p.communicate(timeout=10)
        except Exception:
            pass
        return -9, set()


def eval_uut(args):
    u, cands, node_src = args
    rec = {"uut_id": u["uut_id"], "exposed": False, "n_checked": 0,
           "expose_via": None}
    tb = tempfile.mkdtemp(prefix="fb_")
    tf = tempfile.mkdtemp(prefix="ff_")
    try:
        open(os.path.join(tb, "cut.py"), "w").write(u["cut_code"])
        open(os.path.join(tf, "cut.py"), "w").write(u["cut_code_fixed"])
        if node_src:
            open(os.path.join(tb, "node.py"), "w").write(node_src)
            open(os.path.join(tf, "node.py"), "w").write(node_src)
        for i, c in enumerate(cands[:15]):
            code = sanitize_test(c, "from cut import *")
            if code:
                code = strip_cut_redefs(code, u["cut_code"])
            if not code:
                continue
            fn = f"test_g{i}.py"
            open(os.path.join(tb, fn), "w").write(code)
            rec["n_checked"] += 1
            rc_b, failed_b = run_pytest(tb, fn)
            if rc_b == 0:
                continue                     # nothing fails on buggy
            open(os.path.join(tf, fn), "w").write(code)
            rc_f, failed_f = run_pytest(tf, fn)
            if rc_f == 0:
                rec["exposed"] = True
                rec["expose_via"] = f"{fn}:file"
                break
            if rc_f > 0 and (failed_b - failed_f):
                rec["exposed"] = True
                rec["expose_via"] = f"{fn}:{sorted(failed_b - failed_f)[0]}"
                break
    finally:
        shutil.rmtree(tb, ignore_errors=True)
        shutil.rmtree(tf, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--jobs", type=int, default=40)
    args = ap.parse_args()

    gen = f"{ROOT}/exp/results/gen/{args.method}/quixbugs_seed{args.seed}.jsonl"
    out = f"{ROOT}/exp/results/ftr_v2/{args.method}/quixbugs_seed{args.seed}.json"
    if not os.path.exists(gen):
        print("no gen:", gen)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with open(os.path.join(UUT_DIR, "quixbugs.json")) as f:
        uuts = json.load(f)
    gen_map = {}
    for line in open(gen):
        r = json.loads(line)
        gen_map[r["uut_id"]] = r["candidates"]

    npth = os.path.join(BENCH, "QuixBugs", "python_programs", "node.py")
    node_src = open(npth).read() if os.path.exists(npth) else None

    tasks = [(u, gen_map.get(u["uut_id"], []),
              node_src if u.get("needs_node") else None) for u in uuts]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        recs = list(ex.map(eval_uut, tasks))
    ftr = 100.0 * sum(1 for r in recs if r["exposed"]) / len(recs)
    result = {"method": args.method, "seed": args.seed, "ftr": ftr,
              "n_programs": len(recs), "per_program": recs,
              "elapsed_s": round(time.time() - t0, 1)}
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[ftr_v2] {args.method} seed{args.seed}: FTR {ftr:.1f}")


if __name__ == "__main__":
    main()
