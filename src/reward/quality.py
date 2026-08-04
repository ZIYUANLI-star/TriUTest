"""Quality signals and failure-type classification for DQ-GKD (paper Sec. 3.2).

Implements:
  - classify_failure(): reproducible semantic / non-semantic failure rules based on
    pytest exit codes and JUnitXML reports (paper Table 1).
  - s_exec / s_ass: executability and assertion-adequacy signals (Eqs. 4-5).
  - quality_weights(): softmax quality weights over a candidate set (Eq. 6).
  - ast_feature_set() / jaccard_distance(): structural diversity features (Eqs. 7-8).
"""
import ast, math, os, re, subprocess, sys, tempfile, shutil
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Tuple

# ----------------------------------------------------------------------------
# Failure-type classification (paper Table 1)
# ----------------------------------------------------------------------------
NON_FAILURE = "non_failure"
SEMANTIC = "semantic"
NON_SEMANTIC = "non_semantic"


def classify_failure(exit_code: int, junit_xml: str = "", stdout: str = "") -> str:
    """Classify a pytest run following the reproducible rules of paper Table 1."""
    if exit_code == 0:
        return NON_FAILURE
    if exit_code in (2, 3, 4, 5):
        # interrupted / internal error / usage error / no tests collected
        return NON_SEMANTIC
    # exit_code == 1 (or unknown): inspect the JUnitXML report
    if junit_xml:
        try:
            root = ET.fromstring(junit_xml)
            has_failure = root.iter("failure") is not None and any(True for _ in root.iter("failure"))
            errors = list(root.iter("error"))
            if has_failure:
                return SEMANTIC
            for e in errors:
                when = e.get("when", "")
                if when in ("setup", "teardown"):
                    return NON_SEMANTIC
                msg = (e.get("message") or "") + (e.text or "")
                if "ImportError" in msg or "ModuleNotFoundError" in msg or "collection failure" in msg:
                    return NON_SEMANTIC
                # runtime <error>: attribute by nearest non-framework stack frame
                tb = e.text or ""
                frames = re.findall(r'File "([^"]+)"', tb) or re.findall(r"^(\S+\.py):\d+", tb, re.M)
                for fr in reversed(frames):
                    base = os.path.basename(fr)
                    if "site-packages" in fr or base.startswith("_pytest"):
                        continue
                    if base.startswith("test_"):
                        return NON_SEMANTIC   # error inside generated test code
                    return SEMANTIC           # error attributed to the target module
                return NON_SEMANTIC
            if errors:
                return NON_SEMANTIC
        except ET.ParseError:
            pass
    if stdout and ("ImportError" in stdout or "ERROR collecting" in stdout or
                   "ModuleNotFoundError" in stdout):
        return NON_SEMANTIC
    # exit 1 without a parseable report: assertion failures dominate this case
    return SEMANTIC


def run_pytest_classified(cut_code: str, test_code: str, timeout_s: float = 20,
                          extra_files: Dict[str, str] = None) -> Dict:
    """Run pytest with a JUnitXML report; return dict(exit, category, timeout)."""
    tmp = tempfile.mkdtemp(prefix="qc_")
    try:
        with open(os.path.join(tmp, "cut.py"), "w", encoding="utf-8") as f:
            f.write(cut_code)
        for name, content in (extra_files or {}).items():
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
                f.write(content)
        fixed = test_code if "import cut" in test_code or "from cut" in test_code \
            else "from cut import *\n" + test_code
        with open(os.path.join(tmp, "test_cut.py"), "w", encoding="utf-8") as f:
            f.write(fixed)
        xml_path = os.path.join(tmp, "junit.xml")
        try:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "test_cut.py",
                 "--junitxml", xml_path, "-p", "no:cacheprovider"],
                cwd=tmp, capture_output=True, text=True, timeout=timeout_s)
            exit_code, out = p.returncode, p.stdout + p.stderr
            timed_out = False
        except subprocess.TimeoutExpired:
            exit_code, out, timed_out = 2, "TIMEOUT", True
        xml = ""
        if os.path.exists(xml_path):
            xml = open(xml_path, encoding="utf-8").read()
        cat = NON_SEMANTIC if timed_out else classify_failure(exit_code, xml, out)
        return {"exit": exit_code, "category": cat, "timeout": timed_out}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------
# Quality signals (paper Eqs. 4-6)
# ----------------------------------------------------------------------------
S0_MIN_STMTS = 2       # minimum effective statement threshold s_0
ALPHA_ASS = 1.0        # alpha in Eq. (5)
ETA = (2.0, 1.0, 2.0)  # eta_1, eta_2, eta_3 in Eq. (6)

_ASSERT_LIKE = ("pytest.raises", "approx(", "assertEqual", "assertTrue", "assertRaises")


def _count_asserts_and_stmts(test_code: str) -> Tuple[int, int]:
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return 0, 0
    n_assert, n_stmt = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            n_assert += 1
        if isinstance(node, ast.With):
            src = ast.dump(node.items[0].context_expr) if node.items else ""
            if "raises" in src:
                n_assert += 1
        if isinstance(node, (ast.Expr, ast.Assign, ast.AugAssign, ast.Return,
                             ast.Assert, ast.Raise, ast.If, ast.For, ast.While, ast.With)):
            n_stmt += 1
    return n_assert, n_stmt


def s_exec(cut_code: str, test_code: str, timeout_s: float = 15) -> Tuple[float, str]:
    """Executability + quick sanity check. Returns (score, failure_category)."""
    try:
        ast.parse(test_code)
    except SyntaxError:
        return 0.0, NON_SEMANTIC
    r = run_pytest_classified(cut_code, test_code, timeout_s=timeout_s)
    if r["timeout"]:
        return 0.0, NON_SEMANTIC
    if r["exit"] in (0, 1) and r["category"] != NON_SEMANTIC:
        return 1.0, r["category"]
    if r["exit"] == 0:
        return 1.0, NON_FAILURE
    return 0.0, r["category"]


def assertion_density(test_code: str) -> Tuple[float, int]:
    n_a, n_s = _count_asserts_and_stmts(test_code)
    d = n_a / (n_s + 1e-6)
    return d, n_s


def s_ass_batch(test_codes: List[str]) -> List[float]:
    """Eq. (5): min-max normalized, clipped assertion adequacy within a candidate set."""
    ds, stmts = [], []
    for t in test_codes:
        d, n_s = assertion_density(t)
        ds.append(d)
        stmts.append(n_s)
    lo, hi = min(ds), max(ds)
    out = []
    for d, n_s in zip(ds, stmts):
        if n_s < S0_MIN_STMTS:
            out.append(0.0)
        else:
            norm = (d - lo) / (hi - lo) if hi > lo else 1.0
            out.append(max(0.0, min(1.0, ALPHA_ASS * norm)))
    return out


def quality_weights(execs: List[float], asses: List[float], fails: List[float]) -> List[float]:
    """Eq. (6): softmax over eta1*s_exec + eta2*s_ass - eta3*s_fail."""
    e1, e2, e3 = ETA
    logits = [e1 * a + e2 * b - e3 * c for a, b, c in zip(execs, asses, fails)]
    mx = max(logits)
    exps = [math.exp(l - mx) for l in logits]
    z = sum(exps) or 1.0
    return [x / z for x in exps]


# ----------------------------------------------------------------------------
# Structural diversity features (paper Eqs. 7-8)
# ----------------------------------------------------------------------------
def _bucket_const(v) -> str:
    if isinstance(v, bool):
        return f"bool:{v}"
    if isinstance(v, (int, float)):
        if v == 0:
            return "num:0"
        if v in (1, -1):
            return f"num:{v}"
        return "num:small" if abs(v) < 100 else "num:large"
    if isinstance(v, str):
        return "str:empty" if v == "" else "str:nonempty"
    if v is None:
        return "none"
    return type(v).__name__


def ast_feature_set(test_code: str) -> Set[str]:
    """phi(y): invocation-chain, assertion-structure and input-construction features."""
    feats: Set[str] = set()
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return feats
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (
                fn.attr if isinstance(fn, ast.Attribute) else "?")
            feats.add(f"call:{name}/{len(node.args)}")
            for a in node.args:
                if isinstance(a, ast.Constant):
                    feats.add(f"arg:{_bucket_const(a.value)}")
                elif isinstance(a, (ast.List, ast.Tuple)):
                    feats.add(f"arg:seq{len(a.elts)}")
                elif isinstance(a, ast.Dict):
                    feats.add("arg:dict")
        elif isinstance(node, ast.Assert):
            t = node.test
            if isinstance(t, ast.Compare):
                for op in t.ops:
                    feats.add(f"assert:cmp:{type(op).__name__}")
                for c in ast.walk(t):
                    if isinstance(c, ast.Constant):
                        feats.add(f"assert:val:{_bucket_const(c.value)}")
            elif isinstance(t, ast.Call):
                feats.add("assert:call")
            else:
                feats.add(f"assert:{type(t).__name__}")
        elif isinstance(node, ast.With):
            for item in node.items:
                src = ast.dump(item.context_expr)
                if "raises" in src:
                    exc = re.search(r"id='(\w+Error|\w+Exception)'", src)
                    feats.add(f"raises:{exc.group(1) if exc else '?'}")
    return feats


def jaccard_distance(f1: Set[str], f2: Set[str]) -> float:
    if not f1 and not f2:
        return 0.0
    inter = len(f1 & f2)
    union = len(f1 | f2)
    return 1.0 - inter / union if union else 0.0


def diversity_scores(test_codes: List[str]) -> Tuple[float, List[float]]:
    """Return (mean pairwise distance = -L_div, per-candidate distinctness)."""
    feats = [ast_feature_set(t) for t in test_codes]
    K = len(feats)
    if K < 2:
        return 0.0, [1.0] * K
    dists = [[0.0] * K for _ in range(K)]
    tot, cnt = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            d = jaccard_distance(feats[i], feats[j])
            dists[i][j] = dists[j][i] = d
            tot += d
            cnt += 1
    mean_pair = tot / cnt if cnt else 0.0
    distinct = [sum(row) / (K - 1) for row in dists]
    return mean_pair, distinct
