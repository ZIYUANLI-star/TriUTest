"""Stateful marginal mutant-kill reward for SM-GRPO (paper Sec. 3.3).

Maintains a per-UUT mutant coverage state S[u]; the reward of a candidate test is
the fraction of *newly* killed mutants (marginal incremental kills), raised to a
power gamma, plus light stability / assertion-adequacy regularizers.
"""
import ast, hashlib, re
from typing import Dict, Set, Tuple

from .sandbox import run_pytest_with_code, run_mutation_kill_detail
from .rewarders_new import _basic_sanity, _assertion_richness_score

_FENCE = re.compile(r"```(?:python)?\s*\n([\s\S]*?)```", re.I)


def clean_candidate(text: str) -> str:
    """AST-level salvage of a raw policy sample: take the fenced block if any,
    then keep the longest parseable prefix (drops prose like 'Test Cases:'
    that models sometimes emit between test functions)."""
    m = _FENCE.search(text)
    code = (m.group(1) if m else text).strip()
    lines = code.splitlines()
    while lines:
        try:
            ast.parse("\n".join(lines))
            return "\n".join(lines)
        except SyntaxError as e:
            # cut at the offending line first, then fall back to trailing strip
            bad = (e.lineno or len(lines)) - 1
            if 0 <= bad < len(lines):
                lines = lines[:bad]
            else:
                lines.pop()
    return ""


_TB_LINE_PAT = re.compile(r"test_cut\.py:(\d+):")


def split_asserts(code: str, max_asserts: int = 30) -> str:
    """Per-assert test splitting (mirrors the evaluation pipeline, so the
    training reward sees the same granularity the final metrics use)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.splitlines()
    out_parts = []
    changed = False
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test")):
            out_parts.append("\n".join(lines[node.lineno - 1: node.end_lineno]))
            continue
        assert_idx = [i for i, st_ in enumerate(node.body) if isinstance(st_, ast.Assert)]
        if len(assert_idx) < 2 or len(assert_idx) > max_asserts:
            out_parts.append("\n".join(lines[node.lineno - 1: node.end_lineno]))
            continue
        changed = True
        for k, ai in enumerate(assert_idx):
            body_lines = []
            for i, st_ in enumerate(node.body):
                if isinstance(st_, ast.Assert) and i != ai:
                    continue
                body_lines.append("\n".join(lines[st_.lineno - 1: st_.end_lineno]))
            body = "\n".join(body_lines)
            out_parts.append(f"def {node.name}_a{k}():\n" +
                             (body if body.strip() else "    pass"))
    if not changed:
        return code
    out = "\n\n".join(out_parts)
    try:
        ast.parse(out)
        return out
    except SyntaxError:
        return code


def _count_asserts(code: str) -> int:
    try:
        return sum(1 for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Assert))
    except SyntaxError:
        return 0


def _drop_stmt_at_line(code: str, lineno: int) -> str:
    """Remove the innermost statement covering `lineno` (1-based). Returns ''
    if the result no longer parses or nothing was removed."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and hasattr(node, "lineno") and \
                node.lineno <= lineno <= (node.end_lineno or node.lineno):
            span = (node.end_lineno or node.lineno) - node.lineno
            if best is None or span < best[0]:
                best = (span, node.lineno, node.end_lineno or node.lineno)
    if best is None:
        return ""
    lines = code.splitlines()
    out = "\n".join(ln for i, ln in enumerate(lines, 1)
                    if not (best[1] <= i <= best[2]))
    try:
        ast.parse(out)
    except SyntaxError:
        return ""
    return out


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()[:16]


class StatefulMutantReward:
    def __init__(self, weights: Dict[str, float], time_budget_s: float = 25.0,
                 operators=None):
        self.w_mut = float(weights.get("mutant_kill", 1.5))
        self.w_stab = float(weights.get("stability", 0.2))
        self.w_ass = float(weights.get("assertion", 0.1))
        self.gamma = float(weights.get("_mutant_gamma", 1.7))
        self.pen_syntax = float(weights.get("_penalty_syntax", -1.0))
        self.pen_baseline = float(weights.get("_penalty_baseline", -0.5))
        self.time_budget_s = float(time_budget_s)
        self.operators = operators
        # state: cut_hash -> covered mutant ids
        self.state: Dict[str, Set[str]] = {}
        # cache: (cut_hash, test_hash) -> (killed_ids, total)
        self.kill_cache: Dict[Tuple[str, str], Tuple[Set[str], int]] = {}
        # cache: (cut_hash, cleaned_test_hash) -> (final_code | None, keep_ratio)
        self.prep_cache: Dict[Tuple[str, str], Tuple[object, float]] = {}

    # ---------- deterministic candidate preparation (shared by reward/update) --
    SALVAGE_ROUNDS = 6

    def _prepare(self, cut_code: str, test_code: str):
        """clean -> baseline run -> iterative statement-level salvage.

        Generated tests are typically one `def test():` with many asserts, so
        salvage works at assert granularity: locate the failing statement from
        the pytest traceback, drop it, retry (<= SALVAGE_ROUNDS times).
        Returns (final_code or None, keep_ratio) where keep_ratio is the
        fraction of original assert statements that survived.
        """
        code = clean_candidate(test_code)
        if not code or not _basic_sanity(code):
            return None, 0.0
        code = split_asserts(code)
        key = (_h(cut_code), _h(code))
        if key in self.prep_cache:
            return self.prep_cache[key]

        base_budget = min(self.time_budget_s * 0.2, 5.0)
        n0 = max(1, _count_asserts(code))
        # header offset added by the sandbox before the test body
        offset = 0 if "import CUT" in code else 2

        result = (None, 0.0)
        cur = code
        for _round in range(self.SALVAGE_ROUNDS + 1):
            try:
                r = run_pytest_with_code(cut_code, cur, timeout_s=base_budget)
            except Exception:
                break
            if r.get("ok", False):
                result = (cur, _count_asserts(cur) / n0)
                break
            if r.get("timeout"):
                break
            m = _TB_LINE_PAT.findall(r.get("out", "") + r.get("err", ""))
            if not m:
                break
            lineno = int(m[-1]) - offset
            nxt = _drop_stmt_at_line(cur, lineno)
            if not nxt or not _basic_sanity(nxt):
                break
            cur = nxt

        self.prep_cache[key] = result
        if len(self.prep_cache) > 8192:
            self.prep_cache.pop(next(iter(self.prep_cache)))
        return result

    # ---------- reward for one candidate ----------
    def __call__(self, cut_code: str, test_code: str) -> float:
        prepared, keep_ratio = self._prepare(cut_code, test_code)
        if prepared is None:
            # distinguish unparseable garbage from executable-but-wrong tests
            cleaned = clean_candidate(test_code)
            if not cleaned or not _basic_sanity(cleaned):
                return self.pen_syntax
            return self.pen_baseline
        test_code = prepared

        ck, tk = _h(cut_code), _h(test_code)
        key = (ck, tk)
        if key in self.kill_cache:
            killed, total = self.kill_cache[key]
        else:
            try:
                killed, total = run_mutation_kill_detail(
                    cut_code, test_code,
                    timeout_s=self.time_budget_s,
                    operators=self.operators,
                )
            except Exception:
                killed, total = set(), 0
            self.kill_cache[key] = (killed, total)
            if len(self.kill_cache) > 4096:
                self.kill_cache.pop(next(iter(self.kill_cache)))

        covered = self.state.get(ck, set())
        if total > 0:
            marginal = len(killed - covered) / total
        else:
            marginal = 0.0
        r_mut = marginal ** self.gamma if marginal > 0 else 0.0

        # stability: single deterministic re-run within remaining budget
        stab_budget = min(self.time_budget_s * 0.2, 5.0)
        try:
            r2 = run_pytest_with_code(cut_code, test_code, timeout_s=stab_budget)
            f_stab = 1.0 if bool(r2.get("ok", False)) else 0.0
        except Exception:
            f_stab = 0.0
        f_ass = _assertion_richness_score(test_code)

        # keep_ratio discounts candidates that needed test-level salvage, so a
        # fully-passing candidate always dominates a partially-broken one
        return float(keep_ratio * (self.w_mut * r_mut + self.w_stab * f_stab
                                   + self.w_ass * f_ass))

    # ---------- state update with the selected candidate ----------
    def update_state(self, cut_code: str, test_code: str):
        prepared, _keep = self._prepare(cut_code, test_code)
        if prepared is None:
            return
        ck, tk = _h(cut_code), _h(prepared)
        killed, total = self.kill_cache.get((ck, tk), (set(), 0))
        if total > 0 and killed:
            self.state.setdefault(ck, set()).update(killed)
