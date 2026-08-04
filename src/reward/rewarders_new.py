# rewarders.py
from .sandbox import run_pytest_with_code, run_mutation_kill
from ..augment.robust_code import robust_augment
import time, statistics, random, re
from typing import Dict

# -----------------------------
# 一些轻量工具
# -----------------------------
_TEST_FN_PATTERN = re.compile(
    r"\bdef\s+(?:test\s*\(|test_[A-Za-z0-9_]*\s*\()"
)

def _has_assert(code: str) -> bool:
    return re.search(r"\bassert\b", code) is not None


def _has_test_fn(code: str) -> bool:
    return _TEST_FN_PATTERN.search(code) is not None

def _basic_sanity(code: str) -> bool:
    # 需要：至少一个 assert，且有 def test() 或 def test_xxx()
    return _has_assert(code) and _has_test_fn(code)

def _repair_test_minimal(test_code: str) -> str:
    """
    克制的“最小修复”：
      1) 如无导入，则补 'import cut as CUT'
      2) 如无 test 函数，把顶层 assert 收拢到 def test(): 里
    """
    fixed = test_code



    if not _has_test_fn(fixed):
        lines, body = [], []
        for ln in fixed.splitlines():
            if ln.strip().startswith("assert "):
                body.append("    " + ln.strip())
            else:
                lines.append(ln)
        if body:
            lines.append("\n\ndef test():")
            lines.extend(body)
        fixed = "\n".join(lines)

    return fixed

def _assertion_richness_score(test_code: str) -> float:
    """
    粗粒度“断言丰富度” 0~1：
      - 多种断言/性质：==, !=, in, is None, len(), raises, boundary(0,-1,"",[])
    """
    text = test_code
    feats = 0
    cap = 10  # 归一化上限

    # 出现一个 assert 以上才计分
    if not _has_assert(text):
        return 0.0

    # 多样性计数（是否出现）
    patterns_bool = [
        r"==", r"!=", r"\bin\s", r"is\s+None", r"is\s+not\s+None",
        r"len\s*\(", r"pytest\.raises", r"\.startswith\s*\(", r"\.endswith\s*\("
    ]
    feats += sum(1 for p in patterns_bool if re.search(p, text))

    # 边界字面量计数
    patterns_boundary = [
        r"\b0\b", r"\b-1\b", r"\b1\b", r'""', r"''", r"\[\]", r"\{\}", r"set\s*\(\s*\)"
    ]
    feats += sum(1 for p in patterns_boundary if re.search(p, text))

    # 断言数量（>1 时再加分）
    n_assert = len(re.findall(r"\bassert\s", text))
    feats += max(0, min(n_assert - 1, 4))  # 多写一些断言

    # 归一化到 0~1
    return max(0.0, min(1.0, feats / cap))

# -----------------------------
# 1) 变异杀伤率（带非线性放大）
# -----------------------------
def reward_mutant_kill(
    cut_code: str,
    test_code: str,
    time_budget_s: float = 25.0,
    max_mutants: int = 12,
    nonlinear_gamma: float = 1.0,
) -> float:
    """
    - 限制评测时间预算
    - total==0 / 异常 → 0.0
    - 非线性放大：rate -> rate ** gamma
    - 兼容：底层不支持 max_mutants 时自动回退
    """
    try:
        kwargs = dict(timeout_s=min(40.0, float(time_budget_s)))

        # 先带 max_mutants 调用；不支持则回退
        try:
            kwargs["max_mutants"] = int(max_mutants)
            rate, total = run_mutation_kill(cut_code, test_code, **kwargs)
        except TypeError:
            kwargs.pop("max_mutants", None)
            rate, total = run_mutation_kill(cut_code, test_code, **kwargs)

        if not total or rate is None:
            return 0.0

        rate = max(0.0, min(1.0, float(rate)))
        if nonlinear_gamma and nonlinear_gamma > 1.0:
            rate = rate ** float(nonlinear_gamma)
        return float(rate)
    except Exception:
        return 0.0

# -------------------------------------
# 2) 速度 + 稳定性
# -------------------------------------
def reward_speed_stability(
    cut_code: str,
    test_code: str,
    time_budget_s: float = 10.0,
    repeat: int = 3,
) -> float:
    outcomes, runtimes = [], []
    per_run_budget = max(0.001, float(time_budget_s))

    for _ in range(max(1, int(repeat))):
        t0 = time.time()
        try:
            r = run_pytest_with_code(cut_code, test_code, timeout_s=per_run_budget)
            ok = bool(r.get("ok", False))
        except Exception:
            ok = False
        dt = min(time.time() - t0, per_run_budget)
        outcomes.append(1 if ok else 0)
        runtimes.append(float(dt))

    med_rt = statistics.median(runtimes) if runtimes else per_run_budget
    med_rt = min(max(0.0, med_rt), per_run_budget)

    speed_score = max(0.0, 1.0 - med_rt / max(1e-6, per_run_budget))
    pass_rate = sum(outcomes) / max(1, len(outcomes))
    if len(outcomes) > 1:
        mean = pass_rate
        var = sum((x - mean) ** 2 for x in outcomes) / (len(outcomes) - 1)
        consistency_penalty = min(0.2, var)
    else:
        consistency_penalty = 0.0

    stability_score = max(0.0, pass_rate - consistency_penalty)
    return float(0.5 * speed_score + 0.5 * stability_score)

# -----------------------------------
# 3) 鲁棒不变性（等价扰动后的通过率）
# -----------------------------------
def reward_robust_invariance(
    cut_code: str,
    test_code: str,
    time_budget_s: float = 10.0,
    trials: int = 2,
    seed: int = 1234,
) -> float:
    per_run_budget = max(0.001, float(time_budget_s))
    try:
        base = run_pytest_with_code(cut_code, test_code, timeout_s=per_run_budget)
        if not bool(base.get("ok", False)):
            return 0.0
    except Exception:
        return 0.0

    rnd = random.Random(int(seed))
    oks = 0
    trials = max(1, int(trials))

    for _ in range(trials):
        aug_seed = rnd.randint(0, 2**31 - 1)
        try:
            try:
                aug = robust_augment(cut_code, seed=aug_seed)
            except TypeError:
                aug = robust_augment(cut_code)

            r = run_pytest_with_code(aug, test_code, timeout_s=per_run_budget)
            ok = bool(r.get("ok", False))
        except Exception:
            ok = False
        oks += 1 if ok else 0

    return float(oks / trials)

# -----------------------------------
# 4) 组合奖励（总预算调度 + 闸门 + 断言丰富度）
# -----------------------------------
def composite_reward(
    cut_code: str,
    test_code: str,
    weights: Dict[str, float],
    time_budget_s: float = 20.0,
    repeat_runs: int = 2,
    invariance_aug: bool = True,
    max_score: float = 2.0,
) -> float:
    """
    - GateA：测试结构/语法不过 → penalty_syntax
    - GateB：未变异基线不通过 → penalty_baseline
    - 子奖励：变异杀伤(非线性放大) / 速度稳定 / 不变性 / 断言丰富度
    - 时间：闸门占一小部分预算，其余按权重占比切分
    """
    # ---- 可调超参（从 weights 读取，不影响旧配置）----
    GAMMA = float(weights.get("_mutant_gamma", 1.7))
    MAX_MUTANTS = int(weights.get("_max_mutants", 16))
    PENALTY_SYNTAX = float(weights.get("_penalty_syntax", -1.0))
    PENALTY_BASELINE = float(weights.get("_penalty_baseline", -0.5))

    # ---- GateA：最小修复 + 结构检查 ----
    tc_fixed = test_code
    if not _basic_sanity(tc_fixed):
        return PENALTY_SYNTAX

    # ---- GateB：基线必须先通过 ----
    base_budget = max(0.001, min(float(time_budget_s) * 0.2, 5.0))

    try:
        base = run_pytest_with_code(cut_code, tc_fixed, timeout_s=base_budget)
        if not bool(base.get("ok", False)):
            return PENALTY_BASELINE
    except Exception:
        return PENALTY_BASELINE

    # ---- 权重与预算 ----
    w_m = float(weights.get("mutant_kill", 0.0))
    w_s = float(weights.get("speed", 0.0))
    w_st = float(weights.get("stability", 0.0))   # 与 speed 合并
    w_r = float(weights.get("robustness", 0.0)) if invariance_aug else 0.0
    w_a = float(weights.get("assertion", 0.0))    # 新增：断言丰富度

    enabled = {
        "mutant_kill": w_m > 0.0,
        "speed_stab": (w_s > 0.0) or (w_st > 0.0),
        "robust": w_r > 0.0,
        "assertion": w_a > 0.0,
    }
    parts = {
        "mutant_kill": w_m if enabled["mutant_kill"] else 0.0,
        "speed_stab": (w_s + w_st) if enabled["speed_stab"] else 0.0,
        "robust": w_r if enabled["robust"] else 0.0,
        "assertion": 0.0,  # 静态评分，不吃时间
    }
    total_part = sum(parts.values()) or 1.0

    total_budget = max(0.001, float(time_budget_s) - base_budget)
    budgets = {name: total_budget * (p / total_part) for name, p in parts.items()}

    score = 0.0

    # 1) 变异杀伤（非线性）
    if enabled["mutant_kill"]:
        try:
            s_m = reward_mutant_kill(
                cut_code, tc_fixed,
                time_budget_s=budgets["mutant_kill"],
                max_mutants=MAX_MUTANTS,
                nonlinear_gamma=GAMMA,
            )
        except Exception:
            s_m = 0.0
        score += w_m * float(s_m)

    # 2) 速度 + 稳定性
    if enabled["speed_stab"]:
        try:
            s_ss = reward_speed_stability(
                cut_code, tc_fixed,
                time_budget_s=budgets["speed_stab"],
                repeat=max(1, int(repeat_runs)),
            )
        except Exception:
            s_ss = 0.0
        score += (w_s + w_st) * float(s_ss)

    # 3) 鲁棒不变性
    if enabled["robust"]:
        try:
            s_r = reward_robust_invariance(
                cut_code, tc_fixed,
                time_budget_s=budgets["robust"],
                trials=2,
                seed=1234,
            )
        except Exception:
            s_r = 0.0
        score += w_r * float(s_r)

    # 4) 断言丰富度（静态小权重）
    if enabled["assertion"]:
        try:
            s_a = _assertion_richness_score(tc_fixed)
        except Exception:
            s_a = 0.0
        score += w_a * float(s_a)

    # ---- 裁剪 ----
    max_score = float(max(0.1, max_score))
    score = float(max(0.0, min(score, max_score)))
    return score
