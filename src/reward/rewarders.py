from .sandbox import run_pytest_with_code, run_mutation_kill
from ..augment.robust_code import robust_augment
import time, statistics, random
from typing import Dict

# -----------------------------
# 1) 变异杀伤率奖励（稳健版）
# -----------------------------
def reward_mutant_kill(
    cut_code: str,
    test_code: str,
    time_budget_s: float = 25.0,
    max_mutants: int = 12,
) -> float:
    """
    - 限制评测时间预算（单项）
    - 可选限制参与统计的 mutant 数（若底层支持）
    - total==0 或异常 → 返回 0.0
    """
    t0 = time.time()
    try:
        kwargs = dict(timeout_s=min(40.0, float(time_budget_s)))
        # 若底层支持 max_mutants，可放开下一行；不支持会 TypeError，自动忽略
        try:
            kwargs["max_mutants"] = int(max_mutants)
        except Exception:
            pass

        rate, total = run_mutation_kill(cut_code, test_code, **kwargs)
        # 兜底：越界/异常值清洗
        if total is None or total <= 0:
            return 0.0
        if rate is None:
            return 0.0
        rate = float(max(0.0, min(1.0, rate)))
    except Exception:
        return 0.0

    # 强制遵守时间预算（这里仅作为统计节点，不追加惩罚）
    _ = time.time() - t0
    return float(rate)


# -------------------------------------
# 2) 速度 + 稳定性（合成奖励，鲁棒实现）
# -------------------------------------
def reward_speed_stability(
    cut_code: str,
    test_code: str,
    time_budget_s: float = 10.0,
    repeat: int = 3,
) -> float:
    """
    - 中位数耗时 → speed_score（越快越好）
    - 多次执行通过率 - 一致性惩罚 → stability_score
    - 异常/超时 → 该次记为失败，耗时按预算计
    """
    outcomes = []
    runtimes = []
    per_run_budget = max(0.001, float(time_budget_s))  # 防 0

    for _ in range(max(1, int(repeat))):
        t0 = time.time()
        try:
            r = run_pytest_with_code(cut_code, test_code, timeout_s=per_run_budget)
            ok = bool(r.get("ok", False))
        except Exception:
            ok = False
        dt = time.time() - t0
        # 运行时长截断到预算上方，避免极端值
        dt = min(dt, per_run_budget)

        outcomes.append(1 if ok else 0)
        runtimes.append(float(dt))

    med_rt = statistics.median(runtimes) if runtimes else per_run_budget
    med_rt = min(max(0.0, med_rt), per_run_budget)  # 0..budget

    # 速度分：1 - (中位数耗时/预算)
    speed_score = max(0.0, 1.0 - med_rt / max(1e-6, per_run_budget))

    pass_rate = sum(outcomes) / max(1, len(outcomes))
    # 一致性惩罚（方差上限 0.2）
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
    """
    - 先验证原始 CUT；不过直接 0.0
    - 对 CUT 做 `trials` 次等价扰动；不改全局 PRNG
    - 优先给 robust_augment 传 seed；若不支持则回退无 seed 版本
    - 异常/超时 → 该次记为失败
    """
    per_run_budget = max(0.001, float(time_budget_s))

    # 基线：原始 CUT 必须通过
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
            # 首选：robust_augment(cut, seed=...)
            try:
                aug = robust_augment(cut_code, seed=aug_seed)
            except TypeError:
                # 兼容老签名：无 seed；（注意：不可复现，建议尽快升级）
                aug = robust_augment(cut_code)

            r = run_pytest_with_code(aug, test_code, timeout_s=per_run_budget)
            ok = bool(r.get("ok", False))
        except Exception:
            ok = False
        oks += 1 if ok else 0

    return float(oks / trials)


# -----------------------------
# 4) 组合奖励（总预算调度）
# -----------------------------
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
    - 串行调用各子奖励，但遵守“总时间预算”
    - 预算分配：按启用权重占比进行比例切分
    - 最终分数裁剪到 [0, max_score]
    """
    # 读取与规范化权重
    w_m = float(weights.get("mutant_kill", 0.0))
    w_s = float(weights.get("speed", 0.0))
    w_st = float(weights.get("stability", 0.0))  # 与 speed 合并
    w_r = float(weights.get("robustness", 0.0)) if invariance_aug else 0.0

    # 哪些子项被启用（用于预算划分）
    enabled = {
        "mutant_kill": w_m > 0.0,
        "speed_stab": (w_s > 0.0) or (w_st > 0.0),
        "robust": w_r > 0.0,
    }
    # 用于预算比例的“权重”：与得分权重一致更直观
    parts = {
        "mutant_kill": w_m if enabled["mutant_kill"] else 0.0,
        "speed_stab": (w_s + w_st) if enabled["speed_stab"] else 0.0,
        "robust": w_r if enabled["robust"] else 0.0,
    }
    total_part = sum(parts.values()) or 1.0

    # 分配总预算
    total_budget = max(0.001, float(time_budget_s))
    budgets = {
        name: total_budget * (part / total_part)
        for name, part in parts.items()
    }

    score = 0.0

    # 1) 变异杀伤
    if enabled["mutant_kill"]:
        try:
            s_m = reward_mutant_kill(cut_code, test_code, time_budget_s=budgets["mutant_kill"])
        except Exception:
            s_m = 0.0
        score += w_m * float(s_m)

    # 2) 速度 + 稳定性
    if enabled["speed_stab"]:
        try:
            s_ss = reward_speed_stability(
                cut_code, test_code,
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
                cut_code, test_code,
                time_budget_s=budgets["robust"],
                trials=2,  # 可暴露到 YAML
                seed=1234,
            )
        except Exception:
            s_r = 0.0
        score += w_r * float(s_r)

    # 最终裁剪
    max_score = float(max(0.1, max_score))
    score = float(max(0.0, min(score, max_score)))
    return score
