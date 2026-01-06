import subprocess, tempfile, os, shutil, signal, sys, re
import psutil

def _kill_process_tree(proc: subprocess.Popen):
    """Best-effort kill of a process and all its children (cross-platform)."""
    try:
        # First try psutil (most robust)
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for c in children:
            try:
                c.kill()
            except Exception:
                pass
        try:
            parent.kill()
        except Exception:
            pass
    except Exception:
        # Fallback: platform-specific group kill
        try:
            if os.name == "posix":
                # kill the whole process group
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            elif os.name == "nt":
                # send CTRL_BREAK to the process group, then hard-kill
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                proc.kill()
        except Exception:
            # last resort
            try:
                proc.kill()
            except Exception:
                pass

def run_pytest_with_code(cut_code: str, test_code: str, timeout_s=20):
    """
    在隔离的临时目录下写入 CUT 和测试代码，运行 pytest。
    超时将杀死进程树，避免僵尸子进程残留。
    """
    tmpdir = tempfile.mkdtemp(prefix="rt_")
    try:
        # 写入 CUT
        with open(os.path.join(tmpdir, "cut.py"), "w", encoding="utf-8") as f:
            f.write(cut_code)

        # 写入测试（自动统一导入 cut 为 CUT）
        test_path = os.path.join(tmpdir, "test_cut.py")
        fixed = test_code.replace("import CUT", "import cut as CUT") if "import CUT" in test_code \
                else "import cut as CUT\n" + test_code
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(fixed)

        # 运行 pytest（使用 python -m pytest，跨平台最稳）
        cmd = [sys.executable, "-m", "pytest", "-q", "--disable-warnings", "--maxfail=1", tmpdir]

        creationflags = 0
        preexec = None
        if os.name == "posix":
            preexec = os.setsid  # 新建进程组
        elif os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # Windows 新建进程组

        try:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tmpdir,
                preexec_fn=preexec,
                creationflags=creationflags
            )
        except OSError as e:
            # pytest 未安装或启动异常
            return {"ok": False, "timeout": False, "out": "", "err": str(e)}

        try:
            out, err = p.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_process_tree(p)
            return {"ok": False, "timeout": True, "out": "", "err": "timeout"}

        ok = (p.returncode == 0)
        return {"ok": ok, "timeout": False, "out": out, "err": err}

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)



# src/reward/sandbox.py
# def _which_mutpy():
#     """
#     优先用环境变量指定的解释器与脚本（适合 Windows 跨环境）：
#       - MUTPY_PYTHON: 指向 python.exe（3.9 环境）
#       - MUTPY_SCRIPT: 指向 mut.py 脚本
#     找不到再回退到 PATH 上的 mut.py 或 python -m mutpy。
#     """
#     from shutil import which
#     import os
#
#     py = os.environ.get("MUTPY_PYTHON")
#     mp = os.environ.get("MUTPY_SCRIPT")
#     if py and mp and os.path.exists(py) and os.path.exists(mp):
#         return [py, mp]
#
#     # 回退1：PATH 里有 mut.py
#     m = which("mut.py")
#     if m:
#         return [m]
#
#     # 回退2：python -m mutpy（当前进程的 python）
#     if which(sys.executable):
#         return [sys.executable, "-m", "mutpy"]
#
#     return None

def _which_mutpy(py_override: str = None, script_override: str = None):
    """
    解析 MutPy 启动命令，返回一个可直接用于 subprocess.Popen 的列表。
    优先级：
      1) 函数入参 py_override / script_override
      2) 环境变量 MUTPY_PYTHON / MUTPY_SCRIPT
      3) 硬编码默认路径 HARDCODED_PY / HARDCODED_MP
      4) PATH: mut.py
      5) python -m mutpy
    """
    import os, sys
    from shutil import which

    # ======== 在这里填你自己的硬编码路径 ========
    HARDCODED_PY = r"D:\ANACONDA\envs\MuTAP-master\python.exe"     # 示例：你的 3.9 解释器
    HARDCODED_MP = r"D:\ANACONDA\envs\MuTAP-master\Scripts\mut.py" # 示例：mut.py 的绝对路径
    # =========================================

    def _exists(p: str) -> bool:
        try:
            return bool(p) and os.path.exists(p)
        except Exception:
            return False

    def _wrap_if_py(script_path: str, py_exec: str = None):
        """如果是 .py 脚本，用给定或当前解释器包装；否则直接返回可执行路径。"""
        if not script_path:
            return None
        if script_path.lower().endswith(".py"):
            exe = py_exec if _exists(py_exec) else sys.executable
            return [exe, script_path]
        return [script_path]

    # 1) 入参覆盖
    if _exists(py_override) and _exists(script_override):
        return [py_override, script_override]
    if _exists(script_override):
        return _wrap_if_py(script_override, py_exec=py_override)

    # 2) 环境变量
    env_py = os.environ.get("MUTPY_PYTHON")
    env_mp = os.environ.get("MUTPY_SCRIPT")
    if _exists(env_py) and _exists(env_mp):
        return [env_py, env_mp]
    if _exists(env_mp):
        return _wrap_if_py(env_mp, py_exec=env_py)

    # 3) 硬编码默认
    if _exists(HARDCODED_PY) and _exists(HARDCODED_MP):
        print("aaaaa")
        return [HARDCODED_PY, HARDCODED_MP]
    if _exists(HARDCODED_MP):
        return _wrap_if_py(HARDCODED_MP, py_exec=HARDCODED_PY)

    # 4) PATH 里的 mut.py
    m = which("mut.py")
    if m:
        return _wrap_if_py(m)  # 若是 .py，用当前解释器包装

    # 5) 回退到当前解释器执行模块
    if which(sys.executable):
        return [sys.executable, "-m", "mutpy"]

    return None


# src/reward/sandbox.py
def _parse_mutpy_summary(text: str):
    """
    解析 MutPy 摘要，兼容多种格式：
      1) "Mutation score: K/T"
      2) "Total mutants: N" / "Killed mutants: M" / "Survived mutants: S"
      3) 新格式：
         - all: N
         - killed: M (...)
         - survived: S (...)
    """
    killed = total = survived = None
    for line in text.splitlines():
        # 旧格式 1：K/T
        m = re.search(r"Mutation\s*score.*?:\s*(\d+)\s*/\s*(\d+)", line, re.I)
        if m:
            killed = int(m.group(1)); total = int(m.group(2))

        # 旧格式 2：Total/Killed/Survived mutants
        m = re.search(r"(?:Total\s+mutants|Mutants)\s*:\s*(\d+)", line, re.I)
        if m: total = int(m.group(1))
        m = re.search(r"Killed\s+mutants\s*:\s*(\d+)", line, re.I)
        if m: killed = int(m.group(1))
        m = re.search(r"Survived\s+mutants\s*:\s*(\d+)", line, re.I)
        if m: survived = int(m.group(1))

        # 新格式 3：all / killed / survived
        m = re.search(r"^\s*-\s*all\s*:\s*(\d+)", line, re.I)
        if m: total = int(m.group(1))
        m = re.search(r"^\s*-\s*killed\s*:\s*(\d+)", line, re.I)
        if m: killed = int(m.group(1))
        m = re.search(r"^\s*-\s*survived\s*:\s*(\d+)", line, re.I)
        if m: survived = int(m.group(1))

    if killed is None and survived is not None and total is not None:
        killed = max(0, total - survived)
    if killed is not None and total and total > 0:
        return killed, total
    return None


def run_mutation_kill(cut_code: str, test_code: str, timeout_s=60):
    """
    使用 MutPy（mut.py）进行变异体生成与评测，返回 (kill_rate, total_mutants)。
    - MutPy 将直接运行测试评估击杀率（尽量用 pytest 运行器）。
    - 若本机不可用 mut.py 或解析失败，则回退到轻量文本级变异方案。
    """
    tmpdir = tempfile.mkdtemp(prefix="mutpy_")
    try:
        with open(os.path.join(tmpdir, "cut.py"), "w", encoding="utf-8") as f:
            f.write(cut_code)

        test_code_fixed = test_code if "import CUT" in test_code else "from cut import *\n" + test_code
        with open(os.path.join(tmpdir, "test_cut.py"), "w", encoding="utf-8") as f:
            f.write(test_code_fixed)

        mutpy_cmd = _which_mutpy()
        if mutpy_cmd is not None:
            cmd = mutpy_cmd + [
                "--runner", "pytest",
                "--target", "cut",
                "--unit-test", "test_cut",
                "--timeout", str(timeout_s)
            ]

            creationflags = 0
            preexec = None
            if os.name == "posix":
                preexec = os.setsid
            elif os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            try:
                p = subprocess.Popen(
                    cmd,
                    cwd=tmpdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=preexec,
                    creationflags=creationflags,
                )
                try:
                    out, err = p.communicate(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    _kill_process_tree(p)
                    return _fallback_mutation_kill(cut_code, test_code, timeout_s)
            except OSError:
                # 关键：Windows 上可执行错误（比如 WinError 193）直接回退
                return _fallback_mutation_kill(cut_code, test_code, timeout_s)

            parsed = _parse_mutpy_summary(out) or _parse_mutpy_summary(err)
            if parsed:
                killed, total = parsed
                rate = killed / total if total > 0 else 0.0
                return rate, total
            else:
                # 解析失败 → 回退
                return _fallback_mutation_kill(cut_code, test_code, timeout_s)
        else:
            # 环境没有 mut.py → 回退
            return _fallback_mutation_kill(cut_code, test_code, timeout_s)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)



# -----------------------------
# 轻量级回退（原始实现的改进版）
# -----------------------------
def _fallback_mutation_kill(cut_code: str, test_code: str, timeout_s=40):
    """
    回退策略：用简单的文本替换生成少量变异体并评估。保持与旧接口兼容。
    """
    mutants = []
    pairs = [
        ("+","-"),("-","+"),("*","//"),("//","*"),
        ("==","!="),("!=","=="),(">","<"),("<",">")
    ]
    # 生成最多 ~12 个变异体（每种最多两个位置）
    for a, b in pairs:
        idx = cut_code.find(a)
        if idx != -1:
            mutants.append(cut_code.replace(a, b, 1))
            # 再找第二处（简单策略）
            idx2 = cut_code.find(a, idx + 1)
            if idx2 != -1:
                mutants.append(cut_code[:idx2] + b + cut_code[idx2+len(a):])

    killed = 0
    total = min(len(mutants), 12)
    for i in range(total):
        mcode = mutants[i]
        r = run_pytest_with_code(mcode, test_code, timeout_s=timeout_s)
        if not r["ok"]:
            killed += 1
    rate = (killed / total) if total > 0 else 0.0
    return rate, total





cut_code ="""

def pysiphash(uint64):
    assert 0 <= uint64 < 1 << 64
    if uint64 > (1 << 63) - 1:
        int64 = uint64 - (1 << 64)
    else:
        int64 = uint64
    uint32 = (uint64 ^ uint64 >> 32) & 4294967295
    if uint32 > (1 << 31) - 1:
        int32 = uint32 - (1 << 32)
    else:
        int32 = uint32
    return int32, int64

"""


test_code="""

def test():
    assert pysiphash(0x0000000000000000) == (0, 0)
    assert pysiphash(0) == (0, 0)
    assert pysiphash(0) == (0, 0) == (0, 0)
    assert pysiphash(0x00000000000000000000000000000000) == (0, 0)
    assert pysiphash(0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000) == (0, 0)
"""
rate, total = run_mutation_kill(cut_code, test_code)
print(rate,total)