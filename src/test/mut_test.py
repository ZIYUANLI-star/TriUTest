# file: src/test/mut_test.py
import os, subprocess, tempfile, textwrap, shutil, sys

# 1) 直接写死路径（用你的实际路径；前面加 r 防止反斜杠转义）
MUTPY_PYTHON = r"D:\ANACONDA\envs\MuTAP-master\python.exe"
MUTPY_SCRIPT = r"D:\ANACONDA\envs\MuTAP-master\Scripts\mut.py"

print("MUTPY_PYTHON =", MUTPY_PYTHON, flush=True)
print("MUTPY_SCRIPT =", MUTPY_SCRIPT, flush=True)

# 基本检查
if not os.path.exists(MUTPY_PYTHON):
    raise FileNotFoundError(f"python.exe not found: {MUTPY_PYTHON}")
if not os.path.exists(MUTPY_SCRIPT):
    raise FileNotFoundError(f"mut.py not found: {MUTPY_SCRIPT}")

tmp = tempfile.mkdtemp(prefix="mutpy_check_")
try:
    with open(os.path.join(tmp, "cut.py"), "w", encoding="utf-8") as f:
        f.write(textwrap.dedent("""
from collections.abc import Sequence, Iterable, Iterator
from ast import stmt
def walk_body(body: Sequence[stmt]) -> Iterator[stmt]:
    
    for node in body:
        if isinstance(node, If):
            yield from walk_body(node.body)
            yield from walk_body(node.orelse)
        elif isinstance(node, Try):
            yield from walk_body(node.body)
            for h in node.handlers:
                yield from walk_body(h.body)
            yield from walk_body(node.orelse)
            yield from walk_body(node.finalbody)
        else:
            yield node

        """))
    with open(os.path.join(tmp, "test_cut.py"), "w", encoding="utf-8") as f:
        f.write(textwrap.dedent("""
from cut import *
def test():
    assert walk_body([]) == [] # Test case 1
            """))

    cmd = [MUTPY_PYTHON, MUTPY_SCRIPT,
           "--runner","pytest",
           "--target","cut",
           "--unit-test","test_cut",
           "--timeout","30"]
    # 为了可读性，把带空格的参数加引号打印出来
    fmt = " ".join([f'"{c}"' if " " in c else c for c in cmd])
    print("Running:", fmt, flush=True)

    p = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
    print("RET", p.returncode, flush=True)
    print("--- STDOUT ---\n", p.stdout, flush=True)
    print("--- STDERR ---\n", p.stderr, flush=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
