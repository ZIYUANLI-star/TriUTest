"""Build unified UUT (unit-under-test) lists for the four evaluation subjects.

Output: exp/uuts/{humaneval,quixbugs,codetiming,apimd}.json
Each entry:
  uut_id, subject, prompt, kind ("standalone"|"project"),
  cut_code (standalone), cut_code_fixed (quixbugs), module/project_root (project)
"""
import ast, gzip, json, os, sys

BENCH = "/root/autodl-tmp/benchmarks"
OUT = "/root/autodl-tmp/TriUTest/exp/uuts"
os.makedirs(OUT, exist_ok=True)

PROMPT_TMPL = (
    "Based on the function description and the code snippet below, please generate a "
    "comprehensive set of detailed test cases that cover typical usage, edge cases, and "
    "potential error conditions.\n\nFunction Description:\n{desc}\n\nCode Under Test:\n{code}\n\n"
)

def build_humaneval():
    path = os.path.join(BENCH, "human-eval", "data", "HumanEval.jsonl.gz")
    uuts = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            cut = t["prompt"] + t["canonical_solution"]
            # description: docstring inside prompt
            desc = t["entry_point"]
            try:
                tree = ast.parse(cut)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == t["entry_point"]:
                        d = ast.get_docstring(node)
                        if d:
                            desc = d.strip().split("\n\n")[0]
                        break
            except SyntaxError:
                pass
            uuts.append({
                "uut_id": t["task_id"].replace("/", "_"),
                "subject": "humaneval",
                "kind": "standalone",
                "prompt": PROMPT_TMPL.format(desc=desc, code=cut),
                "cut_code": cut,
                "entry_point": t["entry_point"],
            })
    return uuts

def build_quixbugs():
    root = os.path.join(BENCH, "QuixBugs")
    buggy_dir = os.path.join(root, "python_programs")
    fixed_dir = os.path.join(root, "correct_python_programs")
    uuts = []
    skip = {"node.py", "__init__.py"}
    names = sorted(fn for fn in os.listdir(fixed_dir)
                   if fn.endswith(".py") and fn not in skip and "_test" not in fn)
    for fn in names:
        bp = os.path.join(buggy_dir, fn)
        fp = os.path.join(fixed_dir, fn)
        if not os.path.exists(bp):
            continue
        buggy = open(bp, encoding="utf-8").read()
        fixed = open(fp, encoding="utf-8").read()
        name = fn[:-3]
        desc = name.replace("_", " ")
        try:
            tree = ast.parse(fixed)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    d = ast.get_docstring(node)
                    if d:
                        desc = d.strip()[:500]
                        break
        except SyntaxError:
            pass
        needs_node = "from node import" in buggy or "import node" in buggy
        uuts.append({
            "uut_id": f"quixbugs_{name}",
            "subject": "quixbugs",
            "kind": "standalone",
            "prompt": PROMPT_TMPL.format(desc=desc, code=buggy),
            "cut_code": buggy,
            "cut_code_fixed": fixed,
            "needs_node": needs_node,
        })
    return uuts

def _module_uuts(project_root, package, subject):
    uuts = []
    pkg_dir = os.path.join(project_root, package)
    if not os.path.isdir(pkg_dir):
        # package might be a single dir with different name
        raise RuntimeError(f"package dir not found: {pkg_dir}")
    for dirpath, _dirs, files in os.walk(pkg_dir):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fn)
            rel = os.path.relpath(fpath, project_root)
            modname = rel[:-3].replace(os.sep, ".")
            if modname.endswith(".__init__"):
                modname = modname[: -len(".__init__")]
            src = open(fpath, encoding="utf-8").read()
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            lines = src.splitlines()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name.startswith("_"):
                        continue
                    seg = "\n".join(lines[node.lineno - 1: node.end_lineno])
                    if len(seg) > 6000:
                        seg = seg[:6000]
                    d = ast.get_docstring(node) or node.name
                    desc = (f"Public API `{node.name}` in module `{modname}` of the "
                            f"`{package}` project. {d.strip()[:400]}")
                    code = f"# from module: {modname}\n{seg}"
                    uuts.append({
                        "uut_id": f"{subject}_{modname}.{node.name}".replace(".", "_"),
                        "subject": subject,
                        "kind": "project",
                        "prompt": PROMPT_TMPL.format(desc=desc, code=code),
                        "module": modname,
                        "target_name": node.name,
                        "project_root": project_root,
                        "package": package,
                    })
    return uuts

def main():
    all_out = {}
    all_out["humaneval"] = build_humaneval()
    all_out["quixbugs"] = build_quixbugs()
    all_out["codetiming"] = _module_uuts(os.path.join(BENCH, "codetiming"), "codetiming", "codetiming")
    all_out["apimd"] = _module_uuts(os.path.join(BENCH, "apimd"), "apimd", "apimd")
    for k, v in all_out.items():
        with open(os.path.join(OUT, f"{k}.json"), "w", encoding="utf-8") as f:
            json.dump(v, f, ensure_ascii=False, indent=1)
        print(k, len(v))

if __name__ == "__main__":
    main()
