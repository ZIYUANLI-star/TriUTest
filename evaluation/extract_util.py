"""Shared utilities: extract python test code from LLM output and sanitize it."""
import ast, re

FENCE_PAT = re.compile(r"```(?:python)?\s*\n([\s\S]*?)```", re.I)


def extract_code(text: str) -> str:
    """Take the first fenced code block if present, else raw text."""
    m = FENCE_PAT.search(text)
    code = m.group(1) if m else text
    return code.strip()


def _strip_trailing_garbage(code: str) -> str:
    """Drop trailing lines until the code parses (handles truncated generations)."""
    lines = code.splitlines()
    while lines:
        try:
            ast.parse("\n".join(lines))
            return "\n".join(lines)
        except SyntaxError:
            lines.pop()
    return ""


def prune_failing_tests(code: str, failing_names) -> str:
    """Remove failing top-level test functions from a test file (AST-based).

    Used for test-level (rather than file-level) filtering: a file where some
    regression assertions are environment-sensitive keeps its passing tests.
    Returns '' if nothing runnable remains.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    lines = code.splitlines()
    drop_ranges = []
    kept_tests = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name.startswith("test"):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            if node.name in failing_names:
                drop_ranges.append((start, node.end_lineno))
            else:
                kept_tests += 1
    if kept_tests == 0:
        return ""
    keep = [True] * len(lines)
    for s, e in drop_ranges:
        for i in range(s, min(e, len(lines))):
            keep[i] = False
    out = "\n".join(ln for ln, k in zip(lines, keep) if k)
    try:
        ast.parse(out)
    except SyntaxError:
        return ""
    return out


def split_asserts(code: str, max_asserts: int = 30) -> str:
    """Split every multi-assert test function into per-assert test functions.

    Applied uniformly to all methods: function-level pass/fail then acts at
    assertion granularity, so one wrong assertion no longer voids the whole
    monolithic `def test():` that the training corpus format induces.
    Variant k keeps ALL non-assert statements (setup/state flow preserved in
    original order) and only the k-th top-level assert.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.splitlines()
    out_parts = []
    changed = False
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test")):
            seg = "\n".join(lines[node.lineno - 1: node.end_lineno])
            out_parts.append(seg)
            continue
        assert_idx = [i for i, st_ in enumerate(node.body) if isinstance(st_, ast.Assert)]
        if len(assert_idx) < 2 or len(assert_idx) > max_asserts:
            seg = "\n".join(lines[node.lineno - 1: node.end_lineno])
            out_parts.append(seg)
            continue
        changed = True
        for k, ai in enumerate(assert_idx):
            body_lines = []
            for i, st_ in enumerate(node.body):
                if isinstance(st_, ast.Assert) and i != ai:
                    continue
                seg = "\n".join(lines[st_.lineno - 1: st_.end_lineno])
                body_lines.append(seg)
            hdr_indent = " " * node.body[0].col_offset
            fn = f"def {node.name}_a{k}():\n"
            body = "\n".join(body_lines)
            # body lines already carry their original indentation
            out_parts.append(fn + (body if body.strip() else hdr_indent + "pass"))
    if not changed:
        return code
    out = "\n\n".join(out_parts)
    try:
        ast.parse(out)
        return out
    except SyntaxError:
        return code


def strip_cut_redefs(code: str, cut_code: str) -> str:
    """Remove top-level re-definitions of CUT symbols from a test file.

    Some baselines (MuTAP-style prompts) inline the function under test into
    the generated file; that shadows `from cut import *`, so neither the buggy
    nor the fixed program would ever actually execute (coverage ~= 0, FTR = 0
    by construction). Dropping the inlined copies restores the standard
    protocol: tests must exercise the module under test."""
    try:
        cut_names = {n.name for n in ast.parse(cut_code).body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef))}
    except SyntaxError:
        return code
    if not cut_names:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.splitlines()
    keep = [True] * len(lines)
    dropped = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and node.name in cut_names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            for i in range(start, min(node.end_lineno, len(lines))):
                keep[i] = False
            dropped = True
    if not dropped:
        return code
    out = "\n".join(ln for ln, k in zip(lines, keep) if k)
    try:
        ast.parse(out)
    except SyntaxError:
        return code
    return out


def sanitize_test(code: str, import_line: str) -> str:
    """Return a parseable pytest file or '' if hopeless.

    import_line: e.g. 'from cut import *' or 'from codetiming import *'.
    """
    code = extract_code(code)
    code = _strip_trailing_garbage(code)
    if not code:
        return ""
    # keep any collectable test: oracle strength must be reflected by MKR/FTR,
    # not act as an eval-time gate (raises-based SBST tests carry no `assert`)
    has_oracle = ("assert" in code) or ("pytest.raises" in code) or ("raises(" in code)
    if not has_oracle:
        return ""
    # wrap top-level asserts into a test function if no test fn exists
    if not re.search(r"\bdef\s+test", code):
        lines, body = [], []
        for ln in code.splitlines():
            if ln.startswith("assert ") or ln.strip().startswith("assert "):
                body.append("    " + ln.strip())
            else:
                lines.append(ln)
        if body:
            lines.append("")
            lines.append("def test_generated():")
            lines.extend(body)
            code = "\n".join(lines)
        else:
            return ""
    code = split_asserts(code)
    header = import_line + "\nimport pytest\n\n"
    out = header + code
    try:
        ast.parse(out)
    except SyntaxError:
        return ""
    return out
