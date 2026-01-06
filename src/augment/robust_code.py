# 用AST做局部变量改名，只改函数体内的局部变量、临时变量、for目标、with as目标、赋值目标、参数

# src/augment/robust_code.py
import ast
import builtins
import io
import keyword
import random
import re
import tokenize
from typing import Dict, Iterable, List, Set, Tuple, Optional


# ---------------------------
# Utilities
# ---------------------------

_BUILTIN_NAMES: Set[str] = set(dir(builtins))
_RESERVED: Set[str] = _BUILTIN_NAMES | set(keyword.kwlist) | {"self", "cls"}

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DUNDER_RE = re.compile(r"^__.+__$")


def _flat_names_from_target(node: ast.AST) -> Iterable[str]:
    """Extract all Name ids from an assignment/target-like node."""
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            yield from _flat_names_from_target(elt)
    elif isinstance(node, ast.Starred):
        yield from _flat_names_from_target(node.value)
    # Do NOT touch attributes/subscripts


def _collect_all_names(node: ast.AST) -> Set[str]:
    """Collect all Name ids in a subtree (for collision avoidance)."""
    out: Set[str] = set()
    class V(ast.NodeVisitor):
        def visit_Name(self, n: ast.Name):
            out.add(n.id)
        # do visit everything
    V().visit(node)
    return out


def _collect_names_in_nested_functions(fn_node: ast.AST) -> Set[str]:
    """Collect names that appear inside nested FunctionDef/AsyncFunctionDef under a function node."""
    nested: Set[str] = set()
    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, n: ast.FunctionDef):
            # names used inside nested function bodies
            nested.update(_collect_all_names(n))
            # still recurse to collect deeper nesting
            self.generic_visit(n)

        def visit_AsyncFunctionDef(self, n: ast.AsyncFunctionDef):
            nested.update(_collect_all_names(n))
            self.generic_visit(n)
    V().visit(fn_node)
    return nested


# ---------------------------
# Function-local variable renamer (AST-based)
# ---------------------------

class _FuncLocalInfo(ast.NodeVisitor):
    """Collect candidate local names in one function, excluding risky ones."""
    def __init__(self):
        self.params: Set[str] = set()
        self.assigned: Set[str] = set()
        self.globals: Set[str] = set()
        self.nonlocals: Set[str] = set()
        self.nested_refs: Set[str] = set()

    def visit_arguments(self, a: ast.arguments):
        for arg in (a.posonlyargs + a.args + a.kwonlyargs):
            self.params.add(arg.arg)
        if a.vararg:
            self.params.add(a.vararg.arg)
        if a.kwarg:
            self.params.add(a.kwarg.arg)

    def visit_Global(self, n: ast.Global):
        self.globals.update(n.names)

    def visit_Nonlocal(self, n: ast.Nonlocal):
        self.nonlocals.update(n.names)

    def visit_Assign(self, n: ast.Assign):
        for t in n.targets:
            self.assigned.update(_flat_names_from_target(t))
        self.generic_visit(n)

    def visit_AnnAssign(self, n: ast.AnnAssign):
        if n.target is not None:
            self.assigned.update(_flat_names_from_target(n.target))
        self.generic_visit(n)

    def visit_AugAssign(self, n: ast.AugAssign):
        self.assigned.update(_flat_names_from_target(n.target))
        self.generic_visit(n)

    def visit_For(self, n: ast.For):
        self.assigned.update(_flat_names_from_target(n.target))
        self.generic_visit(n)

    def visit_AsyncFor(self, n: ast.AsyncFor):
        self.assigned.update(_flat_names_from_target(n.target))
        self.generic_visit(n)

    def visit_With(self, n: ast.With):
        for it in n.items:
            if it.optional_vars:
                self.assigned.update(_flat_names_from_target(it.optional_vars))
        self.generic_visit(n)

    def visit_AsyncWith(self, n: ast.AsyncWith):
        for it in n.items:
            if it.optional_vars:
                self.assigned.update(_flat_names_from_target(it.optional_vars))
        self.generic_visit(n)


def _make_stable_name(base: str, used: Set[str], rng: random.Random) -> str:
    """Generate a fresh name not colliding with 'used'."""
    # keep short and readable; ensure not reserved/dunder
    while True:
        suffix = rng.randint(0, 1_000_000)
        cand = f"v_{suffix:x}"
        if cand not in used and cand not in _RESERVED and not _DUNDER_RE.match(cand):
            return cand


class _ScopedRenamer(ast.NodeTransformer):
    """
    Rename local variables safely within functions.
    - Only inside functions (not at module/class level).
    - Do not touch attributes, imports, keywords, strings, comments (AST strips comments anyway).
    - Exclude globals/nonlocals/inner-function referenced names.
    """
    def __init__(self, rng: random.Random, max_renames: int = 8, rename_params: bool = True):
        super().__init__()
        self.rng = rng
        self.max_renames = max_renames
        self.rename_params = rename_params
        self.scope_stack: List[Dict[str, str]] = []

    # ---- apply mapping in current scope ----
    def _current_map(self) -> Optional[Dict[str, str]]:
        return self.scope_stack[-1] if self.scope_stack else None

    def visit_Name(self, node: ast.Name) -> ast.AST:
        m = self._current_map()
        if m and node.id in m:
            return ast.copy_location(ast.Name(id=m[node.id], ctx=node.ctx), node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        m = self._current_map()
        if m and node.arg in m:
            return ast.copy_location(ast.arg(arg=m[node.arg], annotation=node.annotation, type_comment=node.type_comment), node)
        return node

    # ---- scope entry ----
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_function_like(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_function_like(node)

    def _visit_function_like(self, node):
        # Gather local info
        info = _FuncLocalInfo()
        info.visit(node.args)
        for stmt in node.body:
            info.visit(stmt)

        # Names used in nested functions -> exclude
        info.nested_refs = _collect_names_in_nested_functions(node)

        # Existing names in this subtree (for collision avoidance)
        existing = _collect_all_names(node)

        # Build candidate set
        candidates: Set[str] = set(info.assigned)
        if self.rename_params:
            candidates |= set(info.params)

        # Exclusions
        def _safe(name: str) -> bool:
            return (
                name
                and _NAME_RE.match(name)
                and name not in _RESERVED
                and not _DUNDER_RE.match(name)
                and name not in info.globals
                and name not in info.nonlocals
                and name not in info.nested_refs
            )

        cand_list = [n for n in candidates if _safe(n)]
        self.rng.shuffle(cand_list)
        cand_list = cand_list[: min(self.max_renames, len(cand_list))]

        # Build mapping; avoid collisions with any existing name
        mapping: Dict[str, str] = {}
        used = set(existing)
        for old in cand_list:
            new = _make_stable_name(old, used, self.rng)
            mapping[old] = new
            used.add(new)

        # Push mapping for this scope
        self.scope_stack.append(mapping)
        # Transform body (and defaults/returns/annotations)
        node = self.generic_visit(node)
        # Pop
        self.scope_stack.pop()
        return node


def rename_vars_safe(py: str, seed: Optional[int] = None, max_renames: int = 8, rename_params: bool = True) -> str:
    """
    AST-safe variable renaming within function scopes.
    - Does NOT rename: module-level names, class names, function names, attributes, imports, strings/comments.
    - Skips: builtins, keywords, dunder, self/cls, globals/nonlocals, names referenced by nested functions.
    """
    try:
        tree = ast.parse(py)
    except SyntaxError:
        return py  # don't touch invalid code

    rng = random.Random(seed) if seed is not None else random
    renamer = _ScopedRenamer(rng=rng, max_renames=max_renames, rename_params=rename_params)
    new_tree = renamer.visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)  # Python 3.9+
    except Exception:
        # Fallback: return original if unparsing fails
        return py


# ---------------------------
# Comment & spacing insertion (tokenize-safe)
# ---------------------------

def _multiline_string_lines(py: str) -> Set[int]:
    """Return 1-based line numbers that are inside a multi-line STRING token."""
    lines: Set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(py).readline):
            if tok.type == tokenize.STRING:
                sline, scol = tok.start
                eline, ecol = tok.end
                if eline > sline:  # multi-line string
                    for ln in range(sline, eline + 1):
                        lines.add(ln)
    except tokenize.TokenError:
        pass
    return lines


def add_comments_spacing_safe(py: str, p_before: float = 0.2, p_after: float = 0.2, seed: Optional[int] = None) -> str:
    """
    Insert harmless comments and blank lines.
    - Avoid inserting inside multi-line strings.
    - Preserve indentation for comment lines.
    """
    rng = random.Random(seed) if seed is not None else random
    lines = py.splitlines()
    string_lines = _multiline_string_lines(py)
    out: List[str] = []

    for i, ln in enumerate(lines, start=1):
        indent = re.match(r"\s*", ln).group(0)
        safe = i not in string_lines

        if safe and rng.random() < p_before:
            notes = " ".join(["note"] * rng.randint(1, 3))
            out.append(f"{indent}# {notes}")

        out.append(ln)

        if safe and rng.random() < p_after:
            out.append("")  # blank line

    return "\n".join(out)


# ---------------------------
# Public API
# ---------------------------

def robust_augment(py: str,
                   seed: Optional[int] = None,
                   p_rename: float = 0.6,
                   p_comments: float = 0.5,
                   max_renames: int = 6,
                   rename_params: bool = True) -> str:
    """
    Apply robust, semantics-preserving augmentations:
      1) AST-safe variable renaming in function scopes (prob p_rename).
      2) Tokenize-safe comment/blank-line insertion (prob p_comments).
    """
    rng = random.Random(seed) if seed is not None else random
    if rng.random() < p_rename:
        py = rename_vars_safe(py, seed=rng.randint(0, 1_000_000),
                              max_renames=max_renames, rename_params=rename_params)
    if rng.random() < p_comments:
        py = add_comments_spacing_safe(py, seed=rng.randint(0, 1_000_000))
    return py
