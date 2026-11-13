"""Static AST analysis of agent-generated candidate code.

The guard rejects, *before anything is executed*:

* imports of forbidden modules (``os``, ``subprocess``, ``sys``, ``shutil``,
  ``socket`` plus other I/O, reflection and process modules);
* any reference to ``trusted_eval`` (import, name, attribute or string);
* dangerous builtins (``eval``, ``exec``, ``open``, ``__import__`` ...),
  file-I/O attribute calls (``.load``, ``.save`` ...) and dunder attribute
  access used for sandbox escapes;
* top-level executable statements (only imports, defs, classes and
  assignments are allowed at module level);
* a missing or wrongly-shaped ``preprocess_train`` / ``build_model`` interface.

The guard is defence in depth, not a complete security boundary: the Docker
image (run with ``--network none`` and a read-only filesystem) provides the
OS-level isolation layer.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional

FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "sys", "shutil", "socket",
    "ctypes", "importlib", "inspect", "gc", "pickle", "marshal", "builtins",
    "pathlib", "glob", "tempfile", "multiprocessing", "threading", "signal",
    "urllib", "http", "requests", "ftplib", "smtplib", "webbrowser", "runpy",
    "pty", "resource", "atexit", "trusted_eval", "sandbox", "auditor",
})
CORE_FORBIDDEN_NAMES = frozenset({"os", "subprocess", "sys", "shutil", "socket"})
FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
})
FORBIDDEN_ATTRS = frozenset({
    "system", "popen", "load", "save", "savez", "savez_compressed", "fromfile",
    "tofile", "loadtxt", "savetxt", "genfromtxt", "memmap", "read_csv", "to_csv",
})
ALLOWED_DUNDERS = frozenset({"__init__", "__call__", "__name__", "__len__", "__getitem__"})
PROTECTED_NAME = "trusted_eval"
REQUIRED_FUNCTIONS = {"preprocess_train": 2, "build_model": 2}


class SecurityViolation(Exception):
    """Raised by :func:`assert_safe` when candidate code violates the guard."""

    def __init__(self, report: "GuardReport") -> None:
        super().__init__(report.format())
        self.report = report


@dataclass
class GuardReport:
    """Outcome of a guard run."""

    violations: List[str] = field(default_factory=list)
    interface_errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no violation or interface error was found."""
        return not self.violations and not self.interface_errors

    def format(self) -> str:
        """Human/LLM-readable description used as a sandbox log."""
        lines = ["SecurityViolation: candidate rejected by AST guard"] if self.violations else []
        lines += [f"  - {v}" for v in self.violations]
        if self.interface_errors:
            lines.append("InterfaceError: candidate does not match the required interface")
            lines += [f"  - {e}" for e in self.interface_errors]
        return "\n".join(lines)


class _Guard(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: List[str] = []

    def _add(self, node: ast.AST, msg: str) -> None:
        self.violations.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in FORBIDDEN_MODULES:
                self._add(node, f"forbidden import '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if node.level:
            self._add(node, "relative imports are not allowed")
        if root in FORBIDDEN_MODULES:
            self._add(node, f"forbidden import from '{node.module}'")
        for alias in node.names:
            if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                self._add(node, f"forbidden import of '{alias.name}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
            self._add(node, f"forbidden builtin call '{func.id}()'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRS:
            self._add(node, f"forbidden attribute '.{node.attr}'")
        if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in ALLOWED_DUNDERS:
            self._add(node, f"forbidden dunder attribute '.{node.attr}'")
        if node.attr == PROTECTED_NAME:
            self._add(node, f"reference to protected package '{PROTECTED_NAME}'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == PROTECTED_NAME:
            self._add(node, f"reference to protected package '{PROTECTED_NAME}'")
        elif node.id in CORE_FORBIDDEN_NAMES:
            self._add(node, f"reference to forbidden module name '{node.id}'")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and PROTECTED_NAME in node.value:
            self._add(node, f"string mentions protected package '{PROTECTED_NAME}'")


def _check_interface(tree: ast.Module) -> List[str]:
    errors: List[str] = []
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, n_args in REQUIRED_FUNCTIONS.items():
        fn = funcs.get(name)
        if fn is None:
            errors.append(f"missing top-level function '{name}'")
        elif len(fn.args.args) != n_args:
            errors.append(f"'{name}' must take exactly {n_args} positional arguments")
    return errors


def _check_top_level(tree: ast.Module) -> List[str]:
    allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
    out = []
    for node in tree.body:
        is_doc = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        if not isinstance(node, allowed) and not is_doc:
            out.append(f"line {node.lineno}: top-level executable statement '{type(node).__name__}' is not allowed")
    return out


def check_code(source: str, require_interface: bool = True) -> GuardReport:
    """Statically analyse candidate source code.

    Args:
        source: Python source text.
        require_interface: Also verify ``preprocess_train`` and ``build_model`` exist.

    Returns:
        A :class:`GuardReport`; ``report.ok`` is True only for safe, well-formed code.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return GuardReport(violations=[f"SyntaxError at line {exc.lineno}: {exc.msg}"])
    guard = _Guard()
    guard.visit(tree)
    guard.violations.extend(_check_top_level(tree))
    interface = _check_interface(tree) if require_interface else []
    return GuardReport(violations=guard.violations, interface_errors=interface)


def assert_safe(source: str, require_interface: bool = True) -> None:
    """Raise :class:`SecurityViolation` unless ``source`` passes the guard."""
    report = check_code(source, require_interface=require_interface)
    if not report.ok:
        raise SecurityViolation(report)


def first_violation(source: str) -> Optional[str]:
    """Return the first violation message, or ``None`` if the code is clean."""
    report = check_code(source, require_interface=False)
    return report.violations[0] if report.violations else None
