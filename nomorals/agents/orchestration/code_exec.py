"""Safe in-process code execution for code-first tool calls.

The model writes a Python block calling tools as functions (loops and
conditionals in one block instead of N JSON round-trips).  This runner
AST-validates the code before executing it with a restricted namespace:

- no imports, no exec/eval/compile, no open(), no dunder access
- only the tool functions + a small safe-builtin set are visible
- each tool function routes through ``ToolAdapter.call()``, so the
  registry's capability gating and confirmations still apply

This is NOT a general sandbox — for arbitrary user code use
``nomorals.tools.sandbox_code`` (subprocess isolation).  This runner is
for model-generated glue code over a fixed tool namespace.
"""

from __future__ import annotations

import ast
from typing import Any, Callable

__all__ = ["SafeCodeRunner", "CodeSafetyError"]


class CodeSafetyError(Exception):
    """The code block failed AST validation."""


#: nodes that are never allowed in model-generated code
_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Lambda,
    ast.ClassDef,
)

#: calls that are never allowed, even if somehow reachable
_FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "__import__",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "input", "breakpoint", "exit", "quit", "help", "dir",
}

#: the only builtins model code may touch
_SAFE_BUILTINS = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "range": range, "enumerate": enumerate, "zip": zip,
    "min": min, "max": max, "sum": sum, "sorted": sorted,
    "abs": abs, "round": round, "any": any, "all": all,
    "isinstance": isinstance, "print": print,
}


def _validate(tree: ast.Module, tool_names: set[str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise CodeSafetyError(
                f"forbidden construct: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise CodeSafetyError("dunder/private attribute access blocked")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALLS:
                raise CodeSafetyError(f"forbidden call: {func.id}")
            if isinstance(func, ast.Attribute):
                raise CodeSafetyError("method calls on objects are blocked")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise CodeSafetyError("dunder name access blocked")
            if (node.id not in tool_names
                    and node.id not in _SAFE_BUILTINS
                    and node.id != "result"):
                # allow local variable assignment targets — checked below
                pass
    # collect assigned names; every other Name must be a tool/builtin
    assigned: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned.add(node.id)
        if isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                assigned.add(node.target.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if (node.id not in tool_names
                    and node.id not in _SAFE_BUILTINS
                    and node.id not in assigned
                    and node.id != "result"):
                raise CodeSafetyError(f"unknown name: {node.id!r}")


class SafeCodeRunner:
    """Execute model-generated tool-call code with AST validation."""

    def __init__(self, tools: dict[str, Callable[..., Any]]) -> None:
        self.tools = dict(tools)

    def run(self, code: str) -> tuple[bool, str]:
        """Run ``code``.  Returns (success, observation_text)."""
        code = (code or "").strip()
        if not code:
            return False, "empty code block"
        if len(code) > 8000:
            return False, "code block too long (max 8000 chars)"
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"syntax error: {exc}"
        try:
            _validate(tree, set(self.tools))
        except CodeSafetyError as exc:
            return False, f"code blocked: {exc}"

        namespace: dict[str, Any] = dict(self.tools)
        namespace["__builtins__"] = dict(_SAFE_BUILTINS)
        namespace["result"] = None
        try:
            exec(compile(tree, "<model-code>", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            return False, f"code raised {type(exc).__name__}: {exc}"
        result = namespace.get("result")
        if result is None:
            return True, "(code ran, no result set — assign to `result`)"
        return True, str(result)[:4000]
