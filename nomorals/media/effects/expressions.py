"""Safe value expressions — EffectCraft ``expr`` crate's surface.

EffectCraft evaluates per-property value expressions in a JS engine.
Devon deliberately does not port that runtime (PARITY.md §1.3b: an
embedded JS runtime would be a second, weaker agent surface — the
brain is the scripting surface). This module is the honest
equivalent: a small deterministic expression language evaluated by
an AST walker — no ``eval``/``exec``, no attribute access, no
imports, no I/O. Names come from the caller's context (``time``,
``frame``, ``fps``, ``value`` …) and the function list below.
"""

from __future__ import annotations

import ast
import math
from typing import Any

__all__ = ["ExpressionError", "evaluate", "validate"]


class ExpressionError(ValueError):
    """The expression is not in the safe language, or failed to evaluate."""


def _wiggle(freq: float, amp: float, seed: float = 0.0, *, t: float) -> float:
    """Deterministic smooth noise in [-amp, amp].

    A sin-hash value-noise: the same (t, freq, amp, seed) always gives
    the same value, and nearby times give nearby values — the
    ``wiggle`` every motion artist expects, without a random state.
    """
    x = t * float(freq)
    h = math.sin(x * 12.9898 + float(seed) * 78.233) * 43758.5453
    frac = h - math.floor(h)
    smooth = frac * frac * (3.0 - 2.0 * frac)
    return float(amp) * (2.0 * smooth - 1.0)


def _loop(period: float, *, t: float) -> float:
    if period <= 0:
        raise ExpressionError("loop() needs a positive period")
    return t % float(period)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _lerp(a: float, b: float, u: float) -> float:
    return float(a) + (float(b) - float(a)) * float(u)


def _smooth(u: float) -> float:
    u = max(0.0, min(1.0, float(u)))
    return u * u * (3.0 - 2.0 * u)


_FUNCS: dict[str, Any] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sqrt": math.sqrt,
    "abs": abs,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
    "exp": math.exp,
    "log": math.log,
    "pow": pow,
    "clamp": _clamp,
    "lerp": _lerp,
    "smooth": _smooth,
    "loop": _loop,
    "wiggle": _wiggle,
}


def _eval(node: ast.AST, ctx: dict[str, float], t: float) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body, ctx, t)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(f"only numbers are allowed, got {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in ctx:
            return float(ctx[node.id])
        raise ExpressionError(f"unknown name {node.id!r}")
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, ctx, t)
        right = _eval(node.right, ctx, t)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ExpressionError("division by zero")
            return left / right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ExpressionError("modulo by zero")
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ExpressionError("division by zero")
            return left // right
        raise ExpressionError(f"operator {type(node.op).__name__} is not allowed")
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, ctx, t)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        raise ExpressionError(f"operator {type(node.op).__name__} is not allowed")
    if isinstance(node, ast.BoolOp):
        values = [_eval(v, ctx, t) for v in node.values]
        if isinstance(node.op, ast.And):
            return 1.0 if all(v != 0 for v in values) else 0.0
        if isinstance(node.op, ast.Or):
            return 1.0 if any(v != 0 for v in values) else 0.0
        raise ExpressionError(f"operator {type(node.op).__name__} is not allowed")
    if isinstance(node, ast.Compare):
        left = _eval(node.left, ctx, t)
        result = True
        for op, comp in zip(node.ops, node.comparators, strict=True):
            right = _eval(comp, ctx, t)
            if isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            else:
                raise ExpressionError(
                    f"comparison {type(op).__name__} is not allowed")
            result = result and ok
            left = right
        return 1.0 if result else 0.0
    if isinstance(node, ast.IfExp):
        cond = _eval(node.test, ctx, t)
        return _eval(node.body if cond != 0 else node.orelse, ctx, t)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only plain function calls are allowed")
        name = node.func.id
        if name not in _FUNCS:
            raise ExpressionError(f"unknown function {name!r}")
        if node.keywords:
            raise ExpressionError("keyword arguments are not allowed")
        args = [_eval(a, ctx, t) for a in node.args]
        fn = _FUNCS[name]
        try:
            if name in ("wiggle", "loop"):
                return float(fn(*args, t=t))
            return float(fn(*args))
        except ExpressionError:
            raise
        except Exception as exc:  # noqa: BLE001 - a bad call is an expression error
            raise ExpressionError(f"{name}({', '.join(map(str, args))}): {exc}") from exc
    raise ExpressionError(
        f"{type(node).__name__} is not in the expression language")


def validate(expr: str) -> None:
    """Parse-check ``expr``; raises :class:`ExpressionError` if unsafe."""
    if not isinstance(expr, str) or not expr.strip():
        raise ExpressionError("empty expression")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"cannot parse expression: {exc}") from exc
    _eval(tree, {"time": 0.0, "t": 0.0, "frame": 0.0, "fps": 24.0,
                 "value": 0.0}, 0.0)


def evaluate(expr: str, ctx: dict[str, float]) -> float:
    """Evaluate ``expr`` with ``ctx`` (``time``/``t`` default to 0)."""
    if not isinstance(expr, str) or not expr.strip():
        raise ExpressionError("empty expression")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"cannot parse expression: {exc}") from exc
    full = dict(ctx)
    t = float(full.get("time", full.get("t", 0.0)))
    full.setdefault("time", t)
    full.setdefault("t", t)
    return _eval(tree, full, t)
