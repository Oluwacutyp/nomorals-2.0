"""Computational-answer organ (52-websites site #13, WolframAlpha).

Devon answers computable questions through a ladder, per the
implementation-ladder doctrine — the brain formalises the query,
never the arithmetic itself:

* rung ``wolfram`` — the WolframAlpha v2/query API when an AppID is
  configured (``NM_WOLFRAM_APPID``): curated data, symbolic math,
  step-by-step pods, assumptions named. The reference's strength.
* rung ``sympy`` — a local CAS where the profile admits the
  dependency (sympy is import-guarded: a box without it degrades
  honestly to the stdlib rung, never to a fake answer).
* rung ``stdlib`` — exact arithmetic (``fractions``/``decimal``,
  never float), unit conversion, date arithmetic. No dependency at
  all, and exact where a float would lie (``0.1 + 0.2 == 0.3``).

Every answer carries its derivation (``steps``) and names its rung
and source; a question no rung can answer comes back
``ok=False`` with the reason and what would help — curated facts
stay unanswered rather than invented. ``Computation`` is the
record: ``{query_formal, steps, result, rung}``.
"""

from __future__ import annotations

import ast
import datetime as _dt
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from .wolfram import WolframClient

__all__ = [
    "Computation",
    "compute_answer",
    "format_compute_reply",
]

try:  # sympy is a profile-admitted dependency, not a hard one.
    import sympy as _sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        parse_expr,
        standard_transformations,
    )

    _SYMPY = True
except Exception:  # noqa: BLE001 — no sympy on this profile, stdlib only
    _sympy = None  # type: ignore[assignment]
    parse_expr = None  # type: ignore[assignment]
    standard_transformations = ()
    convert_xor = None
    _SYMPY = False

_MAX_QUERY_CHARS = 500
_MAX_POW = 1000


@dataclass
class Computation:
    """One computed answer, with its derivation and its rung."""

    query: str
    formal: str = ""
    result: str = ""
    rung: str = "unanswered"  # wolfram | sympy | stdlib | unanswered
    steps: list[str] = field(default_factory=list)
    source: str = ""
    pods: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    ok: bool = False
    error: str | None = None


def _unanswered(query: str, formal: str = "",
                wolfram_error: str | None = None) -> Computation:
    error = ("no rung could compute that — the local rungs cover "
             "arithmetic, units, dates and symbolic math; curated "
             "facts and deeper computation need WolframAlpha, which "
             "needs an AppID (set NM_WOLFRAM_APPID and ask again)")
    if wolfram_error:
        error = f"WolframAlpha said: {wolfram_error}. {error}"
    return Computation(query=query, formal=formal or query, rung="unanswered",
                       ok=False, error=error)


# ── stdlib rung: exact arithmetic ──────────────────────────────────────

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                   ast.Mod, ast.Pow)
_ALLOWED_UNARY = (ast.USub, ast.UAdd)


def _eval_exact(node: ast.AST) -> Fraction:
    """Evaluate an arithmetic AST exactly; anything else raises."""
    if isinstance(node, ast.Expression):
        return _eval_exact(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("not arithmetic")
        if isinstance(node.value, int):
            return Fraction(node.value)
        if isinstance(node.value, float):
            return Fraction(Decimal(str(node.value)))
        raise ValueError("not arithmetic")
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _eval_exact(node.left)
        right = _eval_exact(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return Fraction(left // right)
        if isinstance(node.op, ast.Mod):
            return Fraction(left % right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_POW or right.denominator != 1:
                raise ValueError("exponent out of range")
            return left ** int(right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
        value = _eval_exact(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("not arithmetic")


def _format_fraction(value: Fraction, decimal_style: bool = False) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    # A decimal-typed question gets a decimal answer ("0.1 + 0.2" →
    # "0.3", never "3/10"); a fraction-typed one keeps its fraction
    # ("1/3 + 1/6" → "1/2"). A repeating decimal stays an exact
    # fraction either way ("1/3").
    if decimal_style:
        den = value.denominator
        while den % 2 == 0:
            den //= 2
        while den % 5 == 0:
            den //= 5
        if den == 1:
            return _format_decimal(
                Decimal(value.numerator) / Decimal(value.denominator))
    return f"{value.numerator}/{value.denominator}"


def _try_arithmetic(query: str) -> Computation | None:
    text = query.strip()
    pct = re.fullmatch(r"([\d.]+)\s*%\s+of\s+(.+)", text, re.IGNORECASE)
    if pct:
        try:
            base = _eval_exact(ast.parse(pct.group(2).strip(), mode="eval"))
        except Exception:  # noqa: BLE001 — not arithmetic, next rung
            return None
        rate = Fraction(Decimal(pct.group(1))) / 100
        value = base * rate
        decimal_style = "." in pct.group(1) or "." in pct.group(2)
        return Computation(
            query=query, formal=f"{pct.group(1)}% of {pct.group(2)}",
            result=_format_fraction(value, decimal_style),
            rung="stdlib",
            steps=[f"{pct.group(1)}% of "
                   f"{_format_fraction(base, decimal_style)}"
                   f" = {_format_fraction(value, decimal_style)}"],
            source="exact arithmetic (stdlib)", ok=True)
    normalised = text.replace("^", "**").replace("×", "*").replace("÷", "/")
    if not re.fullmatch(r"[0-9\s.+\-*/%()]+", normalised):
        return None
    try:
        value = _eval_exact(ast.parse(normalised, mode="eval"))
    except Exception:  # noqa: BLE001 — not arithmetic, next rung
        return None
    decimal_style = "." in normalised
    return Computation(
        query=query, formal=normalised,
        result=_format_fraction(value, decimal_style),
        rung="stdlib",
        steps=[f"evaluate {normalised} exactly"
               f" = {_format_fraction(value, decimal_style)}"],
        source="exact arithmetic (stdlib)", ok=True)


# ── stdlib rung: units ─────────────────────────────────────────────────

def _dec(value: str) -> Decimal:
    return Decimal(value)


_UNIT_TABLE: dict[str, tuple[str, Decimal]] = {}
for _unit, _factor in [
        ("mm", "0.001"), ("cm", "0.01"), ("m", "1"), ("km", "1000"),
        ("in", "0.0254"), ("inch", "0.0254"), ("inches", "0.0254"),
        ("ft", "0.3048"), ("foot", "0.3048"), ("feet", "0.3048"),
        ("yd", "0.9144"), ("yard", "0.9144"), ("yards", "0.9144"),
        ("mi", "1609.344"), ("mile", "1609.344"), ("miles", "1609.344")]:
    _UNIT_TABLE[_unit] = ("length", _dec(_factor))
for _unit, _factor in [
        ("mg", "0.001"), ("g", "1"), ("kg", "1000"),
        ("oz", "28.349523125"), ("lb", "453.59237"),
        ("ounce", "28.349523125"), ("ounces", "28.349523125"),
        ("pound", "453.59237"), ("pounds", "453.59237")]:
    _UNIT_TABLE[_unit] = ("mass", _dec(_factor))
for _unit, _factor in [
        ("ml", "0.001"), ("l", "1"), ("litre", "1"), ("litres", "1"),
        ("liter", "1"), ("liters", "1"),
        ("tsp", "0.00492892159375"), ("tbsp", "0.01478676478125"),
        ("cup", "0.2365882365"), ("cups", "0.2365882365"),
        ("pint", "0.473176473"), ("quart", "0.946352946"),
        ("gallon", "3.785411784"), ("gallons", "3.785411784")]:
    _UNIT_TABLE[_unit] = ("volume", _dec(_factor))
for _unit, _factor in [
        ("s", "1"), ("sec", "1"), ("secs", "1"), ("second", "1"),
        ("seconds", "1"), ("min", "60"), ("minute", "60"),
        ("minutes", "60"), ("hr", "3600"), ("hour", "3600"),
        ("hours", "3600"), ("day", "86400"), ("days", "86400")]:
    _UNIT_TABLE[_unit] = ("time", _dec(_factor))
for _unit, _factor in [
        ("b", "1"), ("byte", "1"), ("bytes", "1"),
        ("kb", "1000"), ("mb", "1000000"), ("gb", "1000000000"),
        ("kib", "1024"), ("mib", "1048576"), ("gib", "1073741824")]:
    _UNIT_TABLE[_unit] = ("data", _dec(_factor))

_TEMPERATURE = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}


def _format_decimal(value: Decimal) -> str:
    text = format(value.quantize(Decimal("0.0000000001")), "f")
    return text.rstrip("0").rstrip(".") or "0"


def _try_units(query: str) -> Computation | None:
    m = re.fullmatch(r"([\d.]+)\s*([a-zA-Z]+)\s+to\s+([a-zA-Z]+)",
                     query.strip(), re.IGNORECASE)
    if not m:
        return None
    try:
        amount = Decimal(m.group(1))
    except InvalidOperation:
        return None
    src, dst = m.group(2).lower(), m.group(3).lower()
    if src in _TEMPERATURE and dst in _TEMPERATURE:
        celsius = {"c": amount, "celsius": amount,
                   "f": (amount - 32) * Decimal(5) / Decimal(9),
                   "fahrenheit": (amount - 32) * Decimal(5) / Decimal(9),
                   "k": amount - Decimal("273.15"),
                   "kelvin": amount - Decimal("273.15")}[src]
        out = {"c": celsius, "celsius": celsius,
               "f": celsius * Decimal(9) / Decimal(5) + 32,
               "fahrenheit": celsius * Decimal(9) / Decimal(5) + 32,
               "k": celsius + Decimal("273.15"),
               "kelvin": celsius + Decimal("273.15")}[dst]
        return Computation(
            query=query, formal=f"{m.group(1)} {src} to {dst}",
            result=_format_decimal(out), rung="stdlib",
            steps=[f"convert {m.group(1)} {src} → {dst}"
                   f" = {_format_decimal(out)}"],
            source="exact unit conversion (stdlib)", ok=True)
    if src not in _UNIT_TABLE or dst not in _UNIT_TABLE:
        return None
    kind_a, factor_a = _UNIT_TABLE[src]
    kind_b, factor_b = _UNIT_TABLE[dst]
    if kind_a != kind_b:
        return None
    out = amount * factor_a / factor_b
    return Computation(
        query=query, formal=f"{m.group(1)} {src} to {dst}",
        result=_format_decimal(out), rung="stdlib",
        steps=[f"convert {m.group(1)} {src} → {dst}"
               f" = {_format_decimal(out)}"],
        source="exact unit conversion (stdlib)", ok=True)


# ── stdlib rung: dates ─────────────────────────────────────────────────

_DATE_RE = r"(\d{4}-\d{2}-\d{2})"


def _parse_date(text: str) -> _dt.date | None:
    try:
        return _dt.date.fromisoformat(text)
    except ValueError:
        return None


def _try_dates(query: str) -> Computation | None:
    text = query.strip().lower()
    m = re.fullmatch(r"days between " + _DATE_RE + r" and " + _DATE_RE, text)
    if m:
        first, second = _parse_date(m.group(1)), _parse_date(m.group(2))
        if first is None or second is None:
            return None
        days = abs((second - first).days)
        return Computation(
            query=query, formal=f"days between {m.group(1)} and {m.group(2)}",
            result=str(days), rung="stdlib",
            steps=[f"|{m.group(2)} − {m.group(1)}| = {days} days"],
            source="date arithmetic (stdlib)", ok=True)
    m = re.fullmatch(r"what day(?: of the week)? is " + _DATE_RE, text)
    if m:
        day = _parse_date(m.group(1))
        if day is None:
            return None
        name = day.strftime("%A")
        return Computation(
            query=query, formal=f"weekday of {m.group(1)}", result=name,
            rung="stdlib", steps=[f"{m.group(1)} is a {name}"],
            source="date arithmetic (stdlib)", ok=True)
    m = re.fullmatch(r"add (\d+) days? to " + _DATE_RE, text)
    if m:
        day = _parse_date(m.group(2))
        if day is None:
            return None
        out = day + _dt.timedelta(days=int(m.group(1)))
        return Computation(
            query=query, formal=f"{m.group(2)} + {m.group(1)} days",
            result=out.isoformat(), rung="stdlib",
            steps=[f"{m.group(2)} + {m.group(1)} days = {out.isoformat()}"],
            source="date arithmetic (stdlib)", ok=True)
    m = re.fullmatch(r"days until " + _DATE_RE, text)
    if m:
        day = _parse_date(m.group(1))
        if day is None:
            return None
        delta = (day - _dt.date.today()).days
        return Computation(
            query=query, formal=f"days until {m.group(1)}",
            result=str(delta), rung="stdlib",
            steps=[f"{m.group(1)} − today = {delta} days"],
            source="date arithmetic (stdlib)", ok=True)
    return None


# ── sympy rung: symbolic math ──────────────────────────────────────────

def _sympy_parse(expr: str):
    return parse_expr(expr, transformations=standard_transformations
                      + (convert_xor,))


def _try_sympy(query: str) -> Computation | None:
    if not _SYMPY:
        return None
    text = query.strip()
    lowered = text.lower()
    var = "x"
    m = re.search(r"with respect to ([a-z])", lowered)
    if m:
        var = m.group(1)
    try:
        symbol = _sympy.Symbol(var)
        if lowered.startswith(("integrate ", "integral of ")):
            body = re.sub(r"^(integrate|integral of)\s+", "", text,
                          flags=re.IGNORECASE)
            body = re.sub(r"\s+with respect to [a-z]\s*$", "", body,
                          flags=re.IGNORECASE)
            body = re.sub(r"\s+d[a-z]\s*$", "", body)
            definite = re.search(r"(.+?)\s+from\s+(.+?)\s+to\s+(.+)$", body,
                                 re.IGNORECASE)
            if definite:
                expr = _sympy_parse(definite.group(1))
                result = _sympy.integrate(
                    expr, (symbol, _sympy_parse(definite.group(2)),
                           _sympy_parse(definite.group(3))))
                steps = [f"∫ {definite.group(1)} d{var}"
                         f" from {definite.group(2)}"
                         f" to {definite.group(3)}"]
            else:
                expr = _sympy_parse(body)
                result = _sympy.integrate(expr, symbol)
                steps = [f"∫ {body} d{var}"]
            steps.append(f"= {result}")
            return Computation(query=query, formal=f"integrate {body}",
                               result=str(result), rung="sympy", steps=steps,
                               source="sympy CAS", ok=True)
        if lowered.startswith(("differentiate ", "derivative of ", "diff ")):
            body = re.sub(r"^(differentiate|derivative of|diff)\s+", "",
                          text, flags=re.IGNORECASE)
            body = re.sub(r"\s+with respect to [a-z]\s*$", "", body,
                          flags=re.IGNORECASE)
            expr = _sympy_parse(body)
            result = _sympy.diff(expr, symbol)
            steps = [f"d/d{var} ({body})", f"= {result}"]
            return Computation(query=query, formal=f"differentiate {body}",
                               result=str(result), rung="sympy", steps=steps,
                               source="sympy CAS", ok=True)
        if lowered.startswith("solve "):
            body = text[len("solve "):].strip()
            if "=" in body:
                left, right = body.split("=", 1)
                eq = _sympy.Eq(_sympy_parse(left), _sympy_parse(right))
            else:
                eq = _sympy.Eq(_sympy_parse(body), 0)
            sols = _sympy.solve(eq, symbol)
            result = ", ".join(str(s) for s in sols)
            steps = [f"solve {body}", f"= {result}"]
            return Computation(query=query, formal=f"solve {body}",
                               result=result, rung="sympy", steps=steps,
                               source="sympy CAS", ok=True)
        if lowered.startswith("factor "):
            body = text[len("factor "):].strip()
            result = _sympy.factor(_sympy_parse(body))
            steps = [f"factor {body}", f"= {result}"]
            return Computation(query=query, formal=f"factor {body}",
                               result=str(result), rung="sympy", steps=steps,
                               source="sympy CAS", ok=True)
        if lowered.startswith("simplify "):
            body = text[len("simplify "):].strip()
            result = _sympy.simplify(_sympy_parse(body))
            steps = [f"simplify {body}", f"= {result}"]
            return Computation(query=query, formal=f"simplify {body}",
                               result=str(result), rung="sympy", steps=steps,
                               source="sympy CAS", ok=True)
        if lowered.startswith("limit "):
            m = re.search(r"limit\s+(.+?)\s+as\s+([a-z])\s*->\s*(.+)$",
                          text, re.IGNORECASE)
            if m:
                result = _sympy.limit(_sympy_parse(m.group(1)),
                                      _sympy.Symbol(m.group(2)),
                                      _sympy_parse(m.group(3)))
                steps = [f"lim {m.group(2)}→{m.group(3)} ({m.group(1)})",
                         f"= {result}"]
                return Computation(query=query, formal=text,
                                   result=str(result), rung="sympy",
                                   steps=steps, source="sympy CAS", ok=True)
        if re.search(r"[a-z]", text) and re.search(r"[+\-*/^()]", text):
            expr = _sympy_parse(text)
            if expr.free_symbols:
                result = _sympy.simplify(expr)
                steps = [f"simplify {text}", f"= {result}"]
                return Computation(query=query, formal=text,
                                   result=str(result), rung="sympy",
                                   steps=steps, source="sympy CAS", ok=True)
    except Exception:  # noqa: BLE001 — not symbolic math, next rung
        return None
    return None


# ── orchestrator ───────────────────────────────────────────────────────

def compute_answer(query: str, *, appid: str | None = None,
                   fetcher: Any = None,
                   rung: str = "auto") -> Computation:
    """Answer a computable question through the ladder.

    ``rung`` is ``auto`` (Wolfram when keyed, then stdlib, then
    sympy) or one rung by name. Never raises and never invents:
    a question no rung can answer comes back ``ok=False``.
    """
    text = (query or "").strip()
    if not text:
        return _unanswered(query, formal="")
    if len(text) > _MAX_QUERY_CHARS:
        return Computation(query=query, formal=text, rung="unanswered",
                           ok=False,
                           error="that question is too long to compute")
    wanted = (rung or "auto").lower()
    if wanted not in ("auto", "wolfram", "sympy", "stdlib"):
        wanted = "auto"
    wolfram_error: str | None = None
    if wanted in ("auto", "wolfram") and appid:
        client = WolframClient(appid=appid, fetcher=fetcher)
        res = client.query(text)
        if res.get("ok"):
            pods = res.get("pods") or []
            result = ""
            for pod in pods:
                if (pod.get("id") or "").lower() == "result":
                    subs = pod.get("subpods") or []
                    if subs:
                        result = (subs[0].get("plaintext") or "").strip()
                        break
            if not result:
                for pod in pods:
                    subs = pod.get("subpods") or []
                    if subs:
                        result = (subs[0].get("plaintext") or "").strip()
                        break
            steps = []
            for pod in pods:
                for sub in pod.get("subpods") or []:
                    plain = (sub.get("plaintext") or "").strip()
                    if plain:
                        steps.append(f"{pod.get('title') or 'Result'}:"
                                     f" {plain}")
            return Computation(
                query=query, formal=text, result=result, rung="wolfram",
                steps=steps, source="WolframAlpha", pods=pods,
                assumptions=list(res.get("assumptions") or []), ok=True)
        wolfram_error = res.get("error")
        if wanted == "wolfram":
            return Computation(query=query, formal=text, rung="unanswered",
                               ok=False, error=wolfram_error)
    if wanted in ("auto", "stdlib"):
        for attempt in (_try_arithmetic, _try_units, _try_dates):
            comp = attempt(text)
            if comp is not None:
                return comp
        if wanted == "stdlib":
            return _unanswered(query, formal=text,
                               wolfram_error=wolfram_error)
    if wanted in ("auto", "sympy"):
        comp = _try_sympy(text)
        if comp is not None:
            return comp
        if wanted == "sympy":
            return _unanswered(query, formal=text,
                               wolfram_error=wolfram_error)
    return _unanswered(query, formal=text, wolfram_error=wolfram_error)


def format_compute_reply(comp: Computation) -> str:
    """The chat rendering: result first, rung named, steps shown."""
    if not comp.ok:
        return f"Can't compute that: {comp.error or 'unknown reason'}"
    lines = [f"🧮 {comp.result}",
             f"— computed via {comp.rung} ({comp.source})"]
    if comp.steps:
        lines.append("Steps:")
        lines.extend(f"{i + 1}. {step}"
                     for i, step in enumerate(comp.steps))
    if comp.rung == "wolfram" and comp.pods:
        for pod in comp.pods:
            title = pod.get("title") or "Result"
            for sub in pod.get("subpods") or []:
                plain = (sub.get("plaintext") or "").strip()
                if plain and f"{title}: {plain}" not in comp.steps:
                    lines.append(f"{title}: {plain}")
    if comp.assumptions:
        lines.append("Assuming: " + "; ".join(comp.assumptions))
    return "\n".join(lines)
