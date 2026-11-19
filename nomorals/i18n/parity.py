"""Machine-checked parity for the craft-i18n port.

``nomorals/i18n/crafti18n-parity.toml`` lists real fixtures +
operations + expected outcomes; :func:`run_case` runs one case
against a translator (the vendored tree by default) and raises
on any mismatch; :func:`run_i18n_parity` runs the whole TOML
and reports pass/fail per case.  Missing fixture = fail, never
skip.

Layering note: this organ is L3 and the shared parity runner
(``craftlibs.parity``) and the tool registry are L4, so neither
is imported here — the case loop lives in this module with the
same result shape as the shared runner, and the ``tool_call``
operation takes its tool surface as an injected ``tool_caller``
(the caller builds it from the registry; tests and the proof
script do).  A caller that runs a TOML containing a
``tool_call`` case without one gets a failure naming that,
never a silent skip.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .catalog import CatalogSet
from .check import check_upstream_text
from .runtime import I18nError, Translator

__all__ = ["run_case", "run_i18n_parity"]

_DEFAULT_TOML = (Path(__file__).resolve().parent
                 / "crafti18n-parity.toml")


def _translator() -> Translator:
    from .upstream import UpstreamCatalogSet

    return Translator(CatalogSet(),
                      upstream=UpstreamCatalogSet.vendored())


def run_case(case: dict[str, Any],
             translator: Translator | None = None,
             tool_caller: Callable[[dict[str, Any]], Any] | None
             = None) -> bool:
    """Run one parity case; ``True`` when it holds, else raise
    :class:`AssertionError` naming the mismatch.  ``translator``
    defaults to the vendored tree.  A ``tool_call`` case needs
    ``tool_caller``: a callable taking the case's ``args`` dict
    and returning the tool outcome's value."""
    tr = translator if translator is not None else _translator()
    op = str(case.get("op", ""))
    locale = str(case.get("locale", "en"))
    if op == "lookup":
        got = tr.get(str(case["source"]),
                     context=str(case.get("context", "")),
                     app=case.get("app"),
                     locale=locale)
        if got != case["expect"]:
            raise AssertionError(
                f"lookup {case['source']!r} in {locale}: "
                f"got {got!r}, want {case['expect']!r}")
        return True
    if op == "fallback":
        got = tr.get(str(case["source"]), locale=locale)
        if got != case["expect"]:
            raise AssertionError(
                f"fallback {case['source']!r} in {locale}: "
                f"got {got!r}, want {case['expect']!r}")
        return True
    if op in ("plural", "plural_fallback"):
        got = tr.trn(str(case["source"]), locale,
                     int(case["count"]))
        if got != case["expect"]:
            raise AssertionError(
                f"plural {case['source']!r} in {locale} "
                f"count {case['count']}: got {got!r}, "
                f"want {case['expect']!r}")
        return True
    if op == "placeholder_mismatch":
        try:
            tr.translate(str(case["source"]), locale)
        except I18nError as exc:
            if exc.details.get("kind") != "missing_parameter":
                raise AssertionError(
                    "placeholder mismatch raised the wrong kind: "
                    f"{exc.details.get('kind')}") from exc
            return True
        raise AssertionError(
            f"translating {case['source']!r} without its "
            "parameters should have raised")
    if op == "checker_finding":
        found = check_upstream_text(str(case["text"]),
                                    str(case["locale"]))
        kinds = {issue["kind"] for issue in found}
        if str(case["kind"]) not in kinds:
            raise AssertionError(
                f"checker on {case['text']!r}: kinds {kinds}, "
                f"want {case['kind']!r}")
        return True
    if op == "tool_call":
        if tool_caller is None:
            raise AssertionError(
                "tool_call case needs a tool_caller (the tool "
                "registry lives a layer up; pass one in)")
        value = tool_caller(dict(case.get("args", {})))
        reply = str(value.get("reply", value)) \
            if isinstance(value, dict) else str(value)
        if str(case["expect_contains"]) not in reply \
                and str(case["expect_contains"]) not in str(value):
            raise AssertionError(
                f"tool call {case.get('args')}: reply {reply!r} "
                f"lacks {case['expect_contains']!r}")
        return True
    raise AssertionError(f"unknown parity op {op!r}")


def _missing_toml(toml_path: Path) -> dict[str, Any]:
    return {"passed": 0, "failed": ["<toml>"], "passed_ids": [],
            "failures": [{"id": "<toml>",
                          "reason": f"missing {toml_path}"}],
            "total": 0, "skipped": []}


def run_i18n_parity(source: str | Path | None = None,
                    run_case_fn: Any | None = None,
                    tool_caller: Callable[[dict[str, Any]], Any]
                    | None = None) -> dict[str, Any]:
    """Run the i18n parity TOML (default: the shipped one).
    One case's failure never stops the rest.  A missing TOML
    fails, never skips.  ``run_case_fn`` runs one case dict;
    the default is :func:`run_case` with ``tool_caller``."""
    toml_path = Path(source) if source is not None \
        else _DEFAULT_TOML
    if not toml_path.exists():
        return _missing_toml(toml_path)
    with toml_path.open("rb") as handle:
        cases = list(tomllib.load(handle).get("cases", []))
    runner = run_case_fn or (lambda case: run_case(
        case, tool_caller=tool_caller))
    passed_ids: list[str] = []
    failures: list[dict[str, str]] = []
    for case in cases:
        case_id = str(case.get("id", "<unnamed>"))
        try:
            runner(case)
        except Exception as exc:  # noqa: BLE001 — one case's
            # failure never stops the rest.
            failures.append({"id": case_id,
                             "reason": f"{type(exc).__name__}: {exc}"})
        else:
            passed_ids.append(case_id)
    return {"passed": len(passed_ids),
            "failed": [entry["id"] for entry in failures],
            "passed_ids": passed_ids,
            "failures": failures,
            "total": len(cases),
            "skipped": []}
