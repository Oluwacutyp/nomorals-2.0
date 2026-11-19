"""Parity runner — the machine-checked parity TOML for bughunter.

Each ``[[case]]`` names an operation on the real ported organ and
the value the upstream semantics demand. Parity is measured, never
asserted: a case whose expectation is wrong fails loudly, a missing
fixture fails (never a silent skip), and an operation that raises
unexpectedly fails with the exception named.

Schema (version 1):

    [schema]
    version = 1

    [[case]]
    id = "scope-allow"
    op = "scope_allow"
    scope = {name = "x", in_scope = [...], ...}
    urls = [...]
    [case.expect]
    allowed = [...]

Ops and their keys (all keys optional unless noted):

* ``scope_allow`` — ``scope`` (dict), ``urls``. Expect: ``allowed``
  (the in-scope subset, in input order).
* ``scope_reject`` — ``scope``, ``url``. Expect: ``reason`` (exact
  reject_reason string).
* ``classify_url`` — ``url``, ``audit_boost`` (list of classes).
  Expect: ``contains`` (skill names that must be among the matches).
* ``rank_priority`` — ``item`` (dict). Expect: ``priority`` (int).
* ``gate`` — ``finding`` (dict with url/param/vuln_class/severity/
  evidence/request). Expect: ``verdict`` (PASS/DOWNGRADE/KILL).
* ``closure`` — ``state`` (dict with surface/tested/
  coverage_by_item). Expect: ``clean`` (bool), ``gaps`` (int).
* ``render_report`` — ``platform``, ``finding`` (meta dict).
  Expect: ``contains`` (substrings that must appear).
* ``hackerone_escape`` — ``text``. Expect: ``escaped``.
* ``audit_worklist`` — ``findings`` (list). Expect: ``units`` (int),
  ``first_class`` (str).
* ``engine_mock`` — ``scope`` (dict). Runs the full
  recon,rank,map,hunt,validate,report loop against the mock
  fixtures in a throwaway home. Expect: ``surface``, ``tested``,
  ``candidates``, ``confirmed`` (ints).
* ``skill_map_selftest`` — no keys. Expect: ``errors`` (int).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from . import closure, skillmap
from .classify import classify_url
from .engine import run_engagement, worklist_from_audit
from .hackerone import _escape
from .report import render_report
from .scope import Scope
from .triage import gate_finding

__all__ = ["DEFAULT_PARITY_PATH", "load_parity", "run_parity"]

DEFAULT_PARITY_PATH = Path(__file__).resolve().parent / "bughunter-parity.toml"


def load_parity(path: str | Path | None = None) -> dict[str, Any]:
    """Load a parity TOML file (stdlib tomllib)."""
    path = Path(path) if path is not None else DEFAULT_PARITY_PATH
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data.get("case"), list) or not data["case"]:
        raise ValueError(f"parity TOML {path} has no [[case]] rows")
    return data


def _run_case(case: dict[str, Any],
              tmp: Path) -> dict[str, Any]:
    """Run one case; returns {passed, measured, error}."""
    op = case.get("op", "")
    expect = case.get("expect", {}) or {}
    try:
        if op == "scope_allow":
            scope = Scope.from_dict(case["scope"])
            measured = {"allowed": [u for u in case["urls"]
                                    if scope.in_scope_host(u)]}
        elif op == "scope_reject":
            scope = Scope.from_dict(case["scope"])
            measured = {"reason": scope.reject_reason(case["url"])}
        elif op == "classify_url":
            result = classify_url(case["url"],
                                  audit_boost=case.get("audit_boost"))
            measured = {"skills": sorted(result.get("matches", {}).keys())}
        elif op == "rank_priority":
            measured = {"priority": skillmap.rank_priority(case["item"])}
        elif op == "gate":
            measured = {"verdict": gate_finding(case["finding"])["verdict"]}
        elif op == "closure":
            decision = closure.evaluate(case["state"])
            measured = {"clean": decision["clean"],
                        "gaps": len(decision["gaps"])}
        elif op == "render_report":
            text = render_report(case["finding"], case["platform"])
            measured = {"contains": [s for s in expect.get("contains", [])
                                     if s in text]}
            expect = {**expect, "contains": expect.get("contains", [])}
            passed = measured["contains"] == expect["contains"]
            return {"passed": passed, "measured": measured, "error": ""}
        elif op == "hackerone_escape":
            measured = {"escaped": _escape(case["text"])}
        elif op == "audit_worklist":
            units = worklist_from_audit(case["findings"])
            measured = {"units": len(units),
                        "first_class": units[0]["vuln_class"]
                                       if units else ""}
        elif op == "engine_mock":
            home = tmp / "home"
            out = run_engagement(case["scope"], hunt=True, mock=True,
                                 home=home)
            summary = out["summary"]
            measured = {"surface": summary["surface"],
                        "tested": summary["tested"],
                        "candidates": summary["candidates"],
                        "confirmed": summary["confirmed"]}
        elif op == "skill_map_selftest":
            result = skillmap.skill_map_selftest()
            measured = {"errors": len(result.get("errors", []))}
        else:
            return {"passed": False, "measured": {},
                    "error": f"unknown op {op!r}"}
    except Exception as exc:  # noqa: BLE001 - a raising case fails loudly
        return {"passed": False, "measured": {},
                "error": f"{type(exc).__name__}: {exc}"}
    passed = True
    for key, wanted in expect.items():
        if key == "contains":
            got = measured.get("skills", [])
            if not all(s in got for s in wanted):
                passed = False
            continue
        if measured.get(key) != wanted:
            passed = False
    return {"passed": passed, "measured": measured, "error": ""}


def run_parity(path: str | Path | None = None, *,
               tmp_dir: str | Path | None = None) -> dict[str, Any]:
    """Run every [[case]]; returns {passed, failed, cases}."""
    import tempfile

    data = load_parity(path)
    results: list[dict[str, Any]] = []
    if tmp_dir is not None:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(tmp_dir) if tmp_dir else None
    ) as tmp:
        tmp_path = Path(tmp)
        for case in data["case"]:
            outcome = _run_case(case, tmp_path)
            results.append(
                {
                    "id": case.get("id", "<unnamed>"),
                    "passed": outcome["passed"],
                    "measured": outcome["measured"],
                    "error": outcome["error"],
                }
            )
    passed = sum(1 for r in results if r["passed"])
    return {"passed": passed, "failed": len(results) - passed,
            "cases": results}
