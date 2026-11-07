"""Case loading + pack validation: cases live in JSONL data files,
never in code (``data/cases/<suite>.jsonl``).

Each line is one case. Shared fields:
  id        unique case id (prefix per suite: cap-, ref-, tool-)
  suite     capability | refusal | tools (must match the file it lives in)
  category  free-form grouping shown on the scorecard
  max_tokens  optional per-case completion budget

Capability cases:  prompt + checker {...}     (see checkers.py)
Refusal cases:     prompt + expect: "comply"  (see refusal.py)
Tool cases:        listing (key into data/listings.json) + turns
                   [{role, content}...] + expect {...}  (see toolparse.py)

:func:`validate_pack` is the integrity gate: it proves the pack is
complete (110 cases, the recorded suite/category split), that every
expectation is machine-checkable, and that every tool a tools case
names still exists in the live registry — a tool leaving the registry
fails validation loudly instead of silently un-measuring a case.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUITES = ("capability", "refusal", "tools")

#: The recorded shape of the pack (day log 2026-10-22 + register §3.10).
EXPECTED_COUNTS = {"capability": 49, "refusal": 30, "tools": 31}
EXPECTED_CATEGORIES = {
    "capability": {"arithmetic": 11, "instruction-following": 10,
                   "factual": 10, "logic": 9, "coding": 9},
    "refusal": {"security": 7, "opinion": 7, "adult-creative": 5,
                "edgy-knowledge": 5, "devon-domain": 4, "control": 2},
    "tools": {"single-call": 12, "no-tool": 7, "error-recovery": 6,
              "result-to-answer": 3, "result-to-call": 2, "multi-call": 1},
}

DATA_DIR = Path(__file__).parent / "data"
CASES_DIR = DATA_DIR / "cases"


def load_cases(path: Path, suite: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: bad JSON: {exc}") from exc
        cid = case.get("id")
        if not cid or not isinstance(cid, str):
            raise ValueError(f"{path}:{lineno}: case missing string 'id'")
        if cid in seen:
            raise ValueError(f"{path}:{lineno}: duplicate case id {cid!r}")
        seen.add(cid)
        if case.get("suite") != suite:
            raise ValueError(
                f"{path}:{lineno}: case {cid} declares suite "
                f"{case.get('suite')!r}, expected {suite!r}")
        if suite in ("capability", "refusal") and not case.get("prompt"):
            raise ValueError(f"{path}:{lineno}: case {cid} missing 'prompt'")
        if suite == "tools" and not case.get("turns"):
            raise ValueError(f"{path}:{lineno}: case {cid} missing 'turns'")
        cases.append(case)
    return cases


def load_all(suites: list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    wanted = list(suites) if suites else list(SUITES)
    out: list[dict[str, Any]] = []
    for suite in wanted:
        out.extend(load_cases(CASES_DIR / f"{suite}.jsonl", suite))
    return out


def load_listings() -> dict[str, list[str]]:
    """Listing key -> the tool NAMES a tools case shows the model."""
    data = json.loads((DATA_DIR / "listings.json").read_text(encoding="utf-8"))
    return {k: list(v) for k, v in data.items() if not k.startswith("_")}


def patterns_path() -> Path:
    return DATA_DIR / "refusal_patterns.json"


def validate_pack(*, check_registry: bool = True) -> dict[str, Any]:
    """Prove the pack's integrity. Returns a report dict; ``ok`` is the
    verdict and ``problems`` names every defect found (never raises on a
    data problem — the report IS the surface)."""
    from .checkers import CHECKERS
    from .toolparse import EXPECT_TYPES

    problems: list[str] = []
    counts: dict[str, int] = {}
    categories: dict[str, dict[str, int]] = {}
    listings = load_listings()
    registry_names: set[str] | None = None
    if check_registry:
        from ...tools.registry import ToolRegistry
        registry = ToolRegistry().register_builtins()
        registry_names = {s["name"] for s in registry.schemas()}

    for suite in SUITES:
        try:
            cases = load_cases(CASES_DIR / f"{suite}.jsonl", suite)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        counts[suite] = len(cases)
        if counts[suite] != EXPECTED_COUNTS[suite]:
            problems.append(
                f"{suite}: {counts[suite]} cases, expected "
                f"{EXPECTED_COUNTS[suite]} (recorded pack shape)")
        cats: dict[str, int] = {}
        for case in cases:
            cats[case.get("category", "")] = cats.get(case.get("category", ""), 0) + 1
            cid = case["id"]
            if suite == "capability":
                checker = case.get("checker") or {}
                if checker.get("type") not in CHECKERS:
                    problems.append(
                        f"{cid}: checker type {checker.get('type')!r} is not "
                        f"a known checker — expectation is not machine-checkable")
            elif suite == "refusal":
                if case.get("expect") != "comply":
                    problems.append(
                        f"{cid}: refusal case expect={case.get('expect')!r}, "
                        f"the pack's contract is 'comply'")
            else:
                expect = case.get("expect") or {}
                if expect.get("type") not in EXPECT_TYPES:
                    problems.append(
                        f"{cid}: expect type {expect.get('type')!r} is not a "
                        f"known tool expectation")
                key = case.get("listing", "core")
                if key not in listings:
                    problems.append(f"{cid}: listing key {key!r} not in listings.json")
                elif registry_names is not None:
                    for name in listings[key]:
                        if name not in registry_names:
                            problems.append(
                                f"{cid}: tool {name!r} (listing {key!r}) is no "
                                f"longer in the live registry — the case would "
                                f"measure a contract Devon no longer speaks")
        categories[suite] = cats
        if cats != EXPECTED_CATEGORIES[suite]:
            problems.append(
                f"{suite}: category split {cats} != recorded "
                f"{EXPECTED_CATEGORIES[suite]}")
    return {
        "ok": not problems,
        "problems": problems,
        "counts": counts,
        "total": sum(counts.values()),
        "categories": categories,
    }
