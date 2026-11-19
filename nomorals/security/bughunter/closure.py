"""Coverage-closure ledger: did this engagement actually cover
what it discovered?

Port of ``engine/closure.py`` from elementalsouls/Claude-BugHunter
(MIT, Copyright (c) 2026 Sachin Sharma), verbatim in semantics.

The engine reports findings. A report that lists only findings
silently implies everything else was checked and clean, which is
the most dangerous default a pentest deliverable can have.
``tested: 47`` cannot answer "which classes were never tried on
host X" — this can, and the report renders its caveats from it
rather than from hand-written prose.

    CLOSE CLEAN when  every discovered (URL, parameter, vuln_class)
                      item has a verdict
    otherwise         MARKED, and the gaps ship as caveats

``coverage_complete`` is a HARD gate. An engagement with one
geo-blocked host never closes clean; it closes MARKED and says
so. That is deliberate — a caveat that ships is worth more than
a clean-looking report that quietly omits what nobody could reach.

Pure functions over the state dict, testable without running
agents. Devon adopts the ledger's "blocked is a gap, not
coverage" rule as the reporting standard for all security
audits, not just engagements (see ``triage.py`` and the
engagement report renderer in ``report.py``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .state import Engagement

__all__ = [
    "COVERED",
    "VERDICTS",
    "coverage_complete",
    "coverage_gaps",
    "evaluate",
    "expected_coverage",
    "render_caveats",
]

#: A work item is only "covered" by one of these verdicts.
#: BLOCKED is deliberately absent — it is a gap, and gaps become caveats.
COVERED = {"clean", "finding", "not_applicable", "out_of_scope"}
VERDICTS = COVERED | {"blocked"}


def _host(url: str) -> str:
    """Bare host (port included) from a URL, tolerant of scheme-less input.

    Recorders and the expectation set must agree on this, or a
    covered pair silently reads as never-tested.
    """
    s = url.split("://", 1)[-1]
    return s.split("/", 1)[0].split("?", 1)[0]


def expected_coverage(state: dict[str, Any]) -> set[tuple[str, str]]:
    """Every (host, vuln_class) the run discovered, and therefore owes a verdict."""
    return {(_host(s["url"]), s.get("vuln_class", ""))
            for s in state.get("surface", [])
            if s.get("vuln_class")}


def coverage_gaps(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Pairs with no verdict, or an explicitly blocked one. Sorted for stable reports."""
    legacy = state.get("coverage", {})
    by_item = state.get("coverage_by_item", {})
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in state.get("surface", []):
        if item.get("vuln_class"):
            pairs[(_host(item["url"]), item["vuln_class"])].append(item)

    gaps = []
    for (host, cls), items in sorted(pairs.items()):
        for item in items:
            rec = by_item.get(Engagement._key(item))
            if rec is None and len(items) == 1:
                rec = legacy.get(f"{host}|{cls}")
            rec = rec or {}
            verdict = rec.get("verdict")
            if verdict not in COVERED:
                gap = {"host": host, "vuln_class": cls,
                       "url": item["url"],
                       "param": item.get("param", ""), "state": "never_tested",
                       "note": ""}
                if verdict == "blocked":
                    gap.update(state="blocked", note=rec.get("note", ""))
                gaps.append(gap)
    return gaps


def coverage_complete(state: dict[str, Any]) -> bool:
    return not coverage_gaps(state)


def evaluate(state: dict[str, Any]) -> dict[str, Any]:
    """{clean: bool, reason: str, gaps: [...]} — the closure verdict for the report."""
    gaps = coverage_gaps(state)
    if not gaps:
        return {"clean": True, "gaps": [],
                "reason": "every discovered work item reached a verdict"}
    blocked = sum(1 for g in gaps if g["state"] == "blocked")
    never = len(gaps) - blocked
    bits = ([f"{blocked} blocked"] if blocked else []) + \
           ([f"{never} never tested"] if never else [])
    return {"clean": False, "gaps": gaps,
            "reason": f"{len(gaps)} coverage gap(s): " + ", ".join(bits)}


def render_caveats(state: dict[str, Any],
                   decision: dict[str, Any] | None = None) -> str:
    """The 'not tested' section, generated from the ledger rather than written by hand."""
    d = decision or evaluate(state)
    if d["clean"]:
        return ("## Coverage\n\nEvery discovered work item reached a "
                "verdict. No coverage caveats apply.\n")

    gaps = d["gaps"]
    blocked = [g for g in gaps if g["state"] == "blocked"]
    never = [g for g in gaps if g["state"] == "never_tested"]
    out = ["## Not tested — do not read as passing\n",
           "**This engagement did not reach full coverage.** The surface "
           "below was discovered and never cleared.\n"]

    def describe(gap: dict[str, Any]) -> str:
        param = f" (parameter `{gap['param']}`)" if gap["param"] else ""
        return (f"- `{gap['host']}` · {gap['vuln_class']} · "
                f"`{gap['url']}`{param}")

    if blocked:
        out.append("\n**Blocked — reachable surface that could not be "
                   "assessed:**\n")
        for g in blocked:
            out.append(describe(g) + (f" — {g['note']}" if g["note"] else ""))
    if never:
        out.append("\n**Never tested — discovered but never reached:**\n")
        for g in never:
            out.append(describe(g))
    out.append("\nAbsence of a finding above is not evidence of absence "
               "of a vulnerability.\n")
    return "\n".join(out)


def closure_selftest() -> bool:
    """Upstream's assertions, verbatim."""
    def st(surface: Any = (), coverage: Any = None) -> dict[str, Any]:
        return {"surface": [{"url": u, "vuln_class": c}
                            for u, c in surface],
                "coverage": coverage or {}}

    # A single untested pair blocks clean closure — the gate is hard.
    s = st([("https://a.test/x", "idor"), ("https://b.test/y", "ssrf")],
           {"a.test|idor": {"verdict": "clean"}})
    d = evaluate(s)
    assert not d["clean"] and len(d["gaps"]) == 1 \
        and d["gaps"][0]["host"] == "b.test"
    assert "never tested" in d["reason"]

    # Blocked never counts as covered, and its note reaches the report.
    s2 = st([("https://a.test/x", "idor")],
            {"a.test|idor": {"verdict": "blocked",
                             "note": "geo-fenced at edge"}})
    assert not coverage_complete(s2)
    out = render_caveats(s2)
    assert "do not read as passing" in out and "geo-fenced at edge" in out

    # out_of_scope and not_applicable ARE verdicts — they close cleanly.
    s3 = st([("https://a.test/x", "idor"), ("https://a.test/y", "xss")],
            {"a.test|idor": {"verdict": "out_of_scope"},
             "a.test|xss": {"verdict": "not_applicable"}})
    assert evaluate(s3)["clean"], \
        "a recorded verdict must close, even a negative one"
    assert "No coverage caveats apply" in render_caveats(s3)

    # Host normalisation must match between recorder and expectation.
    assert _host("https://h.test:8443/a?b=1") == "h.test:8443"
    s4 = st([("https://h.test:8443/a", "idor")],
            {"h.test:8443|idor": {"verdict": "clean"}})
    assert evaluate(s4)["clean"], \
        "recorder and expectation must agree on host form"

    # Surface with no vuln_class owes nothing.
    assert evaluate({"surface": [{"url": "https://a.test/"}],
                     "coverage": {}})["clean"]
    return True
