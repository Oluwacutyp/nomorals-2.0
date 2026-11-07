"""Aggregation + rendering: the scorecard and run comparisons.

Scores are means of per-case scores (0..1), always reported per suite
and per category — the pack's standing rule is that no single overall
number may hide a weak axis (a model that is uncensored but stupid
fails capability; smart but censored fails freedom; either without
tool fluency can't operate her tools when her brain directs it to).
The overall figure is the case-mean
across all attempted cases and is shown last, labeled as such.

Only ATTEMPTED cases (a served response was scored) count toward a
score. Errored and not-run cases are counted and shown — a run that
mostly failed to reach its provider must never average into a
respectable-looking number.
"""
from __future__ import annotations

from typing import Any

SUITE_TITLES = {
    "capability": "Capability",
    "refusal": "Freedom",
    "tools": "Tool fluency",
}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case rows into suite/category score dicts."""
    suites: dict[str, dict[str, Any]] = {}
    overall_scores: list[float] = []
    for row in rows:
        suite = suites.setdefault(row["suite"], {
            "scores": [], "passed": 0, "attempted": 0, "total": 0,
            "errors": 0, "not_run": 0, "categories": {}})
        suite["total"] += 1
        if row.get("not_run"):
            suite["not_run"] += 1
            continue
        if row.get("error"):
            suite["errors"] += 1
            continue
        suite["attempted"] += 1
        suite["scores"].append(row["score"])
        if row["passed"]:
            suite["passed"] += 1
        overall_scores.append(row["score"])
        cat = suite["categories"].setdefault(
            row["category"], {"scores": [], "passed": 0, "attempted": 0})
        cat["attempted"] += 1
        cat["scores"].append(row["score"])
        if row["passed"]:
            cat["passed"] += 1

    def _finish(bucket: dict[str, Any]) -> dict[str, Any]:
        scores = bucket.pop("scores")
        return {
            **bucket,
            "score": round(sum(scores) / len(scores), 4) if scores else None,
        }

    out_suites: dict[str, Any] = {}
    for name, bucket in suites.items():
        cats = {c: _finish(b) for c, b in bucket.pop("categories").items()}
        finished = _finish(bucket)
        finished["categories"] = cats
        out_suites[name] = finished
    return {
        "suites": out_suites,
        "overall": (round(sum(overall_scores) / len(overall_scores), 4)
                    if overall_scores else None),
        "attempted": len(overall_scores),
    }


def _pct(score: float | None) -> str:
    return "—" if score is None else f"{score * 100:.1f}%"


def render_run(run: dict[str, Any]) -> str:
    """The scorecard for one run — the human read."""
    agg = run["aggregate"]
    lines = [
        f"Brain eval — run {run['run_id']}",
        f"pack v{run['pack_version']} · rung {run['rung']} · "
        f"provider {run.get('provider') or '?'} · model {run.get('model') or '?'}",
        f"status {run['status']} · {run['started']} → {run['finished']}",
    ]
    if run.get("note"):
        lines.append(f"note: {run['note']}")
    lines.append("")
    for suite in ("capability", "refusal", "tools"):
        bucket = agg["suites"].get(suite)
        if bucket is None:
            continue
        title = SUITE_TITLES.get(suite, suite)
        lines.append(
            f"{title:<14} {_pct(bucket['score']):>7}  "
            f"({bucket['passed']}/{bucket['attempted']} passed, "
            f"{bucket['errors']} errored, {bucket['not_run']} not run)")
        for cat, cb in sorted(bucket["categories"].items()):
            lines.append(
                f"  {cat:<22} {_pct(cb['score']):>7}  "
                f"({cb['passed']}/{cb['attempted']})")
    lines.append(
        f"{'Overall':<14} {_pct(agg['overall']):>7}  "
        f"(case-mean over {agg['attempted']} attempted cases)")
    imperfect = [r for r in run["cases"]
                 if not r.get("not_run") and not r["passed"]]
    if imperfect:
        lines.append("")
        lines.append(f"Sub-perfect cases ({len(imperfect)}):")
        for row in imperfect:
            lines.append(f"- {row['id']} [{row['suite']}/{row['category']}] "
                         f"score {row['score']:.2f} — {row['detail']}")
            excerpt = (row.get("response") or "").strip().replace("\n", " ")
            if excerpt:
                lines.append(f"    said: {excerpt[:200]}")
    return "\n".join(lines)


def render_compare(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Side-by-side comparison of two runs (a = older, b = newer)."""
    lines = [
        "Brain eval comparison",
        f"A {a['run_id']} — {a.get('provider') or '?'}/{a.get('model') or '?'} "
        f"(pack v{a['pack_version']}, {a['status']})",
        f"B {b['run_id']} — {b.get('provider') or '?'}/{b.get('model') or '?'} "
        f"(pack v{b['pack_version']}, {b['status']})",
        "",
        f"{'suite':<14} {'A':>7} {'B':>7} {'delta':>8}",
    ]
    for suite in ("capability", "refusal", "tools"):
        ba = a["aggregate"]["suites"].get(suite, {})
        bb = b["aggregate"]["suites"].get(suite, {})
        sa, sb = ba.get("score"), bb.get("score")
        delta = (f"{(sb - sa) * 100:+.1f}pp"
                 if sa is not None and sb is not None else "—")
        lines.append(f"{SUITE_TITLES.get(suite, suite):<14} "
                     f"{_pct(sa):>7} {_pct(sb):>7} {delta:>8}")
    oa, ob = a["aggregate"]["overall"], b["aggregate"]["overall"]
    delta = (f"{(ob - oa) * 100:+.1f}pp"
             if oa is not None and ob is not None else "—")
    lines.append(f"{'Overall':<14} {_pct(oa):>7} {_pct(ob):>7} {delta:>8}")
    if a["pack_version"] != b["pack_version"]:
        lines.append("")
        lines.append("⚠ pack versions differ — scores are not strictly comparable.")
    return "\n".join(lines)
