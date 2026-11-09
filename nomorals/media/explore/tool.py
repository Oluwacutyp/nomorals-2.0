"""The ``video_explore`` tool — the exploration substrate's surface.

WS4 Pass 1 ships the manual trigger + the read actions; the
``/explore`` board and the scheduler are P3 (census §6). Actions:

* ``run`` — one sweep over the resolved beats (NewsAgent interest
  profile + interests watches) through the production instruments.
* ``status`` — the substrate summary: sweeps, observations,
  candidates being watched, per-beat baselines.
* ``observations`` — recent observation rows with provenance.
* ``candidates`` — the watching pool with sighting counts.
* ``beats`` — the beats a sweep would cover, with their origins.

Rendered text carries no internal record ids — beats, timestamps
and counts only; ids ride the structured result for programmatic
consumers (P2/P3), per the house rule.
"""

from __future__ import annotations

import time
from typing import Any

from ...core.errors import ToolError
from ...core.policy import Capability
from .engine import explore_for_context, explore_status, run_exploration

__all__ = ["register"]


def _when(ts: float) -> str:
    if not ts:
        return "never"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _render_status(data: dict[str, Any]) -> str:
    summary = data["summary"]
    lines = [
        "🔭 **Video exploration**",
        f"sweeps run: {summary['sweeps']} · observations recorded: "
        f"{summary['observations']} · candidates watched: "
        f"{summary['candidates_watching']} · claims: {summary['claims']}",
    ]
    latest = summary.get("latest_sweep") or {}
    if latest:
        lines.append(
            f"latest sweep: {_when(latest.get('finished_at') or 0)} "
            f"({latest.get('trigger') or 'manual'}) — "
            f"{latest.get('observations') or 0} observations, "
            f"{latest.get('results') or 0} results, "
            f"{latest.get('failed') or 0} failed, "
            f"{latest.get('new_candidates') or 0} new candidates")
    else:
        lines.append("no sweep has run yet — action=run starts one")
    baselines = data.get("baselines") or {}
    if baselines:
        lines.append("\nbeats under watch:")
        for beat, base in sorted(baselines.items()):
            lines.append(
                f"· {beat} — {base['observations']} observations across "
                f"{base['sweep_count']} sweep(s), last "
                f"{_when(base['last_swept_at'])}")
    candidates = data.get("candidates") or []
    if candidates:
        lines.append("\nwatching (discovered, unpromoted):")
        for cand in candidates[:10]:
            lines.append(
                f"· {cand.get('name')} ({cand.get('kind')}) — seen "
                f"{cand.get('sightings')}×, last {_when(cand.get('last_seen') or 0)}")
    return "\n".join(lines)


def _render_run(report: dict[str, Any], beats: list[dict[str, Any]]) -> str:
    lines = [
        "🔭 **Exploration sweep complete**",
        f"beats swept: {len(report['beats'])}"
        + (f" (+{report['beats_deferred']} over the per-sweep cap, "
           "deferred)" if report["beats_deferred"] else ""),
        f"observations: {report['observations']} "
        f"({report['results']} results · {report['empty']} empty · "
        f"{report['failed']} failed)",
        f"new candidates noted: {report['new_candidates']}"
        + (f" (+{report['candidates_deferred']} deferred over the "
           "candidate cap)" if report["candidates_deferred"] else ""),
        "metadata only — no exemplar files were fetched",
    ]
    for row in report["beats"]:
        lines.append(
            f"· {row['beat']} — {row['results']} results, "
            f"{row['corroborated']} corroborated across instruments")
    return "\n".join(lines)


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "video_explore",
        description=(
            "Explore the video field Devon fetches from: sweeps watch "
            "the owner's interest beats (NewsAgent profile + interests "
            "watches) through video/YouTube/web search and record "
            "observations with provenance; channels/topics discovered "
            "are noted as unpromoted candidates. action=run (one "
            "sweep) | status | observations (beat=, limit=) | "
            "candidates | beats. Sweeps are unauthenticated public "
            "reads, metadata only — no files are downloaded."
        ),
        capability=Capability.NET_OUT,
    )
    def video_explore(action: str = "status", beat: str = "",
                      limit: int = 20) -> dict[str, Any]:
        if action == "run":
            store, _engine, beats = explore_for_context(context)
            report = run_exploration(context, store=store, beats=beats,
                                     trigger="tool")
            return {"report": report, "beats": beats,
                    "text": _render_run(report, beats)}
        store, _engine, beats = explore_for_context(context)
        if action == "status":
            data = explore_status(store)
            return {**data, "text": _render_status(data)}
        if action == "observations":
            rows = store.observations(beat=beat or "",
                                      limit=max(1, min(int(limit), 100)))
            return {"observations": rows, "count": len(rows)}
        if action == "candidates":
            rows = store.candidates()
            return {"candidates": rows, "count": len(rows)}
        if action == "beats":
            return {"beats": beats, "count": len(beats)}
        raise ToolError(f"unknown video_explore action {action!r}")
