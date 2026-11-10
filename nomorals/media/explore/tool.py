"""The ``video_explore`` tool — the exploration substrate's surface.

WS4 Pass 1 ships the manual trigger + the read actions; P3 adds
``sweep`` (the scheduled cycle the durable jobs fire) and the
``/explore`` board (``board.py``). Actions:

* ``run`` — one sweep over the resolved beats (NewsAgent interest
  profile + interests watches) through the production instruments.
* ``sweep`` — the autonomous cycle (``jobs.run_sweep_cycle``):
  discipline gates, sweep, trend evaluation. What the durable
  morning/evening jobs call; a deferred cycle says why.
* ``status`` — the substrate summary: sweeps, observations,
  candidates being watched, per-beat baselines.
* ``observations`` — recent observation rows with provenance.
* ``candidates`` — the watching pool with sighting counts.
* ``beats`` — the beats a sweep would cover, with their origins.
* ``trends`` — WS4 Pass 2: evaluate the observation history into
  trend/spike/unverified claims (idempotent) and render the
  current claim set with each claim's kind, basis and freshness —
  the render source P3's board uses. ``refresh=false`` renders
  the stored claims without re-evaluating.
* ``niches`` — WS4 Pass 4: the shared niche registry (shipped
  floor + discovered overlay): proposals awaiting the owner's
  act, adopted entries, the effective set.
* ``adopt`` / ``dismiss`` — the owner's acts on a niche proposal
  (``slug=`` names it, by slug or board number): adoption flips
  the overlay record and scaffold-generates a DRAFT plugin —
  nothing discovered goes live without this act.

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
from .trends import current_claims, eliminated_branches, evaluate_trends

__all__ = ["register", "render_claims_text", "render_run_text"]


def render_claims_text(claims: list[dict[str, Any]],
                       report: dict[str, Any] | None,
                       eliminated: list[dict[str, Any]] | None = None,
                       ) -> str:
    """The claims render (P2's seam, made public for P3's board):
    kind, basis in plain words, freshness — no internal ids.
    PG4: preserved eliminated branches render after the live
    claims, in their own labelled section — never among them."""
    return _render_trends(claims, report, eliminated)


def render_run_text(report: dict[str, Any]) -> str:
    """A sweep report as text (the ``run`` action's render, made
    public for the board's cycle render)."""
    return _render_run(report, report.get("beats") or [])


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


def _render_trends(claims: list[dict[str, Any]],
                   report: dict[str, Any] | None,
                   eliminated: list[dict[str, Any]] | None = None,
                   ) -> str:
    lines = ["🔭 **Exploration trends**"]
    if not claims:
        lines.append("no claims yet — run a sweep (action=run), then "
                     "evaluate again; one sweep alone can only ever "
                     "be a spike or unverified, never a trend")
    for claim in claims:
        legs = claim.get("basis") or {}
        bits = [f"{len(legs.get('sweeps') or [])} sweep(s)"]
        classes = legs.get("source_classes") or []
        if classes:
            bits.append("classes: " + " + ".join(classes))
        velocity = legs.get("velocity") or {}
        if velocity.get("from_zero"):
            bits.append("velocity: burst from a zero baseline")
        elif velocity.get("ratio") is not None:
            bits.append(f"velocity {velocity['ratio']}x baseline")
        if legs.get("corroborated_urls"):
            bits.append(f"{legs['corroborated_urls']} corroborated "
                        "across classes")
        if legs.get("corroborated_sightings"):
            bits.append(f"{legs['corroborated_sightings']} sightings "
                        "corroborated across classes")
        if legs.get("single_source"):
            bits.append("single-source (seed signal only)")
        standing = claim.get("standing") or "fresh"
        grade = str((claim.get("proof_grade") or {}).get("grade")
                    or "G0")
        lines.append(
            f"· [{claim.get('kind')}] {claim.get('text')} — "
            f"{'; '.join(bits)} · proof {grade} · {standing}, "
            f"last evidence {_when(claim.get('observed_at') or 0)}")
    if eliminated:
        lines.append(
            "\neliminated branches (preserved — eliminated, "
            "not live claims):")
        for branch in eliminated[:10]:
            grade = str((branch.get("proof_grade") or {}).get(
                "grade") or "G0")
            line = (
                f"· [{branch.get('kind')}] {branch.get('text')} "
                f"— eliminated: {branch.get('reason')} · proof "
                f"{grade} (its own evidence) · "
                f"{_when(branch.get('eliminated_at') or 0)}")
            winner = branch.get("superseded_by") or {}
            if winner.get("text"):
                line += f" · superseded by: {winner['text']}"
            lines.append(line)
        if len(eliminated) > 10:
            lines.append(
                f"· … and {len(eliminated) - 10} more preserved "
                "in the store")
    if report:
        verdicts = report.get("candidates") or {}
        if verdicts.get("promoted"):
            lines.append(
                f"candidates promoted on evidence this evaluation: "
                f"{len(verdicts['promoted'])}")
        if verdicts.get("shed"):
            lines.append(
                f"candidates shed (absent past the threshold): "
                f"{len(verdicts['shed'])}")
        declined = report.get("declined") or []
        if declined:
            lines.append("\ndeclined to claim:")
            for row in declined:
                lines.append(
                    f"· {row.get('subject')} — {row.get('reason')}")
        if report.get("eliminated_preserved"):
            lines.append(
                f"eliminated branches preserved this evaluation: "
                f"{report['eliminated_preserved']}")
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
            "candidates | beats | trends (evaluate observations into "
            "trend/spike/unverified claims with their evidence basis "
            "and freshness; refresh=false renders without "
            "re-evaluating) | sweep (the scheduled cycle the "
            "durable jobs fire: quiet-hours + per-day-cap "
            "discipline, then a sweep, then trend evaluation — "
            "ends with fresh claims, or a deferral that says why) "
            "| niches (the shared niche registry: shipped floor "
            "+ discovered proposals + the effective set) | "
            "adopt / dismiss (slug= — the owner's acts on a niche "
            "proposal: adoption flips the overlay record and "
            "scaffold-generates a DRAFT plugin; nothing "
            "discovered goes live without the act). "
            "Sweeps are unauthenticated public "
            "reads, metadata only — no files are downloaded."
        ),
        capability=Capability.NET_OUT,
    )
    def video_explore(action: str = "status", beat: str = "",
                      limit: int = 20,
                      refresh: bool = True,
                      slug: str = "") -> dict[str, Any]:
        if action == "sweep":
            from .board import render_sweep_cycle
            from .jobs import run_sweep_cycle

            cycle = run_sweep_cycle(context)
            return {**cycle, "text": render_sweep_cycle(cycle)}
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
        if action == "trends":
            now = time.time()
            report = evaluate_trends(store, now=now) if refresh else None
            claims = current_claims(store, now=now)
            eliminated = eliminated_branches(store)
            return {"claims": claims, "count": len(claims),
                    "evaluation": report,
                    "eliminated": eliminated,
                    "text": _render_trends(claims, report, eliminated)}
        if action == "niches":
            from .board import render_explore_niches
            from .niches import effective_niches

            return {
                "proposals": store.niche_proposals(),
                "effective": effective_niches(store),
                "text": render_explore_niches(context, store=store),
            }
        if action in ("adopt", "dismiss"):
            from .board import render_niche_act
            from .niches import adopt_niche, dismiss_niche

            act = adopt_niche if action == "adopt" else dismiss_niche
            result = act(store, slug)
            return {**result,
                    "text": render_niche_act(result, action)}
        raise ToolError(f"unknown video_explore action {action!r}")
