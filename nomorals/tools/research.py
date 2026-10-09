"""research — spine tools for the autonomous research organ.

The brain schedules watches, inspects them, forces runs, and logs
knowledge gaps from plain language. The organ itself runs on the
durable scheduler (``research organ tick``) — the owner never trips
commands.
"""

from __future__ import annotations

from typing import Any

__all__ = ["register"]


def register(registry: Any) -> None:
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "research_schedule",
        description=(
            "Schedule an autonomous research watch: the research organ "
            "runs these queries on cadence, assesses findings, and delivers "
            "worthy ones as briefings without being asked. USE THIS when "
            "the owner wants to 'keep an eye on X' or 'watch for Y'. "
            "Args: topic, queries (list), cadence_hours (optional, default 24)."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "topic": "str — what the watch is about",
            "queries": "list[str] — the search queries to run each cycle",
            "cadence_hours": "float (optional) — hours between runs, default 24",
            "goals": "list[str] (optional) — goal tags: money, devon, arena, virtual_cam",
        },
    )
    def research_schedule(topic: str, queries: list[str],
                          cadence_hours: float = 24.0,
                          goals: list[str] | None = None,
                          **_: Any) -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.research.autonomy")
        organ = mod.ResearchOrgan(context)
        wid = organ.add_watch(topic, queries, float(cadence_hours or 24.0),
                              goals or [], created_by="brain")
        return {"ok": True, "watch_id": wid, "topic": topic,
                "note": "the research organ runs it on cadence, no commands needed"}

    @registry.register(
        "research_jobs",
        description=(
            "List the autonomous research watches: topic, cadence, enabled, "
            "last run. Use to answer 'what is research watching?'"
        ),
        capability=Capability.NET_OUT,
        parameters={},
    )
    def research_jobs(**_: Any) -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.research.autonomy")
        organ = mod.ResearchOrgan(context)
        return {"ok": True, "watches": organ.list_watches()}

    @registry.register(
        "research_run_now",
        description=(
            "Force an autonomous research watch to run immediately instead "
            "of waiting for its cadence. Args: watch_id (from research_jobs)."
        ),
        capability=Capability.NET_OUT,
        parameters={"watch_id": "str — the watch id"},
    )
    def research_run_now(watch_id: str,
                         **_: Any) -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.research.autonomy")
        organ = mod.ResearchOrgan(context)
        # Zero the last-run so the tick picks it up as due.
        organ.db.execute(
            "INSERT INTO research_job_state (job_id, last_run, last_status)"
            " VALUES (?, 0, 'forced')"
            " ON CONFLICT(job_id) DO UPDATE SET last_run = 0",
            (watch_id,))
        report = organ.tick()
        return {"ok": True, "watches_run": report.watches_run,
                "findings": report.findings, "delivered": report.delivered,
                "errors": report.errors}

    @registry.register(
        "research_remove_watch",
        description=(
            "Stop an autonomous research watch. Args: watch_id."
        ),
        capability=Capability.NET_OUT,
        parameters={"watch_id": "str — the watch id"},
    )
    def research_remove_watch(watch_id: str,
                              **_: Any) -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.research.autonomy")
        organ = mod.ResearchOrgan(context)
        return {"ok": organ.remove_watch(watch_id), "watch_id": watch_id}

    @registry.register(
        "note_knowledge_gap",
        description=(
            "Log something Devon doesn't know but should: the research "
            "organ turns open gaps into autonomous research runs and "
            "reports back. USE THIS whenever you hit a question you can't "
            "answer from memory or the corpus. Args: question, context "
            "(optional)."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "question": "str — the unanswered question",
            "gap_context": "str (optional) — why it matters / where it came up",
        },
    )
    def note_knowledge_gap(question: str, gap_context: str = "",
                           **_: Any) -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.research.autonomy")
        organ = mod.ResearchOrgan(context)
        gid = organ.note_gap(question, gap_context)
        return {"ok": True, "gap_id": gid,
                "note": "the research organ will research this on its own tick"}

    @registry.register(
        "research_weak_sources",
        description=(
            "Domains the research organ has learned are low-quality "
            "(learned from evidence, not a blocklist). Use to answer "
            "'which sources should we stop trusting?'"
        ),
        capability=Capability.NET_OUT,
        parameters={},
    )
    def research_weak_sources(**_: Any) -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.research.autonomy")
        organ = mod.ResearchOrgan(context)
        return {"ok": True, "weak_sources": organ.weak_sources()}

    @registry.register(
        "research_organ_tick",
        description=(
            "Run one autonomous research-organ cycle: due watches, "
            "knowledge gaps, source-quality learning, cross-organ events. "
            "This is what the scheduler runs — the owner never calls it "
            "directly."
        ),
        capability=Capability.NET_OUT,
        parameters={},
    )
    def research_organ_tick(**_: Any) -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.research.autonomy")
        organ = mod.ResearchOrgan(context)
        report = organ.tick()
        return {"ok": True,
                "watches_run": report.watches_run,
                "findings": report.findings,
                "delivered": report.delivered,
                "gaps_opened": report.gaps_opened,
                "gaps_resolved": report.gaps_resolved,
                "events_emitted": report.events_emitted,
                "errors": report.errors}
