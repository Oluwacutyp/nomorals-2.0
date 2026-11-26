"""TrendClip tools — the registry face of the trendclip organ.

Five tools. The deterministic rung runs in every tool. The brain rung
lives in the ``/clip`` handler, not here: a tool call runs inside the
brain's own tool loop, so the brain IS the picker — it reads the
candidates the analyze tool returns and chooses. When the owner runs
``/clip`` in chat, the handler consults the brain for the pick and both
rungs land in the job receipt.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from ..media.trendclip import TrendclipPipeline, topic_from_query, trendclip_root

__all__ = ["register"]


def register(registry: Any) -> None:
    """Attach the trendclip tools to a registry."""
    context = registry.context

    def pipeline() -> TrendclipPipeline:
        return TrendclipPipeline(trendclip_root(context.settings))

    @registry.register(
        "trendclip_discover",
        description="Discover trending YouTube videos for a topic, ranked by views-per-hour and a deterministic relevance/freshness score.",
        capability=Capability.NET_DOWNLOAD,
    )
    def trendclip_discover(query: str, *, strategy: str = "conversation",
                           limit: int = 10) -> dict[str, Any]:
        """Trending videos for a topic — the intake side of the shorts
        factory. Returns the ranked list with titles, URLs, channels,
        views-per-hour and feed scores."""
        topic = topic_from_query(query, strategy=strategy)
        problems = topic.validate()
        if problems:
            return {"ok": False, "error": "; ".join(problems)}
        items = pipeline().discover(topic, limit=limit)
        if items and "error" in items[0]:
            return {"ok": False, "error": str(items[0]["error"])}
        return {"ok": True, "topic": topic.slug, "strategy": topic.strategy,
                "items": items}

    @registry.register(
        "trendclip_analyze",
        description="Ingest a video URL, transcribe it with local Faster Whisper, and score its moments deterministically (signal evidence + editorial quality 0-10). Returns the job and its ranked candidates.",
        capability=Capability.NET_DOWNLOAD,
    )
    def trendclip_analyze(url: str, *, query: str = "",
                          strategy: str = "conversation",
                          model: str = "small") -> dict[str, Any]:
        """Analyze one source into ranked clip candidates. The brain rung
        does not run inside this tool — the calling brain reads the
        candidates and picks; the /clip handler consults the brain when
        the owner runs it in chat."""
        topic = topic_from_query(query or url, strategy=strategy)
        pipe = pipeline()
        job = pipe.analyze(url, topic, model=model)
        if job.stage == "failed":
            return {"ok": False, "job_id": job.job_id,
                    "error": job.error or "analysis failed"}
        receipt = job.receipt or {}
        return {
            "ok": True,
            "job_id": job.job_id,
            "stage": job.stage,
            "source": receipt.get("source") or {},
            "rights": (receipt.get("rights") or {}).get("label") or "",
            "moments": receipt.get("moments") or [],
            "editorial_summary": receipt.get("editorial_summary") or "",
            "editorial_issues": receipt.get("editorial_issues") or [],
            "brain_rung": ("the calling brain picks from these candidates; "
                           "the /clip handler consults the brain when the "
                           "owner runs it in chat"),
        }

    @registry.register(
        "trendclip_render",
        description="Render one analyzed moment into a validated vertical 1080x1920 clip (-14 LUFS) with captions, thumbnail, provenance and a checksummed export package.",
        capability=Capability.FS_WRITE,
    )
    def trendclip_render(job_id: str, pick: int, *,
                         layout: str = "balanced",
                         destinations: list[str] | None = None
                         ) -> dict[str, Any]:
        """Render a picked moment. Third-party media renders as a review
        draft; anything without usable rights is refused. The package is
        ready for manual upload — nothing auto-posts."""
        pipe = pipeline()
        try:
            job = pipe.render(job_id, pick, layout=layout,
                              destinations=destinations)
        except (KeyError, RuntimeError) as exc:
            return {"ok": False, "job_id": job_id, "error": str(exc)}
        receipt = job.receipt or {}
        render_info = receipt.get("render") or {}
        validation = receipt.get("validation") or {}
        if job.stage == "failed":
            return {"ok": False, "job_id": job.job_id,
                    "error": job.error or "render failed",
                    "validation": validation}
        return {
            "ok": True,
            "job_id": job.job_id,
            "stage": job.stage,
            "clip": render_info.get("clip_path") or "",
            "duration": render_info.get("duration"),
            "layout": render_info.get("layout"),
            "validation_passed": bool(validation.get("passed")),
            "validation_problems": validation.get("problems") or [],
            "package": render_info.get("package") or {},
            "repurpose": pipe.repurpose_source(job.job_id),
        }

    @registry.register(
        "trendclip_status",
        description="Status of a trendclip job: its stage, moments, brain picks and rendered clip.",
        capability=Capability.FS_WRITE,
    )
    def trendclip_status(job_id: str = "") -> dict[str, Any]:
        """One job's status, or the recent jobs when no id is given."""
        pipe = pipeline()
        if not job_id:
            return {"ok": True, "jobs": pipe.list_jobs()}
        try:
            return {"ok": True, **pipe.status(job_id)}
        except KeyError as exc:
            return {"ok": False, "error": str(exc)}

    @registry.register(
        "trendclip_queue",
        description="Queue a rendered trendclip clip into the shorts calendar for the publisher sweep, which delivers the pre-rendered clip (it does not re-render it). The owner queues it; the sweep's ledger records the delivery.",
        capability=Capability.FS_WRITE,
    )
    def trendclip_queue(job_id: str, *,
                        platforms: list[str] | None = None,
                        scheduled_for: str = "") -> dict[str, Any]:
        """Queue a rendered clip for the publisher sweep."""
        pipe = pipeline()
        try:
            return {"ok": True,
                    **pipe.queue(job_id, platforms=platforms,
                                 scheduled_for=scheduled_for)}
        except (KeyError, RuntimeError) as exc:
            return {"ok": False, "job_id": job_id, "error": str(exc)}
