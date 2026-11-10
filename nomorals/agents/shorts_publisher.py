"""The shorts publisher (WS3 P6) — the factory's missing last mile.

The shorts factory (``media/contentops``) could plan, render and
— through the CLI's manual ``post`` verb — publish ONE ready job
at a time. Its calendar (``ContentCalendar``) collected planned
posts with due times, and NOTHING ever fired them: no sweep read
``calendar.due()``, no job rendered a due post, no publisher ran
unless a human typed the CLI verb. This module is that sweep,
run as a delivery cell (``agents/content_cell.py``):

    plan evidence → render → skeptic (copy numbers must trace)
    → verifier (the rendered file is real) → per-platform
    delivery through the factory's own publishers → PublishLedger
    + calendar marks + receipt chain

Honesty rules, inherited from the publishers themselves: a
platform whose credentials are absent refuses with
:class:`CapabilityUnavailable` naming the manual step — the sweep
records that refusal in the ledger as a FAILED delivery and moves
on; it never fakes a post. Instagram/Facebook keep the CLI's own
gate (Meta fetches from a public HTTPS URL; a local render cannot
be delivered) — recorded the same way, in words. A planned post
with no platforms named can never publish and is marked failed
with that reason rather than re-swept forever.

The owner's queueing act is the authorization (``nm shorts queue``
/ the calendar he maintains) — the sweep executes his schedule,
the same posture as ``social/drafts.schedule_post``'s owner-named
scheduling. Nothing here invents content or timing.

Entry points:

* :func:`publish_due_shorts` — the sweep itself; the single
  function every trigger calls.
* the ``shorts_publish`` agent tool + :func:`ensure_shorts_publisher_job`
  — the durable path: every 30 minutes the runtime's scheduler
  fires the tool, exactly like the day derivation's evening job.
* :func:`install_shorts_publisher_job` — the unified-scheduler
  installer (register_action + cron), mirroring
  ``content_pipeline.install_weekly_content_job``, for the stacks
  that run that scheduler.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)

__all__ = [
    "PUBLISHER_JOB",
    "default_publisher_for",
    "publish_due_shorts",
    "install_shorts_publisher_job",
    "ensure_shorts_publisher_job",
    "register",
]

PUBLISHER_JOB = "shorts publisher"


# ── publishers (the factory's own, via the CLI's mapping) ────────────

def default_publisher_for(platform: str) -> Any:
    """Platform → publisher instance, the ``nm shorts post``
    mapping. Instagram/Facebook raise CapabilityUnavailable with
    the CLI's honest gate when CALLED through the sweep's adapter
    (see :func:`_deliver_one`)."""
    if platform == "youtube":
        from ..media.contentops.publish.youtube import YouTubePublisher

        return YouTubePublisher()
    if platform == "tiktok":
        from ..media.contentops.publish.tiktok import TikTokPublisher

        return TikTokPublisher()
    if platform in ("instagram", "reels"):
        from ..media.contentops.publish.meta import MetaPublisher

        return MetaPublisher()
    if platform == "facebook":
        from ..media.contentops.publish.meta import MetaPublisher

        return MetaPublisher()
    if platform == "x":
        from ..media.contentops.publish.x import XPublisher

        return XPublisher()
    return None


def _load_post_copy(pipe: Any, job: Any) -> dict[str, Any]:
    """Title/description/hashtags the pipeline wrote for this job
    (the CLI's loader, same file, same fallback)."""
    import json as _json

    if getattr(job, "run_id", ""):
        pc = Path(pipe.runs_root) / job.run_id / "post_copy.json"
        if pc.exists():
            try:
                return _json.loads(pc.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - fall back to the topic
                pass
    return {"title": job.topic, "description": "", "hashtags": []}


def _deliver_one(platform: str, publisher: Any, video_path: str,
                 copy: dict[str, Any], ledger: Any) -> dict[str, Any]:
    """Publish one rendered short to one platform through the
    factory's publisher. Returns the raw result dict; raises
    whatever the publisher raises (CapabilityUnavailable included)
    for the caller to record honestly."""
    from ..media.contentops.publish import CapabilityUnavailable

    title = str(copy.get("title") or "")
    description = str(copy.get("description") or "")
    tags = list(copy.get("hashtags") or [])
    if platform == "youtube":
        return publisher.publish(
            video_path, title, description=description, tags=tags,
            publish_at=None, privacy="public", confirmed=True,
            ledger=ledger)
    if platform == "tiktok":
        return publisher.post_video(
            video_path, caption=f"{title} {description}".strip(),
            tags=tags, ledger=ledger)
    if platform == "x":
        return publisher.post_video(
            video_path, title, tags=tags, ledger=ledger)
    if platform in ("instagram", "reels"):
        raise CapabilityUnavailable(
            platform,
            "instagram reels need a public HTTPS URL for the video"
            " (Meta fetches it); a local render cannot be delivered"
            " by this path",
            manual_step="upload the rendered file somewhere public,"
                      " then publish via the publisher API directly")
    if platform == "facebook":
        raise CapabilityUnavailable(
            platform,
            "facebook video needs a public HTTPS URL for the video"
            " or a page upload flow; a local render cannot be"
            " delivered by this path",
            manual_step="use the publisher API directly with a"
                      " public URL or a page upload")
    raise CapabilityUnavailable(
        platform, f"no publisher for platform {platform!r}",
        manual_step="name one of youtube|tiktok|instagram|facebook|x")


def _external_id(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    for key in ("video_id", "platform_video_id", "tweet_id", "id"):
        value = result.get(key)
        if value:
            return str(value)
    return ""


# ── the sweep ─────────────────────────────────────────────────────────

def _drafting_resolver(db: Any) -> Any:
    """The pipeline's niche resolver, read through the shared
    registry (WS4 P5, contract 2): floor plugins first, then the
    owner-ADOPTED overlay's data-driven plugins, then the shipped
    default. Built over the explore store bound to this database,
    via the adapter (``media/explore/feed.py``) — the sweep never
    learns how the overlay is stored, and a proposed niche is
    invisible to it. None when resolution cannot be built (the
    pipeline then uses its own default contract, exactly as
    before this wire existed)."""
    try:
        from ..media.explore.feed import drafting_plugin_resolver
        from ..media.explore.store import ExploreStore
        from ..storage.kv import KVStore

        return drafting_plugin_resolver(ExploreStore(KVStore(db)))
    except Exception as exc:  # noqa: BLE001 - resolver is an enhancement
        _log.debug("shorts niche resolver not built: %s", exc)
        return None


def publish_due_shorts(
    *,
    pipeline: Any = None,
    db: Any = None,
    publisher_for: Callable[[str], Any] | None = None,
    ledger: Any = None,
    now: Any = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Publish every due planned short, as a delivery cell per post.

    Returns one outcome dict per due post:
    ``{post_id, niche, topic, status: posted|failed|blocked,
    platforms: {platform: {ok, external_id | error}}, contract_id,
    reason}``. Never raises for one bad post — a failure is an
    outcome, recorded in the ledger and on the calendar.
    """
    from datetime import datetime, timezone

    if pipeline is None:
        from ..media.contentops.pipeline import ShortPipeline

        pipeline = ShortPipeline(
            get_niche=_drafting_resolver(db)) \
            if db is not None else ShortPipeline()
    if ledger is None:
        from ..media.contentops.publish.ledger import PublishLedger

        ledger = PublishLedger()
    if publisher_for is None:
        publisher_for = default_publisher_for
    moment = now or datetime.now(timezone.utc)
    due = pipeline.calendar.due(moment)[: max(0, int(limit))]
    outcomes: list[dict[str, Any]] = []
    for post in due:
        outcomes.append(_publish_one_post(
            post, pipeline=pipeline, db=db,
            publisher_for=publisher_for, ledger=ledger))
    return outcomes


def _publish_one_post(post: Any, *, pipeline: Any, db: Any,
                      publisher_for: Callable[[str], Any],
                      ledger: Any) -> dict[str, Any]:
    from .content_cell import ContentCell, ReceiptStore, untraced_numbers

    outcome: dict[str, Any] = {
        "post_id": post.id, "niche": post.niche, "topic": post.topic,
        "status": "failed", "platforms": {}, "contract_id": "",
        "reason": ""}
    cell: ContentCell | None = None
    contracts = None
    if db is not None:
        from .delegation import DelegationStore

        contracts = DelegationStore(db)
        cell = ContentCell(contracts, kind="shorts")
        cell.open(
            goal=(f"publish the due planned short: {post.niche} —"
                  f" {post.topic}"),
            deliverable=("the rendered short published to its named"
                         " platforms, recorded in the publish"
                         " ledger"),
            required_evidence=("plan", "render", "delivery"))
        outcome["contract_id"] = cell.contract_id
        platforms_text = ", ".join(post.platforms) or "none named"
        due_text = post.scheduled_for or "now"
        cell.evidence(
            "plan",
            f"planned short: {post.niche} — {post.topic}"
            f" (platforms: {platforms_text}; due {due_text})",
            producer=cell.researcher,
            evidence_ref=f"shorts:calendar:{post.id}", verify=True)

    platforms = list(post.platforms or [])
    if not platforms:
        outcome["reason"] = (
            "the planned post names no platforms — nothing to"
            " publish to; it can never fire, so it is marked failed"
            " rather than re-swept forever")
        pipeline.calendar.mark(post.id, "failed")
        if cell is not None:
            cell.fail(outcome["reason"], by=cell.steward)
        return outcome

    # Render (the worker).
    job = pipeline.plan(post.niche, post.topic,
                        platforms=platforms, style=post.style or "")
    pipeline.calendar.mark(post.id, "rendering", job_id=job.id)
    result = pipeline.run(job)
    if not getattr(result, "ok", False):
        outcome["reason"] = (
            f"render failed: {getattr(result, 'error', 'unknown')}")
        pipeline.calendar.mark(post.id, "failed", job_id=job.id)
        if cell is not None:
            cell.fail(outcome["reason"])
        return outcome
    video_path = str(getattr(result, "final_path", "")
                     or getattr(job, "final_path", "") or "")
    render_claim = None
    if cell is not None:
        def _file_is_real() -> bool:
            try:
                return Path(video_path).stat().st_size > 0
            except OSError:
                return False

        render_claim = cell.evidence(
            "render",
            f"rendered {job.id} → {video_path}",
            producer=cell.drafter, evidence_ref=video_path,
            verify=_file_is_real)
        if render_claim.verdict != "supported":
            outcome["reason"] = (
                "the render reported success but the file is not"
                " there to verify — nothing was published")
            pipeline.calendar.mark(post.id, "failed", job_id=job.id)
            cell.fail(outcome["reason"], by=cell.verifier)
            return outcome

    # Skeptic: the post copy's numbers must trace to the topic or
    # the script the pipeline wrote — invented figures block the
    # delivery before anything posts.
    copy = _load_post_copy(pipeline, job)
    if cell is not None:
        basis = post.topic + "\n" + str(copy.get("script") or "")
        bad = untraced_numbers(
            f"{copy.get('title') or ''}\n"
            f"{copy.get('description') or ''}", basis)
        if bad:
            claim = cell.evidence(
                "copy",
                f"post copy for {post.id}: "
                f"{str(copy.get('title') or '')[:60]}",
                producer=cell.drafter,
                evidence_ref=f"shorts:copy:{post.id}")
            cell.block(
                claim.id,
                f"post copy states number(s)"
                f" {', '.join(str(b) for b in bad)} with no source"
                " in the topic or the script",
                note="untraced numbers never publish")
            outcome["status"] = "blocked"
            outcome["reason"] = (
                "the skeptic blocked this post: its copy states"
                " numbers the topic and script never gave")
            pipeline.calendar.mark(post.id, "failed", job_id=job.id)
            return outcome

    # Delivery (the steward): one attempt per named platform.
    receipts = ReceiptStore(db) if db is not None else None
    posted = 0
    for platform in platforms:
        entry: dict[str, Any] = {"ok": False, "external_id": "",
                                 "error": ""}
        publisher = None
        try:
            publisher = publisher_for(platform)
            if publisher is None:
                raise ValueError(f"no publisher for {platform!r}")
            res = _deliver_one(platform, publisher, video_path,
                                copy, ledger)
            entry["ok"] = True
            entry["external_id"] = _external_id(res)
            entry["ledger_id"] = (
                str(res.get("ledger_id") or "") if isinstance(res, dict)
                else "")
            posted += 1
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            entry["error"] = f"{type(exc).__name__}: {exc}"[:240]
            # The publisher records its own successes; a refusal
            # before recording gets its honest ledger row here.
            try:
                failed = ledger.record(
                    platform=platform, file=video_path,
                    title=str(copy.get("title") or post.topic),
                    status="failed",
                    notes=entry["error"][:200])
                entry["ledger_id"] = str(failed.get("id") or "")
            except Exception as lexc:  # noqa: BLE001
                _log.warning("shorts ledger failure-record failed:"
                             " %s", lexc)
        outcome["platforms"][platform] = entry
        if receipts is not None and cell is not None:
            receipts.record(
                contract_id=cell.contract_id, kind="shorts",
                label=post.topic, platform=platform,
                post_ref=entry.get("ledger_id") or "",
                chain={"calendar_post": post.id,
                       "external_id": entry.get("external_id", ""),
                       "delivery": ("posted" if entry["ok"]
                                     else "failed"),
                       "error": entry.get("error", "")})
    if cell is not None:
        summary = "; ".join(
            f"{p}: {'posted' if e['ok'] else 'failed — ' + e['error']}"
            for p, e in outcome["platforms"].items())
        cell.deliver("delivery",
                     f"delivery for {post.id}: {summary}",
                     evidence_ref=f"shorts:delivery:{post.id}")
        decision = cell.finish()
        if not decision.ok:
            _log.warning("shorts contract %s could not complete: %s",
                         cell.contract_id,
                         "; ".join(decision.reasons or []))
    if posted:
        outcome["status"] = "posted"
        pipeline.calendar.mark(post.id, "posted", job_id=job.id)
    else:
        outcome["reason"] = (
            "every platform delivery failed — see the per-platform"
            " errors and the publish ledger")
        pipeline.calendar.mark(post.id, "failed", job_id=job.id)
    return outcome


# ── durable job (the runtime's scheduler) ─────────────────────────────

def ensure_shorts_publisher_job(scheduler: Any) -> list[dict[str, Any]]:
    """Register the durable shorts-publisher job (idempotent by
    name, the social/jobs.py pattern): every 30 minutes, fire the
    ``shorts_publish`` tool's ``run`` action. Routine: outcomes
    surface through the calendar, the ledger and /calendar, not
    job alerts — a failed sweep still records as failed."""
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    if PUBLISHER_JOB in have:
        return [{"name": PUBLISHER_JOB, "already_scheduled": True}]
    job = scheduler.add(
        PUBLISHER_JOB, "every 30m", "tool",
        {"tool": "shorts_publish", "args": {"action": "run"}},
        missed_fire_policy="skip", overlap_policy="skip",
        run_timeout_s=1800, routine=True)
    out.append({"name": PUBLISHER_JOB, "scheduled": True,
                "job_id": job.get("id")})
    return out


# ── unified-scheduler installer (the other scheduler's stacks) ───────

def install_shorts_publisher_job(
    scheduler: Any,
    *,
    cron: str = "*/30 * * * *",
    task_id: str = "social.shorts_publisher",
    sweep_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Register the sweep on a ``register_action`` scheduler —
    the ``install_weekly_content_job`` pattern, for the stacks
    that run the unified scheduler instead of the durable one."""
    kwargs = dict(sweep_kwargs or {})

    def _fire(**params: Any) -> Any:
        outcomes = publish_due_shorts(**kwargs)
        _log.info("shorts publisher: %d due post(s), %d posted",
                  len(outcomes),
                  sum(1 for o in outcomes if o["status"] == "posted"))
        return outcomes

    async def _fire_async(**params: Any) -> Any:
        return _fire(**params)

    if hasattr(scheduler, "register_action"):
        try:
            scheduler.register_action(task_id, _fire_async)
        except Exception:  # noqa: BLE001
            _log.debug("register_action failed", exc_info=True)
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        return loop.create_task(
            scheduler.schedule_cron(task_id, cron, task_id))
    _log.info("shorts publisher job registered (cron %s); schedule"
              " when a loop runs", cron)
    return None


# ── the agent tool ────────────────────────────────────────────────────

def register(registry: Any) -> None:
    """The ``shorts_publish`` tool — the durable job's payload and
    the brain's route when the owner asks for due shorts to go
    out. The description is the recognition surface; there is no
    keyword router anywhere in the path."""
    context = registry.context

    @registry.register(
        "shorts_publish",
        description=(
            "Publish the owner's DUE planned shorts (the content"
            " calendar he queues with nm shorts queue / the shorts"
            " calendar). action=run sweeps the calendar: each due"
            " post is rendered and delivered to its named platforms"
            " through the factory's publishers, recorded in the"
            " publish ledger — a platform without credentials"
            " refuses honestly and is recorded as failed, never"
            " faked. action=status reports the calendar's counts."
            " The durable 'shorts publisher' job fires action=run"
            " every 30 minutes; call it directly only when the"
            " owner asks for due shorts to go out now."
        ),
        capability="net.out",
        parameters={
            "action": "str — run | status",
            "limit": "int — max due posts per sweep (default 10)",
        },
    )
    def _shorts_publish(action: str = "run", *,
                        limit: int = 10) -> dict[str, Any]:
        action = (action or "run").strip().lower()
        if action == "status":
            try:
                from ..media.contentops.pipeline import ShortPipeline

                posts = ShortPipeline().calendar.list()
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"calendar unreadable:"
                                              f" {exc}"}
            counts: dict[str, int] = {}
            for p in posts:
                counts[p.status] = counts.get(p.status, 0) + 1
            return {"ok": True, "calendar": counts,
                    "total": len(posts)}
        if action == "run":
            db = getattr(context, "db", None)
            started = time.time()
            outcomes = publish_due_shorts(db=db, limit=int(limit))
            return {
                "ok": True,
                "processed": len(outcomes),
                "posted": sum(1 for o in outcomes
                              if o["status"] == "posted"),
                "outcomes": outcomes,
                "seconds": round(time.time() - started, 2)}
        return {"ok": False,
                "error": f"unknown action {action!r} (run | status)"}
