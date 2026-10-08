"""Research → draft → schedule content pipeline (build-map #45).

Devon's unfair advantage, composed from #42-44 + the research pipeline:

    "what's working in my niche this week?"
        → research (bounded budget)
        → 5-7 drafts in the owner's voice (#44 VoiceProfile)
        → virality-scored (#44)
        → queued for review (#43 DraftQueue — default asks, never auto-posts)
        → scheduled at the audience's REAL optimal windows (engagement curve
           from social_posts, never generic "best times" advice)

Composition only — no duplicated logic. Research is budget-bounded.
Nothing posts without the owner's explicit override (#43 rule).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Sequence

_log = logging.getLogger(__name__)

__all__ = [
    "PipelineDraft",
    "PipelineResult",
    "optimal_windows",
    "weekly_content",
    "install_weekly_content_job",
    "FALLBACK_WINDOWS",
]

#: Fallback posting windows (dow 0=Mon, hour 24h) used only when there is
#: NO engagement data at all. Documented as fallback, not advice.
#: Tue-Thu 9:00 and 18:00 — sensible defaults, nothing more.
FALLBACK_WINDOWS: list[tuple[int, int]] = [
    (1, 9), (1, 18), (2, 9), (2, 18), (3, 9), (3, 18),
]

#: Metrics keys that count as "engagement" across adapters.
_ENGAGEMENT_KEYS = (
    "likes", "favorites", "favourites", "reposts", "reblogs",
    "replies", "comments", "shares", "quotes", "bookmarks",
    "impressions", "views", "engagement",
)

#: Default research budget for one weekly run (USD). Bounded, always.
DEFAULT_RESEARCH_BUDGET_USD = 1.0


@dataclass
class PipelineDraft:
    """One draft produced by the pipeline."""

    content: str
    platform: str
    virality: float
    virality_grade: str
    finding_title: str = ""
    finding_url: str = ""
    scheduled_for: float = 0.0  # unix ts, 0 = unscheduled
    draft_id: str = ""
    weak: bool = False  # flagged, never silently dropped


@dataclass
class PipelineResult:
    """Everything one weekly run produced."""

    niche: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    drafts: list[PipelineDraft] = field(default_factory=list)
    windows: list[datetime] = field(default_factory=list)
    windows_from_data: bool = False
    message: str = ""
    seconds: float = 0.0


# ── 1. optimal windows from real engagement data ────────────────────────────


def _engagement_of(metrics: Any) -> float:
    """Extract an engagement number from a post's metrics JSON."""
    if not metrics:
        return 0.0
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except Exception:  # noqa: BLE001
            return 0.0
    if not isinstance(metrics, dict):
        return 0.0
    total = 0.0
    for key in _ENGAGEMENT_KEYS:
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            total += val
    return total


def optimal_windows(
    db: Any,
    *,
    days: int = 90,
    n: int = 5,
    now: datetime | None = None,
) -> tuple[list[datetime], bool]:
    """Compute the audience's optimal posting windows from REAL data.

    Groups past posted posts by (day-of-week, hour), ranks by average
    engagement, and returns the next ``n`` upcoming datetimes at the top
    windows. Returns ``(windows, from_data)`` — ``from_data`` is False
    when falling back to :data:`FALLBACK_WINDOWS` (no engagement data).
    """
    now = now or datetime.now()
    cutoff = time.time() - days * 86400
    buckets: dict[tuple[int, int], list[float]] = {}
    from_data = False
    try:
        rows = db.query(
            "SELECT posted_at, metrics FROM social_posts "
            "WHERE posted_at IS NOT NULL AND posted_at > ? "
            "AND status = 'posted'",
            (cutoff,))
        for row in rows:
            try:
                posted = row["posted_at"]
                dt = datetime.fromtimestamp(float(posted))
            except Exception:  # noqa: BLE001 - bad row, skip
                continue
            eng = _engagement_of(row["metrics"] if "metrics" in row.keys()
                                 else None)
            buckets.setdefault((dt.weekday(), dt.hour), []).append(eng)
    except Exception as exc:  # noqa: BLE001 - no table / bad db → fallback
        _log.debug("optimal_windows query failed, using fallback: %s", exc)
        buckets = {}

    ranked: list[tuple[int, int]] = []
    if buckets:
        scored = sorted(
            buckets.items(),
            key=lambda kv: (sum(kv[1]) / len(kv[1]), len(kv[1])),
            reverse=True,
        )
        ranked = [dow_hour for dow_hour, _ in scored[:n]]
        from_data = True
    else:
        ranked = list(FALLBACK_WINDOWS[:n])

    windows: list[datetime] = []
    for dow, hour in ranked:
        delta_days = (dow - now.weekday()) % 7
        candidate = (now + timedelta(days=delta_days)).replace(
            hour=hour, minute=0, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=7)
        windows.append(candidate)
    windows.sort()
    return windows[:n], from_data


# ── 2. research (bounded) ────────────────────────────────────────────────────


def _default_research(
    niche: str,
    rctx: Any,
    *,
    budget_usd: float = DEFAULT_RESEARCH_BUDGET_USD,
) -> list[dict[str, Any]]:
    """Run a bounded ResearchJob on the niche. Never raises."""
    try:
        from ..research.pipeline import (
            ResearchBudget, ResearchContext, ResearchJob, run_job)
        from ..core.ids import new_id
        job = ResearchJob(
            id=new_id("research"),
            topic=f"what's working in {niche} this week",
            queries=[
                f"what's trending in {niche} this week",
                f"{niche} viral posts formats 2026",
                f"{niche} news this week",
            ],
            budget_usd=budget_usd,
            max_results=6,
            fetch_top=1,
        )
        ctx = rctx if isinstance(rctx, ResearchContext) else ResearchContext(
            db=getattr(rctx, "db", None), registry=getattr(rctx, "registry", None))
        budget = ResearchBudget(limit_usd=budget_usd)
        findings = run_job(job, ctx, budget=budget)
        return [
            {"title": f.title, "snippet": f.snippet, "url": f.url}
            for f in findings
        ]
    except Exception as exc:  # noqa: BLE001 - research is best-effort
        _log.warning("pipeline research failed: %s", exc)
        return []


# ── 3. drafting ─────────────────────────────────────────────────────────────


def _template_draft(finding: dict[str, Any], variant: int = 0) -> str:
    """Honest structured drafts from a finding (no-LLM fallback).

    ``variant`` rotates the angle so one finding yields several distinct
    drafts: 0 = news hook, 1 = contrarian take, 2 = practical implication.
    """
    title = (finding.get("title") or "").strip()
    snippet = (finding.get("snippet") or "").strip()
    if len(snippet) > 160:
        snippet = snippet[:157].rsplit(" ", 1)[0] + "…"
    url = (finding.get("url") or "").strip()
    suffix = f"\n{url}" if url else ""
    v = variant % 3
    if v == 0:
        body = f"🔥 {title}" if title else "Worth your attention:"
        if snippet:
            body += f"\n\n{snippet}"
        body += "\n\nWhat's your take — hype or real shift?"
    elif v == 1:
        body = f"Unpopular opinion on \"{title}\":" if title else \
            "Unpopular opinion:"
        body += ("\n\nEveryone's excited. I'm watching what breaks first. "
                 "The second-order effects are where the real story is.")
        body += "\n\nAgree or am I wrong?"
    else:
        body = "Here's what this actually means for you:\n\n" if title else ""
        if title:
            body = f"\"{title}\"\n\nHere's what this actually means for you:\n\n"
        if snippet:
            body += f"{snippet}\n\n"
        body += "The winners won't be the ones who read about it — they'll be the ones who act on it this week."
    return body + suffix


def _llm_draft(
    finding: dict[str, Any],
    voice_brief: str,
    llm_fn: Callable[[str], str],
    variant: int = 0,
) -> str:
    angles = ("a news-style hook", "a contrarian take", "a practical implication")
    prompt = (
        "Write ONE social media post draft (no preamble, no quotes around it) "
        f"about this: {(finding.get('title') or '').strip()} — "
        f"{(finding.get('snippet') or '').strip()[:300]}. "
        f"Angle: {angles[variant % 3]}. "
        f"Write it in this voice: {voice_brief}. "
        "Keep it under 240 characters. End with a question or CTA."
    )
    try:
        text = (llm_fn(prompt) or "").strip()
    except Exception as exc:  # noqa: BLE001
        _log.warning("pipeline llm draft failed: %s", exc)
        text = ""
    return text


def _generate_drafts(
    findings: list[dict[str, Any]],
    profile: Any,
    *,
    llm_fn: Callable[[str], str] | None = None,
    platforms: Sequence[str] = ("x",),
    min_drafts: int = 5,
    max_drafts: int = 7,
) -> list[str]:
    """5-7 draft texts from findings, shaped toward the voice profile."""
    from .voice import apply_voice
    drafts: list[str] = []
    pool = list(findings) or [{"title": "", "snippet": "", "url": ""}]
    i = 0
    while len(drafts) < max_drafts and i < len(pool) * 3:
        finding = pool[i % len(pool)]
        variant = i // len(pool)
        i += 1
        if llm_fn is not None and finding.get("title"):
            text = _llm_draft(finding, profile.describe(), llm_fn, variant)
            if not text:
                text = _template_draft(finding, variant)
        else:
            text = _template_draft(finding, variant)
            # Rule-based voice shaping (honest fallback, documented in voice.py).
            try:
                text = apply_voice(text, profile)
            except Exception:  # noqa: BLE001
                pass
        text = text.strip()
        if text and text not in drafts:
            drafts.append(text)
    # Pad with generic drafts if findings were thin — marked honestly.
    _PADS = [
        "Building in public this week — what's one thing you shipped "
        "that you're proud of? Drop it below 👇",
        "What's the hardest lesson you learned building your thing? "
        "Mine: nobody cares until it works. Then everyone has opinions.",
        "Quick question for builders: what's your current bottleneck — "
        "time, money, or distribution? Be honest.",
        "Shipped > perfect. What are you shipping this week?",
        "The gap between idea and execution is just reps. "
        "What's one rep you're doing today?",
    ]
    pi = 0
    while len(drafts) < min_drafts and pi < len(_PADS):
        pad = _PADS[pi]
        pi += 1
        if pad not in drafts:
            drafts.append(pad)
    return drafts[:max_drafts]


# ── 4. the pipeline ─────────────────────────────────────────────────────────


def weekly_content(
    niche: str,
    *,
    db: Any = None,
    queue: Any = None,
    voice_profile: Any = None,
    platforms: Sequence[str] = ("x",),
    llm_fn: Callable[[str], str] | None = None,
    findings: list[dict[str, Any]] | None = None,
    rctx: Any = None,
    scheduler: Any = None,
    research_budget_usd: float = DEFAULT_RESEARCH_BUDGET_USD,
    min_drafts: int = 5,
    max_drafts: int = 7,
    weak_threshold: float = 40.0,
) -> PipelineResult:
    """Run the full loop: research → draft → score → queue → schedule.

    Drafts are queued for REVIEW (default per #43) — never auto-posted.
    ``findings`` injects research (tests); otherwise a bounded ResearchJob
    runs when ``rctx`` is given; otherwise honest generic templates.
    """
    from .drafts import DraftQueue, schedule_post
    from .voice import VoiceProfile, virality_score

    started = time.perf_counter()
    niche = (niche or "").strip() or "general"

    # Research (bounded).
    found = list(findings) if findings is not None else []
    if findings is None and rctx is not None:
        found = _default_research(niche, rctx, budget_usd=research_budget_usd)

    # Voice profile.
    profile = voice_profile
    if profile is None:
        try:
            profile = VoiceProfile.load()
        except Exception:  # noqa: BLE001
            profile = VoiceProfile()

    # Draft in the owner's voice.
    texts = _generate_drafts(
        found, profile, llm_fn=llm_fn, platforms=platforms,
        min_drafts=min_drafts, max_drafts=max_drafts)

    # Score each draft (past winners from engagement data when available).
    past_winners = _past_winners(db) if db is not None else None
    scored: list[PipelineDraft] = []
    for text in texts:
        platform = platforms[0] if platforms else "x"
        try:
            vs = virality_score(text, platform=platform,
                                past_winners=past_winners)
        except Exception:  # noqa: BLE001
            from .voice import ViralityScore as _VS
            vs = _VS(50.0, ["scoring unavailable"])
        scored.append(PipelineDraft(
            content=text, platform=platform,
            virality=vs.score, virality_grade=vs.grade,
            weak=vs.score < weak_threshold,
        ))
    # Strongest first; the weakest are flagged, never silently dropped.
    scored.sort(key=lambda d: d.virality, reverse=True)
    for idx, d in enumerate(scored):
        if found and idx < len(found):
            d.finding_title = str(found[idx].get("title", ""))
            d.finding_url = str(found[idx].get("url", ""))

    # Optimal windows from real engagement data.
    windows: list[datetime] = []
    windows_from_data = False
    if db is not None:
        try:
            windows, windows_from_data = optimal_windows(
                db, n=max(len(scored), 1))
        except Exception:  # noqa: BLE001
            _log.debug("optimal_windows failed", exc_info=True)
    if not windows:
        windows, windows_from_data = optimal_windows(
            _NullDB(), n=max(len(scored), 1))

    # Queue for review + schedule at the windows (round-robin).
    q = queue or DraftQueue()
    for i, d in enumerate(scored):
        win = windows[i % len(windows)]
        win_ts = win.timestamp()
        try:
            draft = q.create_draft(
                d.content, platforms=list(platforms),
                metadata={"niche": niche, "virality": round(d.virality, 1),
                          "grade": d.virality_grade,
                          "finding_url": d.finding_url,
                          "pipeline": "weekly_content"})
            q.propose(draft.id)  # → pending_review: Devon asks, never auto-posts
            d.draft_id = draft.id
            if scheduler is not None:
                try:
                    schedule_post(scheduler, q, draft.id, win_ts,
                                  platforms=list(platforms))
                except Exception:  # noqa: BLE001 - scheduler optional
                    q.schedule(draft.id, win_ts)
            else:
                q.schedule(draft.id, win_ts)
            d.scheduled_for = win_ts
        except Exception as exc:  # noqa: BLE001 - one bad draft ≠ dead pipeline
            _log.warning("pipeline draft %d failed: %s", i, exc)

    message = _batch_message(niche, scored, windows, windows_from_data)
    return PipelineResult(
        niche=niche, findings=found, drafts=scored, windows=windows,
        windows_from_data=windows_from_data, message=message,
        seconds=time.perf_counter() - started)


class _NullDB:
    """DB that answers nothing — forces the documented fallback windows."""

    def query(self, *args: Any, **kwargs: Any) -> list:
        return []


def _past_winners(db: Any, *, limit: int = 5) -> list[str] | None:
    """Top-engagement past post contents for virality novelty comparison."""
    try:
        rows = db.query(
            "SELECT content, metrics FROM social_posts "
            "WHERE status = 'posted' ORDER BY created_at DESC LIMIT 200")
    except Exception:  # noqa: BLE001
        return None
    scored = []
    for row in rows:
        try:
            keys = row.keys()
            content = row["content"] if "content" in keys else ""
        except Exception:  # noqa: BLE001
            continue
        eng = _engagement_of(row["metrics"] if "metrics" in keys else None)
        if content:
            scored.append((eng, str(content)))
    scored.sort(key=lambda t: t[0], reverse=True)
    winners = [c for _, c in scored[:limit] if c]
    return winners or None


def _batch_message(
    niche: str,
    drafts: list[PipelineDraft],
    windows: list[datetime],
    from_data: bool,
) -> str:
    lines = [
        f"📰 This week's content for *{niche}*: "
        f"{len(drafts)} drafts ready for review.",
        "",
    ]
    for i, d in enumerate(drafts, 1):
        preview = d.content[:90].replace("\n", " ")
        if len(d.content) > 90:
            preview += "…"
        when = (datetime.fromtimestamp(d.scheduled_for).strftime("%a %H:%M")
                if d.scheduled_for else "unscheduled")
        flag = " ⚠️ weak" if d.weak else ""
        lines.append(
            f"{i}. [virality {d.virality:.0f} {d.virality_grade}]{flag} "
            f"⏰ {when}\n   {preview}")
    lines.append("")
    lines.append(
        "Scheduled at your audience's best windows "
        + ("(from your engagement data)" if from_data
           else "(fallback windows — no engagement data yet)")
        + ". Say the word and I'll post one, or tap a draft to review.")
    return "\n".join(lines)


# ── 5. weekly cron ───────────────────────────────────────────────────────────


def install_weekly_content_job(
    scheduler: Any,
    niche: str,
    *,
    cron: str = "0 8 * * MON",
    task_id: str = "social.weekly_content",
    present_fn: Callable[[str], Any] | None = None,
    pipeline_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Register the weekly pipeline run.

    Each run presents the batch for review — it never auto-posts
    (respects the #43 rule). ``present_fn(message)`` delivers the batch
    (chat send, …); without it the result is logged.
    Follows the ``install_evergreen_job`` pattern from voice.py.
    """
    kwargs = dict(pipeline_kwargs or {})

    def _fire(**params: Any) -> Any:
        result = weekly_content(niche, **kwargs)
        _log.info("weekly content: %d drafts for %s (%.1fs)",
                  len(result.drafts), niche, result.seconds)
        if present_fn is not None:
            try:
                res = present_fn(result.message)
                if hasattr(res, "__await__"):
                    return res
            except Exception as exc:  # noqa: BLE001
                _log.warning("weekly content present_fn failed: %s", exc)
        return result

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
    _log.info("weekly content job registered (cron %s); schedule when a loop runs",
              cron)
    return None
