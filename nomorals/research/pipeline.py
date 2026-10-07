"""Core research pipeline: run a job, judge worth, deliver to the owner.

``run_job`` uses the tool registry's ``web_search``/``web_fetch`` (DuckDuckGo,
no API key — the repo's existing research path). ``assess_worth`` is the
conservative gate: a finding must be new, relevant to the owner's stated
goals, and fresh/actionable enough to earn an interruption. ``deliver`` sends
through the ChatGateway to owner chats and records the delivery so nothing is
ever sent twice.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

# ── schema ───────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_job_state (
  job_id     TEXT PRIMARY KEY,
  last_run   REAL NOT NULL DEFAULT 0,
  last_status TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS research_seen (
  url_hash   TEXT PRIMARY KEY,
  job_id     TEXT NOT NULL,
  title      TEXT NOT NULL DEFAULT '',
  first_seen REAL NOT NULL,
  best_score REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS research_deliveries (
  url_hash     TEXT PRIMARY KEY,
  job_id       TEXT NOT NULL,
  title        TEXT NOT NULL DEFAULT '',
  delivered_at REAL NOT NULL,
  score        REAL NOT NULL DEFAULT 0,
  channel      TEXT NOT NULL DEFAULT ''
);
"""


def ensure_schema(db: Any) -> None:
    """Create the research tables. Idempotent."""
    db.executescript(_SCHEMA)


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().lower().encode("utf-8")).hexdigest()[:32]


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass
class ResearchJob:
    """A scheduled research task."""

    id: str
    topic: str
    queries: list[str]
    cadence_hours: float = 24.0
    goals: list[str] = field(default_factory=list)
    max_results: int = 6
    fetch_top: int = 2
    enabled: bool = True


@dataclass
class ResearchFinding:
    job_id: str
    title: str
    url: str
    snippet: str
    fetched_at: float = field(default_factory=time.time)
    detail: str = ""


@dataclass
class Assessment:
    worth: bool
    score: float
    reasons: list[str]


@dataclass
class ResearchContext:
    """Everything the pipeline needs, injected so tests can fake it."""

    db: Any
    registry: Any  # ToolRegistry with web_search / web_fetch
    memory: Any | None = None  # MemoryManager, optional
    gateway: Any | None = None  # ChatGateway, optional
    goal_keywords: dict[str, list[str]] | None = None
    worth_threshold: float = 0.65
    daily_delivery_cap: int = 3


#: Goal tags -> keywords the owner actually cares about. Jobs tag themselves
#: with goals; findings must hit at least one goal's keywords to be relevant.
DEFAULT_GOAL_KEYWORDS: dict[str, list[str]] = {
    "money": [
        "outlier", "mindrift", "ai training", "ai trainer", "data annotation",
        "freelance", "remote job", "hiring", "pay per hour", "$", "earn",
        "gig", "upwork", "turing",
    ],
    "virtual_cam": [
        "virtual camera", "face swap", "face fusion", "deepfake", "obs virtual",
        "android camera", "xposed", "camera2",
    ],
    "devon": [
        "telegram bot", "llm agent", "coding agent", "llama.cpp", "gguf",
        "fine-tune", "qlora", "local llm",
    ],
    "arena": [
        "lagos life", "browser game", "webgl game", "multiplayer game",
        "indie game nigeria",
    ],
}

_RECENCY_RE = re.compile(
    r"\b(today|yesterday|this week|just launched|launched|"
    r"new release|breaking|2026)\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(deadline|closes?|apply|sign ?up|hiring|payout|release|"
    r"discount|free|limited time|ends? (soon|today))\b",
    re.IGNORECASE,
)


# ── run ──────────────────────────────────────────────────────────────────────

def _tool_text(registry: Any, name: str, **kwargs: Any) -> dict[str, Any] | None:
    outcome = registry.call(name, actor="system", **kwargs)
    if not outcome.ok:
        _log.warning("research tool %s failed: %s", name, outcome.error)
        return None
    value = outcome.value
    return value if isinstance(value, dict) else None


def run_job(job: ResearchJob, rctx: ResearchContext) -> list[ResearchFinding]:
    """Execute a job's queries and return deduplicated findings.

    Raises on total failure (no query produced results and no tool at all);
    partial failures are logged and skipped — a flaky endpoint must not kill
    the whole run.
    """
    if not job.queries:
        raise ValueError(f"research job {job.id!r} has no queries")
    seen_urls: set[str] = set()
    findings: list[ResearchFinding] = []
    any_ok = False
    for query in job.queries:
        payload = _tool_text(
            rctx.registry, "web_search", query=query, max_results=job.max_results
        )
        if payload is None:
            continue
        any_ok = True
        for item in payload.get("results", []):
            url = str(item.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            findings.append(
                ResearchFinding(
                    job_id=job.id,
                    title=str(item.get("title", "")).strip(),
                    url=url,
                    snippet=str(item.get("snippet", "")).strip(),
                )
            )
    if not any_ok:
        raise RuntimeError(f"research job {job.id!r}: all web_search calls failed")
    # Depth pass: fetch full text for the top N so assessment sees more than a snippet.
    for finding in findings[: max(0, job.fetch_top)]:
        payload = _tool_text(
            rctx.registry, "web_fetch", url=finding.url, max_chars=6000
        )
        if payload and payload.get("text"):
            finding.detail = str(payload["text"])[:2000]
    _log.info("research job %s: %d findings", job.id, len(findings))
    return findings


# ── assess ───────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _word_overlap(a: str, b: str) -> float:
    wa = set(_norm(a).split())
    wb = set(_norm(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def _already_known(finding: ResearchFinding, memory: Any) -> tuple[bool, str]:
    """Check the owner's memory for prior knowledge of this finding."""
    try:
        result = memory.recall(finding.title, limit=3)
    except Exception as exc:  # noqa: BLE001 - memory is advisory, never fatal
        _log.debug("research memory recall failed: %s", exc)
        return False, ""
    records = getattr(result, "records", result) or []
    for record in records:
        text = str(getattr(record, "text", "") or "")
        if _word_overlap(finding.title, text) >= 0.5:
            return True, text[:80]
    return False, ""


def assess_worth(finding: ResearchFinding, rctx: ResearchContext) -> Assessment:
    """Conservative gate: default to silence unless genuinely useful.

    Hard rejects (score 0): already delivered, empty content, no goal
    relevance. Otherwise: novelty + recency + actionability, thresholded.
    """
    reasons: list[str] = []
    db = rctx.db
    urlh = _url_hash(finding.url)

    # 1. Already delivered? Never send twice.
    row = db.query_one(
        "SELECT delivered_at FROM research_deliveries WHERE url_hash = ?",
        (urlh,),
    )
    if row:
        return Assessment(False, 0.0, ["already delivered"])

    # 2. Content sanity.
    if len(finding.title) < 15 or len(finding.snippet) < 40:
        return Assessment(False, 0.0, ["too thin to judge"])

    text = f"{finding.title} {finding.snippet} {finding.detail}"
    goals = rctx.goal_keywords or DEFAULT_GOAL_KEYWORDS

    # 3. Relevance: the finding must hit at least one goal's keywords.
    # The job's queries already scope the topic; this check ensures the
    # result is about something the owner actually cares about.
    matched_goals: list[str] = []
    keyword_hits = 0
    for goal, keywords in goals.items():
        hits = sum(1 for kw in keywords if kw.lower() in text.lower())
        if hits:
            matched_goals.append(goal)
            keyword_hits += hits
    if not matched_goals:
        return Assessment(False, 0.0, ["no relevance to owner goals"])
    # (untagged jobs are treated as general interest; the threshold below
    # still applies, so they need freshness/actionability to earn delivery.)

    score = 0.0
    score += min(0.45, 0.15 * len(matched_goals) + 0.05 * min(keyword_hits, 6))
    reasons.append(f"relevant to {', '.join(matched_goals)}")

    # 4. Novelty: not in memory, not seen before (or seen long ago, unseen).
    if rctx.memory is not None:
        known, excerpt = _already_known(finding, rctx.memory)
        if known:
            return Assessment(False, 0.0, [f"already known: {excerpt}"])
        reasons.append("not in memory")
    seen = db.query_one(
        "SELECT first_seen, best_score FROM research_seen WHERE url_hash = ?",
        (urlh,),
    )
    now = time.time()
    if seen is None:
        score += 0.25
        reasons.append("never seen before")
    else:
        age_days = (now - float(seen["first_seen"])) / 86400.0
        if age_days < 7:
            # Seen recently and not delivered then: it wasn't worth it then,
            # don't re-surface it now.
            return Assessment(False, 0.0, ["seen recently, not delivered"])
        score += 0.10
        reasons.append("seen before but stale")
    db.execute(
        "INSERT INTO research_seen (url_hash, job_id, title, first_seen, best_score)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(url_hash) DO UPDATE SET best_score = max(best_score, excluded.best_score)",
        (urlh, finding.job_id, finding.title[:200], now, score),
    )

    # 5. Recency and actionability.
    if _RECENCY_RE.search(text):
        score += 0.15
        reasons.append("fresh")
    if _ACTION_RE.search(text):
        score += 0.15
        reasons.append("actionable")

    worth = score >= rctx.worth_threshold
    reasons.append(f"score {score:.2f} vs threshold {rctx.worth_threshold:.2f}")
    return Assessment(worth, score, reasons)


# ── deliver ──────────────────────────────────────────────────────────────────

def _owner_chats(rctx: ResearchContext) -> list[Any]:
    gateway = rctx.gateway
    if gateway is None:
        raise RuntimeError("research deliver: no chat gateway configured")
    chats = list(getattr(gateway, "owner_chats", None) or ())
    if not chats:
        raise RuntimeError("research deliver: no owner chats configured")
    return chats


def _deliveries_today(db: Any) -> int:
    day_start = time.time() - 86400.0
    return int(
        db.scalar(
            "SELECT COUNT(*) FROM research_deliveries WHERE delivered_at >= ?",
            (day_start,),
            default=0,
        )
        or 0
    )


def format_finding(finding: ResearchFinding, assessment: Assessment) -> str:
    why = next(
        (r for r in assessment.reasons if r not in ("fresh", "actionable")
         and not r.startswith("score")),
        "matched your interests",
    )
    lines = [f"🔍 {finding.title}", "", why, "", finding.url]
    text = "\n".join(lines)
    return text[:900]


def deliver(
    finding: ResearchFinding, assessment: Assessment, rctx: ResearchContext
) -> list[str]:
    """Send a worthy finding to the owner's DMs. Fail-fast, never silent.

    Returns the chat keys it was sent to. Raises if delivery is impossible
    (no gateway, no owner chats, daily cap hit) — the scheduler logs the
    failure instead of pretending it worked.
    """
    if not assessment.worth:
        raise ValueError("research deliver called on an unworthy finding")
    if _deliveries_today(rctx.db) >= rctx.daily_delivery_cap:
        raise RuntimeError(
            f"research deliver: daily cap ({rctx.daily_delivery_cap}) reached"
        )
    text = format_finding(finding, assessment)
    sent: list[str] = []
    failures: list[str] = []
    for chat_key in _owner_chats(rctx):
        try:
            from ..social.chat.gateway import ChatRef  # local import: layering
        except Exception:  # noqa: BLE001
            ChatRef = None  # type: ignore[assignment]
        if ChatRef is not None:
            ref = ChatRef.parse(str(chat_key))
            platform = ref.platform
        else:
            platform = str(chat_key).split(":", 1)[0]
        result = rctx.gateway.send(platform, chat_key, text)
        if result.ok:
            sent.append(str(chat_key))
        else:
            failures.append(f"{chat_key}: {result.error}")
    if not sent:
        raise RuntimeError(f"research deliver failed everywhere: {failures}")
    if failures:
        _log.warning("research partial delivery failure: %s", failures)
    now = time.time()
    rctx.db.execute(
        "INSERT OR REPLACE INTO research_deliveries"
        " (url_hash, job_id, title, delivered_at, score, channel)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            _url_hash(finding.url),
            finding.job_id,
            finding.title[:200],
            now,
            assessment.score,
            ",".join(sent),
        ),
    )
    _log.info("research delivered %r to %d chat(s)", finding.title[:60], len(sent))
    return sent


# ── one-shot ─────────────────────────────────────────────────────────────────

@dataclass
class JobReport:
    job_id: str
    findings: int
    delivered: int
    skipped: int
    errors: list[str] = field(default_factory=list)


def execute_job(job: ResearchJob, rctx: ResearchContext) -> JobReport:
    """Run one job end-to-end: research, assess each finding, deliver the
    worthy ones. Never raises for per-finding problems; raises only if the
    job itself could not run at all."""
    report = JobReport(job_id=job.id, findings=0, delivered=0, skipped=0)
    findings = run_job(job, rctx)
    report.findings = len(findings)
    for finding in findings:
        try:
            assessment = assess_worth(finding, rctx)
        except Exception as exc:  # noqa: BLE001 - one bad finding != dead job
            report.errors.append(f"assess {finding.url[:60]}: {exc}")
            report.skipped += 1
            continue
        if not assessment.worth:
            report.skipped += 1
            _log.debug(
                "research skip %r: %s", finding.title[:60], "; ".join(assessment.reasons)
            )
            continue
        try:
            deliver(finding, assessment, rctx)
            report.delivered += 1
        except Exception as exc:  # noqa: BLE001 - deliver failure is real, record it
            report.errors.append(f"deliver {finding.url[:60]}: {exc}")
            report.skipped += 1
    return report
