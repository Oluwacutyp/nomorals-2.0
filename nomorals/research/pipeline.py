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
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

# ── budget ───────────────────────────────────────────────────────────────────
#
# "Spend $0.50 researching this" — budget as the stopping condition. Costs
# are planning estimates, NOT metered billing: the real web_search/web_fetch
# path here is DuckDuckGo (free), and LLM cost varies wildly by provider.
# These numbers exist so a run can be *bounded*, not so it can be invoiced.

#: approximate USD per operation. Planning estimates, not metered billing.
COST_TABLE: dict[str, float] = {
    "web_search": 0.001,   # one web_search call
    "web_fetch": 0.002,    # one web_fetch call (full page read)
    "llm_call": 0.0008,    # one decompose / synthesize / clarify LLM call
}


class ResearchBudget:
    """A spend cap for one research run. Thread-safe.

    ``charge(op)`` debits ``COST_TABLE[op]`` (or an explicit ``amount``)
    and returns True. When a charge would exceed the budget it returns
    False instead — the balance never goes negative and ``exhausted``
    latches True. Exhaustion is a *signal to degrade gracefully* (stop
    searching, synthesize with what you have), never a reason to raise.
    """

    def __init__(self, budget_usd: float) -> None:
        if budget_usd < 0:
            raise ValueError(f"budget must be >= 0, got {budget_usd}")
        self._budget = round(float(budget_usd), 9)
        self._spent = 0.0
        self.exhausted = False
        self._lock = threading.Lock()

    def charge(self, op: str, amount: float | None = None) -> bool:
        """Debit one operation. False when it would exceed the budget."""
        cost = round(float(amount if amount is not None else COST_TABLE[op]), 9)
        with self._lock:
            if self.exhausted:
                return False
            if self._spent + cost > self._budget:
                self.exhausted = True
                return False
            self._spent = round(self._spent + cost, 9)
            return True

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(0.0, round(self._budget - self._spent, 9))

    def spent_usd(self) -> float:
        """Spend, rounded for reports."""
        return round(self.spent, 6)


class _BudgetExhausted(Exception):
    """Internal: an LLM phase hit the budget cap. Caught by the phase's
    existing fallback (templates / extractive brief / heuristic clarify),
    so exhaustion degrades the run instead of killing it."""


def _budgeted_llm(llm_fn: Any, budget: "ResearchBudget | None") -> Any:
    """Wrap an llm_fn so each call charges ``llm_call`` first.

    When the charge fails the wrapper raises ``_BudgetExhausted``, which
    the pipeline's LLM phases already catch (they all degrade to free
    fallbacks on LLM failure).
    """
    if llm_fn is None or budget is None:
        return llm_fn

    def wrapper(prompt: str) -> str:
        if not budget.charge("llm_call"):
            raise _BudgetExhausted("research budget exhausted")
        return llm_fn(prompt)

    return wrapper


def _resolve_budget(job: Any,
                    budget: "ResearchBudget | None") -> "ResearchBudget | None":
    """Explicit budget wins; otherwise build one from the job's budget_usd."""
    if budget is not None:
        return budget
    job_budget = getattr(job, "budget_usd", None)
    if job_budget is not None:
        return ResearchBudget(job_budget)
    return None

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
    max_results: int | None = None  # None → profile default
    fetch_top: int | None = None  # None → profile default
    enabled: bool = True
    budget_usd: float | None = None  # None → unlimited (today's behavior)

    def __post_init__(self) -> None:
        from ..core.profiles import profile_value
        if self.max_results is None:
            self.max_results = int(profile_value("max_results", 6))
        if self.fetch_top is None:
            self.fetch_top = int(profile_value("fetch_top", 2))


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
    daily_delivery_cap: int | None = None  # None → profile default

    def __post_init__(self) -> None:
        from ..core.profiles import profile_value
        if self.daily_delivery_cap is None:
            self.daily_delivery_cap = int(profile_value("daily_delivery_cap", 3))


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


def _research_workers(n_calls: int) -> int:
    """Profile-gated concurrency for research fan-out.

    ``research_workers`` profile key, default 3; termux never exceeds 2 —
    a phone must not melt running parallel fetches.
    """
    from ..core.profiles import get_profile_kind, profile_value
    workers = int(profile_value("research_workers", 3))
    try:
        if get_profile_kind() == "termux":
            workers = min(workers, 2)
    except Exception:  # noqa: BLE001 - profile detection never breaks research
        pass
    return max(1, min(max(0, n_calls), workers))


def _dispatch_calls(registry: Any, calls: list[tuple[str, dict[str, Any]]],
                    *, actor: str = "system") -> list[Any]:
    """Run tool calls in order, concurrently when the registry supports it.

    Prefers ``registry.call_many`` (order-preserving, bounded workers);
    falls back to the serial loop for registries that predate it.
    """
    if not calls:
        return []
    call_many = getattr(registry, "call_many", None)
    if callable(call_many):
        return list(call_many(calls,
                              max_workers=_research_workers(len(calls)),
                              actor=actor))
    return [registry.call(name, actor=actor, **kwargs)
            for name, kwargs in calls]


def _emit_progress(progress: Any, phase: str, item: str) -> None:
    if progress is None:
        return
    try:
        progress(phase, item)
    except Exception:  # noqa: BLE001 - progress must never break a run
        pass


def run_job(job: ResearchJob, rctx: ResearchContext,
            *, progress: Any = None,
            budget: "ResearchBudget | None" = None) -> list[ResearchFinding]:
    """Execute a job's queries and return deduplicated findings.

    Queries run concurrently (profile-gated workers); results are merged
    in query order with URL dedup, exactly like the old serial loop.

    ``budget`` caps spend: each query charges ``web_search`` and each
    depth fetch charges ``web_fetch`` *before* dispatching, in order, so
    exhaustion simply stops issuing new calls — findings gathered so far
    are kept. When the budget is exhausted before any search, returns []
    instead of raising: a budget stop is not a tool failure. ``budget``
    defaults to the job's ``budget_usd``; None means unlimited.

    Raises on total failure (no query produced results and no tool at all);
    partial failures are logged and skipped — a flaky endpoint must not kill
    the whole run. ``progress`` is an optional ``(phase, item)`` callback
    (phases: "search", "fetch").
    """
    if not job.queries:
        raise ValueError(f"research job {job.id!r} has no queries")
    budget = _resolve_budget(job, budget)
    # Budget gate: charge serially, in query order, before dispatching.
    # Only affordable queries are sent; the rest are skipped, never failed.
    queries = job.queries
    if budget is not None:
        queries = []
        for q in job.queries:
            if budget.charge("web_search"):
                queries.append(q)
            else:
                _log.info("research job %s: budget exhausted, stopping "
                          "after %d quer(ies)", job.id, len(queries))
                break
        if not queries:
            return []
    seen_urls: set[str] = set()
    findings: list[ResearchFinding] = []
    any_ok = False
    search_calls = [
        ("web_search", {"query": q, "max_results": job.max_results})
        for q in queries
    ]
    for query, outcome in zip(queries,
                              _dispatch_calls(rctx.registry, search_calls)):
        _emit_progress(progress, "search", query)
        if not outcome.ok:
            _log.warning("research tool web_search failed: %s", outcome.error)
            continue
        payload = outcome.value
        if not isinstance(payload, dict):
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
    from ..core.profiles import profile_value
    _detail_chars = int(profile_value("detail_chars", 2000))
    targets = findings[: max(0, job.fetch_top)]
    if budget is not None:
        affordable: list[ResearchFinding] = []
        for f in targets:
            if budget.charge("web_fetch"):
                affordable.append(f)
            else:
                _log.info("research job %s: budget exhausted, skipping "
                          "remaining fetches", job.id)
                break
        targets = affordable
    fetch_calls = [("web_fetch", {"url": f.url, "max_chars": 6000})
                   for f in targets]
    for finding, outcome in zip(targets,
                                _dispatch_calls(rctx.registry, fetch_calls)):
        _emit_progress(progress, "fetch", finding.url)
        if not outcome.ok:
            _log.warning("research tool web_fetch failed: %s", outcome.error)
            continue
        payload = outcome.value
        if isinstance(payload, dict) and payload.get("text"):
            finding.detail = str(payload["text"])[:_detail_chars]
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
    from ..core.profiles import profile_value
    return text[:int(profile_value("summary_chars", 900))]


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


def execute_job(job: ResearchJob, rctx: ResearchContext,
                *, progress: Any = None,
                budget: "ResearchBudget | None" = None) -> JobReport:
    """Run one job end-to-end: research, assess each finding, deliver the
    worthy ones. Never raises for per-finding problems; raises only if the
    job itself could not run at all. ``progress`` is an optional
    ``(phase, item)`` callback (phases: "search", "fetch", "assess").
    ``budget`` caps research spend (see ``run_job``); delivery itself is
    not charged."""
    report = JobReport(job_id=job.id, findings=0, delivered=0, skipped=0)
    findings = run_job(job, rctx, progress=progress, budget=budget)
    report.findings = len(findings)
    for finding in findings:
        _emit_progress(progress, "assess", finding.url)
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


# ── deep research: decompose → concurrent search → synthesize ────────────────
#
# The scheduled path above runs a job's fixed queries. The deep path takes a
# single question, decomposes it into angled sub-queries, fans them out
# concurrently, and synthesizes one cited brief. Used by the ``research_deep``
# tool and by chat/NL flows that need more than one angle of evidence.

_DECOMPOSE_PROMPT = """Break this research question into {max_queries} or fewer specific web-search queries covering different angles: core facts, practical how-to guidance, criticisms and limitations, and recent developments.

Reply with one query per line. No numbering, no bullets, no commentary — just the queries.

QUESTION: {question}
"""

#: template angles (mirrors ResearchSwarm.angles_for): always available,
#: no LLM needed.
def _template_queries(question: str, max_queries: int) -> list[str]:
    q = question.strip().rstrip("?")
    angles = [
        q,
        f"{q} best practices how to",
        f"{q} criticism problems limitations",
        f"{q} recent developments",
    ]
    return list(dict.fromkeys(a for a in angles if a.strip()))[:max_queries]


def decompose(question: str, llm_fn: Any = None, *,
              max_queries: int = 6) -> list[str]:
    """Split a research question into 1-``max_queries`` search queries.

    LLM-driven when ``llm_fn`` is given; template angles otherwise (or when
    the LLM fails / returns nothing parseable). Never raises for a
    well-formed question — always returns at least one query.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("decompose needs a non-empty question")
    max_queries = max(1, int(max_queries))
    if llm_fn is not None:
        try:
            raw = llm_fn(_DECOMPOSE_PROMPT.format(
                question=question, max_queries=max_queries))
            queries: list[str] = []
            for line in (raw or "").splitlines():
                line = line.strip().lstrip("-•*").strip()
                line = re.sub(r"^\d+[.)]\s*", "", line).strip()
                if len(line) > 3:
                    queries.append(line)
            queries = list(dict.fromkeys(queries))
            if queries:
                return queries[:max_queries]
            _log.debug("decompose: LLM returned nothing parseable, "
                       "using templates")
        except Exception as exc:  # noqa: BLE001 - templates always work
            _log.debug("decompose LLM failed (%s), using templates", exc)
    return _template_queries(question, max_queries)


_SYNTH_PROMPT = """Answer the research question using ONLY the findings below.

Rules:
- Every factual claim must cite its source with [S<n>], using the exact
  labels shown (e.g. [S1], [S2]). Cite every paragraph that states facts.
- If the findings do not support an answer, reply with exactly:
  SYNTHESIS_EMPTY
- Do not use knowledge outside the findings. Be concise and structured.

QUESTION: {question}

FINDINGS:
{numbered}
"""

#: the model cites with [S<n>] (or a title fragment); code assigns the
#: deterministic numbers. Mirrors grounded.py::_number_citations discipline:
#: the model never numbers citations itself.
_CITE_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9 _-]{0,40})\]")

#: returned when nothing supports an answer.
SYNTHESIS_EMPTY = "SYNTHESIS_EMPTY"


def _map_citations(raw: str,
                   findings: list[ResearchFinding]
                   ) -> tuple[str, list[ResearchFinding]]:
    """Map the model's [S<n>] markers to deterministic [1..n] numbers.

    Labels the model invented (not matching any source) are stripped — a
    citation to nothing is worse than no citation. Returns the remapped
    text and the sources actually cited, in first-appearance order.
    """
    label_to_num: dict[str, int] = {}
    used: list[ResearchFinding] = []

    def replace(m: "re.Match[str]") -> str:
        label = m.group(1).strip().upper()
        idx: int | None = None
        if label.startswith("S") and label[1:].isdigit():
            n = int(label[1:])
            if 1 <= n <= len(findings):
                idx = n - 1
        if idx is None and label:
            for i, f in enumerate(findings):
                if label in f.title.upper():
                    idx = i
                    break
        if idx is None:
            return ""  # invented citation — strip it
        key = f"S{idx + 1}"
        if key not in label_to_num:
            label_to_num[key] = len(used) + 1
            used.append(findings[idx])
        return f"[{label_to_num[key]}]"

    return _CITE_RE.sub(replace, raw), used


def _render_synthesis(text: str, used: list[ResearchFinding]) -> str:
    lines = [text.strip(), "", "Sources:"]
    for i, f in enumerate(used, 1):
        lines.append(f"[{i}] {f.title} — {f.url}")
    return "\n".join(lines)


def _extractive_brief(question: str,
                      findings: list[ResearchFinding]) -> str:
    """No-LLM fallback: top findings' titles + snippets, honestly cited.

    No invented prose — just what the sources actually say.
    """
    lines = [question.strip(), ""]
    for i, f in enumerate(findings, 1):
        snippet = re.sub(r"\s+", " ", f.snippet).strip()
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "…"
        lines.append(f"[{i}] {f.title} — {snippet or '(no snippet)'}")
    lines += ["", "Sources:"]
    for i, f in enumerate(findings, 1):
        lines.append(f"[{i}] {f.title} — {f.url}")
    return "\n".join(lines)


def synthesize(question: str, findings: list[ResearchFinding],
               llm_fn: Any = None) -> str:
    """Merge findings into one coherent, cited brief.

    LLM path: answer from the findings with [S<n>] citations, remapped to
    deterministic numbers + a Sources section; invented citations stripped.
    Fallback (no LLM, LLM failure, or no valid citations): extractive brief.
    Returns ``SYNTHESIS_EMPTY`` when there is nothing to synthesize.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("synthesize needs a non-empty question")
    findings = [f for f in (findings or []) if f is not None]
    if not findings:
        return SYNTHESIS_EMPTY
    if llm_fn is not None:
        try:
            numbered = "\n\n".join(
                f"[S{i}] {f.title}\n{re.sub(r'\s+', ' ', f.snippet).strip()[:600]}"
                for i, f in enumerate(findings, 1)
            )
            raw = (llm_fn(_SYNTH_PROMPT.format(question=question,
                                               numbered=numbered)) or "").strip()
            if SYNTHESIS_EMPTY in raw.upper():
                return SYNTHESIS_EMPTY
            text, used = _map_citations(raw, findings)
            if used:
                return _render_synthesis(text, used)
            _log.debug("synthesize: no valid citations survived, "
                       "falling back to extractive brief")
        except Exception as exc:  # noqa: BLE001 - extractive always works
            _log.debug("synthesize LLM failed (%s), extractive fallback", exc)
    return _extractive_brief(question, findings)


_CLARIFY_PROMPT = """You are scoping a research question before any searching happens.

If the question is specific and unambiguous — clear topic, clear angle —
reply with exactly: CLEAR

Otherwise reply with 1-3 sharp clarifying questions, one per line, that
would narrow the scope (angle, time frame, geography, depth). No numbering,
no commentary.

QUESTION: {question}
"""

_STOPWORDS = frozenset(
    "a an the and or but if then else when at by for with about into through "
    "during before after above below to from up down in out on off over under "
    "again further once here there all any both each few more most other some "
    "such no nor not only own same so than too very can will just should now "
    "me my i you your we us is are was were be been being have has had do "
    "does did of as it its this that these those am s t tell".split()
)

_TIME_ANCHORS = frozenset(
    "today yesterday week month year recent latest current now upcoming soon "
    "deadline new just 2024 2025 2026 2027 2028".split()
)

_SCOPE_ANCHORS = frozenset(
    "nigeria nigerian africa african lagos abuja usa us america american uk "
    "europe european global worldwide international local beginner advanced "
    "free paid remote online best top worst vs versus how-to guide tutorial "
    "comparison".split()
)

_GENERIC_CLARIFICATION = (
    "What angle matters most here: a quick overview, a practical how-to, "
    "or the latest developments?"
)


def _content_words(question: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", question.lower())
            if w not in _STOPWORDS and len(w) > 1]


def _has_anchor(question: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    return bool(words & _TIME_ANCHORS or words & _SCOPE_ANCHORS)


def clarify(question: str, llm_fn: Any = None) -> list[str]:
    """Scope-clarifying questions for an ambiguous research question.

    Returns [] when the question is sharp. Heuristic gate first (no LLM
    needed for the obvious cases): with no ``llm_fn``, a question with
    fewer than 4 content words or no time/scope anchor gets one generic
    clarification. With ``llm_fn``, the model decides — it replies CLEAR
    for unambiguous questions. Never raises for a well-formed question.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("clarify needs a non-empty question")
    if llm_fn is None:
        if len(_content_words(question)) < 4 or not _has_anchor(question):
            return [_GENERIC_CLARIFICATION]
        return []
    try:
        raw = (llm_fn(_CLARIFY_PROMPT.format(question=question)) or "")
        lines = [ln.strip().lstrip("-•*").strip() for ln in raw.splitlines()]
        lines = [re.sub(r"^\d+[.)]\s*", "", ln).strip() for ln in lines]
        lines = [ln for ln in lines if ln]
        if lines and lines[0].upper().rstrip(".") == "CLEAR":
            return []
        questions = [ln for ln in lines
                     if ln.upper() != "CLEAR" and len(ln) > 8]
        return questions[:3]
    except Exception as exc:  # noqa: BLE001 - unclear is not fatal
        _log.debug("clarify LLM failed (%s)", exc)
        return []


@dataclass
class DeepReport:
    """Outcome of one deep-research run."""

    question: str
    sub_queries: list[str]
    findings: list[ResearchFinding]
    synthesis: str
    clarifications: list[str]
    needs_clarification: bool = False
    spent_usd: float = 0.0
    budget_exhausted: bool = False


def _router_llm_fn(router: Any) -> Any:
    """Adapt an LLM router to the ``prompt -> text`` shape, or None."""
    if router is None:
        return None

    def llm_fn(prompt: str) -> str:
        resp = router.complete(prompt)
        return resp.text if hasattr(resp, "text") else str(resp)

    return llm_fn


def research_deep(question: str, rctx: ResearchContext, *,
                  llm_fn: Any = None, progress: Any = None,
                  max_queries: int = 6,
                  budget_usd: float | None = None) -> DeepReport:
    """One-shot deep research: clarify → decompose → concurrent search →
    synthesize.

    When the question is ambiguous, returns early with
    ``needs_clarification=True`` and the clarifying questions — the run is
    never burned on a vague query.

    ``budget_usd`` caps total spend: the question is clarified first, then
    ``max_queries`` is cut so the planned searches fit the remaining
    budget (each LLM phase charges ``llm_call``, each search charges
    ``web_search``, each depth fetch charges ``web_fetch``). When the
    budget runs out mid-run the run stops issuing calls and synthesizes
    from whatever findings exist — exhaustion never raises.

    Raises ``ValueError`` on an empty question; ``RuntimeError`` when
    every search fails (via ``run_job``).
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("research_deep needs a non-empty question")
    budget = ResearchBudget(budget_usd) if budget_usd is not None else None

    def _report(**kw: Any) -> DeepReport:
        return DeepReport(
            spent_usd=budget.spent_usd() if budget else 0.0,
            budget_exhausted=budget.exhausted if budget else False,
            **kw,
        )

    # LLM phases charge through the wrapper; exhaustion raises
    # _BudgetExhausted, which each phase already degrades on (templates /
    # heuristic clarify / extractive brief).
    bllm = _budgeted_llm(llm_fn, budget)
    clarifications = clarify(question, llm_fn=bllm)
    if clarifications:
        _log.info("research_deep: needs clarification for %r", question[:80])
        return _report(question=question, sub_queries=[], findings=[],
                       synthesis="", clarifications=clarifications,
                       needs_clarification=True)
    if budget is not None:
        # Cap the fan-out so the planned searches fit the remaining budget.
        # Rough per-query cost: one search + one depth fetch.
        per_query = COST_TABLE["web_search"] + COST_TABLE["web_fetch"]
        affordable = int(budget.remaining / per_query) + 1
        max_queries = max(1, min(int(max_queries), affordable))
        _log.info("research_deep: budget $%.4f remaining, capping at %d "
                  "sub-quer(ies)", budget.remaining, max_queries)
    sub_queries = decompose(question, llm_fn=bllm, max_queries=max_queries)
    job_id = f"deep-{hashlib.sha256(question.encode()).hexdigest()[:12]}"
    job = ResearchJob(id=job_id, topic=question, queries=sub_queries)
    findings = run_job(job, rctx, progress=progress, budget=budget)
    synthesis = synthesize(question, findings, llm_fn=bllm)
    return _report(question=question, sub_queries=sub_queries,
                   findings=findings, synthesis=synthesis,
                   clarifications=[], needs_clarification=False)


# ── tool registration ──────────────────────────────────────────────────────

def register(registry: Any) -> None:
    """Expose ``research_deep`` as a registry tool (NET_OUT).

    Wired via ``nomorals/tools/agents.py`` ``AGENT_TOOL_MODULES`` (the
    ``nomorals.*`` fallback), next to the ``research_swarm`` tool.
    """
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "research_deep",
        description=(
            "deep research: decompose a question into sub-queries, search "
            "them concurrently, and return one cited synthesis. Asks for "
            "clarification first when the question is ambiguous instead of "
            "burning a run. Use when a question needs more than one angle "
            "of evidence."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "question": "str — the research question",
            "max_queries": "int (optional) — max sub-queries to fan out, default 6",
            "budget_usd": "float (optional) — spend cap in USD; the run stops "
                          "issuing calls when exhausted and synthesizes from "
                          "what it has",
        },
    )
    def research_deep_tool(question: str, max_queries: int = 6,
                           budget_usd: float | None = None,
                           **_: Any) -> dict[str, Any]:
        rctx = ResearchContext(
            db=getattr(context, "db", None),
            registry=registry,
            memory=getattr(context, "memory", None),
            gateway=getattr(context, "gateway", None),
        )
        report = research_deep(
            question, rctx,
            llm_fn=_router_llm_fn(getattr(context, "router", None)),
            max_queries=int(max_queries or 6),
            budget_usd=budget_usd,
        )
        return {
            "question": report.question,
            "needs_clarification": report.needs_clarification,
            "clarifications": report.clarifications,
            "sub_queries": report.sub_queries,
            "synthesis": report.synthesis,
            "spent_usd": report.spent_usd,
            "budget_exhausted": report.budget_exhausted,
            "findings": [
                {"title": f.title, "url": f.url, "snippet": f.snippet,
                 "detail": f.detail}
                for f in report.findings
            ],
        }
