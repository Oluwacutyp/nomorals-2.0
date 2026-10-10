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
# Cost constants live in .costs so that nomorals.llm.router (which needs
# the flat per-call USD cost) can import them without pulling in this
# module's brain dependency — the old direct import created a
# research.pipeline <-> llm.router cycle (research first => ImportError).
from .costs import COST_TABLE


class ResearchBudget:
    """A spend cap for one research run. Thread-safe.

    ``charge(op)`` debits ``COST_TABLE[op]`` (or an explicit ``amount``)
    and returns True. When a charge would exceed the budget it returns
    False instead — the balance never goes negative and ``exhausted``
    latches True. Exhaustion is a *signal to degrade gracefully* (stop
    searching, synthesize with what you have), never a reason to raise.
    """

    def __init__(self, budget_usd: float,
                 deadline_s: float | None = None) -> None:
        if budget_usd < 0:
            raise ValueError(f"budget must be >= 0, got {budget_usd}")
        self._budget = round(float(budget_usd), 9)
        self._spent = 0.0
        self.exhausted = False
        self._lock = threading.Lock()
        # Optional wall-clock cap: a run that spends forever researching a
        # question is as broken as one that spends infinite money.
        self._deadline = (time.time() + float(deadline_s)
                          if deadline_s is not None else None)

    def time_exceeded(self) -> bool:
        """True when the wall-clock deadline (if any) has passed."""
        with self._lock:
            if self._deadline is None:
                return False
            if time.time() >= self._deadline:
                self.exhausted = True
                return True
            return False

    @property
    def time_left_s(self) -> float | None:
        with self._lock:
            if self._deadline is None:
                return None
            return max(0.0, self._deadline - time.time())

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

    def refund(self, amount: float) -> float:
        """Credit back an over-charged estimate.  Never lets spend go
        negative.  Used by the measured-cost true-up (build-map #18):
        the pre-call charge is a planning estimate; when the router
        meters the real cost and it came in lower, the difference is
        refunded so the ledger reflects reality."""
        credit = round(float(amount), 9)
        with self._lock:
            self._spent = round(max(0.0, self._spent - credit), 9)
            return self._spent


class _BudgetExhausted(Exception):
    """Internal: an LLM phase hit the budget cap. Caught by the phase's
    existing fallback (templates / extractive brief / heuristic clarify),
    so exhaustion degrades the run instead of killing it."""


def _true_up_llm_cost(llm_fn: Any, budget: "ResearchBudget") -> None:
    """Replace the flat planning estimate with the metered cost, when known.

    The pre-call ``charge("llm_call")`` is the *guard*: it stops a run from
    starting work it can't afford.  When the router-measured cost is
    available on ``llm_fn.last_response`` (set by :func:`_router_llm_fn`),
    true-up the delta so the budget ledger reflects reality instead of the
    estimate — charged when the call cost more, refunded when it cost less.
    Never raises and never double-counts.
    """
    try:
        from ..llm.router import estimate_cost
        resp = getattr(llm_fn, "last_response", None)
        usage = getattr(resp, "usage", None) if resp is not None else None
        if usage is None:
            return
        prompt_t = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_t = int(getattr(usage, "completion_tokens", 0) or 0)
        if not (prompt_t or completion_t):
            return
        measured = estimate_cost(
            getattr(resp, "provider", "") or "",
            getattr(resp, "model", "") or "",
            prompt_t,
            completion_t,
        )
        delta = round(measured - COST_TABLE["llm_call"], 9)
        if delta > 0:
            # Already have the result; exhaustion latches for later calls.
            budget.charge("llm_call", amount=delta)
        elif delta < 0:
            budget.refund(-delta)
    except Exception:  # noqa: BLE001 — ledger accuracy never breaks research
        _log.debug("budget true-up failed", exc_info=True)


def _budgeted_llm(llm_fn: Any, budget: "ResearchBudget | None") -> Any:
    """Wrap an llm_fn so each call charges ``llm_call`` first.

    When the charge fails the wrapper raises ``_BudgetExhausted``, which
    the pipeline's LLM phases already catch (they all degrade to free
    fallbacks on LLM failure).  After a successful call the flat estimate
    is trued-up against the router-metered cost when available.
    """
    if llm_fn is None or budget is None:
        return llm_fn

    def wrapper(prompt: str) -> str:
        if not budget.charge("llm_call"):
            raise _BudgetExhausted("research budget exhausted")
        result = llm_fn(prompt)
        _true_up_llm_cost(llm_fn, budget)
        return result

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
    digest: bool = False  # True → worthy findings batched into one digest

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

#: independence signals (beyondseo reputation-rubric pattern): a press
#: release or sponsored placement is not an independent editorial source.
_PRESS_RE = re.compile(
    r"\b(press release|sponsored( content)?|paid partnership|advertorial|"
    r"promoted content|advertisement)\b",
    re.IGNORECASE,
)
#: primary-source signals: docs, official publications, code, studies.
_PRIMARY_RE = re.compile(
    r"\b(official|documentation|white ?paper|peer[- ]reviewed|"
    r"published study|changelog|api reference)\b",
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


def _browser_fetch_fallback(registry: Any, url: str, max_chars: int) -> str:
    """Fetch a page through the rendered Chromium engine.

    Used when the plain ``web_fetch`` can't read a page (JS-rendered,
    bot-gated). Returns the page text, or "" when the rendered engine
    isn't available. Never raises.
    """
    try:
        outcome = registry.call("browser", actor="system", action="open",
                                url=url, engine="rendered",
                                session="research")
        if not outcome.ok:
            return ""
        outcome = registry.call("browser", actor="system", action="wait",
                                engine="rendered", session="research",
                                timeout=10000)
        # wait failing is fine — the page may already be settled
        outcome = registry.call("browser", actor="system", action="text",
                                engine="rendered", session="research",
                                max_chars=max_chars)
        if not outcome.ok:
            return ""
        value = outcome.value
        if isinstance(value, dict):
            return str(value.get("text", ""))[:max_chars]
        return ""
    except Exception as exc:  # noqa: BLE001 - fallback must never break research
        _log.debug("browser fetch fallback failed for %s: %s", url, exc)
        return ""
    finally:
        try:
            registry.call("browser", actor="system", action="close",
                          engine="rendered", session="research")
        except Exception:  # noqa: BLE001
            pass


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


def _learned_domain_score(db: Any, url: str) -> float | None:
    """Rolling worth-rate for a URL's domain, learned from evidence.

    Reads the ``source_quality`` table (written by the research organ on
    every assessment). Returns None when the table is missing or the
    domain is unseen — unseen domains are never penalized, only
    evidence-demonstrated noise is. Never raises.
    """
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower().lstrip("www.")
        if not domain:
            return None
        row = db.query_one(
            "SELECT runs, worth_runs FROM source_quality WHERE domain = ?",
            (domain,))
        if not row or not row["runs"]:
            return None
        return float(row["worth_runs"]) / float(row["runs"])
    except Exception:  # noqa: BLE001 - learned ranking is advisory
        return None


def _rerank_by_learned_quality(findings: list[ResearchFinding],
                               db: Any) -> list[ResearchFinding]:
    """Demote findings from evidence-demonstrated weak domains.

    Domains with a learned worth-rate below 0.25 over ≥3 runs go last;
    everything else keeps its relative order. Never filters — a weak
    domain can still deliver a worthy finding, it just doesn't get the
    depth-fetch slots first. Stable sort, so equal scores keep position.
    """
    def sort_key(f: ResearchFinding) -> tuple[int, float]:
        score = _learned_domain_score(db, f.url)
        weak = (score is not None and score < 0.25
                and _domain_runs(db, f.url) >= 3)
        return (1 if weak else 0, -(score or 0.0))

    return sorted(findings, key=sort_key)


def _domain_runs(db: Any, url: str) -> int:
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower().lstrip("www.")
        row = db.query_one(
            "SELECT runs FROM source_quality WHERE domain = ?", (domain,))
        return int(row["runs"]) if row else 0
    except Exception:  # noqa: BLE001
        return 0


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
    # Learned quality: domains that have consistently produced noise get
    # demoted (evidence-based, never a hardcoded blocklist). Rerank before
    # the depth pass so the fetch slots go to the best sources first.
    findings = _rerank_by_learned_quality(findings, rctx.db)
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
            # Fallback: the rendered Chromium engine for JS-heavy pages
            # that the plain fetcher can't read. She's got a browser —
            # use it, don't just skip the source.
            detail = _browser_fetch_fallback(rctx.registry, finding.url,
                                             _detail_chars)
            if detail:
                finding.detail = detail
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

    # 6. Independence: earned-media evidence outranks placed content.
    # A press release or sponsored placement about the topic is not an
    # independent source — discount it; primary sources get a small bonus.
    if _PRESS_RE.search(text):
        score -= 0.15
        reasons.append("placed/press content — independence discount")
    elif _PRIMARY_RE.search(text) or "github.com" in finding.url.lower():
        score += 0.05
        reasons.append("primary-source signals")

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


def _send_to_chats(rctx: ResearchContext,
                   text: str) -> tuple[list[str], list[str]]:
    """Send ``text`` to every owner chat. Returns (sent, failures)."""
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
    return sent, failures


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


#: delivery themes for format_finding. "classic" is the historical output;
#: "card" is the rich default for new callers; "brief" is TL;DR-first;
#: "verbose" is the full audit-style card.
_FINDING_STYLES = ("classic", "card", "brief", "verbose")


def _score_bar(score: float, width: int = 10) -> str:
    filled = max(0, min(width, int(round(score * width))))
    return "▰" * filled + "▱" * (width - filled)


def format_finding(finding: ResearchFinding, assessment: Assessment,
                   style: str = "classic") -> str:
    """Render a finding for delivery. Styles:

    - ``classic`` — the historical plain template (default, unchanged).
    - ``card`` — rich markdown card: headline, why-it-matters, score bar,
      URL. The god-tier default for chat delivery.
    - ``brief`` — TL;DR-first: one line + URL, for tight digests.
    - ``verbose`` — full audit card: snippet excerpt, every reason, domain
      quality note, score breakdown.
    """
    from ..core.profiles import profile_value
    limit = int(profile_value("summary_chars", 900))
    why = next(
        (r for r in assessment.reasons if r not in ("fresh", "actionable")
         and not r.startswith("score")),
        "matched your interests",
    )
    if style == "brief":
        return f"🔍 {finding.title}\n{why}\n{finding.url}"[:limit]
    if style == "card":
        tags = " ".join(f"#{t}" for t in ("fresh", "actionable")
                        if t in assessment.reasons)
        lines = [
            f"🔍 *{finding.title}*",
            "",
            f"_{why}_",
            "",
            f"{_score_bar(assessment.score)} `{assessment.score:.0%}`"
            + (f"  {tags}" if tags else ""),
            "",
            finding.url,
        ]
        return "\n".join(lines)[:limit]
    if style == "verbose":
        snippet = re.sub(r"\s+", " ", finding.snippet).strip()
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "…"
        lines = [
            f"🔍 {finding.title}",
            "",
            f"Why: {why}",
            f"Excerpt: {snippet}" if snippet else "Excerpt: (none)",
            "",
            "Signals:",
        ]
        lines += [f"  • {r}" for r in assessment.reasons]
        lines += ["", f"Score: {assessment.score:.2f}  {finding.url}"]
        return "\n".join(lines)[: limit * 2]
    if style != "classic":
        raise ValueError(f"unknown finding style {style!r}; "
                         f"expected one of {_FINDING_STYLES}")
    lines = [f"🔍 {finding.title}", "", why, "", finding.url]
    text = "\n".join(lines)
    return text[:limit]


def deliver_digest(pairs: list[tuple[ResearchFinding, Assessment]],
                   rctx: ResearchContext,
                   style: str = "card",
                   title: str | None = None) -> list[str]:
    """Batch several worthy findings into ONE delivered message.

    The scheduled path delivers per-finding; a watch that surfaces five
    worthy items should not send five interruptions. Every finding still
    gets its own ``research_deliveries`` row (never re-sent), and the
    daily cap applies to the whole digest. Raises on unworthy findings,
    missing gateway/chats, or a blown cap — fail-fast like ``deliver``.
    Returns the chat keys it was sent to.
    """
    pairs = list(pairs or [])
    if not pairs:
        raise ValueError("deliver_digest needs at least one finding")
    for finding, assessment in pairs:
        if not assessment.worth:
            raise ValueError(
                f"deliver_digest: unworthy finding {finding.url[:60]!r}")
    remaining = rctx.daily_delivery_cap - _deliveries_today(rctx.db)
    if len(pairs) > remaining:
        raise RuntimeError(
            f"research deliver_digest: daily cap ({rctx.daily_delivery_cap}) "
            f"reached")
    from ..core.profiles import profile_value
    limit = int(profile_value("summary_chars", 900)) * max(1, len(pairs))
    head = title or "🔍 Research digest"
    parts = [head, ""]
    for i, (finding, assessment) in enumerate(pairs, 1):
        body = format_finding(finding, assessment, style=style)
        parts.append(f"— {i} —\n{body}")
    text = "\n\n".join(parts)[:limit]
    sent, failures = _send_to_chats(rctx, text)
    if not sent:
        raise RuntimeError(
            f"research deliver_digest failed everywhere: {failures}")
    if failures:
        _log.warning("research partial digest failure: %s", failures)
    now = time.time()
    for finding, assessment in pairs:
        rctx.db.execute(
            "INSERT OR REPLACE INTO research_deliveries"
            " (url_hash, job_id, title, delivered_at, score, channel)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (_url_hash(finding.url), finding.job_id, finding.title[:200],
             now, assessment.score, ",".join(sent)),
        )
    _log.info("research digest: %d finding(s) to %d chat(s)",
              len(pairs), len(sent))
    return sent


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
    sent, failures = _send_to_chats(rctx, text)
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
    digest_pairs: list[tuple[ResearchFinding, Assessment]] = []
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
        if job.digest:
            # Digest mode: batch every worthy finding into one message.
            digest_pairs.append((finding, assessment))
            continue
        try:
            deliver(finding, assessment, rctx)
            report.delivered += 1
        except Exception as exc:  # noqa: BLE001 - deliver failure is real, record it
            report.errors.append(f"deliver {finding.url[:60]}: {exc}")
            report.skipped += 1
    if digest_pairs:
        _emit_progress(progress, "deliver", f"digest of {len(digest_pairs)}")
        try:
            deliver_digest(digest_pairs, rctx, title=f"🔍 {job.topic}")
            report.delivered += len(digest_pairs)
        except Exception as exc:  # noqa: BLE001 - digest failure is real
            report.errors.append(f"deliver_digest: {exc}")
            report.skipped += len(digest_pairs)
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


def _resolve_label(label: str,
                   findings: list[ResearchFinding]) -> int | None:
    """Resolve a citation label (``S3`` or a title fragment) to a finding
    index, or None when the label matches no source (invented label)."""
    label = (label or "").strip().upper()
    if label.startswith("S") and label[1:].isdigit():
        n = int(label[1:])
        if 1 <= n <= len(findings):
            return n - 1
    if label:
        for i, f in enumerate(findings):
            if label in f.title.upper():
                return i
    return None


def _verify_cited_sentences(raw: str,
                            findings: list[ResearchFinding]) -> str:
    """Back-compat wrapper — see ``_verify_cited_sentences_stats``."""
    text, _, _ = _verify_cited_sentences_stats(raw, findings)
    return text


def _verify_cited_sentences_stats(
        raw: str, findings: list[ResearchFinding]) -> tuple[str, int, int]:
    """Strip sentences whose citations don't verify against the sources.

    Closes the loop on citations: a valid-looking ``[S<n>]`` marker is
    not enough — the sentence's factual content must actually appear in
    the cited source (see ``citations.sentence_supported``). Sentences
    with no citation markers are left alone (uncited prose, not smuggled
    claims). Sentences citing unknown labels are stripped too. Never
    raises; on any internal error the text passes through unchanged.

    Returns ``(cleaned_text, sentences_checked, sentences_stripped)``.
    """
    try:
        from .citations import sentence_supported
    except Exception:  # noqa: BLE001 - verification is advisory
        return raw, 0, 0
    try:
        texts = [(f.snippet or "") + "\n" + (f.detail or "") for f in findings]
        kept: list[str] = []
        checked = stripped = 0
        for sent in re.split(r"(?<=[.!?])\s+", (raw or "").strip()):
            if not sent.strip():
                continue
            labels = [m.group(1) for m in _CITE_RE.finditer(sent)]
            if not labels:
                kept.append(sent)
                continue
            checked += 1
            idxs = {i for lab in labels
                    if (i := _resolve_label(lab, findings)) is not None}
            if not idxs:
                stripped += 1
                _log.debug("synthesize: stripped sentence with unverifiable "
                           "citation labels: %r", sent[:80])
                continue
            body = _CITE_RE.sub("", sent)
            if any(sentence_supported(body, texts[i]) for i in idxs):
                kept.append(sent)
            else:
                stripped += 1
                _log.info("synthesize: stripped sentence whose citation did "
                          "not verify: %r", sent[:100])
        return " ".join(kept), checked, stripped
    except Exception as exc:  # noqa: BLE001 - never break synthesis
        _log.debug("synthesize: citation verification failed (%s)", exc)
        return raw, 0, 0


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
        idx = _resolve_label(m.group(1), findings)
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
    text, _ = synthesize_with_stats(question, findings, llm_fn)
    return text


def synthesize_with_stats(question: str, findings: list[ResearchFinding],
                          llm_fn: Any = None
                          ) -> tuple[str, dict[str, Any]]:
    """``synthesize`` plus the evidence-verification audit.

    Returns ``(text, stats)`` where stats carries
    ``sentences_checked``, ``sentences_stripped``, ``sources_cited`` and
    ``corroborated_sources`` (sources confirmed by ≥2 independent
    domains — the cross-validation rule from the deep-research
    best-practices literature).
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("synthesize needs a non-empty question")
    findings = [f for f in (findings or []) if f is not None]
    if not findings:
        return SYNTHESIS_EMPTY, {"sentences_checked": 0,
                                 "sentences_stripped": 0,
                                 "sources_cited": 0,
                                 "corroborated_sources": 0}
    stats: dict[str, Any] = {"sentences_checked": 0, "sentences_stripped": 0,
                             "sources_cited": 0, "corroborated_sources": 0}
    if llm_fn is not None:
        try:
            numbered = "\n\n".join(
                f"[S{i}] {f.title}\n{re.sub(r'\s+', ' ', f.snippet).strip()[:600]}"
                for i, f in enumerate(findings, 1)
            )
            raw = (llm_fn(_SYNTH_PROMPT.format(question=question,
                                               numbered=numbered)) or "").strip()
            if SYNTHESIS_EMPTY in raw.upper():
                return SYNTHESIS_EMPTY, stats
            # Claim-level check: strip sentences whose citations don't
            # verify against the source texts before deterministic mapping.
            raw, checked, stripped = _verify_cited_sentences_stats(
                raw, findings)
            stats["sentences_checked"] = checked
            stats["sentences_stripped"] = stripped
            text, used = _map_citations(raw, findings)
            if used:
                stats["sources_cited"] = len(used)
                stats["corroborated_sources"] = _corroborated_count(
                    findings, used)
                return _render_synthesis(text, used), stats
            _log.debug("synthesize: no valid citations survived, "
                       "falling back to extractive brief")
        except Exception as exc:  # noqa: BLE001 - extractive always works
            _log.debug("synthesize LLM failed (%s), extractive fallback", exc)
    text = _extractive_brief(question, findings)
    stats["sources_cited"] = len(findings)
    stats["corroborated_sources"] = _corroborated_count(findings, findings)
    return text, stats


def _corroborated_count(findings: list[ResearchFinding],
                        used: list[ResearchFinding]) -> int:
    """How many used sources are corroborated by ≥2 independent domains.

    Independent corroboration is the deep-research best-practice rule:
    a key finding should be confirmed by at least two sources that don't
    share a publisher. Counts sources whose topic cluster (content-word
    overlap) contains findings from at least two distinct domains.
    """
    try:
        from .citations import corroboration_map
        corrob = corroboration_map(findings)
        used_urls = {f.url for f in used}
        n = 0
        for f in used:
            idx = next((i for i, x in enumerate(findings)
                        if x.url == f.url), None)
            if idx is not None and corrob.get(idx):
                n += 1
        return n
    except Exception:  # noqa: BLE001 - corroboration is advisory
        return 0


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
    citations: list[dict] = field(default_factory=list)  # audit trail
    learnings: list[str] = field(default_factory=list)
    running_summary: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
    conflicts: list[dict] = field(default_factory=list)
    depth: int = 3

    def to_markdown(self) -> str:
        """Structured report: TL;DR → learnings → synthesis → evidence.

        The export shape the open deep-research tools converge on: an
        executive summary up front, then the full cited brief, then the
        audit trail. ``verification`` and ``conflicts`` make the
        evidence quality visible instead of implied.
        """
        tldr = ""
        if self.learnings:
            tldr = "\n".join(f"- {ln}" for ln in self.learnings[:8])
        elif self.synthesis and self.synthesis != SYNTHESIS_EMPTY:
            first = self.synthesis.split("\n\n")[0].strip()
            tldr = first[:600]
        lines = [f"# Research: {self.question}", ""]
        lines += ["## TL;DR", "", tldr or "_No findings._", ""]
        if self.running_summary:
            lines += ["## Working summary", "", self.running_summary, ""]
        lines += ["## Brief", "",
                  self.synthesis or "_No synthesis produced._", ""]
        if self.conflicts:
            lines += ["## ⚠️ Conflicting sources", ""]
            for c in self.conflicts:
                lines.append(
                    f"- {c.get('topic', 'conflict')}: "
                    f"{c.get('value_a', '?')} ({c.get('url_a', '')}) vs "
                    f"{c.get('value_b', '?')} ({c.get('url_b', '')})")
            lines.append("")
        v = self.verification or {}
        if v:
            lines += ["## Evidence check", "",
                      f"- sentences citation-checked: "
                      f"{v.get('sentences_checked', 0)}",
                      f"- sentences stripped (unverifiable): "
                      f"{v.get('sentences_stripped', 0)}",
                      f"- sources cited: {v.get('sources_cited', 0)}",
                      f"- corroborated by ≥2 independent domains: "
                      f"{v.get('corroborated_sources', 0)}", ""]
        lines += [f"_Depth {self.depth} · {len(self.findings)} findings · "
                  f"${self.spent_usd:.4f} spent"
                  + (" · budget exhausted" if self.budget_exhausted else "")
                  + "_"]
        return "\n".join(lines)


def _citation_audit_trail(findings: list[ResearchFinding]) -> list[dict]:
    """Build the claim-level evidence audit trail for a run's findings.

    Registers each finding's source (content hash + access time) in a
    ``CitationManager`` and returns the audit trail, so ``DeepReport``
    carries its evidence. Never raises — an empty trail beats a failed
    run.
    """
    try:
        from .citations import CitationManager
        mgr = CitationManager()
        for f in findings or []:
            mgr.register_source(
                f.url, f.title,
                (f.snippet or "") + "\n" + (f.detail or ""),
                accessed_ts=f.fetched_at)
        return mgr.audit_trail()
    except Exception as exc:  # noqa: BLE001 - evidence is advisory
        _log.debug("research_deep: citation audit trail failed (%s)", exc)
        return []


def _router_llm_fn(router: Any) -> Any:
    """Adapt an LLM router to the ``prompt -> text`` shape, or None.

    Uses the minimal router interface — ``complete(prompt)`` returning a
    response with ``.text`` — so strict fakes and the full LLMRouter both
    work. The full response is stashed on ``llm_fn.last_response`` so the
    budget wrapper can true-up the flat planning estimate against the
    router-metered cost (build-map #18).
    """
    if router is None:
        return None

    def llm_fn(prompt: str) -> str:
        resp = router.complete(prompt)
        llm_fn.last_response = resp
        return resp.text if hasattr(resp, "text") else str(resp)

    llm_fn.last_response = None
    return llm_fn


def _conflict_dicts(findings: list[ResearchFinding]) -> list[dict]:
    """Heuristic conflict detection for a deep-research run.

    Uses ``citations.find_conflicts`` (same topic, different numbers from
    different domains) and flattens to plain dicts for ``DeepReport``.
    Never raises — conflicts are advisory, the run is not.
    """
    try:
        from .citations import find_conflicts
        return [
            {"topic": c.topic, "value_a": c.value_a, "url_a": c.url_a,
             "value_b": c.value_b, "url_b": c.url_b}
            for c in find_conflicts(findings)
        ]
    except Exception as exc:  # noqa: BLE001
        _log.debug("research_deep: conflict detection failed (%s)", exc)
        return []


#: max refinement iterations in research_deep (initial pass + follow-ups).
_MAX_REFINEMENTS = 3

_REFINE_PROMPT = """You are reviewing research findings for gaps. Read the findings below and decide if more searching is needed.

Reply with either:
- DONE — the findings fully answer the question, no gaps
- Or 1-3 follow-up search queries (one per line) that would fill specific gaps, resolve contradictions, or add missing angles

Be specific. Reference what's missing, not what's already covered.

QUESTION: {question}

FINDINGS:
{numbered}
"""


def _reasoning_pass(question: str, findings: list[ResearchFinding],
                    llm_fn: Any = None) -> list[str]:
    """Identify gaps in findings, return follow-up queries or [] if done.

    This is the iterative refinement loop from deep-research systems:
    read sources → identify contradictions/gaps → decide if more searches
    are needed. Returns [] when the findings are sufficient.
    """
    if not findings or llm_fn is None:
        return []
    try:
        numbered = "\n\n".join(
            f"[S{i}] {f.title}\n{re.sub(r'\\s+', ' ', f.snippet).strip()[:400]}"
            for i, f in enumerate(findings[:20], 1)
        )
        raw = (llm_fn(_REFINE_PROMPT.format(question=question,
                                            numbered=numbered)) or "").strip()
        if "DONE" in raw.upper()[:20]:
            return []
        queries = []
        for line in raw.splitlines():
            line = line.strip().lstrip("-•*").strip()
            line = re.sub(r"^\\d+[.)]\\s*", "", line).strip()
            if len(line) > 5 and "DONE" not in line.upper():
                queries.append(line)
        return queries[:3]
    except Exception as exc:  # noqa: BLE001 - no refinement on failure
        _log.debug("reasoning pass failed (%s), no follow-ups", exc)
        return []


_REFINE_LEARN_PROMPT = """You are distilling research findings into durable learnings.

QUESTION: {question}

FINDINGS:
{numbered}

Reply with exactly two sections:

LEARNINGS:
- one crisp factual learning per bullet, each ending with its [S<n>] citation
  (only learnings the findings actually support — no outside knowledge)

FOLLOW-UPS:
- one follow-up search query per bullet for real gaps, contradictions, or
  missing angles
- or the single line NONE when the findings fully answer the question
"""


@dataclass
class Refinement:
    """One reasoning pass: distilled learnings + follow-up directions."""

    learnings: list[str] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    done: bool = False


def _refinement_pass(question: str, findings: list[ResearchFinding],
                     llm_fn: Any = None) -> Refinement:
    """One combined reasoning pass (Open Deep Research pattern).

    Unlike ``_reasoning_pass`` (queries only), this also distills
    **learnings** — compressed facts with citations — so later iterations
    build on insight instead of re-reading raw snippets. Offline
    fallback: learnings from the top finding titles, no follow-ups.
    Never raises.
    """
    if not findings:
        return Refinement(done=True)
    if llm_fn is None:
        learnings = [f"{f.title} [S{i}]"
                     for i, f in enumerate(findings[:8], 1) if f.title]
        return Refinement(learnings=learnings, done=True)
    try:
        numbered = "\n\n".join(
            f"[S{i}] {f.title}\n{re.sub(r'\\s+', ' ', f.snippet).strip()[:400]}"
            for i, f in enumerate(findings[:20], 1)
        )
        raw = (llm_fn(_REFINE_LEARN_PROMPT.format(
            question=question, numbered=numbered)) or "").strip()
        learnings: list[str] = []
        follow_ups: list[str] = []
        section = None
        for line in raw.splitlines():
            s = line.strip()
            up = s.upper().rstrip(":")
            if up == "LEARNINGS":
                section = "learnings"
                continue
            if up == "FOLLOW-UPS":
                section = "follow_ups"
                continue
            s = s.lstrip("-•*").strip()
            s = re.sub(r"^\\d+[.)]\\s*", "", s).strip()
            if not s or len(s) < 8:
                continue
            if section == "learnings":
                learnings.append(s)
            elif section == "follow_ups":
                if s.upper().startswith("NONE") or "DONE" in s.upper():
                    continue
                follow_ups.append(s)
        return Refinement(learnings=learnings[:10], follow_ups=follow_ups[:3],
                          done=not follow_ups)
    except Exception as exc:  # noqa: BLE001 - refinement is advisory
        _log.debug("refinement pass failed (%s)", exc)
        return Refinement(done=True)


def _update_working_summary(summary: str, learnings: list[str],
                            llm_fn: Any = None) -> str:
    """Fold new learnings into the running summary (IterDRAG pattern).

    The working summary is the compressed memory of everything the run
    has learned so far — later refinement passes read this instead of
    the full finding list. LLM path compresses; template path appends
    and caps. Never raises.
    """
    if not learnings:
        return summary
    addition = "\n".join(f"- {ln}" for ln in learnings)
    merged = f"{summary}\n{addition}".strip() if summary else addition
    if llm_fn is not None:
        try:
            prompt = ("Compress these research notes into a tight running "
                      "summary (<= 1200 chars). Keep every distinct fact and "
                      "its [S<n>] citation.\n\nNOTES:\n" + merged[:6000])
            compressed = (llm_fn(prompt) or "").strip()
            if compressed:
                return compressed[:2000]
        except Exception as exc:  # noqa: BLE001
            _log.debug("working summary compress failed (%s)", exc)
    # Template fallback: append, keep the tail — newest insight survives.
    return merged[-4000:]


def research_deep(question: str, rctx: ResearchContext, *,
                  llm_fn: Any = None, progress: Any = None,
                  max_queries: int = 6,
                  budget_usd: float | None = None,
                  depth: int = 3,
                  time_budget_s: float | None = None) -> DeepReport:
    """One-shot deep research: clarify → decompose → concurrent search →
    refine → synthesize.

    When the question is ambiguous, returns early with
    ``needs_clarification=True`` and the clarifying questions — the run is
    never burned on a vague query.

    After the initial search, a reasoning pass distills **learnings** and
    reviews the findings for gaps and contradictions, generating follow-up
    queries. This repeats for ``depth - 1`` refinement rounds (Open Deep
    Research pattern, hard-capped by ``_MAX_REFINEMENTS``) — the iterative
    loop that separates deep research from single-pass search. Each round
    folds its learnings into a **running summary** (IterDRAG pattern) so
    later rounds reason over compressed insight, not raw snippets.

    ``budget_usd`` caps total spend and ``time_budget_s`` caps wall-clock
    time: the question is clarified first, then ``max_queries`` is cut so
    the planned searches fit the remaining budget (each LLM phase charges
    ``llm_call``, each search charges ``web_search``, each depth fetch
    charges ``web_fetch``). When either budget runs out mid-run the run
    stops issuing calls and synthesizes from whatever findings exist —
    exhaustion never raises.

    Raises ``ValueError`` on an empty question; ``RuntimeError`` when
    every search fails (via ``run_job``).
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("research_deep needs a non-empty question")
    depth = max(1, min(int(depth or 3), _MAX_REFINEMENTS))
    budget = (ResearchBudget(budget_usd, deadline_s=time_budget_s)
              if budget_usd is not None else None)

    def _report(**kw: Any) -> DeepReport:
        return DeepReport(
            spent_usd=budget.spent_usd() if budget else 0.0,
            budget_exhausted=budget.exhausted if budget else False,
            depth=depth,
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
    # Iterative refinement: learnings + follow-up directions → more search.
    all_queries = list(sub_queries)
    all_learnings: list[str] = []
    working_summary = ""
    refinement = _refinement_pass(question, findings, llm_fn=bllm)
    all_learnings.extend(refinement.learnings)
    working_summary = _update_working_summary(
        working_summary, refinement.learnings, llm_fn=bllm)
    for iteration in range(depth - 1):
        if budget is not None and (budget.exhausted or budget.time_exceeded()):
            _log.info("research_deep: budget exhausted, stopping refinement")
            break
        _emit_progress(progress, "refine",
                       f"iteration {iteration + 2}: "
                       f"{len(refinement.learnings)} learning(s)")
        if refinement.done or not refinement.follow_ups:
            _log.info("research_deep: findings sufficient after %d iteration(s)",
                      iteration + 1)
            break
        _log.info("research_deep: refinement %d, %d follow-up quer(ies)",
                  iteration + 2, len(refinement.follow_ups))
        all_queries.extend(refinement.follow_ups)
        follow_job = ResearchJob(id=f"{job_id}-r{iteration + 1}",
                                topic=question, queries=refinement.follow_ups)
        new_findings = run_job(follow_job, rctx, progress=progress,
                               budget=budget)
        # Deduplicate by URL against existing findings
        seen = {f.url for f in findings}
        for f in new_findings:
            if f.url not in seen:
                seen.add(f.url)
                findings.append(f)
        refinement = _refinement_pass(question, findings, llm_fn=bllm)
        all_learnings.extend(refinement.learnings)
        working_summary = _update_working_summary(
            working_summary, refinement.learnings, llm_fn=bllm)
    synthesis, verification = synthesize_with_stats(question, findings,
                                                    llm_fn=bllm)
    conflicts = _conflict_dicts(findings)
    citations = _citation_audit_trail(findings)
    return _report(question=question, sub_queries=all_queries,
                   findings=findings, synthesis=synthesis,
                   clarifications=[], needs_clarification=False,
                   citations=citations, learnings=all_learnings,
                   running_summary=working_summary,
                   verification=verification, conflicts=conflicts)


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
            "depth": "int (optional) — refinement rounds, 1-3, default 3. "
                     "Each round distills learnings and chases follow-up "
                     "directions; 1 = single pass, no refinement",
            "time_budget_s": "float (optional) — wall-clock cap in seconds",
            "style": "str (optional) — report style: 'brief' (synthesis + "
                     "sources) or 'full' (markdown report with TL;DR, "
                     "learnings, evidence check); default 'brief'",
        },
    )
    def research_deep_tool(question: str, max_queries: int = 6,
                           budget_usd: float | None = None,
                           depth: int = 3,
                           time_budget_s: float | None = None,
                           style: str = "brief",
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
            depth=int(depth or 3),
            time_budget_s=time_budget_s,
        )
        synthesis = (report.to_markdown() if str(style).lower() == "full"
                     else report.synthesis)
        return {
            "question": report.question,
            "needs_clarification": report.needs_clarification,
            "clarifications": report.clarifications,
            "sub_queries": report.sub_queries,
            "synthesis": synthesis,
            "style": style,
            "learnings": report.learnings,
            "running_summary": report.running_summary,
            "verification": report.verification,
            "conflicts": report.conflicts,
            "depth": report.depth,
            "spent_usd": report.spent_usd,
            "budget_exhausted": report.budget_exhausted,
            "citations": report.citations,
            "findings": [
                {"title": f.title, "url": f.url, "snippet": f.snippet,
                 "detail": f.detail}
                for f in report.findings
            ],
        }
