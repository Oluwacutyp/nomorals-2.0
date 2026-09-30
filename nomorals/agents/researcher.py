"""Always-on research & suggestion engine.

One top-level agent, many sub-agents: a :class:`DomainResearcher` per topic
area, each with its own curated topic bank. The three major domains:

* **lifestyle** — fitness, food, travel, money, home, wellness
* **tech**      — web, networking, data, AI, automation, systems
* **cyber**     — both sides, on purpose: good (defence, privacy tooling,
  security research, bug bounties) and dark (ransomware post-mortems,
  threat-actor TTPs, incident write-ups, honeypot findings). Research is
  research: public reporting, analysed and summarised — never acted on.

Cycle: pick a sub-agent → pick a fresh topic → research (search engine) →
digest → ONE actionable suggestion (model-sourced when a live model is
answering, extractive otherwise) → ``research_log`` → notifier to the owner
(daily-capped, feature-gated).
"""

from __future__ import annotations

import json
import random
import re
import time
import threading
from typing import Any, Sequence

from ..core.ids import new_id
from .features import feature_enabled
from .search.engine import SearchEngine

__all__ = [
    "DOMAINS", "DomainResearcher", "ResearchAgent", "DEFAULT_FEEDS_PLACEHOLDER",
    "score_idea", "actionability", "novelty", "signal_strength", "freshness",
    "is_code_actionable",
]

# ── idea quality scoring ─────────────────────────────────────────────────────
# The old engine notified on every cycle, so the owner's channel got "one
# thing worth reading" spam. The fix is a quality gate: every proposal gets a
# 0..1 composite score from four measurable dimensions, and only ideas that
# clear the bar (with real novelty) reach the owner. Low ideas are still
# logged — the pipeline needs the full distribution to stay honest.

_ACTION_VERBS = (
    "add", "deploy", "switch", "enable", "adopt", "try", "build", "write",
    "test", "run", "configure", "set up", "replace", "upgrade", "move",
    "install", "automate", "script", "rotate", "pin", "block", "monitor",
    "budget", "schedule", "batch", "precompute", "refactor", "migrate",
    "profile", "benchmark", "cache", "quantize", "version", "split", "merge",
    "alert", "flag", "cap", "isolate", "snapshot", "backup", "prune", "tune",
)
_GENERIC_PHRASES = (
    "worth reading", "worth a look", "keep an eye on", "stay tuned",
    "might be interesting", "consider looking into", "in the news",
    "people are talking about", "trending topic",
)


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    stop = {"the", "and", "that", "this", "with", "from", "for", "about",
            "how", "what", "why", "are", "was", "will", "your", "you", "one"}
    return {w for w in words if w not in stop}


def novelty(topic: str, recent_topics: Sequence[str]) -> float:
    """1.0 when the topic shares no substance with anything recently done."""
    mine = _tokens(topic)
    if not mine or not recent_topics:
        return 0.6 if recent_topics else 1.0
    best = 0.0
    for other in recent_topics[:24]:
        theirs = _tokens(other)
        if not theirs:
            continue
        overlap = len(mine & theirs) / len(mine | theirs)  # jaccard
        best = max(best, overlap)
    return max(0.0, 1.0 - best)


def actionability(suggestion: str) -> float:
    """Concrete + imperative scores high; vague platitudes score zero."""
    text = (suggestion or "").strip()
    if not text:
        return 0.0
    low = text.lower()
    score = 0.0
    if any(re.search(rf"\b{re.escape(v)}\b", low) for v in _ACTION_VERBS):
        score += 0.5
    if len(re.findall(r"\d", low)) >= 1:  # a number = a specific target
        score += 0.2
    if 40 <= len(text) <= 320:
        score += 0.15
    elif len(text) < 20:
        score -= 0.1
    if any(p in low for p in _GENERIC_PHRASES):
        score -= 0.35
    return max(0.0, min(1.0, score))


def signal_strength(digest: str, sources: Any) -> float:
    """Does the research actually contain substance?"""
    text = (digest or "").strip()
    if not text:
        return 0.0
    score = 0.0
    score += 0.3 if len(text) >= 400 else (0.15 if len(text) >= 150 else 0.0)
    domains: set[str] = set()
    count = 0
    if isinstance(sources, list):
        for item in sources[:8]:
            url = str((item or {}).get("url") or (item or {}).get("link") or "")
            count += 1
            m = re.match(r"https?://([^/]+)/?", url)
            if m:
                domains.add(m.group(1).removeprefix("www."))
    if count >= 3:
        score += 0.2
    if len(domains) >= 3:  # independent sources, not one site echoed x3
        score += 0.2
    if len(re.findall(r"\d{2,}", text)) >= 3:  # numbers = facts, not vibes
        score += 0.1
    return max(0.0, min(1.0, score))


def freshness(sources: Any, digest: str) -> float:
    """Recent material outranks evergreen padding (2025/2026 markers)."""
    now_year = time.gmtime().tm_year
    haystacks: list[str] = []
    if isinstance(sources, list):
        for item in sources[:8]:
            haystacks.append(json.dumps(item, ensure_ascii=False)[:400])
    haystacks.append((digest or "")[:2000])
    blob = " ".join(haystacks)
    if not blob:
        return 0.5
    recent = len(re.findall(rf"20{now_year - 1:02d}|{now_year}", blob))
    return max(0.0, min(1.0, 0.2 + 0.2 * min(recent, 5)))


def score_idea(topic: str, suggestion: str, digest: str,
               sources: Any, recent_topics: Sequence[str]) -> dict[str, float]:
    """Composite 0..1 quality score with the per-dimension breakdown."""
    a = actionability(suggestion)
    n = novelty(topic, recent_topics)
    s = signal_strength(digest, sources)
    f = freshness(sources, digest)
    total = 0.35 * a + 0.25 * n + 0.20 * s + 0.20 * f
    return {
        "total": round(total, 3),
        "actionability": round(a, 3),
        "novelty": round(n, 3),
        "signal": round(s, 3),
        "freshness": round(f, 3),
    }


_CODE_AUTHORSHIP = (
    "build", "write", "add", "implement", "create", "extend", "refactor",
    "automate", "a tool", "a script", "a command", "a function", "a module",
    "a cli", "parse", "export", "wire up", "wire", "patch", "scaffold",
)
_SOFTWARE_NOUN = re.compile(
    r"\b(nomorals|the repo|the codebase|the agent|the cli|tool|module|"
    r"function|file|command|test|api|server|pipeline|fetcher|scheduler|"
    r"worker|queue|cache|config|endpoint|handler|daemon|service|parser|"
    r"limiter|router|gateway|bridge|script|class|method|flag|cli)\b")


def is_code_actionable(suggestion: str) -> bool:
    """Can the coding bot plausibly turn this into a working code change?

    'Add a rate limiter to the fetcher' → yes. 'Try a new sleep schedule' →
    no, that's for the owner, not the compiler. We require BOTH a
    code-authorship signal and a software noun, so lifestyle advice that
    happens to use the word 'run' or 'test' never triggers auto-apply.
    """
    low = (suggestion or "").lower()
    if not any(re.search(rf"\b{re.escape(v)}\b", low) for v in _CODE_AUTHORSHIP):
        return False
    return bool(_SOFTWARE_NOUN.search(low))


class DomainResearcher:
    """One sub-agent: a topic area with its own bank and suggestion style."""

    def __init__(self, domain: str, name: str, topics: tuple[str, ...],
                 suggestion_style: str) -> None:
        self.domain = domain
        self.name = name
        self.topics = topics
        self.suggestion_style = suggestion_style

    def pick_topic(self, used: set[str], rng: random.Random) -> str:
        fresh = [t for t in self.topics if t not in used]
        pool = fresh or list(self.topics)
        return rng.choice(pool)

    def suggest_prompt(self, topic: str, digest: str) -> str:
        return (
            f"You are a {self.domain} researcher for a busy operator. "
            f"Topic: {topic}. Digest:\n{digest[:3000]}\n\n"
            f"Write ONE concrete, actionable suggestion ({self.suggestion_style}) "
            "in at most two sentences. Plain text, no preamble, no bullet lists."
        )


DOMAINS: dict[str, tuple[DomainResearcher, ...]] = {
    "lifestyle": (
        DomainResearcher("lifestyle", "fitness", (
            "zone 2 training and why it beats all-or-nothing cardio",
            "protein targets for strength training in 2026",
            "sleep hygiene: the interventions that actually move the needle",
            "progressive overload: how to structure a 3-day split",
            "walking volume and metabolic health: what the data says",
            "mobility work that takes 10 minutes and pays off",
        ), "one habit change you can start this week"),
        DomainResearcher("lifestyle", "food", (
            "batch cooking: the 30-minute template that actually holds",
            "umami: how to make cheap food taste expensive",
            "meal prepping on a small kitchen and a tight budget",
            "fermentation basics: what a beginner can safely make",
            "reading nutrition labels like an engineer, not a marketer",
            "air fryer vs oven: where each genuinely wins",
        ), "one recipe or swap to try this week"),
        DomainResearcher("lifestyle", "travel", (
            "shoulder season travel: when prices crater but weather holds",
            "the 24-hour layover: what's actually worth doing",
            "carry-on only: the packing system that survives 10 days",
            "how frequent flyer miles actually work in 2026",
            "border wait times and the data behind the quiet gates",
            "digital nomad visas: what's real and what's marketing",
        ), "one booking or route decision made easier"),
        DomainResearcher("lifestyle", "money", (
            "high-yield savings vs money markets in a rate-cut cycle",
            "index funds: the boring math that beats 95% of active picks",
            "emergency funds: how big is actually enough",
            "credit utilization: the number that quietly moves your score",
            "negotiating salaries: the levers that work in 2026",
            "the real cost of 'buy now pay later' over five years",
        ), "one money move with clear expected value"),
        DomainResearcher("lifestyle", "home", (
            "home maintenance: the 6 things that fail silently first",
            "insulation upgrades ranked by cost per saved degree",
            "smart home: the devices that earn their place",
            "water leaks: how to find the ones you can't see",
            "generators and surge protection for unreliable grids",
            "the ventilation mistakes that ruin new builds",
        ), "one fix or check to do before it becomes expensive"),
        DomainResearcher("lifestyle", "wellness", (
            "cold exposure: what's real, what's vibes",
            "breathwork protocols with actual evidence",
            "screen time and sleep: the boundary that works",
            "daily stand-up with yourself: the 5-minute version",
            "micro-breaks and the ultradian rhythm",
            "journaling formats that survive a busy month",
        ), "one small protocol worth trying for a week"),
    ),
    "tech": (
        DomainResearcher("tech", "web", (
            "HTTP/3 in production: what still breaks in the wild",
            "edge rendering vs SSR: where each one actually wins",
            "the state of web auth: passkeys, WebAuthn, and the gaps",
            "browser caching in 2026: why your CDN bill still surprises you",
            "webAssembly on the frontend: real wins vs demos",
            "how modern bundlers decide what to ship",
        ), "one architecture or tooling decision clarified"),
        DomainResearcher("tech", "networking", (
            "WPA3 and the attacks that still work anyway",
            "how BGP hijacks happen: the 2026 write-ups",
            "home lab routing: OPNsense, pfSense, or OpenWrt",
            "DNS privacy: DoH, DoT, and what they hide from whom",
            "10G at the desktop: what the bottleneck actually is",
            "how cellular data planes work under the hood",
        ), "one network decision made with real numbers"),
        DomainResearcher("tech", "data", (
            "vector databases: the indexes that work for RAG",
            "SQLite at scale: the patterns that hold to a million rows",
            "data lakes into data swamps: the failure modes",
            "feature stores: where the complexity actually lives",
            "streaming vs batch: choosing without religion",
            "the economics of storage tiers in 2026",
        ), "one data decision with the trade-offs named"),
        DomainResearcher("tech", "ai", (
            "mixture-of-experts: why inference got cheap",
            "small language models that beat big ones on narrow tasks",
            "RLHF vs DPO: what the loops optimize and where they lie",
            "prompt caching and KV reuse: the engineering that saves money",
            "local LLMs on a phone: what actually fits",
            "evals: how to know your model change was an improvement",
        ), "one model or pipeline decision backed by numbers"),
        DomainResearcher("tech", "automation", (
            "idempotent pipelines: why your retry logic is lying to you",
            "flaky tests: the triage that kills them for good",
            "self-healing infrastructure and where it backfires",
            "the design of dead-letter queues that don't rot",
            "CI on a budget: what a small team actually needs",
            "feature flags without config drift",
        ), "one automation fix that removes a recurring fire"),
        DomainResearcher("tech", "systems", (
            "how Linux namespaces compose into a container cage",
            "NUMA and the lies multi-socket servers tell",
            "JIT compilation: speed of compile vs speed of code",
            "the page cache and how it evicts",
            "memory barriers and why ARM code surprises x86 devs",
            "io_uring: what changed and what it costs",
        ), "one systems concept that makes the next incident predictable"),
    ),
    "cyber": (
        DomainResearcher("cyber", "threats", (
            "the latest ransomware post-mortems: what the public reports share",
            "threat-actor TTPs this quarter from public threat intel",
            "how initial access brokers actually operate (public reporting)",
            "supply chain attacks: the 2025-26 public incidents",
            "credential stuffing defence: the layered playbook",
            "phishing kits in 2026: what the public analyses show",
        ), "one defensive measure worth adopting this week"),
        DomainResearcher("cyber", "defence", (
            "eBPF in the security stack: what it finally made possible",
            "passkeys at the org level: the migration that works",
            "zero-trust for small teams: the realistic version",
            "SIEM without the 10k-seat contract",
            "incident response playbooks: the ones that survive a real night",
            "security awareness that isn't a compliance checkbox",
        ), "one control to add that costs little and pays a lot"),
        DomainResearcher("cyber", "research", (
            "bug bounty write-ups: the classes of bugs that still pay",
            "honeypot findings: what attackers tried this quarter",
            "malware analysis: the techniques in recent public dissections",
            "OSINT methods from public research (and their limits)",
            "privacy tooling: what's real vs marketing",
            "CTF write-ups: the tricks that transfer to defence",
        ), "one technique to test on your own systems"),
        DomainResearcher("cyber", "dark", (
            "what the dark web marketplaces are actually selling this quarter",
            "stolen-data trade patterns from public takedowns and reporting",
            "scam infrastructure: how the kits are built and sold publicly",
            "the economics of carding operations from public court records",
            "botnet anatomy from public takedown reports",
            "exploit brokers and the public side of that economy",
        ), "one red flag to add to your monitoring"),
    ),
}


class ResearchAgent:
    """Top-level agent: rotates sub-agents, journals everything, notifies."""

    def __init__(self, context: Any, notifier: Any = None) -> None:
        self.context = context
        self.settings = getattr(context, "settings", None)
        self.db = getattr(context, "db", None)
        self.notifier = notifier
        self.rng = random.Random()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── domain helpers ───────────────────────────────────────────────────────
    def _domains(self) -> tuple[str, ...]:
        raw = getattr(self.settings, "research", None)
        wanted = [d.strip() for d in (getattr(raw, "domains", "") or "lifestyle,tech,cyber").split(",") if d.strip()]
        return tuple(d for d in wanted if d in DOMAINS) or ("lifestyle", "tech", "cyber")

    def _used_topics(self, domain: str) -> set[str]:
        used: set[str] = set()
        if self.db is None:
            return used
        try:
            rows = self.db.query(
                "SELECT topic FROM research_log WHERE domain = ? LIMIT 200", (domain,)
            )
            for row in rows:
                used.add(str(row.get("topic", "")))
        except Exception:  # noqa: BLE001
            pass
        return used

    def _model_suggestion(self, prompt: str) -> str:
        router = getattr(self.context, "router", None)
        if router is None:
            return ""
        from ..llm.base import Message

        try:
            response = router.chat([Message(role="user", content=prompt)])
            text = (getattr(response, "text", "") or "").strip()
            if getattr(response, "error", None):
                return ""
            return text
        except Exception:  # noqa: BLE001
            return ""

    # ── one cycle ────────────────────────────────────────────────────────────
    def run_cycle(self, domain: str | None = None) -> dict[str, Any]:
        started = time.time()
        domains = self._domains()
        dom = domain or self.rng.choice(domains)
        researchers = DOMAINS[dom]
        researcher = self.rng.choice(researchers)
        topic = researcher.pick_topic(self._used_topics(dom), self.rng)
        try:
            pages = 3
            raw = getattr(self.settings, "research", None)
            pages = max(1, min(int(getattr(raw, "research_pages", 3)), 8))
            report = SearchEngine(self.context).run(topic, mode="quick", pages=pages)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "domain": dom, "topic": topic, "error": f"research failed: {exc}"}
        digest = str(report.get("summary") or "").strip()
        pages_read = report.get("pages_read") or []
        n_pages = len(pages_read) if isinstance(pages_read, (list, tuple)) \
            else int(pages_read or 0)
        if not digest or n_pages == 0:
            return {"ok": False, "domain": dom, "topic": topic,
                    "error": "research returned nothing readable"}
        suggestion = self._model_suggestion(researcher.suggest_prompt(topic, digest))
        if not suggestion:
            first = next((line.strip() for line in digest.splitlines() if line.strip()), digest[:200])
            suggestion = f"worth 10 minutes: {first[:180]}"

        sources = report.get("results", [])[:6]
        scores = score_idea(topic, suggestion, digest, sources,
                            self._recent_topics())
        # The value gate: the owner's channel is precious. An idea only gets
        # pushed when it scores well AND is actually new — repeats of what
        # was already delivered are the failure mode that makes people mute
        # the whole feature.
        valuable = (scores["total"] >= self._notify_threshold()
                    and scores["novelty"] >= 0.3)
        # wave 85: the reasoning agent critiques anything about to be
        # pushed — a concrete weakness demotes the idea to pending so the
        # owner's channel only gets ideas that survive a sharp review.
        if valuable:
            critique = self._reasoning_critique(topic, suggestion)
            if critique:
                scores = dict(scores)
                scores["total"] = round(max(0.0, scores["total"] - 0.2), 3)
                scores["reasoning_note"] = critique
                valuable = (scores["total"] >= self._notify_threshold()
                            and scores["novelty"] >= 0.3)
        if valuable and self._under_cap() and self.notifier is not None:
            status = "notified"
            delivered = 1
        elif valuable:
            status = "pending"  # valuable but over the daily cap — wait in line
            delivered = 0
        else:
            status = "skipped"  # logged, never pushed
            delivered = 0

        rid = new_id()
        if self.db is not None:
            try:
                with self.db.transaction():
                    self.db.execute(
                        "INSERT INTO research_log (id, domain, topic, digest, suggestion, sources, "
                        "delivered, created_at, score, score_detail, status) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (rid, dom, topic, digest[:4000], suggestion,
                         json.dumps(sources, ensure_ascii=False),
                         delivered, time.time(),
                         scores["total"], json.dumps(scores, ensure_ascii=False), status),
                    )
            except Exception:  # noqa: BLE001
                pass
        if status == "notified":
            self.notifier.publish(
                "research",
                f"research ({dom}/{researcher.name}) score {scores['total']:.2f}: {topic[:60]}",
                f"suggestion: {suggestion}\ndigest: {digest[:900]}",
            )
        if (status == "notified" and self._auto_apply_enabled()
                and dom == "tech" and scores["total"] >= self._auto_apply_threshold()
                and is_code_actionable(suggestion)):
            self._apply_in_background(
                rid, self._apply_task(topic, suggestion, digest),
                self._apply_filename(rid),
            )
        return {"ok": True, "domain": dom, "researcher": researcher.name, "topic": topic,
                "suggestion": suggestion, "score": scores["total"], "status": status,
                "seconds": round(time.time() - started, 1)}

    # ── wave 85: the reasoning layer in the research loop ──────────────────
    def _reasoning_critique(self, topic: str, suggestion: str) -> str:
        """One bounded adversarial pass over a would-be notification.

        Returns the concrete weakness when the idea is weak, ``""`` when
        it stands (or reasoning is off / budget exhausted) — a critique
        must never block a research cycle.
        """
        try:
            from .reasoning import ReasoningAgent, reasoning_enabled

            if not reasoning_enabled(self.context):
                return ""
            agent = ReasoningAgent(self.context)
            verdict = agent.advise(
                "Critique this research suggestion that is about to be "
                "pushed to the owner's notification channel. Be sharp: "
                "vague, generic, not actionable, or already-known ideas "
                "fail. Topic: " + topic[:160] + ". Suggestion: " +
                suggestion[:300] + ". Reply with one short sentence — "
                "either 'ok' or the concrete weakness.",
                focus="critique", max_tokens=160,
            )
        except Exception:  # noqa: BLE001 — critique is a bonus, never a gate
            return ""
        verdict = (verdict or "").strip()
        if not verdict or verdict.lower().startswith("ok"):
            return ""
        return verdict[:200]

    # ── proposal pipeline (score → approve → auto-apply) ───────────────────
    def _notify_threshold(self) -> float:
        raw = getattr(self.settings, "research", None)
        return max(0.0, min(1.0, float(getattr(raw, "notify_threshold", 0.55) or 0.55)))

    def _auto_apply_enabled(self) -> bool:
        raw = getattr(self.settings, "research", None)
        return bool(getattr(raw, "auto_apply", True))

    def _auto_apply_threshold(self) -> float:
        raw = getattr(self.settings, "research", None)
        return max(0.0, min(1.0, float(getattr(raw, "auto_apply_threshold", 0.8) or 0.8)))

    def _recent_topics(self, *, limit: int = 24) -> list[str]:
        if self.db is None:
            return []
        try:
            rows = self.db.query(
                "SELECT topic FROM research_log ORDER BY created_at DESC LIMIT ?", (limit,))
            return [str(row.get("topic", "")) for row in rows]
        except Exception:  # noqa: BLE001
            return []

    def list_proposals(self, limit: int = 5, status: str = "") -> list[dict[str, Any]]:
        """Pending-first view of the proposal queue, best score on top."""
        if self.db is None:
            return []
        if status:
            rows = self.db.query(
                "SELECT id, domain, topic, suggestion, score, status, delivered, created_at "
                "FROM research_log WHERE status = ? ORDER BY score DESC, created_at DESC LIMIT ?",
                (status, max(1, min(int(limit), 50))))
        else:
            rows = self.db.query(
                "SELECT id, domain, topic, suggestion, score, status, delivered, created_at "
                "FROM research_log ORDER BY "
                "CASE status WHEN 'pending' THEN 0 WHEN 'notified' THEN 1 ELSE 2 END, "
                "score DESC, created_at DESC LIMIT ?", (max(1, min(int(limit), 50)),))
        return [dict(r) for r in rows]

    def resolve_proposal(self, ref: str) -> dict[str, Any] | None:
        """Resolve an id or 'latest' to one proposal row."""
        if self.db is None or not ref:
            return None
        ref = ref.strip()
        if ref in {"latest", "last"}:
            row = self.db.query_one(
                "SELECT id, domain, topic, suggestion, digest, score, status, created_at "
                "FROM research_log ORDER BY created_at DESC LIMIT 1")
        else:
            row = self.db.query_one(
                "SELECT id, domain, topic, suggestion, digest, score, status, created_at "
                "FROM research_log WHERE id LIKE ?", (ref + "%",))
        return dict(row) if row else None

    def _set_status(self, rid: str, status: str) -> None:
        if self.db is None:
            return
        try:
            self.db.execute("UPDATE research_log SET status = ? WHERE id = ?", (status, rid))
        except Exception:  # noqa: BLE001
            pass

    def approve(self, ref: str) -> str:
        """Approve a proposal; tech + code-actionable proposals start applying."""
        row = self.resolve_proposal(ref)
        if row is None:
            return "no such proposal (ids: try /research ideas)"
        if row.get("status") in {"applied", "applying"}:
            return f"already {'applied' if row['status'] == 'applied' else 'applying'}: {row['topic'][:60]}"
        self._set_status(row["id"], "approved")
        if row.get("domain") == "tech" and is_code_actionable(row.get("suggestion", "")):
            self._apply_in_background(
                row["id"],
                self._apply_task(row.get("topic", ""), row.get("suggestion", ""),
                                 row.get("digest", "")),
                self._apply_filename(row["id"]),
            )
            return f"approved & applying in the background: {row['topic'][:70]}"
        return f"approved (not auto-applicable — keep it as a suggestion): {row['topic'][:70]}"

    def deny(self, ref: str) -> str:
        row = self.resolve_proposal(ref)
        if row is None:
            return "no such proposal (ids: try /research ideas)"
        self._set_status(row["id"], "denied")
        return f"denied: {row['topic'][:70]}"

    def _apply_filename(self, rid: str) -> str:
        return f"research/auto_{rid[:8]}.py"

    def _apply_task(self, topic: str, suggestion: str, digest: str) -> str:
        return (
            "Implement this improvement to the NoMorals repo. It was suggested by the "
            f"research engine (topic: {topic}).\n\n"
            f"Suggestion: {suggestion}\n\n"
            f"Research context:\n{digest[:1500]}\n\n"
            "Rules: keep it small and self-contained — one Python file. Make it a real, "
            'importable, runnable module (no placeholders). Under an `if __name__ == '
            '"__main__":` guard add a self-check that exits non-zero if the module is broken. '
            "No network access, no third-party packages beyond the stdlib and the nomorals "
            "package itself."
        )

    def _apply_in_background(self, rid: str, task: str, filename: str) -> None:
        def _work() -> None:
            from .coding import CodingAgent

            self._set_status(rid, "applying")
            try:
                result = CodingAgent(self.context).run(
                    task, filename=filename, max_iterations=3, timeout=60.0)
                status = "applied" if result.ok else "apply_failed"
                detail = (result.output if result.ok else result.error)[:2000]
            except Exception as exc:  # noqa: BLE001 - journal, never crash the loop
                status, detail = "apply_failed", str(exc)[:2000]
            if self.db is not None:
                try:
                    self.db.execute(
                        "UPDATE research_log SET status = ?, applied_at = ?, apply_result = ? "
                        "WHERE id = ?", (status, time.time(), detail, rid))
                except Exception:  # noqa: BLE001
                    pass

        threading.Thread(target=_work, name=f"research-apply-{rid[:8]}", daemon=True).start()

    def _under_cap(self) -> bool:
        cap = 3
        raw = getattr(self.settings, "research", None)
        cap = int(getattr(raw, "daily_suggestion_cap", 3))
        if self.db is None:
            return cap > 0
        try:
            today = time.time() - 86400
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM research_log WHERE delivered = 1 AND created_at > ?",
                (today,),
            )
            return int(row.get("n", 0)) < cap
        except Exception:  # noqa: BLE001
            return True

    # ── background loop ──────────────────────────────────────────────────────
    def start_loop(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    raw = getattr(self.settings, "research", None)
                    if getattr(raw, "enabled", False) and feature_enabled(self.context, "research"):
                        self.run_cycle()
                except Exception:  # noqa: BLE001 - the loop must never die
                    pass
                hours = 12.0
                raw = getattr(self.settings, "research", None)
                hours = max(0.05, float(getattr(raw, "interval_hours", 12.0)))
                self._stop.wait(hours * 3600.0)

        self._thread = threading.Thread(target=_loop, name="research-loop", daemon=True)
        self._thread.start()
        return True

    def stop_loop(self) -> None:
        self._stop.set()

    def loop_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
