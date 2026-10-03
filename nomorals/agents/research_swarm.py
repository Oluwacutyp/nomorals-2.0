"""Research swarm (wave 86): parallel specialized researchers.

One question, N angles, run at once:

* **angle decomposition** — the goal is split into 2-4 research angles
  (core / practical / critical / recent). Pass ``specialists=[...]`` to
  decompose along domain-specialist templates instead (systems, security,
  product, ml, tooling, competitors); findings are tagged with their
  domain. When a router is configured the Reasoning Agent refines the
  angles; the deterministic templates are the fallback so the swarm never
  depends on the model being up.
* **specialized workers** — one researcher per angle, in parallel
  (profile-tuned worker count). Each worker searches its angle, annotates
  the results with source trust, and reads the most credible pages to pull
  out concrete claims. A worker that dies is recorded, not fatal.
* **conflict-aware synthesis** — claims are grouped by subject and
  opposed statements are flagged as conflicts instead of being silently
  averaged away; the synthesis (model-written when available,
  deterministic otherwise) states where sources disagree.
* **cross-specialist dedup** — when two angles/specialists surface the
  same claim, ``dedupe_findings`` merges them into one finding with the
  union of sources (every URL kept), the union of domains/angles, and
  the max confidence; affirmations are never merged with negations.
* **supervision** — when the Reasoning Agent is reachable, the finished
  synthesis gets an adversarial critique that lands in the report.
* **feeds** — the report can be filed into memory (``to_memory``), run as
  a mission, streamed to chat (``/swarm research <topic>``), or called by
  any agent through the ``research_swarm`` tool.

Everything degrades: no router → template angles + deterministic
synthesis; no network for one angle → that angle's worker reports the
failure and the rest of the report stands.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from ..core.trust import SourceTrust
from .search.engine import SearchEngine

_log = get_logger(__name__)

__all__ = ["SwarmFinding", "SwarmReport", "ResearchSwarm"]

#: Specialist domains for angle decomposition: each entry holds 3-4
#: domain-specific angle templates (``{q}`` is the query) plus the
#: keywords that characterize the domain.  In domain order, the first
#: ``workers`` templates become the swarm's angles, and every finding is
#: tagged with the domain that produced it.
SPECIALIST_DOMAINS: dict[str, dict[str, list[str]]] = {
    "systems": {
        "angles": [
            "{q} distributed systems architecture design",
            "{q} scalability reliability patterns",
            "{q} performance latency throughput tuning",
            "{q} fault tolerance failover consensus",
        ],
        "keywords": [
            "distributed", "architecture", "scalability", "reliability",
            "latency", "throughput", "fault-tolerance", "consensus",
            "sharding", "caching", "load-balancing", "observability",
        ],
    },
    "security": {
        "angles": [
            "{q} threat model vulnerabilities",
            "{q} security best practices hardening",
            "{q} CVE exploit mitigations",
            "{q} authentication authorization access control",
        ],
        "keywords": [
            "threat-model", "vulnerability", "CVE", "exploit",
            "hardening", "authentication", "authorization", "encryption",
            "zero-trust", "audit", "pentest", "sandboxing",
        ],
    },
    "product": {
        "angles": [
            "{q} product market fit user research",
            "{q} product strategy roadmap prioritization",
            "{q} UX onboarding activation retention",
            "{q} product metrics analytics growth",
        ],
        "keywords": [
            "product-market-fit", "user-research", "roadmap", "UX",
            "onboarding", "retention", "activation", "conversion",
            "pricing", "positioning", "churn", "engagement",
        ],
    },
    "ml": {
        "angles": [
            "{q} machine learning model architecture",
            "{q} training fine-tuning hyperparameters",
            "{q} ML evaluation benchmarks metrics",
            "{q} model deployment inference optimization",
        ],
        "keywords": [
            "neural-network", "transformer", "training", "fine-tuning",
            "hyperparameters", "benchmark", "inference", "quantization",
            "embedding", "dataset", "overfitting", "LLM",
        ],
    },
    "tooling": {
        "angles": [
            "{q} developer tooling workflow automation",
            "{q} CI/CD build pipeline tooling",
            "{q} CLI developer experience tooling",
            "{q} debugging observability tooling",
        ],
        "keywords": [
            "CLI", "CI/CD", "build", "pipeline", "automation",
            "debugging", "observability", "profiling", "linting",
            "testing", "packaging", "DX",
        ],
    },
    "competitors": {
        "angles": [
            "{q} competitor landscape comparison",
            "{q} alternatives to {q}",
            "{q} competitive pricing differentiation",
            "{q} market leaders vs {q}",
        ],
        "keywords": [
            "competitor", "alternative", "comparison", "pricing",
            "market-share", "landscape", "benchmark", "differentiation",
            "incumbent", "disruptor", "vendor", "open-source",
        ],
    },
}

_CONTENT_WORD = re.compile(r"[a-z][a-z0-9\-]{3,}")
_NEGATION = re.compile(
    r"\b(not|no|never|cannot|can'?t|won'?t|false|wrong|disprove\w*|myth|"
    r"unreliable|scam|fraud|debunk\w*)\b", re.IGNORECASE)
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "was",
    "were", "are", "will", "would", "about", "into", "over", "under", "than",
    "then", "them", "they", "their", "what", "when", "where", "which", "who",
    "how", "why", "can", "could", "should", "does", "doing", "done", "best",
    "new", "using", "use", "used", "based", "such", "like", "also", "more",
    "most", "much", "many", "some", "any", "all", "its", "it's", "you", "your",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _CONTENT_WORD.findall((text or "").lower()) if w not in _STOP}


def _stem(word: str) -> str:
    """Crude normalizer so near-duplicate detection sees past inflections:
    'sessions'/'session', 'reduces'/'reducing'. Only ever shortens to a
    stem of length >= 4 — never invents a root."""
    w = word
    for suffix in ("ing", "ies", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            w = w[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    return w


def _stemmed_words(text: str) -> set[str]:
    return {_stem(w) for w in _content_words(text)}


def _same_claim(a: str, b: str) -> bool:
    """Near-duplicate test for cross-specialist dedup: the same claim in
    different words. Never true across an affirmation/negation boundary —
    that is a conflict for ``_detect_conflicts``, not a duplicate."""
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return False
    if a.lower() == b.lower():
        return True
    if bool(_NEGATION.search(a)) != bool(_NEGATION.search(b)):
        return False
    wa, wb = _stemmed_words(a), _stemmed_words(b)
    if not wa or not wb:
        return False
    inter = wa & wb
    if len(inter) >= 4 and (inter == wa or inter == wb):
        return True  # one claim's substance sits inside the other
    if len(inter) >= 6:
        return True  # six shared topic words is the same claim
    union = wa | wb
    return len(inter) / len(union) >= 0.45 if union else False


def _union_tag(cur: str, add: str) -> str:
    """Merge domain/angle tags without duplication: 'ml' + 'systems' ->
    'ml+systems'."""
    parts = [p for p in str(cur or "").split("+") if p]
    for p in str(add or "").split("+"):
        if p and p not in parts:
            parts.append(p)
    return "+".join(parts)


@dataclass
class SwarmFinding:
    """One concrete claim, backed by named sources."""

    angle: str
    claim: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.3
    domain: str = ""  # specialist domain that produced the finding ("" = unknown)

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle": self.angle,
            "claim": self.claim,
            "sources": self.sources,
            "confidence": round(self.confidence, 2),
            "domain": self.domain,
        }

    def evidence_urls(self) -> list[str]:
        """The finding's cited URLs, in order — its evidence."""
        urls: list[str] = []
        for s in self.sources or []:
            if isinstance(s, dict):
                url = str(s.get("url") or "").strip()
                if url and url not in urls:
                    urls.append(url)
        return urls


@dataclass
class SwarmReport:
    query: str
    angles: list[str]
    findings: list[SwarmFinding] = field(default_factory=list)
    synthesis: str = ""
    conflicts: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    workers: int = 0
    failed_angles: list[str] = field(default_factory=list)
    supervised: bool = False
    critique: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "angles": self.angles,
            "workers": self.workers,
            "seconds": round(self.seconds, 1),
            "failed_angles": self.failed_angles,
            "conflicts": self.conflicts,
            "findings": [f.to_dict() for f in self.findings],
            "sources": self.sources,
            "supervised": self.supervised,
            "critique": self.critique,
            "synthesis": self.synthesis,
        }

    def to_text(self, max_chars: int = 4000) -> str:
        """Chat-friendly rendering: synthesis first, then the evidence."""
        parts: list[str] = []
        parts.append(self.synthesis.strip() or "(no synthesis)")
        if self.conflicts:
            parts.append("\nwhere sources disagree:\n"
                         + "\n".join(f"  - {c}" for c in self.conflicts[:5]))
        top = sorted(self.findings, key=lambda f: -f.confidence)[:8]
        if top:
            lines = []
            for f in top:
                src = f.sources[0]["title"][:40] if f.sources else "web"
                lines.append(f"  - {f.claim[:160]}  [{f.confidence:.0%} via {src}]")
            parts.append("key findings:\n" + "\n".join(lines))
        if self.critique:
            parts.append(f"self-critique: {self.critique[:200]}")
        text = "\n".join(parts)
        return text[:max_chars]


class ResearchSwarm:
    """The coordinator: angles out, parallel workers in, synthesis out."""

    def __init__(
        self,
        context: Any,
        *,
        workers: int | None = None,
        timeout: float = 120.0,
        read_pages: bool = True,
        session_prefix: str = "swarm",
        specialists: list[str] | None = None,
    ) -> None:
        self.context = context
        self.timeout = float(timeout)
        self.read_pages = bool(read_pages)
        self.session_prefix = session_prefix
        if specialists is not None:
            unknown = [s for s in specialists
                       if s not in SPECIALIST_DOMAINS]
            if unknown:
                raise ValueError(
                    f"unknown specialist domain(s): {', '.join(unknown)} "
                    f"(choose from {', '.join(SPECIALIST_DOMAINS)})")
            self.specialists: tuple[str, ...] = tuple(specialists)
        else:
            self.specialists = ()
        self._angle_domains: list[str] = []  # parallel to the last angles_for
        # profile-tuned default: half the thread pool, 2..4, and never more
        # parallel than the profile's mission aggressiveness allows
        # (a phone gets 2 light web workers, a workstation 4).
        tune = (getattr(context, "extras", None) or {}).get("tune")
        default = max(2, min(4, int(getattr(tune, "threads", 8) or 8) // 2))
        if tune is not None:
            default = min(default, max(2, int(getattr(tune, "mission_max_concurrent", 2) or 2)))
        self.workers = max(1, min(6, int(workers) if workers else default))
        self.search_engine = SearchEngine(context)
        self.trust = SourceTrust(context)

    # ── angle decomposition ────────────────────────────────────────────────
    def angles_for(self, query: str) -> list[str]:
        """2-4 research angles for the goal: specialist domain templates
        when ``specialists`` is set (findings tagged per domain), the
        generic core/practical/critical/recent templates otherwise.
        Model-refined when a router is configured (domains unknown then,
        tagged ``""`` — honest)."""
        q = (query or "").strip().rstrip("?")
        if not q:
            raise ValueError("swarm needs a query")
        if self.specialists:
            angles, domains = self._specialist_angles(q)
        else:
            angles = [
                q,
                f"{q} best practices how to",
                f"{q} criticism problems limitations",
                f"{q} recent developments",
            ]
            angles = list(dict.fromkeys(angles))
            domains = [""] * len(angles)
        if self._router_available():
            try:
                refined = self._model_angles(q)
                if refined:
                    angles = refined[: self.workers]
                    # model-generated angles don't map to a known domain
                    domains = [""] * len(angles)
            except Exception as exc:  # noqa: BLE001 - templates always work
                _log.debug("model angle refinement failed: %s", exc)
        angles = angles[: self.workers]
        self._angle_domains = (domains[: len(angles)]
                               + [""] * max(0, len(angles) - len(domains)))
        return angles

    def _specialist_angles(self, q: str) -> tuple[list[str], list[str]]:
        """Expand specialist templates in domain order, capped at the
        worker count.  Returns (angles, parallel domain tags)."""
        pairs: list[tuple[str, str]] = []
        for domain in self.specialists:
            for template in SPECIALIST_DOMAINS[domain]["angles"]:
                pairs.append((template.format(q=q), domain))
                if len(pairs) >= self.workers:
                    break
            if len(pairs) >= self.workers:
                break
        # de-dupe angles, keeping the first (domain-order) owner
        seen: set[str] = set()
        angles: list[str] = []
        domains: list[str] = []
        for angle, domain in pairs:
            if angle not in seen:
                seen.add(angle)
                angles.append(angle)
                domains.append(domain)
        return angles, domains

    def _router_available(self) -> bool:
        router = getattr(self.context, "router", None)
        return router is not None and hasattr(router, "chat")

    def _model_angles(self, query: str) -> list[str]:
        from ..llm.base import Message, SamplingParams

        prompt = (
            "You are planning a parallel research swarm. Split the goal into "
            f"{self.workers} distinct research angles (2-6 words each): one "
            "broad, one practical, one critical, one recent. Respond with a "
            "JSON array of strings only.\n\nGoal: " + query[:400]
        )
        response = self.context.router.chat(
            [Message.user(prompt)], SamplingParams(temperature=0.4, max_tokens=200))
        if not getattr(response, "ok", False) or not response.text:
            return []
        text = response.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        out = [str(a).strip().strip('"') for a in data if str(a).strip()]
        return [a for a in out if 3 <= len(a) <= 80][: self.workers]

    # ── one worker ─────────────────────────────────────────────────────────
    def _research_angle(self, angle: str, worker_index: int,
                        domain: str = "") -> dict[str, Any]:
        started = time.time()
        # adaptive breadth: the angle's own phrasing sets the result count
        results = self.search_engine.search(angle)
        try:
            results = self.trust.annotate(results)
        except Exception:  # noqa: BLE001 - trust is an enhancement
            pass
        if not results:
            return {"angle": angle, "ok": False, "error": "no search results",
                    "findings": [], "seconds": time.time() - started}
        results = sorted(results, key=lambda r: -float(r.get("trust", 0.5)))
        findings: list[SwarmFinding] = []
        angle_words = _content_words(angle)
        pages_read = 0
        for rank, result in enumerate(results[:4]):
            url = str(result.get("url", ""))
            title = str(result.get("title", "") or url)
            trust = float(result.get("trust", 0.5))
            snippet = str(result.get("snippet", "") or "")
            # 1) the search snippet itself is always usable evidence
            claim = self._best_sentence(snippet, angle_words)
            if claim:
                findings.append(
                    self._finding(angle, claim, [(url, title, trust)], trust,
                                  domain))
            # 2) read the most credible pages for deeper claims
            if self.read_pages and rank < 2 and pages_read < 2:
                page_text = self._read_page(url, worker_index)
                if page_text:
                    pages_read += 1
                    for sentence in self._page_claims(page_text, angle_words, limit=3):
                        findings.append(
                            self._finding(angle, sentence, [(url, title, trust)],
                                          trust, domain))
        # de-duplicate, keep the strongest source set per claim
        deduped = self._dedupe(findings)
        return {
            "angle": angle, "ok": True, "findings": deduped,
            "sources": [
                {"url": str(r.get("url", "")), "title": str(r.get("title", "")),
                 "trust": float(r.get("trust", 0.5))}
                for r in results[:6]
            ],
            "pages_read": pages_read,
            "seconds": round(time.time() - started, 2),
        }

    def _read_page(self, url: str, worker_index: int) -> str:
        try:
            from ..tools.browser import get_session, drop_session

            tune = (getattr(self.context, "extras", None) or {}).get("tune")
            max_chars = 8000 if str(getattr(getattr(tune, "profile", None), "kind", "")) \
                in ("termux", "mobile", "embedded") else 12000
            session = f"{self.session_prefix}-{worker_index}"
            sess = get_session(session)
            try:
                result = sess.do("open", url=url)
                if not result.get("ok"):
                    return ""
                return str(sess.do("text", max_chars=max_chars).get("text", ""))
            finally:
                drop_session(session)
        except Exception as exc:  # noqa: BLE001 - a dead page is not a dead swarm
            _log.debug("swarm page read failed (%s): %s", url, exc)
            return ""

    # ── claim mining ───────────────────────────────────────────────────────
    _SENTENCE = re.compile(r"(?<=[.!?])\s+")

    @staticmethod
    def _best_sentence(text: str, angle_words: set[str]) -> str:
        sentences = [s.strip() for s in ResearchSwarm._SENTENCE.split(text or "")
                     if 40 <= len(s.strip()) <= 300]
        if not sentences:
            return ""
        best, best_score = sentences[0], -1.0
        for s in sentences:
            overlap = len(_content_words(s) & angle_words)
            score = overlap * 2.0 + min(1.0, len(s) / 200.0)
            if score > best_score:
                best, best_score = s, overlap * 2.0 + min(1.0, len(s) / 200.0)
        return best if best_score > 0 else ""

    def _page_claims(self, text: str, angle_words: set[str], limit: int = 3) -> list[str]:
        sentences = [s.strip() for s in ResearchSwarm._SENTENCE.split(text or "")
                     if 60 <= len(s.strip()) <= 320]
        scored: list[tuple[float, str]] = []
        for s in sentences:
            overlap = len(_content_words(s) & angle_words)
            if overlap >= 2:
                scored.append((overlap, s))
        scored.sort(key=lambda pair: -pair[0])
        out: list[str] = []
        seen: set[str] = set()
        for _score, s in scored:
            key = s[:60].lower()
            if key not in seen:
                seen.add(key)
                out.append(s)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _finding(angle: str, claim: str, sources: list[tuple[str, str, float]],
                 trust: float, domain: str = "") -> SwarmFinding:
        support = min(3, len({s[0] for s in sources}))
        confidence = min(0.95, 0.25 + 0.15 * support + 0.25 * float(trust))
        return SwarmFinding(
            angle=angle,
            claim=claim.strip(),
            sources=[{"url": u, "title": t, "trust": tr} for u, t, tr in sources[:4]],
            confidence=confidence,
            domain=domain,
        )

    @staticmethod
    def _dedupe(findings: list[SwarmFinding]) -> list[SwarmFinding]:
        seen: dict[str, SwarmFinding] = {}
        for f in findings:
            key = f.claim[:80].lower()
            existing = seen.get(key)
            if existing is None or f.confidence > existing.confidence:
                seen[key] = f
        return list(seen.values())[:10]

    @staticmethod
    def dedupe_findings(findings: list[SwarmFinding]) -> list[SwarmFinding]:
        """Merge near-duplicate findings across angles/specialists.

        Two specialists reporting the same finding become ONE finding: the
        union of sources (deduped by URL — every source kept), the union
        of domains/angles, and the max confidence. Strongest first so the
        surviving record is the best-evidenced one. Affirmations are never
        merged with their negations.
        """
        merged: list[SwarmFinding] = []
        ordered = sorted(findings,
                         key=lambda x: -float(getattr(x, "confidence", 0.0)
                                              or 0.0))
        for f in ordered:
            placed = False
            for m in merged:
                if not _same_claim(f.claim, m.claim):
                    continue
                seen_urls = {s.get("url") for s in m.sources
                             if isinstance(s, dict)}
                for s in f.sources or []:
                    if isinstance(s, dict) and s.get("url") not in seen_urls:
                        m.sources.append(s)
                        seen_urls.add(s.get("url"))
                m.confidence = max(float(m.confidence or 0.0),
                                   float(f.confidence or 0.0))
                m.angle = _union_tag(m.angle, f.angle)
                m.domain = _union_tag(m.domain, f.domain)
                placed = True
                break
            if not placed:
                merged.append(copy.copy(f))
        return merged

    # ── conflict detection ─────────────────────────────────────────────────
    @staticmethod
    def _detect_conflicts(findings: list[SwarmFinding]) -> list[str]:
        """Flag opposed claims: two findings about the SAME subject, one of
        them negating, coming from different angles.

        Deliberately conservative — the shared-subject test requires two
        real topic words (≥4 letters), and only an affirmation vs a
        negation counts. Opposing SOURCES conflict, same angle or not
        (that is the normal case: one search, two sides); emphasis
        differences are left to the synthesis to mention.
        """
        conflicts: list[str] = []
        seen: set[tuple[str, str]] = set()
        ordered = sorted(findings, key=lambda f: -f.confidence)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                shared = {
                    w for w in (_content_words(a.claim) & _content_words(b.claim))
                    if len(w) >= 4
                }
                if len(shared) < 2:
                    continue
                a_neg = bool(_NEGATION.search(a.claim))
                b_neg = bool(_NEGATION.search(b.claim))
                if a_neg == b_neg:
                    continue
                affirmed, negated = (a, b) if a_neg else (b, a)
                key = (affirmed.angle, negated.angle)
                if key in seen:
                    continue
                seen.add(key)
                src_a = affirmed.sources[0]["url"] if affirmed.sources else ""
                src_b = negated.sources[0]["url"] if negated.sources else ""
                conflicts.append(
                    f"sources disagree — {affirmed.claim[:120]}… [{src_a[:60]}] "
                    f"vs {negated.claim[:120]}… [{src_b[:60]}]"
                )
        return conflicts[:6]

    # ── synthesis ──────────────────────────────────────────────────────────
    def _synthesize(self, query: str, report: SwarmReport) -> str:
        findings = sorted(report.findings, key=lambda f: -f.confidence)[:12]
        if self._router_available():
            try:
                return self._model_synthesis(query, report, findings)
            except Exception as exc:  # noqa: BLE001
                _log.debug("model synthesis failed: %s", exc)
        return self._deterministic_synthesis(query, report, findings)

    def _model_synthesis(self, query: str, report: SwarmReport,
                         findings: list[SwarmFinding]) -> str:
        from ..llm.base import Message, SamplingParams

        payload = json.dumps(
            {"findings": [f.to_dict() for f in findings], "conflicts": report.conflicts},
            ensure_ascii=False)[:6000]
        prompt = (
            "You are the synthesis stage of a research swarm. Below are the "
            "verified findings (with sources and confidence) from parallel "
            "researchers on the goal. Write a concise synthesis (150-350 "
            "words, plain text, no markdown): the answer to the goal, the "
            "strongest evidence, and — explicitly — where the sources "
            "disagree and how confident you are. Do not invent facts that "
            "are not in the findings. Cite URLs inline where they matter.\n\n"
            f"Goal: {query}\n\nFindings:\n{payload}"
        )
        response = self.context.router.chat(
            [Message.user(prompt)], SamplingParams(temperature=0.3, max_tokens=500))
        text = (getattr(response, "text", "") or "").strip()
        if text and len(text) > 40:
            return text
        raise ValueError("model synthesis returned nothing usable")

    @staticmethod
    def _deterministic_synthesis(query: str, report: SwarmReport,
                                 findings: list[SwarmFinding]) -> str:
        lines = [f"Research swarm on {query!r}: {len(report.findings)} findings "
                 f"across {len(report.angles)} angle(s), "
                 f"{len(report.sources)} sources."]
        for f in findings[:5]:
            src = f.sources[0]["title"][:50] if f.sources else "web"
            lines.append(f"- [{f.confidence:.0%} via {src}] {f.claim[:220]}")
        if report.conflicts:
            lines.append("Sources disagree on: " + "; ".join(c[:120] for c in report.conflicts[:3]))
        if report.failed_angles:
            lines.append(f"(angles that failed: {', '.join(report.failed_angles)})")
        lines.append("(deterministic synthesis — no model was reachable)")
        return "\n".join(lines)

    # ── the run ────────────────────────────────────────────────────────────
    def run(self, query: str, *, angles: list[str] | None = None,
            save_memory: bool = False) -> SwarmReport:
        started = time.time()
        query = (query or "").strip()
        if not query:
            raise ValueError("swarm needs a query")
        angle_list = list(angles) if angles else self.angles_for(query)
        if angles:
            # caller-supplied angles: no known domains
            self._angle_domains = [""] * len(angle_list)
        report = SwarmReport(query=query, angles=angle_list, workers=self.workers)
        if not angle_list:
            report.synthesis = "no angles to research"
            report.seconds = time.time() - started
            return report

        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.workers,
                                thread_name_prefix="research-swarm") as pool:
            futures = {
                pool.submit(
                    self._research_angle, angle, i,
                    self._angle_domains[i]
                    if i < len(self._angle_domains) else ""): (i, angle)
                for i, angle in enumerate(angle_list)
            }
            for future, (i, angle) in futures.items():
                try:
                    results[i] = future.result(timeout=self.timeout)
                except FutureTimeout:
                    results[i] = {"angle": angle, "ok": False, "error": "timeout",
                                  "findings": [], "sources": []}
                except Exception as exc:  # noqa: BLE001 - one dead worker, not a dead swarm
                    _log.warning("swarm angle %r failed: %s", angle, exc)
                    results[i] = {"angle": angle, "ok": False, "error": str(exc),
                                  "findings": [], "sources": []}

        for i in range(len(angle_list)):
            result = results.get(i, {"ok": False, "error": "missing", "findings": []})
            if not result.get("ok"):
                report.failed_angles.append(result.get("angle", angle_list[i]))
                continue
            report.findings.extend(result.get("findings", []))
            report.sources.extend(result.get("sources", []))
        # de-duplicate sources across angles
        seen: set[str] = set()
        unique_sources: list[dict[str, Any]] = []
        for s in report.sources:
            key = str(s.get("url", ""))
            if key and key not in seen:
                seen.add(key)
                unique_sources.append(s)
        report.sources = unique_sources[:40]
        # cross-specialist dedup: one merged finding per distinct claim,
        # every source kept — before conflicts and synthesis see them
        report.findings = self.dedupe_findings(report.findings)
        report.conflicts = self._detect_conflicts(report.findings)
        report.synthesis = self._synthesize(query, report)
        report.critique = self._supervise(query, report)
        report.seconds = time.time() - started
        if save_memory:
            self.to_memory(report)
        return report

    def _supervise(self, query: str, report: SwarmReport) -> str:
        """Reasoning-Agent critique of the synthesis (best-effort)."""
        router = getattr(self.context, "router", None)
        if router is None or not hasattr(router, "chat"):
            return ""
        try:
            from .reasoning import ReasoningAgent

            verdict = ReasoningAgent(self.context).course_correct(
                report.synthesis, scope="swarm", attempts=1)
            report.supervised = True
            parts: list[str] = []
            if verdict.get("should_pivot"):
                parts.append("weak: " + str(verdict.get("reason", ""))[:160])
            if verdict.get("pivot"):
                parts.append("suggested pivot: " + str(verdict["pivot"])[:160])
            return " | ".join(parts) or "ok"
        except Exception as exc:  # noqa: BLE001 - supervision is a bonus
            _log.debug("swarm supervision failed: %s", exc)
            return ""

    # ── feeds ──────────────────────────────────────────────────────────────
    def to_memory(self, report: SwarmReport) -> int:
        """File the report into long-term memory. Returns items stored."""
        memory = getattr(self.context, "memory", None)
        if memory is None or not hasattr(memory, "remember"):
            return 0
        from ..memory.base import MemoryKind

        stored = 0
        try:
            memory.remember(
                f"Research swarm on {report.query!r}: {report.synthesis[:600]}",
                kind=MemoryKind.EPISODE, importance=0.7, source="research_swarm")
            stored += 1
            for f in sorted(report.findings, key=lambda x: -x.confidence)[:5]:
                src = f.sources[0]["url"] if f.sources else ""
                memory.remember(
                    f"[research: {report.query}] {f.claim} (source: {src})",
                    kind=MemoryKind.FACT, importance=0.5,
                    source=f"research_swarm:{f.angle[:32]}")
                stored += 1
        except Exception as exc:  # noqa: BLE001 - memory is best-effort
            _log.debug("swarm memory filing failed: %s", exc)
        return stored


# ── tool registration (main AI + sub-agents) ─────────────────────────────────

def register(registry: Any) -> None:
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "research_swarm",
        description=(
            "parallel research swarm: split the goal into angles, research "
            "them simultaneously (trusted sources first), flag conflicting "
            "claims, return a structured report with synthesis. Use for "
            "anything that needs more than one angle of evidence."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "goal": "str — the research goal",
            "workers": "int (optional, 1-6) — parallel researchers",
            "angles": "list|str (optional) — explicit angles; default auto-decompose",
            "save_memory": "bool (optional) — file the report into memory",
        },
    )
    def research_swarm(
        goal: str,
        workers: int = 0,
        angles: Any = None,
        save_memory: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        if isinstance(angles, str):
            angles = [a.strip() for a in angles.split(",") if a.strip()]
        swarm = ResearchSwarm(context, workers=workers or None)
        report = swarm.run(goal, angles=angles or None, save_memory=bool(save_memory))
        return report.to_dict()
