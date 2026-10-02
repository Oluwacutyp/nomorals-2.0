"""Structured research knowledge + digest formats (wave C stream 1).

The ResearchSwarm (``research_swarm.py``) produces flat ``SwarmReport``s and
files them as flat memories via ``to_memory()``. This module is the
structured successor: each finding becomes a durable, versioned,
domain-tagged ``ResearchClaim`` node in the knowledge graph, conflicts are
detected against live graph state (never silently averaged away), and three
digest formats serve three consumers:

* ``operator_brief`` — chat-ready, hard-capped at 20 lines
* ``technical_note`` — structured plain-text note for the record
* ``upgrade_ticket`` — actionable ticket for the upgrade backlog

Pipeline: ``ResearchSwarm.run`` -> ``claim_from_finding`` ->
``check_conflicts`` -> ``promote`` -> ``ResearchDigest`` renderers,
orchestrated thinly by ``ResearchPipeline.run``.

Reuse, not duplication: the affirmation-vs-negation test imports
``_NEGATION`` and ``_content_words`` from ``research_swarm`` (the same test
the swarm's own conflict detector uses), and persistence goes through
``KnowledgeGraph`` from ``kg``. The domain keyword lists below are aligned
with the swarm's ``SPECIALIST_DOMAINS`` vocabulary but live here so the
classifier stays stable while the swarm's angle templates evolve.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .kg import KnowledgeGraph
from .research_swarm import (
    ResearchSwarm,
    SwarmFinding,
    SwarmReport,
    _content_words,
    _NEGATION,
)

_log = get_logger(__name__)

__all__ = [
    "ResearchClaim",
    "classify_domain",
    "claim_from_finding",
    "check_conflicts",
    "promote",
    "ResearchDigest",
    "ResearchPipeline",
    "register",
]

_CLAIM_STATUSES = ("active", "superseded", "contradicted")

# ── domain classification ──────────────────────────────────────────────────
#: Priority order for first-match-wins. Rationale: security vocabulary
#: (exploit, CVE, ransomware) is high-signal and rare, so it wins first;
#: competitor language ("vs", "alternative") disambiguates comparisons
#: before product words ("feature", "pricing") can claim them; ml before
#: systems because "training/inference" rarely means anything else; systems
#: before tooling because kernel/server/container terms are stronger than
#: generic build/CI words; tooling before product because dev-workflow
#: terms are more specific; product last because "user/feature" are weak
#: signals that would otherwise shadow everything.
_DOMAIN_PRIORITY = ("security", "competitors", "ml", "systems", "tooling",
                    "product")

#: Keyword sets per domain (14+ each). Matched whole-token, case-insensitive;
#: hyphenated entries also match their space-separated form ("threat-model"
#: hits "threat model"). Anything unmatched falls through to "general".
_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "security": (
        "vulnerability", "vulnerabilities", "exploit", "exploits", "cve",
        "ransomware", "malware", "phishing", "breach", "breaches",
        "firewall", "encryption", "encrypted", "zero-day", "zeroday",
        "zero-trust", "threat-model", "threat", "intrusion",
        "authentication", "authorization", "access-control", "backdoor",
        "pentest", "hardening", "sandboxing", "audit", "xss", "csrf",
        "injection", "sqli", "ddos", "botnet", "spyware", "rootkit",
    ),
    "competitors": (
        "competitor", "competitors", "rival", "rivals", "versus", "vs",
        "alternative", "alternatives", "comparison", "compare", "compared",
        "market-share", "landscape", "differentiation", "incumbent",
        "disruptor", "disruptors", "vendor", "vendors", "open-source",
        "instead-of", "replaces", "replacement", "migration", "migrated",
        "switched", "head-to-head", "bake-off", "shootout",
    ),
    "ml": (
        "neural-network", "neural", "transformer", "transformers",
        "training", "trained", "fine-tuning", "finetune", "hyperparameters",
        "benchmark", "benchmarks", "inference", "quantization", "quantized",
        "embedding", "embeddings", "dataset", "datasets", "overfitting",
        "llm", "llms", "gradient", "gradients", "epoch", "epochs",
        "attention", "regularization", "classification", "regression",
        "tokenizer", "tokens", "alignment", "hallucination", "rag",
    ),
    "systems": (
        "distributed", "architecture", "architectures", "scalability",
        "scalable", "reliability", "reliable", "latency", "throughput",
        "fault-tolerance", "fault-tolerant", "consensus", "sharding",
        "sharded", "caching", "cache", "load-balancing", "load-balancer",
        "observability", "kernel", "kernels", "filesystem", "scheduler",
        "scheduling", "replication", "replica", "failover", "container",
        "containers", "orchestration", "cluster", "clusters", "syscall",
        "virtualization",
    ),
    "tooling": (
        "cli", "ci/cd", "build", "builds", "pipeline", "pipelines",
        "automation", "automated", "debugging", "debugger", "profiling",
        "profiler", "linting", "linter", "testing", "packaging", "package",
        "dx", "developer-experience", "sdk", "sdks", "api", "apis", "ide",
        "ides", "git", "compiler", "compilers", "interpreter", "workflow",
        "workflows", "deploy", "deployment", "monitoring", "logging",
    ),
    "product": (
        "product-market-fit", "user-research", "roadmap", "ux", "onboarding",
        "retention", "activation", "conversion", "pricing", "positioning",
        "churn", "engagement", "customer", "customers", "stakeholder",
        "mvp", "funnel", "saas", "acquisition", "monetization", "freemium",
        "upsell", "persona", "personas", "a/b-test", "cohort",
    ),
}


def _keyword_hit(keyword: str, text: str) -> bool:
    """Whole-token, case-insensitive match; hyphenated keywords also match
    their space-separated form."""
    kw = (keyword or "").strip().lower()
    if not kw:
        return False
    variants = {kw, kw.replace("-", " ")}
    for variant in variants:
        if re.search(r"(?<![a-z0-9])" + re.escape(variant) + r"(?![a-z0-9])",
                      text, re.IGNORECASE):
            return True
    return False


def classify_domain(claim_text: str) -> str:
    """Route a claim to one of the swarm's specialist domains
    (systems/security/product/ml/tooling/competitors), first-match-wins in
    ``_DOMAIN_PRIORITY`` order; "general" when nothing hits."""
    text = claim_text or ""
    for domain in _DOMAIN_PRIORITY:
        for keyword in _DOMAIN_KEYWORDS[domain]:
            if _keyword_hit(keyword, text):
                return domain
    return "general"


# ── the durable claim record ───────────────────────────────────────────────

@dataclass
class ResearchClaim:
    """One research claim as a durable, versioned record.

    ``status`` is one of "active" | "superseded" | "contradicted".
    ``supersedes`` holds the older claim's id when this claim replaced it
    ("" otherwise). ``sources`` are {"url","title","trust"} dicts.
    """

    id: str
    claim: str
    domain: str = "general"
    angle: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.3
    fetched_at: float = 0.0
    query: str = ""
    status: str = "active"
    supersedes: str = ""
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "domain": self.domain,
            "angle": self.angle,
            "sources": self.sources,
            "confidence": round(float(self.confidence), 3),
            "fetched_at": self.fetched_at,
            "query": self.query,
            "status": self.status,
            "supersedes": self.supersedes,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchClaim":
        data = data or {}
        return cls(
            id=str(data.get("id", "")),
            claim=str(data.get("claim", "")),
            domain=str(data.get("domain", "general") or "general"),
            angle=str(data.get("angle", "")),
            sources=list(data.get("sources", []) or []),
            confidence=float(data.get("confidence", 0.3) or 0.0),
            fetched_at=float(data.get("fetched_at", 0.0) or 0.0),
            query=str(data.get("query", "")),
            status=str(data.get("status", "active") or "active"),
            supersedes=str(data.get("supersedes", "")),
            version=int(data.get("version", 1) or 1),
        )


def claim_from_finding(finding: SwarmFinding, query: str) -> ResearchClaim:
    """Convert a ``SwarmFinding`` into a durable claim: fresh id, classified
    domain, timestamp, version 1, status active."""
    sources = []
    for s in finding.sources or []:
        if not isinstance(s, dict):
            continue
        url = str(s.get("url", "") or "")
        if not url:
            continue
        try:
            trust = float(s.get("trust", 0.5))
        except (TypeError, ValueError):
            trust = 0.5
        sources.append({"url": url,
                        "title": str(s.get("title", "") or ""),
                        "trust": trust})
    try:
        confidence = float(finding.confidence)
    except (TypeError, ValueError):
        confidence = 0.3
    return ResearchClaim(
        id=new_id("claim"),
        claim=str(finding.claim or ""),
        domain=classify_domain(str(finding.claim or "")),
        angle=str(finding.angle or ""),
        sources=sources,
        confidence=confidence,
        fetched_at=time.time(),
        query=str(query or ""),
        status="active",
        supersedes="",
        version=1,
    )


# ── conflict detection against the knowledge graph ─────────────────────────

def _contradicts(a: str, b: str) -> bool:
    """The swarm's affirmation-vs-negation test on a claim pair: the same
    subject (>= 2 shared content words of length >= 4) with opposite
    negation polarity. Deliberately conservative — emphasis differences do
    not count."""
    shared = {w for w in (_content_words(a) & _content_words(b)) if len(w) >= 4}
    if len(shared) < 2:
        return False
    return bool(_NEGATION.search(a or "")) != bool(_NEGATION.search(b or ""))


def check_conflicts(claims: list[ResearchClaim],
                    kg: KnowledgeGraph) -> list[dict[str, Any]]:
    """Flag claims that contradict ACTIVE claim nodes already in the KG.

    Candidates come from ``kg.recall`` (graph-text overlap); each is run
    through the shared affirmation-vs-negation test. Flag only — the graph
    is never merged or modified here.

    Returns a list of {"claim_id", "existing_id" (KG node id — what the
    supersede path needs), "existing_label", "existing_claim",
    "existing_claim_id", "existing_confidence", "kind": "contradiction"}.
    """
    out: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for claim in claims or []:
        text = (claim.claim or "").strip()
        if not text:
            continue
        try:
            candidates = kg.recall(text, limit=12)
        except Exception as e:  # noqa: BLE001 - recall is best-effort
            _log.warning("conflict recall failed for claim %s: %s",
                         claim.id, e)
            continue
        for node in candidates:
            if node.get("type") != "claim":
                continue
            props = node.get("properties") or {}
            if str(props.get("status", "active")) != "active":
                continue  # only live claims can be contradicted
            existing_text = str(props.get("claim", "") or "").strip()
            if not existing_text:
                continue
            label = str(node.get("label", "") or "")
            if label == f"claim:{claim.id}":
                continue  # never conflict with yourself
            node_id = str(node.get("id", "") or "")
            pair = (claim.id, node_id)
            if pair in seen_pairs:
                continue
            if _contradicts(text, existing_text):
                seen_pairs.add(pair)
                try:
                    old_conf = float(props.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    old_conf = 0.0
                out.append({
                    "claim_id": claim.id,
                    "existing_id": node_id,
                    "existing_label": label,
                    "existing_claim": existing_text,
                    "existing_claim_id": str(
                        props.get("claim_id", "") or label.split(":", 1)[-1]),
                    "existing_confidence": old_conf,
                    "kind": "contradiction",
                })
    return out


# ── promotion into the knowledge graph ─────────────────────────────────────

def promote(claims: list[ResearchClaim], db: Any, *,
            min_confidence: float = 0.5) -> dict[str, Any]:
    """Durably promote claims into the knowledge graph.

    * claims below ``min_confidence`` are skipped (counted, reported).
    * each claim becomes a ``claim:<id>`` node (type "claim") carrying the
      full provenance: claim text, domain, angle, confidence, fetched_at,
      originating query, source URLs, version, status.
    * ``domain:<name>`` node (type "domain"), linked claim -> domain
      "about" (weight 1.5).
    * one ``source:<url>`` node (type "source") per cited URL, linked
      claim -> source "cites" (weight 1.0).
    * supersede: when a new claim contradicts an existing ACTIVE claim and
      the new confidence >= the old, the old node is marked
      status="superseded" (+ ``superseded_by`` = new claim id) and a
      new -> old "supersedes" edge is added. Nodes are never deleted.
    * when a new claim contradicts a *stronger* active claim, the new node
      is still stored but marked status="contradicted" (+ ``contradicts``).

    Returns {"promoted", "skipped_low_confidence", "superseded",
    "conflicts" (from check_conflicts)}.
    """
    kg = KnowledgeGraph(db)
    result: dict[str, Any] = {"promoted": 0, "skipped_low_confidence": 0,
                              "superseded": 0, "conflicts": []}
    for claim in claims or []:
        try:
            confidence = float(claim.confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < float(min_confidence):
            result["skipped_low_confidence"] += 1
            continue
        conflicts = check_conflicts([claim], kg)
        result["conflicts"].extend(conflicts)

        # decide supersede vs contradicted before writing the node
        superseded_ids: list[str] = []
        contradicted_by = ""
        for c in conflicts:
            old_id = str(c.get("existing_claim_id", "") or "")
            if confidence >= float(c.get("existing_confidence", 0.0) or 0.0):
                superseded_ids.append(old_id)
                if c.get("existing_label"):
                    # merge, never clobber: upsert_node unions properties
                    kg.upsert_node(str(c["existing_label"]), type="claim",
                                   properties={
                                       "status": "superseded",
                                       "superseded_by": claim.id,
                                       "superseded_at": time.time(),
                                   })
            elif not contradicted_by:
                contradicted_by = old_id
        if superseded_ids:
            claim.supersedes = superseded_ids[0]
            result["superseded"] += len(superseded_ids)
        if contradicted_by and not superseded_ids:
            claim.status = "contradicted"

        props = {
            "claim_id": claim.id,
            "claim": claim.claim,
            "domain": claim.domain,
            "angle": claim.angle,
            "confidence": round(confidence, 3),
            "fetched_at": claim.fetched_at,
            "query": claim.query,
            "sources": json.dumps(
                [s["url"] for s in claim.sources if s.get("url")]),
            "version": claim.version,
            "status": claim.status,
            "supersedes": claim.supersedes,
        }
        if contradicted_by and not superseded_ids:
            props["contradicts"] = contradicted_by
        node = kg.upsert_node(f"claim:{claim.id}", type="claim",
                              properties=props)

        domain_node = kg.upsert_node(f"domain:{claim.domain}",
                                     type="domain",
                                     properties={"name": claim.domain})
        kg.link(node.id, domain_node.id, "about", weight=1.5)

        for s in claim.sources[:10]:
            url = str(s.get("url", "") or "").strip()
            if not url:
                continue
            try:
                trust = float(s.get("trust", 0.5))
            except (TypeError, ValueError):
                trust = 0.5
            src_node = kg.upsert_node(
                f"source:{url}", type="source",
                properties={"url": url, "title": str(s.get("title", "") or ""),
                            "trust": trust})
            kg.link(node.id, src_node.id, "cites", weight=1.0)

        for c in conflicts:
            if not c.get("existing_id"):
                continue
            if confidence >= float(c.get("existing_confidence", 0.0) or 0.0):
                kg.link(node.id, str(c["existing_id"]), "supersedes",
                        weight=2.0)
        result["promoted"] += 1
    return result


# ── digests ────────────────────────────────────────────────────────────────

#: Domain -> repo areas that own that kind of work. HEURISTIC, labeled as
#: suggestions in every ticket — verify the owning module before editing.
_SUGGESTED_FILES: dict[str, list[str]] = {
    "security": ["nomorals/core/policy.py", "nomorals/core/trust.py",
                 "nomorals/core/ids.py"],
    "systems": ["nomorals/core/logging_setup.py", "nomorals/native/",
                "nomorals/storage/"],
    "product": ["nomorals/cli.py", "nomorals/api/", "nomorals/tui/",
                "nomorals/partner/"],
    "ml": ["nomorals/llm/base.py", "nomorals/training/", "nomorals/ta/"],
    "tooling": ["nomorals/tools/browser.py", "nomorals/builders/",
                "nomorals/cli.py"],
    "competitors": ["nomorals/agents/research_swarm.py",
                    "nomorals/agents/opportunities.py"],
    "general": ["nomorals/agents/research_swarm.py"],
}

#: Domain -> what the ticket's test plan should stress.
_DOMAIN_TEST_HINTS: dict[str, str] = {
    "security": "adversarial inputs, auth bypass attempts, and secret leakage",
    "systems": "concurrency, failure injection, and resource exhaustion",
    "product": "the owner-facing flow end to end, including empty states",
    "ml": "deterministic small-model fixtures plus a real-model smoke test",
    "tooling": "CLI arg parsing, exit codes, and idempotent re-runs",
    "competitors": "the comparison matrix staying current and sourced",
    "general": "the happy path plus one realistic failure mode",
}


class ResearchDigest:
    """The three digest formats. Pure renderers — no I/O, no network."""

    @staticmethod
    def operator_brief(report_or_claims: Any,
                       min_confidence: float = 0.0) -> str:
        """Chat-ready brief, HARD-capped at 20 lines: synthesis, top
        findings, conflicts, gaps. Returns "" when there is nothing to say.
        Accepts a ``SwarmReport`` or a list of claims/findings.

        ``min_confidence`` filters the findings: weak claims below the bar
        are padding, not signal — the owner's channel is not the place for
        them.
        """
        if isinstance(report_or_claims, SwarmReport):
            report = report_or_claims
            query = report.query or ""
            synthesis = (report.synthesis or "").strip()
            findings = sorted(
                (f for f in report.findings
                 if float(getattr(f, "confidence", 0.0) or 0.0)
                 >= float(min_confidence)),
                key=lambda f: -float(f.confidence or 0.0))
            conflicts = [str(c) for c in (report.conflicts or [])]
            failed = list(report.failed_angles or [])
        elif isinstance(report_or_claims, (list, tuple)):
            query, synthesis = "", ""
            findings = sorted(
                (f for f in list(report_or_claims)
                 if float(getattr(f, "confidence", 0.0) or 0.0)
                 >= float(min_confidence)),
                key=lambda f: -float(getattr(f, "confidence", 0.0) or 0.0))
            conflicts, failed = [], []
        else:
            return ""
        if not synthesis and not findings and not conflicts:
            return ""
        lines: list[str] = []
        lines.append(f"Research brief: {query}" if query else "Research brief")
        if synthesis:
            lines.extend(synthesis.splitlines())
        if findings:
            lines.append("Top findings:")
            for f in findings[:6]:
                text = str(getattr(f, "claim", "") or "").strip()
                try:
                    conf = float(getattr(f, "confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                src = ""
                sources = getattr(f, "sources", None) or []
                if sources and isinstance(sources[0], dict):
                    src = str(sources[0].get("title")
                              or sources[0].get("url") or "")[:40]
                line = f"- [{conf:.0%}] {text[:140]}"
                if src:
                    line += f" ({src})"
                lines.append(line)
        if conflicts:
            lines.append("Conflicts:")
            for c in conflicts[:4]:
                lines.append(f"- {c[:140]}")
        if failed:
            lines.append("Gaps: %d angle(s) failed (%s)"
                         % (len(failed), ", ".join(failed[:3])))
        # the hard cap: never more than 20 lines, whatever came in
        return "\n".join("\n".join(lines).splitlines()[:20])

    @staticmethod
    def technical_note(report: SwarmReport) -> str:
        """Structured plain-text note: Objective, Key findings (confidence +
        source URLs), Conflicts, Gaps/failed angles, Critique. No markdown
        tables — readable in chat and in logs."""
        lines: list[str] = []
        lines.append(f"RESEARCH NOTE: {report.query or '(no query)'}")
        lines.append("")
        lines.append("OBJECTIVE")
        lines.append(report.query or "(no query)")
        lines.append("")
        lines.append("KEY FINDINGS")
        if report.findings:
            for i, f in enumerate(
                    sorted(report.findings,
                           key=lambda x: -float(x.confidence or 0.0)), 1):
                urls = ", ".join(
                    str(s.get("url", ""))
                    for s in (f.sources or [])
                    if isinstance(s, dict) and s.get("url"))
                lines.append(f"{i}. [{float(f.confidence or 0.0):.0%}] "
                             f"{f.claim}")
                detail = f"   angle: {f.angle or '(unspecified)'}"
                if urls:
                    detail += f" | sources: {urls}"
                lines.append(detail)
        else:
            lines.append("(none)")
        lines.append("")
        lines.append("CONFLICTS")
        if report.conflicts:
            lines.extend(f"- {c}" for c in report.conflicts)
        else:
            lines.append("(none)")
        lines.append("")
        lines.append("GAPS / FAILED ANGLES")
        if report.failed_angles:
            lines.extend(f"- {a}" for a in report.failed_angles)
        else:
            lines.append("(none)")
        lines.append("")
        lines.append("CRITIQUE")
        lines.append((report.critique or "").strip()
                     or "(no critique — unsupervised run)")
        return "\n".join(lines)

    @staticmethod
    def upgrade_ticket(claim: ResearchClaim) -> dict[str, Any]:
        """Build an actionable upgrade ticket from a claim.

        ``suggested_files`` is a heuristic domain -> repo-area map, labeled
        as suggestions (see ``suggested_files_basis``) — verify the owning
        module before editing; it is not certainty.
        """
        domain = claim.domain if claim.domain in _SUGGESTED_FILES else "general"
        short = claim.claim[:90].rstrip()
        if len(claim.claim) > 90:
            short += "..."
        hint = _DOMAIN_TEST_HINTS.get(domain, _DOMAIN_TEST_HINTS["general"])
        return {
            "title": f"[{domain}] {short}",
            "rationale": (
                f"Research finding from query {claim.query!r} "
                f"(angle: {claim.angle or 'unspecified'}, "
                f"confidence {float(claim.confidence or 0.0):.0%}): "
                f"{claim.claim}"),
            "domain": domain,
            "suggested_files": list(_SUGGESTED_FILES[domain]),
            "suggested_files_basis": (
                "heuristic: keyword-mapped domain -> repo area; verify the "
                "owning module before editing — suggestions, not certainty"),
            "test_plan": [
                f"Encode the claim as a test: assert the behavior '{short}'.",
                "Add a negative test for the contradicting case (check the "
                "KG for a flagged conflict on this claim first).",
                f"Stress the domain edge cases: {hint}.",
                "Run the owning module's test suite green before and after.",
            ],
            "patch_plan": [
                "Confirm the owning module from suggested_files (read it; "
                "do not assume the heuristic is right).",
                "Write the failing test from the test plan first.",
                "Implement the smallest change that makes the test pass.",
                "Run the module tests plus error_scan; fix regressions.",
                "Promote the outcome back to the KG (research_digest "
                "promote) with the new confidence.",
            ],
            "claim_ids": [claim.id],
            "confidence": round(float(claim.confidence or 0.0), 3),
        }


# ── the thin pipeline ──────────────────────────────────────────────────────

class ResearchPipeline:
    """Thin orchestration: the swarm does the fetching, this module does
    the structuring. ``run`` returns the full dict; the ``research_digest``
    tool selects which digests to surface."""

    @staticmethod
    def run(query: str, context: Any, *,
            specialists: list[str] | None = None,
            min_confidence: float = 0.5,
            notify: bool = False,
            promote_claims: bool = True) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            raise ValueError("pipeline needs a query")
        swarm = ResearchSwarm(context, specialists=specialists)
        report = swarm.run(query)
        claims = [claim_from_finding(f, query) for f in report.findings]

        db = getattr(context, "db", None)
        if promote_claims and db is not None:
            promotion = promote(claims, db, min_confidence=min_confidence)
            conflicts = promotion["conflicts"]
        else:
            kg = KnowledgeGraph(db) if db is not None else None
            conflicts = check_conflicts(claims, kg) if kg is not None else []
            promotion = {"promoted": 0, "skipped_low_confidence": 0,
                         "superseded": 0, "conflicts": conflicts,
                         "skipped_reason": ("no db on context" if db is None
                                            else "promote_claims=False")}

        brief = ResearchDigest.operator_brief(report,
                                              min_confidence=min_confidence)
        note = ResearchDigest.technical_note(report)
        tickets = [ResearchDigest.upgrade_ticket(c) for c in claims
                   if float(c.confidence or 0.0) >= float(min_confidence)]

        notified = False
        if notify and brief:
            try:
                from .notifier import notify as _send

                res = _send(context, "research",
                            f"Research done: {query[:60]}", brief)
                notified = bool(isinstance(res, dict)
                                and res.get("delivered"))
            except Exception as e:  # noqa: BLE001 - notify is best-effort
                _log.warning("research pipeline notify failed: %s", e)

        return {
            "query": query,
            "report": report.to_dict(),
            "claims": [c.to_dict() for c in claims],
            "conflicts": conflicts,
            "promotion": promotion,
            "brief": brief,
            "note": note,
            "tickets": tickets,
            "notified": notified,
        }


# ── tool registration (main AI + sub-agents) ─────────────────────────────────

def register(registry: Any) -> None:
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "research_digest",
        description=(
            "structured research knowledge: run a research swarm, convert "
            "findings to durable versioned claims in the knowledge graph "
            "(with conflict detection and supersede), and render digests — "
            "a chat-ready brief, a technical note, and actionable upgrade "
            "tickets. digest=brief|note|tickets|all."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "goal": "str — the research goal",
            "specialists": "list|str (optional) — specialist domains "
                           "(systems/security/product/ml/tooling/competitors); "
                           "default auto-decompose",
            "promote": "bool (optional, default true) — file claims into "
                       "the knowledge graph",
            "digest": "str (optional: brief|note|tickets|all, default all)",
            "min_confidence": "float (optional, default 0.5) — minimum "
                              "claim confidence for promotion/tickets",
            "notify": "bool (optional) — send the brief to the owner",
        },
    )
    def research_digest(
        goal: str,
        specialists: Any = None,
        promote: bool = True,
        digest: str = "all",
        min_confidence: float = 0.5,
        notify: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        try:
            if isinstance(specialists, str):
                specialists = [s.strip() for s in specialists.split(",")
                               if s.strip()]
            try:
                mc = float(min_confidence)
            except (TypeError, ValueError):
                mc = 0.5
            result = ResearchPipeline.run(
                goal, context,
                specialists=specialists or None,
                min_confidence=mc,
                notify=bool(notify),
                promote_claims=bool(promote),
            )
        except Exception as e:  # noqa: BLE001 - tool boundary returns errors
            _log.warning("research_digest tool failed: %s", e)
            return {"ok": False, "error": str(e)}
        which = (digest or "all").strip().lower()
        out: dict[str, Any] = {
            "ok": True,
            "query": result["query"],
            "claims": result["claims"],
            "conflicts": result["conflicts"],
            "promotion": result["promotion"],
            "notified": result["notified"],
        }
        if which in ("brief", "all"):
            out["brief"] = result["brief"]
        if which in ("note", "all"):
            out["note"] = result["note"]
        if which in ("tickets", "all"):
            out["tickets"] = result["tickets"]
        return out
