"""Hypothesis-driven diagnosis — never anchor on the first cause.

When something fails ("it didn't work", "no reply", "it crashed"), the
failure is diagnosed by generating MULTIPLE competing hypotheses across
different layers, testing EACH against evidence, and reporting which are
ruled out, which fit, and which are unresolved — with the evidence for
each. A single-cause claim without the alternatives considered is not a
diagnosis, it is a guess.

Layers covered (every diagnosis spans at least three):
  config, routing, provider, model, gating, regression, resource.

This module is deterministic and model-free: hypotheses come from layer
templates + error-signal templates, evidence comes from caller-supplied
checks (callables or verdict mappings). The LLM reasoning strategy
(``agents/reasoning.py`` ``_hypothesize``) is a different tool — it
reasons about open questions with the model. This one diagnoses
failures with checkable evidence.

Output shape (enforced by :func:`format_report`):
  symptom -> hypotheses (evidence for / against each) -> verdict +
  confidence -> next evidence to collect if unresolved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "Hypothesis", "Diagnosis", "LAYERS",
    "generate_hypotheses", "evaluate", "diagnose", "format_report",
]

LAYERS = ("config", "routing", "provider", "model", "gating",
          "regression", "resource")

# ── layer templates ──────────────────────────────────────────────────────
# Each template: (layer, claim, falsifier, test). {symptom} is the
# reported failure. The falsifier names what would prove this hypothesis
# wrong — a hypothesis without one is not checkable.

_LAYER_TEMPLATES: tuple[tuple[str, str, str, str], ...] = (
    ("config",
     "A config value is missing or wrong (env key, path, toggle, threshold), so {symptom}.",
     "The config values the path reads are present and correct, yet the symptom persists.",
     "Read the config the failing path actually reads; verify each value is present and valid."),
    ("routing",
     "The input/event is never routed to the handler that should process it, so {symptom}.",
     "The handler is invoked (logged/traced) for the failing input, yet the symptom persists.",
     "Trace one failing input end to end: does it reach the handler?"),
    ("provider",
     "An external provider (LLM, API, service) is down, erroring, or rate-limiting, so {symptom}.",
     "The provider answers healthy for a direct probe, yet the symptom persists.",
     "Probe the provider directly (health/status call) outside the failing path."),
    ("model",
     "The active model (LLM or voice) is wrong, failing, or cooling down, so {symptom}.",
     "A different healthy model on the same path produces the correct result, yet the symptom persists.",
     "Swap the active model for a known-healthy one and retry the failing input."),
    ("gating",
     "A gate (permission, mode, chat-mode, capability) silently drops the work, so {symptom}.",
     "The gate logs a pass for the failing input, yet the symptom persists.",
     "Check the gate decision for the failing input: pass, deny, or silent drop?"),
    ("regression",
     "A recent change broke a previously working path, so {symptom}.",
     "The path fails identically on the pre-change code, yet the symptom persists.",
     "Reproduce on the pre-change code (or bisect the recent changes)."),
    ("resource",
     "A resource limit is hit (RAM, disk, quota, rate limit, timeout), so {symptom}.",
     "Resources are well within limits during the failure, yet the symptom persists.",
     "Check RAM/disk/quota/rate-limit state at the moment of failure."),
)

# ── error-signal templates ───────────────────────────────────────────────
# Error text narrows the hypothesis space. Each entry: (regex, layer,
# claim, falsifier, test). Signal matches ADD a hypothesis; they never
# remove the cross-layer set — the point is competing hypotheses.

_ERROR_SIGNALS: tuple[tuple[str, str, str, str, str], ...] = (
    (r"ModuleNotFound|ImportError|No module named", "config",
     "A required dependency is missing or unimportable, so {symptom}.",
     "The dependency imports cleanly in the failing environment.",
     "Import the named module in the failing environment."),
    (r"timeout|timed out|TimeoutError|ConnectionError|refused|RemoteDisconnected", "provider",
     "The call times out or the connection is refused, so {symptom}.",
     "The same call succeeds with a longer timeout / from another host.",
     "Retry the call with timing; check provider status."),
    (r"denied|permission|401|403|unauthorized|forbidden", "gating",
     "The call is denied by permissions or gating, so {symptom}.",
     "The same call succeeds with the same credentials after the gate is bypassed in a test.",
     "Check the gate/permission decision for this call."),
    (r"KeyError|AttributeError|TypeError|IndexError|NoneType", "regression",
     "The code hits an unexpected shape (missing key/attr/None), so {symptom}.",
     "The same input succeeds on the pre-change code.",
     "Reproduce with the failing input; inspect the value at the crash line."),
    (r"MemoryError|OutOfMemory|disk full|No space left|quota|rate limit|429", "resource",
     "A resource limit (memory, disk, quota, rate limit) is hit, so {symptom}.",
     "Limits are not hit during a reproduced failure.",
     "Check memory/disk/quota at failure time."),
    (r"no reply|silent|dropped|ignored|no output|empty", "routing",
     "The input is silently dropped before reaching a handler, so {symptom}.",
     "The handler logs receipt of the input.",
     "Trace one failing input: does any handler log it?"),
)


@dataclass
class Hypothesis:
    """One competing explanation for a symptom."""

    id: str
    layer: str
    claim: str
    falsifier: str
    test: str
    status: str = "unresolved"  # ruled_out | fits | unresolved
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "layer": self.layer, "claim": self.claim,
            "falsifier": self.falsifier, "test": self.test,
            "status": self.status,
            "evidence_for": list(self.evidence_for),
            "evidence_against": list(self.evidence_against),
        }


@dataclass
class Diagnosis:
    symptom: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    verdict: str = "unresolved"
    confidence: float = 0.0
    next_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symptom": self.symptom,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "verdict": self.verdict,
            "confidence": self.confidence,
            "next_evidence": list(self.next_evidence),
        }


def generate_hypotheses(symptom: str, *, error: str = "",
                        source: str = "") -> list[Hypothesis]:
    """Generate competing hypotheses across layers for a symptom.

    Always spans at least three layers (the full layer set, minus
    duplicates), plus error-signal hypotheses when the error text
    matches a known signal. Every hypothesis carries its falsifier —
    a hypothesis without one is not checkable.
    """
    symptom = (symptom or "").strip() or "the reported failure"
    hyps: list[Hypothesis] = []
    seen_layers: set[str] = set()
    for i, (layer, claim, falsifier, test) in enumerate(_LAYER_TEMPLATES):
        hyps.append(Hypothesis(
            id=f"H{i + 1}",
            layer=layer,
            claim=claim.format(symptom=symptom),
            falsifier=falsifier,
            test=test,
        ))
        seen_layers.add(layer)
    # Error-signal hypotheses: additional, more specific candidates.
    # They share the layer of an existing hypothesis when the layer
    # already has one — no duplicate layers, but a sharper claim.
    error_text = f"{error or ''} {source or ''}"
    for pattern, layer, claim, falsifier, test in _ERROR_SIGNALS:
        if re.search(pattern, error_text, re.I):
            hyps.append(Hypothesis(
                id=f"H{len(hyps) + 1}",
                layer=layer,
                claim=claim.format(symptom=symptom),
                falsifier=falsifier,
                test=test,
            ))
    return hyps


def evaluate(hypotheses: list[Hypothesis], evidence: dict | None = None) -> list[Hypothesis]:
    """Test each hypothesis against evidence.

    ``evidence`` maps hypothesis id -> {"verdict": "supports"|"refutes"|"unknown",
    "detail": str} or a bare verdict string. A hypothesis is:
      - ruled_out when evidence refutes it,
      - fits when evidence supports it,
      - unresolved otherwise (no evidence, or evidence is inconclusive).
    Evidence detail is recorded verbatim on the hypothesis — the report
    shows why each verdict was reached, never just the verdict.
    """
    evidence = evidence or {}
    for h in hypotheses:
        entry = evidence.get(h.id)
        if entry is None:
            h.status = "unresolved"
            continue
        if isinstance(entry, str):
            verdict, detail = entry, ""
        else:
            verdict = str((entry or {}).get("verdict", "unknown")).lower()
            detail = str((entry or {}).get("detail", "") or "")
        if verdict in ("refutes", "refuted", "ruled_out", "ruled-out"):
            h.status = "ruled_out"
            if detail:
                h.evidence_against.append(detail)
        elif verdict in ("supports", "fits", "confirmed"):
            h.status = "fits"
            if detail:
                h.evidence_for.append(detail)
        else:
            h.status = "unresolved"
            if detail:
                h.evidence_for.append(f"inconclusive: {detail}")
    return hypotheses


def diagnose(symptom: str, *, evidence: dict | None = None,
             error: str = "", source: str = "",
             hypotheses: list[Hypothesis] | None = None) -> Diagnosis:
    """Full diagnosis: generate hypotheses, test each, report verdict.

    Verdict: the single fitting hypothesis with the strongest evidence
    (most evidence_for entries; ties broken by hypothesis order). When
    several fit, the verdict names the first and lists the rest as
    still-fitting — never silently picks one. When none fit, the verdict
    is "unresolved" and next_evidence names the test for each unresolved
    hypothesis (its falsifier test), so the owner knows exactly what
    evidence would settle it.
    """
    hyps = hypotheses if hypotheses is not None else generate_hypotheses(
        symptom, error=error, source=source)
    evaluate(hyps, evidence)
    fitting = [h for h in hyps if h.status == "fits"]
    unresolved = [h for h in hyps if h.status == "unresolved"]
    if fitting:
        # Strongest evidence first; stable order breaks ties.
        best = max(fitting, key=lambda h: len(h.evidence_for))
        verdict = best.claim
        if len(fitting) > 1:
            others = ", ".join(f"{h.id} ({h.layer})" for h in fitting if h is not best)
            verdict += f" [also still fitting: {others}]"
        # Confidence: fitting with evidence beats fitting without; a
        # single fitting hypothesis out of many tested is stronger than
        # several fitting (ambiguity lowers confidence).
        confidence = 0.8 if len(fitting) == 1 and best.evidence_for else 0.5
        if len(fitting) > 1:
            confidence = 0.5
    else:
        verdict = "unresolved"
        confidence = 0.0
    next_evidence = [f"{h.id} ({h.layer}): {h.test}" for h in unresolved]
    return Diagnosis(symptom=symptom, hypotheses=hyps, verdict=verdict,
                     confidence=confidence, next_evidence=next_evidence)


def format_report(diagnosis: Diagnosis) -> str:
    """The required shape: symptom -> hypotheses (for/against) -> verdict
    + confidence -> next evidence. Never a single-cause claim without the
    alternatives considered."""
    lines = [f"Symptom: {diagnosis.symptom}", "", "Hypotheses considered:"]
    for h in diagnosis.hypotheses:
        lines.append(f"  {h.id} [{h.layer}] {h.claim}")
        lines.append(f"    status: {h.status}")
        for e in h.evidence_for:
            lines.append(f"    for: {e}")
        for e in h.evidence_against:
            lines.append(f"    against: {e}")
        if h.status == "unresolved":
            lines.append(f"    falsifier test: {h.test}")
    ruled_out = [h.id for h in diagnosis.hypotheses if h.status == "ruled_out"]
    fitting = [h.id for h in diagnosis.hypotheses if h.status == "fits"]
    lines += [
        "",
        f"Ruled out: {', '.join(ruled_out) or 'none'}",
        f"Fitting: {', '.join(fitting) or 'none'}",
        f"Verdict: {diagnosis.verdict}",
        f"Confidence: {diagnosis.confidence:.2f}",
    ]
    if diagnosis.next_evidence:
        lines.append("Next evidence to collect:")
        for e in diagnosis.next_evidence:
            lines.append(f"  - {e}")
    return "\n".join(lines)
