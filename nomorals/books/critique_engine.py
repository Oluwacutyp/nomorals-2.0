"""Critique engine (L3) — generalized from books/collab.py.

Finding 3 fix: critique existed in three costumes (books collab.py
writer-critic, media critique engine, music critique loop). This is the
ONE engine; every creative organ is a client.

Model: named dimensions, each with a score + evidence, combined by
weights into an overall score, judged against an explicit quality floor.
``verdict`` is derived from the floor, not vibes:

- overall >= floor and no error-severity continuity finding → ``ship``
- overall >= floor - 0.15 → ``revise``
- otherwise → ``rewrite``

Checks are callables ``(text, context) -> DimensionScore``. Built-ins are
pure-Python heuristics over prose/continuity; a caller may register a
model-backed check with the same signature. ``to_collab()`` converts a
report back to the collab.py ``Critique`` shape so existing clients keep
working without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .continuity import ContinuityEngine, ContinuityState
from .prose import analyze as prose_analyze

__all__ = ["DimensionScore", "CritiqueReport", "CritiqueEngine",
           "DEFAULT_FLOOR"]

DEFAULT_FLOOR = 0.6


@dataclass
class DimensionScore:
    dimension: str
    score: float  # 0..1
    evidence: list[str] = field(default_factory=list)
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"dimension": self.dimension,
                "score": round(max(0.0, min(1.0, self.score)), 4),
                "evidence": self.evidence, "weight": self.weight}


@dataclass
class CritiqueReport:
    overall: float
    dimensions: list[DimensionScore]
    strengths: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    verdict: str = "revise"  # ship | revise | rewrite
    floor: float = DEFAULT_FLOOR
    floor_passed: bool = False
    continuity_findings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 4),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "strengths": self.strengths, "issues": self.issues,
            "suggestions": self.suggestions, "verdict": self.verdict,
            "floor": self.floor, "floor_passed": self.floor_passed,
            "continuity_findings": self.continuity_findings,
        }

    def to_collab(self) -> dict[str, Any]:
        """collab.py ``Critique`` shape (score/strengths/issues/
        suggestions/verdict) — no import of collab needed."""
        return {"score": round(self.overall, 4),
                "strengths": self.strengths, "issues": self.issues,
                "suggestions": self.suggestions, "verdict": self.verdict}


CheckFn = Callable[[str, dict[str, Any]], DimensionScore]


class CritiqueEngine:
    """Checklist + named scores + quality floor, organ-agnostic."""

    def __init__(self, *, floor: float = DEFAULT_FLOOR) -> None:
        if not 0.0 <= floor <= 1.0:
            raise ValueError(f"floor must be in [0, 1], got {floor!r}")
        self.floor = floor
        self._checks: dict[str, tuple[CheckFn, float]] = {}
        for name, weight in (("prose", 1.0), ("pacing", 0.8),
                             ("dialogue", 0.6), ("hook", 0.7),
                             ("structure", 0.8), ("continuity", 1.0)):
            self.register(name, getattr(self, f"_check_{name}"), weight)

    def register(self, name: str, fn: CheckFn, weight: float = 1.0) -> None:
        if not callable(fn):
            raise TypeError("critique check must be callable")
        self._checks[name] = (fn, weight)

    def review(self, text: str, *, context: dict[str, Any] | None = None,
               floor: float | None = None) -> CritiqueReport:
        ctx = dict(context or {})
        ctx.setdefault("text", text or "")
        dimensions: list[DimensionScore] = []
        continuity: list[dict[str, Any]] = []
        for name, (fn, weight) in self._checks.items():
            try:
                dim = fn(text or "", ctx)
            except Exception as exc:  # noqa: BLE001 - one bad check
                # must not sink the whole review; it scores 0 with evidence
                dim = DimensionScore(dimension=name, score=0.0,
                                     evidence=[f"check failed: {exc}"])
            dim.weight = weight
            dimensions.append(dim)
            if name == "continuity":
                continuity = [dict(e) for e in
                              ctx.get("_continuity_findings", [])]
        total_w = sum(d.weight for d in dimensions) or 1.0
        overall = sum(d.score * d.weight for d in dimensions) / total_w
        use_floor = self.floor if floor is None else floor
        has_error = any(f.get("severity") == "error" for f in continuity)
        if overall >= use_floor and not has_error:
            verdict = "ship"
        elif overall >= use_floor - 0.15:
            verdict = "revise"
        else:
            verdict = "rewrite"
        strengths = [f"{d.dimension}: strong ({d.score:.2f})"
                     for d in dimensions if d.score >= 0.8]
        issues: list[str] = []
        suggestions: list[str] = []
        for d in dimensions:
            for ev in d.evidence:
                if d.score < 0.6:
                    issues.append(f"[{d.dimension}] {ev}")
        if has_error:
            issues.append("continuity error — a dead/retired entity "
                          "appears, or a rule is violated")
            suggestions.append("fix the continuity error before shipping")
        for d in dimensions:
            if d.score < 0.6:
                suggestions.append(f"strengthen {d.dimension} "
                                   f"(now {d.score:.2f})")
        return CritiqueReport(
            overall=round(overall, 4), dimensions=dimensions,
            strengths=strengths, issues=issues, suggestions=suggestions,
            verdict=verdict, floor=use_floor,
            floor_passed=overall >= use_floor and not has_error,
            continuity_findings=continuity)

    # ── built-in checks (pure Python, no model) ──────────────────────
    @staticmethod
    def _check_prose(text: str, ctx: dict[str, Any]) -> DimensionScore:
        rep = prose_analyze(text)
        ev: list[str] = []
        penalties = 0.0
        if rep.filter_density > 0.015:
            ev.append(f"filter-word heavy ({rep.filter_density:.3f})")
            penalties += 0.15
        if rep.adverb_density > 0.04:
            ev.append(f"adverb-dense ({rep.adverb_density:.3f})")
            penalties += 0.10
        if rep.passive_density > 0.02:
            ev.append(f"passive-heavy ({rep.passive_density:.3f})")
            penalties += 0.08
        if rep.pov_drift:
            ev.append("POV drift between halves")
            penalties += 0.12
        if rep.tense_drift:
            ev.append("tense drift between halves")
            penalties += 0.12
        if not ev:
            ev.append("clean prose scan: no filter/adverb/passive/drift flags")
        return DimensionScore(dimension="prose",
                              score=max(0.0, 1.0 - penalties), evidence=ev)

    @staticmethod
    def _check_pacing(text: str, ctx: dict[str, Any]) -> DimensionScore:
        rep = prose_analyze(text)
        ev = [f"sentence-length CV {rep.sentence_len_cv:.2f} "
              f"(mean {rep.sentence_len_mean:.1f} words)"]
        # healthy prose varies: CV < 0.25 is monotonous, > 1.2 is erratic
        if rep.sentence_len_cv < 0.25:
            return DimensionScore(dimension="pacing", score=0.45,
                                  evidence=ev + ["sentence lengths too "
                                                 "uniform — vary them"])
        if rep.sentence_len_cv > 1.2:
            return DimensionScore(dimension="pacing", score=0.6,
                                  evidence=ev + ["sentence lengths erratic"])
        return DimensionScore(dimension="pacing", score=0.85, evidence=ev)

    @staticmethod
    def _check_dialogue(text: str, ctx: dict[str, Any]) -> DimensionScore:
        rep = prose_analyze(text)
        ratio = rep.dialogue_ratio
        if ratio == 0.0:
            return DimensionScore(dimension="dialogue", score=0.4,
                                  evidence=["no dialogue in this passage"])
        if ratio > 0.7:
            return DimensionScore(dimension="dialogue", score=0.6,
                                  evidence=[f"dialogue ratio {ratio:.2f} — "
                                            "mostly talking heads"])
        return DimensionScore(dimension="dialogue", score=0.85,
                              evidence=[f"dialogue ratio {ratio:.2f}"])

    @staticmethod
    def _check_hook(text: str, ctx: dict[str, Any]) -> DimensionScore:
        rep = prose_analyze(text)
        hook = rep.hook
        if hook["kind"] in ("question", "revelation", "cliffhanger"):
            score = 0.9
        elif hook["kind"] in ("threat", "arrival", "deferred_decision"):
            score = 0.75
        elif hook["kind"] == "trailing":
            score = 0.45
        else:
            score = 0.3
        return DimensionScore(
            dimension="hook", score=score,
            evidence=[f"chapter-end hook: {hook['kind']} — {hook['note']}"])

    @staticmethod
    def _check_structure(text: str, ctx: dict[str, Any]) -> DimensionScore:
        rep = prose_analyze(text)
        ev: list[str] = []
        score = 0.7
        if rep.words < 200:
            ev.append(f"very short ({rep.words} words) — scene, not chapter")
            score -= 0.25
        if rep.paragraphs < 3:
            ev.append("fewer than 3 paragraphs — no scene shape")
            score -= 0.15
        if not ev:
            ev.append(f"{rep.words} words, {rep.paragraphs} paragraphs — "
                      "chapter-shaped")
        return DimensionScore(dimension="structure",
                              score=max(0.0, score), evidence=ev)

    @staticmethod
    def _check_continuity(text: str, ctx: dict[str, Any]) -> DimensionScore:
        state = ctx.get("continuity_state") or ctx.get("bible_state")
        bible = ctx.get("bible")
        findings: list[dict[str, Any]] = []
        if state is not None:
            if not isinstance(state, ContinuityState):
                engine = ContinuityEngine()
                state = engine.from_story_bible(state)
            chapter = int(ctx.get("chapter", 0) or 0)
            findings = [f.to_dict() for f in
                        ContinuityEngine().check_chapter(state, chapter,
                                                         text)]
        elif isinstance(bible, dict):
            engine = ContinuityEngine()
            state = engine.from_story_bible(bible)
            findings = [f.to_dict() for f in
                        engine.check_chapter(
                            state, int(ctx.get("chapter", 0) or 0), text)]
        ctx["_continuity_findings"] = findings
        if not findings:
            return DimensionScore(dimension="continuity", score=0.9,
                                  evidence=["no continuity flags"])
        errors = [f for f in findings if f["severity"] == "error"]
        score = 0.9 - 0.4 * len(errors) - 0.05 * (
            len(findings) - len(errors))
        return DimensionScore(
            dimension="continuity", score=max(0.0, score),
            evidence=[f["message"] for f in findings])
