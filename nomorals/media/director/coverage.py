"""Script coverage — the reader's report a studio would pay for.

Brain-driven, structured, and honest: the brain reads the script
(with its parsed scene outline as the map) and returns coverage as
JSON against a fixed contract — logline, synopsis, structure read,
graded assessments (premise / character / dialogue / visual /
pacing), and a verdict on the industry scale (pass / consider /
recommend) with the reasons attached.

If the brain is unavailable or its answer breaks the contract, this
raises :class:`DirectorError`. There is no canned coverage and no
template verdict — a coverage report nobody read is worse than none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.logging_setup import get_logger
from .errors import DirectorError
from .screenplay import Screenplay

_log = get_logger(__name__)

__all__ = ["Coverage", "Assessment", "cover_script", "VERDICTS", "GRADES"]

VERDICTS = ("pass", "consider", "recommend")
GRADES = ("excellent", "strong", "adequate", "weak", "poor")
_ASSESSMENT_KEYS = ("premise", "character", "dialogue", "visual", "pacing")

#: script characters fed to the reader — long scripts are read via
#: their outline + opening/middle/end samples rather than truncated
#: blind at the first N characters (the brain fits its own window
#: too, but the reader should choose what it reads deliberately).
_READ_BUDGET = 36_000


@dataclass
class Assessment:
    grade: str = "adequate"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"grade": self.grade, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Assessment":
        grade = str(d.get("grade", "adequate")).strip().lower()
        return cls(grade=grade if grade in GRADES else "adequate",
                   notes=str(d.get("notes", "")))


@dataclass
class Coverage:
    title: str = ""
    logline: str = ""
    synopsis: str = ""
    genre: str = ""
    structure: dict[str, Any] = field(default_factory=dict)
    assessments: dict[str, Assessment] = field(default_factory=dict)
    verdict: str = ""
    verdict_reasons: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "logline": self.logline,
            "synopsis": self.synopsis, "genre": self.genre,
            "structure": self.structure,
            "assessments": {k: a.to_dict()
                            for k, a in self.assessments.items()},
            "verdict": self.verdict,
            "verdict_reasons": list(self.verdict_reasons),
            "strengths": list(self.strengths),
            "concerns": list(self.concerns),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Coverage":
        return cls(
            title=str(d.get("title", "")), logline=str(d.get("logline", "")),
            synopsis=str(d.get("synopsis", "")),
            genre=str(d.get("genre", "")),
            structure=dict(d.get("structure") or {}),
            assessments={k: Assessment.from_dict(v)
                         for k, v in (d.get("assessments") or {}).items()
                         if isinstance(v, dict)},
            verdict=str(d.get("verdict", "")),
            verdict_reasons=[str(x) for x in d.get("verdict_reasons", [])],
            strengths=[str(x) for x in d.get("strengths", [])],
            concerns=[str(x) for x in d.get("concerns", [])],
        )

    def report(self) -> str:
        """The coverage sheet as a reader would hand it over."""
        lines = [f"COVERAGE — {self.title or 'untitled script'}"]
        if self.genre:
            lines.append(f"genre: {self.genre}")
        lines.append(f"logline: {self.logline}")
        lines.append(f"synopsis: {self.synopsis}")
        acts = (self.structure or {}).get("acts") or []
        beats = (self.structure or {}).get("beats") or []
        if acts:
            lines.append("structure:")
            for a in acts:
                lines.append(f"  - {a}")
        if beats:
            lines.append("major beats: " + " · ".join(str(b) for b in beats))
        if self.assessments:
            lines.append("assessments:")
            for key in _ASSESSMENT_KEYS:
                a = self.assessments.get(key)
                if a:
                    lines.append(f"  {key:10} {a.grade:9} {a.notes}")
        lines.append(f"VERDICT: {self.verdict.upper()}")
        for r in self.verdict_reasons:
            lines.append(f"  - {r}")
        if self.strengths:
            lines.append("strengths: " + "; ".join(self.strengths))
        if self.concerns:
            lines.append("concerns: " + "; ".join(self.concerns))
        return "\n".join(lines)


def _reading_sample(text: str, play: Screenplay | None) -> str:
    """What the reader reads: outline + full text if it fits, else a
    deliberate head/middle/tail sample (never a blind head-cut)."""
    outline = play.outline() if play and play.scenes else ""
    if len(text) <= _READ_BUDGET:
        body = text
    else:
        head = text[: _READ_BUDGET // 2]
        mid_start = max(0, len(text) // 2 - _READ_BUDGET // 4)
        middle = text[mid_start: mid_start + _READ_BUDGET // 4]
        tail = text[-(_READ_BUDGET // 4):]
        body = (f"{head}\n\n[… middle sample …]\n{middle}"
                f"\n\n[… final pages …]\n{tail}")
    return (f"SCENE OUTLINE:\n{outline}\n\nSCRIPT:\n{body}"
            if outline else body)


def cover_script(text: str, *, brain: Any,
                 play: Screenplay | None = None,
                 title: str = "") -> Coverage:
    """Read ``text`` and return structured coverage.

    ``brain`` is a :class:`nomorals.llm.brain.Brain` (anything with a
    compatible ``chat_json``). Raises :class:`DirectorError` when the
    brain fails or its answer breaks the coverage contract.
    """
    from ...llm.base import Message

    if brain is None:
        raise DirectorError(
            "coverage needs the brain — no brain is configured, and "
            "there is no canned coverage to fall back on")
    if not (text or "").strip():
        raise DirectorError("coverage needs a script — the text is empty")
    sample = _reading_sample(text, play)
    prompt = (
        "You are a professional script reader writing studio coverage. "
        "Read the script and reply with ONLY a JSON object of exactly "
        "this shape:\n"
        '{"title": str, "logline": str (one sentence), '
        '"synopsis": str (one paragraph), "genre": str, '
        '"structure": {"acts": [str, ...], "beats": [str, ...]}, '
        '"assessments": {"premise": {"grade": g, "notes": str}, '
        '"character": {"grade": g, "notes": str}, '
        '"dialogue": {"grade": g, "notes": str}, '
        '"visual": {"grade": g, "notes": str}, '
        '"pacing": {"grade": g, "notes": str}}, '
        '"verdict": "pass" | "consider" | "recommend", '
        '"verdict_reasons": [str, ...], "strengths": [str, ...], '
        '"concerns": [str, ...]}\n'
        f"where every grade g is one of {list(GRADES)}. "
        "Judge like a working reader: specific, evidence-based, no "
        "flattery. The verdict is the industry scale — pass means do "
        "not produce, consider means promising with real reservations, "
        "recommend means push it forward.\n\n" + sample)
    try:
        data, resp = brain.chat_json(
            [Message.user(prompt)], task_kind="coverage",
            params=_coverage_params())
    except Exception as exc:  # noqa: BLE001 - surfaced as DirectorError
        raise DirectorError(f"coverage failed: {exc}") from exc
    if not isinstance(data, dict):
        err = getattr(resp, "error", "") or "no usable reply"
        raise DirectorError(
            f"coverage failed: the brain returned no structured "
            f"coverage ({err})")
    cov = Coverage.from_dict(data)
    cov.title = cov.title or title or (play.title if play else "")
    missing = [k for k in ("logline", "synopsis", "verdict")
               if not getattr(cov, k)]
    if cov.verdict not in VERDICTS:
        missing.append(f"verdict must be one of {VERDICTS}")
    if missing:
        raise DirectorError(
            "coverage broke its contract — missing/invalid: "
            + ", ".join(missing))
    for key in _ASSESSMENT_KEYS:
        cov.assessments.setdefault(key, Assessment(
            grade="adequate", notes="(reader gave no note)"))
    return cov


def _coverage_params():
    from ...llm.base import SamplingParams
    return SamplingParams(temperature=0.3, max_tokens=2200)
