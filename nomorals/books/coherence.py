"""Long-form coherence at scale — queries over the continuity state.

Covers the BookForge coherence gap: foreshadowing payoff, timeline
consistency (dates / ages / seasons / travel), character-arc completion,
Chekhov's gun, contradiction detection. All of it runs over
:class:`ContinuityState` + the chapter texts — no model, deterministic
regex + state queries, so a 400-chapter work gets the same checks as a
4-chapter one.

Each finding carries ``kind`` / ``severity`` / ``chapter`` / ``evidence``
so the caller can render, count, or gate on it. Nothing here deletes or
rewrites story content: it reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from .continuity import ContinuityEngine, ContinuityState, Finding

__all__ = ["CoherenceReport", "analyze_coherence", "contradiction_scan"]

_AGE_RE = re.compile(r"\b(?-i:([A-Z][a-z]+))\b[^.]{0,60}?\b(?:aged?|age)\s+"
                     r"(\d{1,3})\b", re.IGNORECASE)
_AGE_RE2 = re.compile(r"\b(\d{1,3})-year-old\s+([A-Z][a-z]+)\b")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_SEASONS = ("spring", "summer", "autumn", "fall", "winter")
_SEASON_RE = re.compile(r"\b(spring|summer|autumn|fall|winter)\b",
                        re.IGNORECASE)
_TRAVEL_RE = re.compile(r"\b(\d+)\s+(days?|weeks?|months?)\s+(later|"
                        r"of travel|on the road|journey)\b", re.IGNORECASE)
_DEAD_RE = re.compile(r"\b(?-i:([A-Z][a-z]+))\b[^.]{0,50}?\b(died|was dead|"
                       r"lay dead|killed)\b", re.IGNORECASE)
#: Sentence openers / function words the name capture can land on when
#: it scans a lazy window ("The old man died" captures "The").  They are
#: never character names; a capture in this set is discarded, not
#: reported — a false "error" finding is worse than a missed one.
_NOT_A_NAME = frozenset({
    "The", "A", "An", "He", "She", "It", "They", "We", "This", "That",
    "These", "Those", "His", "Her", "Their", "Its", "But", "And", "When",
    "Then", "As", "In", "On", "At", "To", "For", "From", "With", "By",
    "After", "Before", "While", "Until", "Since", "If", "So", "Yet",
    "Or", "Nor", "Now", "Here", "There", "Once", "Just", "Still",
    "Even", "One", "Two", "Every", "Each", "All", "Some", "Many",
})
_ALIVE_VERBS = ("said", "asked", "walked", "ran", "laughed", "shouted",
                "whispered", "looked", "turned", "nodded", "smiled")


@dataclass
class CoherenceReport:
    """Everything the coherence pass found, grouped by check."""

    work_slug: str
    chapters_checked: int
    findings: list[Finding]
    foreshadowing: list[Finding]
    timeline: list[Finding]
    arcs: list[Finding]
    chekhov: list[Finding]
    contradictions: list[Finding]

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_slug": self.work_slug,
            "chapters_checked": self.chapters_checked,
            "counts": {
                "foreshadowing": len(self.foreshadowing),
                "timeline": len(self.timeline),
                "arcs": len(self.arcs),
                "chekhov": len(self.chekhov),
                "contradictions": len(self.contradictions),
                "total": len(self.findings),
            },
            "findings": [f.to_dict() for f in self.findings],
        }

    @property
    def clean(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)


def _as_state(state_or_bible: Any) -> ContinuityState:
    if isinstance(state_or_bible, ContinuityState):
        return state_or_bible
    return ContinuityEngine().from_story_bible(state_or_bible)


def analyze_coherence(state_or_bible: Any,
                      chapters: Sequence[tuple[int, str]],
                      *, stall_after: int = 10,
                      current_chapter: int = 0) -> CoherenceReport:
    """Run all five coherence checks over ``chapters`` (number, text)."""
    state = _as_state(state_or_bible)
    chapters = sorted(chapters, key=lambda c: c[0])
    current = current_chapter or (chapters[-1][0] if chapters else 0)
    findings: list[Finding] = []
    fore = _foreshadowing(state, chapters, current, stall_after)
    time_f = _timeline(state, chapters)
    arcs = _arcs(state, chapters, current, stall_after)
    chek = _chekhov(state, chapters)
    contra = contradiction_scan(chapters, state)
    findings = fore + time_f + arcs + chek + contra
    return CoherenceReport(
        work_slug=state.work_slug, chapters_checked=len(chapters),
        findings=findings, foreshadowing=fore, timeline=time_f,
        arcs=arcs, chekhov=chek, contradictions=contra)


def _foreshadowing(state: ContinuityState,
                   chapters: Sequence[tuple[int, str]],
                   current: int, stall_after: int) -> list[Finding]:
    out: list[Finding] = []
    texts = {n: (t or "").lower() for n, t in chapters}
    for thread in state.threads:
        if thread.status != "open":
            continue
        age = current - (thread.last_seen or thread.introduced)
        # payoff detection: thread keywords resurface in a later chapter
        payoff = 0
        for n, low in texts.items():
            if n > thread.introduced and thread.keywords and any(
                    k.lower() in low for k in thread.keywords):
                payoff = n
                break
        if payoff:
            continue  # paid off (or at least revisited) — not a finding
        if age >= stall_after:
            out.append(Finding(
                kind="unpaid_setup", severity="warning",
                message=f"Thread {thread.id!r} ({thread.summary[:80]}) "
                        f"introduced ch.{thread.introduced}, unseen for "
                        f"{age} chapters — pay it off or cut the setup",
                chapter=current, evidence=thread.summary[:200]))
    return out


def _timeline(state: ContinuityState,
              chapters: Sequence[tuple[int, str]]) -> list[Finding]:
    out: list[Finding] = []
    last_year: int | None = None
    last_season_idx: int | None = None
    ages: dict[str, int] = {}
    for number, text in sorted(chapters, key=lambda c: c[0]):
        low = (text or "").lower()
        # years must not run backwards across chapters
        years = [int(y) for y in _YEAR_RE.findall(text or "")
                 if 1000 <= int(y) <= 2100]
        for y in years:
            if last_year is not None and y < last_year - 1:
                out.append(Finding(
                    kind="timeline_year_backwards", severity="warning",
                    message=f"year {y} in ch.{number} after year "
                            f"{last_year} — time runs backwards",
                    chapter=number, evidence=text[:160]))
            last_year = max(last_year or y, y)
        # seasons: a jump of >1 season without travel mention is suspect
        seasons = [m.group(1).lower() for m in _SEASON_RE.finditer(text or "")]
        season_idx = {"spring": 0, "summer": 1, "autumn": 2, "fall": 2,
                      "winter": 3}
        for s in seasons:
            idx = season_idx[s]
            if (last_season_idx is not None
                    and abs(idx - last_season_idx) > 1
                    and not _TRAVEL_RE.search(text or "")):
                out.append(Finding(
                    kind="timeline_season_jump", severity="info",
                    message=f"season jumps to {s} in ch.{number} with no "
                            "travel/time passage noted",
                    chapter=number, evidence=text[:160]))
            last_season_idx = idx
        # ages: a named character must not get younger
        for m in _AGE_RE.finditer(text or ""):
            name, age = m.group(1), int(m.group(2))
            if name in _NOT_A_NAME:
                continue
            if name in ages and age < ages[name]:
                out.append(Finding(
                    kind="timeline_age_regression", severity="error",
                    message=f"{name} was {ages[name]} then {age} in "
                            f"ch.{number} — ages don't decrease",
                    chapter=number, entity=name,
                    evidence=m.group(0)))
            else:
                ages[name] = age
        for m in _AGE_RE2.finditer(text or ""):
            age, name = int(m.group(1)), m.group(2)
            if name in _NOT_A_NAME:
                continue
            if name in ages and age < ages[name]:
                out.append(Finding(
                    kind="timeline_age_regression", severity="error",
                    message=f"{name} was {ages[name]} then {age} in "
                            f"ch.{number}",
                    chapter=number, entity=name,
                    evidence=m.group(0)))
            else:
                ages[name] = age
    return out


def _arcs(state: ContinuityState,
          chapters: Sequence[tuple[int, str]],
          current: int, stall_after: int) -> list[Finding]:
    out: list[Finding] = []
    last_seen: dict[str, int] = {}
    for number, text in chapters:
        low = (text or "").lower()
        for ent in state.entities:
            if ent.kind != "character":
                continue
            forms = [ent.name, *ent.aliases]
            if any(f and re.search(r"\b" + re.escape(f) + r"\b", low, re.IGNORECASE)
                   for f in forms):
                last_seen[ent.name] = number
    for ent in state.entities:
        if ent.kind != "character" or ent.status != "alive":
            continue
        seen = last_seen.get(ent.name, ent.first_seen)
        if current - seen >= stall_after and ent.mentions > 0:
            out.append(Finding(
                kind="stalled_arc", severity="info",
                message=f"{ent.name} last seen ch.{seen}, {current - seen} "
                        "chapters ago — arc stalled or quietly dropped",
                chapter=current, entity=ent.name))
        elif ent.mentions == 0 and current - ent.first_seen >= stall_after:
            out.append(Finding(
                kind="abandoned_character", severity="info",
                message=f"{ent.name} introduced ch.{ent.first_seen} and "
                        "never mentioned again",
                chapter=current, entity=ent.name))
    return out


def _chekhov(state: ContinuityState,
             chapters: Sequence[tuple[int, str]]) -> list[Finding]:
    out: list[Finding] = []
    if not chapters:
        return out
    current = max(n for n, _ in chapters)
    counts: dict[str, int] = {}
    for _, text in chapters:
        low = (text or "").lower()
        for ent in state.entities:
            if ent.kind not in ("item", "concept"):
                continue
            forms = [ent.name, *ent.aliases]
            if any(f and re.search(r"\b" + re.escape(f) + r"\b", low, re.IGNORECASE)
                   for f in forms):
                counts[ent.name] = counts.get(ent.name, 0) + 1
    for ent in state.entities:
        if ent.kind not in ("item", "concept"):
            continue
        if counts.get(ent.name, 0) == 1 and current - ent.first_seen >= 5:
            out.append(Finding(
                kind="chekhov_gun", severity="info",
                message=f"{ent.name!r} ({ent.kind}) introduced "
                        f"ch.{ent.first_seen} and never used again — "
                        "fire it or cut it",
                chapter=current, entity=ent.name))
    for rule in state.rules:
        # a world rule stated once and never referenced again
        key_words = [w for w in re.findall(r"[a-z]{4,}",
                                           rule.statement.lower())][:4]
        if not key_words:
            continue
        hits = sum(1 for _, text in chapters
                   if any(k in (text or "").lower() for k in key_words))
        if hits <= 1 and current - rule.source_chapter >= 5:
            out.append(Finding(
                kind="chekhov_rule", severity="info",
                message=f"World rule {rule.id!r} ({rule.statement[:60]}) "
                        "never referenced again after ch."
                        f"{rule.source_chapter}",
                chapter=current, evidence=rule.statement[:160]))
    return out


def contradiction_scan(chapters: Sequence[tuple[int, str]],
                       state: ContinuityState | None = None
                       ) -> list[Finding]:
    """Direct textual contradictions: dead-then-speaking, age flips,
    'never X' then X. State-backed dead-checks run when state given."""
    out: list[Finding] = []
    dead_at: dict[str, int] = {}
    for number, text in sorted(chapters, key=lambda c: c[0]):
        text = text or ""
        for m in _DEAD_RE.finditer(text):
            name = m.group(1)
            # a sentence opener ("The") or a lowercase word matched under
            # the keyword's IGNORECASE is not a character — discarding it
            # is strictly better than a false contradiction error.
            if name in _NOT_A_NAME or not name[0].isupper():
                continue
            dead_at[name] = number
        for name, died_ch in list(dead_at.items()):
            if number <= died_ch:
                continue
            for verb in _ALIVE_VERBS:
                if re.search(r"\b" + re.escape(name) + r"\b[^.]{0,40}?\b"
                             + verb + r"\b", text, re.IGNORECASE):
                    out.append(Finding(
                        kind="contradiction_dead_speaks", severity="error",
                        message=f"{name} died ch.{died_ch} but {verb} in "
                                f"ch.{number}",
                        chapter=number, entity=name,
                        evidence=text[:160]))
                    break
    if state is not None:
        for ent in state.entities:
            if ent.status == "dead":
                for number, text in chapters:
                    if re.search(r"\b" + re.escape(ent.name) + r"\b",
                                 text or "", re.IGNORECASE):
                        out.append(Finding(
                            kind="contradiction_dead_speaks",
                            severity="error",
                            message=f"{ent.name} is dead in the bible but "
                                    f"appears in ch.{number}",
                            chapter=number, entity=ent.name))
    return out
