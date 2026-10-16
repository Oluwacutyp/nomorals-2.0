"""Anti-AI-tell library — counts + locations, never a hard block.

The pattern follows the :data:`COACH_BANNED_PHRASES` precedent in
``nomorals.health.coach`` — a named tuple of banned phrasing + a guard
function — with one deliberate inversion: the coach guard RAISES, this
one never does.  An AI-tell is a report ("this line reads like model
output"), not a veto.  The owner overrules any hit, always.

Each entry carries ``why`` so the report teaches instead of nagging.
Structural tells (rule-of-three overuse, em-dash overuse, a dialogue
tag carrying an emotion the line already carries, the sentence after
dialogue restating the line, chapter-close tells, the explanatory
``..., which meant that...`` clause, a summary beat in past-perfect)
are detected by small dedicated checks, not just string matching —
the same counts + locations shape either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .prose import split_paragraphs, split_sentences
from .voiceprint import FILTERING_VERBS

__all__ = [
    "AI_TELLS", "AITell", "scan_ai_tells", "ai_tell_summary",
]

#: Phrases / patterns that read as model output.  ``kind`` is
#: "literal" (case-insensitive substring) or "regex".  Structural tells
#: live in the dedicated check functions below, not here.
AI_TELLS: tuple[dict[str, str], ...] = (
    {"id": "testament_to", "label": "testament to",
     "pattern": "testament to", "kind": "literal",
     "why": "Hollow praise-phrase: asserts significance instead of showing it."},
    {"id": "little_did_know", "label": "little did X know",
     "pattern": r"little did (he|she|they|we|i|it|\w+) know",
     "kind": "regex",
     "why": "Narrator winking at the reader from above the story."},
    {"id": "in_that_moment", "label": "in that moment",
     "pattern": "in that moment", "kind": "literal",
     "why": "Generic intensifier; a specific image would do the work."},
    {"id": "mix_of", "label": "mix of X and Y",
     "pattern": r"mix of [^.!?]{2,60} and [^.!?]{2,60}",
     "kind": "regex",
     "why": "Emotion by shopping list instead of one true detail."},
    {"id": "not_just_but", "label": "not just X but Y",
     "pattern": r"not just [^.!?]{2,60} but [^.!?]{2,60}",
     "kind": "regex",
     "why": "Correction-rhetoric: the sentence argues with itself for emphasis."},
    {"id": "wasnt_x_was_y", "label": "It wasn't X, it was Y",
     "pattern": r"it wasn't [^.!?]{2,60},? it was [^.!?]{2,60}",
     "kind": "regex",
     "why": "Same correction-rhetoric in past tense; usually the Y alone lands harder."},
    {"id": "something_shifted", "label": "something shifted",
     "pattern": "something shifted", "kind": "literal",
     "why": "Vague change with no mechanism — what moved, exactly?"},
    {"id": "air_thick_with", "label": "air thick with",
     "pattern": "air thick with", "kind": "literal",
     "why": "Stock atmosphere: the air is always thick with something."},
    {"id": "silence_deafening", "label": "silence deafening",
     "pattern": r"silence was deafening|deafening silence",
     "kind": "regex",
     "why": "Dead metaphor; silence has better verbs available."},
    {"id": "time_slow", "label": "time seemed slow",
     "pattern": r"time seemed (to )?slow|time slowed to",
     "kind": "regex",
     "why": "Slow-motion cliché; the moment deserves its own clock."},
    {"id": "felt_wave", "label": "felt a wave of",
     "pattern": r"felt (a|the) wave of",
     "kind": "regex",
     "why": "Emotion as weather report instead of a body doing a thing."},
    {"id": "couldnt_help_but", "label": "couldn't help but",
     "pattern": "couldn't help but", "kind": "literal",
     "why": "Removes the character's agency mid-sentence."},
    {"id": "chill_spine", "label": "chill ran down spine",
     "pattern": r"chill ran down (his|her|their|my|the) spine|shiver (ran )?down (his|her|their|my) spine",
     "kind": "regex",
     "why": "Horror/thriller stock body cue."},
    {"id": "breath_holding", "label": "breath he didn't know he was holding",
     "pattern": r"breath (he|she|they|i) (didn't|hadn't|never) know (he|she|they|i) (was|were|'d been|had been) holding|breath (he|she|they) didn't know (he|she|they) (were )?holding",
     "kind": "regex",
     "why": "The most-quoted AI tell in the wild; cut it and the scene survives."},
    {"id": "was_devastated", "label": "was devastated / elated / furious",
     "pattern": r"\b(was|felt) (devastated|elated|furious)\b",
     "kind": "regex",
     "why": "Emotion label doing the work a beat should do."},
    {"id": "heart_raced", "label": "heart raced / pounded / hammered",
     "pattern": r"heart (raced|pounded|hammered|thudded|thumped)",
     "kind": "regex",
     "why": "Stock arousal cue; a specific physical detail beats a racing heart."},
    {"id": "explanatory_clause", "label": "explanatory clause '..., which meant that...'",
     "pattern": r", which meant that\b",
     "kind": "regex",
     "why": "The narrator explaining the scene to the reader instead of trusting it."},
)

#: emotion adverbs / exotic tags that carry an emotion the dialogue
#: line itself already carries.
_EMOTION_ADVERBS = (
    "angrily", "sadly", "happily", "furiously", "joyfully", "bitterly",
    "tearfully", "nervously", "anxiously", "desperately", "lovingly",
    "hatefully", "jealously", "enviously", "mournfully",
)
_EXOTIC_TAGS = (
    "hissed", "growled", "snarled", "sobbed", "wailed", "sneered",
    "gasped", "choked", "whimpered", "bellowed", "shrieked",
)
_MORALISING_CLOSE = (
    "in the end", "what mattered", "the lesson", "learned that",
    "the meaning", "and that was", "after all", "the truth was",
    "what it means to", "the real meaning",
)


@dataclass
class AITell:
    """One detected tell: what, where, why — never a verdict."""
    id: str
    label: str
    count: int
    locations: list[dict[str, Any]]
    why: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "count": self.count,
                "locations": self.locations, "why": self.why}


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _excerpt(text: str, offset: int, radius: int = 90) -> str:
    start = max(0, offset - radius)
    end = min(len(text), offset + radius)
    return " ".join(text[start:end].split())


def _locations(text: str, matches: list[re.Match]) -> list[dict[str, Any]]:
    return [{"line": _line_of(text, m.start()),
             "offset": m.start(),
             "excerpt": _excerpt(text, m.start())}
            for m in matches]


def _check_rule_of_three(text: str) -> AITell | None:
    """Sentences built as X, Y, and Z — three parallel beats, one too many.

    Only sentences with exactly two commas and a final 'and' count, and
    only when the three segments are short parallel phrases.  Reported
    as 'possible' — lists exist for honest reasons too.
    """
    locs: list[dict[str, Any]] = []
    for sent in split_sentences(text):
        if sent.count(",") != 2 or " and " not in sent.lower():
            continue
        parts = re.split(r",\s*|\s+and\s+", sent.strip())
        if len(parts) == 3 and all(2 <= len(p.split()) <= 7 for p in parts):
            offset = text.find(sent)
            locs.append({"line": _line_of(text, max(0, offset)),
                         "offset": max(0, offset),
                         "excerpt": sent[:160]})
    if not locs:
        return None
    return AITell("rule_of_three", "rule of three (possible overuse)",
                  len(locs), locs,
                  "Three parallel beats in one sentence, repeated, reads "
                  "like a generated cadence. Two beats or one true detail "
                  "usually lands harder.")


def _check_em_dash_overuse(text: str) -> AITell | None:
    words = len(re.findall(r"[A-Za-z']+", text))
    dashes = text.count("—") + text.count("--")
    per_1k = dashes / max(1, words) * 1000
    # overuse = dense em-dash habit, not one dash doing a job
    if words < 150 or (dashes < 6 and per_1k < 8):
        return None
    locs: list[dict[str, Any]] = []
    for m in re.finditer(r"—|--", text):
        locs.append({"line": _line_of(text, m.start()), "offset": m.start(),
                     "excerpt": _excerpt(text, m.start())})
    return AITell("em_dash_overuse", "em-dash overuse", len(locs), locs[:12],
                  f"{dashes} em-dashes ({per_1k:.1f}/1k words) — the dash "
                  "stops being punctuation and becomes the rhythm.")


def _check_filtering_verbs(text: str) -> AITell | None:
    low = text.lower()
    locs: list[dict[str, Any]] = []
    for verb in FILTERING_VERBS:
        for m in re.finditer(r"\b" + re.escape(verb) + r"\b", low):
            locs.append({"line": _line_of(text, m.start()),
                         "offset": m.start(),
                         "excerpt": _excerpt(text, m.start())})
    if not locs:
        return None
    return AITell("filtering_verbs", "filtering verbs", len(locs),
                  sorted(locs, key=lambda l: l["offset"])[:15],
                  "Perception reported through a filter verb ('she saw the "
                  "door') instead of rendered ('the door stood open').")


def _check_dialogue_tag_emotion(text: str) -> AITell | None:
    """A tag carrying an emotion the line already carries."""
    locs: list[dict[str, Any]] = []
    quote_re = re.compile(r'"[^"]{2,}"|“[^”]{2,}”')
    for m in quote_re.finditer(text):
        after = text[m.end():m.end() + 80]
        tag_m = re.search(
            r"\b(?:said|asked|replied|whispered|shouted|muttered)\b\s+"
            r"(" + "|".join(_EMOTION_ADVERBS) + r")\b", after, re.IGNORECASE)
        exotic_m = re.search(r"\b(" + "|".join(_EXOTIC_TAGS) + r")\b",
                             after, re.IGNORECASE)
        hit = tag_m or exotic_m
        if hit:
            locs.append({"line": _line_of(text, m.start()),
                         "offset": m.start(),
                         "excerpt": (m.group(0)[:90] + " " +
                                     after.strip()[:70])})
    if not locs:
        return None
    return AITell("dialogue_tag_emotion",
                  "dialogue tag carrying an emotion the line already carries",
                  len(locs), locs,
                  "If the line is angry, the tag need not say 'angrily'. "
                  "'Said' is invisible; the line does the feeling.")


def _check_restating_sentence(text: str) -> AITell | None:
    """The sentence after dialogue restating what the line just said."""
    locs: list[dict[str, Any]] = []
    quote_re = re.compile(r'"[^"]{2,}"|“[^”]{2,}”')
    stop = set("the a an and or of to in on at for with is was were be "
               "been are had has have it its they them he she his her we "
               "you your i me my as by from that this these those not no "
               "but so if then than too".split())
    for m in quote_re.finditer(text):
        rest = text[m.end():]
        # walk the following sentences: skip pure attribution
        # ("Mara said.") and test the first substantive sentence
        nxt = ""
        for cand in re.findall(r"[^.!?]{1,300}[.!?]", rest):
            stripped = cand.strip()
            if len(stripped) < 40 and re.search(
                    r"\b(said|asked|replied|whispered|shouted|muttered)\b",
                    stripped.lower()):
                continue
            nxt = cand
            break
        if not nxt:
            continue
        # skip if the next sentence is itself a new speaker's line
        if nxt.strip().startswith(('"', "“")):
            continue
        quote_words = {w.lower() for w in
                       re.findall(r"[a-z']+", m.group(0)) if w.lower() not in stop}
        next_words = {w.lower() for w in
                      re.findall(r"[a-z']+", nxt) if w.lower() not in stop}
        if len(quote_words & next_words) >= 3:
            locs.append({"line": _line_of(text, m.start()),
                         "offset": m.start(),
                         "excerpt": (m.group(0)[:80] + " → " + nxt.strip()[:100])})
    if not locs:
        return None
    return AITell("restating_sentence",
                  "sentence after dialogue restating the line",
                  len(locs), locs,
                  "The narrative repeats the dialogue's content instead of "
                  "moving the scene forward.")


def _check_chapter_close(text: str) -> AITell | None:
    """The chapter closes on theme / a rhetorical question / 'hope'."""
    paras = split_paragraphs(text)
    if not paras:
        return None
    last = paras[-1]
    low = last.lower()
    hits: list[dict[str, Any]] = []
    offset = text.rfind(last)
    if last.strip().endswith("?"):
        hits.append({"line": _line_of(text, max(0, offset)),
                     "offset": max(0, offset),
                     "excerpt": last[-200:],
                     "kind": "rhetorical-question close"})
    if re.search(r"\bhope\b", low):
        hits.append({"line": _line_of(text, max(0, offset)),
                     "offset": max(0, offset),
                     "excerpt": last[-200:],
                     "kind": "closes on the word 'hope'"})
    for phrase in _MORALISING_CLOSE:
        if phrase in low:
            hits.append({"line": _line_of(text, max(0, offset)),
                         "offset": max(0, offset),
                         "excerpt": last[-200:],
                         "kind": f"thematic/moralising close ({phrase!r})"})
            break
    if not hits:
        return None
    return AITell("chapter_close",
                  "chapter closes on theme / rhetorical question / 'hope'",
                  len(hits), hits,
                  "A chapter that ends on a theme explains itself. End on "
                  "an image or an action and let the theme stay unsaid.")


def _check_summary_past_perfect(text: str) -> AITell | None:
    """A summary beat in past-perfect ('had ...') — telling the aftermath."""
    paras = split_paragraphs(text)
    if not paras:
        return None
    last = paras[-1]
    matches = list(re.finditer(r"\bhad\s+(?:\w+ed|been|gone|come|seen|taken)\b",
                               last.lower()))
    if len(matches) < 2:
        return None
    offset = text.rfind(last)
    return AITell("summary_past_perfect",
                  "summary beat in past-perfect (closing 'had ...')",
                  len(matches),
                  [{"line": _line_of(text, max(0, offset)),
                    "offset": max(0, offset),
                    "excerpt": last[-260:]}],
                  "Past-perfect summary at the close tells what happened "
                  "instead of ending inside the scene.")


def scan_ai_tells(text: str) -> dict[str, Any]:
    """Scan ``text`` for AI-tells. Counts + locations, never a hard block.

    Returns ``{"total", "tells": [...], "blocked": False, ...}``.  The
    owner overrules any hit — a tell is a report, not a veto.
    """
    text = text or ""
    tells: list[AITell] = []
    low = text.lower()
    for entry in AI_TELLS:
        if entry["kind"] == "literal":
            needle = entry["pattern"].lower()
            matches = [m for m in re.finditer(re.escape(needle), low)]
        else:
            matches = list(re.finditer(entry["pattern"], low))
        if matches:
            tells.append(AITell(entry["id"], entry["label"], len(matches),
                                 _locations(text, matches), entry["why"]))
    for check in (_check_rule_of_three, _check_em_dash_overuse,
                  _check_filtering_verbs, _check_dialogue_tag_emotion,
                  _check_restating_sentence, _check_chapter_close,
                  _check_summary_past_perfect):
        hit = check(text)
        if hit is not None:
            tells.append(hit)
    total = sum(t.count for t in tells)
    words = max(1, len(re.findall(r"[A-Za-z']+", text)))
    return {
        "total": total,
        "per_1k": round(total / words * 1000, 2),
        "tells": [t.to_dict() for t in tells],
        "blocked": False,
        "owner_may_overrule": True,
        "note": ("AI-tells are reported, never blocking — like "
                 "COACH_BANNED_PHRASES in shape, inverted in force: the "
                 "coach guard raises, this one reports."),
    }


def ai_tell_summary(report: dict[str, Any]) -> str:
    """One-line summary of a scan report."""
    tells = report.get("tells", [])
    if not tells:
        return "No AI-tells detected."
    top = sorted(tells, key=lambda t: t["count"], reverse=True)[:3]
    bits = ", ".join(f"{t['label']} ×{t['count']}" for t in top)
    return (f"{report['total']} AI-tell hits "
            f"({report.get('per_1k', 0)}/1k words) — top: {bits}. "
            f"Reported, not blocked.")
