"""Prose craft metrics — measured, not vibes.

Covers the BookForge prose gap: show-vs-tell, adverb density,
sentence-length variance, filter words, passive voice, POV/tense drift,
chapter-end hook. Every number is computed from the text in front of it;
nothing is inferred from a model.

The chapter-end hook detector looks at the final paragraph for
unresolved-signal patterns (question, revelation, threat, arrival,
decision deferred) and reports kind + strength + excerpt, so a caller
can decide whether the hook is real or whether the chapter just stops.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any

__all__ = ["ProseReport", "analyze", "chapter_end_hook", "split_sentences",
           "split_paragraphs"]

FILTER_WORDS = (
    "saw", "heard", "felt", "noticed", "realized", "seemed",
    "appeared", "watched", "observed", "knew", "thought", "wondered",
    "decided", "remembered", "could see", "could hear",
)
TELLING_VERBS = (
    "was sad", "was happy", "was angry", "was scared", "was nervous",
    "was excited", "was tired", "felt sad", "felt happy", "felt angry",
)
_PAST_VERBS = frozenset(
    "was were had did said went came saw took made knew thought felt "
    "looked turned walked stood sat lay began became brought built "
    "caught chose fought found gave grew heard kept left lost met "
    "paid ran rose spoke understood wrote asked replied".split())
_PRESENT_VERBS = frozenset(
    "is are has have does says goes comes sees takes makes knows thinks "
    "feels looks turns walks stands sits lies begins becomes brings "
    "builds catches chooses fights finds gives grows hears keeps leaves "
    "loses meets pays runs rises speaks understands writes asks".split())
_FIRST_PRONOUNS = frozenset("i me my mine we us our ours".split())
_THIRD_PRONOUNS = frozenset(
    "he him his she her hers they them their theirs it its".split())
_BE_FORMS = ("am", "is", "are", "was", "were", "be", "been", "being")
_HOOK_PATTERNS: list[tuple[str, str]] = [
    ("question", r"\?\s*[\"”']?\s*$"),
    ("revelation", r"\b(it was|she was|he was|the truth was|all along|"
                   r"finally understood|realized that)\b[^.]{0,80}$"),
    ("threat", r"\b(coming for|will kill|dead by|too late|behind (him|her|"
               r"them)|watching|footsteps|door (opened|creaked))\b"),
    ("arrival", r"\b(arrived|appeared|stood in the doorway|knock at|"
                r"a voice|someone (was|stood))\b"),
    ("deferred_decision", r"\b(had to decide|tomorrow|at dawn|one (more|"
                          r"last) (chance|night|day)|before it was too late)\b"),
    ("cliffhanger", r"\b(and then|before (he|she|they) could|the floor "
                    r"(gave|shook)|everything went (dark|silent))\b"),
]
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-Z\"“‘(])")


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def split_sentences(text: str) -> list[str]:
    flat = " ".join(split_paragraphs(text))
    return [s.strip() for s in _SENT_SPLIT.split(flat) if s.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text or "")


def chapter_end_hook(text: str) -> dict[str, Any]:
    """Classify the final paragraph's hook signal (or its absence)."""
    paras = split_paragraphs(text)
    if not paras:
        return {"kind": "none", "strength": 0.0, "excerpt": "",
                "note": "empty chapter — no hook possible"}
    last = paras[-1]
    low = last.lower()
    for kind, pattern in _HOOK_PATTERNS:
        if re.search(pattern, low):
            return {"kind": kind, "strength": 0.8, "excerpt": last[-220:],
                    "note": f"final paragraph carries a {kind} signal"}
    if last.rstrip().endswith(("...", "—", "–")):
        return {"kind": "trailing", "strength": 0.5, "excerpt": last[-220:],
                "note": "chapter trails off — weak hook, consider a "
                        "question, revelation, or deferred decision"}
    return {"kind": "none", "strength": 0.0, "excerpt": last[-220:],
            "note": "final paragraph resolves without an unresolved "
                    "signal — the chapter ends, it does not hook"}


@dataclass
class ProseReport:
    words: int
    sentences: int
    paragraphs: int
    sentence_len_mean: float
    sentence_len_std: float
    sentence_len_cv: float
    adverb_density: float
    adverb_count: int
    filter_density: float
    filter_hits: dict[str, int]
    passive_count: int
    passive_density: float
    dialogue_ratio: float
    pov_first_ratio: float
    pov_third_ratio: float
    pov_drift: bool
    tense_past_ratio: float
    tense_present_ratio: float
    tense_drift: bool
    show_tell_note: str
    hook: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "words": self.words, "sentences": self.sentences,
            "paragraphs": self.paragraphs,
            "sentence_len_mean": round(self.sentence_len_mean, 2),
            "sentence_len_std": round(self.sentence_len_std, 2),
            "sentence_len_cv": round(self.sentence_len_cv, 3),
            "adverb_density": round(self.adverb_density, 4),
            "adverb_count": self.adverb_count,
            "filter_density": round(self.filter_density, 4),
            "filter_hits": self.filter_hits,
            "passive_count": self.passive_count,
            "passive_density": round(self.passive_density, 4),
            "dialogue_ratio": round(self.dialogue_ratio, 3),
            "pov_first_ratio": round(self.pov_first_ratio, 3),
            "pov_third_ratio": round(self.pov_third_ratio, 3),
            "pov_drift": self.pov_drift,
            "tense_past_ratio": round(self.tense_past_ratio, 3),
            "tense_present_ratio": round(self.tense_present_ratio, 3),
            "tense_drift": self.tense_drift,
            "show_tell_note": self.show_tell_note,
            "hook": self.hook,
        }


def analyze(text: str) -> ProseReport:
    """Full prose scan of one chapter (or any passage)."""
    text = text or ""
    words = _words(text)
    n_words = len(words)
    sentences = split_sentences(text)
    paras = split_paragraphs(text)
    sent_lens = [len(_words(s)) for s in sentences] or [0]
    mean = statistics.fmean(sent_lens)
    std = statistics.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0
    low = text.lower()
    # adverbs: words ending -ly (minus a small non-adverb stoplist)
    non_adv = {"only", "early", "really", "actually", "finally", "family",
               "likely", "nearly", "mostly", "partly", "daily", "weekly"}
    adverbs = [w for w in (w.lower() for w in words)
               if w.endswith("ly") and len(w) > 4 and w not in non_adv]
    # filter words
    fhits = {w: len(re.findall(r"\b" + re.escape(w) + r"\b", low))
             for w in FILTER_WORDS}
    fhits = {w: c for w, c in fhits.items() if c}
    # passive: be-form + past participle (word ending ed/en, not a noun)
    passive = len(re.findall(
        r"\b(?:am|is|are|was|were|be|been|being)\s+"
        r"[a-z]+(?:ed|en)\b", low))
    # dialogue: quoted characters / total characters
    quoted = sum(len(m) for m in re.findall(r'"[^"]*"|“[^”]*”', text))
    dialogue_ratio = quoted / max(1, len(text))
    # POV / tense drift: first half vs second half ratios
    halves = _half_ratios(text)
    pov_drift = abs(halves["first_a"] - halves["first_b"]) > 0.15
    tense_drift = abs(halves["past_a"] - halves["past_b"]) > 0.25
    if fhits and sum(fhits.values()) / max(1, n_words) > 0.015:
        show_note = ("filter-word heavy — render the perception, not the "
                     "perceiving")
    elif fhits:
        show_note = "some filter words — check the flagged spots"
    else:
        show_note = "no filter-word telling detected"
    return ProseReport(
        words=n_words, sentences=len(sentences), paragraphs=len(paras),
        sentence_len_mean=round(mean, 2),
        sentence_len_std=round(std, 2),
        sentence_len_cv=round(std / mean, 3) if mean else 0.0,
        adverb_density=len(adverbs) / max(1, n_words),
        adverb_count=len(adverbs),
        filter_density=sum(fhits.values()) / max(1, n_words),
        filter_hits=fhits, passive_count=passive,
        passive_density=passive / max(1, n_words),
        dialogue_ratio=dialogue_ratio,
        pov_first_ratio=halves["first_all"],
        pov_third_ratio=halves["third_all"],
        pov_drift=pov_drift,
        tense_past_ratio=halves["past_all"],
        tense_present_ratio=halves["present_all"],
        tense_drift=tense_drift, show_tell_note=show_note,
        hook=chapter_end_hook(text))


def _half_ratios(text: str) -> dict[str, float]:
    def _ratios(chunk: str) -> tuple[float, float, float, float]:
        toks = [w.lower() for w in _words(chunk)]
        n = max(1, len(toks))
        first = sum(1 for w in toks if w in _FIRST_PRONOUNS) / n
        third = sum(1 for w in toks if w in _THIRD_PRONOUNS) / n
        past = sum(1 for w in toks if w in _PAST_VERBS) / n
        present = sum(1 for w in toks if w in _PRESENT_VERBS) / n
        return first, third, past, present
    mid = len(text) // 2
    fa, ta, pa, pra = _ratios(text[:mid])
    fb, tb, pb, prb = _ratios(text[mid:])
    fall, tall, pall, prall = _ratios(text)
    return {"first_a": fa, "first_b": fb, "past_a": pa, "past_b": pb,
            "first_all": fall, "third_all": tall,
            "past_all": pall, "present_all": prall}
