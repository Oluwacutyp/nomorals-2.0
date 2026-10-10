"""Voice fingerprint: do characters actually sound like themselves?

Every character has an ``expression`` (catchphrases, emoji habits) — but
nothing verified they *stay* in voice. This module computes a voice
fingerprint from a character's dialogue history and scores new
utterances for consistency. Drift gets flagged in the post-session
processing pass, not punished mid-conversation.

The fingerprint is deliberately shallow (lexical + structural, no LLM):
- top content words (excluding stopwords)
- sentence-length distribution (mean, spread)
- emoji rate, question rate, exclamation rate
- catchphrase hit rate (from character.expression)

Two characters with near-identical fingerprints probably need more
distinct expression settings — ``distinctness(a, b)`` quantifies it.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "VoiceFingerprint",
    "fingerprint",
    "consistency",
    "distinctness",
    "function_word_profile",
    "style_report",
]

# Stylometry gold: function words are the most robust authorship
# discriminator — used unconsciously, hard to fake, topic-independent.
# Content words say what you talk about; function words say who you are.
_FUNCTION_WORDS = frozenset(
    "i me my mine we us our ours you your yours he him his she her hers "
    "it its they them their theirs the a an and or but if so as at in on "
    "of to for with by from is are was were be been being do does did "
    "done have has had having not no nor that this these those then than "
    "too very just really well now here there what when where who whom "
    "why how all any some such can will would could should shall may "
    "might must own same other another each every".split()
)

_STOPWORDS = frozenset(
    "i me my we us you he she it they them the a an and or but if so as at "
    "in on of to for with is are was were be been do does did have has had "
    "not no yes oh ah um uh like just really very so too".split()
)

_WORD_RE = re.compile(r"[a-z']+")


def _words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall((text or "").lower())
            if w not in _STOPWORDS and len(w) > 2]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?…]+", text or "")
    return [p.strip() for p in parts if p.strip()]


@dataclass
class VoiceFingerprint:
    """Lexical + structural voice signature.

    Stylometry-informed: content ``top_words`` are topic-contaminated,
    so the fingerprint ALSO carries function-word rates (the robust
    discriminator), vocabulary richness (type-token ratio), and
    word-length distribution.
    """

    top_words: tuple[tuple[str, int], ...] = ()
    mean_sentence_len: float = 0.0
    sentence_len_spread: float = 0.0
    emoji_rate: float = 0.0
    question_rate: float = 0.0
    exclaim_rate: float = 0.0
    catchphrase_hits: float = 0.0
    n_samples: int = 0
    # ── sweep additions (stylometry gold) ──
    function_words: tuple[tuple[str, float], ...] = ()  # word -> rate per 100 words
    ttr: float = 0.0                    # type-token ratio: vocabulary richness
    mean_word_len: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_words": [list(p) for p in self.top_words],
            "mean_sentence_len": self.mean_sentence_len,
            "sentence_len_spread": self.sentence_len_spread,
            "emoji_rate": self.emoji_rate,
            "question_rate": self.question_rate,
            "exclaim_rate": self.exclaim_rate,
            "catchphrase_hits": self.catchphrase_hits,
            "n_samples": self.n_samples,
            "function_words": [list(p) for p in self.function_words],
            "ttr": self.ttr,
            "mean_word_len": self.mean_word_len,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VoiceFingerprint":
        d = d or {}
        return cls(
            top_words=tuple((str(w), int(c)) for w, c in
                            (d.get("top_words") or [])),
            mean_sentence_len=float(d.get("mean_sentence_len", 0.0)),
            sentence_len_spread=float(d.get("sentence_len_spread", 0.0)),
            emoji_rate=float(d.get("emoji_rate", 0.0)),
            question_rate=float(d.get("question_rate", 0.0)),
            exclaim_rate=float(d.get("exclaim_rate", 0.0)),
            catchphrase_hits=float(d.get("catchphrase_hits", 0.0)),
            n_samples=int(d.get("n_samples", 0)),
            function_words=tuple((str(w), float(r)) for w, r in
                                 (d.get("function_words") or [])),
            ttr=float(d.get("ttr", 0.0)),
            mean_word_len=float(d.get("mean_word_len", 0.0)),
        )


def _emoji_count(text: str) -> int:
    return sum(1 for ch in (text or "") if ord(ch) > 0x1F300)


def fingerprint(utterances: list[str],
                catchphrases: list[str] | tuple[str, ...] = ()) -> VoiceFingerprint:
    """Build a fingerprint from a character's past utterances."""
    utterances = [u for u in (utterances or []) if (u or "").strip()]
    if not utterances:
        return VoiceFingerprint()
    word_counts: Counter[str] = Counter()
    sent_lens: list[float] = []
    word_lens: list[float] = []
    func_counts: Counter[str] = Counter()
    total_words = 0
    emoji_total = 0
    q_total = 0
    e_total = 0
    cp_hits = 0
    cp_lows = [c.lower() for c in (catchphrases or []) if c]
    for u in utterances:
        toks = _WORD_RE.findall(u.lower())
        for w in toks:
            total_words += 1
            word_lens.append(float(len(w)))
            if w in _FUNCTION_WORDS:
                func_counts[w] += 1
            elif w not in _STOPWORDS and len(w) > 2:
                word_counts[w] += 1
        sents = _sentences(u)
        for s in sents:
            sent_lens.append(float(len(s.split())))
        emoji_total += _emoji_count(u)
        q_total += u.count("?")
        e_total += u.count("!")
        low = u.lower()
        if any(cp in low for cp in cp_lows):
            cp_hits += 1
    n = len(utterances)
    mean_len = sum(sent_lens) / len(sent_lens) if sent_lens else 0.0
    spread = (math.sqrt(sum((x - mean_len) ** 2 for x in sent_lens)
                        / len(sent_lens)) if sent_lens else 0.0)
    # vocabulary richness: type-token over content words
    content_toks = sum(word_counts.values())
    ttr = (len(word_counts) / content_toks) if content_toks else 0.0
    mean_wlen = sum(word_lens) / len(word_lens) if word_lens else 0.0
    func_profile = tuple(
        (w, round(c / total_words * 100, 2))
        for w, c in func_counts.most_common(20)) if total_words else ()
    return VoiceFingerprint(
        top_words=tuple(word_counts.most_common(25)),
        mean_sentence_len=mean_len,
        sentence_len_spread=spread,
        emoji_rate=emoji_total / n,
        question_rate=q_total / n,
        exclaim_rate=e_total / n,
        catchphrase_hits=cp_hits / n,
        n_samples=n,
        function_words=func_profile,
        ttr=ttr,
        mean_word_len=mean_wlen,
    )


def function_word_profile(utterances: list[str]) -> dict[str, float]:
    """Rate-per-100-words for each function word — the stylometric core."""
    fp = fingerprint(utterances)
    return {w: r for w, r in fp.function_words}


def _word_overlap(a: VoiceFingerprint, b: VoiceFingerprint) -> float:
    wa = {w for w, _ in a.top_words[:15]}
    wb = {w for w, _ in b.top_words[:15]}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _func_divergence(a: VoiceFingerprint, b: VoiceFingerprint) -> float:
    """Burrows-flavored: mean absolute difference of function-word rates,
    normalized. Function words are the stylometric core — unconscious,
    topic-independent, hard to fake."""
    da = {w: r for w, r in a.function_words}
    db = {w: r for w, r in b.function_words}
    vocab = set(da) | set(db)
    if not vocab:
        return 0.0
    diff = sum(abs(da.get(w, 0.0) - db.get(w, 0.0)) for w in vocab)
    return min(1.0, diff / (len(vocab) * 2.0))


def consistency(fp: VoiceFingerprint, utterance: str,
                catchphrases: list[str] | tuple[str, ...] = ()) -> float:
    """Score 0–1: does this utterance sound like the fingerprint?

    Compares the single utterance's structural stats against the
    fingerprint's distributions. Needs n_samples >= 5 to be meaningful —
    below that it returns 0.5 (not enough data to judge).
    """
    if fp.n_samples < 5 or not (utterance or "").strip():
        return 0.5
    single = fingerprint([utterance], catchphrases)
    score = 0.5
    # sentence length within 2 spreads of the mean
    if fp.sentence_len_spread > 0:
        z = abs(single.mean_sentence_len - fp.mean_sentence_len) \
            / fp.sentence_len_spread
        score += 0.2 * max(0.0, 1.0 - z / 3.0)
    else:
        score += 0.1
    # word overlap with the character's vocabulary
    score += 0.2 * _word_overlap(fp, single)
    # punctuation habits within tolerance
    for attr in ("emoji_rate", "question_rate", "exclaim_rate"):
        a = getattr(fp, attr)
        b = getattr(single, attr)
        score += 0.033 * (1.0 - min(1.0, abs(a - b) * 2.0))
    return max(0.0, min(1.0, score))


def distinctness(a: VoiceFingerprint, b: VoiceFingerprint) -> float:
    """Score 0–1: how different do these two voices sound?

    1.0 = completely distinct, 0.0 = interchangeable. Below ~0.4 the two
    characters probably need more differentiated expression settings.

    Stylometry-weighted: function-word divergence leads (it's the
    robust discriminator); content vocabulary is down-weighted because
    it's topic-contaminated.
    """
    if a.n_samples < 3 or b.n_samples < 3:
        return 0.5
    diff = 0.0
    # function-word divergence — the stylometric core
    diff += 0.35 * _func_divergence(a, b)
    # vocabulary divergence (down-weighted: topic-contaminated)
    diff += 0.2 * (1.0 - _word_overlap(a, b))
    # vocabulary richness gap
    diff += 0.1 * min(1.0, abs(a.ttr - b.ttr) * 3.0)
    # structural divergence
    len_diff = abs(a.mean_sentence_len - b.mean_sentence_len)
    diff += 0.15 * min(1.0, len_diff / 8.0)
    diff += 0.05 * min(1.0, abs(a.mean_word_len - b.mean_word_len))
    for attr in ("emoji_rate", "question_rate", "exclaim_rate"):
        diff += 0.05 * min(1.0, abs(getattr(a, attr) - getattr(b, attr)) * 2.0)
    # catchphrase habits
    diff += 0.05 * min(1.0, abs(a.catchphrase_hits - b.catchphrase_hits) * 2.0)
    return max(0.0, min(1.0, diff))


def style_report(name: str, fp: VoiceFingerprint) -> str:
    """A character's voice, described like a person — not a stat block.

    God-tier presentation: this is what the owner sees when they ask
    'what does Zara sound like?'
    """
    if fp.n_samples < 3:
        return f"🎙️ {name}: not enough recorded speech yet ({fp.n_samples} samples)."
    lines = [f"🎙️ {name}'s voice — {fp.n_samples} utterances on record:"]
    # rhythm
    if fp.mean_sentence_len >= 14:
        rhythm = "long, winding sentences — thinks out loud"
    elif fp.mean_sentence_len >= 8:
        rhythm = "medium sentences, conversational"
    else:
        rhythm = "short bursts — punchy, lands and moves on"
    lines.append(f"  Rhythm: {rhythm} (~{fp.mean_sentence_len:.0f} words/sentence).")
    # vocabulary
    if fp.ttr >= 0.6:
        vocab = "rich, varied vocabulary — rarely repeats themselves"
    elif fp.ttr >= 0.4:
        vocab = "solid everyday vocabulary"
    else:
        vocab = "tight, repetitive word-hoard — signature phrases over variety"
    lines.append(f"  Vocabulary: {vocab}.")
    if fp.mean_word_len >= 5.2:
        lines.append("  Word choice leans long and considered.")
    elif fp.mean_word_len <= 4.2:
        lines.append("  Word choice leans short and direct.")
    # punctuation personality
    habits = []
    if fp.question_rate >= 0.4:
        habits.append("asks a lot of questions")
    if fp.exclaim_rate >= 0.4:
        habits.append("exclaims — high energy on the page")
    if fp.emoji_rate >= 0.3:
        habits.append("emoji-heavy")
    if habits:
        lines.append("  Habits: " + ", ".join(habits) + ".")
    if fp.catchphrase_hits > 0.05:
        lines.append(f"  Catchphrases land in ~{fp.catchphrase_hits:.0%} of utterances.")
    # function-word fingerprint — the stylometric signature
    top_func = [w for w, _ in fp.function_words[:5]]
    if top_func:
        lines.append("  Signature function words: " + ", ".join(top_func) + ".")
    return "\n".join(lines)
