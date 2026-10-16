"""Prose voiceprint — a numeric fingerprint of how a passage *sounds*.

Descriptor, not prescriber: this module MEASURES a voice. It never
decides whether a voice is good, and it never blocks a chapter for
breaking voice — a break is reported ("told, not blocked") and the owner
overrules it.

Four uses:

* continuation validates — a new chapter's fingerprint is compared
  against ``bible.voice`` (the target fingerprint built from the chapters
  already digested); drift is a number, not a veto;
* chapter drift score — ``compare()`` / ``drift_score()`` between any
  two passages, e.g. chapter 1 vs chapter 40;
* ``bible.voice`` becomes the target — :mod:`nomorals.books.bible`
  stores the fingerprint of the digested chapters as
  ``StoryBible.voiceprint`` and the continuation prompt steers by it;
* the owner's own prose is the default voiceprint — built from the
  owner's messages / notes / letters (see
  :func:`build_owner_voiceprint` and ``collab.CollaborativeSession``),
  so "write like me" means "match this measured fingerprint", not a
  vibe.

Relationship to the existing :mod:`nomorals.books.voice` module:
``voice`` measures a small style vector (sentence length, dialogue
ratio, punctuation rates) for continuity drift. This module is the
full fingerprint on top of it: punctuation habits, register, filtering,
interiority, sensory density by sense, specificity, and per-character
idiolect. It does not replace ``voice`` — ``voice_similarity`` stays
the cheap continuity check, this is the deep one.

Everything is deterministic pure Python over the text in front of it.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "Voiceprint", "fingerprint", "drift_score", "compare",
    "fingerprint_for_character", "build_owner_voiceprint",
    "describe_voiceprint",
    "FILTERING_VERBS", "SENSORY_WORDS",
]

# ── lexicons ─────────────────────────────────────────────────────────────────
#
# Small curated word lists, kept deliberately narrow so the rates are
# honest signals rather than dictionary coverage theatre.  A word not in
# either list is simply uncounted, never guessed at.

FILTERING_VERBS = (
    "saw", "heard", "felt", "noticed", "realized", "realised", "seemed",
    "appeared", "watched", "observed", "knew", "thought", "wondered",
    "decided", "remembered", "could see", "could hear", "could feel",
)

_INTERIORITY_MARKERS = (
    "thought", "wondered", "realized", "realised", "knew", "told herself",
    "told himself", "told themself", "of course", "after all", "somehow",
    "maybe", "perhaps", "she told herself", "he told himself",
    "in her head", "in his head", "to herself", "to himself",
)

# latin-ate suffixes: a cheap, honest proxy for Latinate register.  Words
# carrying one of these endings are counted Latinate; the rest of the
# content words count Anglo-Saxon / Germanic for the ratio.
_LATINATE_SUFFIXES = (
    "tion", "sion", "ity", "ities", "ous", "ive", "ives", "al", "ic",
    "ical", "ism", "ist", "ment", "ance", "ence", "ancy", "ency",
    "ation", "ition", "ization", "isation", "ology", "osis",
)
_LATINATE_WORDS = frozenset("""
    consequence consider decide decision important necessary possible
    probable require requirement sufficient sufficient sufficient
    establish establish communicate communication particular particular
    situation condition experience existence existence
""".split())

_CONCRETE_WORDS = frozenset("""
    stone hand bread knife table chair door window road river cup bowl
    rope nail hammer wood iron glass bottle key coin shirt shoe hat
    bread butter salt pepper milk cheese apple wood fire smoke ash
    dust mud rain snow wind leaf branch root bone skin hair tooth
    eye ear nose mouth finger toe knee elbow shoulder neck back
    hammer saw axe rope chain bucket basket bag pocket boot coat
    candle lamp match matchstick paper pen ink book page letter
    """.split())

_ABSTRACT_WORDS = frozenset("""
    freedom truth love grief hope fear courage justice mercy honour honor
    fate destiny meaning purpose memory desire ambition pride shame
    guilt anger sorrow joy peace war power wisdom folly beauty
    innocence experience existence essence nature spirit soul mind
    """.split())

#: sensory lexicon by sense.  Sight is deliberately NOT the star: the
#: fingerprint reports touch / smell / sound alongside sight so a
#: sight-only book shows up as sight-only.
SENSORY_WORDS: dict[str, frozenset[str]] = {
    "sight": frozenset("""
        red blue green yellow black white grey gray bright dark dim pale
        gleam glitter shine shadow colour color light glow flicker glint
        scarlet crimson amber gold silver copper vivid vivid
        """.split()),
    "sound": frozenset("""
        whisper murmur roar thud bang clang clatter creak groan sigh
        hum buzz rumble thunder echo silence quiet loud soft shrill
        rustle crunch snap crackle hiss thump patter knock bell rang
        ring ringing chime toll song sung voice shout scream cry
        """.split()),
    "smell": frozenset("""
        smoke ash sweat perfume rot stench scent odour odor fragrance
        musty stale fresh pungent acrid sweet sour damp mildew rain
        leather woodsmoke brine salt
        """.split()),
    "touch": frozenset("""
        cold warm hot rough smooth sharp soft hard heavy light wet dry
        sticky gritty silky coarse prickly numb tingling burning
        freezing trembling shivering pressure grip clutch
        """.split()),
    "taste": frozenset("""
        bitter sweet sour salty savoury savory bland metallic spicy
        tart tangy rich stale burnt
        """.split()),
}

_EVASION_MARKERS = (
    "i don't know", "don't know", "maybe", "perhaps", "whatever",
    "never mind", "forget it", "it's nothing", "it's fine",
    "i guess", "suppose", "sort of", "kind of", "not really",
    "let's not", "drop it", "don't worry about it",
)

_KNOWN_BRANDS = frozenset("""
    coke pepsi nike adidas apple samsung sony lego ikea guinness
    heineken nestle cadbury nescafe lipton toyota honda ford bmw
    mercedes rolex casio gucci prada chanel versace
""".split())

_MEASUREMENT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km|inches?|feet|foot|yards?|miles?|"
    r"kg|grams?|pounds?|lbs?|oz|litres?|liters?|ml|seconds?|minutes?|"
    r"hours?|days?|weeks?|months?|years?|dollars?|cents?|euros?|"
    r"naira|pounds?)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_PROPER_NOUN_RE = re.compile(
    r"(?<=[a-z,;: \u201c\u201d])\b([A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]{2,}){0,1})\b")
_DIALOGUE_RE = re.compile(r'"[^"]{2,}"|“[^”]{2,}”')
_DIALOGUE_VERBS = (
    "said", "asked", "replied", "shouted", "whispered", "muttered",
    "growled", "laughed", "sighed", "exclaimed", "demanded", "answered",
    "snarled", "hissed", "yelled", "murmured", "declared", "warned",
)
_VERB_HINTS = frozenset("""
    was were is are had has have did do does went came saw took made
    knew thought felt looked turned walked stood sat lay began became
    brought built caught chose fought found gave grew heard kept left
    lost met paid ran rose spoke stood understood wrote asked replied
    moved opened closed waited listened watched remembered decided
    wanted needed tried called fell hit broke reached pulled pushed
""".split())

_STOP = frozenset(
    "the a an and or of to in on at for with is was were be been are "
    "had has have it its they them he she his her we you your i me my "
    "as by from that this these those not no but so if then than too "
    "who whom what when where why how all any each every some such "
    "own out up down over under again once".split())


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+(?=[A-Z\"“‘(])", text or "")
    return [p.strip() for p in parts if p.strip()]


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text or "")


def _syllables(word: str) -> int:
    word = word.lower().strip("'")
    if len(word) <= 3:
        return 1
    groups = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e"):
        groups = max(1, groups - 1)
    return max(1, groups)


@dataclass
class Voiceprint:
    """The numeric fingerprint of a passage's prose voice.

    All rates are per 1,000 words unless named otherwise, so passages of
    different lengths compare fairly.  ``blocked`` is always False:
    this object describes, it never prescribes.
    """

    words: int = 0
    sentences: int = 0
    # sentence-length distribution
    sentence_len_mean: float = 0.0
    sentence_len_variance: float = 0.0
    pct_under_8_words: float = 0.0
    # paragraph shape
    pct_one_line_paragraphs: float = 0.0
    longest_paragraph_words: int = 0
    paragraph_len_mean: float = 0.0
    # punctuation habits (per sentence unless noted)
    em_dash_per_1k: float = 0.0
    semicolon_per_1k: float = 0.0
    ellipsis_per_1k: float = 0.0
    fragments_per_1k: float = 0.0
    exclam_per_1k: float = 0.0
    question_per_1k: float = 0.0
    # lexical register
    anglo_saxon_rate: float = 0.0
    latinate_rate: float = 0.0
    concrete_per_1k: float = 0.0
    abstract_per_1k: float = 0.0
    type_token_ratio: float = 0.0
    # dialogue
    dialogue_ratio: float = 0.0
    unattributed_dialogue_pct: float = 0.0
    # filtering / interiority
    filtering_per_1k: float = 0.0
    interiority_per_1k: float = 0.0
    # sensory density by sense (per 1k words)
    sensory: dict[str, float] = field(default_factory=dict)
    # specificity (per 1k words)
    specificity_per_1k: float = 0.0
    proper_nouns_per_1k: float = 0.0
    numbers_per_1k: float = 0.0
    brands_per_1k: float = 0.0
    measurements_per_1k: float = 0.0
    # per-character idiolect: name -> {lines, words, vocab_top,
    # sentence_len_mean, evasion_per_1k, type_token_ratio}
    idiolect: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: descriptor contract — always False: a voiceprint never blocks.
    blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["blocked"] = False
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Voiceprint":
        d = dict(d or {})
        d.pop("blocked", None)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def describe(self) -> str:
        """One-paragraph prose description of the measured voice."""
        bits = [
            f"~{self.sentence_len_mean:.0f}-word sentences "
            f"({self.pct_under_8_words:.0%} under 8 words)",
            f"{self.dialogue_ratio:.0%} dialogue "
            f"({self.unattributed_dialogue_pct:.0%} unattributed)",
            f"register {'Latinate' if self.latinate_rate > self.anglo_saxon_rate else 'Anglo-Saxon'}-leaning",
            f"filtering {self.filtering_per_1k:.1f}/1k, "
            f"interiority {self.interiority_per_1k:.1f}/1k",
            f"specificity {self.specificity_per_1k:.1f}/1k",
        ]
        top_sense = max(self.sensory, key=lambda k: self.sensory.get(k, 0.0),
                        default="sight")
        bits.append(f"strongest sense: {top_sense}")
        return "; ".join(bits) + "."


def _is_fragment(sentence: str) -> bool:
    """A fragment: short and carrying no verb hint at all."""
    toks = [w.lower() for w in _words(sentence)]
    if not toks or len(toks) > 9:
        return False
    return not any(t in _VERB_HINTS or t.endswith("ing") or t.endswith("ed")
                   for t in toks)


def _idiolect(text: str, characters: list[str] | None) -> dict[str, dict[str, Any]]:
    """Per-character idiolect from attributed dialogue lines.

    A quoted line counts for a character when the character's name (or
    first name) appears within ~60 chars of the quote next to a dialogue
    verb.  Characters with no attributed lines get no entry — absence
    is data, not a zero.
    """
    names = [c for c in (characters or []) if c]
    if not names:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        first = name.split()[0]
        lines: list[str] = []
        for m in _DIALOGUE_RE.finditer(text):
            start = max(0, m.start() - 70)
            end = min(len(text), m.end() + 70)
            window = text[start:end]
            if re.search(r"\b" + re.escape(first) + r"\b", window) and \
                    any(v in window.lower() for v in _DIALOGUE_VERBS):
                lines.append(m.group(0).strip("\"“”"))
        if not lines:
            continue
        joined = " ".join(lines)
        toks = [w.lower() for w in _words(joined)]
        n = max(1, len(toks))
        content = [w for w in toks if w not in _STOP and len(w) > 2]
        sents = _sentences(joined)
        sent_lens = [len(_words(s)) for s in sents] or [0]
        evasion = sum(joined.lower().count(m) for m in _EVASION_MARKERS)
        out[name] = {
            "lines": len(lines),
            "words": len(toks),
            "vocab_top": [w for w, _ in Counter(content).most_common(8)],
            "sentence_len_mean": round(statistics.fmean(sent_lens), 2),
            "evasion_per_1k": round(evasion / n * 1000, 2),
            "type_token_ratio": round(len(set(toks)) / n, 3),
        }
    return out


def fingerprint(text: str, characters: list[str] | None = None) -> Voiceprint:
    """Measure the voiceprint of ``text``.

    ``characters`` (names) enables per-character idiolect.  Empty text
    yields a zero fingerprint, never a fabricated voice.
    """
    text = text or ""
    words = _words(text)
    n_words = len(words)
    low_words = [w.lower() for w in words]
    sents = _sentences(text)
    paras = _paragraphs(text)
    sent_lens = [len(_words(s)) for s in sents]
    para_lens = [len(_words(p)) for p in paras]
    per_1k = 1000.0 / max(1, n_words)

    # sentence-length distribution
    mean = statistics.fmean(sent_lens) if sent_lens else 0.0
    var = statistics.pvariance(sent_lens) if len(sent_lens) > 1 else 0.0
    pct_short = (sum(1 for x in sent_lens if x < 8) / max(1, len(sent_lens)))

    # paragraph shape: "one-line" = a paragraph of a single short sentence
    one_line = sum(1 for p in paras
                   if len(_sentences(p)) <= 1 and len(_words(p)) <= 14)
    pct_one = one_line / max(1, len(paras))

    # punctuation habits
    em_dash = text.count("—") + text.count("--")
    fragments = sum(1 for s in sents if _is_fragment(s))

    # lexical register: content words only
    content = [w for w in low_words if w not in _STOP and len(w) > 2]
    latinate = sum(
        1 for w in content
        if w in _LATINATE_WORDS or w.endswith(_LATINATE_SUFFIXES))
    lat_rate = latinate / max(1, len(content))
    concrete = sum(1 for w in low_words if w in _CONCRETE_WORDS)
    abstract = sum(1 for w in low_words if w in _ABSTRACT_WORDS)

    # dialogue + attribution
    quoted_chars = sum(len(m.group(0)) for m in _DIALOGUE_RE.finditer(text))
    dialogue_ratio = quoted_chars / max(1, len(text))
    quotes = list(_DIALOGUE_RE.finditer(text))
    unattributed = 0
    for m in quotes:
        window = text[max(0, m.start() - 60):min(len(text), m.end() + 60)]
        if not any(v in window.lower() for v in _DIALOGUE_VERBS):
            unattributed += 1
    unattributed_pct = unattributed / max(1, len(quotes))

    # filtering / interiority
    low = text.lower()
    filtering = sum(len(re.findall(r"\b" + re.escape(v) + r"\b", low))
                    for v in FILTERING_VERBS if " " not in v)
    filtering += sum(low.count(v) for v in FILTERING_VERBS if " " in v)
    interiority = sum(low.count(m) for m in _INTERIORITY_MARKERS)
    # unquoted questions read as free-indirect interiority
    narrative = _DIALOGUE_RE.sub(" ", text)
    interiority += len(re.findall(r"\b\w[^.!?]*\?\s", narrative))

    # sensory density by sense
    sensory = {sense: round(
        sum(1 for w in low_words if w in lex) * per_1k, 2)
        for sense, lex in SENSORY_WORDS.items()}

    # specificity
    proper = len(_PROPER_NOUN_RE.findall(text))
    numbers = len(_NUMBER_RE.findall(text))
    brands = sum(1 for w in low_words if w in _KNOWN_BRANDS)
    measurements = len(_MEASUREMENT_RE.findall(text))
    specificity = (proper + numbers + brands + measurements) * per_1k

    return Voiceprint(
        words=n_words, sentences=len(sents),
        sentence_len_mean=round(mean, 2),
        sentence_len_variance=round(var, 2),
        pct_under_8_words=round(pct_short, 3),
        pct_one_line_paragraphs=round(pct_one, 3),
        longest_paragraph_words=max(para_lens) if para_lens else 0,
        paragraph_len_mean=round(statistics.fmean(para_lens), 2) if para_lens else 0.0,
        em_dash_per_1k=round(em_dash * per_1k, 2),
        semicolon_per_1k=round(text.count(";") * per_1k, 2),
        ellipsis_per_1k=round(text.count("…") * per_1k + text.count("...") * per_1k, 2),
        fragments_per_1k=round(fragments * per_1k, 2),
        exclam_per_1k=round(text.count("!") * per_1k, 2),
        question_per_1k=round(text.count("?") * per_1k, 2),
        anglo_saxon_rate=round(1.0 - lat_rate, 3),
        latinate_rate=round(lat_rate, 3),
        concrete_per_1k=round(concrete * per_1k, 2),
        abstract_per_1k=round(abstract * per_1k, 2),
        type_token_ratio=round(len(set(low_words)) / max(1, n_words), 3),
        dialogue_ratio=round(dialogue_ratio, 3),
        unattributed_dialogue_pct=round(unattributed_pct, 3),
        filtering_per_1k=round(filtering * per_1k, 2),
        interiority_per_1k=round(interiority * per_1k, 2),
        sensory=sensory,
        specificity_per_1k=round(specificity, 2),
        proper_nouns_per_1k=round(proper * per_1k, 2),
        numbers_per_1k=round(numbers * per_1k, 2),
        brands_per_1k=round(brands * per_1k, 2),
        measurements_per_1k=round(measurements * per_1k, 2),
        idiolect=_idiolect(text, characters),
    )


#: numeric voiceprint keys compared by drift_score, with the scale each
#: difference is normalised against (so a 5-word sentence-length gap and
#: a 5/1k filtering gap weigh comparably).
_DRIFT_SCALES: dict[str, float] = {
    "sentence_len_mean": 20.0, "pct_under_8_words": 1.0,
    "pct_one_line_paragraphs": 1.0, "paragraph_len_mean": 80.0,
    "em_dash_per_1k": 10.0, "semicolon_per_1k": 10.0,
    "ellipsis_per_1k": 10.0, "fragments_per_1k": 10.0,
    "latinate_rate": 1.0, "concrete_per_1k": 30.0,
    "abstract_per_1k": 30.0, "type_token_ratio": 1.0,
    "dialogue_ratio": 1.0, "unattributed_dialogue_pct": 1.0,
    "filtering_per_1k": 20.0, "interiority_per_1k": 20.0,
    "specificity_per_1k": 60.0,
}


def drift_score(a: Voiceprint, b: Voiceprint) -> float:
    """0..1 distance between two voiceprints.  0 = same measured voice.

    Sensory densities join the scalar features (each sense normalised
    against a 20/1k scale).  Deterministic; symmetric.
    """
    diffs: list[float] = []
    for key, scale in _DRIFT_SCALES.items():
        av = float(getattr(a, key, 0.0) or 0.0)
        bv = float(getattr(b, key, 0.0) or 0.0)
        diffs.append(min(1.0, abs(av - bv) / scale))
    for sense in SENSORY_WORDS:
        av = float((a.sensory or {}).get(sense, 0.0))
        bv = float((b.sensory or {}).get(sense, 0.0))
        diffs.append(min(1.0, abs(av - bv) / 20.0))
    if not diffs:
        return 0.0
    return round(sum(diffs) / len(diffs), 4)


def compare(new: Voiceprint, target: Voiceprint) -> dict[str, Any]:
    """Compare a fresh passage against a target voiceprint.

    Returns the drift score, the largest per-feature deltas (so the
    owner can see WHAT drifted, not just that something did), and the
    descriptor contract: ``blocked`` is always False — a voice break is
    told, never blocked, and the owner overrules it.
    """
    drift = drift_score(new, target)
    deltas: list[dict[str, Any]] = []
    for key, scale in _DRIFT_SCALES.items():
        av = float(getattr(new, key, 0.0) or 0.0)
        bv = float(getattr(target, key, 0.0) or 0.0)
        deltas.append({"feature": key, "new": round(av, 3),
                       "target": round(bv, 3),
                       "delta": round(abs(av - bv), 3)})
    deltas.sort(key=lambda d: d["delta"], reverse=True)
    # idiolect drift: characters whose sentence length / evasion moved
    idiolect_notes: list[str] = []
    for name, idio in (new.idiolect or {}).items():
        base = (target.idiolect or {}).get(name)
        if not base:
            continue
        gap = abs(idio.get("sentence_len_mean", 0) -
                  base.get("sentence_len_mean", 0))
        if gap > 6:
            idiolect_notes.append(
                f"{name}'s dialogue length moved "
                f"{base.get('sentence_len_mean')} → "
                f"{idio.get('sentence_len_mean')} words/sentence")
    if drift > 0.35:
        verdict = ("voice break — this passage does not sound like the "
                   "target voice. Reported, not blocked: the owner "
                   "overrules.")
    elif drift > 0.18:
        verdict = "noticeable voice drift — worth a look, not a veto."
    else:
        verdict = "on-voice: within the target fingerprint."
    return {
        "drift": drift,
        "verdict": verdict,
        "blocked": False,
        "owner_may_overrule": True,
        "largest_deltas": deltas[:6],
        "idiolect_notes": idiolect_notes,
        "target_description": target.describe(),
        "new_description": new.describe(),
    }


def fingerprint_for_character(text: str, name: str) -> Voiceprint:
    """The voiceprint of one character's attributed dialogue only.

    Falls back to the whole-passage fingerprint when the character has
    no attributed lines — and says so in the result via ``words == 0``
    on the caller side is NOT the case; instead an empty dialogue set
    yields a zero fingerprint, never the passage's voice in disguise.
    """
    lines: list[str] = []
    first = name.split()[0]
    for m in _DIALOGUE_RE.finditer(text or ""):
        start = max(0, m.start() - 70)
        end = min(len(text), m.end() + 70)
        window = text[start:end]
        if re.search(r"\b" + re.escape(first) + r"\b", window) and \
                any(v in window.lower() for v in _DIALOGUE_VERBS):
            lines.append(m.group(0).strip("\"“”"))
    if not lines:
        return Voiceprint()
    return fingerprint(" ".join(lines))


def build_owner_voiceprint(texts: list[str]) -> Voiceprint:
    """The owner's own prose as the default voiceprint.

    ``texts`` are the owner's messages / notes / letters — any prose
    they actually wrote.  They are concatenated and fingerprinted as
    one sample, so the default "write like me" target is measured from
    real owner prose, not from a persona prompt.  Empty input yields a
    zero fingerprint; callers treat ``words == 0`` as "no owner sample
    yet" and fall back to the story's own voice.
    """
    joined = "\n\n".join(t.strip() for t in (texts or []) if (t or "").strip())
    if not joined.strip():
        return Voiceprint()
    return fingerprint(joined)


def describe_voiceprint(vp: Voiceprint) -> str:
    """Short descriptor for a voiceprint (describes, never prescribes)."""
    return vp.describe()
