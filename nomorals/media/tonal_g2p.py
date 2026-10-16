"""Global tonal G2P + tone-aware melody — every language, one architecture.

Tier 1 (generalized): Interface first -> Yoruba G2P -> second G2P to
prove generality -> phrase-level tone melody -> mixed-language per
line -> groove templates.

SCOPE — universal, Nigeria as DEFAULT, not boundary. The model is N
packs: Nigeria is the default pack (owner context, least-served, best
place to start), never the boundary. Sources are nation-generic; media
culture packs ship Nigeria-by-default with every other nation as an
equal pack.

THE MOAT is tone languages, not Yoruba. Yoruba is the FIRST and
HARDEST test case, not the moat itself — the majority of the world
speaks a tonal language. Tone systems covered:

* Yoruba / Igbo / Twi / Ewe / Fon — 2–3 levels + downstep + vowel harmony
* Hausa — 2 levels + falling + downstep
* Mandarin — 4 + neutral
* Cantonese — 6–9 (hardest: 6 tones + 3 checked entering tones)
* Vietnamese — 6 + VOICE QUALITY (creaky / breathy, not just pitch)
* Thai / Lao — 5
* Punjabi — 3 (rising / level / falling)
* Zulu / Shona / Somali — 2–3

ARCHITECTURE — G2P descriptors per language + ONE language-agnostic
tone-aware melody engine. The engine consumes (phoneme, tone[,
quality]) units and rewrites the contour; it NEVER knows the language.
Adding a language = adding a G2P descriptor, not a new subsystem.
The voice-quality slot is in the interface from day one (Vietnamese
needs it; retrofit is expensive). The second G2P (Mandarin) was
written early on purpose: if the engine needed changes for language
#2, the abstraction was wrong — it didn't.

What this module does
----------------------
* :func:`g2p` — word -> phoneme list, per language, rule-based.
* :func:`syllabify` — word -> syllables, per language syllable shape.
* :func:`tone_of` / :func:`tones_of` — syllable -> tone level, per
  language tone system (see table above).
* :func:`quality_of` — syllable -> voice quality (modal / creaky /
  breathy / glottalized / short). Vietnamese ngã/nặng need it.
* :func:`units_of` — lyric line -> list[:class:`ToneUnit`], the
  language-agnostic interface: (phoneme, tone[, quality]).
* :func:`check_units` / :func:`fix_units` — the ONE tone-aware melody
  engine. Takes units + melody, never a language string.
* :func:`tone_check` / :func:`tone_fix_melody` — thin wrappers that
  build units from (text, language) and delegate to the engine.
* :func:`detect_languages` — per-LINE mixed-language detection for
  code-switching (Yoruba / Pidgin / English and beyond): script +
  diacritic + marker-lexicon evidence, deterministic, no model needed.

Yoruba specifics (first and hardest test case for the tone-language moat, spelled out)
----------------------------------------
* 7 oral vowels: a e ẹ i o ọ u (+ nasal vowels an ẹn in ọn un).
* Tone marks: acute = HIGH, grave = LOW, unmarked = MID. Marks ride on
  the vowel (or syllabic nasal m/n).
* Consonants: ṣ (sh), gb and kp are single digraph phonemes (labial-
  velar stops), j, y, w; no consonant clusters in native words.
* Syllable shapes: V, CV, syllabic N (m/n carrying tone). Nasal vowel
  = vowel + n written as one nucleus.
* Vowel harmony: within a native word, vowels agree in ATR-ish sets
  {e,o} vs {ẹ,ọ} (a/i/u neutral). Used as a *detection* signal and a
  *validation* warning, never to rewrite the user's spelling.
* Tone sandhi (phrase-level, approximate): a MID tone immediately
  before a HIGH tone in the same phrase tends to rise; a LOW tone
  before HIGH may assimilate upward in fast speech. :func:`sandhi`
  reports the predicted surface tones; the melody check uses surface
  tones, because singers sing the surface.

Honest limits: rule-based G2P is exact for shallow orthographies
(Yoruba, Swahili, Spanish, Vietnamese pinyin-style) and approximate
for deep ones (English, French, Thai script). Thai tone assignment
uses consonant-class + mark + live/dead rules and is labelled
approximate in the result. Mandarin/Cantonese/Vietnamese G2P assumes
romanized input (pinyin / jyutping / quốc ngữ); Han characters are
detected but not converted (no dictionary bundled).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "LanguageProfile",
    "LANGUAGE_PROFILES",
    "profile_for",
    "syllabify",
    "g2p",
    "tone_of",
    "tones_of",
    "quality_of",
    "ToneUnit",
    "units_of",
    "check_units",
    "fix_units",
    "sandhi",
    "detect_languages",
    "detect_line_languages",
    "tone_check",
    "tone_fix_melody",
    "ToneMismatch",
    "YORUBA_VOWELS",
    "check_vowel_harmony",
]

# ── Yoruba constants ────────────────────────────────────────────────────────

YORUBA_VOWELS = ("a", "e", "ẹ", "i", "o", "ọ", "u")
#: ATR-ish harmony sets (a/i/u neutral, appear with either set)
_HARMONY_SET_A = frozenset("eo")
_HARMONY_SET_B = frozenset("ẹọ")
_ACUTE = "\u0301"
_GRAVE = "\u0300"
_MACRON = "\u0304"
_TILDE = "\u0303"
_DOT_BELOW = "\u0323"

# ── tone systems ────────────────────────────────────────────────────────────
# Tone levels are on a 0..4 scale so cross-language comparison is uniform:
# 0 = extra-low / low, 1 = low-mid, 2 = mid, 3 = high-mid, 4 = high.
# Contour tones are (start, end) pairs; level tones are single ints.


@dataclass(frozen=True)
class LanguageProfile:
    """Descriptor for one language's sound system. Data, not code."""

    code: str                    # "yo", "ig", "zh", ...
    name: str                    # "Yoruba"
    tonal: bool                  # does lexical tone change meaning?
    tone_system: str             # "level-3", "level-2", "contour-4", "none", ...
    tone_levels: tuple[int, ...]  # levels used, low→high
    script: str = "latin"        # "latin" | "han" | "thai" | "arabic" | ...
    syllable_shape: str = "CV"   # dominant shape: "CV", "CVC", "complex"
    unmarked_tone: int | None = 2  # tone assumed when no mark present
    digraphs: tuple[str, ...] = ()  # multi-letter single phonemes
    nasal_vowels: bool = False
    notes: str = ""


LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    # ── the default pack: Nigerian / West African tonal languages ──
    "yo": LanguageProfile("yo", "Yoruba", True, "level-3", (0, 2, 4),
                          syllable_shape="CV", unmarked_tone=2,
                          digraphs=("gb", "kp", "ṣ", "sh"),
                          nasal_vowels=True,
                          notes="7 vowels a e ẹ i o ọ u; acute=HIGH grave=LOW unmarked=MID; ṣ gb kp; CV/V/syllabic-N; nasal vowels; vowel harmony; tone sandhi. FIRST and HARDEST test case for the tone-language moat — not the moat itself."),
    "ekiti": LanguageProfile("ekiti", "Ekiti (Yoruba dialect)", True,
                             "level-3", (0, 2, 4), syllable_shape="CV",
                             unmarked_tone=2, digraphs=("gb", "kp", "ṣ", "sh"),
                             nasal_vowels=True,
                             notes="Yoruba system; dialect vowel/consonant variants ride on the same tone engine."),
    "ig": LanguageProfile("ig", "Igbo", True, "level-2+downstep", (0, 4),
                          syllable_shape="CV", unmarked_tone=0,
                          digraphs=("gb", "kp", "gw", "kw", "nw", "ch", "sh", "ny"),
                          nasal_vowels=True,
                          notes="Two tones HIGH/LOW + downstep (!H); unmarked syllables default LOW in most marking conventions."),
    "ee": LanguageProfile("ee", "Ewe", True, "level-3+contour+downstep", (0, 2, 4),
                          syllable_shape="CV", unmarked_tone=2,
                          digraphs=("gb", "kp", "dz", "ts", "ny", "ŋ"),
                          notes="Three level tones + rising/falling contours + downstep; open syllables dominate."),
    "tw": LanguageProfile("tw", "Twi/Akan", True, "level-2+downstep", (0, 4),
                          syllable_shape="CV", unmarked_tone=2,
                          digraphs=("dw", "gy", "hy", "ky", "kw", "nw", "tw", "ny"),
                          notes="Two tones HIGH/LOW + downstep; tone terracing (downstep) matters in phrases."),
    "fon": LanguageProfile("fon", "Fon", True, "level-2+contour+downstep", (0, 4),
                           syllable_shape="CV", unmarked_tone=2,
                           digraphs=("gb", "kp", "hw", "ny", "x"),
                           notes="Two level tones + rising/falling contours + downstep; nasal vowels."),
    "ha": LanguageProfile("ha", "Hausa", True, "level-2+falling+downstep", (0, 4),
                          syllable_shape="CVC", unmarked_tone=2,
                          digraphs=("sh", "ts", "kw", "gw", "ky", "gy", "'y", "ɓ", "ɗ", "ƙ"),
                          notes="HIGH/LOW + falling (HL) tone + downstep; long/short vowel length is phonemic; implosives ɓ ɗ ƙ."),
    "pcm": LanguageProfile("pcm", "Nigerian Pidgin", False, "none", (),
                           syllable_shape="CV", unmarked_tone=None,
                           digraphs=("gb", "kp", "sh", "ch"),
                           notes="Not tonal — stress/pitch accent only. Tone check always passes."),
    # ── other tonal languages worldwide ──
    "zh": LanguageProfile("zh", "Mandarin", True, "contour-4+neutral",
                          (0, 1, 2, 3, 4), script="latin-pinyin",
                          syllable_shape="CVC", unmarked_tone=None,
                          notes="4 tones + neutral: 1 high-level 2 rising 3 dipping 4 falling. Input: pinyin with tone numbers (ma1) or diacritics (mā)."),
    "yue": LanguageProfile("yue", "Cantonese", True, "contour-6+checked",
                           (0, 1, 2, 3, 4), script="latin-jyutping",
                           syllable_shape="CVC", unmarked_tone=None,
                           notes="6 tones (+3 checked entering tones on -p/-t/-k). Input: jyutping with tone numbers (nei5)."),
    "vi": LanguageProfile("vi", "Vietnamese", True, "contour-6", (0, 1, 2, 3, 4),
                          syllable_shape="CVC", unmarked_tone=2,
                          notes="6 tones marked by diacritics on the vowel: ngang (unmarked, mid) huyền (grave, low-falling) sắc (acute, high-rising) hỏi (hook, dipping) ngã (tilde, broken-rising) nặng (dot-below, low-short)."),
    "th": LanguageProfile("th", "Thai", True, "contour-5", (0, 1, 2, 3, 4),
                          script="thai", syllable_shape="CVC",
                          unmarked_tone=None,
                          notes="5 tones. APPROXIMATE: needs consonant class + mark + live/dead syllable; romanized input uses tone marks 1-5 (maa1..maa5)."),
    "my": LanguageProfile("my", "Burmese", True, "contour-4", (0, 1, 2, 3),
                          script="myanmar", syllable_shape="CVC",
                          unmarked_tone=None,
                          notes="4 tones (low, high, creaky, checked). Romanized input: tone numbers."),
    "km": LanguageProfile("km", "Khmer", False, "none", (),
                          script="khmer", syllable_shape="CVC",
                          notes="Not tonal (registers instead)."),
    "lo": LanguageProfile("lo", "Lao", True, "contour-5", (0, 1, 2, 3, 4),
                          script="lao", syllable_shape="CVC",
                          unmarked_tone=None,
                          notes="5 tones (Thai/Lao family); romanized input with tone numbers."),
    "pa": LanguageProfile("pa", "Punjabi", True, "contour-3", (0, 2, 4),
                          syllable_shape="CVC", unmarked_tone=2,
                          digraphs=("bh", "ch", "dh", "gh", "jh", "kh", "ph",
                                    "th", "ḍ", "ṇ", "ṛ", "ś"),
                          notes="3 tones: rising / level / falling. Historically from lost voiced aspirates; romanized input uses tone numbers (pa1 rising, pa2 level, pa3 falling) or diacritics."),
    "sn": LanguageProfile("sn", "Shona", True, "level-3", (0, 2, 4),
                          syllable_shape="CV", unmarked_tone=2,
                          digraphs=("bv", "dz", "mb", "nd", "ng", "ny", "nz",
                                    "sv", "ts", "zv", "sh", "ch"),
                          notes="High / Mid / Low tones; downstep-like terracing in phrases."),
    "so": LanguageProfile("so", "Somali", True, "level-3-pitch-accent", (0, 2, 4),
                          syllable_shape="CVC", unmarked_tone=2,
                          digraphs=("dh", "kh", "sh"),
                          notes="Pitch-accent system with High / Mid / Low distinctions; tone carries grammatical (gender/case) contrasts."),
    "zu": LanguageProfile("zu", "Zulu", True, "level-3+downstep", (0, 2, 4),
                          syllable_shape="CV", unmarked_tone=2,
                          digraphs=("hl", "dl", "kh", "ng", "ny", "sh", "th",
                                    "bh", "ph", "gc", "gq", "gx", "nc", "nq",
                                    "nx", "qh", "xh", "c", "q", "x"),
                          notes="High / Mid / Low tones + downstep; clicks hl/dl/q/gq kept as digraph phonemes."),
    # ── non-tonal languages (same architecture, tone check passes) ──
    "en": LanguageProfile("en", "English", False, "none", (),
                          syllable_shape="complex",
                          notes="Deep orthography: rule-based G2P is APPROXIMATE. Stress, not tone, drives singability."),
    "es": LanguageProfile("es", "Spanish", False, "none", (),
                          syllable_shape="CVC",
                          notes="Shallow orthography; G2P near-exact."),
    "fr": LanguageProfile("fr", "French", False, "none", (),
                          syllable_shape="CVC",
                          notes="Rule-based G2P approximate (liaison, silent letters)."),
    "de": LanguageProfile("de", "German", False, "none", (),
                          syllable_shape="complex",
                          notes="Rule-based G2P approximate (umlaut, ch)."),
    "pt": LanguageProfile("pt", "Portuguese", False, "none", (),
                          syllable_shape="CVC", nasal_vowels=True,
                          notes="Nasal vowels ã õ; G2P approximate."),
    "it": LanguageProfile("it", "Italian", False, "none", (),
                          syllable_shape="CVC", notes="Shallow; G2P near-exact."),
    "sw": LanguageProfile("sw", "Swahili", False, "none", (),
                          syllable_shape="CV",
                          notes="Shallow orthography; G2P near-exact."),
    "hi": LanguageProfile("hi", "Hindi", False, "none", (),
                          script="devanagari", syllable_shape="CVC",
                          notes="Romanized input assumed; Devanagari not converted."),
    "ja": LanguageProfile("ja", "Japanese", False, "pitch-accent", (),
                          script="kana", syllable_shape="CV",
                          notes="Pitch accent, not lexical tone: tone check passes; accent handled by stress layer."),
    "ko": LanguageProfile("ko", "Korean", False, "none", (),
                          script="hangul", syllable_shape="CVC",
                          notes="Romanized input assumed."),
    "ar": LanguageProfile("ar", "Arabic", False, "none", (),
                          script="arabic", syllable_shape="CVC",
                          notes="Romanized input assumed; emphatic consonants kept as digraphs."),
    "ru": LanguageProfile("ru", "Russian", False, "none", (),
                          script="cyrillic", syllable_shape="complex",
                          notes="Romanized input assumed."),
    "am": LanguageProfile("am", "Amharic", False, "none", (),
                          script="ethiopic", syllable_shape="CVC",
                          notes="Romanized input assumed."),
    "xh": LanguageProfile("xh", "Xhosa", False, "none", (),
                          syllable_shape="CV",
                          notes="Clicks c/q/x kept as phonemes."),
}

# Back-compat aliases
_ALIASES = {"yoruba": "yo", "igbo": "ig", "ewe": "ee", "twi": "tw",
            "akan": "tw", "hausa": "ha", "pidgin": "pcm",
            "mandarin": "zh", "chinese": "zh", "cantonese": "yue",
            "vietnamese": "vi", "thai": "th", "english": "en",
            "spanish": "es", "french": "fr", "german": "de",
            "portuguese": "pt", "italian": "it", "swahili": "sw",
            "hindi": "hi", "japanese": "ja", "korean": "ko",
            "arabic": "ar", "russian": "ru", "amharic": "am",
            "zulu": "zu", "xhosa": "xh", "fon": "fon", "burmese": "my",
            "khmer": "km", "lao": "lo", "yorùbá": "yo",
            "punjabi": "pa", "shona": "sn", "somali": "so", "zulu": "zu"}


def profile_for(language: str) -> LanguageProfile:
    """Resolve a language name/code to its descriptor. Unknown -> English."""
    key = (language or "en").strip().lower()
    key = _ALIASES.get(key, key)
    return LANGUAGE_PROFILES.get(key, LANGUAGE_PROFILES["en"])


# ── normalization helpers ───────────────────────────────────────────────────


def _nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text or "")


def _strip_marks(text: str) -> str:
    """Strip tone/accent combining marks — but NOT dot-below (U+0323).

    Dot-below is part of the letters ẹ/ọ/ṣ in Yoruba (and part of the
    tone mark in Vietnamese nặng). Stripping it would turn ẹ→e, ọ→o,
    ṣ→s and destroy Yoruba's phoneme inventory. Callers that need
    Vietnamese tone-stripping remove U+0323 explicitly.
    """
    return unicodedata.normalize(
        "NFC", re.sub(r"[\u0300-\u0302\u0304-\u0322\u0324-\u036f]", "",
                      _nfd(text)))


def _marks_of(syllable: str) -> str:
    out = []
    for ch in _nfd(syllable):
        if unicodedata.combining(ch):
            out.append(ch)
    return "".join(out)


# ── syllabification ─────────────────────────────────────────────────────────


_YORUBA_VOWEL_SET = frozenset("aeẹioọuAEẸIOỌU")
_ASCII_VOWELS = frozenset("aeiouAEIOU")


def syllabify(word: str, language: str = "yo") -> list[str]:
    """Split a word into syllables per the language's syllable shape.

    Yoruba-family (CV): each syllable = onset consonants + one vowel
    nucleus, or a syllabic nasal (m/n carrying tone, not followed by a
    vowel). Nasal vowel (vowel + n) stays one nucleus. Digraphs gb/kp/ṣ
    never split.
    Other Latin-script languages: vowel-nucleus split with consonant
    clusters attached to the following onset when possible; a trailing
    consonant cluster stays with the last syllable (CVC).
    """
    prof = profile_for(language)
    w = (word or "").strip()
    if not w:
        return []
    if prof.code in ("yo", "ekiti", "ig", "ee", "tw", "fon", "pcm", "sw",
                     "sn", "zu"):
        return _syllabify_cv(w, prof)
    if prof.code in ("zh", "yue"):
        # Romanized: one pinyin/jyutping syllable per token already.
        return [w]
    return _syllabify_generic(w)


def _syllabify_cv(word: str, prof: LanguageProfile) -> list[str]:
    groups: list[tuple[str, str]] = []
    for ch in _nfd(word):
        if unicodedata.combining(ch):
            if groups:
                b, m = groups[-1]
                groups[-1] = (b, m + ch)
            else:
                groups.append(("", ch))
        else:
            groups.append((ch, ""))
    syllables: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    i = 0
    n = len(groups)
    while i < n:
        base, marks = groups[i]
        bl = base.lower()
        cur.append((base, marks))
        if bl in _YORUBA_VOWEL_SET or bl in _ASCII_VOWELS:
            # Nasal vowel: vowel + n (no tone mark on the n) stays together.
            if (i + 1 < n and groups[i + 1][0].lower() == "n"
                    and not groups[i + 1][1] and prof.nasal_vowels):
                # only absorb the n if what follows is not a vowel
                nxt2 = groups[i + 2][0].lower() if i + 2 < n else ""
                if nxt2 not in _YORUBA_VOWEL_SET and nxt2 not in _ASCII_VOWELS:
                    cur.append(groups[i + 1])
                    i += 1
            syllables.append(cur)
            cur = []
        elif bl in ("m", "n"):
            nxt = groups[i + 1][0].lower() if i + 1 < n else ""
            if nxt not in _YORUBA_VOWEL_SET and nxt not in _ASCII_VOWELS:
                syllables.append(cur)  # syllabic nasal
                cur = []
        i += 1
    if cur:
        if syllables:
            syllables[-1].extend(cur)
        else:
            syllables.append(cur)
    return [unicodedata.normalize("NFC", "".join(b + m for b, m in s))
            for s in syllables]


def _syllabify_generic(word: str) -> list[str]:
    """Vowel-nucleus split for CVC/complex languages (approximation)."""
    chars = list(word)
    # find vowel nuclei (a vowel group = one nucleus)
    nuclei = [i for i, c in enumerate(chars)
              if c.lower() in "aeiouáéíóúàèìòùâêîôûäëïöüāēīōū"]
    if not nuclei:
        return [word]
    if len(nuclei) == 1:
        return [word]
    bounds = []
    start = 0
    for k in range(len(nuclei) - 1):
        # split the inter-nuclear consonants: all but last go left
        gap_start, gap_end = nuclei[k] + 1, nuclei[k + 1]
        gap = gap_end - gap_start
        split = gap_end - 1 if gap >= 2 else gap_end
        bounds.append((start, split))
        start = split
    bounds.append((start, len(chars)))
    return ["".join(chars[a:b]) for a, b in bounds if b > a]


# ── G2P ─────────────────────────────────────────────────────────────────────


def g2p(word: str, language: str = "yo") -> list[str]:
    """Word -> phoneme list (rule-based, per-language).

    Yoruba-family: orthography is shallow, so phonemes ≈ letters with
    digraphs (gb/kp/ṣ/sh/ch) kept whole and tone marks stripped into
    the tone layer (see :func:`tone_of`), not the phoneme layer.
    Mandarin: pinyin syllable -> [initial, final] (tone digit stripped).
    Cantonese: jyutping syllable -> [onset, nucleus+coda].
    Vietnamese: quốc ngữ syllable -> [onset, rhyme] (tone diacritic
    stripped). English/French/German: letter-cluster approximation —
    documented as approximate; a real English bank needs a lexicon.
    """
    prof = profile_for(language)
    w = (word or "").strip()
    if not w:
        return []
    if prof.code in ("yo", "ekiti", "ig", "ee", "tw", "fon", "ha", "pcm",
                     "sw", "es", "it", "sn", "zu", "so"):
        return _g2p_shallow(w, prof)
    if prof.code == "zh":
        return _g2p_pinyin(w)
    if prof.code == "yue":
        return _g2p_jyutping(w)
    if prof.code == "vi":
        return _g2p_vietnamese(w)
    if prof.code in ("th", "lo", "my", "pa"):
        return _g2p_tone_numbered(w)
    return _g2p_approx(w)


def _g2p_shallow(word: str, prof: LanguageProfile) -> list[str]:
    plain = _strip_marks(word).lower()
    phonemes: list[str] = []
    i = 0
    digraphs = sorted(prof.digraphs, key=len, reverse=True)
    while i < len(plain):
        matched = ""
        for d in digraphs:
            if plain.startswith(d, i):
                matched = d
                break
        if matched:
            phonemes.append(matched)
            i += len(matched)
            continue
        ch = plain[i]
        # nasal vowel: vowel + n at syllable end -> "ṽ" style marker
        if (ch in "aeiouẹọ" and i + 1 < len(plain) and plain[i + 1] == "n"
                and prof.nasal_vowels):
            nxt = plain[i + 2] if i + 2 < len(plain) else ""
            if nxt not in "aeiouẹọ":
                phonemes.append(ch + "̃")  # nasalized vowel phoneme
                i += 2
                continue
        phonemes.append(ch)
        i += 1
    return phonemes


def _g2p_pinyin(word: str) -> list[str]:
    """Pinyin syllable (ma1, mā, zhong4) -> [initial, final]."""
    m = re.match(r"^([a-zü]+?)(\d)?$", _strip_marks(word).lower())
    if not m:
        return [word]
    syl = m.group(1)
    initials = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
                "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w")
    for init in initials:
        if syl.startswith(init) and len(syl) > len(init):
            return [init, syl[len(init):]]
    return [syl]


def _g2p_jyutping(word: str) -> list[str]:
    m = re.match(r"^([a-z]+?)(\d)$", word.lower())
    if not m:
        return [word]
    syl = m.group(1)
    initials = ("gw", "kw", "ng", "b", "p", "m", "f", "d", "t", "n", "l",
                "g", "k", "h", "z", "c", "s", "j", "w")
    for init in initials:
        if syl.startswith(init) and len(syl) > len(init):
            return [init, syl[len(init):]]
    return [syl]


def _g2p_vietnamese(word: str) -> list[str]:
    plain = _strip_marks(word).lower()
    # Vietnamese nặng tone = dot-below: strip it here (Yoruba keeps it).
    plain = unicodedata.normalize(
        "NFC", re.sub(r"[\u0323]", "", unicodedata.normalize("NFD", plain)))
    # strip tone diacritics already done; onset = leading consonants
    m = re.match(r"^([^aeiouy]+)?(.*)$", plain)
    if not m:
        return [plain]
    onset, rhyme = m.group(1) or "", m.group(2) or ""
    out = []
    if onset:
        out.append(onset)
    if rhyme:
        out.append(rhyme)
    return out or [plain]


def _g2p_tone_numbered(word: str) -> list[str]:
    m = re.match(r"^([a-z]+?)(\d)?$", word.lower())
    if not m:
        return [word]
    return [m.group(1)]


def _g2p_approx(word: str) -> list[str]:
    """Letter-cluster approximation for deep orthographies."""
    plain = _strip_marks(word).lower()
    out: list[str] = []
    i = 0
    clusters = ("tch", "sch", "ch", "sh", "th", "ph", "gh", "ng", "ck",
                "qu", "ou", "ow", "ee", "oo", "ea", "ai", "ay", "ei", "ey")
    while i < len(plain):
        matched = ""
        for c in clusters:
            if plain.startswith(c, i):
                matched = c
                break
        if matched:
            out.append(matched)
            i += len(matched)
        else:
            out.append(plain[i])
            i += 1
    return out


# ── tones ───────────────────────────────────────────────────────────────────


def tone_of(syllable: str, language: str = "yo") -> int | tuple[int, int]:
    """Tone level of one syllable (0=low .. 4=high), or (start, end) contour.

    Yoruba-family: acute=HIGH(4), grave=LOW(0), macron/unmarked=MID(2)
    (Igbo unmarked=LOW(0)). Mandarin: pinyin tone number/diacritic ->
    1: (4,4) level-high, 2: (2,4) rising, 3: (2,0) dipping-ish,
    4: (4,0) falling, 5/neutral: (2,2). Cantonese: jyutping number ->
    1 (4,4) 2 (3,4) 3 (2,2) 4 (0,1) 5 (1,2) 6 (1,1). Vietnamese:
    ngang (2,2) huyền (1,0) sắc (3,4) hỏi (2,0) ngã (2,4 broken)
    nặng (0,0). Thai romanized maa1..maa5: 1 mid 2 low 3 falling 4 high
    5 rising (numbering per standard Thai romanization).
    Non-tonal: returns the unmarked/default (2) — callers must check
    ``profile_for(lang).tonal`` before treating it as lexical tone.
    """
    prof = profile_for(language)
    syl = syllable or ""
    if prof.code in ("zh",):
        return _mandarin_tone(syl)
    if prof.code == "yue":
        return _cantonese_tone(syl)
    if prof.code == "vi":
        return _vietnamese_tone(syl)
    if prof.code in ("th", "lo", "my"):
        return _numbered_tone(syl, prof.code)
    if prof.code == "pa":
        return _punjabi_tone(syl)
    marks = _marks_of(syl)
    if _ACUTE in marks:
        return 4
    if _GRAVE in marks:
        return 0
    if _MACRON in marks:
        return 2
    if prof.code == "ig":
        return 0  # Igbo unmarked defaults LOW
    if prof.unmarked_tone is not None:
        return prof.unmarked_tone
    return 2


def _tone_level(t) -> float:
    if isinstance(t, tuple):
        return (t[0] + t[1]) / 2.0
    return float(t)


def _tone_dir(t) -> int:
    if isinstance(t, tuple):
        return 1 if t[1] > t[0] else (-1 if t[1] < t[0] else 0)
    return 0


_PINYIN_DIACRITIC_TONE = {
    "ā": 1, "á": 2, "ǎ": 3, "à": 4,
    "ē": 1, "é": 2, "ě": 3, "è": 4,
    "ī": 1, "í": 2, "ǐ": 3, "ì": 4,
    "ō": 1, "ó": 2, "ǒ": 3, "ò": 4,
    "ū": 1, "ú": 2, "ǔ": 3, "ù": 4,
    "ǖ": 1, "ǘ": 2, "ǚ": 3, "ǜ": 4,
}
_MANDARIN_CONTOURS = {1: (4, 4), 2: (2, 4), 3: (2, 0), 4: (4, 0), 5: (2, 2)}


def _mandarin_tone(syl: str) -> tuple[int, int]:
    m = re.search(r"(\d)$", syl.strip())
    if m:
        return _MANDARIN_CONTOURS.get(int(m.group(1)), (2, 2))
    for ch in syl:
        if ch in _PINYIN_DIACRITIC_TONE:
            return _MANDARIN_CONTOURS[_PINYIN_DIACRITIC_TONE[ch]]
    return (2, 2)  # neutral / unmarked


_CANTONESE_CONTOURS = {1: (4, 4), 2: (3, 4), 3: (2, 2),
                       4: (0, 1), 5: (1, 2), 6: (1, 1)}


def _cantonese_tone(syl: str) -> tuple[int, int]:
    m = re.search(r"(\d)$", syl.strip())
    if m:
        return _CANTONESE_CONTOURS.get(int(m.group(1)), (2, 2))
    return (2, 2)


def _vietnamese_tone(syl: str) -> tuple[int, int]:
    marks = _marks_of(syl)
    if _ACUTE in marks:
        return (3, 4)   # sắc — high rising
    if _GRAVE in marks:
        return (1, 0)   # huyền — low falling
    if "\u0309" in marks:  # hook above
        return (2, 0)   # hỏi — dipping
    if _TILDE in marks:
        return (2, 4)   # ngã — broken rising
    if _DOT_BELOW in marks:
        return (0, 0)   # nặng — low, short
    return (2, 2)       # ngang — mid level


_THAI_NUMBERED = {1: (2, 2), 2: (0, 0), 3: (4, 0), 4: (4, 4), 5: (2, 4)}


def _numbered_tone(syl: str, code: str) -> tuple[int, int]:
    m = re.search(r"(\d)$", syl.strip())
    if m and code == "th":
        return _THAI_NUMBERED.get(int(m.group(1)), (2, 2))
    if m:
        return (int(m.group(1)), int(m.group(1)))
    return (2, 2)


_PUNJABI_CONTOURS = {1: (2, 4), 2: (2, 2), 3: (4, 0)}  # rising/level/falling


def _punjabi_tone(syl: str) -> tuple[int, int]:
    m = re.search(r"(\d)$", syl.strip())
    if m:
        return _PUNJABI_CONTOURS.get(int(m.group(1)), (2, 2))
    return (2, 2)


def quality_of(syllable: str, language: str = "yo") -> str:
    """Voice quality of one syllable: modal / creaky / breathy / glottalized / short.

    Vietnamese is the driver: ngã is creaky (glottalized rising),
    nặng is short + glottalized, huyền is breathy, hỏi is creaky,
    sắc/ngang are modal. Cantonese checked (entering) tones — jyutping
    1/3/6 on a -p/-t/-k coda — are short. Everything else defaults to
    modal. The quality slot exists in :class:`ToneUnit` from day one
    so the vocal stage can render it (creaky/breathy voice) instead of
    reducing every tone to pitch.
    """
    prof = profile_for(language)
    syl = syllable or ""
    if prof.code == "vi":
        marks = _marks_of(syl)
        if _TILDE in marks:
            return "creaky"       # ngã
        if _DOT_BELOW in marks:
            return "glottalized"  # nặng — short, glottalized
        if _GRAVE in marks:
            return "breathy"      # huyền
        if "\u0309" in marks:
            return "creaky"       # hỏi
        return "modal"
    if prof.code == "yue":
        if re.search(r"[ptk]\d$", syl.strip().lower()):
            return "short"        # checked / entering tone
        return "modal"
    return "modal"


@dataclass
class ToneUnit:
    """ONE language-agnostic unit for the tone-aware melody engine.

    This is the interface: the engine consumes ToneUnits and NEVER
    knows the language. ``phoneme`` is the joined G2P string for the
    syllable; ``tone`` is a level (0..4) or a (start, end) contour;
    ``quality`` is the voice-quality slot (modal/creaky/breathy/
    glottalized/short) — Vietnamese needs it, everyone gets it.
    """

    phoneme: str
    tone: int | tuple[int, int]
    quality: str = "modal"
    syllable: str = ""

    @property
    def level(self) -> float:
        return _tone_level(self.tone)

    @property
    def contour_dir(self) -> int:
        return _tone_dir(self.tone)


def units_of(text: str, language: str = "yo") -> list[ToneUnit]:
    """Lyric line -> ToneUnits (the language-agnostic interface).

    G2P + tone + quality per syllable. Sandhi applies where modelled
    (Yoruba): the unit carries the SURFACE tone, because singers sing
    the surface.
    """
    surf = sandhi(text, language)
    units: list[ToneUnit] = []
    for item in surf:
        syl = item["syllable"]
        phonemes = g2p(syl, language)
        units.append(ToneUnit(phoneme="".join(phonemes),
                              tone=item["surface"],
                              quality=quality_of(syl, language),
                              syllable=syl))
    return units
    """Tone of every syllable in a lyric line."""
    out: list[int | tuple[int, int]] = []
    for word in (text or "").split():
        for syl in syllabify(word, language):
            out.append(tone_of(syl, language))
    return out


def check_vowel_harmony(word: str) -> dict[str, Any]:
    """Yoruba vowel-harmony check: does the word mix {e,o} with {ẹ,ọ}?

    Returns {ok, sets_found, note}. a/i/u are neutral. Loanwords and
    names legitimately break harmony — this is a warning, not an error.
    """
    plain = _strip_marks(word).lower()
    vowels = [c for c in plain if c in "eoẹọ"]
    sets = set()
    if any(c in _HARMONY_SET_A for c in vowels):
        sets.add("e/o")
    if any(c in _HARMONY_SET_B for c in vowels):
        sets.add("ẹ/ọ")
    return {"ok": len(sets) <= 1, "sets_found": sorted(sets),
            "note": ("" if len(sets) <= 1 else
                     "mixes e/o with ẹ/ọ — possible loanword, name, or typo")}


def tones_of(text: str, language: str = "yo") -> list[int | tuple[int, int]]:
    """Tone of every syllable in a lyric line."""
    out: list[int | tuple[int, int]] = []
    for word in (text or "").split():
        for syl in syllabify(word, language):
            out.append(tone_of(syl, language))
    return out


def sandhi(text: str, language: str = "yo") -> list[dict[str, Any]]:
    """Phrase-level tone sandhi: predicted SURFACE tones.

    Yoruba approximation: a MID tone immediately before a HIGH tone in
    the same phrase surfaces raised (2 -> 3); a LOW before HIGH in fast
    speech may surface raised (0 -> 1). Other languages: surface =
    lexical (no sandhi rule modelled — returned as-is, honestly).
    Singers sing the surface, so the melody check uses this output.
    """
    prof = profile_for(language)
    syls: list[str] = []
    for word in (text or "").split():
        syls.extend(syllabify(word, language))
    lexical = [tone_of(s, language) for s in syls]
    surface = list(lexical)
    if prof.code in ("yo", "ekiti"):
        for i in range(len(lexical) - 1):
            nxt = _tone_level(lexical[i + 1])
            cur = _tone_level(lexical[i])
            if nxt >= 4 and cur == 2:
                surface[i] = 3  # MID raised before HIGH
            elif nxt >= 4 and cur == 0:
                surface[i] = 1  # LOW raised before HIGH (fast speech)
    return [{"syllable": s, "lexical": lex, "surface": sur}
            for s, lex, sur in zip(syls, lexical, surface)]


# ── mixed-language detection ────────────────────────────────────────────────


_YORUBA_MARKERS = frozenset(
    "ṣ ẹ ọ á à é è í ì ó ò ú ù ń ǹ mọ nǹkan ṣe".split())
_YORUBA_WORDS = frozenset(
    "mo ni ko se ti o wa nla rere ife ayo orun aye omo baba iya oko".split())
_PIDGIN_WORDS = frozenset(
    "na dey go come chop wahala abeg jare sapa gbedu ehn oga madam".split())
_IGBO_WORDS = frozenset(
    "nna nwanne obi ego ife unwa afa ndu ezigbo dalu".split())
_HAUSA_WORDS = frozenset(
    "ina gida ruwa yau gobe sai dai lafiya godiya".split())
_SPANISH_WORDS = frozenset("el la los las que con por para una esto".split())
_FRENCH_WORDS = frozenset("le la les des que une dans pour avec".split())


def _line_language_scores(line: str) -> dict[str, float]:
    text = (line or "")
    low = _strip_marks(text).lower()
    words = re.findall(r"[a-zà-ỹẹọṣ]+", low)
    scores: dict[str, float] = {}
    if not words:
        return scores
    # Diacritic evidence: Yoruba/Vietnamese-style marks on Latin text.
    nfd = _nfd(text)
    if "ẹ" in text or "ọ" in text or "ṣ" in text:
        scores["yo"] = scores.get("yo", 0.0) + 3.0
    if _ACUTE in nfd or _GRAVE in nfd:
        scores["yo"] = scores.get("yo", 0.0) + 0.5
        scores["vi"] = scores.get("vi", 0.0) + 0.5
    if re.search(r"[0-9]$", text.strip()) and re.search(
            r"\b[a-z]+[1-5]\b", low):
        scores["zh"] = scores.get("zh", 0.0) + 2.0
        scores["yue"] = scores.get("yue", 0.0) + 1.0
    for w in words:
        if w in _YORUBA_WORDS:
            scores["yo"] = scores.get("yo", 0.0) + 1.0
        if w in _PIDGIN_WORDS:
            scores["pcm"] = scores.get("pcm", 0.0) + 1.5
        if w in _IGBO_WORDS:
            scores["ig"] = scores.get("ig", 0.0) + 1.0
        if w in _HAUSA_WORDS:
            scores["ha"] = scores.get("ha", 0.0) + 1.0
        if w in _SPANISH_WORDS:
            scores["es"] = scores.get("es", 0.0) + 1.0
        if w in _FRENCH_WORDS:
            scores["fr"] = scores.get("fr", 0.0) + 1.0
        if w in ("the", "and", "you", "your", "love", "baby", "girl", "boy"):
            scores["en"] = scores.get("en", 0.0) + 0.7
    # Script evidence
    if re.search(r"[\u4e00-\u9fff]", text):
        scores["zh"] = scores.get("zh", 0.0) + 3.0
    if re.search(r"[\u0e00-\u0e7f]", text):
        scores["th"] = scores.get("th", 0.0) + 3.0
    if re.search(r"[\u3040-\u30ff]", text):
        scores["ja"] = scores.get("ja", 0.0) + 3.0
    if re.search(r"[\uac00-\ud7a3]", text):
        scores["ko"] = scores.get("ko", 0.0) + 3.0
    if re.search(r"[\u0600-\u06ff]", text):
        scores["ar"] = scores.get("ar", 0.0) + 3.0
    if re.search(r"[\u0400-\u04ff]", text):
        scores["ru"] = scores.get("ru", 0.0) + 3.0
    return scores


def detect_line_languages(line: str) -> list[dict[str, Any]]:
    """Per-line language detection for code-switching.

    Returns a list of {language, score, confidence} sorted by score.
    Empty = no evidence (caller should fall back to the song's default
    language). Deterministic; marker-lexicon + script + diacritic.
    """
    scores = _line_language_scores(line)
    if not scores:
        return []
    best = max(scores.values())
    out = [{"language": k, "score": round(v, 2),
            "confidence": round(v / best, 2) if best else 0.0}
           for k, v in sorted(scores.items(), key=lambda kv: -kv[1])]
    return out


def detect_languages(text: str) -> list[dict[str, Any]]:
    """Per-line detection for a whole lyric block.

    Returns [{line, languages: [...]}, ...]. A line with 2+ languages
    above threshold is code-switched — the caller should run G2P and
    the tone check per line with its dominant language, and per-word
    when a line mixes.
    """
    out = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        langs = detect_line_languages(line)
        out.append({"line": line, "languages": langs,
                    "code_switched": len([l for l in langs
                                          if l["confidence"] >= 0.5]) >= 2})
    return out


# ── tone ↔ melody (global) ──────────────────────────────────────────────────


@dataclass
class ToneMismatch:
    index: int
    syllable_a: str
    syllable_b: str
    tone_dir: int
    note_a: int
    note_b: int
    mel_dir: int
    language: str = "yo"

    @property
    def message(self) -> str:
        tdir = {1: "rises", -1: "falls", 0: "stays level"}[self.tone_dir]
        mdir = {1: "rises", -1: "falls", 0: "stays level"}[self.mel_dir]
        return (f"'{self.syllable_a}' → '{self.syllable_b}' ({self.language}): "
                f"tone {tdir} but melody {mdir}")


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def check_units(units: list[ToneUnit], melody: list) -> dict[str, Any]:
    """THE tone-aware melody engine — language-agnostic.

    Consumes (phoneme, tone[, quality]) units + melody notes. Never
    sees a language string. A pair mismatches only when tone and
    melody move in OPPOSITE directions; level moves are flexible.
    For contour tones, the contour's own direction also counts: a
    rising contour tone against a falling melody is a mismatch.
    Voice quality does NOT change the pitch verdict (creaky vs modal
    is a phonation difference, not a pitch difference) — it rides
    along on the unit for the vocal stage.
    """
    notes = [(int(p), float(s), float(d)) for p, s, d in (melody or [])]
    mismatches: list[ToneMismatch] = []
    pairs = min(len(units) - 1, len(notes) - 1)
    for i in range(max(0, pairs)):
        tdir = _sign(units[i + 1].level - units[i].level)
        if tdir == 0:
            tdir = units[i + 1].contour_dir or units[i].contour_dir
        mdir = _sign(notes[i + 1][0] - notes[i][0])
        if tdir != 0 and mdir != 0 and tdir != mdir:
            mismatches.append(ToneMismatch(
                index=i, syllable_a=units[i].syllable or units[i].phoneme,
                syllable_b=units[i + 1].syllable or units[i + 1].phoneme,
                tone_dir=tdir, note_a=notes[i][0], note_b=notes[i + 1][0],
                mel_dir=mdir, language=""))
    score = 1.0 if pairs <= 0 else 1.0 - len(mismatches) / pairs
    return {"ok": not mismatches, "score": round(score, 3),
            "mismatches": [m.message for m in mismatches],
            "mismatch_objects": mismatches}


def fix_units(units: list[ToneUnit], melody: list,
              max_nudge: int = 3) -> list:
    """Rewrite clashing notes so melody follows the units' contour.

    Same contract as the old language-specific fix, language-free:
    for each mismatch, move the second note at least level with the
    first in the tone direction, capped at max_nudge past the
    compliant floor. Input untouched; returns a new melody list.
    """
    check = check_units(units, melody)
    if check["ok"]:
        return list(melody)
    fixed = [(int(p), float(s), float(d)) for p, s, d in melody]
    for m in check["mismatch_objects"]:
        i = m.index
        orig = fixed[i + 1][0]
        if m.tone_dir == 1:
            floor = fixed[i][0]
            target = min(max(orig, floor + 1), floor + 1 + max_nudge)
        else:
            ceil = fixed[i][0]
            target = max(min(orig, ceil - 1), ceil - 1 - max_nudge)
        fixed[i + 1] = (target, fixed[i + 1][1], fixed[i + 1][2])
    return fixed


def tone_check(lyrics: str, melody: list, language: str = "yo") -> dict[str, Any]:
    """Global tone↔melody check — thin wrapper over :func:`check_units`.

    Builds units from (text, language) via :func:`units_of` (G2P +
    tone + quality + sandhi), then the language-agnostic engine does
    the checking. Non-tonal languages return score 1.0.
    """
    prof = profile_for(language)
    if not prof.tonal:
        return {"ok": True, "score": 1.0, "mismatches": [],
                "language": prof.code, "tonal": False}
    result = check_units(units_of(lyrics, language), melody)
    result["language"] = prof.code
    result["tonal"] = True
    for m in result["mismatch_objects"]:
        m.language = prof.code
    return result


def tone_fix_melody(lyrics: str, melody: list, language: str = "yo",
                    max_nudge: int = 3) -> list:
    """Tone-compliant melody rewrite — thin wrapper over :func:`fix_units`."""
    prof = profile_for(language)
    if not prof.tonal:
        return list(melody)
    return fix_units(units_of(lyrics, language), melody, max_nudge)


def register(registry: Any) -> None:
    """Tool registration: ``tonal_g2p`` — global G2P + tone check."""
    from ..core.policy import Capability

    @registry.register(
        "tonal_g2p",
        description=(
            "Global tonal G2P + tone↔melody (tone languages are the moat; "
            "Yoruba is the first/hardest test case): action=g2p (word, "
            "language → phonemes) | tones | detect | check | fix | units "
            "(text, language → language-agnostic ToneUnits). "
        ),
        capability=Capability.FS_WRITE,
    )
    def tonal_g2p(action: str = "g2p", word: str = "", text: str = "",
                  language: str = "yo", lyrics: str = "",
                  melody: str = "") -> dict:
        import json as _json
        action = (action or "g2p").lower()
        if action == "g2p":
            return {"ok": True, "phonemes": g2p(word, language),
                    "syllables": syllabify(word, language),
                    "language": profile_for(language).code}
        if action == "tones":
            return {"ok": True, "sandhi": sandhi(text, language),
                    "language": profile_for(language).code}
        if action == "detect":
            return {"ok": True, "lines": detect_languages(text)}
        if action == "units":
            return {"ok": True, "language": profile_for(language).code,
                    "units": [{"phoneme": u.phoneme,
                               "tone": list(u.tone) if isinstance(u.tone, tuple) else u.tone,
                               "quality": u.quality, "syllable": u.syllable}
                              for u in units_of(text, language)]}
        if action in ("check", "fix"):
            try:
                mel = _json.loads(melody) if melody else []
            except Exception as exc:  # noqa: BLE001
                from ..core.errors import ToolError
                raise ToolError(f"melody must be JSON: {exc}")
            if action == "check":
                r = tone_check(lyrics, mel, language)
                r.pop("mismatch_objects", None)
                return {"ok": True, **r}
            return {"ok": True,
                    "melody": tone_fix_melody(lyrics, mel, language),
                    "language": profile_for(language).code}
        from ..core.errors import ToolError
        raise ToolError(f"unknown action {action!r}")
