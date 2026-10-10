"""Ekiti / Ilawe Ekiti dialect tutoring — the flagship Yoruba dialect.

**User directive (2026-10-07):** Ekiti/Ilawe Ekiti is the god-tier target
for ALL Yoruba work. Not generic Yoruba. This is the moat — nobody does
dialect-level tonal tutoring.

**Phase 1 (this module):** conversation practice, debate mode, word-level
pronunciation assessment, XTTS reference audio (private stack), real tonal
guidance (Yoruba's three tones are phonemic — pitch changes meaning),
minimal-pair drilling.

**Phase 2 (phoneme-level tonal alignment):** research-grade. The module
exposes :meth:`EkitiTutor.phoneme_feedback` as a real, structured
interface that returns word-level feedback today and per-phoneme scores
when a tonal alignment model is available. Per-phoneme precision is
NEVER faked.

**Honest limits:** the verifiable Ekiti core is the subject-position
``mi`` vs Standard ``mo`` distinction (Ekiti "mi lọ" vs Standard "mo lọ" =
"I go/went"). This module does NOT invent Ekiti vocabulary. Full dialect
coverage needs native-speaker validation. Ambiguous Yoruba leans Ekiti
(flagship bias, documented).

**Linguistics mined (real sources, cited):**
- Tone facts: Shittu (BUCLD), citing Ward 1952, Bamgbose 1966b,
  Akinlabi & Liberman 2000 — three lexical tones H/M/L; H strongest,
  M weakest; L2 learners (English) use only H/L and misread initial-M as H
  and final-M as L (Orie 2006b). The ``igba`` quintuple below is the
  canonical textbook set — it also fixed a real bug: ``igbá`` (MH) is
  'calabash', not 'garden egg' (which is ``ìgbá``, LH).
- Nasal contrasts are phonemic (Ajiboye): àdá/àdán, rù/rùn, ẹrí/ẹ̀rín,
  yẹ/yẹn, ìwọ̀/ìwọ̀n.
- Èkìtì is Central Yorùbá (with Ìjẹ̀ṣà, Ìfẹ̀, Mọ̀bà — Awóbùlúyì 1998);
  Standard's /r/-deletion is largely absent in Èkìtì (Aturamu 2024).
"""

from __future__ import annotations

import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .tutor import MasteryModel

__all__ = [
    "DialectError",
    "DialectUnsupported",
    "EkitiTutor",
    "DialectTurn",
    "PronunciationReport",
    "PhonemeFeedback",
    "WordFeedback",
    "detect_dialect_detail",
    "TONE_GUIDE",
    "YORUBA_MINIMAL_PAIRS",
    "MINIMAL_PAIR_SETS",
    "EKITI_DIALECT_NOTES",
    "SUPPORTED_DIALECTS",
    "tone_pattern",
    "mid_tone_diagnosis",
    "tonal_note_for",
    "arm_check",
    "consume_check",
    "detect_dialect_detail",
    "get_tutor",
    "pending_check",
]

_log = get_logger(__name__)

#: Dialects the tutor accepts. Only Ekiti is fully built — the flagship.
SUPPORTED_DIALECTS = ("ekiti",)

#: Honest roadmap, not fake coverage.
COMING_DIALECTS = ("hausa", "igbo", "yoruba-std")


class DialectError(Exception):
    """Base error for dialect tutoring."""


class DialectUnsupported(DialectError):
    """Raised when a non-flagship dialect is requested."""


# ── dialect detection ────────────────────────────────────────────────────────
# Reuses the verified mi/mo marker banks from voice/money.py — imported LAZILY
# because voice.money ↔ finance.send have a circular import: importing
# nomorals.learn first must not crash. If the import ever fails at runtime, a
# minimal local bank (the documented subject-position mi/mo core) keeps
# detection working, flagged in the result.

_MONEY_IMPORT_ERROR: str | None = None


def _money_markers() -> tuple[Any, Any, Any]:
    """(_EKITI_MARKERS, _YORUBA_STD_MARKERS, detect_language).

    Lazy import; falls back to the documented mi/mo core on failure.
    """
    global _MONEY_IMPORT_ERROR
    try:
        from ..voice.money import (
            _EKITI_MARKERS, _YORUBA_STD_MARKERS, detect_language)
        return _EKITI_MARKERS, _YORUBA_STD_MARKERS, detect_language
    except Exception as exc:  # noqa: BLE001
        _MONEY_IMPORT_ERROR = str(exc)
        _log.debug("voice.money import failed, using local mi/mo core: %s",
                   exc)
        ekiti = ((" mi lọ ", 2.0), (" mi lo ", 1.5), (" mi fẹ́ ", 2.0),
                 (" mi fe ", 1.5), (" mi ní ", 1.5), (" mi ni ", 1.0))
        std = ((" mo lọ ", 2.0), (" mo lo ", 1.5), (" mo fẹ́ ", 2.0),
               (" mo fe ", 1.5), (" mo ní ", 1.5), (" mo ni ", 1.0))
        return ekiti, std, None


def _score(text: str, bank: tuple[tuple[str, float], ...]) -> float:
    return sum(w for phrase, w in bank if phrase in text)


def detect_dialect_detail(text: str) -> dict[str, Any]:
    """(label, confidence, ekiti_score, std_score).

    Reuses the verified ``mi``/``mo`` marker banks from voice/money.py.
    Never raises; empty text → ("other", 0.0, ...).
    """
    try:
        ekiti_bank, std_bank, detect_language = _money_markers()
        lowered = f" {(text or '').lower()} "
        ekiti = _score(lowered, ekiti_bank)
        std = _score(lowered, std_bank)
        if detect_language is not None:
            label, confidence = detect_language(text or "")
        else:  # local fallback scoring
            if ekiti > std:
                label, confidence = "ekiti", ekiti / (ekiti + std)
            elif std > 0:
                label, confidence = "yoruba", std / (ekiti + std)
            else:
                label, confidence = "other", 0.0
        if label not in ("ekiti", "yoruba"):
            label = "other"
        out = {"label": label, "confidence": confidence,
               "ekiti_score": ekiti, "std_score": std}
        if _MONEY_IMPORT_ERROR:
            out["fallback"] = "local mi/mo core (voice.money unavailable)"
        return out
    except Exception:  # noqa: BLE001
        _log.debug("detect_dialect_detail failed", exc_info=True)
        return {"label": "other", "confidence": 0.0,
                "ekiti_score": 0.0, "std_score": 0.0}


# Standard "mo X" → Ekiti "mi X" gentle corrections. Subject-position only —
# bare possessive "mi" ("owo mi" = "my money") exists in Standard too and is
# never "corrected".
_MO_MI_VERBS = ("lọ", "lo", "fẹ́", "fe", "ná", "na", "ti", "ó", "o",
                "ní", "ni", "ṣe", "se", "jẹ", "je", "mọ̀", "mo", "rí", "ri")


def suggest_ekiti_fix(text: str) -> str | None:
    """If the user slipped into Standard "mo + verb", suggest the Ekiti "mi".

    Returns e.g. "in Ilawe we'd say 'mi lọ', not 'mo lọ'" or None.
    """
    lowered = f" {(text or '').lower()} "
    for verb in _MO_MI_VERBS:
        std = f" mo {verb} "
        if std in lowered or lowered.startswith(f"mo {verb} "):
            return (f"in Ilawe we'd say 'mi {verb}', not 'mo {verb}' — "
                    f"'mo' is the Lagos/standard form")
    return None


# ── tonal guidance: real Yoruba linguistics ──────────────────────────────────

TONE_GUIDE = """Yoruba has THREE tones, and they change meaning — pitch is phonemic:

• HIGH (´) — pitch raised, like the stressed beat in English "REcord" (noun).
  Say it like you're calling someone's name across the room.
• MID (¯, unmarked) — your normal speaking pitch, held level. The default.
• LOW (`) — pitch dropped, like a low hum. Let your voice sink.

Physical drill: hum the tone contour FIRST (mm-MM-mm), then put vowels on it.
If a word sounds "flat", you're probably singing every syllable at mid tone —
push the highs up and let the lows drop. Exaggerate at first; natives compress
the range, learners need the full range to be understood.

Research note: H is the strongest/most stable tone, M the weakest
(Akinlabi & Liberman 2000). English speakers learning Yoruba typically use
only H and L, hearing word-initial M as H and word-final M as L (Orie 2006b)
— if your mid tones keep "disappearing", that's the classic pattern to fix."""

#: Canonical textbook minimal pairs — tone alone changes the word.
#: The igba quintuple (Ward 1952; Bamgbose 1966b; Akinlabi & Liberman 2000):
#: MH 'calabash', LL 'time', LH 'garden egg', MM '200', ML 'climbing rope'.
#: (A previous version of this table wrongly glossed igbá as 'garden egg' —
#: igbá is MH = 'calabash'; garden egg is ìgbá, LH. Fixed via mining.)
YORUBA_MINIMAL_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("igbá", "calabash", "MH — mid then HIGH on á"),
    ("ìgbà", "time / season", "LL — low on both"),
    ("ìgbá", "garden egg", "LH — low then HIGH on á"),
    ("igba", "two hundred", "MM — all mid, level pitch"),
    ("igbà", "climbing rope", "ML — mid then LOW on à"),
    ("ọkọ", "hoe (farm tool)", "MM — all mid"),
    ("ọkọ́", "husband / vehicle", "MH — HIGH on final ọ́"),
    ("owó", "money", "MH — HIGH on final ó"),
    ("oko", "farm", "MM — all mid"),
)

#: Nasal contrasts are phonemic in Yoruba (Ajiboye) — a second drill category.
NASAL_MINIMAL_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("àdá", "cutlass", "oral vowel"),
    ("àdán", "bat (animal)", "nasal vowel — n nasalizes the vowel"),
    ("rù", "carry", "oral"),
    ("rùn", "smell", "nasal"),
    ("ẹrí", "witness", "oral"),
    ("ẹ̀rín", "laughter", "nasal"),
    ("yẹ", "be fit / befit", "oral"),
    ("yẹn", "that (one)", "nasal"),
    ("ìwọ̀", "hook", "oral"),
    ("ìwọ̀n", "measuring scale", "nasal"),
)

#: Grouped drill sets: category -> list of (form, gloss, tone note).
MINIMAL_PAIR_SETS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "tone": YORUBA_MINIMAL_PAIRS,
    "nasal": NASAL_MINIMAL_PAIRS,
}

#: Verified Ekiti dialect facts (cited — no invented vocabulary).
EKITI_DIALECT_NOTES: tuple[str, ...] = (
    "Èkìtì is a Central Yorùbá dialect, grouped with Ìjẹ̀ṣà, Ìfẹ̀ and Mọ̀bà "
    "(Awóbùlúyì 1998).",
    "Consonant deletion — especially /r/ deletion, prominent in Standard "
    "Yorùbá — is largely absent in Èkìtì (Aturamu 2024): Ekiti keeps sounds "
    "that Standard drops.",
    "The flagship distinction stays the subject pronoun: Ekiti 'mi lọ' vs "
    "Standard 'mo lọ' (I go/went).",
)


def _strip_tones(word: str) -> str:
    """Remove tone marks regardless of precomposed/combining form."""
    nfd = unicodedata.normalize("NFD", word or "")
    return re.sub(r"[́̀̄]", "", nfd)


def tone_pattern(word: str) -> str:
    """Per-vowel tone pattern of a word, e.g. ``tone_pattern("igbá") ==
    "MH"``. Unmarked vowels count as M (mid is the default)."""
    nfd = unicodedata.normalize("NFD", word or "").lower()
    vowels = set("aeẹioọu")
    out: list[str] = []
    i = 0
    while i < len(nfd):
        ch = nfd[i]
        if ch in vowels:
            tone = "M"
            j = i + 1
            while j < len(nfd) and unicodedata.combining(nfd[j]):
                if nfd[j] == "́":
                    tone = "H"
                elif nfd[j] == "̀":
                    tone = "L"
                j += 1
            out.append(tone)
            i = j
        else:
            i += 1
    return "".join(out)


def mid_tone_diagnosis(heard: str, expected: str) -> str:
    """Orie (2006b) diagnostic: English-speaking learners collapse M tones
    to H/L (initial-M heard as H, final-M as L). Returns a targeted note
    when the heard pattern shows exactly that, else ""."""
    exp_pat = tone_pattern(expected)
    heard_pat = tone_pattern(heard)
    if not exp_pat or not heard_pat or "M" not in exp_pat:
        return ""
    flattened = sum(1 for e, h in zip(exp_pat, heard_pat)
                    if e == "M" and h in ("H", "L"))
    if flattened and heard_pat != exp_pat:
        return ("mid-tone flattening: you sang a mid tone as "
                f"{'high' if heard_pat[0] == 'H' else 'low/high'} — the "
                "classic English-speaker pattern (Orie 2006b). Mid is your "
                "normal speaking pitch, held level — don't push it up or "
                "let it drop.")
    return ""


def tonal_note_for(word: str) -> str:
    """Actionable tonal guidance for a missed word, or "" when none applies."""
    base = (word or "").lower().strip(".,!?;:'\"")
    base_plain = _strip_tones(base)
    for form, gloss, tone in YORUBA_MINIMAL_PAIRS:
        if base == form or base_plain == _strip_tones(form):
            return (f"'{form}' ({gloss}) — {tone}. "
                    f"Humming the contour first helps: get the pitch right "
                    f"before the vowels.")
    for form, gloss, tone in NASAL_MINIMAL_PAIRS:
        if base == form or base_plain == _strip_tones(form):
            return (f"'{form}' ({gloss}) — {tone}. "
                    f"Nasal vs oral changes the word: let the air through "
                    f"your nose on the vowel.")
    nfd = unicodedata.normalize("NFD", word or "")
    if "́" in nfd:
        return ("watch the HIGH tone — raise your pitch on the marked vowel, "
                "don't sing it flat at mid.")
    if "̀" in nfd:
        return ("watch the LOW tone — let your pitch drop on the marked "
                "vowel, like a low hum.")
    return ""


# ── data shapes ──────────────────────────────────────────────────────────────

@dataclass
class DialectTurn:
    """One tutor turn in the dialect session."""
    response: str            # tutor's reply, in Ekiti-marked phrasing
    correction: str = ""     # gentle correction, "" when none needed
    dialect: str = "ekiti"   # detected dialect of the user's input
    mastery: dict[str, float] = field(default_factory=dict)
    debate_round: int = 0


@dataclass
class WordFeedback:
    word: str
    heard: str
    ok: bool
    tonal_note: str = ""
    tone_pattern: str = ""   # expected per-vowel pattern, e.g. "MH"
    tone_miss: bool = False  # right letters, wrong tones


@dataclass
class PronunciationReport:
    accuracy: float          # 0..1 word-level
    words: list[WordFeedback]
    heard: str
    expected: str
    summary: str
    tonal_accuracy: float = 0.0  # 0..1 — tone patterns right (subset of words)


@dataclass
class PhonemeFeedback:
    """Phase-2 interface. Honest about precision.

    ``precision`` is "word" today. Per-phoneme scores arrive when a tonal
    alignment model is available — the ``phonemes`` list stays empty until
    then, and ``note`` says so plainly. Nothing is faked.
    """
    precision: str            # "word" (today) | "phoneme" (future)
    words: list[WordFeedback]
    phonemes: list[dict[str, Any]] = field(default_factory=list)
    note: str = ("per-phoneme tonal alignment is research-grade and not "
                 "available yet — feedback below is word-level, which is "
                 "honest. Phoneme precision upgrades with models.")


# ── the tutor ────────────────────────────────────────────────────────────────

_EKITI_OPENERS = (
    "Ẹ káàárọ̀! Báwo ni? Sọ fún mi — kí ni o fẹ́ kọ́ nípa Èkìtì lónìí?",
    "Káàárọ̀! Mi mọ̀ pé o fẹ́ kọ́ Èkìtì. Kí ni ìbéèrè rẹ?",
)

_DEBATE_SEEDS = (
    "Mi lòdì sí ọ̀rọ̀ yìí — ṣe òótọ́ ni? Fún mi ní ìdí rẹ.",
    "Kò dára tó — fún mi ní ẹ̀rí tó lágbára jù.",
)


class EkitiTutor:
    """Ekiti/Ilawe Ekiti dialect tutor.

    ``dialect``: only "ekiti" is built. Anything else raises
    :class:`DialectUnsupported` with an honest "coming — Ekiti first".
    """

    DIALECT_SKILLS = ("particles (mi vs mo)", "vocabulary",
                      "tonal patterns", "fluency", "nasal contrasts")

    def __init__(
        self,
        dialect: str = "ekiti",
        *,
        llm_fn: Callable[[str, str], str] | None = None,
        tts: Any | None = None,
    ) -> None:
        dialect = (dialect or "ekiti").lower().strip()
        if dialect not in SUPPORTED_DIALECTS:
            if dialect in COMING_DIALECTS:
                raise DialectUnsupported(
                    f"{dialect} tutoring is coming — Ekiti/Ilawe Ekiti first, "
                    f"per the flagship directive. Not faked, not half-built.")
            raise DialectUnsupported(f"unknown dialect {dialect!r}")
        self.dialect = dialect
        self.llm_fn = llm_fn
        self.tts = tts
        self.mastery = MasteryModel(list(self.DIALECT_SKILLS))
        self.id = uuid.uuid4().hex[:12]
        self.turns = 0
        self._debate_round = 0
        self._debate_topic = ""
        self._drill: dict[str, Any] | None = None

    # -- conversation ------------------------------------------------------

    def converse(self, user_text: str) -> DialectTurn:
        """One conversation turn in Ekiti. Corrects gently, tracks mastery."""
        self.turns += 1
        text = (user_text or "").strip()
        detail = detect_dialect_detail(text)
        correction = suggest_ekiti_fix(text) or ""

        # Mastery: reward Ekiti markers, note slips.
        if detail["label"] == "ekiti":
            self.mastery.update("particles (mi vs mo)", correct=True)
            self.mastery.update("fluency", correct=True)
        elif detail["label"] == "yoruba" and correction:
            self.mastery.update("particles (mi vs mo)", correct=False)
        if text:
            self.mastery.update("vocabulary", correct=detail["label"] != "other")

        if self.llm_fn is not None:
            try:
                reply = self.llm_fn(
                    "You are an Ekiti (Ilawe Ekiti) Yoruba dialect tutor. "
                    "Reply IN Ekiti dialect. Keep it to 1-2 sentences, "
                    "conversational, encouraging. Never invent vocabulary "
                    "you are unsure of — prefer the verified mi/mo core.",
                    f"Student said: {text}\n"
                    f"Detected: {detail['label']}. "
                    + (f"Gently note: {correction}" if correction else ""),
                ).strip()
            except Exception:  # noqa: BLE001
                _log.debug("ekiti llm_fn failed", exc_info=True)
                reply = self._fallback_reply(text, correction)
        else:
            reply = self._fallback_reply(text, correction)

        return DialectTurn(response=reply, correction=correction,
                           dialect=detail["label"],
                           mastery=self.mastery.snapshot())

    def _fallback_reply(self, text: str, correction: str) -> str:
        opener = _EKITI_OPENERS[self.turns % len(_EKITI_OPENERS)]
        if not text:
            return opener
        if correction:
            return f"{correction}. Tún sọ ọ́ — báwo ni ọjọ́ rẹ ṣe lọ?"
        return ("Ó dára! Mi gbọ́ ọ. Sọ sí i — kí ni o ṣe lónìí? "
                "(Ó dára = good, mi gbọ́ ọ = I hear you)")

    def dialect_notes(self) -> str:
        """Why Ekiti sounds the way it does — cited dialect facts."""
        return "🗣️ Èkìtì notes:\n" + "\n".join(f"• {n}" for n in EKITI_DIALECT_NOTES)

    # -- minimal-pair drilling -----------------------------------------------

    def drill_pair(self, category: str = "tone") -> dict[str, Any]:
        """Start a minimal-pair drill: which form matches the gloss?

        Returns {"prompt", "options", "answer", "gloss", "category"}; answer
        with :meth:`grade_drill`. Cycles deterministically through the set.
        """
        pairs = MINIMAL_PAIR_SETS.get(category, MINIMAL_PAIR_SETS["tone"])
        idx = self.turns % len(pairs)
        form, gloss, tone_note = pairs[idx]
        # The distractor: the closest other form in the same set.
        others = [p for p in pairs if p[0] != form]
        distractor = min(
            others,
            key=lambda p: abs(len(_strip_tones(p[0])) - len(_strip_tones(form))))
        options = [form, distractor[0]]
        if self.turns % 2:
            options.reverse()
        self._drill = {"answer": form, "gloss": gloss, "tone_note": tone_note,
                       "category": category}
        self.turns += 1
        return {"prompt": f"Which word means '{gloss}'?",
                "options": options, "answer": form, "gloss": gloss,
                "category": category, "tone_note": tone_note}

    def grade_drill(self, choice: str) -> dict[str, Any]:
        """Grade the pending drill. Updates tonal/nasal mastery."""
        drill = self._drill or {}
        self._drill = None
        answer = drill.get("answer", "")
        ok = (choice or "").strip().lower() == answer.lower()
        skill = ("nasal contrasts" if drill.get("category") == "nasal"
                 else "tonal patterns")
        self.mastery.update(skill, correct=ok)
        self.mastery.bkt_update(skill, correct=ok)
        note = "" if ok else tonal_note_for(answer)
        return {"correct": ok, "answer": answer,
                "gloss": drill.get("gloss", ""),
                "tone_note": drill.get("tone_note", ""),
                "feedback": ("Ó dára! " if ok else "Not quite. ")
                            + (note or f"The answer was '{answer}'."),
                "mastery": self.mastery.snapshot()}

    # -- debate (Talkpal pattern) ------------------------------------------

    def debate(self, topic: str) -> DialectTurn:
        """Argue WITH the user in Ekiti — takes the opposing side.

        Escalates complexity each round. With llm_fn: real dialect debate.
        Without: structured scaffolding (honest, documented).
        """
        topic = (topic or "").strip()
        if not topic:
            raise DialectError("debate needs a topic — e.g. /ekiti debate <topic>")
        if topic != self._debate_topic:
            self._debate_topic = topic
            self._debate_round = 0
        self._debate_round += 1
        self.turns += 1

        if self.llm_fn is not None:
            try:
                reply = self.llm_fn(
                    "You are debating IN Ekiti (Ilawe Ekiti) Yoruba dialect. "
                    "Take the OPPOSING side of the student's position. "
                    f"This is round {self._debate_round} — escalate: harder "
                    "arguments, richer vocabulary each round. 2-3 sentences. "
                    "Never invent vocabulary you are unsure of.",
                    f"Debate topic: {topic}",
                ).strip()
            except Exception:  # noqa: BLE001
                _log.debug("ekiti debate llm_fn failed", exc_info=True)
                reply = self._fallback_debate(topic)
        else:
            reply = self._fallback_debate(topic)

        self.mastery.update("fluency", correct=True)
        self.mastery.update("vocabulary", correct=True)
        return DialectTurn(response=reply, dialect="ekiti",
                           mastery=self.mastery.snapshot(),
                           debate_round=self._debate_round)

    def _fallback_debate(self, topic: str) -> str:
        seed = _DEBATE_SEEDS[self._debate_round % len(_DEBATE_SEEDS)]
        return (f"Jẹ́ kí a jiyàn nípa '{topic}'. {seed} "
                f"(Round {self._debate_round} — dáhùn ní Èkìtì!)")

    # -- pronunciation (Phase 1: word-level, honest) ------------------------

    def assess_pronunciation(self, transcript: str,
                             expected_text: str) -> PronunciationReport:
        """Word-level pronunciation assessment.

        ``transcript`` is the STT output (the caller transcribes via the
        voice bridge first). Compares word-by-word against the expected
        text; tonal notes flag minimal-pair risks; a separate tonal_accuracy
        tracks tone patterns (Orie 2006b mid-tone diagnostic included).
        Never raises.
        """
        try:
            heard_words = (transcript or "").lower().split()
            exp_words = (expected_text or "").lower().split()
            words: list[WordFeedback] = []
            hits = 0
            tone_hits = 0
            tone_total = 0
            for i, exp in enumerate(exp_words):
                heard = heard_words[i] if i < len(heard_words) else ""
                ok = heard == exp
                # Tonal near-miss: same letters, different tone marks.
                tonal_miss = (not ok and heard
                              and _strip_tones(heard) == _strip_tones(exp))
                exp_pat = tone_pattern(exp)
                heard_pat = tone_pattern(heard) if heard else ""
                if exp_pat:
                    tone_total += 1
                    if heard_pat == exp_pat:
                        tone_hits += 1
                note = ""
                if tonal_miss:
                    note = tonal_note_for(exp)
                    diag = mid_tone_diagnosis(heard, exp)
                    if diag:
                        note += f"\n  {diag}"
                elif not ok and heard and exp_pat and heard_pat:
                    diag = mid_tone_diagnosis(heard, exp)
                    if diag:
                        note = diag
                if ok:
                    hits += 1
                words.append(WordFeedback(
                    word=exp, heard=heard, ok=ok, tonal_note=note,
                    tone_pattern=exp_pat, tone_miss=tonal_miss))
            accuracy = hits / len(exp_words) if exp_words else 0.0
            tonal_accuracy = tone_hits / tone_total if tone_total else 0.0
            missed = [w.word for w in words if not w.ok]
            summary = (
                f"{hits}/{len(exp_words)} words correct "
                f"({accuracy:.0%}), tones {tone_hits}/{tone_total} "
                f"({tonal_accuracy:.0%})."
                + (f" Work on: {', '.join(missed[:3])}." if missed
                   else " Clean run — ó dára pupọ̀!")
            )
            for w in words:
                self.mastery.update("tonal patterns", correct=w.ok)
            self.mastery.bkt_update("tonal patterns",
                                    correct=tonal_accuracy >= 0.8)
            return PronunciationReport(accuracy=accuracy, words=words,
                                       heard=transcript or "",
                                       expected=expected_text or "",
                                       summary=summary,
                                       tonal_accuracy=round(tonal_accuracy, 3))
        except Exception:  # noqa: BLE001
            _log.debug("assess_pronunciation failed", exc_info=True)
            return PronunciationReport(accuracy=0.0, words=[],
                                       heard=transcript or "",
                                       expected=expected_text or "",
                                       summary="assessment failed — try again.")

    def phoneme_feedback(self, transcript: str,
                         expected_text: str) -> PhonemeFeedback:
        """Phase-2 interface: per-phoneme tonal feedback.

        Returns word-level feedback TODAY (``precision="word"``). The
        ``phonemes`` list populates when a tonal alignment model lands.
        Nothing is faked — the note says so plainly.
        """
        report = self.assess_pronunciation(transcript, expected_text)
        return PhonemeFeedback(precision="word", words=report.words)

    # -- reference audio (XTTS, private stack) -------------------------------

    def reference_audio(self, text: str, *, voice_name: str | None = None,
                        out_path: str = "") -> dict[str, Any]:
        """Native-speaker reference audio for shadowing practice.

        Uses XTTS on the PRIVATE stack (audience="private") — the
        non-commercial license is fine for the owner's own tutoring.
        """
        text = (text or "").strip()
        if not text:
            raise DialectError("reference_audio needs text")
        tts = self.tts
        if tts is None:
            from ..voice.tts import UniversalTTS
            tts = UniversalTTS(backend="xtts", audience="private")
        result = tts.speak(text, voice_name=voice_name, out_path=out_path,
                           audience="private")
        result["dialect"] = "ekiti"
        return result

    def shadowing_drill(self, text: str) -> dict[str, Any]:
        """Repeat-after-me drill: reference audio + the text to shadow.

        Returns {"text", "tone_pattern", "audio"} — the caller plays the
        audio; the student repeats; then /ekiti check scores it.
        """
        text = (text or "").strip()
        if not text:
            raise DialectError("shadowing_drill needs text")
        audio: dict[str, Any] = {}
        try:
            audio = self.reference_audio(text)
        except Exception as exc:  # noqa: BLE001
            audio = {"error": str(exc)}
        return {"text": text,
                "tone_pattern": " ".join(tone_pattern(w) for w in text.split()),
                "audio": audio}


# ── pending pronunciation checks (chat wiring) ───────────────────────────────
# /ekiti check "<expected>" arms a check; the next voice note in that chat
# is transcribed and assessed against it.

_pending: dict[str, dict[str, Any]] = {}

#: Per-chat tutors (debate rounds + mastery persist across turns).
_tutors: dict[str, EkitiTutor] = {}


def get_tutor(chat_key: str, *, llm_fn: Callable[[str, str], str] | None = None,
              tts: Any | None = None) -> EkitiTutor:
    """Per-chat EkitiTutor (debate rounds + mastery persist). Never raises."""
    tutor = _tutors.get(chat_key)
    if tutor is None:
        tutor = EkitiTutor(llm_fn=llm_fn, tts=tts)
        _tutors[chat_key] = tutor
    else:
        if llm_fn is not None:
            tutor.llm_fn = llm_fn
        if tts is not None:
            tutor.tts = tts
    return tutor


def arm_check(chat_key: str, expected_text: str) -> str:
    """Arm a pronunciation check. Returns the instruction for the user."""
    expected = (expected_text or "").strip().strip("\"'")
    if not expected:
        raise DialectError('usage: /ekiti check "<expected text>"')
    _pending[chat_key] = {"expected": expected, "armed_at": time.time()}
    return (f"🎙️ say this: \"{expected}\"\n"
            f"Send a voice note and I'll score your pronunciation.")


def pending_check(chat_key: str) -> dict[str, Any] | None:
    entry = _pending.get(chat_key)
    if entry and time.time() - entry["armed_at"] < 600:
        return entry
    _pending.pop(chat_key, None)
    return None


def consume_check(chat_key: str, transcript: str,
                  *, tutor: EkitiTutor | None = None) -> str | None:
    """Score a voice-note transcript against an armed check. None if none."""
    entry = pending_check(chat_key)
    if not entry:
        return None
    _pending.pop(chat_key, None)
    t = tutor or EkitiTutor()
    report = t.assess_pronunciation(transcript, entry["expected"])
    lines = [f"🗣️ pronunciation: {report.summary}"]
    for w in report.words:
        if not w.ok:
            line = f"• '{w.word}' — you said '{w.heard or '—'}'"
            if w.tone_pattern:
                line += f" (tones: {w.tone_pattern})"
            if w.tonal_note:
                line += f"\n  {w.tonal_note}"
            lines.append(line)
    return "\n".join(lines)
