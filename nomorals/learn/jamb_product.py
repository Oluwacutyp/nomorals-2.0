"""JAMB tutor as a product — ``nm learn jamb`` flow.

Wraps the real engines in :mod:`nomorals.learn` (Socratic tutor, BKT
mastery, FSRS-4.5 flashcards, Ekiti dialect tutor, WAEC/JAMB curriculum,
Chief-Examiners misconception bank). This module adds NO new pedagogy —
it is the product surface: exam/curriculum packs, the answer guard that
closes the leak, and the ``nm learn jamb`` flow.

Universal design: JAMB/WAEC is the Nigeria pack. Other exams (SAT, GCSE,
JEE, …) are pack *descriptors* — structured, honest, empty until a real
syllabus mapping lands. A pack never invents syllabus content.

Answer-guard leak closure: :class:`AttemptGate` refuses to show an answer
until an attempt is recorded. The Socratic engine's own
:meth:`SocraticEngine.guard` / ``sanitize_feedback`` redact leaks in
generated text; the gate closes the structural leak (showing the answer
before the student tried).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .curriculum import (
    EXAM_FORMATS,
    MISCONCEPTION_BANK,
    SUBJECTS,
    SYLLABI,
    find_topic,
    misconceptions_for,
    subject_topics,
    syllabus_code,
)
from .dialect import EKITI_DIALECT_NOTES, MINIMAL_PAIR_SETS, EkitiTutor
from .flashcards import FSRS, MistakeNotebook, card_from_mistake, grade_for_verdict
from .tutor import MasteryModel, SocraticEngine, TutorSession

__all__ = [
    "EXAM_PACKS",
    "EKITI_DESCRIPTOR",
    "AttemptGate",
    "JambMastery",
    "JambFlashcards",
    "curriculum_alignment",
    "misconceptions",
    "misconception_bank_status",
    "ekiti_tutor",
    "start_socratic_session",
    "guard_text",
    "jamb_flow",
    "JAMB_FLOW_USAGE",
]


# ── universal exam / curriculum packs ──────────────────────────────────────
# A pack descriptor: code, region, exam board, subjects, format, status.
# "mapped" = syllabus topics exist in nomorals.learn.curriculum.
# "descriptor_only" = structured placeholder; subjects/format are empty
# and NO syllabus content is invented for it.

def _pack(code: str, region: str, board: str, *, subjects: tuple[str, ...] = (),
          exam_format: dict[str, Any] | None = None, status: str = "descriptor_only",
          note: str = "") -> dict[str, Any]:
    return {"code": code, "region": region, "board": board,
            "subjects": list(subjects),
            "format": exam_format if exam_format is not None else {},
            "status": status, "note": note}


EXAM_PACKS: dict[str, dict[str, Any]] = {
    "JAMB": _pack(
        "JAMB", "Nigeria", "Joint Admissions and Matriculation Board",
        subjects=SUBJECTS, exam_format=EXAM_FORMATS.get("JAMB", {}),
        status="mapped",
        note="UTME format + WAEC/JAMB syllabus topics mapped in learn/curriculum.py."),
    "WAEC": _pack(
        "WAEC", "Nigeria / West Africa", "West African Examinations Council",
        subjects=SUBJECTS, exam_format=EXAM_FORMATS.get("WAEC (generic)", {}),
        status="mapped",
        note="Syllabus topics mapped; per-subject paper schemes use the "
             "generic fallback except WAEC Physics (verified scheme)."),
    "SAT": _pack(
        "SAT", "USA / international", "College Board",
        status="descriptor_only",
        note="Pack descriptor only — no SAT syllabus mapping yet. "
             "Add a real College Board syllabus mapping before tutoring."),
    "GCSE": _pack(
        "GCSE", "UK / international", "AQA / Edexcel / OCR",
        status="descriptor_only",
        note="Pack descriptor only — no GCSE syllabus mapping yet. "
             "Add a real board syllabus mapping before tutoring."),
    "JEE": _pack(
        "JEE", "India", "National Testing Agency",
        status="descriptor_only",
        note="Pack descriptor only — no JEE syllabus mapping yet. "
             "Add a real NTA syllabus mapping before tutoring."),
    "NECO": _pack(
        "NECO", "Nigeria", "National Examinations Council",
        status="descriptor_only",
        note="Pack descriptor only — NECO largely follows the WAEC topic "
             "structure, but no NECO-specific mapping is claimed until "
             "verified against the NECO syllabus."),
}


def exam_pack(code: str) -> dict[str, Any] | None:
    """One pack descriptor (None for unknown codes). Never invents."""
    return EXAM_PACKS.get((code or "").strip().upper())


# ── curriculum alignment (wraps learn/curriculum.py) ─────────────────────────

def curriculum_alignment(subject: str = "", query: str = "") -> dict[str, Any]:
    """Align a query to the WAEC/JAMB syllabus. Wraps ``find_topic``.

    Returns the matched topic (code/title/keywords), the subject's topic
    list, and the pack descriptor — never invented syllabus content.
    """
    topic = find_topic(query, subject=subject) if query else None
    topics = subject_topics(subject) if subject else []
    return {
        "pack": "JAMB/WAEC (Nigeria)",
        "subject": subject,
        "query": query,
        "matched": ({"code": topic.code, "title": topic.title,
                     "syllabus_code": syllabus_code(topic),
                     "keywords": list(topic.keywords),
                     "jamb": topic.jamb}
                    if topic is not None else None),
        "subject_topics": [{"code": t.code, "title": t.title,
                            "jamb": t.jamb} for t in topics],
        "subject_topic_count": len(topics),
    }


# ── misconception bank (wraps the real bank — never invented) ───────────────

def misconception_bank_status() -> dict[str, Any]:
    """Honest bank status: wrapped entries + their examiner sources."""
    return {
        "source": "WAEC Chief Examiners' Reports, via learn/curriculum.py "
                  "MISCONCEPTION_BANK (wrapped, not copied)",
        "entries": len(MISCONCEPTION_BANK),
        "codes": sorted({m.code for m in MISCONCEPTION_BANK}),
        "empty": len(MISCONCEPTION_BANK) == 0,
    }


def misconceptions(query: str) -> list[dict[str, Any]]:
    """Misconceptions relevant to ``query`` (wraps ``misconceptions_for``).

    Each entry carries its examiner source. If the bank were empty this
    returns [] with the status flag — it never invents examiner content.
    """
    return [{"code": m.code, "pattern": m.pattern,
             "correction": m.correction, "probe": m.probe,
             "source": m.source}
            for m in misconceptions_for(query)]


# ── BKT mastery (wraps learn/tutor.py MasteryModel) ──────────────────────────

class JambMastery:
    """BKT mastery tracking for JAMB prep. Delegates to MasteryModel."""

    def __init__(self, skills: list[str] | None = None) -> None:
        self._model = MasteryModel(skills or ["general"])

    def record(self, skill: str, *, correct: bool) -> dict[str, float]:
        """One attempt -> updated linear + BKT mastery for the skill."""
        linear = self._model.update(skill, correct=correct)
        bkt = self._model.bkt_update(skill, correct=correct)
        return {"linear": linear, "bkt_p_knows": bkt,
                "p_correct_next": round(self._model.predict_correct(skill), 3),
                "recall_prob": round(self._model.recall_prob(skill), 3)}

    def bkt(self, skill: str) -> float:
        return self._model.bkt(skill)

    def snapshot(self) -> dict[str, float]:
        return self._model.snapshot()

    def bkt_snapshot(self) -> dict[str, float]:
        return self._model.bkt_snapshot()

    def weakest(self) -> str:
        return self._model.weakest()


# ── FSRS-4.5 flashcards from wrong answers (wraps learn/flashcards.py) ──────

class JambFlashcards:
    """Wrong answers -> FSRS-4.5 cards. Delegates to MistakeNotebook."""

    def __init__(self) -> None:
        self._notebook = MistakeNotebook()
        self._fsrs = FSRS()

    def card_for_wrong_answer(self, question: str, student_answer: str,
                              correct: str, *, topic: str = "",
                              why: str = "") -> dict[str, Any]:
        """Record a wrong answer; returns the card (front/back + FSRS state)."""
        card = self._notebook.record(
            question, student_answer, correct, why=why, topic=topic,
            tags=["jamb", topic] if topic else ["jamb"])
        return {"id": card.id, "front": card.front, "back": card.back,
                "topic": card.topic, "fsrs_state": card.fsrs.state}

    def review(self, card_id: str, verdict: str) -> bool:
        """Grade a review (verdict: wrong/partial/correct)."""
        return self._notebook.review(card_id, grade_for_verdict(verdict))

    def due_cards(self, limit: int = 20) -> list[dict[str, Any]]:
        return [{"id": c.id, "front": c.front, "back": c.back,
                 "topic": c.topic, "fsrs_state": c.fsrs.state}
                for c in self._notebook.due_cards(limit=limit)]

    def stats(self) -> dict[str, Any]:
        return self._notebook.stats()

    @staticmethod
    def standalone_card(question: str, student_answer: str,
                        correct: str) -> dict[str, Any]:
        """One card without a notebook (pure ``card_from_mistake``)."""
        card = card_from_mistake(question, student_answer, correct)
        return {"id": card.id, "front": card.front, "back": card.back}


# ── Ekiti / Ilawe dialect tutoring (the moat) ────────────────────────────────

EKITI_DESCRIPTOR: dict[str, Any] = {
    "dialect": "Ekiti / Ilawe Ekiti (Central Yoruba)",
    "status": "flagship — dialect-level tonal tutoring",
    "verified_core": ("subject-position 'mi' vs Standard 'mo' "
                      "(Ekiti 'mi lo' vs Standard 'mo lo' = I go/went)"),
    "notes": list(EKITI_DIALECT_NOTES),
    "minimal_pair_sets": {k: [{"form": f, "gloss": g, "tone_note": n}
                               for f, g, n in v]
                          for k, v in MINIMAL_PAIR_SETS.items()},
    "honest_limit": ("Full dialect coverage needs native-speaker "
                     "validation; no Ekiti vocabulary is invented."),
}


def ekiti_tutor(llm_fn: Any = None, tts: Any = None) -> EkitiTutor:
    """The real Ekiti tutor (wraps learn/dialect.py EkitiTutor)."""
    return EkitiTutor(llm_fn=llm_fn, tts=tts)


# ── Socratic tutor (wraps learn/tutor.py) ───────────────────────────────────

def start_socratic_session(topic: str, *, mode: str = "socratic",
                           llm_fn: Any = None,
                           expected_answer: str = "") -> TutorSession:
    """Start a Socratic (or direct) tutoring session on ``topic``."""
    return TutorSession(topic, mode=mode, llm_fn=llm_fn,
                        expected_answer=expected_answer)


def guard_text(text: str, expected_answer: str) -> str:
    """Redact + sanitize tutor text so the answer cannot leak."""
    return SocraticEngine.sanitize_feedback(
        SocraticEngine.guard(text, expected_answer), expected_answer)


# ── answer guard: the leak closure ──────────────────────────────────────────
# The structural rule: NO answer is shown until an attempt is recorded
# for that question. The guard holds the expected answer and refuses
# before an attempt; after an attempt it releases it. The pre-attempt
# refusal contains no part of the answer.

@dataclass
class AttemptGate:
    """Per-question answer gate. ``reveal()`` refuses before an attempt."""

    _attempts: dict[str, str] = field(default_factory=dict)
    _answers: dict[str, str] = field(default_factory=dict)

    def register(self, question_id: str, expected_answer: str) -> None:
        """Register a question's answer with the gate (kept server-side)."""
        self._answers[question_id] = expected_answer

    def record_attempt(self, question_id: str, student_answer: str) -> dict[str, Any]:
        """Record the student's attempt. Returns attempt status only."""
        self._attempts[question_id] = student_answer or ""
        return {"question_id": question_id, "attempt_recorded": True,
                "attempts": len(self._attempts)}

    def check_attempt(self, question_id: str) -> dict[str, Any]:
        """Has this question been attempted? Never includes the answer."""
        return {"question_id": question_id,
                "attempted": question_id in self._attempts}

    def reveal(self, question_id: str) -> dict[str, Any]:
        """Release the answer ONLY after an attempt is recorded.

        Before an attempt: ``allowed=False`` and NO answer content.
        After an attempt: ``allowed=True`` with the answer.
        """
        if question_id not in self._attempts:
            return {"question_id": question_id, "allowed": False,
                    "answer": None,
                    "message": ("Attempt first — work through the question "
                                "and submit an answer before the model "
                                "answer is shown.")}
        return {"question_id": question_id, "allowed": True,
                "answer": self._answers.get(question_id, ""),
                "message": "Attempt recorded — here is the model answer."}

    def socratic_feedback(self, question_id: str, student_answer: str,
                          feedback: str) -> dict[str, Any]:
        """Feedback for an attempt, answer-guard sanitized."""
        expected = self._answers.get(question_id, "")
        self.record_attempt(question_id, student_answer)
        return {"question_id": question_id,
                "feedback": guard_text(feedback, expected),
                "revealed": False}


# ── nm learn jamb flow ──────────────────────────────────────────────────────

JAMB_FLOW_USAGE = (
    "nm learn jamb — JAMB tutor (Nigeria pack)\n"
    "  nm learn jamb subjects — JAMB subjects with topic counts\n"
    "  nm learn jamb topics <subject> — syllabus topics for a subject\n"
    "  nm learn jamb align <query> — align a query to the syllabus\n"
    "  nm learn jamb misconceptions <query> — Chief-Examiners misconceptions\n"
    "  nm learn jamb packs — all exam/curriculum packs (JAMB mapped, "
    "others descriptors)\n"
    "  nm learn jamb ekiti — Ekiti/Ilawe dialect tutoring descriptor\n"
    "  nm learn jamb bank — misconception-bank status\n"
    "Answers are never shown before an attempt (AttemptGate)."
)


def jamb_flow(command: str) -> str:
    """The ``nm learn jamb`` flow. Never raises; never leaks answers."""
    try:
        rest = (command or "").strip()
        low = rest.lower()
        if not low or low in ("help", "usage"):
            return JAMB_FLOW_USAGE
        if low == "subjects":
            lines = ["📚 JAMB subjects (Nigeria pack):"]
            for subj in SUBJECTS:
                lines.append(f"• {subj} — {len(subject_topics(subj))} syllabus topics")
            return "\n".join(lines)
        if low.startswith("topics "):
            subj = rest[len("topics "):].strip()
            topics = subject_topics(subj)
            if not topics:
                return (f"no JAMB subject '{subj}' — try: "
                        + ", ".join(SUBJECTS))
            lines = [f"📖 JAMB {subj} — syllabus topics:"]
            for t in topics:
                lines.append(f"• {t.code} — {t.title}"
                             + (" (JAMB)" if t.jamb else ""))
            return "\n".join(lines)
        if low.startswith("align "):
            q = rest[len("align "):].strip()
            hit = curriculum_alignment(query=q)
            m = hit["matched"]
            if m is None:
                return f"no syllabus topic matched '{q}'."
            return (f"📖 aligned: {m['syllabus_code']}\n"
                    f"keywords: {', '.join(m['keywords'])}")
        if low.startswith("misconceptions "):
            q = rest[len("misconceptions "):].strip()
            items = misconceptions(q)
            if not items:
                return f"no Chief-Examiners misconceptions matched '{q}'."
            lines = [f"⚠️ misconceptions for '{q}':"]
            for m in items:
                lines.append(f"• [{m['code']}] {m['correction']}\n"
                             f"  probe: {m['probe']}\n  source: {m['source']}")
            return "\n".join(lines)
        if low == "packs":
            lines = ["🗺️ exam / curriculum packs:"]
            for code, p in EXAM_PACKS.items():
                lines.append(f"• {code} ({p['region']}, {p['board']}) — "
                             f"{p['status']}, {len(p['subjects'])} subjects")
            return "\n".join(lines)
        if low == "ekiti":
            return ("🗣️ Ekiti / Ilawe Ekiti — dialect-level tonal tutoring "
                    "(the moat).\n"
                    f"Verified core: {EKITI_DESCRIPTOR['verified_core']}\n"
                    + "\n".join(f"• {n}" for n in EKITI_DESCRIPTOR["notes"])
                    + f"\n{EKITI_DESCRIPTOR['honest_limit']}")
        if low == "bank":
            st = misconception_bank_status()
            return (f"📦 misconception bank: {st['entries']} entries from "
                    f"{st['source']}\ncodes: {', '.join(st['codes'])}")
        return "unknown jamb subcommand.\n" + JAMB_FLOW_USAGE
    except Exception as exc:  # noqa: BLE001 — flow never raises
        return f"jamb flow hit an error ({exc}).\n{JAMB_FLOW_USAGE}"
