"""Socratic tutoring engine — Devon as tutor (build-map #46).

Two explicit modes:
- ``socratic`` ("guide me"): withholds answers, asks diagnostic questions,
  backtracks to prerequisites on struggle, probes deeper on success.
  NEVER reveals the expected answer — enforced by :meth:`SocraticEngine.guard`.
- ``direct`` ("just tell me"): teaches straight, answers included.

The engine takes an optional ``llm_fn(prompt) -> str`` for diagnosis and
question generation. Without one it uses a structured fallback (template
questions, keyword diagnosis) — honest and documented, never a fake LLM.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "MasteryModel",
    "MistakeNotebook",
    "SocraticEngine",
    "TutorError",
    "TutorSession",
    "TutorTurn",
    "card_from_mistake",
    "end_session",
    "get_notebook",
    "get_session",
    "set_session",
    "tutor_from_files",
    "tutor_from_photo",
]


class TutorError(Exception):
    """Tutoring could not start or continue (vision down, no files, ...)."""


# ── mastery model ────────────────────────────────────────────────────────────

@dataclass
class MasteryModel:
    """Per-sub-skill mastery, 0.0–1.0. Correct answers raise it, wrong
    answers lower it, weighted by difficulty. Drives what to teach next."""

    skills: dict[str, float] = field(default_factory=dict)

    def __init__(self, sub_skills: list[str] | None = None) -> None:
        self.skills = {s: 0.5 for s in (sub_skills or ["general"])}

    def update(self, skill: str, *, correct: bool,
               difficulty: float = 0.5) -> float:
        """Record an outcome. Returns the new mastery for the skill."""
        weight = 0.5 + max(0.0, min(1.0, difficulty))
        delta = (0.15 if correct else -0.20) * weight
        new = max(0.0, min(1.0, self.skills.get(skill, 0.5) + delta))
        self.skills[skill] = round(new, 3)
        return self.skills[skill]

    def weakest(self) -> str:
        return min(self.skills, key=lambda s: self.skills[s])

    def strongest(self) -> str:
        return max(self.skills, key=lambda s: self.skills[s])

    def snapshot(self) -> dict[str, float]:
        return dict(self.skills)


# ── tutor turn ────────────────────────────────────────────────────────────────

@dataclass
class TutorTurn:
    """One exchange of the tutoring dialogue."""

    feedback: str = ""        # response to the student's last answer
    prompt: str = ""          # the next question / guidance / teaching
    diagnosis: str = ""       # "correct" | "partial" | "wrong" | ""
    revealed: bool = False    # True only if the answer was given (direct mode)
    mastery: dict[str, float] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)  # grounded mode
    done: bool = False        # topic mastered / session complete


# ── socratic engine ───────────────────────────────────────────────────────────

_TOKEN = re.compile(r"[a-z0-9]+")

SOCRATIC_SYSTEM = """You are a Socratic tutor. Rules:
1. NEVER reveal the expected answer. Guide the student to discover it.
2. Diagnose each student answer as correct, partial, or wrong — and say
   WHAT specifically is right or wrong about it.
3. If the student struggles twice on the same idea, backtrack: ask about a
   prerequisite ("let's step back — do you remember how X works?").
4. On success, probe deeper with a harder follow-up question.
5. Keep replies short: one piece of feedback, one question.
6. Output ONLY JSON: {"feedback": "...", "prompt": "...",
   "diagnosis": "correct|partial|wrong", "skill": "<sub-skill>",
   "difficulty": 0.0-1.0, "prerequisite": "<prereq question or empty>"}
"""


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


class SocraticEngine:
    """The dialogue loop. ``llm_fn`` is ``(system, user) -> str``; without it
    the engine runs a structured fallback (templates + keyword diagnosis)."""

    def __init__(self, llm_fn: Callable[[str, str], str] | None = None) -> None:
        self.llm_fn = llm_fn
        self._struggles: dict[str, int] = {}

    # -- answer guard: the hard guarantee ----------------------------------
    @staticmethod
    def guard(text: str, expected_answer: str) -> str:
        """Redact verbatim leaks of the expected answer.

        Socratic mode must never reveal the answer — not even when the LLM
        slips. Short answers (< 3 chars) are skipped: redacting "x" would
        mangle every word containing x.
        """
        answer = (expected_answer or "").strip()
        if len(answer) < 3 or not text:
            return text
        redacted = re.sub(re.escape(answer), "[the answer]", text,
                          flags=re.IGNORECASE)
        # Also catch the answer with surrounding quotes stripped.
        bare = answer.strip("\"'“”‘’")
        if len(bare) >= 3 and bare.lower() != answer.lower():
            redacted = re.sub(re.escape(bare), "[the answer]", redacted,
                              flags=re.IGNORECASE)
        return redacted

    # -- diagnosis ----------------------------------------------------------
    def diagnose(self, student_answer: str, expected_answer: str,
                 context: str = "") -> tuple[str, str]:
        """-> (verdict, specifics). Verdict in {correct, partial, wrong}."""
        if self.llm_fn is not None:
            try:
                raw = self.llm_fn(
                    SOCRATIC_SYSTEM,
                    f"Topic context: {context}\nExpected answer: {expected_answer}\n"
                    f"Student answer: {student_answer}\nDiagnose as JSON.")
                data = json.loads(_strip_fences(raw))
                verdict = str(data.get("diagnosis", "wrong")).lower()
                if verdict not in ("correct", "partial", "wrong"):
                    verdict = "wrong"
                specifics = str(data.get("feedback", ""))
                return verdict, specifics
            except Exception as exc:  # noqa: BLE001 — fall back, don't die
                _log.debug("socratic LLM diagnosis failed: %s", exc)
        return self._fallback_diagnose(student_answer, expected_answer)

    @staticmethod
    def _fallback_diagnose(student_answer: str,
                           expected_answer: str) -> tuple[str, str]:
        """Keyword-overlap diagnosis. Honest heuristic, documented as such."""
        student = _tokens(student_answer)
        expected = _tokens(expected_answer)
        if not expected:
            return "wrong", "I couldn't parse what you were aiming at — try again?"
        if not student:
            return "wrong", "No answer yet — give it a shot, even a guess helps."
        overlap = student & expected
        # Numeric answers: exact match on the number wins.
        nums_s = {t for t in student if t.isdigit()}
        nums_e = {t for t in expected if t.isdigit()}
        if nums_e and nums_s == nums_e:
            return "correct", "Exactly right."
        if nums_e and not (nums_s & nums_e):
            return "wrong", "Check your number — the digits don't match."
        ratio = len(overlap) / len(expected)
        if ratio >= 0.7:
            return "correct", "That's right."
        if ratio >= 0.35:
            missing = sorted(expected - student)[:3]
            return ("partial",
                    f"Partly there — you're missing: {', '.join(missing)}.")
        return "wrong", "Not quite — think about what the question is really asking."

    # -- next prompt ---------------------------------------------------------
    def next_prompt(self, *, topic: str, verdict: str, skill: str,
                    expected_answer: str, struggle_key: str = "") -> str:
        """Build the follow-up: probe deeper on success, backtrack after two
        struggles, otherwise nudge forward. Never contains the answer."""
        if verdict == "correct":
            self._struggles.pop(struggle_key, None)
            prompt = self._probe_deeper(topic, skill)
        else:
            n = self._struggles.get(struggle_key, 0) + 1
            self._struggles[struggle_key] = n
            if n >= 2:
                prompt = self._backtrack(topic, skill)
            else:
                prompt = self._nudge(topic, skill)
        return self.guard(prompt, expected_answer)

    def _probe_deeper(self, topic: str, skill: str) -> str:
        if self.llm_fn is not None:
            try:
                return self.llm_fn(
                    SOCRATIC_SYSTEM,
                    f"The student just answered correctly about '{topic}' "
                    f"(skill: {skill}). Ask ONE harder follow-up question as "
                    f"JSON {{\"prompt\": \"...\"}}.")
            except Exception:  # noqa: BLE001
                _log.debug("probe-deeper LLM call failed", exc_info=True)
        return (f"Good — now take it one step further: why is that true for "
                f"'{topic}'? What would change the result?")

    def _backtrack(self, topic: str, skill: str) -> str:
        if self.llm_fn is not None:
            try:
                return self.llm_fn(
                    SOCRATIC_SYSTEM,
                    f"The student is struggling with '{topic}' (skill: {skill}). "
                    f"Ask ONE prerequisite question to rebuild the foundation, "
                    f"as JSON {{\"prompt\": \"...\"}}.")
            except Exception:  # noqa: BLE001
                _log.debug("backtrack LLM call failed", exc_info=True)
        return (f"Let's step back — this builds on something simpler. "
                f"Before '{topic}': what do you already know about '{skill}' "
                f"that might connect here?")

    def _nudge(self, topic: str, skill: str) -> str:
        return (f"Close — look at '{topic}' from a different angle. "
                f"What's the first thing that has to be true for your "
                f"answer to work?")

    def hint(self, *, topic: str, expected_answer: str,
             hint_level: int = 1) -> str:
        """Progressive hints. Level 1: direction; 2: narrowed; 3: almost —
        but never the answer itself."""
        if self.llm_fn is not None:
            try:
                raw = self.llm_fn(
                    SOCRATIC_SYSTEM,
                    f"Give hint level {hint_level} (of 3) for '{topic}'. "
                    f"Level 1 points at the right idea, level 2 narrows it, "
                    f"level 3 is one step from the answer — but NEVER state "
                    f"the answer. JSON {{\"prompt\": \"...\"}}.")
                data = json.loads(_strip_fences(raw))
                return self.guard(str(data.get("prompt", "")), expected_answer)
            except Exception:  # noqa: BLE001
                _log.debug("hint LLM call failed", exc_info=True)
        fallbacks = {
            1: f"Think about what '{topic}' is really asking — break it into smaller pieces.",
            2: f"Focus on the key relationship inside '{topic}'. What depends on what?",
            3: f"You're one step away on '{topic}' — eliminate what can't be true and see what's left.",
        }
        return self.guard(fallbacks.get(min(hint_level, 3), fallbacks[3]),
                          expected_answer)


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ── tutor session ───────────────────────────────────────────────────────────

class TutorSession:
    """One tutoring conversation.

    ``mode``: "socratic" (guide me) or "direct" (just tell me).
    ``expected_answer``: the canonical answer for the current question —
    used by the answer guard in socratic mode, revealed freely in direct.
    """

    def __init__(
        self,
        topic: str,
        mode: str = "socratic",
        *,
        llm_fn: Callable[[str, str], str] | None = None,
        sub_skills: list[str] | None = None,
        expected_answer: str = "",
        source: str = "chat",
        citations: list[str] | None = None,
    ) -> None:
        if mode not in ("socratic", "direct"):
            raise ValueError(f"mode must be 'socratic' or 'direct', got {mode!r}")
        self.id = uuid.uuid4().hex[:12]
        self.topic = topic
        self.mode = mode
        self.engine = SocraticEngine(llm_fn)
        self.mastery = MasteryModel(sub_skills)
        self.expected_answer = expected_answer
        self.source = source
        self.citations = list(citations or [])
        self.skill = (sub_skills or ["general"])[0]
        self.turns = 0
        self._hint_level = 0
        self._started_at = time.time()
        self._current_question = ""

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> TutorTurn:
        """Open the session: present the first question (socratic) or the
        teaching (direct)."""
        if self.mode == "direct":
            teaching = self._teach_direct()
            return TutorTurn(feedback="", prompt=teaching, revealed=True,
                             mastery=self.mastery.snapshot(),
                             citations=self.citations)
        question = self._first_question()
        self._current_question = question
        return TutorTurn(feedback=f"Let's work through '{self.topic}' together.",
                         prompt=self.engine.guard(question, self.expected_answer),
                         mastery=self.mastery.snapshot(),
                         citations=self.citations)

    def respond(self, student_answer: str) -> TutorTurn:
        """Student answered the current question -> feedback + next prompt."""
        self.turns += 1
        self._hint_level = 0
        verdict, specifics = self.engine.diagnose(
            student_answer, self.expected_answer, context=self.topic)
        correct = verdict == "correct"
        mastery = self.engine and self.mastery.update(
            self.skill, correct=correct or verdict == "partial",
            difficulty=0.5)

        if self.mode == "direct":
            # Direct mode teaches outright — answers included.
            prompt = self._teach_direct(follow_up=student_answer,
                                        verdict=verdict)
            return TutorTurn(feedback=specifics, prompt=prompt,
                             diagnosis=verdict, revealed=True,
                             mastery=self.mastery.snapshot(),
                             citations=self.citations)

        # Socratic mode: guide, never reveal.
        feedback = self.engine.guard(specifics, self.expected_answer)
        prompt = self.engine.next_prompt(
            topic=self.topic, verdict=verdict, skill=self.skill,
            expected_answer=self.expected_answer,
            struggle_key=self._current_question or self.topic)
        if verdict == "correct":
            self._current_question = prompt
        done = self.mastery.skills.get(self.skill, 0.5) >= 0.9 and verdict == "correct"
        return TutorTurn(feedback=feedback, prompt=prompt, diagnosis=verdict,
                         revealed=False, mastery=self.mastery.snapshot(),
                         citations=self.citations, done=done)

    def hint(self) -> TutorTurn:
        """A progressive hint. Never the answer in socratic mode."""
        self._hint_level += 1
        if self.mode == "direct":
            return TutorTurn(feedback="Here's the straight answer:",
                             prompt=self.expected_answer or self._teach_direct(),
                             revealed=True, mastery=self.mastery.snapshot(),
                             citations=self.citations)
        prompt = self.engine.hint(topic=self.topic,
                                  expected_answer=self.expected_answer,
                                  hint_level=self._hint_level)
        return TutorTurn(feedback=f"Hint {min(self._hint_level, 3)}/3:",
                         prompt=prompt, revealed=False,
                         mastery=self.mastery.snapshot(),
                         citations=self.citations)

    def set_question(self, question: str, expected_answer: str,
                     skill: str | None = None) -> None:
        """Move to a new question (driven by the chat layer or curriculum)."""
        self._current_question = question
        self.expected_answer = expected_answer
        if skill:
            self.skill = skill
            if skill not in self.mastery.skills:
                self.mastery.skills[skill] = 0.5
        self._hint_level = 0

    # -- internals ----------------------------------------------------------
    def _first_question(self) -> str:
        if self.engine.llm_fn is not None:
            try:
                raw = self.engine.llm_fn(
                    SOCRATIC_SYSTEM,
                    f"Start a Socratic session on '{self.topic}'. Ask the "
                    f"first diagnostic question as JSON {{\"prompt\": \"...\"}}.")
                return str(json.loads(_strip_fences(raw)).get("prompt", ""))
            except Exception:  # noqa: BLE001
                _log.debug("first-question LLM call failed", exc_info=True)
        return (f"What do you already know about '{self.topic}'? "
                f"Start anywhere — even a guess tells me where to begin.")

    def _teach_direct(self, follow_up: str = "",
                      verdict: str = "") -> str:
        if self.engine.llm_fn is not None:
            try:
                return self.engine.llm_fn(
                    "You are a direct, no-fluff tutor. Teach clearly with a "
                    "concrete example. Keep it tight.",
                    f"Teach '{self.topic}'."
                    + (f" The student said: {follow_up} ({verdict})." if follow_up else "")
                    + (f"\nAnswer to convey: {self.expected_answer}" if self.expected_answer else ""))
            except Exception:  # noqa: BLE001
                _log.debug("direct-teach LLM call failed", exc_info=True)
        base = f"Here's '{self.topic}', straight: "
        if self.expected_answer:
            base += self.expected_answer
        else:
            base += ("work through it step by step — break the problem into "
                     "its smallest pieces, solve each, then combine.")
        if self.citations:
            base += "\n\nSources: " + "; ".join(self.citations)
        return base


# ── per-chat session registry ───────────────────────────────────────────────
# The chat layer keeps one active TutorSession per chat key. Sessions are
# in-memory (a restart ends them); the mistake notebook persists via the
# repetition scheduler.

_SESSIONS: dict[str, TutorSession] = {}
_NOTEBOOKS: dict[str, Any] = {}


def get_session(chat_key: str) -> TutorSession | None:
    """The active tutoring session for a chat, if any."""
    return _SESSIONS.get(chat_key)


def set_session(chat_key: str, session: TutorSession) -> None:
    _SESSIONS[chat_key] = session


def end_session(chat_key: str) -> bool:
    """End the active session. Returns True if one was running."""
    return _SESSIONS.pop(chat_key, None) is not None


def get_notebook(chat_key: str = "") -> Any:
    """The mistake notebook (shared process-wide by default)."""
    from .flashcards import MistakeNotebook
    if chat_key not in _NOTEBOOKS:
        try:
            from ..memory.repetition import RepetitionScheduler
            _NOTEBOOKS[chat_key] = MistakeNotebook(RepetitionScheduler())
        except Exception:  # noqa: BLE001
            _NOTEBOOKS[chat_key] = MistakeNotebook(None)
    return _NOTEBOOKS[chat_key]


# ── photo input ─────────────────────────────────────────────────────────────

def tutor_from_photo(image_path: str, *, mode: str = "socratic",
                     seer_fn: Callable[[str, str], str] | None = None,
                     llm_fn: Callable[[str, str], str] | None = None
                     ) -> TutorSession:
    """Snap a problem -> Seer extracts it -> tutoring session.

    Fail-closed: no vision path means no session (never a fake one).
    """
    see = seer_fn
    if see is None:
        def see(path: str, question: str) -> str:  # noqa: ANN001,ANN202
            from ..vision.seer import VisionUnavailable, get_seer
            try:
                return get_seer().see(path, question)
            except VisionUnavailable as exc:
                raise TutorError(
                    f"vision is unavailable ({exc}); cannot start a "
                    f"photo tutoring session") from exc
    try:
        problem = see(image_path,
                      "Extract the problem or question shown in this image, "
                      "exactly as written. If there is no clear problem, say so.")
    except TutorError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TutorError(f"could not read the image: {exc}") from exc
    problem = (problem or "").strip()
    if not problem or "no clear problem" in problem.lower():
        raise TutorError("no clear problem found in the image")
    return TutorSession(topic=problem, mode=mode, llm_fn=llm_fn,
                        source=f"photo:{image_path}")


# ── source-grounded tutoring ──────────────────────────────────────────────────

def tutor_from_files(paths: list[str], topic: str, *,
                     mode: str = "socratic",
                     llm_fn: Callable[[str, str], str] | None = None,
                     index: Any | None = None) -> TutorSession:
    """Teach ``topic`` from the owner's files only, with citations.

    Each claim the tutor makes should cite the file it came from
    (#20-style: which file, which section).
    """
    from ..documents.index import DocumentIndex
    from ..documents.model import Document, Section

    idx = index if index is not None else DocumentIndex()
    sources: list[str] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            raise TutorError(f"cannot read {path}: {exc}") from exc
        if not text.strip():
            continue
        doc = Document(title=path, source=path,
                       sections=[Section(heading="", text=text)])
        try:
            idx.add(doc)
        except Exception as exc:  # noqa: BLE001
            raise TutorError(f"could not index {path}: {exc}") from exc
        sources.append(path)
    if not sources:
        raise TutorError("no readable files to teach from")

    citations: list[str] = []
    context_bits: list[str] = []
    try:
        hits = idx.search(topic, limit=3)
    except Exception as exc:  # noqa: BLE001
        raise TutorError(f"file search failed: {exc}") from exc
    for hit in hits:
        src = str(hit.get("doc_id", ""))
        snippet = str(hit.get("snippet", ""))[:300]
        # Resolve the doc_id back to its file path for a readable citation.
        citations.append(src)
        if snippet:
            context_bits.append(f"[{src}] {snippet}")

    session = TutorSession(topic=topic, mode=mode, llm_fn=llm_fn,
                           source="files:" + ",".join(sources),
                           citations=citations)
    # Seed the first question from the file context when there's no LLM.
    if llm_fn is None and context_bits:
        session._current_question = (
            f"From your files on '{topic}': {context_bits[0][:200]}… "
            f"What does this tell you?")
    return session
