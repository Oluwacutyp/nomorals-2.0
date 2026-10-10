"""Socratic tutoring engine — Devon as tutor (build-map #46).

Two explicit modes:
- ``socratic`` ("guide me"): withholds answers, asks diagnostic questions,
  backtracks to prerequisites on struggle, probes deeper on success.
  NEVER reveals the expected answer — enforced by :meth:`SocraticEngine.guard`
  plus :meth:`SocraticEngine.sanitize_feedback` (the fallback diagnosis used
  to leak expected keywords like "you're missing: wavelength" — fixed).
- ``direct`` ("just tell me"): teaches straight, answers included.

Dialogue policy follows AutoTutor's EMT cycle (Graesser et al.): every hard
question carries expectations + anticipated misconceptions, and the engine
runs the pump → hint → prompt → assertion ladder. Pedagogical shaping follows
LearnLM's five principles (active learning, cognitive load, adaptivity,
curiosity, metacognition).

Mastery is tracked two ways: the original linear ``update()`` (kept for
back-compat — drives ``weakest()``/``snapshot()`` and the chat layer) and a
Bayesian Knowledge Tracing posterior per skill (Corbett & Anderson 1995;
pyBKT), plus Duolingo-style half-life regression (Settles & Meeder 2016) for
forgetting. Sessions persist to JSON so a restart doesn't kill them.

The engine takes an optional ``llm_fn(prompt) -> str`` for diagnosis and
question generation. Without one it uses a structured fallback (template
questions, keyword diagnosis) — honest and documented, never a fake LLM.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "MasteryModel",
    "BKT_DEFAULTS",
    "SocraticEngine",
    "TutorError",
    "TutorSession",
    "TutorTurn",
    "Expectation",
    "Misconception",
    "QuestionScript",
    "end_session",
    "get_session",
    "set_session",
    "persist_session",
    "restore_session",
    "sessions_dir",
    "tutor_from_files",
    "tutor_from_photo",
]


class TutorError(Exception):
    """Tutoring could not start or continue (vision down, no files, ...)."""


# ── mastery model ────────────────────────────────────────────────────────────

#: BKT defaults from the pyBKT README example
#: (p_T=0.30, p_G=0.10, p_S=0.03, p_L0=0.10). p_forget is the BKT+forget
#: extension (standard BKT assumes 0); overridable per skill.
BKT_DEFAULTS: dict[str, float] = {
    "p_init": 0.10,
    "p_learn": 0.30,
    "p_guess": 0.10,
    "p_slip": 0.03,
    "p_forget": 0.0,
}


@dataclass
class MasteryModel:
    """Per-sub-skill mastery.

    Two trackers live side by side:
    - ``skills``: the original linear heuristic (kept for back-compat —
      drives ``weakest()``, ``snapshot()``, and the chat layer).
    - BKT posterior per skill (``bkt_update``): P(knows) via Corbett &
      Anderson 1995 with the pyBKT update equations, guess/slip aware.
    - HLR half-life per skill (``halflife_update``/``recall_prob``):
      p(recall) = 2^(-elapsed/h), after Settles & Meeder 2016 — BKT says
      whether the student knows it, HLR says whether they'll still know it
      tomorrow.
    """

    skills: dict[str, float] = field(default_factory=dict)

    def __init__(self, sub_skills: list[str] | None = None) -> None:
        self.skills = {s: 0.5 for s in (sub_skills or ["general"])}
        self._bkt: dict[str, float] = {}
        self._bkt_params: dict[str, dict[str, float]] = {}
        self._halflife: dict[str, float] = {}   # days
        self._hl_seen: dict[str, float] = {}    # last observation timestamp

    # -- original linear heuristic (unchanged semantics) --------------------
    def update(self, skill: str, *, correct: bool,
               difficulty: float = 0.5) -> float:
        """Record an outcome. Returns the new mastery for the skill."""
        weight = 0.5 + max(0.0, min(1.0, difficulty))
        delta = (0.15 if correct else -0.20) * weight
        new = max(0.0, min(1.0, self.skills.get(skill, 0.5) + delta))
        self.skills[skill] = round(new, 3)
        return self.skills[skill]

    # -- Bayesian Knowledge Tracing ------------------------------------------
    def set_bkt_params(self, skill: str, **params: float) -> None:
        merged = dict(BKT_DEFAULTS)
        merged.update({k: v for k, v in params.items() if k in merged})
        self._bkt_params[skill] = merged

    def _params(self, skill: str) -> dict[str, float]:
        return self._bkt_params.get(skill, BKT_DEFAULTS)

    def bkt_update(self, skill: str, *, correct: bool,
                   now: float | None = None) -> float:
        """One BKT observation. Returns P(knows) after the update."""
        prm = self._params(skill)
        p = self._bkt.get(skill, prm["p_init"])
        g, s = prm["p_guess"], prm["p_slip"]
        if correct:
            denom = p * (1 - s) + (1 - p) * g
            p_obs = p * (1 - s) / denom if denom > 0 else p
        else:
            denom = p * s + (1 - p) * (1 - g)
            p_obs = p * s / denom if denom > 0 else p
        p_next = p_obs + (1 - p_obs) * prm["p_learn"]
        p_next *= (1.0 - prm["p_forget"])
        self._bkt[skill] = round(min(1.0, max(0.0, p_next)), 4)
        self.halflife_update(skill, correct=correct, now=now)
        return self._bkt[skill]

    def bkt(self, skill: str) -> float:
        """Current P(knows) for a skill (prior if never observed)."""
        return self._bkt.get(skill, self._params(skill)["p_init"])

    def predict_correct(self, skill: str) -> float:
        """P(student gets the next item right) = P(L)(1-S) + (1-P(L))G."""
        prm = self._params(skill)
        p = self.bkt(skill)
        return p * (1 - prm["p_slip"]) + (1 - p) * prm["p_guess"]

    # -- half-life regression (forgetting) ------------------------------------
    def halflife_update(self, skill: str, *, correct: bool,
                        now: float | None = None) -> float:
        """HLR-lite: double the half-life on success, halve on failure
        (floor 6h). Full HLR fits per-item weights from data; this cold-start
        heuristic keeps the shape honest without pretending to be fitted."""
        now = time.time() if now is None else now
        h = self._halflife.get(skill, 1.0)
        h = max(0.25, h * (2.0 if correct else 0.5))
        self._halflife[skill] = round(h, 3)
        self._hl_seen[skill] = now
        return self._halflife[skill]

    def recall_prob(self, skill: str, now: float | None = None) -> float:
        """p(recall) = 2^(-elapsed_days / half_life)."""
        now = time.time() if now is None else now
        h = self._halflife.get(skill)
        seen = self._hl_seen.get(skill)
        if h is None or seen is None:
            return 1.0
        elapsed_days = max(0.0, (now - seen) / 86400.0)
        return 2.0 ** (-elapsed_days / h)

    # -- accessors -------------------------------------------------------------
    def weakest(self) -> str:
        return min(self.skills, key=lambda s: self.skills[s])

    def strongest(self) -> str:
        return max(self.skills, key=lambda s: self.skills[s])

    def weakest_bkt(self) -> str:
        pool = self._bkt or {s: self._params(s)["p_init"]
                             for s in self.skills}
        return min(pool, key=lambda s: pool[s])

    def snapshot(self) -> dict[str, float]:
        return dict(self.skills)

    def bkt_snapshot(self) -> dict[str, float]:
        return {s: self.bkt(s) for s in self.skills}

    # -- persistence -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"skills": dict(self.skills), "bkt": dict(self._bkt),
                "bkt_params": {k: dict(v)
                               for k, v in self._bkt_params.items()},
                "halflife": dict(self._halflife),
                "hl_seen": dict(self._hl_seen)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MasteryModel":
        m = cls(list((d.get("skills") or {}).keys()) or None)
        m.skills.update({str(k): float(v)
                         for k, v in (d.get("skills") or {}).items()})
        m._bkt.update({str(k): float(v)
                       for k, v in (d.get("bkt") or {}).items()})
        for k, v in (d.get("bkt_params") or {}).items():
            m._bkt_params[str(k)] = {pk: float(pv)
                                     for pk, pv in v.items()}
        m._halflife.update({str(k): float(v)
                            for k, v in (d.get("halflife") or {}).items()})
        m._hl_seen.update({str(k): float(v)
                           for k, v in (d.get("hl_seen") or {}).items()})
        return m


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
    move: str = ""            # EMT dialog move used: pump|hint|prompt|assert|...


# ── EMT: expectations & misconceptions ───────────────────────────────────────
# AutoTutor (Graesser et al.): each hard question carries anticipated good
# answers (expectations) and anticipated bugs (misconceptions); the tutor
# matches student turns against both and runs the
# feedback → pump → hint → prompt → assertion ladder.

@dataclass
class Expectation:
    """One anticipated good-answer component. ``keywords`` default to the
    content tokens of ``text``."""
    text: str
    keywords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.keywords:
            toks = [t for t in _tokens(self.text) if len(t) >= 4]
            object.__setattr__(self, "keywords", tuple(toks))


@dataclass
class Misconception:
    """One anticipated bug. ``pattern`` is matched fuzzily against the
    student's answer; ``probe`` is the socratic question that targets it."""
    pattern: str
    correction: str
    probe: str = ""
    source: str = ""          # e.g. "WAEC Chief Examiner 2025 Maths"

    def matches(self, answer: str) -> bool:
        ptoks = {t for t in _tokens(self.pattern) if len(t) >= 3}
        atoks = _tokens(answer)
        if not ptoks or not atoks:
            return False
        return len(ptoks & atoks) / len(ptoks) >= 0.6


@dataclass
class QuestionScript:
    """The EMT script for one question."""
    question: str
    expected_answer: str
    skill: str = "general"
    expectations: list[Expectation] = field(default_factory=list)
    misconceptions: list[Misconception] = field(default_factory=list)


# ── socratic engine ───────────────────────────────────────────────────────────

_TOKEN = re.compile(r"[a-z0-9]+")

#: Rewritten around LearnLM's five pedagogy principles (DeepMind 2024):
#: active learning, cognitive load, adaptivity, curiosity, metacognition.
SOCRATIC_SYSTEM = """You are a Socratic tutor grounded in learning science.
Principles: (1) ACTIVE LEARNING — never reveal the expected answer; guide the
student to discover it. Ask one question at a time. (2) COGNITIVE LOAD — keep
replies SHORT: one piece of feedback, one question. Use bullets for structure.
Logical order, no repetition, no contradiction. (3) ADAPTIVITY — diagnose each
answer as correct, partial, or wrong, and say WHAT specifically is right or
wrong. If the student struggles twice on the same idea, backtrack to a
prerequisite. If they sound frustrated, encourage and simplify. (4) CURIOSITY —
on success, probe deeper with a harder follow-up. (5) METACOGNITION — help the
student reflect: name what improved, state the plan.
Output ONLY JSON: {"feedback": "...", "prompt": "...",
"diagnosis": "correct|partial|wrong", "skill": "<sub-skill>",
"difficulty": 0.0-1.0, "prerequisite": "<prereq question or empty>"}
"""

#: LearnLM adaptivity: frustration signals that trigger encouragement +
#: an easier sub-question instead of the next hard step.
_FRUSTRATION = ("i don't get", "dont get", "don't understand",
                "this is hard", "too hard", "confusing", "i give up",
                "give up", "stupid", "hate this", "ugh", "idk", "i dunno")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been being "
    "it its this that these those as at by from into over after such what "
    "which when where who how why will would can could should than then there "
    "their them they you your we our he she his her".split())


class SocraticEngine:
    """The dialogue loop. ``llm_fn`` is ``(system, user) -> str``; without it
    the engine runs a structured fallback (templates + keyword diagnosis)."""

    def __init__(self, llm_fn: Callable[[str, str], str] | None = None) -> None:
        self.llm_fn = llm_fn
        self._struggles: dict[str, int] = {}
        self._script: QuestionScript | None = None
        self._covered: set[int] = set()
        self._move_counts: dict[str, int] = {}

    # -- EMT script ----------------------------------------------------------
    def set_script(self, script: QuestionScript) -> None:
        """Arm the expectation/misconception script for the current question."""
        self._script = script
        self._covered = set()
        self._move_counts = {}

    def match_expectations(self, answer: str) -> tuple[list[int], list[int]]:
        """-> (covered expectation indexes, missing indexes)."""
        if not self._script or not self._script.expectations:
            return [], []
        atoks = _tokens(answer)
        covered, missing = [], []
        for i, exp in enumerate(self._script.expectations):
            keys = set(exp.keywords) or _tokens(exp.text)
            hit = len(keys & atoks) / max(1, len(keys))
            (covered if hit >= 0.5 else missing).append(i)
        self._covered.update(covered)
        return covered, missing

    def match_misconceptions(self, answer: str) -> list[Misconception]:
        if not self._script:
            return []
        return [m for m in self._script.misconceptions if m.matches(answer)]

    # -- answer guard: the hard guarantee ----------------------------------
    @staticmethod
    def guard(text: str, expected_answer: str) -> str:
        """Redact verbatim leaks of the expected answer.

        Socratic mode must never reveal the answer — not even when the LLM
        slips. Short answers (< 3 chars) are skipped: redacting "x" would
        mangle every word containing x. Normalized variants (case/punct/
        quote differences) are caught too.
        """
        answer = (expected_answer or "").strip()
        if len(answer) < 3 or not text:
            return text
        redacted = text
        variants = {answer, answer.strip("\"'“”‘’")}
        # Also catch the answer with punctuation stripped (e.g. "42 m/s").
        bare = re.sub(r"[^\w\s]", "", answer).strip()
        if len(bare) >= 3:
            variants.add(bare)
        for variant in variants:
            if len(variant) >= 3:
                redacted = re.sub(re.escape(variant), "[the answer]",
                                  redacted, flags=re.IGNORECASE)
        return redacted

    @staticmethod
    def sanitize_feedback(feedback: str, expected_answer: str) -> str:
        """Mask expected-answer content tokens leaking through feedback.

        The fallback diagnosis used to say "you're missing: wavelength,
        frequency" — handing the student the answer's keywords. In socratic
        mode those tokens are masked; if masking guts the feedback, a
        generic partial-credit line replaces it.
        """
        if not feedback or not expected_answer:
            return feedback
        out = feedback
        for tok in _tokens(expected_answer):
            if len(tok) >= 4 and tok not in _STOPWORDS:
                out = re.sub(r"\b" + re.escape(tok) + r"s?\b", "…", out,
                             flags=re.IGNORECASE)
        # If we masked away most of the meaning, say so generically.
        words = out.split()
        if words and sum(1 for w in words if "…" in w) / len(words) > 0.4:
            return ("Partly there — some key pieces are still missing. "
                    "Think about what the question is really asking.")
        return out

    @staticmethod
    def detect_frustration(text: str) -> bool:
        lowered = (text or "").lower()
        return any(sig in lowered for sig in _FRUSTRATION)

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
            # Socratic-safe: never name the missing keywords (that leaks the
            # answer — see sanitize_feedback).
            return ("partial",
                    "Partly there — some key pieces are still missing. "
                    "What else does the question need?")
        return "wrong", "Not quite — think about what the question is really asking."

    # -- EMT move selection ----------------------------------------------------
    def emt_turn(self, answer: str) -> dict[str, Any]:
        """One EMT cycle step. Returns feedback, the dialog move
        (pump|hint|prompt|assert|done), the next prompt, covered/missing
        expectation indexes, and matched misconceptions."""
        script = self._script
        if script is None or (not script.expectations
                              and not script.misconceptions):
            # No EMT content armed: the classic nudge/backtrack/probe path
            # handles this question (back-compat).
            return {"feedback": "", "move": "nudge",
                    "prompt": "", "covered": [], "missing": [],
                    "misconceptions": []}
        misconceptions = self.match_misconceptions(answer)
        covered, missing = self.match_expectations(answer)

        if misconceptions:
            m = misconceptions[0]
            hits = self._move_counts.get("misconception:" + m.pattern, 0) + 1
            self._move_counts["misconception:" + m.pattern] = hits
            probe = m.probe or (
                f"Careful — that's a common trap. {m.correction} "
                f"Now, rethink: {script.question}")
            src = f" (WAEC examiners flag this: {m.source})" if m.source else ""
            return {"feedback": f"Not quite — {m.correction}{src}",
                    "move": "assert" if hits >= 2 else "hint",
                    "prompt": self.guard(probe, script.expected_answer),
                    "covered": covered, "missing": missing,
                    "misconceptions": [m.pattern for m in misconceptions]}

        if not missing:
            return {"feedback": "That's the full picture — well done.",
                    "move": "done", "prompt": "",
                    "covered": covered, "missing": [],
                    "misconceptions": []}

        # Pump → hint → prompt → assertion ladder per missing expectation.
        key = f"exp:{missing[0]}"
        n = self._move_counts.get(key, 0) + 1
        self._move_counts[key] = n
        exp = script.expectations[missing[0]]
        if n == 1:
            move, prompt = "pump", "What else? There's more to it."
        elif n == 2:
            move = "hint"
            prompt = (f"Think about the part involving "
                      f"'{exp.keywords[0] if exp.keywords else 'the key idea'}' — "
                      f"what do you know about that?")
        elif n == 3:
            move = "prompt"
            prompt = self._fill_in_blank(exp)
        else:
            move = "assert"
            prompt = (f"Here's the missing piece so we can move on: "
                      f"{exp.text} Now put it together — {script.question}")
        return {"feedback": "Good start — keep going." if covered else "Not yet.",
                "move": move,
                "prompt": self.guard(prompt, script.expected_answer),
                "covered": covered, "missing": missing,
                "misconceptions": []}

    @staticmethod
    def _fill_in_blank(exp: Expectation) -> str:
        """AutoTutor 'prompt' move: fill-in-the-blank for a missing keyword."""
        text = exp.text
        for kw in exp.keywords:
            if len(kw) >= 4:
                blanked = re.sub(r"\b" + re.escape(kw) + r"\b", "_____",
                                 text, count=1, flags=re.IGNORECASE)
                if blanked != text:
                    return f"Complete this: {blanked}"
        return f"Say more about: {text}"

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

def sessions_dir() -> Path:
    return Path.home() / ".nomorals" / "learn" / "sessions"


class TutorSession:
    """One tutoring conversation.

    ``mode``: "socratic" (guide me) or "direct" (just tell me).
    ``expected_answer``: the canonical answer for the current question —
    used by the answer guard in socratic mode, revealed freely in direct.
    ``notebook``: an optional :class:`MistakeNotebook` — wrong answers are
    recorded automatically (build-map #46's promise).
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
        syllabus: Any | None = None,
        notebook: Any | None = None,
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
        # Optional WAEC/JAMB syllabus grounding (#48): a SyllabusTopic or a
        # topic string to resolve. When set, the tutor references the
        # syllabus code and follows syllabus topic order.
        self.syllabus = syllabus
        self.skill = (sub_skills or ["general"])[0]
        self.notebook = notebook
        self.turns = 0
        self._hint_level = 0
        self._started_at = time.time()
        self._current_question = ""
        self._verdicts: list[str] = []
        self._mastery_start: dict[str, float] = {}

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> TutorTurn:
        """Open the session: present the first question (socratic) or the
        teaching (direct)."""
        # Resolve a syllabus string to a real topic (#48, additive).
        if isinstance(self.syllabus, str):
            try:
                from .curriculum import find_topic
                found = find_topic(self.syllabus)
                self.syllabus = found if found is not None else self.syllabus
            except Exception:  # noqa: BLE001
                pass
        code_line = ""
        if self.syllabus is not None and not isinstance(self.syllabus, str):
            try:
                from .curriculum import syllabus_code
                code_line = f"\n📖 {syllabus_code(self.syllabus)}"
            except Exception:  # noqa: BLE001
                pass
        self._mastery_start = self.mastery.snapshot()
        if self.mode == "direct":
            teaching = self._teach_direct()
            return TutorTurn(feedback="", prompt=teaching + code_line,
                             revealed=True,
                             mastery=self.mastery.snapshot(),
                             citations=self.citations)
        question = self._first_question()
        self._current_question = question
        return TutorTurn(
            feedback=f"Let's work through '{self.topic}' together.{code_line}\n"
                     f"{self.plan()}",
            prompt=self.engine.guard(question, self.expected_answer),
            mastery=self.mastery.snapshot(),
            citations=self.citations)

    def plan(self) -> str:
        """LearnLM metacognition: state the session objective up front."""
        return (f"📋 Plan: I'll ask questions about '{self.topic}' and guide "
                f"you to the answers yourself — no answers handed over. "
                f"Say 'hint' any time; 'report' shows your progress.")

    def respond(self, student_answer: str) -> TutorTurn:
        """Student answered the current question -> feedback + next prompt."""
        self.turns += 1
        self._hint_level = 0
        answer = student_answer or ""

        # LearnLM adaptivity: frustration → encourage + simplify, don't push.
        frustrated = self.engine.detect_frustration(answer)
        if frustrated:
            key = self._current_question or self.topic
            self.engine._struggles[key] = max(
                0, self.engine._struggles.get(key, 1) - 1)

        verdict, specifics = self.engine.diagnose(
            answer, self.expected_answer, context=self.topic)
        self._verdicts.append(verdict)
        correct = verdict == "correct"
        self.mastery.update(self.skill, correct=correct or verdict == "partial",
                            difficulty=0.5)
        self.mastery.bkt_update(self.skill, correct=correct)

        # EMT cycle when a script with expectations/misconceptions is armed.
        move = ""
        emt_prompt = ""
        script = self.engine._script
        if (self.mode == "socratic" and script is not None
                and (script.expectations or script.misconceptions)):
            emt = self.engine.emt_turn(answer)
            move = emt["move"]
            if move == "done":
                emt_prompt = self.engine.next_prompt(
                    topic=self.topic, verdict="correct", skill=self.skill,
                    expected_answer=self.expected_answer,
                    struggle_key=self._current_question or self.topic)
                self._current_question = emt_prompt
            else:
                emt_prompt = emt["prompt"] or self.engine.next_prompt(
                    topic=self.topic, verdict=verdict, skill=self.skill,
                    expected_answer=self.expected_answer,
                    struggle_key=self._current_question or self.topic)

        if self.mode == "direct":
            # Direct mode teaches outright — answers included.
            prompt = self._teach_direct(follow_up=answer, verdict=verdict)
            return TutorTurn(feedback=specifics, prompt=prompt,
                             diagnosis=verdict, revealed=True,
                             mastery=self.mastery.snapshot(),
                             citations=self.citations, move=move)

        # Socratic mode: guide, never reveal. Sanitize the feedback so
        # expected-answer keywords can't leak through it.
        feedback = self.engine.sanitize_feedback(
            self.engine.guard(specifics, self.expected_answer),
            self.expected_answer)
        if frustrated:
            feedback = ("I hear you — this one's tough, and that's normal. "
                        "Let's make it smaller. ") + feedback
        prompt = emt_prompt or self.engine.next_prompt(
            topic=self.topic, verdict=verdict, skill=self.skill,
            expected_answer=self.expected_answer,
            struggle_key=self._current_question or self.topic)
        if verdict == "correct":
            self._current_question = prompt

        # Record the mistake (build-map #46's promise). Never breaks tutoring.
        if verdict == "wrong" and self.notebook is not None:
            try:
                self.notebook.record(
                    self._current_question or self.topic, answer,
                    self.expected_answer,
                    why=feedback, topic=self.topic,
                    tags=["tutor", self.skill])
            except Exception:  # noqa: BLE001
                _log.debug("mistake recording failed", exc_info=True)

        done = (self.mastery.skills.get(self.skill, 0.5) >= 0.9
                and verdict == "correct")
        return TutorTurn(feedback=feedback, prompt=prompt, diagnosis=verdict,
                         revealed=False, mastery=self.mastery.snapshot(),
                         citations=self.citations, done=done, move=move)

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
                     skill: str | None = None,
                     script: QuestionScript | None = None) -> None:
        """Move to a new question (driven by the chat layer or curriculum).

        Pass a ``QuestionScript`` to arm the EMT expectation/misconception
        cycle for this question.
        """
        self._current_question = question
        self.expected_answer = expected_answer
        if skill:
            self.skill = skill
            if skill not in self.mastery.skills:
                self.mastery.skills[skill] = 0.5
        if script is not None:
            self.engine.set_script(script)
        else:
            self.engine.set_script(QuestionScript(
                question=question, expected_answer=expected_answer,
                skill=self.skill))
        self._hint_level = 0

    def next_item(self) -> dict[str, Any]:
        """What to do next: interleave due FSRS reviews (retrieval practice)
        with the current question. The chat layer calls this between turns."""
        if self.notebook is not None:
            try:
                due = self.notebook.due_cards(limit=1)
                if due:
                    card = due[0]
                    return {"kind": "review", "card_id": card.id,
                            "front": card.front, "back": card.back,
                            "topic": card.topic}
            except Exception:  # noqa: BLE001
                _log.debug("due-card lookup failed", exc_info=True)
        return {"kind": "question",
                "question": self._current_question or self.topic}

    def report(self) -> str:
        """LearnLM metacognition: a progress reflection for the student."""
        total = len(self._verdicts)
        correct = sum(1 for v in self._verdicts if v == "correct")
        partial = sum(1 for v in self._verdicts if v == "partial")
        lines = [f"📊 Session report — '{self.topic}' ({self.turns} turns)"]
        if total:
            lines.append(
                f"• answers: {correct} correct, {partial} partial, "
                f"{total - correct - partial} wrong")
        for skill, start in self._mastery_start.items():
            now_v = self.mastery.skills.get(skill, start)
            arrow = "↑" if now_v > start else ("↓" if now_v < start else "→")
            bkt_v = self.mastery.bkt(skill)
            recall = self.mastery.recall_prob(skill)
            lines.append(
                f"• {skill}: {start:.2f} → {now_v:.2f} {arrow} "
                f"(BKT P(knows)={bkt_v:.2f}, recall≈{recall:.0%})")
        try:
            weak = self.mastery.weakest()
            lines.append(f"• weakest skill: {weak} — that's where we focus next")
        except ValueError:
            pass
        if self.notebook is not None:
            try:
                stats = self.notebook.stats()
                lines.append(
                    f"• mistake notebook: {stats['total']} cards, "
                    f"{stats['due']} due for review")
            except Exception:  # noqa: BLE001
                pass
        elapsed = int(time.time() - self._started_at)
        lines.append(f"• time: {elapsed // 60}m {elapsed % 60}s")
        return "\n".join(lines)

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path | None = None) -> str:
        """Persist the session (restart-safe). The notebook is NOT saved
        here — it has its own persistence."""
        p = Path(path) if path else sessions_dir() / f"{self.id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "id": self.id, "topic": self.topic, "mode": self.mode,
            "expected_answer": self.expected_answer, "source": self.source,
            "citations": self.citations, "skill": self.skill,
            "mastery": self.mastery.to_dict(),
            "turns": self.turns, "hint_level": self._hint_level,
            "started_at": self._started_at,
            "current_question": self._current_question,
            "verdicts": self._verdicts,
            "mastery_start": self._mastery_start,
        }, indent=2), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path: str | Path,
             llm_fn: Callable[[str, str], str] | None = None,
             notebook: Any | None = None) -> "TutorSession":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        sub_skills = list((d.get("mastery") or {}).get("skills") or ["general"])
        s = cls(d.get("topic", ""), d.get("mode", "socratic"), llm_fn=llm_fn,
                sub_skills=sub_skills, notebook=notebook)
        s.id = d.get("id", s.id)
        s.expected_answer = d.get("expected_answer", "")
        s.source = d.get("source", "chat")
        s.citations = list(d.get("citations") or [])
        s.skill = d.get("skill", s.skill)
        s.mastery = MasteryModel.from_dict(d.get("mastery") or {})
        s.turns = int(d.get("turns", 0))
        s._hint_level = int(d.get("hint_level", 0))
        s._started_at = float(d.get("started_at", time.time()))
        s._current_question = d.get("current_question", "")
        s._verdicts = list(d.get("verdicts") or [])
        s._mastery_start = {str(k): float(v)
                            for k, v in (d.get("mastery_start") or {}).items()}
        return s

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
# persisted on demand (persist_session) so a restart doesn't lose them;
# the mistake notebook persists via its own save/load.

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


def persist_session(chat_key: str) -> str | None:
    """Save the active session for ``chat_key`` to disk. Path or None."""
    session = _SESSIONS.get(chat_key)
    if session is None:
        return None
    try:
        return session.save(sessions_dir() / f"chat_{chat_key}.json")
    except Exception:  # noqa: BLE001
        _log.debug("persist_session failed", exc_info=True)
        return None


def restore_session(chat_key: str, *,
                    llm_fn: Callable[[str, str], str] | None = None,
                    notebook: Any | None = None) -> TutorSession | None:
    """Restore a persisted session into the registry. None if none saved."""
    path = sessions_dir() / f"chat_{chat_key}.json"
    if not path.is_file():
        return None
    try:
        session = TutorSession.load(path, llm_fn=llm_fn, notebook=notebook)
    except Exception:  # noqa: BLE001
        _log.debug("restore_session failed", exc_info=True)
        return None
    _SESSIONS[chat_key] = session
    return session


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
