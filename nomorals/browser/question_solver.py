"""Local question-captcha solver: the brain answers, in-session.

Some captchas are neither pictures nor tokens — they are a plain
text question rendered into the page: "What is 2 + 3?", "What is
the sum of 4 and 5?" with an ordinary text input beside it. The
estate had NO kind for them (Pass #79 survey: no detection, no
stage, nothing) — they fell through every pipeline as "unknown".

This module adds the kind ``question_captcha`` with a deliberately
conservative detector (an arithmetic/question phrase AND a text
input AND a captcha/security context on the page — a blog article
that asks "what is 2+3?" in prose is not a captcha) and a solver
that asks the brain. The brain call is bounded: one short prompt,
answer-only instruction, the reply trimmed to its first line. No
brain bound or resolvable → the solver registers as unavailable
and routing falls through, honestly.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..tools.captcha import (
    CaptchaBackend,
    SolveResult,
    register_backend,
)
from .kind_solvers import register_kind_solver
from .tab_tools import press_key, type_answer

_log = get_logger(__name__)

__all__ = [
    "QUESTION_KIND",
    "detect_question_challenge",
    "brain_status",
    "QuestionSolver",
    "solve_question_kind",
    "LocalQuestionBackend",
    "register_local_backend",
]

#: The estate's kind name for text question captchas.
QUESTION_KIND = "question_captcha"

_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_INPUT_RE = re.compile(r"<input\b([^>]*)>", re.I | re.S)
_ATTR_RE = re.compile(r"([\w-]+)\s*=\s*[\"']([^\"']*)[\"']")

_QUESTION_RES = (
    re.compile(r"what is\s+[\d\s\+\-\*/x×÷().]+(?:\?|=)", re.I),
    re.compile(r"what is the (?:sum|total|product|difference) of\s+"
               r"[\d\sand,]+?\?", re.I),
    re.compile(r"(?:solve|answer|calculate)[:\s]+[\d\s\+\-\*/x×÷().]+"
               r"(?:\?|=)", re.I),
    re.compile(r"\b\d+\s*(?:plus|minus|times|\+|\-|\*|x|×|÷|/)\s*\d+"
               r"\s*(?:=|\?)", re.I),
)
_CONTEXT_HINTS = ("captcha", "security question", "human verification",
                  "prove you are human", "anti-spam", "antispam",
                  "verification question")
_INPUT_HINTS = ("captcha", "answer", "verify", "security", "human",
                "challenge", "question")


def _selector_for(attrs: dict[str, str]) -> str:
    if attrs.get("id"):
        return f"#{attrs['id']}"
    if attrs.get("name"):
        return f"input[name='{attrs['name']}']"
    classes = (attrs.get("class") or "").split()
    if classes:
        return f"input.{classes[0]}"
    return "input"


def detect_question_challenge(html: str) -> dict[str, Any] | None:
    """Recognise a text question captcha in one document.

    Returns ``{"question", "input_selector"}`` or None. All three
    signals are required — question phrase, a text input, and a
    captcha/security context — so prose questions and ordinary
    forms are never mistaken for one.
    """
    if not html:
        return None
    text = re.sub(r"\s+", " ", _TAG_STRIP_RE.sub(" ", html))
    low_text = text.lower()
    low_html = html.lower()
    if not any(h in low_text or h in low_html for h in _CONTEXT_HINTS) \
            and "captcha" not in low_html:
        return None
    question = ""
    for pattern in _QUESTION_RES:
        m = pattern.search(text)
        if m:
            question = m.group(0).strip()
            break
    if not question:
        return None
    fallback = ""
    for m in _INPUT_RE.finditer(html):
        attrs = {a.group(1).lower(): a.group(2)
                 for a in _ATTR_RE.finditer(m.group(1))}
        itype = (attrs.get("type") or "text").lower()
        if itype not in ("text", "", "number", "tel"):
            continue
        identity = " ".join(str(attrs.get(k) or "") for k in
                            ("name", "id", "placeholder", "aria-label",
                             "class")).lower()
        if any(h in identity for h in _INPUT_HINTS):
            return {"question": question,
                    "input_selector": _selector_for(attrs)}
        if not fallback:
            fallback = _selector_for(attrs)
    if fallback:
        return {"question": question, "input_selector": fallback}
    return None


# ── the brain ────────────────────────────────────────────────────────────

_ANSWER_PROMPT = (
    "Answer this captcha question with ONLY the answer — no words, "
    "no punctuation, no explanation. For arithmetic, reply with "
    "just the number.\nQuestion: {question}")


def brain_status() -> dict[str, Any]:
    """Can a brain be reached to answer questions right now?

    Structural probe (same posture as the click solver's grounding
    probe): resolution is cheap; a provider failure at answer time
    is reported by the solve itself, honestly.
    """
    try:
        from ..llm.brain import brain_for
        brain_for(None)
        return {"available": True,
                "detail": "brain resolvable — answers confirmed at "
                          "solve time"}
    except Exception as exc:  # noqa: BLE001 - no brain, no answers
        return {"available": False,
                "detail": f"brain unresolvable: {exc}"}


def _brain_answer(question: str) -> str:
    """Default answerer: the shared brain, one bounded prompt."""
    from ..llm.brain import brain_for
    resp = brain_for(None).complete(
        _ANSWER_PROMPT.format(question=question), timeout_s=30.0)
    if getattr(resp, "error", ""):
        raise RuntimeError(f"brain: {resp.error}")
    text = str(getattr(resp, "text", "") or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    answer = lines[0] if lines else ""
    return answer[:64].strip().strip("\"'")


class QuestionSolver:
    """Answer question captchas on a live tab, in its own session."""

    def __init__(self, *,
                 answer_fn: Callable[[str], str] | None = None,
                 probe: Callable[[], tuple[bool, str]] | None = None,
                 sleeper: Callable[[float], None] | None = None) -> None:
        self._answer_fn = answer_fn
        self._probe = probe
        self._sleep = sleeper or time.sleep

    def available(self) -> tuple[bool, str]:
        if self._probe is not None:
            return self._probe()
        if self._answer_fn is not None:
            return True, "brain answerer bound"
        st = brain_status()
        return bool(st["available"]), str(st["detail"])

    def answer(self, question: str) -> str:
        fn = self._answer_fn or _brain_answer
        return str(fn(question) or "").strip()

    def solve_tab(self, tab: Any, challenge: dict | None = None
                  ) -> dict[str, Any]:
        ok_avail, reason = self.available()
        if not ok_avail:
            return {"ok": False, "backend": "local_brain",
                    "detail": reason}
        info = None
        if challenge and challenge.get("question"):
            info = {"question": str(challenge["question"]),
                    "input_selector":
                        str(challenge.get("input_selector") or "")}
        if info is None:
            try:
                info = detect_question_challenge(str(tab.html() or ""))
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "backend": "local_brain",
                        "detail": f"could not read the page: {exc}"}
        if info is None:
            return {"ok": False, "backend": "local_brain",
                    "detail": "no question captcha detected on "
                              "this page"}
        if not info["input_selector"]:
            return {"ok": False, "backend": "local_brain",
                    "detail": "question found but no answer input "
                              "could be located"}
        try:
            answer = self.answer(info["question"])
        except Exception as exc:  # noqa: BLE001 - brain failure is data
            return {"ok": False, "backend": "local_brain",
                    "detail": f"the brain could not answer: {exc}"}
        if not answer:
            return {"ok": False, "backend": "local_brain",
                    "detail": "the brain returned no answer"}
        if not type_answer(tab, info["input_selector"], answer):
            return {"ok": False, "backend": "local_brain",
                    "detail": f"answered {answer!r} but could not "
                              f"type it into "
                              f"{info['input_selector']!r}"}
        press_key(tab, "Enter")
        self._sleep(0.3)
        try:
            after = str(tab.html() or "")
        except Exception:  # noqa: BLE001
            after = ""
        if detect_question_challenge(after) is None:
            return {"ok": True, "backend": "local_brain",
                    "detail": f"question captcha cleared in-session: "
                              f"{info['question']!r} answered"}
        return {"ok": False, "backend": "local_brain",
                "detail": "answer typed but the question captcha is "
                          "still present — the answer was likely "
                          "wrong"}


def solve_question_kind(tab: Any, challenge: dict) -> dict[str, Any]:
    """Kind-router entry point for ``question_captcha``."""
    if tab is None:
        return {"ok": False, "backend": "local_brain",
                "detail": "question solving needs a live tab"}
    return QuestionSolver().solve_tab(tab, challenge)


# ── captcha backend registration (tools/captcha.py untouched) ────────────

class LocalQuestionBackend(CaptchaBackend):
    """Brain answers as a captcha backend (``local_brain``).

    Needs the live tab in ``metadata["session"]`` — the question
    lives in the page.
    """

    name = "local_brain"

    def __init__(self, settings: Any = None, *,
                 solver: QuestionSolver | None = None,
                 **_: Any) -> None:
        self._settings = settings
        self._solver = solver

    def available(self) -> bool:
        solver = self._solver or QuestionSolver()
        return bool(solver.available()[0])

    def solve(self, challenge: Any) -> SolveResult:
        started = time.time()
        kind = str(getattr(challenge, "kind", "") or "")
        if kind != QUESTION_KIND:
            return SolveResult(
                ok=False, kind=kind, backend=self.name,
                detail=f"local brain backend solves question "
                       f"captchas, not {kind!r}")
        metadata = getattr(challenge, "metadata", None) or {}
        session = metadata.get("session") or metadata.get("tab")
        if session is None:
            return SolveResult(
                ok=False, kind=kind, backend=self.name,
                detail="question solving needs the live browser "
                       "session in challenge.metadata['session']")
        solver = self._solver or QuestionSolver()
        out = solver.solve_tab(session)
        return SolveResult(
            ok=bool(out.get("ok")), kind=kind, backend=self.name,
            elapsed_ms=int((time.time() - started) * 1000),
            detail=str(out.get("detail") or ""))


_REGISTERED = False


def register_local_backend() -> bool:
    """Register ``local_brain`` in the captcha backend registry."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        register_backend("local_brain", LocalQuestionBackend)
        _REGISTERED = True
        return True
    except Exception as exc:  # noqa: BLE001 - registration is additive
        _log.warning("local brain backend registration failed: %s",
                     exc)
        return False


register_local_backend()
register_kind_solver(QUESTION_KIND, solve_question_kind)
