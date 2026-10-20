"""Chat control for /learn — one front door over nomorals.learn.

Never raises. Owner-only (same gate as /tutor and /course).

Usage:
    /learn tutor [guide me] <topic> | answer <text> | hint | status | stop
    /learn quiz formats
    /learn quiz start <exam> <subject>
    /learn flashcards due [n] | stats | list | review <id> <grade 1-4> | export <path>
    /learn mastery [weakest]
    /learn waec <subject> | waec grade <pct>
    /learn jamb <subject> | jamb formats

Every verb delegates to nomorals.learn functions — nothing is
re-implemented here. ``quiz start`` is honestly unavailable: ExamSim
needs caller-supplied questions and learn/ ships no question bank,
so this command refuses to fabricate one.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger
from ..core.tool_render import describe_exc

_log = get_logger(__name__)

__all__ = ["control_learn", "LEARN_USAGE"]


def LEARN_USAGE() -> str:
    return (
        "📚 /learn — tutoring, flashcards, mastery, WAEC/JAMB\n"
        "  /learn tutor [guide me] <topic> — start a session\n"
        "  /learn tutor answer <text> | hint | status | stop\n"
        "  /learn quiz formats — exam formats learn/ knows\n"
        "  /learn quiz start <exam> <subject> — mock exam (needs a question bank)\n"
        "  /learn flashcards due [n] | stats | list | review <id> <1-4> | export <path>\n"
        "  /learn mastery [weakest] — BKT mastery snapshot\n"
        "  /learn waec <subject> | waec grade <pct>\n"
        "  /learn jamb <subject> | jamb formats"
    )


def control_learn(tail: str, context: Any = None, chat: Any = None,
                  sender_id: str = "", sender: str = "",
                  chat_key: str = "") -> str:
    """``/learn`` — one front door over nomorals.learn. Never raises."""
    try:
        return _control_learn(tail, chat_key=chat_key)
    except Exception as exc:  # noqa: BLE001 — never raise from chat
        return f"Learn hit an error ({exc})."


def _control_learn(tail: str, *, chat_key: str = "") -> str:
    raw = (tail or "").strip()
    if not raw or raw.lower() == "help":
        return LEARN_USAGE()
    verb, _, rest = raw.partition(" ")
    verb = verb.lower()
    rest = rest.strip()
    if verb == "tutor":
        return _tutor(rest, chat_key=chat_key)
    if verb == "quiz":
        return _quiz(rest)
    if verb == "flashcards":
        return _flashcards(rest, chat_key=chat_key)
    if verb == "mastery":
        return _mastery(rest, chat_key=chat_key)
    if verb == "waec":
        return _waec(rest)
    if verb == "jamb":
        return _jamb(rest)
    return LEARN_USAGE()


# ── tutor ────────────────────────────────────────────────────────────────


def _tutor(rest: str, *, chat_key: str = "") -> str:
    from .tutor import (
        TutorSession,
        end_session,
        get_notebook,
        get_session,
        set_session,
    )

    if not rest:
        return ("usage: /learn tutor [guide me] <topic> — e.g. "
                "/learn tutor guide me fractions\n"
                "       /learn tutor answer <text> | hint | status | stop")
    verb, _, tail = rest.partition(" ")
    verb = verb.lower()
    tail = tail.strip()

    if verb == "stop":
        return ("session ended." if end_session(chat_key)
                else "no tutoring session running.")
    if verb == "status":
        sess = get_session(chat_key)
        if sess is None:
            return "no tutoring session running."
        snap = sess.mastery.snapshot()
        nb = get_notebook(chat_key)
        lines = [f"📚 tutoring '{sess.topic}' ({sess.mode} mode, "
                 f"{sess.turns} turns)"]
        for skill, val in sorted(snap.items(), key=lambda kv: kv[1]):
            bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
            lines.append(f"  {skill}: {bar} {val:.0%}")
        lines.append(f"  mistakes notebooked: {nb.count()}")
        return "\n".join(lines)
    if verb == "hint":
        sess = get_session(chat_key)
        if sess is None:
            return "no tutoring session running — /learn tutor <topic> to start."
        try:
            turn = sess.hint()
        except Exception as exc:  # noqa: BLE001
            return f"hint failed: {describe_exc(exc)}"
        return f"{turn.feedback}\n{turn.prompt}"
    if verb == "answer":
        sess = get_session(chat_key)
        if sess is None:
            return "no tutoring session running — /learn tutor <topic> to start."
        if not tail:
            return "usage: /learn tutor answer <your answer>"
        try:
            turn = sess.respond(tail)
        except Exception as exc:  # noqa: BLE001
            return f"couldn't process that: {describe_exc(exc)}"
        out = f"{turn.feedback}\n\n{turn.prompt}" if turn.feedback else turn.prompt
        # Wrong answers become flashcards automatically (same rule as /tutor).
        if turn.diagnosis == "wrong" and sess.mode == "socratic":
            try:
                nb = get_notebook(chat_key)
                question = getattr(sess, "current_question", None) or sess.topic
                expected = getattr(sess, "expected_answer", "") or ""
                nb.record(question, tail, expected, topic=sess.topic)
                out += "\n\n📝 noted in your mistake notebook for review."
            except Exception:  # noqa: BLE001
                pass
        if turn.done:
            out += "\n\n🎓 you've got this one — nice work."
        return out

    # Start a new session: "guide me <topic>" -> socratic, else direct.
    if verb == "guide" and tail.lower().startswith("me "):
        mode, topic = "socratic", tail[3:].strip()
    elif verb == "guide":
        return "did you mean: /learn tutor guide me <topic>?"
    else:
        mode, topic = "direct", rest
    if not topic:
        return "usage: /learn tutor [guide me] <topic>"
    try:
        sess = TutorSession(topic=topic, mode=mode)
        set_session(chat_key, sess)
        turn = sess.start()
    except Exception as exc:  # noqa: BLE001
        return f"couldn't start tutoring: {describe_exc(exc)}"
    opener = ("🔍 socratic mode — I'll guide you, not tell you. "
              "Answer with /learn tutor answer <text>."
              if mode == "socratic" else
              "📖 direct mode — straight teaching. "
              "Check yourself with /learn tutor answer <text>.")
    return f"{opener}\n\n{turn.prompt}"


# ── quiz ─────────────────────────────────────────────────────────────────


def _quiz(rest: str) -> str:
    from .curriculum import EXAM_FORMATS

    low = rest.lower()
    if low == "formats" or not rest:
        lines = ["📝 Exam formats learn/ knows:"]
        for name, fmt in sorted(EXAM_FORMATS.items()):
            papers = fmt.get("papers", [])
            lines.append(f"  • {name} — {len(papers)} paper(s), "
                         f"negative marking: {fmt.get('negative_marking', False)}")
        lines.append("Start one: /learn quiz start <exam> <subject>")
        return "\n".join(lines)
    if low.startswith("start"):
        return ("Quiz start is not available: a mock exam needs a question "
                "bank, and learn/ ships no question bank — ExamSim grades "
                "caller-supplied questions only. This command will not "
                "fabricate questions. Use /learn quiz formats to see the "
                "formats, or /learn waec <subject> for the syllabus.")
    return ("usage: /learn quiz formats | /learn quiz start <exam> <subject>")


# ── flashcards ───────────────────────────────────────────────────────────


def _flashcards(rest: str, *, chat_key: str = "") -> str:
    from .flashcards import MistakeNotebook

    verb, _, tail = rest.partition(" ")
    verb = verb.lower()
    tail = tail.strip()
    nb = MistakeNotebook.load()
    if verb in ("", "help"):
        return ("usage: /learn flashcards due [n] | stats | list | "
                "review <id> <grade 1-4> | export <path>")
    if verb == "due":
        try:
            n = int(tail.split()[0]) if tail else 20
        except ValueError:
            n = 20
        due = nb.due_cards(limit=n)
        if not due:
            return "✅ nothing due — your mistake notebook is clear."
        lines = [f"📝 {len(due)} card(s) due:"]
        for c in due:
            lines.append(f"  • {c.id}: {c.front[:120]}")
        lines.append("Grade one: /learn flashcards review <id> <grade 1-4>")
        return "\n".join(lines)
    if verb == "stats":
        st = nb.stats()
        return ("📊 flashcards — total %(total)s · due %(due)s · "
                "new %(new)s · young %(young)s · mature %(mature)s"
                % {k: st.get(k, 0) for k in
                   ("total", "due", "new", "young", "mature")})
    if verb == "list":
        entries = nb.notebook()
        if not entries:
            return "📝 mistake notebook is empty."
        lines = [f"📝 mistake notebook ({len(entries)}):"]
        for e in entries[:20]:
            lines.append(f"  • {e.get('id', '?')}: "
                         f"{str(e.get('front', ''))[:100]} "
                         f"[due: {e.get('due')}]")
        return "\n".join(lines)
    if verb == "review":
        parts = tail.split()
        if len(parts) < 2:
            return "usage: /learn flashcards review <id> <grade 1-4>"
        try:
            grade = int(parts[1])
        except ValueError:
            return "grade must be 1-4."
        if not 1 <= grade <= 4:
            return "grade must be 1-4."
        if not nb.review(parts[0], grade):
            return f"no card with id {parts[0]}."
        return (f"✅ graded card {parts[0]}: {grade}/4 (saved — "
                "the grade is in your mistake notebook).")
    if verb == "export":
        if not tail:
            return "usage: /learn flashcards export <path>"
        path = nb.export_anki(tail)
        return f"📤 exported {nb.count()} card(s) to {path} (Anki TSV)."
    return ("usage: /learn flashcards due [n] | stats | list | "
            "review <id> <grade 1-4> | export <path>")


# ── mastery ───────────────────────────────────────────────────────────────


def _mastery(rest: str, *, chat_key: str = "") -> str:
    from .tutor import get_session, restore_session

    sess = get_session(chat_key) or restore_session(chat_key)
    if sess is None:
        return "no tutoring session — /learn tutor <topic> to start one."
    snap = sess.mastery.snapshot()
    if not snap:
        return f"📚 mastery for '{sess.topic}': no skills tracked yet."
    if rest.lower() == "weakest":
        skill = sess.mastery.weakest()
        return f"📉 weakest skill: {skill} ({snap.get(skill, 0):.0%})"
    lines = [f"📚 mastery — '{sess.topic}':"]
    for skill, val in sorted(snap.items(), key=lambda kv: kv[1]):
        bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
        lines.append(f"  {skill}: {bar} {val:.0%}")
    lines.append(f"weakest: {sess.mastery.weakest()} · "
                 f"strongest: {sess.mastery.strongest()}")
    return "\n".join(lines)


# ── WAEC / JAMB ────────────────────────────────────────────────────────────


def _waec(rest: str) -> str:
    from .curriculum import SUBJECTS, subject_topics, waec_grade

    low = rest.lower()
    if low.startswith("grade"):
        arg = rest[len("grade"):].strip().strip("%")
        try:
            pct = float(arg)
        except ValueError:
            return "usage: /learn waec grade <pct> — e.g. /learn waec grade 72"
        return f"WAEC grade for {pct:.0f}%: {waec_grade(pct)}"
    if not rest:
        return ("usage: /learn waec <subject> | /learn waec grade <pct>\n"
                f"subjects: {', '.join(SUBJECTS)}")
    topics = subject_topics(rest)
    if not topics:
        return ("unknown subject — try: " + ", ".join(SUBJECTS))
    lines = [f"📖 WAEC {topics[0].subject} syllabus ({len(topics)} topics):"]
    for t in topics:
        jamb = " ·JAMB" if t.jamb else ""
        lines.append(f"  {t.code.split()[-1]} {t.title}{jamb}")
    lines.append("Codes are the stable internal scheme (see learn/curriculum.py) — "
                 "not official WAEC section numbers.")
    return "\n".join(lines)


def _jamb(rest: str) -> str:
    from .curriculum import EXAM_FORMATS, SUBJECTS, subject_topics

    low = rest.lower()
    if low == "formats" or low == "format":
        fmt = EXAM_FORMATS.get("JAMB", {})
        papers = fmt.get("papers", [])
        lines = ["📝 JAMB format:"]
        for p in papers:
            lines.append(f"  • {p.get('name', '?')} — "
                         f"{p.get('minutes', '?')} min")
        lines.append(f"negative marking: {fmt.get('negative_marking', False)}")
        return "\n".join(lines)
    if not rest:
        return ("usage: /learn jamb <subject> | /learn jamb formats\n"
                f"subjects: {', '.join(SUBJECTS)}")
    topics = [t for t in subject_topics(rest) if t.jamb]
    if not topics:
        all_topics = subject_topics(rest)
        if not all_topics:
            return ("unknown subject — try: " + ", ".join(SUBJECTS))
        return (f"no JAMB-flagged topics for {all_topics[0].subject} — "
                "try /learn waec <subject> for the full syllabus.")
    lines = [f"📖 JAMB {topics[0].subject} topics ({len(topics)}):"]
    for t in topics:
        lines.append(f"  {t.code.split()[-1]} {t.title}")
    return "\n".join(lines)
