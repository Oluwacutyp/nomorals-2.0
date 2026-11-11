"""Goal-class surface: board, briefings, snapshot (WS2 P2).

The presentation layer over :class:`GoalTracker` + the class
registry (:mod:`nomorals.goals.classes`), shared by every surface
that shows the owner's goals grouped by class:

* chat ``/goals`` (agents/partner/runtime_system.py),
* the ``nm goals`` CLI (cmdline/commands/goals.py),
* the morning briefing's goal-classes section
  (agents/morning_briefing.py),
* the brain context snapshot (:func:`goals_context_snapshot`,
  rendered by context/engine.py's goals section).

Everything here is synchronous: the tracker is async, so its
coroutines are driven through :func:`run_sync` (the house pattern
for sync callers — cf. accounts/provisioning._run_coro).  Rendering
is pure text; surfaces that can carry buttons wrap the board text
in their own actionable message.

Honesty contract (from P1, carried to every render): a goal's
progress number is shown WITH its basis — ``(measured · <source>)``
when a class provider measured it from a real store, or
``(self-reported)`` when it is the goal's own manual %.  The two are
never rendered indistinguishable.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from . import Goal, GoalTracker
from .classes import (
    DEFAULT_CLASS_ID,
    ClassProgress,
    GoalClass,
    get_class,
    list_classes,
    validate_class_id,
)

__all__ = [
    "OWNER_USER_ID",
    "class_briefing_lines",
    "create_goal_filed",
    "goals_context_snapshot",
    "interest_watch_note",
    "progress_label",
    "refile_goal",
    "render_briefing",
    "render_class_board",
    "render_classes",
    "render_focus",
    "render_goal_progress",
    "render_review",
    "resolve_goal",
    "run_sync",
    "suggest_class",
]

#: The chat/CLI surfaces are the owner's own; GoalTracker rows for
#: the owner live under this user id (the ``nm idea`` CLI default).
OWNER_USER_ID = "owner"


def run_sync(coro: Any) -> Any:
    """Drive a tracker coroutine from synchronous surface code.

    Plain ``asyncio.run`` when no loop is running; when called from
    inside a running loop, the coroutine runs on a throwaway thread
    with its own loop and the result (or exception) is ferried back
    — the accounts/provisioning pattern.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - ferried to caller
            box["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ── progress labels ───────────────────────────────────────────────────


def progress_label(reading: ClassProgress) -> str:
    """One goal's progress, basis always attached.

    ``62% (measured · ledger)`` / ``40% (self-reported)`` — the
    label is rendered verbatim on every surface so a measured
    number and a manual one can never be confused.
    """
    if reading.basis == "measured":
        if reading.percent is None:
            return f"not measured yet (provider: {reading.source})"
        return f"{reading.percent:.0f}% (measured · {reading.source})"
    percent = reading.percent if reading.percent is not None else 0.0
    return f"{percent:.0f}% (self-reported)"


def _goal_line(tracker: GoalTracker, goal: Goal, *, now: float) -> str:
    """One board line: title, progress with basis, overdue called out."""
    reading = tracker.class_progress(goal, now=now)
    line = f"  • {goal.title} — {progress_label(reading)}"
    if goal.target_date is not None and goal.status == "active":
        if goal.target_date < now:
            days = int((now - goal.target_date) / 86400)
            line += f"  ⛔ overdue by {days}d"
        else:
            days = int((goal.target_date - now) / 86400)
            if days <= 7:
                line += f"  ⏰ due in {days}d"
    return line


# ── the class board ───────────────────────────────────────────────────


def render_class_board(
    tracker: GoalTracker,
    user_id: str,
    *,
    only_class: str | None = None,
    now: float | None = None,
) -> str:
    """Goals grouped by class, with per-class rollups from stats().

    Every registered class renders: classes with goals get their
    rollup (counts, average progress, overdue) plus one line per
    active goal; classes without goals get a designed empty state
    (what the class is + how a goal joins it) — never a bare blank.
    ``only_class`` renders just that class's section (validated —
    an unknown class raises GoalClassError listing the classes).

    Presentation Phase 3: the board is a leaderboard-class Card —
    one untitled block per class, every line byte-identical to the
    old hand-joined text (the class heads carry their own
    ``**Title** (`id`)`` markers as prose, so the pinned copy
    survives), framed by the card title and stamp.  Rendered
    canonical; the gateway converts per platform at send.
    """
    now = time.time() if now is None else now
    stats = run_sync(tracker.stats(user_id))
    by_class = stats.get("by_class") or {}
    active = run_sync(tracker.list_goals(user_id, status="active"))
    active_by_class: dict[str, list[Goal]] = {}
    for goal in active:
        active_by_class.setdefault(goal.goal_class or DEFAULT_CLASS_ID,
                                   []).append(goal)

    descriptors = ([get_class(only_class)] if only_class is not None
                   else list_classes())

    from ..social.chat.typography import Card, card_text

    card = Card("leaderboard", title="Goals")
    if only_class is None:
        focus = run_sync(tracker.today_focus(user_id, limit=3, now=now))
        if focus:
            focus_lines = ["🎯 **Focus now:**"]
            for entry in focus:
                goal = entry["goal"]
                reading = tracker.class_progress(goal, now=now)
                focus_lines.append(
                    f"  • {goal.title} — {progress_label(reading)}")
            card.block_lines(focus_lines, title="")
        total = stats.get("total", 0)
        card.block_lines(
            [f"🧭 **Goal classes** — {total} goal(s) on file"], title="")

    for descriptor in descriptors:
        rollup = by_class.get(descriptor.id)
        section = [f"**{descriptor.title}** (`{descriptor.id}`)"]
        if not rollup:
            section.append(
                f"  🗒 nothing here yet — {descriptor.description}")
            section.append(
                f"  file one: /goals add <title> in {descriptor.id}")
            card.block_lines(section, title="")
            continue
        overdue = rollup.get("overdue", 0)
        section.append(
            f"  {rollup.get('total', 0)} goal(s) · avg "
            f"{rollup.get('avg_progress', 0.0):.0f}%"
            + (f" · ⛔ {overdue} overdue" if overdue else ""))
        for goal in active_by_class.get(descriptor.id, []):
            section.append(_goal_line(tracker, goal, now=now))
        card.block_lines(section, title="")
    return card_text(card)


def render_classes() -> str:
    """The class picker as text: every class, what it is, how to file.

    Presentation Phase 3: a menu-class Card, the picker lines
    verbatim inside it."""
    from ..social.chat.typography import Card, card_text

    card = Card("menu", title="Goal classes")
    lines = ["🧭 **Goal classes:**"]
    for descriptor in list_classes():
        lines.append(
            f"  `{descriptor.id}` — {descriptor.title}: "
            f"{descriptor.description}")
    lines.append("")
    lines.append("file a goal: /goals add <title> in <class>")
    lines.append("move a goal: /goals class <goal> <class>")
    card.block_lines(lines, title="")
    return card_text(card)


# ── the dead methods, wired ───────────────────────────────────────────


def render_briefing(tracker: GoalTracker, user_id: str) -> str:
    """GoalTracker's get_briefing — its first production caller.

    Presentation Phase 3 remainder: a briefing-class Card
    framing the tracker's own lines verbatim (the tracker owns
    the formatting; this surface owns the presentation).  The
    empty answer stays a plain honest line — no board exists
    to frame."""
    text = run_sync(tracker.get_briefing(user_id))
    if not text.strip() or text.strip() == "No active goals.":
        return text
    from ..social.chat.typography import card_for_text

    return card_for_text(text, "briefing",
                         title="Goals — progress briefing")


def render_review(tracker: GoalTracker, user_id: str, *,
                  now: float | None = None) -> str:
    """GoalTracker's weekly_review, formatted by its own formatter.

    Presentation Phase 3 remainder: a briefing-class Card
    framing the tracker's own lines verbatim."""
    review = run_sync(tracker.weekly_review(user_id, now=now))
    text = tracker.format_weekly_review(review)
    from ..social.chat.typography import card_for_text

    return card_for_text(text, "briefing",
                         title="Goals — weekly review")


def render_focus(tracker: GoalTracker, user_id: str, *,
                 now: float | None = None) -> str:
    """today_focus as a chat board: ranked goals with their scores.

    Presentation Phase 3: a leaderboard-class Card, the ranked
    lines verbatim inside it.  The empty answer stays a plain
    honest line — no board exists to frame."""
    focus = run_sync(tracker.today_focus(user_id, limit=5, now=now))
    if not focus:
        return "no active goals to focus on yet — /goals add one first."
    from ..social.chat.typography import Card, card_text

    card = Card("leaderboard", title="Focus")
    lines = ["🎯 **Focus now** (ranked by urgency, deadlines, momentum):"]
    for i, entry in enumerate(focus, 1):
        goal = entry["goal"]
        descriptor = get_class(goal.goal_class)
        reading = tracker.class_progress(goal, now=now)
        lines.append(
            f"  {i}. {goal.title} — {progress_label(reading)}"
            f" · {descriptor.title} · score {entry['score']:.1f}")
    card.block_lines(lines, title="")
    return card_text(card)


def render_goal_progress(tracker: GoalTracker, user_id: str, ref: str, *,
                         now: float | None = None) -> str:
    """One goal's progress: class reading + pace verdict (pace_report).

    Presentation Phase 3: a briefing-class Card, every line
    verbatim inside it.  The unknown-goal answer stays a plain
    honest line — no record exists to frame."""
    goal, candidates = resolve_goal(tracker, user_id, ref)
    if goal is None:
        return _unknown_goal_text(ref, candidates)
    descriptor = get_class(goal.goal_class)
    reading = run_sync(tracker.measured_progress(goal.goal_id, now=now))
    lines = [
        f"**{goal.title}** — {descriptor.title} (`{descriptor.id}`)",
        f"  progress: {progress_label(reading)}",
    ]
    if reading.headline and reading.headline != "self-reported":
        lines.append(f"  {reading.headline}")
    if goal.target_date is not None:
        report = run_sync(tracker.pace_report(goal.goal_id, now=now))
        verdict = str(report.get("verdict") or "")
        detail = ""
        req, act = report.get("required_daily_pace"), report.get("actual_daily_pace")
        if req is not None and act is not None:
            detail = f" — needs {req:.1f}%/day, doing {act:.1f}%/day"
        elif req is not None:
            detail = f" — needs {req:.1f}%/day"
        days_left = report.get("days_left")
        if days_left is not None:
            detail += f" · {days_left:.0f} days left"
        lines.append(f"  pace: {verdict}{detail}")
    subgoals = goal.subgoals or []
    if subgoals:
        done = sum(1 for s in subgoals if s.is_completed)
        lines.append(f"  subgoals: {done}/{len(subgoals)} completed")
    lines.append(f"  goal id: {goal.goal_id}")
    from ..social.chat.typography import Card, card_text

    card = Card("briefing", title="Goal progress")
    card.block_lines(lines, title="")
    return card_text(card)


# ── goal resolution / filing ──────────────────────────────────────────


def resolve_goal(
    tracker: GoalTracker, user_id: str, ref: str,
) -> tuple[Goal | None, list[Goal]]:
    """Resolve a chat goal reference to one goal.

    Accepts a goal id (exact or unique prefix), ``#n`` (1-based into
    the active list, board order), or a unique title substring.
    Returns ``(goal, [])`` on a hit, ``(None, candidates)`` when the
    reference is ambiguous or matches nothing (candidates = the
    near matches, or every active goal when nothing matched).
    """
    ref = (ref or "").strip()
    active = run_sync(tracker.list_goals(user_id, status="active"))
    if not ref:
        return None, active
    exact = run_sync(tracker.get(ref))
    if exact is not None and exact.user_id == user_id:
        return exact, []
    if ref.startswith("#") and ref[1:].isdigit():
        idx = int(ref[1:]) - 1
        if 0 <= idx < len(active):
            return active[idx], []
        return None, active
    by_prefix = [g for g in active if g.goal_id.startswith(ref)]
    if len(by_prefix) == 1:
        return by_prefix[0], []
    needle = ref.lower()
    by_title = [g for g in active if needle in g.title.lower()]
    if len(by_title) == 1:
        return by_title[0], []
    if len(by_title) > 1:
        return None, by_title
    return None, active


def _unknown_goal_text(ref: str, candidates: list[Goal]) -> str:
    lines = [f"no single goal matches {ref!r}."]
    if candidates:
        lines.append("active goals:")
        for goal in candidates[:10]:
            descriptor = get_class(goal.goal_class)
            lines.append(
                f"  • {goal.title} ({descriptor.id}) — {goal.goal_id}")
    else:
        lines.append("no active goals yet — /goals add one first.")
    return "\n".join(lines)


#: Extra whole-word cues for the create-time class suggestion,
#: beyond the class ids/titles themselves.  Drawn from the built-in
#: templates' own vocabulary (save-money → finance, fitness →
#: health, learn-language / read-books → learning): a suggestion is
#: named in the create reply and re-filing is one command, so a
#: conservative cue list is enough.
_SUGGEST_WORDS: dict[str, str] = {
    "save": "finance", "saving": "finance", "savings": "finance",
    "money": "finance", "budget": "finance",
    "fitness": "health", "workout": "health",
    "language": "learning", "books": "learning", "reading": "learning",
    "study": "learning", "learn": "learning",
    "job": "careers", "jobs": "careers", "interview": "careers",
    "trade": "trading", "trading": "trading",
}


def suggest_class(title: str) -> GoalClass | None:
    """Deterministic create-time class suggestion from the title.

    Whole-word match against class ids/titles first, then the
    template-vocabulary cue words.  No model call; returns None when
    nothing matches (the goal files under ``general``).
    """
    words = {w.strip(".,;:!?()[]{}\"'").lower()
             for w in (title or "").split()}
    words.discard("")
    if not words:
        return None

    def _matches(word: str, token: str) -> bool:
        return word == token or word == token + "s" or word + "s" == token

    for descriptor in list_classes():
        if descriptor.id == DEFAULT_CLASS_ID:
            continue
        names = {descriptor.id.lower(), descriptor.title.lower()}
        if any(_matches(w, n) for w in words for n in names):
            return descriptor
    for word in sorted(words):
        target = _SUGGEST_WORDS.get(word)
        if target:
            return get_class(target)
    return None


def create_goal_filed(
    tracker: GoalTracker,
    user_id: str,
    title: str,
    class_arg: str | None,
) -> tuple[Goal, str, bool]:
    """Create a goal filed under a class.

    An explicit ``class_arg`` wins (validated — unknown ids raise
    GoalClassError).  Without one, the deterministic title matcher
    proposes a class; the returned flag says whether the filing was
    a suggestion (the caller names it in the reply so it is never
    silent).  Returns ``(goal, class_id, suggested)``.

    An interests filing also registers the goal's watch topics as
    news beats (WS2 P5 — the census's settled behaviour change;
    :func:`nomorals.goals.providers.register_interest_beats` is
    best-effort and never fails the creation).  Callers state the
    live watch in the reply via :func:`interest_watch_note`.
    """
    if class_arg:
        class_id = validate_class_id(class_arg)
        goal = run_sync(tracker.create(user_id, title, goal_class=class_id))
        suggested = False
    else:
        suggestion = suggest_class(title)
        if suggestion is not None:
            goal = run_sync(
                tracker.create(user_id, title, goal_class=suggestion.id))
            class_id, suggested = suggestion.id, True
        else:
            goal = run_sync(tracker.create(user_id, title))
            class_id = goal.goal_class or DEFAULT_CLASS_ID
            suggested = False
    if class_id == "interests":
        from . import providers

        providers.register_interest_beats(tracker.db, goal)
    return goal, class_id, suggested


def interest_watch_note(tracker: GoalTracker, goal: Goal) -> str:
    """The create-reply statement of an interests goal's live watch.

    Names the goal's topics that are news beats right now (its own
    registration just ran, or they were already covered) — the
    effect stated truthfully in both cases.  Empty when the goal
    is not an interests goal or no topic is a beat (e.g. the news
    profile could not be read).  Never raises.
    """
    try:
        if (goal.goal_class or DEFAULT_CLASS_ID) != "interests":
            return ""
        from . import providers

        profile = providers._load_news_profile(tracker.db)
        if profile is None:
            return ""
        beats = {str(b.get("topic") or "").casefold()
                 for b in profile["beats"]}
        live = [t for t in providers.interest_topics(goal)
                if t.casefold() in beats]
        if not live:
            return ""
        return ("📡 this watch is live — its topics are news beats: "
                + " · ".join(live)
                + " (re-file the goal out of Interests to stop "
                  "watching)")
    except Exception:  # noqa: BLE001 - a reply extra, never a failure
        return ""


def refile_goal(
    tracker: GoalTracker, user_id: str, ref: str, class_arg: str,
) -> str:
    """Move one goal to another class (update(goal_class=...)).

    Moving INTO interests registers the goal's watch topics as
    news beats; moving OUT un-files this goal's beats (seed beats,
    stated briefing topics and topics other interests goals still
    claim are kept — see
    :func:`nomorals.goals.providers.remove_interest_beats`; the
    profile carries no per-beat provenance, so the un-filed set is
    the goal's topics minus those protections).  Both acts are
    named in the reply.
    """
    class_id = validate_class_id(class_arg)
    goal, candidates = resolve_goal(tracker, user_id, ref)
    if goal is None:
        return _unknown_goal_text(ref, candidates)
    before = get_class(goal.goal_class)
    updated = run_sync(
        tracker.update(goal.goal_id, goal_class=class_id))
    after = get_class(updated.goal_class)
    text = (f"moved **{updated.title}** from {before.title} "
            f"to {after.title} (`{after.id}`).")
    if after.id == "interests" and before.id != "interests":
        from . import providers

        added = providers.register_interest_beats(tracker.db, updated)
        if added:
            text += ("\n  📡 registered as news beat(s): "
                     + " · ".join(added))
        else:
            text += "\n  📡 its watch topics were already news beats."
    elif before.id == "interests" and after.id != "interests":
        from . import providers

        keep: list[str] = []
        for other in run_sync(tracker.list_goals(
                user_id, status="active", goal_class="interests")):
            keep.extend(providers.interest_topics(other))
        removed = providers.remove_interest_beats(
            tracker.db, goal, keep_topics=keep)
        if removed:
            text += ("\n  📡 stopped watching — removed news beat(s): "
                     + " · ".join(removed))
    return text


# ── morning briefing one-liners ───────────────────────────────────────


def class_briefing_lines(
    tracker: GoalTracker, user_id: str, *, now: float | None = None,
) -> list[str]:
    """Per-class one-liners for the morning briefing's goals section.

    One line per class that actually has goals: counts, average
    progress, overdue called out.  Empty list when the owner has no
    goals at all (the briefing section then omits itself).
    """
    now = time.time() if now is None else now
    stats = run_sync(tracker.stats(user_id))
    by_class = stats.get("by_class") or {}
    if not by_class:
        return []
    active = run_sync(tracker.list_goals(user_id, status="active"))
    active_counts: dict[str, int] = {}
    for goal in active:
        key = goal.goal_class or DEFAULT_CLASS_ID
        active_counts[key] = active_counts.get(key, 0) + 1
    lines: list[str] = []
    for descriptor in list_classes():
        rollup = by_class.get(descriptor.id)
        if not rollup:
            continue
        overdue = rollup.get("overdue", 0)
        line = (
            f"• {descriptor.title}: {active_counts.get(descriptor.id, 0)} "
            f"active of {rollup.get('total', 0)} · avg "
            f"{rollup.get('avg_progress', 0.0):.0f}%"
            + (f" · ⛔ {overdue} overdue" if overdue else ""))
        lines.append(line)
    return lines


# ── brain context snapshot ────────────────────────────────────────────

#: Hard bound for the context snapshot: the top goals by the
#: tracker's own focus scoring.  The context engine caps again at
#: render time; this bound keeps the snapshot itself cheap.
SNAPSHOT_LIMIT = 8


def goals_context_snapshot(
    tracker: GoalTracker,
    user_id: str,
    *,
    limit: int = SNAPSHOT_LIMIT,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Top active goals by focus score, as context-section entries.

    Each entry carries everything the brain needs to speak about the
    goal honestly: title, class (id + title), the progress reading
    WITH its basis and source, and the pace verdict for dated goals
    that are not simply on track.  Empty list when there are no
    active goals (the context section then omits itself).
    """
    now = time.time() if now is None else now
    limit = max(1, limit)
    ranked = run_sync(
        tracker.today_focus(user_id, limit=limit, now=now))
    entries: list[dict[str, Any]] = []
    for entry in ranked:
        goal = entry["goal"]
        descriptor = get_class(goal.goal_class)
        reading = tracker.class_progress(goal, now=now)
        item: dict[str, Any] = {
            "title": goal.title,
            "goal_class": descriptor.id,
            "class_title": descriptor.title,
            "percent": reading.percent,
            "basis": reading.basis,
            "source": reading.source,
            "focus": entry["score"],
            "overdue": bool(goal.target_date is not None
                            and goal.target_date < now
                            and goal.status == "active"),
        }
        if goal.target_date is not None:
            report = run_sync(tracker.pace_report(goal.goal_id, now=now))
            verdict = str(report.get("verdict") or "")
            if verdict and verdict not in ("on_track", "complete"):
                item["pace"] = verdict
        entries.append(item)
    return entries
