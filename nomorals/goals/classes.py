"""Goal classes — the domain axis for goals (WS2).

Descriptors as data, following the repo's own precedent
(``preferences/store.py`` — *"descriptors are data… adding a domain
never touches the learning logic"*): :data:`GOAL_CLASSES` names the
classes; nothing else in the system branches on class ids.  Adding a
class = one registry entry (+ one provider function when its organ
lands).

A class is not a label.  A domain earns a class when it has a distinct
*measured* progress model (progress computed from a real store, not a
self-reported %), a distinct periodic review, a distinct tool/organ
set, and a distinct memory shape.  Domains that fail that test stay
tags (see ``nomorals/goals/__init__.py``'s ``GOAL_TEMPLATES`` — the
``declutter`` template's ``home`` tag stays a tag).

The axis is orthogonal to ``GOAL_KINDS`` (the goal's *shape*:
outcome / habit / project).  ``general`` is the unclassed default: a
goal with no class keeps the pre-WS2 behaviour exactly.

Progress providers
------------------
A provider is a read-only callable with the signature::

    provider(db, goal, *, now) -> ClassProgress | None

It returns a *measured* reading from its organ's real store, or
``None`` when it cannot measure this goal.  When no provider measures
a goal, :func:`goal_progress` falls back to the goal's manual % and
labels it ``basis="self-reported"`` — measured and manual numbers are
never indistinguishable (the OFF-vs-BROKEN honesty rule applied to
numbers).  Providers register per class with
:func:`register_progress_provider`; P1 ships the registry empty on
purpose — the per-class providers land with their organs in WS2
P3/P5/P6 and plug in here without touching the goal store.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a cycle
    from ..storage.db import Database
    from . import Goal

__all__ = [
    "CADENCE_DAYS",
    "DEFAULT_CLASS_ID",
    "GOAL_CLASSES",
    "ClassProgress",
    "GoalClass",
    "GoalClassError",
    "cadence_days",
    "class_for_template",
    "define_custom_class",
    "ensure_goal_classes_table",
    "get_class",
    "get_class_for",
    "goal_progress",
    "list_all_classes",
    "list_classes",
    "list_custom_classes",
    "register_progress_provider",
    "remove_custom_class",
    "require_class",
    "validate_class_id",
    "validate_class_id_for",
]


class GoalClassError(ValueError):
    """Raised when a goal-class id is not in the registry.

    A ValueError subclass so the goal store's existing input-error
    handling (``create``/``update`` raise ValueError for bad kinds)
    treats an unknown class the same way.
    """


#: The unclassed default.  Goals created before the class axis (or
#: without a class) read as this and behave exactly as they did.
DEFAULT_CLASS_ID = "general"


@dataclass(frozen=True)
class GoalClass:
    """One goal class, as data.

    ``progress_model`` names the provider key that measures progress
    for this class (see :func:`register_progress_provider`); a model
    of ``"manual"`` means the class has no measuring organ yet and its
    goals' progress is self-reported by construction.
    ``review_cadence`` names where the class's periodic review rides
    (an existing digest/briefing cadence, or ``"weekly"``).
    ``summarizer_key`` is the key the focus weekly review's injected
    summarizers resolve for this class's section.  ``template_names``
    are the ``GOAL_TEMPLATES`` keys this class owns — a goal created
    from one of those templates is filed under this class.
    ``tool_hints`` are the organ/tool names the brain may reach for
    when working a goal of this class.  ``milestone_templates`` are
    the starter milestones a goal of this class is seeded with at
    creation (see ``GoalTracker.seed_class_milestones``) — what a
    milestone looks like for this class, as data, not a code branch.
    """

    id: str
    title: str
    description: str
    progress_model: str
    review_cadence: str
    summarizer_key: str = ""
    template_names: tuple[str, ...] = ()
    tool_hints: tuple[str, ...] = ()
    milestone_templates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "progress_model": self.progress_model,
            "review_cadence": self.review_cadence,
            "summarizer_key": self.summarizer_key,
            "template_names": list(self.template_names),
            "tool_hints": list(self.tool_hints),
            "milestone_templates": list(self.milestone_templates),
        }


#: The class registry.  Order is presentation order (owner-named
#: classes first, ``general`` last as the fallback).
GOAL_CLASSES: dict[str, GoalClass] = {
    "careers": GoalClass(
        id="careers",
        title="Careers",
        description=(
            "Career moves and the job pipeline: target roles and "
            "companies, applications by stage, interviews, offers."),
        progress_model="application_pipeline",
        review_cadence="weekly",
        summarizer_key="careers",
        tool_hints=("matching", "documents", "artifacts", "learn"),
        milestone_templates=("Submit the first application", "Reach the interview stage", "Receive an offer"),
    ),
    "productivity": GoalClass(
        id="productivity",
        title="Productivity",
        description=(
            "Getting the right things done: commitments kept, focus "
            "time, plans that survive contact with the week."),
        progress_model="commitment_reliability",
        review_cadence="weekly",
        summarizer_key="productivity",
        tool_hints=("commitments", "focus", "planning", "reminders"),
        milestone_templates=("Keep a 7-day streak", "Keep a 30-day streak", "Clear the week's focus list"),
    ),
    "finance": GoalClass(
        id="finance",
        title="Finance",
        description=(
            "Household money: savings targets, budgets, bills and "
            "the ledger — measured from the ledger, not guessed."),
        progress_model="ledger_funded",
        review_cadence="digest:finance",
        summarizer_key="finance",
        template_names=("save-money",),
        tool_hints=("finance", "ledger", "budgets", "alerts"),
        milestone_templates=("Reach 25% funded", "Reach 50% funded", "Fully funded"),
    ),
    "interests": GoalClass(
        id="interests",
        title="Interests",
        description=(
            "Watches over topics that matter: news beats, findings "
            "delivered, what surfaced and what went quiet."),
        progress_model="watch_coverage",
        review_cadence="daily-brief",
        summarizer_key="interests",
        tool_hints=("news", "watchers", "briefing"),
        milestone_templates=("First finding delivered", "Watch reaches 10 findings"),
    ),
    "health": GoalClass(
        id="health",
        title="Health",
        description=(
            "Body and mind: training, sleep, nutrition and the "
            "signals from synced devices."),
        progress_model="health_adherence",
        review_cadence="weekly",
        summarizer_key="health",
        template_names=("fitness",),
        tool_hints=("health", "coach", "timeline"),
        milestone_templates=("Log 7 days in a row", "Log 30 days in a row"),
    ),
    "learning": GoalClass(
        id="learning",
        title="Learning",
        description=(
            "Skills and study: mastery per topic, flashcards due, "
            "curriculum coverage."),
        progress_model="mastery",
        review_cadence="weekly",
        summarizer_key="learning",
        template_names=("learn-language", "read-books"),
        tool_hints=("learn", "tutor", "flashcards", "curriculum"),
        milestone_templates=("Finish the first study session", "Reach 80% mastery on a topic"),
    ),
    "build": GoalClass(
        id="build",
        title="Build",
        description=(
            "Things being made and shipped: projects, missions and "
            "the work that turns an idea into a launched thing."),
        progress_model="execution_steps",
        review_cadence="weekly",
        summarizer_key="build",
        template_names=("ship-project",),
        tool_hints=("goal", "missions", "skills", "projects"),
        milestone_templates=("Ship the first version", "Launch"),
    ),
    "relationships": GoalClass(
        id="relationships",
        title="Relationships",
        description=(
            "People and promises: commitments to specific people, "
            "kept-rate and cadence with who matters."),
        progress_model="person_commitments",
        review_cadence="weekly",
        summarizer_key="relationships",
        tool_hints=("commitments", "family", "relationships"),
        milestone_templates=("Keep every promise this week",),
    ),
    "trading": GoalClass(
        id="trading",
        title="Trading",
        description=(
            "The trading desk: equity and P&L against target, read "
            "from the trading journal and GoldenSwing evidence."),
        progress_model="trading_journal",
        review_cadence="digest:trading",
        summarizer_key="trading",
        tool_hints=("trading_desk", "trade_pipeline", "goldenswing"),
        milestone_templates=("First realized profit", "Hit the P&L target"),
    ),
    DEFAULT_CLASS_ID: GoalClass(
        id=DEFAULT_CLASS_ID,
        title="General",
        description=(
            "Unclassed goals — no domain organ measures these; "
            "progress is whatever the goal itself records."),
        progress_model="manual",
        review_cadence="weekly",
        summarizer_key="",
        tool_hints=(),
    ),
}


def list_classes() -> list[GoalClass]:
    """All registered classes, in registry (presentation) order."""
    return list(GOAL_CLASSES.values())


def get_class(class_id: str | None) -> GoalClass:
    """The descriptor for ``class_id``; ``None``/empty → ``general``.

    Raises :class:`GoalClassError` for an id that is not registered —
    an unknown class is refused in words, never silently coerced.
    """
    if class_id is None or not str(class_id).strip():
        return GOAL_CLASSES[DEFAULT_CLASS_ID]
    key = str(class_id).strip().lower()
    descriptor = GOAL_CLASSES.get(key)
    if descriptor is None:
        known = ", ".join(sorted(GOAL_CLASSES))
        raise GoalClassError(
            f"unknown goal class {class_id!r} (known classes: {known})")
    return descriptor


def validate_class_id(class_id: str | None) -> str:
    """Normalise a class id for storage; unknown ids raise.

    ``None``/empty means the default class.
    """
    return get_class(class_id).id


def require_class(class_id: str) -> GoalClass:
    """Strict lookup for a stored/explicit id (no defaulting)."""
    if class_id is None or not str(class_id).strip():
        raise GoalClassError(
            "a goal class id is required (known classes: "
            f"{', '.join(sorted(GOAL_CLASSES))})")
    return get_class(class_id)


def class_for_template(template_name: str | None) -> str | None:
    """The class a built-in template's goals are filed under.

    Resolves through each class's ``template_names``.  A built-in
    template owned by no class (``declutter`` — home is a tag, not a
    class) files under ``general``.  Returns ``None`` when the name is
    not a built-in template at all (custom templates carry no class
    mapping; their goals stay ``general`` unless filed explicitly).
    """
    if not template_name:
        return None
    name = str(template_name).strip()
    for descriptor in GOAL_CLASSES.values():
        if name in descriptor.template_names:
            return descriptor.id
    # Imported lazily: goals/__init__ imports this module, so the
    # reverse import must not happen at module load time.
    from . import GOAL_TEMPLATES

    if name in GOAL_TEMPLATES:
        return DEFAULT_CLASS_ID
    return None


# ── custom classes: "something else", as data ─────────────────────────
#
# The built-in registry above is code.  A class the owner names that is
# not built in ("something else") must NOT need a code change to exist:
# custom classes live in the ``goal_classes`` table of the same database
# the goals live in (created by ``GoalTracker._ensure_schema``; the
# functions below create it idempotently too, so a bare database works).
# A custom class is data with the same shape as a built-in descriptor —
# title, description, progress model, review cadence, tool hints,
# milestone templates — and its goals measure ``manual`` (self-reported)
# unless a provider is registered for it later: no measuring organ
# exists for a class nobody has wired one for, and the reading says so.
# ``list_classes()`` / ``get_class()`` / ``validate_class_id()`` stay
# built-in-only (the ten-class pin); the ``*_for`` / ``list_all_*``
# functions below are the db-aware path every writer and renderer
# uses, so a custom-class goal never breaks a board, a briefing, or
# the brain snapshot.

import json as _json
import re as _re

#: Custom class ids: lowercase slug, must start with a letter.
_CUSTOM_ID_RE = _re.compile(r"^[a-z][a-z0-9-]{0,31}$")

#: Review-cadence names → check-in window in days.  ``digest:*``
#: cadences ride the weekly digest, so they read as weekly here.
CADENCE_DAYS: dict[str, int] = {
    "daily-brief": 1,
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
}


def cadence_days(cadence: str | None) -> int:
    """Check-in window in days for a review-cadence name.

    ``digest:*`` → 7; an unknown/empty cadence → 7 (weekly is the
    standing default, never a crash).
    """
    name = (cadence or "").strip()
    if name in CADENCE_DAYS:
        return CADENCE_DAYS[name]
    if name.startswith("digest:"):
        return 7
    return 7


def ensure_goal_classes_table(db: Any) -> None:
    """Create the ``goal_classes`` table when it is missing (idempotent)."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS goal_classes (
            class_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            progress_model TEXT NOT NULL DEFAULT 'manual',
            review_cadence TEXT NOT NULL DEFAULT 'weekly',
            summarizer_key TEXT NOT NULL DEFAULT '',
            tool_hints TEXT NOT NULL DEFAULT '[]',
            milestone_templates TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL
        )
    """)


def _row_to_custom_class(row: dict[str, Any]) -> GoalClass:
    def _str_list(raw: Any) -> tuple[str, ...]:
        try:
            values = _json.loads(raw or "[]")
        except (ValueError, TypeError):
            return ()
        return tuple(str(v) for v in values if str(v).strip())

    return GoalClass(
        id=str(row["class_id"]),
        title=str(row["title"]),
        description=str(row.get("description") or ""),
        progress_model=str(row.get("progress_model") or "manual"),
        review_cadence=str(row.get("review_cadence") or "weekly"),
        summarizer_key=str(row.get("summarizer_key") or ""),
        tool_hints=_str_list(row.get("tool_hints")),
        milestone_templates=_str_list(row.get("milestone_templates")),
    )


def _custom_classes(db: Any) -> dict[str, GoalClass]:
    """Custom classes from the db, keyed by id; {} when unavailable.

    Any store problem (no table yet, bare :memory: db, no db at all)
    reads as "no custom classes" — the built-ins still resolve, so a
    board over a database that predates the table never breaks.
    """
    if db is None:
        return {}
    try:
        rows = db.query("SELECT * FROM goal_classes ORDER BY class_id")
    except Exception:  # noqa: BLE001 - no table yet = no custom classes
        return {}
    out: dict[str, GoalClass] = {}
    for row in rows or []:
        try:
            descriptor = _row_to_custom_class(row)
        except Exception:  # noqa: BLE001 - one bad row is not all of them
            continue
        out[descriptor.id] = descriptor
    return out


def list_custom_classes(db: Any) -> list[GoalClass]:
    """Custom (owner-defined) classes, in id order."""
    return [_custom_classes(db)[k] for k in sorted(_custom_classes(db))]


def list_all_classes(db: Any) -> list[GoalClass]:
    """Every class: the built-ins in presentation order, then customs."""
    customs = _custom_classes(db)
    return list(GOAL_CLASSES.values()) + [customs[k] for k in sorted(customs)]


def get_class_for(db: Any, class_id: str | None) -> GoalClass:
    """The descriptor for ``class_id`` — built-in first, then custom.

    ``None``/empty → ``general``.  Raises :class:`GoalClassError`
    for an id in neither registry, listing the known classes.
    """
    if class_id is None or not str(class_id).strip():
        return GOAL_CLASSES[DEFAULT_CLASS_ID]
    key = str(class_id).strip().lower()
    descriptor = GOAL_CLASSES.get(key)
    if descriptor is not None:
        return descriptor
    descriptor = _custom_classes(db).get(key)
    if descriptor is not None:
        return descriptor
    known = ", ".join(sorted(list(GOAL_CLASSES) + list(_custom_classes(db))))
    raise GoalClassError(f"unknown goal class {class_id!r} (known classes: {known})")


def validate_class_id_for(db: Any, class_id: str | None) -> str:
    """Normalise a class id for storage (built-in or custom); unknown ids raise."""
    return get_class_for(db, class_id).id


def define_custom_class(
    db: Any,
    class_id: str,
    title: str,
    description: str = "",
    *,
    progress_model: str = "manual",
    review_cadence: str = "weekly",
    summarizer_key: str = "",
    tool_hints: tuple[str, ...] | list[str] = (),
    milestone_templates: tuple[str, ...] | list[str] = (),
) -> GoalClass:
    """Define a new goal class as data (persisted in ``goal_classes``).

    Refuses a built-in id (those are code, not data), a malformed id,
    and an empty title.  Re-defining an existing custom id replaces
    its descriptor — a class is data, and data can be corrected.
    """
    key = (class_id or "").strip().lower()
    if not _CUSTOM_ID_RE.match(key):
        raise GoalClassError(
            f"invalid goal class id {class_id!r} — use a lowercase slug "
            "starting with a letter, e.g. 'gardening'")
    if key in GOAL_CLASSES:
        raise GoalClassError(
            f"goal class {key!r} is built in — custom classes cannot "
            "shadow a built-in id")
    title = (title or "").strip()
    if not title:
        raise GoalClassError("a goal class needs a title")
    ensure_goal_classes_table(db)
    db.execute("""
        INSERT INTO goal_classes
            (class_id, title, description, progress_model, review_cadence,
             summarizer_key, tool_hints, milestone_templates, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(class_id) DO UPDATE SET
            title = excluded.title,
            description = excluded.description,
            progress_model = excluded.progress_model,
            review_cadence = excluded.review_cadence,
            summarizer_key = excluded.summarizer_key,
            tool_hints = excluded.tool_hints,
            milestone_templates = excluded.milestone_templates
    """, (key, title, description,
          (progress_model or "manual").strip() or "manual",
          (review_cadence or "weekly").strip() or "weekly",
          (summarizer_key or "").strip(),
          _json.dumps([str(t) for t in tool_hints]),
          _json.dumps([str(t) for t in milestone_templates]),
          time.time()))
    return get_class_for(db, key)


def remove_custom_class(db: Any, class_id: str) -> int:
    """Remove a custom class; returns how many goals were filed under it.

    Refuses built-in ids and unknown ids with :class:`GoalClassError`.
    A class still in use is NOT removed — the caller must re-file or
    delete those goals first (the count returned names them in the
    error); deleting a class out from under live goals would orphan
    their filing.  The goals table may not exist on a bare db — then
    the class is trivially unused.
    """
    key = (class_id or "").strip().lower()
    if key in GOAL_CLASSES:
        raise GoalClassError(
            f"goal class {key!r} is built in — it cannot be removed")
    if key not in _custom_classes(db):
        raise GoalClassError(f"unknown goal class {class_id!r} (not a custom class)")
    in_use = 0
    try:
        row = db.query_one(
            "SELECT COUNT(*) AS n FROM goals WHERE goal_class = ?", (key,))
        in_use = int((row or {}).get("n") or 0)
    except Exception:  # noqa: BLE001 - no goals table yet = unused
        in_use = 0
    if in_use:
        raise GoalClassError(
            f"goal class {key!r} is in use by {in_use} goal(s) — "
            "re-file them first (/goals class <goal> <class>)")
    db.execute("DELETE FROM goal_classes WHERE class_id = ?", (key,))
    return in_use


# ── progress providers ────────────────────────────────────────────────


@dataclass
class ClassProgress:
    """One progress reading for a goal.

    ``percent`` is 0–100, or ``None`` when nothing can be said.
    ``basis`` is ``"measured"`` (computed from a real store by a
    provider) or ``"self-reported"`` (the goal's manual % — used only
    when no provider measures the goal, and labelled so everywhere it
    renders).  ``source`` names where the number came from (the
    provider key, or ``"manual"``).  ``measured_at`` is when the
    reading was taken (epoch seconds).  ``evidence`` carries the
    provider's supporting facts for briefings to show their work.
    """

    percent: float | None
    basis: str  # "measured" | "self-reported"
    source: str
    measured_at: float | None = None
    headline: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "percent": self.percent,
            "basis": self.basis,
            "source": self.source,
            "measured_at": self.measured_at,
            "headline": self.headline,
            "evidence": self.evidence,
        }


#: Provider signature: (db, goal, *, now) -> ClassProgress | None.
ProgressProvider = Callable[..., ClassProgress | None]

#: class id -> provider.  Empty in P1 by design (see module docstring).
_PROGRESS_PROVIDERS: dict[str, ProgressProvider] = {}


def register_progress_provider(class_id: str, provider: ProgressProvider) -> None:
    """Register (or replace) the measured-progress provider for a class.

    The class must exist — a provider for an unknown class is a wiring
    bug and is refused in words.  Providers must be read-only over
    their organ's store.
    """
    descriptor = require_class(class_id)
    if not callable(provider):
        raise GoalClassError(
            f"progress provider for class {descriptor.id!r} is not callable")
    _PROGRESS_PROVIDERS[descriptor.id] = provider


def registered_provider(class_id: str) -> ProgressProvider | None:
    """The provider registered for a class, or None."""
    return _PROGRESS_PROVIDERS.get(validate_class_id(class_id))


def goal_progress(
    db: Database,
    goal: Goal,
    *,
    now: float | None = None,
) -> ClassProgress:
    """The progress reading for one goal — measured when possible.

    The goal's class provider gets first say: a provider reading is
    returned with its basis forced to ``"measured"`` and any missing
    fields filled (percent clamped to 0–100, source defaulted to the
    class's progress-model key).  When no provider is registered or
    the provider returns ``None``, the goal's manual % is returned
    labelled ``basis="self-reported"`` — never dressed up as measured.
    """
    descriptor = get_class_for(db, getattr(goal, "goal_class", None))
    provider = _PROGRESS_PROVIDERS.get(descriptor.id)
    if provider is not None:
        reading = provider(db, goal, now=now)
        if reading is not None:
            percent = reading.percent
            if percent is not None:
                percent = max(0.0, min(100.0, float(percent)))
            return ClassProgress(
                percent=percent,
                basis="measured",
                source=reading.source or descriptor.progress_model,
                measured_at=reading.measured_at
                if reading.measured_at is not None
                else (time.time() if now is None else now),
                headline=reading.headline,
                evidence=dict(reading.evidence or {}),
            )
    return ClassProgress(
        percent=float(getattr(goal, "progress", 0.0) or 0.0),
        basis="self-reported",
        source="manual",
        measured_at=None,
        headline="self-reported",
        evidence={},
    )
