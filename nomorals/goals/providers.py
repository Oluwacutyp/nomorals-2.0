"""Measured-progress providers for goal classes (WS2 P3).

P1 shipped the provider registry empty by design
(:mod:`nomorals.goals.classes`); this module is where the first two
real providers land — **finance** and **trading** — and
:func:`register_providers` wires them into the registry.  The wiring
runs when the goals package loads (see the module-end registration
at the bottom of :mod:`nomorals.goals`, the ``agents/goals.py``
precedent), so every entry point that builds a ``GoalTracker`` —
partner runtime, ``nm goals`` CLI, morning briefing — measures these
classes from their real stores with zero surface changes: the P2
board/progress/brain renders read through
``GoalTracker.class_progress``, which consults the registry.

Both providers are read-only and failure-contained: any store
problem (missing file, corrupt bytes, unreadable home) is logged
with the goal named and answered ``None`` — the goal falls back to
its self-reported % and no board ever breaks on a progress reading.

Linkage — how a tracker goal meets its measuring record
-------------------------------------------------------
The goal record carries no foreign keys into the finance stores, so
the link rides the fields each store already owns:

* finance: the savings store (:mod:`nomorals.finance.goals`) is
  keyed by *name* in every surface that touches it (``/goal add
  <name> …``).  Resolution order: (1) an explicit tag
  ``finance-goal:<savings-goal-id>`` on the tracker goal — the
  strongest key, settable through the ordinary ``tags`` field;
  (2) a unique name match, titles compared normalised (casefolded,
  punctuation collapsed) — deterministic, never fuzzy: an ambiguous
  or absent match measures nothing.  A present-but-stale explicit
  tag does NOT fall through to name matching: a deliberate link that
  no longer resolves is a broken link, not a licence to guess.
* trading: the desk journal records outcomes, not targets, so the
  goal itself declares its target P&L in a tag
  ``pnl-target:<amount>`` (account currency, must be > 0).  No tag
  (or an unparsable/non-positive one) = no measurable target =
  ``None``.

The finance directory is resolved through
:func:`nomorals.finance.budgets.finance_paths` over the process
settings — the same resolution ``finance/commands.py`` uses for the
ledger behind ``/goal`` — so provider and surfaces read the same
files (in a default-home deployment these are exactly the files the
bare ``GoalStore()`` construction in the ``/goal`` handler uses too).
The tracker ``db`` argument is unused by both providers: the finance
and trading stores are file stores (JSON / JSONL) beside that
directory, not tables in the context database.

Measures
--------
* finance (``ledger_funded``): funded % of the linked savings goal,
  computed by the savings store's own :func:`goal_progress` — ledger
  income/spend in the goal's category since the goal was created,
  divided by its target.  The provider adds nothing to the math; it
  only links and labels.
* trading (``trading_journal``): cumulative *realized* P&L from the
  desk journal's close records (``paper_close`` / ``live_close``,
  the same records ``TradingDesk``'s own analytics sum), counting
  closes at or after the goal's creation, divided by the goal's
  declared target.  Realized closes only — open-position marks need
  a live connector snapshot and would make a stored reading depend
  on when you looked; the decision journal
  (:mod:`nomorals.finance.trading_journal`) records reads and
  judgments, not money outcomes, so it is deliberately not the
  measure.  GoldenSwing bundle evidence (WS1 P7 state/harvest
  bundles imported as artifacts) is NOT read here: a bundle's
  manifest carries member hashes, not desk stats, and parsing
  STATE.md prose into a progress number would manufacture
  precision.  That reader is WS2 P7's scope, where the census put
  it; this provider ships complete over Devon's own journal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.config import get_settings
from ..core.logging_setup import get_logger
from ..finance.budgets import finance_paths
from ..finance.goals import GoalStore
from ..finance.goals import goal_progress as _savings_progress
from ..finance.ledger import Ledger
from .classes import ClassProgress, register_progress_provider

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a cycle
    from ..finance.goals import Goal as SavingsGoal
    from . import Goal

_log = get_logger(__name__)

__all__ = [
    "FINANCE_GOAL_TAG",
    "PNL_TARGET_TAG",
    "finance_progress",
    "register_providers",
    "trading_progress",
]

#: Explicit finance link: a tracker-goal tag ``finance-goal:<id>``
#: naming a savings-store goal id (the ``goal_<hex>`` ids
#: :func:`nomorals.finance.goals.create_goal` mints).
FINANCE_GOAL_TAG = "finance-goal:"

#: Trading target declaration: a tracker-goal tag
#: ``pnl-target:<amount>`` — realized P&L (account currency) the
#: goal is aiming at, e.g. ``pnl-target:1000``.
PNL_TARGET_TAG = "pnl-target:"

#: The desk's execution journal, beside the desk state file —
#: the filename :mod:`nomorals.finance.trading_desk` writes.
DESK_JOURNAL_FILENAME = "trading_journal.jsonl"

#: Journal events that close a trade and carry realized ``pnl`` —
#: the same pair the desk's own analytics (_closed_trades,
#: daily_pnl, streaks) sum over.
_CLOSE_EVENTS = ("paper_close", "live_close")


def _finance_dir() -> Path:
    """The finance store directory (ledger, savings goals, desk).

    Resolved via ``finance_paths`` over the process settings —
    the resolution the finance surfaces themselves use.
    """
    _ledger_path, budgets_path = finance_paths(get_settings())
    return Path(budgets_path).parent


# ── finance: funded % from the savings ledger ─────────────────────────


def _normalise_name(text: Any) -> str:
    """Deterministic name key: casefold, non-alnum → space, collapse."""
    folded = re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold())
    return " ".join(folded.split())


def _explicit_finance_link(goal: Goal) -> str | None:
    """The savings-goal id from a ``finance-goal:<id>`` tag, if any."""
    for tag in getattr(goal, "tags", None) or []:
        text = str(tag).strip()
        if text.startswith(FINANCE_GOAL_TAG):
            return text[len(FINANCE_GOAL_TAG):].strip() or None
    return None


def _linked_savings_goal(goal: Goal, store: GoalStore) -> SavingsGoal | None:
    """Resolve the savings goal that measures this tracker goal.

    Explicit tag first (a stale tag resolves to None — never a
    fall-through to guessing); otherwise a unique normalised-name
    match against the savings store (done goals included — a funded
    savings goal is a completed measurement, not an absence).
    """
    linked_id = _explicit_finance_link(goal)
    if linked_id is not None:
        return store.get(linked_id)
    wanted = _normalise_name(getattr(goal, "title", ""))
    if not wanted:
        return None
    matches = [g for g in store.list(include_done=True)
               if _normalise_name(g.name) == wanted]
    if len(matches) == 1:
        return matches[0]
    return None


def _finance_progress(goal: Goal, *, now: float | None) -> ClassProgress | None:
    fin_dir = _finance_dir()
    store = GoalStore(fin_dir / "goals.json")
    linked = _linked_savings_goal(goal, store)
    if linked is None:
        return None
    ledger = Ledger(fin_dir / "ledger.jsonl")
    reading = _savings_progress(linked, ledger, now=now)
    pace = ""
    if reading.get("on_track") is True:
        pace = " · on pace"
    elif reading.get("on_track") is False:
        pace = " · off pace"
    headline = (
        f"{reading['contributed']} of {reading['target']} toward "
        f"{linked.name} ({linked.category}){pace}")
    evidence: dict[str, Any] = {
        "finance_goal_id": linked.id,
        "finance_goal_name": linked.name,
        "link": "tag" if _explicit_finance_link(goal) else "name",
        "category": linked.category,
        "goal_type": linked.goal_type,
        "contributed_kobo": reading["contributed_kobo"],
        "target_kobo": linked.target_kobo,
        "remaining_kobo": reading["remaining_kobo"],
        "on_track": reading.get("on_track"),
        "monthly_need_kobo": reading.get("monthly_need_kobo"),
        "streak_months": reading.get("streak_months"),
    }
    return ClassProgress(
        percent=float(reading["pct"]) * 100.0,
        basis="measured",
        source="savings ledger",
        headline=headline,
        evidence=evidence,
    )


def finance_progress(
    db: Any, goal: Goal, *, now: float | None = None,
) -> ClassProgress | None:
    """Finance-class provider: funded % from the savings ledger.

    ``None`` when no savings goal links to this goal (no explicit
    tag, no unique name match) or when the finance stores cannot
    be read — the failure is logged with the goal named, never
    raised into a board.
    """
    try:
        return _finance_progress(goal, now=now)
    except Exception as exc:  # noqa: BLE001 - contained: see docstring
        _log.warning(
            "finance progress provider failed for goal %s (%r): %s",
            getattr(goal, "goal_id", "?"), getattr(goal, "title", ""), exc)
        return None


# ── trading: realized P&L from the desk journal ───────────────────────


def _pnl_target(goal: Goal) -> float | None:
    """The goal's declared target P&L from a ``pnl-target:`` tag.

    A present but unparsable or non-positive tag declares nothing
    measurable — it returns None rather than scanning on for a
    second, weaker signal.
    """
    for tag in getattr(goal, "tags", None) or []:
        text = str(tag).strip()
        if text.startswith(PNL_TARGET_TAG):
            raw = text[len(PNL_TARGET_TAG):].strip()
            try:
                value = float(raw)
            except ValueError:
                return None
            return value if value > 0 else None
    return None


def _realized_pnl(
    journal_path: Path, *, since: float,
) -> tuple[float, int, int, int]:
    """Sum realized P&L over desk-journal close records.

    Returns ``(realized, closes, unreadable_lines, bad_pnl_records)``.
    Parsing mirrors the desk's own journal reader: unparseable lines
    are skipped (and counted here, so the reading can name them in
    its evidence instead of pretending the file was clean).
    """
    if not journal_path.exists():
        return 0.0, 0, 0, 0
    realized = 0.0
    closes = 0
    unreadable = 0
    bad_pnl = 0
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            unreadable += 1
            continue
        if not isinstance(record, dict):
            unreadable += 1
            continue
        if record.get("event") not in _CLOSE_EVENTS:
            continue
        try:
            ts = float(record.get("ts") or 0)
        except (TypeError, ValueError):
            unreadable += 1
            continue
        if ts < since:
            continue
        try:
            realized += float(record.get("pnl") or 0)
        except (TypeError, ValueError):
            bad_pnl += 1
            continue
        closes += 1
    return realized, closes, unreadable, bad_pnl


def _trading_progress(goal: Goal, *, now: float | None) -> ClassProgress | None:
    target = _pnl_target(goal)
    if target is None:
        return None
    journal_path = _finance_dir() / DESK_JOURNAL_FILENAME
    since = float(getattr(goal, "created_at", 0.0) or 0.0)
    realized, closes, unreadable, bad_pnl = _realized_pnl(
        journal_path, since=since)
    percent = realized / target * 100.0
    if closes:
        headline = (
            f"realized {realized:+.2f} of {target:.2f} target across "
            f"{closes} closed trade(s) since this goal was filed")
    else:
        headline = (
            f"no closed trades in the desk journal since this goal "
            f"was filed — realized 0.00 of {target:.2f} target")
    evidence: dict[str, Any] = {
        "realized_pnl": round(realized, 2),
        "target_pnl": target,
        "closed_trades": closes,
        "since": since,
        "journal": DESK_JOURNAL_FILENAME,
    }
    if unreadable:
        evidence["unreadable_journal_lines"] = unreadable
    if bad_pnl:
        evidence["unreadable_pnl_records"] = bad_pnl
    return ClassProgress(
        percent=percent,
        basis="measured",
        source="trading journal",
        headline=headline,
        evidence=evidence,
    )


def trading_progress(
    db: Any, goal: Goal, *, now: float | None = None,
) -> ClassProgress | None:
    """Trading-class provider: realized P&L vs the goal's target.

    ``None`` when the goal declares no measurable target (no valid
    ``pnl-target:`` tag) or when the desk journal cannot be read —
    failures are logged with the goal named, never raised.
    """
    try:
        return _trading_progress(goal, now=now)
    except Exception as exc:  # noqa: BLE001 - contained: see docstring
        _log.warning(
            "trading progress provider failed for goal %s (%r): %s",
            getattr(goal, "goal_id", "?"), getattr(goal, "title", ""), exc)
        return None


# ── registration ──────────────────────────────────────────────────────


def register_providers() -> None:
    """Register the finance + trading providers with the registry.

    Idempotent (re-registration replaces).  Called by
    :mod:`nomorals.goals` when the package loads — the one wiring
    point every GoalTracker entry point shares.
    """
    register_progress_provider("finance", finance_progress)
    register_progress_provider("trading", trading_progress)
