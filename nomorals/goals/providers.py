"""Measured-progress providers for goal classes (WS2 P3–P6).

P1 shipped the provider registry empty by design
(:mod:`nomorals.goals.classes`); this module is where the real
providers land — **finance** and **trading** (P3), **careers**
(P4), **productivity** and **interests** (P5), **health**,
**learning**, **build** and **relationships** (P6) — and
:func:`register_providers` wires them into the registry.  The wiring
runs when the goals package loads (see the module-end registration
at the bottom of :mod:`nomorals.goals`, the ``agents/goals.py``
precedent), so every entry point that builds a ``GoalTracker`` —
partner runtime, ``nm goals`` CLI, morning briefing — measures these
classes from their real stores with zero surface changes: the P2
board/progress/brain renders read through
``GoalTracker.class_progress``, which consults the registry.

The providers are read-only and failure-contained: any store
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
* careers: the pipeline store (:mod:`nomorals.careers.store`)
  holds applications, not goal targets, so the link rides tags
  again: an explicit ``career-app:<application-id>`` naming ONE
  application whose stage is the reading, else a declared
  ``career-target:<n>`` — submit ``n`` applications (counted from
  the pipeline's own stage histories, filed at/after the goal).
  A present-but-stale ``career-app:`` tag measures nothing (the
  P3 rule: a broken deliberate link is not a licence to guess).
* productivity: the class organs are the owner's own ledgers on
  the home's ``data/`` directory — the commitments ledger
  (:mod:`nomorals.commitments.ledger`, the file the partner
  runtime's ``/commitments`` surface writes) and the focus timer
  (:mod:`nomorals.focus.sessions`, the ``/focus`` file).  A goal
  declaring ``focus-target:<minutes>`` measures focus; any other
  productivity goal measures commitment reliability.  No tag links
  a goal to individual commitments — the ledger is the owner's
  whole obligation layer, and the goal's window (its own creation)
  is the scope.
* interests: an interests goal IS a watch.  Its scope is its
  topics — ``watch:<topic>`` tags when present, else the goal's
  title — and its store is the news organ's tables (``news_items``
  / ``news_interest``) in the context database itself: the tracker
  ``db`` the provider receives is the same database the NewsAgent
  is constructed over (``runtime_live`` builds ``NewsAgent(context)``
  and the runtime builds ``GoalTracker(context.db)``).  The goals
  layer may not import ``agents.news`` (L5 sits above goals' L4),
  so the profile/item reads here mirror the NewsAgent's own access
  pattern over the storage seam, and the seed-beat fallback is a
  local mirror of ``INTEREST_SEED_BEATS`` — pinned against the real
  constant by the P5 test suite, so the mirror cannot drift
  silently.  Creating an interests goal registers its topics as
  beats in that profile (see :func:`register_interest_beats`) —
  the census's settled behaviour change: a watch that watches
  nothing is a label, not a class.
* health: the health organ is L5 (it imports agents), above
  goals' L4, so — the interests pattern — the provider reads its
  store directly: the ``health_events`` table of the timeline
  database (:mod:`nomorals.health.timeline`) at the settings home,
  opened read-only via sqlite3.  The timeline is the owner's one
  health record, so there is no per-goal link: the goal's window
  (its own creation) is the scope, exactly the productivity
  stance.  The table/column names and event vocabulary here are
  a mirror of the timeline's DDL, pinned against the real store
  by the P6 suite seeding through ``HealthTimeline`` itself.
* learning: the tutor's persisted session files
  (``<home>/learn/sessions/*.json``, the files
  :meth:`nomorals.learn.tutor.TutorSession.save` writes).  Scope
  is a ``learn:<topic>`` tag, else the goal's title matching a
  session topic exactly after normalisation (the P3 finance
  name-match rule — deterministic, never fuzzy).  Each file's
  ``mastery`` payload is read with ``MasteryModel.bkt()``
  semantics: a skill's P(knows) is its ``bkt`` entry, or the BKT
  prior for a skill the session named but never observed.  The
  prior is a local mirror of ``BKT_DEFAULTS["p_init"]``, pinned
  against the real constant by the P6 suite.
* build: execution records the tracker ecosystem itself keeps.
  An explicit ``agent-goal:<id>`` tag links the goal to one of
  the brain's agent goals in the SAME context database (the
  tracker and ``agents/goals.py`` share ``context.db``) and
  measures its step table; a present-but-stale tag measures
  nothing.  Otherwise the goal's own ``subgoals`` rows in the
  tracker store are the steps.  No other link exists on a tracker
  goal today — there is no mission foreign key to read, and none
  is invented here.
* relationships: the commitments ledger again, read from the
  other side — owner promises TO one person.  Scope is a
  ``person:<name>`` tag, else the goal's title matching a
  promisee exactly after normalisation (unique match only, the
  finance rule).  The people index/memory is NOT a measure
  source: no kept-promise records live there; the ledger is the
  only record of promises and their verdicts.

The finance directory is resolved through
:func:`nomorals.finance.budgets.finance_paths` over the process
settings — the same resolution ``finance/commands.py`` uses for the
ledger behind ``/goal`` — so provider and surfaces read the same
files (in a default-home deployment these are exactly the files the
bare ``GoalStore()`` construction in the ``/goal`` handler uses too).
The tracker ``db`` argument is unused by the finance, trading,
careers and productivity providers: their stores are file stores
(JSON / JSONL) beside that directory, not tables in the context
database.  The interests provider is the exception — its store IS
the context database (above).

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
* careers (``application_pipeline``): for a ``career-app:``-linked
  goal, the linked application's stage weight
  (:data:`nomorals.careers.store.STAGE_PROGRESS` — saved 10,
  applied 30, screen 50, interview 70, offer 90, accepted 100;
  a rejected/withdrawn link reads 0 with the closure named in the
  headline — it ended, and it did not succeed).  For a
  ``career-target:<n>`` goal, applications submitted since the
  goal was filed (stage history reached ``applied``) divided by
  ``n``.  Both are computed from the pipeline store's own records
  — the append-only stage histories, never a self-report.
* productivity (``commitment_reliability``): kept ÷ due over the
  goal's window.  *Due* = an owner commitment whose deadline fell
  in [goal creation, now]; *kept* is the ledger's own ``kept``
  status.  An open commitment whose deadline has passed is due
  and not kept, so it sits in the denominator (broken down in the
  evidence); cancelled and superseded commitments are NOT misses
  (the ledger's own semantics — a cancelled promise was withdrawn
  by agreement, a superseded one was replaced) and are counted in
  the evidence as exclusions, alongside no-deadline, not-yet-due,
  pre-window and other-promisor rows.  Nothing due in the window
  measures nothing (``None``) — a 0/0 rate would be a fabricated
  number.  For a ``focus-target:<minutes>`` goal the measure is
  instead completed focus minutes in the window (``done`` sessions
  started at/after the goal; in-progress sessions never count —
  a stored reading must not depend on when you looked, the P3
  trading rule) divided by the declared target.
* interests (``watch_coverage``): findings on the goal's watch
  that REACHED the owner, from the ``news_items`` delivery state
  machine (``pushed`` / ``digested`` count as delivered; legacy
  NULL-state rows count as delivered — storage WAS delivery
  before the state machine existed, the store's own reading;
  ``undelivered`` / ``archived`` do not).  An item matches the
  watch when every content word of a topic appears in the item's
  title/summary/source/publisher text (deterministic whole-word
  matching, the P3 normalisation family — never the brain's
  per-profile score, which is not attributable to one watch).
  With a ``watch-target:<n>`` tag the reading is delivered ÷ n;
  without one it is delivered ÷ flowed, and a watch with scope
  but no flow yet reads a measured 0% with that headline — "no
  scope" (no usable topic) is the only ``None``.  Rows whose
  timestamps cannot be read are counted in the evidence, never
  silently absorbed into the flow.
* health (``health_adherence``): check-in adherence — the days
  in the goal's window carrying at least one timeline event,
  divided by the days in the window (the filing day through the
  reading day, inclusive).  The timeline records what the owner
  actually logged (symptoms, sleep, mood, measurements,
  medications, visits, notes) — a day with an entry is a day the
  health record was kept up; a day without one was not.  The
  challenges streak table cannot carry this measure (it stores
  only each member's last check-in, no day history) and the
  training store's completed/incomplete workout logs are one
  plan's record, not the class's — the timeline is the
  class-wide check-in record.  A window with no events reads a
  measured 0% saying so (the organ has simply not been fed);
  only an unreadable store is ``None``.
* learning (``mastery``): mean BKT P(knows) over the scoped
  topic's skills, each skill read from the most recent session
  that covered it (mastery is a state, like the careers linked
  application's stage — the newest record of the state is the
  current state).  Sessions are not windowed by the goal's
  creation: knowledge gained before the goal was filed is still
  knowledge, and the evidence discloses how many sessions
  predate the goal.  A declared topic with no sessions yet reads
  a measured 0% — no demonstrated mastery — with that headline.
  The FSRS mistake notebook is deliberately NOT read: it is one
  global deck with no topic attribution, so its due/cleared
  counts cannot honestly measure one goal's topic.
* build (``execution_steps``): completed steps ÷ total steps —
  the linked agent goal's step rows (``done`` over all steps
  except ``skipped``: a skipped step was deliberately dropped
  from the plan, the cancelled-commitment rule; skipped counts
  land in the evidence), or the tracker goal's own subgoal rows
  (``is_completed`` ÷ total).  A goal with no steps recorded
  anywhere measures nothing (``None``) — there is no execution
  record to read, and the manual % would be a guess dressed as
  a measurement if the provider invented one.
* relationships (``person_commitments``): kept ÷ due for owner
  promises to the scoped person over the goal's window — the
  productivity counting loop with a promisee filter: same due
  semantics (deadline in [goal creation, now]; open past
  deadline is due and not kept), same exclusions (cancelled /
  superseded are not misses), plus owner promises to OTHER
  people counted as their own exclusion class.  Nothing due to
  that person measures nothing (``None``) — a 0/0 rate would be
  fabricated, the productivity rule.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..careers.store import STAGE_PROGRESS, CareerStore, careers_path
from ..commitments.ledger import CommitmentLedger
from ..core.config import get_settings
from ..core.logging_setup import get_logger
from ..finance.budgets import finance_paths
from ..finance.goals import GoalStore
from ..finance.goals import goal_progress as _savings_progress
from ..finance.ledger import Ledger
from ..focus.sessions import FocusTimer
from ..storage.rowstore import attached_rows
from .classes import ClassProgress, register_progress_provider

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a cycle
    from ..finance.goals import Goal as SavingsGoal
    from . import Goal

_log = get_logger(__name__)

__all__ = [
    "AGENT_GOAL_TAG",
    "CAREER_APP_TAG",
    "CAREER_TARGET_TAG",
    "FINANCE_GOAL_TAG",
    "FOCUS_TARGET_TAG",
    "LEARN_TAG",
    "PERSON_TAG",
    "PNL_TARGET_TAG",
    "WATCH_TAG",
    "WATCH_TARGET_TAG",
    "build_progress",
    "careers_progress",
    "finance_progress",
    "health_progress",
    "interest_topics",
    "interests_progress",
    "learning_progress",
    "productivity_progress",
    "register_interest_beats",
    "register_providers",
    "relationships_progress",
    "remove_interest_beats",
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


# ── careers: stage funnel from the pipeline store ─────────────────────

#: Explicit careers link: a tracker-goal tag
#: ``career-app:<application-id>`` naming one application in the
#: pipeline store (the ``app_<hex>`` ids
#: :class:`nomorals.careers.store.CareerStore` mints).  The linked
#: application's stage IS the goal's measured progress.
CAREER_APP_TAG = "career-app:"

#: Careers target declaration: a tracker-goal tag
#: ``career-target:<n>`` — submit ``n`` applications (counted from
#: stage histories, filed at/after the goal's creation).
CAREER_TARGET_TAG = "career-target:"


def _career_store() -> CareerStore:
    """The pipeline store, resolved via the process settings —
    the resolution the /careers surface itself uses."""
    return CareerStore(careers_path(get_settings()))


def _tag_value(goal: Goal, prefix: str) -> str | None:
    """The value of the goal's first ``prefix`` tag, if any."""
    for tag in getattr(goal, "tags", None) or []:
        text = str(tag).strip()
        if text.startswith(prefix):
            return text[len(prefix):].strip() or None
    return None


def _careers_progress(goal: Goal, *, now: float | None) -> ClassProgress | None:
    store = _career_store()
    linked_id = _tag_value(goal, CAREER_APP_TAG)
    if linked_id is not None:
        app = store.get(linked_id)
        if app is None:
            return None  # stale deliberate link: never fall through
        percent = STAGE_PROGRESS[app.stage]
        if app.stage in ("rejected", "withdrawn"):
            headline = (
                f"{app.company} — {app.role} closed ({app.stage}): "
                "the linked application ended without an offer")
        elif app.stage == "accepted":
            headline = f"{app.company} — {app.role}: offer accepted"
        else:
            headline = (
                f"{app.company} — {app.role} is at the {app.stage} "
                f"stage ({percent:.0f}% of the pipeline)")
        evidence: dict[str, Any] = {
            "application_id": app.id,
            "link": "career-app",
            "company": app.company,
            "role": app.role,
            "stage": app.stage,
            "stage_progress": percent,
            "stages_visited": [h.get("stage") for h in app.history],
        }
        if app.next_action:
            evidence["next_action"] = app.next_action
            evidence["next_action_due"] = app.next_action_due
        return ClassProgress(
            percent=percent,
            basis="measured",
            source="application pipeline",
            headline=headline,
            evidence=evidence,
        )
    raw_target = _tag_value(goal, CAREER_TARGET_TAG)
    if raw_target is None:
        return None
    try:
        target = int(raw_target)
    except ValueError:
        return None
    if target <= 0:
        return None
    since = float(getattr(goal, "created_at", 0.0) or 0.0)
    submitted = store.submitted_since(since)
    percent = submitted / target * 100.0
    headline = (
        f"{submitted} of {target} application(s) submitted since "
        "this goal was filed")
    return ClassProgress(
        percent=percent,
        basis="measured",
        source="application pipeline",
        headline=headline,
        evidence={
            "link": "career-target",
            "submitted": submitted,
            "target_applications": target,
            "since": since,
            "funnel": store.funnel_counts(),
        },
    )


def careers_progress(
    db: Any, goal: Goal, *, now: float | None = None,
) -> ClassProgress | None:
    """Careers-class provider: the application pipeline, measured.

    ``None`` when the goal declares no pipeline link (no
    ``career-app:`` / valid ``career-target:`` tag), when a
    deliberate link is stale, or when the pipeline store cannot
    be read — the failure is logged with the goal named, never
    raised into a board.
    """
    try:
        return _careers_progress(goal, now=now)
    except Exception as exc:  # noqa: BLE001 - contained: see docstring
        _log.warning(
            "careers progress provider failed for goal %s (%r): %s",
            getattr(goal, "goal_id", "?"), getattr(goal, "title", ""), exc)
        return None


def _tag_values(goal: Goal, prefix: str) -> list[str]:
    """The values of EVERY ``prefix`` tag on the goal, in tag order."""
    out: list[str] = []
    for tag in getattr(goal, "tags", None) or []:
        text = str(tag).strip()
        if text.startswith(prefix):
            value = text[len(prefix):].strip()
            if value:
                out.append(value)
    return out


# ── productivity: commitment reliability + focus minutes ──────────────

#: Focus target declaration: a tracker-goal tag
#: ``focus-target:<minutes>`` — completed focus minutes the goal is
#: aiming at, e.g. ``focus-target:600`` (ten hours of deep work).
#: A goal carrying this tag measures focus; any other productivity
#: goal measures commitment reliability.
FOCUS_TARGET_TAG = "focus-target:"

#: The class-organ files on the home's ``data/`` directory — the
#: exact paths the partner runtime's own surfaces use
#: (``runtime_reach._commitments_ledger`` / ``._focus_timer``).
_COMMITMENTS_FILENAME = "commitments.json"
_FOCUS_FILENAME = "focus.json"


def _data_file(filename: str) -> Path:
    """A file on the home's ``data/`` directory, resolved via the
    process settings — the ``careers_path`` pattern."""
    settings = get_settings()
    home = getattr(settings, "home_path", None)
    if home is not None:
        return Path(home) / "data" / filename
    return Path.home() / ".nomorals" / "data" / filename


def _raw_store_rows(path: Path, key: str) -> list[Any] | None:
    """The raw ``key`` list from a JSON store file, or None.

    None means the file cannot be read as JSON at all (missing is
    NOT None — a missing file reads as an empty list: the organ
    simply has not run yet).  A present-but-unparseable file raises
    ValueError: silently treating corruption as emptiness would
    let a broken store masquerade as an empty one.
    """
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: top level is not an object")
    rows = raw.get(key)
    return list(rows) if isinstance(rows, list) else []


def _focus_progress(goal: Goal, *, raw_target: str,
                    now: float | None) -> ClassProgress | None:
    """The focus measure: completed minutes vs a declared target."""
    try:
        target = float(raw_target)
    except ValueError:
        return None
    if target <= 0:
        return None
    path = _data_file(_FOCUS_FILENAME)
    raw_rows = _raw_store_rows(path, "sessions")
    timer = FocusTimer(str(path))
    since = float(getattr(goal, "created_at", 0.0) or 0.0)
    seconds = 0.0
    completed = 0
    predating = 0
    in_progress = 0
    for session in timer.sessions.values():
        if float(session.started_at or 0.0) < since:
            if session.status == "done":
                predating += 1
            continue
        if session.status == "done":
            completed += 1
            seconds += session.elapsed()
        else:
            in_progress += 1
    minutes = seconds / 60.0
    percent = minutes / target * 100.0
    if completed:
        headline = (
            f"focused {minutes:.0f} of {target:.0f} target minutes "
            f"across {completed} completed session(s) since this "
            "goal was filed")
    else:
        headline = (
            f"no completed focus sessions since this goal was "
            f"filed — 0 of {target:.0f} target minutes")
    evidence: dict[str, Any] = {
        "measure": "focus_minutes",
        "link": "focus-target",
        "focus_minutes": round(minutes, 1),
        "target_minutes": target,
        "completed_sessions": completed,
        "since": since,
    }
    unreadable = len(raw_rows or []) - len(timer.sessions)
    if unreadable > 0:
        evidence["unreadable_session_rows"] = unreadable
    if predating:
        evidence["sessions_predating_goal"] = predating
    if in_progress:
        evidence["sessions_in_progress"] = in_progress
    return ClassProgress(
        percent=percent,
        basis="measured",
        source="focus sessions",
        headline=headline,
        evidence=evidence,
    )


def _commitment_counts(
    ledger: CommitmentLedger, *, since: float, until: float,
    promisee_norm: str | None = None,
) -> tuple[int, int, int, dict[str, int], int]:
    """Kept/broken/overdue counts + exclusions over a window.

    The one counting loop behind the productivity (whole-ledger)
    and relationships (per-person) measures — the ledger's own
    semantics, shared: cancelled and superseded are not misses,
    and an open commitment past its deadline is due and not kept.
    With ``promisee_norm`` set, only owner promises TO that person
    (normalised) are in scope; owner promises to anyone else are
    counted as the ``other_promisee`` exclusion.  Returns
    ``(kept, broken, overdue_open, excluded, unreadable_rows)``.
    """
    kept = broken = overdue_open = 0
    unreadable_rows = 0
    excluded = {
        "cancelled": 0, "superseded": 0, "no_deadline": 0,
        "due_before_window": 0, "not_yet_due": 0, "other_promisor": 0,
    }
    if promisee_norm is not None:
        excluded["other_promisee"] = 0
    for commitment in ledger.commitments.values():
        if str(commitment.promisor or "owner") != "owner":
            excluded["other_promisor"] += 1
            continue
        if (promisee_norm is not None
                and _normalise_name(commitment.promisee) != promisee_norm):
            excluded["other_promisee"] += 1
            continue
        if commitment.status == "cancelled":
            excluded["cancelled"] += 1
            continue
        if commitment.status == "superseded":
            excluded["superseded"] += 1
            continue
        if commitment.deadline is None:
            excluded["no_deadline"] += 1
            continue
        try:
            deadline = float(commitment.deadline)
        except (TypeError, ValueError):
            unreadable_rows += 1
            continue
        if deadline < since:
            excluded["due_before_window"] += 1
            continue
        if deadline > until:
            excluded["not_yet_due"] += 1
            continue
        # Due in the window: the ledger's own verdicts, plus the
        # still-open promises whose deadline has passed (due, not
        # kept).  Promises closed early (kept/broken before their
        # deadline) are verdicts about a due commitment all the
        # same — status wins over the deadline arithmetic.
        if commitment.status == "kept":
            kept += 1
        elif commitment.status == "broken":
            broken += 1
        elif commitment.status == "open":
            overdue_open += 1
    return kept, broken, overdue_open, excluded, unreadable_rows


def _reliability_progress(goal: Goal, *,
                          now: float | None) -> ClassProgress | None:
    """The reliability measure: kept ÷ due over the goal's window."""
    path = _data_file(_COMMITMENTS_FILENAME)
    raw_rows = _raw_store_rows(path, "entries")
    ledger = CommitmentLedger(str(path))
    since = float(getattr(goal, "created_at", 0.0) or 0.0)
    until = time.time() if now is None else float(now)
    kept, broken, overdue_open, excluded, unreadable_rows = \
        _commitment_counts(ledger, since=since, until=until)
    due = kept + broken + overdue_open
    if due == 0:
        return None
    percent = kept / due * 100.0
    headline = (
        f"kept {kept} of {due} commitment(s) due since this goal "
        "was filed")
    misses = []
    if broken:
        misses.append(f"{broken} broken")
    if overdue_open:
        misses.append(f"{overdue_open} overdue")
    if misses:
        headline += " (" + ", ".join(misses) + ")"
    evidence: dict[str, Any] = {
        "measure": "commitment_reliability",
        "kept": kept,
        "broken": broken,
        "overdue_open": overdue_open,
        "due": due,
        "since": since,
        "excluded": {k: v for k, v in excluded.items() if v},
    }
    unreadable = len(raw_rows or []) - len(ledger.entries)
    if unreadable > 0:
        evidence["unreadable_ledger_entries"] = unreadable
    if unreadable_rows:
        evidence["unreadable_commitment_rows"] = unreadable_rows
    return ClassProgress(
        percent=percent,
        basis="measured",
        source="commitment ledger",
        headline=headline,
        evidence=evidence,
    )


def _productivity_progress(goal: Goal, *,
                           now: float | None) -> ClassProgress | None:
    raw_target = _tag_value(goal, FOCUS_TARGET_TAG)
    if raw_target is not None:
        return _focus_progress(goal, raw_target=raw_target, now=now)
    return _reliability_progress(goal, now=now)


def productivity_progress(
    db: Any, goal: Goal, *, now: float | None = None,
) -> ClassProgress | None:
    """Productivity-class provider: kept-rate / focus, measured.

    ``None`` when the goal declares a ``focus-target:`` that does
    not parse, when nothing has come due in the goal's window, or
    when the class stores cannot be read — failures are logged
    with the goal named, never raised into a board.
    """
    try:
        return _productivity_progress(goal, now=now)
    except Exception as exc:  # noqa: BLE001 - contained: see docstring
        _log.warning(
            "productivity progress provider failed for goal %s (%r): %s",
            getattr(goal, "goal_id", "?"), getattr(goal, "title", ""), exc)
        return None


# ── interests: watch coverage from the news store ─────────────────────

#: Watch scope declaration: a tracker-goal tag ``watch:<topic>``
#: naming one topic the goal watches.  Several may ride one goal;
#: with none, the goal's title is its topic.
WATCH_TAG = "watch:"

#: Watch target declaration: a tracker-goal tag
#: ``watch-target:<n>`` — findings on this watch to be delivered,
#: e.g. ``watch-target:10``.  Present-but-invalid declares nothing
#: measurable (the P3 rule), it does not fall through.
WATCH_TARGET_TAG = "watch-target:"

#: Mirror of ``agents.news.INTEREST_SEED_BEATS`` — the fallback
#: beats when no profile was ever saved.  agents/ is L5, above
#: goals' L4, so the constant cannot be imported here; the P5 test
#: suite pins this mirror against the real tuple so it can never
#: drift silently.
_SEED_BEATS: tuple[tuple[str, float], ...] = (
    ("XAUUSD and gold trading", 1.0),
    ("AI and model work", 0.9),
    ("Nigerian tech and Lagos", 0.9),
    ("Devon herself — the owner's own agent and her systems", 1.0),
    ("USA money opportunities, including loans", 0.8),
)

#: Mirror of ``NewsAgent._ITEM_EXTRA`` (the delivery/attribution
#: columns declared to the rowstore mapping) — same L4/L5 reason.
_ITEM_EXTRA = ("publisher", "publisher_url", "delivery_state",
               "interest", "delivered_at", "delivery_channel")

#: Delivery states that mean the finding REACHED the owner.
#: A NULL state is the legacy world (storage was delivery) and
#: also counts — see the module docstring.
_DELIVERED_STATES = ("pushed", "digested")

#: Whole words that carry no topic signal — scaffolding verbs and
#: connectives.  A topic reduced to only these words has no scope.
_TOPIC_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "in", "into", "of",
    "on", "the", "to", "with", "follow", "following", "track",
    "tracking", "watch", "watching", "keep", "keeping", "my", "me",
    "i", "updates", "update", "news",
})


def interest_topics(goal: Goal) -> list[str]:
    """The goal's watch topics: ``watch:`` tags, else its title.

    Order-stable, deduplicated case-insensitively.  This is the
    scope both the provider measures and beat registration files —
    one definition, two consumers.
    """
    topics = _tag_values(goal, WATCH_TAG)
    if not topics:
        title = str(getattr(goal, "title", "") or "").strip()
        if title:
            topics = [title]
    out: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        topic = str(topic).strip()
        key = topic.casefold()
        if topic and key not in seen:
            seen.add(key)
            out.append(topic)
    return out


def _topic_words(topic: str) -> frozenset[str]:
    """A topic's content words (normalised, stopwords removed)."""
    folded = re.sub(r"[^a-z0-9]+", " ", str(topic or "").casefold())
    return frozenset(w for w in folded.split()
                     if w and w not in _TOPIC_STOPWORDS)


def _briefing_topics() -> list[str]:
    """The owner's stated briefing topics (settings) — the same
    list the NewsAgent folds into its effective profile."""
    settings = get_settings()
    briefing = getattr(settings, "briefing", None)
    return [str(t).strip() for t in
            (getattr(briefing, "topics", None) or []) if str(t).strip()]


def _load_news_profile(db: Any) -> dict[str, Any] | None:
    """The news interest profile {beats, pins}, NewsAgent-shaped.

    Mirrors ``NewsAgent._load_profile``: the saved row wins; absent
    a row, the seeds are the profile.  ``None`` when there is no
    database at all.  A corrupt saved value falls back to the
    seeds, exactly as the NewsAgent falls back.
    """
    if db is None:
        return None
    profile: dict[str, Any] = {
        "beats": [{"topic": t, "weight": w} for t, w in _SEED_BEATS],
        "pins": [],
    }
    raw: Any = None
    profile_t = attached_rows(db, "news_interest")
    if profile_t is not None:
        row = profile_t.get("profile")
        raw = (row or {}).get("value")
    else:
        try:
            row = db.query_one(
                "SELECT value FROM news_interest WHERE key = ?",
                ("profile",))
        except Exception as exc:  # noqa: BLE001 - table may not exist
            if "no such table" not in str(exc).lower():
                raise
            row = None
        raw = (row or {}).get("value")
    if raw:
        try:
            saved = json.loads(raw)
        except (TypeError, ValueError):
            saved = None
        if isinstance(saved, dict):
            beats = saved.get("beats")
            if isinstance(beats, list) and beats:
                profile["beats"] = [
                    {"topic": str(b.get("topic") or ""),
                     "weight": float(b.get("weight") or 0.5)}
                    for b in beats
                    if isinstance(b, dict) and b.get("topic")]
            pins = saved.get("pins")
            if isinstance(pins, list):
                profile["pins"] = [str(p) for p in pins if str(p).strip()]
    return profile


def _save_news_profile(db: Any, profile: dict[str, Any]) -> None:
    """Persist the profile the way ``NewsAgent._save_profile`` does."""
    payload = json.dumps(profile)
    profile_t = attached_rows(db, "news_interest")
    if profile_t is not None:
        profile_t.put({"key": "profile", "value": payload})
        return
    db.execute(
        "CREATE TABLE IF NOT EXISTS news_interest "
        "(key TEXT PRIMARY KEY, value TEXT)")
    db.execute(
        "INSERT OR REPLACE INTO news_interest (key, value) "
        "VALUES (?, ?)", ("profile", payload))


def register_interest_beats(db: Any, goal: Goal) -> list[str]:
    """File an interests goal's topics as news beats.

    The census's settled behaviour change: creating an interests
    goal makes Devon actually watch its topics.  A topic already
    covered — a saved beat, a seed beat, or a stated briefing
    topic — is not duplicated; genuinely new topics are appended
    at weight 1.0 (a stated interest, like the briefing topics).
    Returns the topics NEWLY registered ([] when the watch was
    already covered, or when there is no database / the profile
    cannot be written — registration is best-effort and logged,
    never a reason to fail a goal creation).
    """
    try:
        if db is None:
            return []
        topics = interest_topics(goal)
        if not topics:
            return []
        profile = _load_news_profile(db)
        if profile is None:
            return []
        covered = {str(b.get("topic") or "").casefold()
                   for b in profile["beats"]}
        covered |= {t.casefold() for t in _briefing_topics()}
        added: list[str] = []
        for topic in topics:
            if topic.casefold() not in covered:
                profile["beats"].append({"topic": topic, "weight": 1.0})
                covered.add(topic.casefold())
                added.append(topic)
        if added:
            _save_news_profile(db, profile)
        return added
    except Exception as exc:  # noqa: BLE001 - best-effort, see docstring
        _log.warning(
            "interest beat registration failed for goal %s (%r): %s",
            getattr(goal, "goal_id", "?"), getattr(goal, "title", ""), exc)
        return []


def remove_interest_beats(db: Any, goal: Goal, *,
                          keep_topics: Any = ()) -> list[str]:
    """Un-file an interests goal's beats (the registration reversal).

    Only beats this goal could have registered are removed: a
    topic is kept when it is a seed beat, a stated briefing topic,
    or claimed by another interests goal (``keep_topics`` — the
    caller passes the surviving goals' topics).  Returns the
    topics actually removed.  Best-effort and logged, like
    registration.
    """
    try:
        if db is None:
            return []
        topics = interest_topics(goal)
        if not topics:
            return []
        profile = _load_news_profile(db)
        if profile is None:
            return []
        protected = {t.casefold() for t, _w in _SEED_BEATS}
        protected |= {t.casefold() for t in _briefing_topics()}
        protected |= {str(t).strip().casefold() for t in keep_topics
                      if str(t).strip()}
        wanted = {t.casefold() for t in topics} - protected
        if not wanted:
            return []
        removed: list[str] = []
        beats: list[dict[str, Any]] = []
        for beat in profile["beats"]:
            topic = str(beat.get("topic") or "")
            if topic.casefold() in wanted:
                removed.append(topic)
            else:
                beats.append(beat)
        if removed:
            profile["beats"] = beats
            _save_news_profile(db, profile)
        return removed
    except Exception as exc:  # noqa: BLE001 - best-effort, see docstring
        _log.warning(
            "interest beat removal failed for goal %s (%r): %s",
            getattr(goal, "goal_id", "?"), getattr(goal, "title", ""), exc)
        return []


def _news_items(db: Any) -> list[dict[str, Any]]:
    """Every stored news item row, via the NewsAgent's own access
    pattern (rowstore seam when attached, native SQL otherwise).
    A missing table reads as no items — the organ has not run."""
    items_t = attached_rows(db, "news_items", extra_columns=_ITEM_EXTRA)
    if items_t is not None:
        return [r for r in items_t.all() if isinstance(r, dict)]
    try:
        rows = db.query(
            "SELECT title, summary, source, publisher, delivery_state, "
            "delivered_at, created_at FROM news_items")
    except Exception as exc:  # noqa: BLE001 - table may not exist
        if "no such table" not in str(exc).lower():
            raise
        return []
    return [r for r in (rows or []) if isinstance(r, dict)]


def _interests_progress(db: Any, goal: Goal, *,
                        now: float | None) -> ClassProgress | None:
    topics = interest_topics(goal)
    pairs = [(t, _topic_words(t)) for t in topics]
    pairs = [(t, tw) for t, tw in pairs if tw]
    if not pairs:
        return None  # no usable scope: nothing to measure against
    raw_target = _tag_value(goal, WATCH_TARGET_TAG)
    target: int | None = None
    if raw_target is not None:
        try:
            target = int(raw_target)
        except ValueError:
            return None
        if target <= 0:
            return None
    if db is None:
        return None
    since = float(getattr(goal, "created_at", 0.0) or 0.0)
    until = time.time() if now is None else float(now)
    flowed = 0
    delivered = 0
    unreadable = 0
    matched_topics: set[str] = set()
    for item in _news_items(db):
        try:
            created = float(item.get("created_at") or 0)
        except (TypeError, ValueError):
            unreadable += 1
            continue
        if created < since or created > until:
            continue
        text = " ".join(str(item.get(k) or "") for k in
                        ("title", "summary", "source", "publisher"))
        folded = re.sub(r"[^a-z0-9]+", " ", text.casefold())
        words = frozenset(folded.split())
        hit = [t for t, tw in pairs if tw <= words]
        if not hit:
            continue
        flowed += 1
        matched_topics.update(hit)
        state = item.get("delivery_state")
        if state is None or str(state) in _DELIVERED_STATES:
            delivered += 1
    shown = ", ".join(topics[:3]) + ("…" if len(topics) > 3 else "")
    evidence: dict[str, Any] = {
        "topics": topics,
        "flowed": flowed,
        "delivered": delivered,
        "since": since,
        "match": "all topic words in title/summary/source/publisher",
    }
    if unreadable:
        evidence["unreadable_item_rows"] = unreadable
    if target is not None:
        percent = delivered / target * 100.0
        evidence["link"] = "watch-target"
        evidence["target_findings"] = target
        headline = (
            f"{delivered} of {target} watched findings delivered "
            f"since this goal was filed ({flowed} matched the watch: "
            f"{shown})")
    elif flowed == 0:
        percent = 0.0
        headline = (
            f"no findings on this watch yet — nothing matching "
            f"{shown} has flowed since this goal was filed")
    else:
        percent = delivered / flowed * 100.0
        headline = (
            f"{delivered} of {flowed} findings on this watch "
            f"({shown}) reached you since this goal was filed")
    return ClassProgress(
        percent=percent,
        basis="measured",
        source="news watch",
        headline=headline,
        evidence=evidence,
    )


def interests_progress(
    db: Any, goal: Goal, *, now: float | None = None,
) -> ClassProgress | None:
    """Interests-class provider: watch coverage, measured.

    ``None`` when the goal declares no usable watch scope (no
    ``watch:`` tags and a title with no content words), when a
    ``watch-target:`` tag is present but invalid, when there is
    no database, or when the news store cannot be read — failures
    are logged with the goal named, never raised into a board.
    A scoped watch with no flow yet is NOT None: it reads a
    measured 0% saying so.
    """
    try:
        return _interests_progress(db, goal, now=now)
    except Exception as exc:  # noqa: BLE001 - contained: see docstring
        _log.warning(
            "interests progress provider failed for goal %s (%r): %s",
            getattr(goal, "goal_id", "?"), getattr(goal, "title", ""), exc)
        return None


# ── P6 shared store resolution ────────────────────────────────────────


def _settings_home() -> Path:
    """The settings home when one is in force, else ``~/.nomorals``
    — the ``careers_path`` pattern the P6 file/SQLite stores use."""
    settings = get_settings()
    home = getattr(settings, "home_path", None)
    if home is not None:
        return Path(home)
    return Path.home() / ".nomorals"


# ── health: check-in adherence from the timeline store ────────────────

#: The timeline database under the health home — the file
#: :func:`nomorals.health.timeline.health_db_path` names in a
#: default deployment (``<home>/health/timeline.db``).
_HEALTH_DB_PARTS = ("health", "timeline.db")


def _health_event_rows(
    path: Path,
) -> tuple[list[tuple[str, float]], int, int]:
    """``(event_type, ts)`` rows from the timeline store.

    Returns ``(rows, unreadable)``: rows carry only readable
    timestamps, and ``unreadable`` counts the rows whose ``ts``
    could not be read as a number.  A missing file or a
    missing table reads as no events (the organ has not run);
    a corrupt database raises (contained by the caller).  The
    database is opened read-only: a progress reading must never
    create the owner's health store as a side effect.
    """
    if not path.exists():
        return [], 0
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        try:
            raw = con.execute(
                "SELECT event_type, ts FROM health_events").fetchall()
        except sqlite3.Error as exc:
            if "no such table" in str(exc).lower():
                return [], 0
            raise
    finally:
        con.close()
    rows: list[tuple[str, float]] = []
    unreadable = 0
    for event_type, ts in raw:
        try:
            stamp = float(ts)
        except (TypeError, ValueError):
            unreadable += 1
            continue
        rows.append((str(event_type or ""), stamp))
    return rows, unreadable


def _health_progress(goal: Goal, *, now: float | None) -> ClassProgress:
    path = _settings_home().joinpath(*_HEALTH_DB_PARTS)
    rows, unreadable = _health_event_rows(path)
    since = float(getattr(goal, "created_at", 0.0) or 0.0)
    until = time.time() if now is None else float(now)
    if until < since:
        until = since
    days_in_window = int((until - since) // 86400) + 1
    active_days: set[tuple[int, int, int]] = set()
    by_type: dict[str, int] = {}
    in_window = 0
    pre_window = 0
    after_window = 0
    last_event_at: float | None = None
    for event_type, stamp in rows:
        if stamp < since:
            pre_window += 1
            continue
        if stamp > until:
            after_window += 1
            continue
        in_window += 1
        day = time.localtime(stamp)
        active_days.add((day.tm_year, day.tm_mon, day.tm_mday))
        if event_type:
            by_type[event_type] = by_type.get(event_type, 0) + 1
        if last_event_at is None or stamp > last_event_at:
            last_event_at = stamp
    active = len(active_days)
    percent = active / days_in_window * 100.0
    if in_window:
        headline = (
            f"health logged on {active} of {days_in_window} day(s) "
            f"since this goal was filed ({in_window} entries)")
    else:
        headline = (
            f"nothing logged in the health timeline since this "
            f"goal was filed — 0 of {days_in_window} day(s)")
    evidence: dict[str, Any] = {
        "measure": "checkin_adherence",
        "active_days": active,
        "days_in_window": days_in_window,
        "events_in_window": in_window,
        "events_by_type": by_type,
        "since": since,
    }
    if last_event_at is not None:
        evidence["last_event_at"] = last_event_at
    if pre_window:
        evidence["events_before_window"] = pre_window
    if after_window:
        evidence["events_after_reading"] = after_window
    if unreadable:
        evidence["unreadable_event_rows"] = unreadable
    return ClassProgress(
        percent=percent,
        basis="measured",
        source="health timeline",
        headline=headline,
        evidence=evidence,
    )


def health_progress(
    db: Any, goal: Goal, *, now: float | None = None,
) -> ClassProgress | None:
    """Health-class provider: check-in adherence, measured.

    Days carrying at least one timeline event ÷ days in the
    goal's window.  A window with no events is a measured 0%
    (the organ has not been fed); ``None`` only when the
    timeline store cannot be read — logged with the goal named,
    never raised into a board.
    """
    try:
        return _health_progress(goal, now=now)
    except Exception as exc:  # noqa: BLE001 - contained: see docstring
        _log.warning(
            "health progress provider failed for goal %s (%r): %s",
            getattr(goal, "goal_id", "?"), getattr(goal, "title", ""), exc)
        return None


# ── learning: BKT mastery from the tutor session files ────────────────

#: Learning scope declaration: a tracker-goal tag
#: ``learn:<topic>`` naming the tutor topic the goal is about,
#: e.g. ``learn:Yoruba greetings``.  Without one, the goal's
#: title must match a session topic exactly (normalised).
LEARN_TAG = "learn:"

#: Mirror of ``learn.tutor.BKT_DEFAULTS["p_init"]`` — the P(knows)
#: a skill carries before any observation, which is what
#: ``MasteryModel.bkt()`` returns for a named-but-unobserved
#: skill.  The P6 test suite pins this mirror against the real
#: constant so it cannot drift silently.
_BKT_P_INIT = 0.10


def _learn_session_records(
    sessions_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Readable tutor session records + an unreadable-file count.

    A record is one persisted session file's measuring payload:
    its topic, start time, turn count, and per-skill mastery
    numbers (``skills`` names, ``bkt`` posteriors) as floats.
    Files that are not JSON objects with a mastery object are
    counted unreadable, never absorbed silently.
    """
    records: list[dict[str, Any]] = []
    unreadable = 0
    if not sessions_dir.is_dir():
        return records, unreadable
    for path in sorted(sessions_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable += 1
            continue
        if not isinstance(data, dict):
            unreadable += 1
            continue
        mastery = data.get("mastery")
        if not isinstance(mastery, dict):
            unreadable += 1
            continue
        skills_raw = mastery.get("skills")
        bkt_raw = mastery.get("bkt")
        if not isinstance(skills_raw, dict) and not isinstance(bkt_raw, dict):
            unreadable += 1
            continue
        skills: dict[str, float] = {}
        bad_values = 0
        for name, value in (skills_raw or {}).items():
            try:
                skills[str(name)] = float(value)
            except (TypeError, ValueError):
                bad_values += 1
        bkt: dict[str, float] = {}
        for name, value in (bkt_raw or {}).items():
            try:
                bkt[str(name)] = float(value)
            except (TypeError, ValueError):
                bad_values += 1
        try:
            started_at = float(data.get("started_at") or 0.0)
        except (TypeError, ValueError):
            started_at = 0.0
        try:
            turns = int(data.get("turns") or 0)
        except (TypeError, ValueError):
            turns = 0
        records.append({
            "topic": str(data.get("topic") or "").strip(),
            "started_at": started_at,
            "turns": turns,
            "skills": skills,
            "bkt": bkt,
            "bad_values": bad_values,
        })
    return records, unreadable


def _learning_progress(goal: Goal, *, now: float | None) -> ClassProgress | None:
    sessions_dir = _settings_home() / "learn" / "sessions"
    records, unreadable_files = _learn_session_records(sessions_dir)
    tagged = _tag_value(goal, LEARN_TAG)
    scope: str | None = None
    link = ""
    if tagged is not None:
        scope = tagged
        link = "learn-tag"
    else:
        wanted = _normalise_name(getattr(goal, "title", ""))
        if wanted:
            matches = {r["topic"] for r in records
                       if r["topic"]
                       and _normalise_name(r["topic"]) == wanted}
            if len(matches) == 1:
                scope = matches.pop()
                link = "title"
    if scope is None:
        return None  # no declared or exactly-matching topic: no scope
    scope_norm = _normalise_name(scope)
    scoped = [r for r in records
              if r["topic"] and _normalise_name(r["topic"]) == scope_norm]
    evidence: dict[str, Any] = {
        "measure": "mastery",
        "topic": scope,
        "link": link,
        "sessions": len(scoped),
    }
    if unreadable_files:
        evidence["unreadable_session_files"] = unreadable_files
    if not scoped:
        return ClassProgress(
            percent=0.0,
            basis="measured",
            source="tutor mastery",
            headline=(
                f"no tutor sessions on {scope!r} yet — no mastery "
                "demonstrated"),
            evidence=evidence,
        )
    # Mastery is a state: per skill, the most recent session that
    # covered it carries the current P(knows) — its bkt entry, or
    # the prior for a skill named but never observed
    # (MasteryModel.bkt() semantics).
    since = float(getattr(goal, "created_at", 0.0) or 0.0)
    latest: dict[str, float] = {}
    total_turns = 0
    predating = 0
    bad_values = 0
    for record in sorted(scoped, key=lambda r: r["started_at"]):
        total_turns += record["turns"]
        bad_values += record["bad_values"]
        if record["started_at"] < since:
            predating += 1
        for skill in set(record["skills"]) | set(record["bkt"]):
            latest[skill] = record["bkt"].get(skill, _BKT_P_INIT)
    if not latest:
        return ClassProgress(
            percent=0.0,
            basis="measured",
            source="tutor mastery",
            headline=(
                f"tutor sessions on {scope!r} recorded no skills "
                "yet — no mastery demonstrated"),
            evidence=evidence,
        )
    percent = sum(latest.values()) / len(latest) * 100.0
    headline = (
        f"mastery averages {percent:.0f}% across {len(latest)} "
        f"skill(s) in {len(scoped)} tutor session(s) on {scope}")
    evidence["skills"] = {k: round(v, 4) for k, v in
                          sorted(latest.items())}
    evidence["skill_count"] = len(latest)
    evidence["total_turns"] = total_turns
    if predating:
        evidence["sessions_predating_goal"] = predating
    if bad_values:
        evidence["unreadable_mastery_values"] = bad_values
    return ClassProgress(
        percent=percent,
        basis="measured",
        source="tutor mastery",
        headline=headline,
        evidence=evidence,
    )


def learning_progress(
    db: Any, goal: Goal, *, now: float | None = None,
) -> ClassProgress | None:
    """Learning-class provider: BKT mastery over the goal's topic.

    ``None`` when the goal declares no topic (no ``learn:`` tag
    and no exact title match against a session topic) or when
    the session store cannot be read — failures are logged with
    the goal named, never raised into a board.  A declared topic
    with no sessions yet reads a measured 0%.
    """
    try:
        return _learning_progress(goal, now=now)
    except Exception as exc:  # noqa: BLE001 - contained: see docstring
        _log.warning(
            "learning progress provider failed for goal %s (%r): %s",
            getattr(goal, "goal_id", "?"), getattr(goal, "title", ""), exc)
        return None


# ── build: execution steps from the tracker / agent-goal stores ───────

#: Explicit build link: a tracker-goal tag ``agent-goal:<id>``
#: naming one of the brain's agent goals (``agents/goals.py``) in
#: the same context database.  The linked goal's step completion
#: IS this goal's measured progress.  A present-but-stale tag
#: measures nothing (the P3 rule).
AGENT_GOAL_TAG = "agent-goal:"


def _build_progress(db: Any, goal: Goal, *,
                    now: float | None) -> ClassProgress | None:
    linked_id = _tag_value(goal, AGENT_GOAL_TAG)
    if linked_id is not None:
        if db is None:
            return None  # the agent store lives in the shared db
        found = db.query_one(
            "SELECT id FROM agent_goals WHERE id = ?", (linked_id,))
        if found is None:
            return None  # stale deliberate link: never fall through
        rows = db.query(
            "SELECT status FROM agent_goal_steps WHERE goal_id = ?",
            (linked_id,))
        by_status: dict[str, int] = {}
        for row in rows or []:
            status = str((row or {}).get("status") or "")
            by_status[status] = by_status.get(status, 0) + 1
        done = by_status.get("done", 0)
        skipped = by_status.get("skipped", 0)
        total = sum(by_status.values()) - skipped
        if total <= 0:
            return None  # no executable steps recorded to measure
        percent = done / total * 100.0
        headline = (
            f"{done} of {total} step(s) done on the linked agent "
            "goal")
        evidence: dict[str, Any] = {
            "measure": "execution_steps",
            "link": "agent-goal",
            "agent_goal_id": linked_id,
            "steps_done": done,
            "steps_total": total,
            "steps_by_status": by_status,
        }
        if skipped:
            evidence["steps_skipped"] = skipped
        return ClassProgress(
            percent=percent,
            basis="measured",
            source="agent goal steps",
            headline=headline,
            evidence=evidence,
        )
    if db is not None:
        rows = db.query(
            "SELECT is_completed FROM subgoals WHERE goal_id = ?",
            (getattr(goal, "goal_id", ""),))
        total = len(rows or [])
        done = sum(1 for row in rows or []
                   if (row or {}).get("is_completed"))
    else:
        subgoals = getattr(goal, "subgoals", None) or []
        total = len(subgoals)
        done = sum(1 for sub in subgoals
                   if getattr(sub, "is_completed", False))
    if total == 0:
        return None  # no execution steps recorded anywhere
    percent = done / total * 100.0
    return ClassProgress(
        percent=percent,
        basis="measured",
        source="goal subgoals",
        headline=f"{done} of {total} step(s) completed",
        evidence={
            "measure": "execution_steps",
            "link": "subgoals",
            "steps_done": done,
            "steps_total": total,
        },
    )


def build_progress(
    db: Any, goal: Goal, *, now: float | None = None,
) -> ClassProgress | None:
    """Build-class provider: execution steps, measured.

    The linked agent goal's steps when an ``agent-goal:`` tag
    declares one, else the goal's own tracker subgoals.
    ``None`` when a deliberate link is stale, or when no steps
    are recorded anywhere — failures are logged with the goal
    named, never raised into a board.
    """
    try:
        return _build_progress(db, goal, now=now)
    except Exception as exc:  # noqa: BLE001 - contained: see docstring
        _log.warning(
            "build progress provider failed for goal %s (%r): %s",
            getattr(goal, "goal_id", "?"), getattr(goal, "title", ""), exc)
        return None


# ── relationships: kept promises to one person ────────────────────────

#: Person scope declaration: a tracker-goal tag ``person:<name>``
#: naming the promisee the goal is about, e.g. ``person:Mum``.
#: Without one, the goal's title must match a promisee exactly
#: (normalised, unique — the finance name-match rule).
PERSON_TAG = "person:"


def _relationships_progress(goal: Goal, *,
                            now: float | None) -> ClassProgress | None:
    path = _data_file(_COMMITMENTS_FILENAME)
    raw_rows = _raw_store_rows(path, "entries")
    ledger = CommitmentLedger(str(path))
    tagged = _tag_value(goal, PERSON_TAG)
    link = ""
    person_norm = ""
    person_display = ""
    if tagged is not None:
        person_norm = _normalise_name(tagged)
        person_display = tagged
        link = "person-tag"
        if not person_norm:
            return None
    else:
        wanted = _normalise_name(getattr(goal, "title", ""))
        if not wanted:
            return None
        spellings: dict[str, str] = {}
        for commitment in ledger.commitments.values():
            if str(commitment.promisor or "owner") != "owner":
                continue
            norm = _normalise_name(commitment.promisee)
            if norm and norm not in spellings:
                spellings[norm] = str(commitment.promisee).strip()
        if wanted not in spellings:
            return None  # no promisee carries this goal's name
        person_norm = wanted
        person_display = spellings[wanted]
        link = "title"
    since = float(getattr(goal, "created_at", 0.0) or 0.0)
    until = time.time() if now is None else float(now)
    kept, broken, overdue_open, excluded, unreadable_rows = \
        _commitment_counts(ledger, since=since, until=until,
                           promisee_norm=person_norm)
    due = kept + broken + overdue_open
    if due == 0:
        return None
    percent = kept / due * 100.0
    headline = (
        f"kept {kept} of {due} promise(s) to {person_display} "
        "due since this goal was filed")
    misses = []
    if broken:
        misses.append(f"{broken} broken")
    if overdue_open:
        misses.append(f"{overdue_open} overdue")
    if misses:
        headline += " (" + ", ".join(misses) + ")"
    evidence: dict[str, Any] = {
        "measure": "person_commitments",
        "person": person_display,
        "link": link,
        "kept": kept,
        "broken": broken,
        "overdue_open": overdue_open,
        "due": due,
        "since": since,
        "excluded": {k: v for k, v in excluded.items() if v},
    }
    unreadable = len(raw_rows or []) - len(ledger.entries)
    if unreadable > 0:
        evidence["unreadable_ledger_entries"] = unreadable
    if unreadable_rows:
        evidence["unreadable_commitment_rows"] = unreadable_rows
    return ClassProgress(
        percent=percent,
        basis="measured",
        source="commitment ledger",
        headline=headline,
        evidence=evidence,
    )


def relationships_progress(
    db: Any, goal: Goal, *, now: float | None = None,
) -> ClassProgress | None:
    """Relationships-class provider: promises to one person, kept.

    ``None`` when the goal names no measurable person (no
    ``person:`` tag and no exact title-to-promisee match), when
    nothing has come due to that person in the goal's window,
    or when the ledger cannot be read — failures are logged
    with the goal named, never raised into a board.
    """
    try:
        return _relationships_progress(goal, now=now)
    except Exception as exc:  # noqa: BLE001 - contained: see docstring
        _log.warning(
            "relationships progress provider failed for goal %s "
            "(%r): %s",
            getattr(goal, "goal_id", "?"), getattr(goal, "title", ""), exc)
        return None


# ── registration ──────────────────────────────────────────────────────


def register_providers() -> None:
    """Register the WS2 measured-progress providers.

    Finance + trading + careers (P3/P4), productivity + interests
    (P5), health + learning + build + relationships (P6) — every
    class with a measuring organ; ``general`` stays manual by
    design.  Idempotent (re-registration replaces).  Called by
    :mod:`nomorals.goals` when the package loads — the one wiring
    point every GoalTracker entry point shares.
    """
    register_progress_provider("finance", finance_progress)
    register_progress_provider("trading", trading_progress)
    register_progress_provider("careers", careers_progress)
    register_progress_provider("productivity", productivity_progress)
    register_progress_provider("interests", interests_progress)
    register_progress_provider("health", health_progress)
    register_progress_provider("learning", learning_progress)
    register_progress_provider("build", build_progress)
    register_progress_provider("relationships", relationships_progress)
