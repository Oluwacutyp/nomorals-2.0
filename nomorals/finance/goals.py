"""Savings goals — native, no API.

A goal is a named target ("emergency fund ₦500k by December"). Progress
is measured from the ledger: income-kind transactions tagged to the
goal's category (default "savings") count toward it. Goals keep Devon
honest: the weekly digest flags goals that are off-pace.

Goal types follow YNAB's three (the best goal UX in budgeting):
  target   — have ₦X total (optionally by a date)
  monthly  — put away ₦X every month (builder goal)
  by_date  — need ₦X for a specific date (YNAB auto-breaks it into
             monthly chunks — so do we)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .budgets import finance_paths
from .ledger import Ledger, format_naira
from .style import bar, current_theme

_log = get_logger("nomorals.finance")

__all__ = [
    "Goal", "GoalStore", "create_goal", "goal_progress", "render_goals",
    "GOAL_TARGET", "GOAL_MONTHLY", "GOAL_BY_DATE",
]

GOAL_TARGET = "target"
GOAL_MONTHLY = "monthly"
GOAL_BY_DATE = "by_date"
GOAL_TYPES = (GOAL_TARGET, GOAL_MONTHLY, GOAL_BY_DATE)


@dataclass
class Goal:
    id: str
    name: str
    target_kobo: int
    deadline_ts: float | None = None
    category: str = "savings"   # ledger category that funds this goal
    created_at: float = 0.0
    done: bool = False
    goal_type: str = GOAL_TARGET  # target | monthly | by_date
    snooze_until: float | None = None  # YNAB's beloved snooze

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Goal":
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            target_kobo=int(data.get("target_kobo", 0) or 0),
            deadline_ts=(float(data["deadline_ts"])
                         if data.get("deadline_ts") else None),
            category=str(data.get("category", "savings") or "savings"),
            created_at=float(data.get("created_at", 0) or 0),
            done=bool(data.get("done", False)),
            goal_type=str(data.get("goal_type", GOAL_TARGET) or GOAL_TARGET),
            snooze_until=(float(data["snooze_until"])
                          if data.get("snooze_until") else None),
        )

    @property
    def snoozed(self) -> bool:
        return bool(self.snooze_until) and self.snooze_until > time.time()


class GoalStore:
    """Persistent goal registry. JSON file."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            _, budgets_path = finance_paths(None)
            path = Path(budgets_path).parent / "goals.json"
        self.path = Path(path)

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, ValueError):
            pass
        return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, goal: Goal) -> Goal:
        data = self._load()
        data[goal.id] = goal.to_dict()
        self._save(data)
        return goal

    def get(self, goal_id: str) -> Goal | None:
        raw = self._load().get(goal_id or "")
        if not raw:
            return None
        try:
            return Goal.from_dict(raw)
        except Exception:  # noqa: BLE001
            return None

    def list(self, *, include_done: bool = False) -> list[Goal]:
        out = []
        for raw in self._load().values():
            try:
                g = Goal.from_dict(raw)
            except Exception:  # noqa: BLE001
                continue
            if g.done and not include_done:
                continue
            out.append(g)
        return sorted(out, key=lambda g: g.created_at)

    def remove(self, goal_id: str) -> bool:
        data = self._load()
        if goal_id not in data:
            return False
        del data[goal_id]
        self._save(data)
        return True

    def mark_done(self, goal_id: str) -> bool:
        g = self.get(goal_id)
        if g is None:
            return False
        g.done = True
        data = self._load()
        data[g.id] = g.to_dict()
        self._save(data)
        return True

    def snooze(self, goal_id: str, days: float = 30.0) -> bool:
        """Pause pace-nagging on a goal (YNAB's snooze). Not deletion."""
        g = self.get(goal_id)
        if g is None:
            return False
        g.snooze_until = time.time() + float(days) * 86400
        data = self._load()
        data[g.id] = g.to_dict()
        self._save(data)
        return True

    def unsnooze(self, goal_id: str) -> bool:
        g = self.get(goal_id)
        if g is None:
            return False
        g.snooze_until = None
        data = self._load()
        data[g.id] = g.to_dict()
        self._save(data)
        return True


def _parse_deadline(text: str) -> float | None:
    """'2026-12-31' or 'dec' (next Dec 31) or 'in 90d' → epoch, else None."""
    text = (text or "").strip().lower()
    if not text:
        return None
    import re
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)),
                            int(m.group(3))).astimezone().timestamp()
        except ValueError:
            return None
    m = re.fullmatch(r"in\s+(\d+)\s*d", text)
    if m:
        return time.time() + int(m.group(1)) * 86400
    months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
              "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    m = re.fullmatch(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                     text[:3])
    if m:
        now = datetime.now().astimezone()
        year = now.year + (1 if months[m.group(1)] <= now.month else 0)
        last_day = {2: 28}.get(months[m.group(1)], 30)
        if months[m.group(1)] in (1, 3, 5, 7, 8, 10, 12):
            last_day = 31
        return datetime(year, months[m.group(1)], last_day,
                        23, 59).astimezone().timestamp()
    return None


def create_goal(
    name: str,
    target_kobo: int,
    *,
    deadline: str = "",
    category: str = "savings",
    goal_type: str = GOAL_TARGET,
    store: GoalStore | None = None,
) -> Goal:
    """Validate and persist a savings goal. Raises ValueError on bad input.

    goal_type: "target" (have ₦X), "monthly" (save ₦X every month),
    "by_date" (need ₦X by the deadline — auto-chunked into months).
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("goal name is required")
    if target_kobo <= 0:
        raise ValueError("goal target must be positive (kobo)")
    goal_type = (goal_type or GOAL_TARGET).strip().lower()
    if goal_type not in GOAL_TYPES:
        raise ValueError(f"goal_type must be one of {GOAL_TYPES}")
    deadline_ts = _parse_deadline(deadline) if deadline else None
    if deadline and deadline_ts is None:
        raise ValueError(
            f"couldn't parse deadline {deadline!r} — try 2026-12-31, "
            "'in 90d', or 'dec'")
    if goal_type == GOAL_BY_DATE and deadline_ts is None:
        raise ValueError("by_date goals need a deadline")
    goal = Goal(
        id="goal_" + uuid.uuid4().hex[:10],
        name=name, target_kobo=int(target_kobo),
        deadline_ts=deadline_ts,
        category=(category or "savings").strip().lower() or "savings",
        created_at=time.time(),
        goal_type=goal_type,
    )
    return (store or GoalStore()).add(goal)


def _contributions(goal: Goal, ledger: Ledger, now: float) -> list[Any]:
    """All ledger moves funding this goal since creation."""
    out = list(ledger.transactions(
        since=goal.created_at, until=now, kind="income",
        category=goal.category))
    out += list(ledger.transactions(
        since=goal.created_at, until=now, kind="spend",
        category=goal.category))
    return sorted(out, key=lambda t: t.ts)


def contribution_streak(goal: Goal, ledger: Ledger,
                        *, now: float | None = None) -> int:
    """Consecutive months (incl. current) with at least one contribution."""
    now = time.time() if now is None else now
    months: set[str] = set()
    for t in _contributions(goal, ledger, now):
        d = datetime.fromtimestamp(t.ts).astimezone()
        months.add(f"{d.year}-{d.month:02d}")
    streak = 0
    d = datetime.fromtimestamp(now).astimezone()
    while True:
        key = f"{d.year}-{d.month:02d}"
        if key in months:
            streak += 1
        else:
            break
        d = d.replace(day=1)
        d = d.replace(year=d.year - 1, month=12) if d.month == 1 else \
            d.replace(month=d.month - 1)
    return streak


def monthly_need(goal: Goal, contributed_kobo: int,
                 *, now: float | None = None) -> int | None:
    """Kobo/month still needed — YNAB's "by date" chunking math.

    target  → remaining ÷ months left (when there's a deadline)
    monthly → the target itself, every month
    by_date → remaining ÷ months left (deadline required)
    """
    now = time.time() if now is None else now
    if goal.goal_type == GOAL_MONTHLY:
        return goal.target_kobo
    if not goal.deadline_ts or goal.deadline_ts <= now:
        return None
    months_left = _months_between(now, goal.deadline_ts)
    if months_left <= 0:
        return None
    remaining = max(0, goal.target_kobo - contributed_kobo)
    return int(-(-remaining // months_left))  # ceil division


def _months_between(start_ts: float, end_ts: float) -> int:
    s = datetime.fromtimestamp(start_ts).astimezone()
    e = datetime.fromtimestamp(end_ts).astimezone()
    n = (e.year - s.year) * 12 + (e.month - s.month)
    if e.day >= s.day:
        n += 1
    return max(1, n)


def goal_progress(goal: Goal, ledger: Ledger,
                  *, now: float | None = None) -> dict[str, Any]:
    """Progress of one goal from the ledger.

    Contributions = income-kind transactions in the goal's category since
    the goal was created (savings deposits, ajo payouts, etc.).
    """
    now = time.time() if now is None else now
    contributed = sum(t.amount_kobo for t in _contributions(goal, ledger, now))
    pct = contributed / goal.target_kobo if goal.target_kobo else 0.0
    remaining = max(0, goal.target_kobo - contributed)
    need = monthly_need(goal, contributed, now=now)
    pace: dict[str, Any] = {"on_track": None}
    if goal.deadline_ts and goal.deadline_ts > now:
        days_left = (goal.deadline_ts - now) / 86400
        elapsed = max(1.0, (now - goal.created_at) / 86400)
        expected_pct = elapsed / (elapsed + days_left)
        pace = {
            "on_track": pct >= expected_pct * 0.9,
            "expected_pct": round(expected_pct, 3),
            "days_left": round(days_left, 1),
            "needed_per_day_kobo": int(remaining / max(1.0, days_left)),
        }
    # Monthly-builder goals: on track = this month's contribution ≥ target.
    if goal.goal_type == GOAL_MONTHLY:
        mk = datetime.fromtimestamp(now).astimezone().strftime("%Y-%m")
        year, mon = (int(x) for x in mk.split("-", 1))
        start = datetime(year, mon, 1).astimezone().timestamp()
        mtd = sum(t.amount_kobo for t in _contributions(goal, ledger, now)
                  if t.ts >= start)
        pace["on_track"] = mtd >= goal.target_kobo * 0.9
        pace["month_contributed_kobo"] = mtd
    return {
        "id": goal.id, "name": goal.name,
        "target": format_naira(goal.target_kobo),
        "contributed": format_naira(contributed),
        "contributed_kobo": contributed,
        "pct": round(pct, 3),
        "remaining": format_naira(remaining),
        "remaining_kobo": remaining,
        "goal_type": goal.goal_type,
        "monthly_need_kobo": need,
        "monthly_need": format_naira(need) if need else "",
        "streak_months": contribution_streak(goal, ledger, now=now),
        "snoozed": goal.snoozed,
        "deadline": (datetime.fromtimestamp(goal.deadline_ts).astimezone()
                     .strftime("%Y-%m-%d") if goal.deadline_ts else ""),
        "done": goal.done or contributed >= goal.target_kobo,
        **pace,
    }


def render_goals(goals: list[Goal], ledger: Ledger,
                 theme: str | None = None,
                 now: float | None = None) -> str:
    """Styled goals view: bars, pace, monthly need, streaks."""
    th = current_theme(theme)
    lines = [th.paint(f"{th.money_bag} savings goals", th.bold)]
    if not goals:
        return "\n".join(
            lines + ["  none yet — /goal add emergency-fund 500k dec"])
    for g in goals:
        p = goal_progress(g, ledger, now=now)
        pace_txt = ""
        if g.snoozed:
            pace_txt = th.paint(" 💤 snoozed", th.dim)
        elif p.get("on_track") is False:
            pace_txt = th.paint(" ⚠️ off pace", th.warn)
        elif p.get("on_track") is True:
            pace_txt = th.paint(" ✅ on pace", th.good)
        need = (f" · need {p['monthly_need']}/mo" if p["monthly_need"]
                else "")
        streak = (f" · 🔥{p['streak_months']}mo" if p["streak_months"] >= 2
                  and th.use_emoji else
                  f" · {p['streak_months']}mo streak"
                  if p["streak_months"] >= 2 else "")
        dl = f" by {p['deadline']}" if p["deadline"] else ""
        done_mark = " 🎉" if p["done"] and th.use_emoji else (
            " DONE" if p["done"] else "")
        lines.append(
            f"  {th.bullet} {g.name} [{g.goal_type}]: "
            f"{bar(p['pct'], theme=th)} {p['pct']:.0%} "
            f"({p['contributed']} of {p['target']}){dl}{need}"
            f"{streak}{pace_txt}{done_mark}")
    return "\n".join(lines)
