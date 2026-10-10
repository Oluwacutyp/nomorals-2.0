"""Savings goals — native, no API.

A goal is a named target ("emergency fund ₦500k by December"). Progress
is measured from the ledger: income-kind transactions tagged to the
goal's category (default "savings") count toward it. Goals keep Devon
honest: the weekly digest flags goals that are off-pace.
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

_log = get_logger("nomorals.finance")

__all__ = ["Goal", "GoalStore", "create_goal", "goal_progress"]


@dataclass
class Goal:
    id: str
    name: str
    target_kobo: int
    deadline_ts: float | None = None
    category: str = "savings"   # ledger category that funds this goal
    created_at: float = 0.0
    done: bool = False

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
        )


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
    store: GoalStore | None = None,
) -> Goal:
    """Validate and persist a savings goal. Raises ValueError on bad input."""
    name = (name or "").strip()
    if not name:
        raise ValueError("goal name is required")
    if target_kobo <= 0:
        raise ValueError("goal target must be positive (kobo)")
    deadline_ts = _parse_deadline(deadline) if deadline else None
    if deadline and deadline_ts is None:
        raise ValueError(
            f"couldn't parse deadline {deadline!r} — try 2026-12-31, "
            "'in 90d', or 'dec'")
    goal = Goal(
        id="goal_" + uuid.uuid4().hex[:10],
        name=name, target_kobo=int(target_kobo),
        deadline_ts=deadline_ts,
        category=(category or "savings").strip().lower() or "savings",
        created_at=time.time(),
    )
    return (store or GoalStore()).add(goal)


def goal_progress(goal: Goal, ledger: Ledger,
                  *, now: float | None = None) -> dict[str, Any]:
    """Progress of one goal from the ledger.

    Contributions = income-kind transactions in the goal's category since
    the goal was created (savings deposits, ajo payouts, etc.).
    """
    now = time.time() if now is None else now
    contributed = sum(
        t.amount_kobo for t in ledger.transactions(
            since=goal.created_at, until=now, kind="income",
            category=goal.category))
    # Also count spend-kind "savings" moves (e.g. /spend logged to savings
    # as money set aside) — both directions fund the goal.
    contributed += sum(
        t.amount_kobo for t in ledger.transactions(
            since=goal.created_at, until=now, kind="spend",
            category=goal.category))
    pct = contributed / goal.target_kobo if goal.target_kobo else 0.0
    remaining = max(0, goal.target_kobo - contributed)
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
    return {
        "id": goal.id, "name": goal.name,
        "target": format_naira(goal.target_kobo),
        "contributed": format_naira(contributed),
        "contributed_kobo": contributed,
        "pct": round(pct, 3),
        "remaining": format_naira(remaining),
        "remaining_kobo": remaining,
        "deadline": (datetime.fromtimestamp(goal.deadline_ts).astimezone()
                     .strftime("%Y-%m-%d") if goal.deadline_ts else ""),
        "done": goal.done or contributed >= goal.target_kobo,
        **pace,
    }
