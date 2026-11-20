"""Hypothesis ledger and next-action selection.

The autonomy layer. It does not call tools; it ranks what to do next and
records what happened, so the agent has a reason to pick one action over
another instead of cycling on the first tool it remembers.

Each hypothesis carries a class, a target, a confidence that moves with
evidence, and a state. Ranking is expected-value: confidence x class
weight / cost. A confirmed hypothesis spawns children (a confirmed sqli
suggests the neighbouring params); a rejected one is remembered so it is
not retried.

Durable JSONL, append-only, so a run survives a restart.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ..core.policy import Capability
from ..core.logging_setup import get_logger

_log = get_logger(__name__)
__all__ = ["Hypothesis", "Ledger", "CLASS_COST", "CLASS_WEIGHT", "register"]

#: relative cost of testing each class (requests), and its payoff weight
CLASS_COST: dict[str, float] = {
    "params": 30, "sqli": 40, "xss": 20, "ssti": 15, "lfi": 15,
    "redirect": 8, "ssrf": 12, "cors": 6, "graphql": 4, "jwt": 2,
    "race": 25, "takeover": 3, "exposed_file": 2,
}
CLASS_WEIGHT: dict[str, float] = {
    "sqli": 1.0, "ssti": 0.9, "lfi": 0.85, "ssrf": 0.7, "takeover": 0.9,
    "graphql": 0.6, "cors": 0.5, "xss": 0.55, "jwt": 0.8, "race": 0.6,
    "redirect": 0.35, "params": 0.9, "exposed_file": 0.4,
}


@dataclass
class Hypothesis:
    cls: str
    target: str
    param: str = ""
    confidence: float = 0.5
    state: str = "open"        # open | testing | confirmed | rejected
    parent: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    notes: list[str] = field(default_factory=list)

    @property
    def value(self) -> float:
        return (self.confidence * CLASS_WEIGHT.get(self.cls, 0.3)
                / max(1.0, CLASS_COST.get(self.cls, 10)))

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "class": self.cls, "target": self.target,
                "param": self.param, "confidence": round(self.confidence, 2),
                "state": self.state, "parent": self.parent,
                "value": round(self.value, 5), "notes": self.notes[-5:]}


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, Hypothesis] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            item = Hypothesis(cls=row["class"], target=row["target"],
                              param=row.get("param", ""),
                              confidence=row.get("confidence", 0.5),
                              state=row.get("state", "open"),
                              parent=row.get("parent", ""),
                              id=row.get("id") or uuid.uuid4().hex[:12])
            self._items[item.id] = item

    def _append(self, item: Hypothesis, note: str = "") -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**item.as_dict(), "note": note}) + "\n")

    def add(self, cls: str, target: str, *, param: str = "",
            confidence: float = 0.5, parent: str = "") -> Hypothesis:
        item = Hypothesis(cls=cls, target=target, param=param,
                          confidence=max(0.05, min(0.95, confidence)),
                          parent=parent)
        self._items[item.id] = item
        self._append(item, "created")
        return item

    def next(self, limit: int = 5) -> list[Hypothesis]:
        open_items = [h for h in self._items.values() if h.state == "open"]
        open_items.sort(key=lambda h: -h.value)
        return open_items[:limit]

    def resolve(self, hid: str, *, confirmed: bool,
                note: str = "") -> Optional[Hypothesis]:
        item = self._items.get(hid)
        if item is None:
            return None
        item.state = "confirmed" if confirmed else "rejected"
        item.confidence = 1.0 if confirmed else 0.05
        if note:
            item.notes.append(note[:200])
        self._append(item, "resolved")
        if confirmed:
            for cls in self._children(item):
                self.add(cls, item.target, param=item.param,
                         confidence=0.55, parent=item.id)
        return item

    @staticmethod
    def _children(item: Hypothesis) -> list[str]:
        if item.cls == "params":
            return ["sqli", "xss", "redirect"]
        if item.cls == "sqli":
            return ["ssti", "lfi"]
        if item.cls == "xss":
            return ["cors"]
        return []

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for item in self._items.values():
            counts[item.state] = counts.get(item.state, 0) + 1
        return {"total": len(self._items), "states": counts,
                "open_best": [h.as_dict() for h in self.next(3)]}


def register(registry: Any) -> None:
    from .case import case_dir, home_of

    context = registry.context
    home = home_of(context)

    @registry.register("plan_next",
                       description="Ranked next hypotheses to test.",
                       capability=Capability.DB_READ,
                       parameters={"case": "str", "limit": "int (optional, 5)"})
    def plan_next(case: str, *, limit: str = "5") -> dict[str, Any]:
        ledger = Ledger(case_dir(home, case) / "hypotheses.jsonl")
        return {"next": [h.as_dict() for h in ledger.next(int(limit or 5))],
                "stats": ledger.stats()}

    @registry.register("plan_add",
                       description="Add a hypothesis to the ledger.",
                       capability=Capability.DB_WRITE,
                       parameters={"case": "str", "class": "str",
                                   "target": "str", "param": "str (optional)",
                                   "confidence": "float (optional, 0.5)"})
    def plan_add(case: str, cls: str, target: str, *, param: str = "",
                 confidence: str = "0.5") -> dict[str, Any]:
        ledger = Ledger(case_dir(home, case) / "hypotheses.jsonl")
        item = ledger.add(cls, target, param=param,
                          confidence=float(confidence or 0.5))
        return item.as_dict()

    @registry.register("plan_resolve",
                       description="Resolve a hypothesis; confirmation spawns "
                                   "child hypotheses.",
                       capability=Capability.DB_WRITE,
                       parameters={"case": "str", "id": "str",
                                   "confirmed": "bool", "note": "str (optional)"})
    def plan_resolve(case: str, id: str, confirmed: str = "false",
                     *, note: str = "") -> dict[str, Any]:
        ledger = Ledger(case_dir(home, case) / "hypotheses.jsonl")
        item = ledger.resolve(id, confirmed=(confirmed or "").lower()
                              in {"1", "true", "yes"}, note=note)
        return item.as_dict() if item else {"error": "unknown id"}
