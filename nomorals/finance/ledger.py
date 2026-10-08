"""Expense ledger: integer-kobo transactions, Nigerian amount parsing.

Money is stored and summed as integer kobo — ``0.1 + 0.2`` style float
drift is impossible by construction.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger("nomorals.finance")

DEFAULT_CURRENCY = "₦"

#: Extensible keyword → category map. First match wins; order matters.
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "transport": (
        "transport", "fuel", "petrol", "diesel", "uber", "bolt", "indrive",
        "danfo", "keke", "tricycle", "okada", "bus", "taxi", "cab", "fare",
        "toll", "parking", "car wash", "mechanic",
    ),
    "food": (
        "food", "rice", "beans", "mama put", "mamaput", "restaurant",
        "shawarma", "suya", "eatery", "canteen", "lunch", "dinner",
        "breakfast", "shaki", "amala", "fufu", "jollof", "groceries",
        "market food", "snack",
    ),
    "data": (
        "data", "airtime", "mtn", "glo", "airtel", "9mobile", "etisalat",
        "recharge", "topup", "top-up", "top up", "subscription",
    ),
    "housing": (
        "rent", "landlord", "accommodation", "hostel", "nepa", "phcn",
        "electricity", "light bill", "prepaid meter", "water bill",
        "service charge", "estate",
    ),
    "health": (
        "hospital", "pharmacy", "drugs", "drug", "doctor", "clinic",
        "chemist", "medical", "lab test", "hmo",
    ),
    "shopping": (
        "shopping", "clothes", "shirt", "shoe", "dress", "boutique",
        "jumia", "konga", "jiji", "market", "tailor", "phone case",
    ),
    "entertainment": (
        "cinema", "movie", "club", "party", "concert", "game", "bet",
        "betting", "sportybet", "netflix", "dstv", "spotify", "show",
    ),
    "savings": (
        "savings", "ajo", "thrift", "esusu", "investment", "piggyvest",
        "cowrywise", "fixed deposit",
    ),
    "education": (
        "school", "tuition", "course", "book", "tutorial", "exam",
        "lesson", "training",
    ),
    "giving": (
        "tithe", "offering", "charity", "donation", "gift", "dash",
    ),
}

#: Canonical category list (order used in summaries).
CATEGORIES: tuple[str, ...] = tuple(CATEGORY_KEYWORDS) + ("income", "other")

# ── amount parsing ───────────────────────────────────────────────────────────

# Matches: ₦5k · 5k · 2.5m · ₦1,500 · 2000 naira · NGN 3.2m · 750
_RE_AMOUNT = re.compile(
    r"^\s*(?:₦|ngn|naira)?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<suffix>[kKmM])?"
    r"\s*(?:naira|ngn)?\s*$",
    re.IGNORECASE,
)

_SUFFIX_MULTIPLIER = {"k": 1_000, "m": 1_000_000}


def naira_to_kobo(naira: Decimal | int | str) -> int:
    """Exact naira → kobo conversion. Never floats."""
    return int(Decimal(str(naira)) * 100)


def parse_amount(text: str) -> int | None:
    """Parse Nigerian shorthand into integer kobo.

    "5k" → 500000 · "2.5m" → 250000000 · "₦1,500" → 150000 ·
    "2000 naira" → 200000. Returns None when unparseable.
    """
    if not text:
        return None
    m = _RE_AMOUNT.match(text.strip())
    if not m:
        return None
    try:
        num = Decimal(m.group("num").replace(",", ""))
    except InvalidOperation:
        return None
    if num < 0:
        return None
    suffix = (m.group("suffix") or "").lower()
    naira = num * _SUFFIX_MULTIPLIER.get(suffix, 1)
    return int(naira * 100)


def format_naira(kobo: int, currency: str = DEFAULT_CURRENCY) -> str:
    """Format integer kobo as ₦5,000. Handles negatives."""
    sign = "-" if kobo < 0 else ""
    naira, kob = divmod(abs(kobo), 100)
    if kob:
        return f"{sign}{currency}{naira:,}.{kob:02d}"
    return f"{sign}{currency}{naira:,}"


def categorize(note: str) -> str:
    """Rule-based, offline categorizer. Returns a canonical category."""
    text = (note or "").lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return category
    return "other"


# ── ledger ───────────────────────────────────────────────────────────────────

@dataclass
class Transaction:
    """One money movement. Amounts are integer kobo."""

    ts: float
    amount_kobo: int
    category: str
    note: str = ""
    kind: str = "spend"  # "spend" | "income"
    source: str = "manual"  # "manual" | "mono" (bank import hook) | ...

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "amount_kobo": self.amount_kobo,
            "category": self.category,
            "note": self.note,
            "kind": self.kind,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transaction":
        return cls(
            ts=float(data.get("ts", 0)),
            amount_kobo=int(data.get("amount_kobo", 0)),
            category=str(data.get("category", "other")),
            note=str(data.get("note", "")),
            kind=str(data.get("kind", "spend")),
            source=str(data.get("source", "manual")),
        )


class Ledger:
    """Append-only JSON-lines transaction log. Thread-safe."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".nomorals" / "finance" / "ledger.jsonl"
        self.path = Path(path)
        self._lock = threading.RLock()

    def _ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def log(
        self,
        amount_kobo: int,
        category: str | None = None,
        note: str = "",
        kind: str = "spend",
        ts: float | None = None,
        source: str = "manual",
    ) -> Transaction:
        """Append a transaction. Category auto-detected when omitted."""
        if amount_kobo <= 0:
            raise ValueError("amount_kobo must be positive")
        if kind not in ("spend", "income"):
            raise ValueError(f"kind must be spend|income, got {kind!r}")
        category = (category or "").strip().lower() or categorize(note)
        txn = Transaction(
            ts=time.time() if ts is None else ts,
            amount_kobo=int(amount_kobo),
            category=category,
            note=note,
            kind=kind,
            source=source,
        )
        with self._lock:
            self._ensure()
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(txn.to_dict()) + "\n")
        _log.debug("logged %s %s (%s)", kind, format_naira(txn.amount_kobo), category)
        return txn

    def transactions(
        self,
        since: float | None = None,
        until: float | None = None,
        category: str | None = None,
        kind: str | None = None,
    ) -> list[Transaction]:
        """Read back transactions with optional filters."""
        out: list[Transaction] = []
        with self._lock:
            if not self.path.exists():
                return out
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    txn = Transaction.from_dict(json.loads(line))
                except (ValueError, TypeError, AttributeError):
                    _log.debug("skipping malformed ledger line")
                    continue
                if since is not None and txn.ts < since:
                    continue
                if until is not None and txn.ts > until:
                    continue
                if category is not None and txn.category != category:
                    continue
                if kind is not None and txn.kind != kind:
                    continue
                out.append(txn)
        return out

    def total_spent(
        self,
        since: float | None = None,
        until: float | None = None,
        category: str | None = None,
    ) -> int:
        """Sum of spend in kobo. Integer arithmetic — no float drift."""
        return sum(
            t.amount_kobo
            for t in self.transactions(since=since, until=until,
                                       category=category, kind="spend")
        )

    def total_income(
        self,
        since: float | None = None,
        until: float | None = None,
    ) -> int:
        return sum(
            t.amount_kobo
            for t in self.transactions(since=since, until=until, kind="income")
        )
