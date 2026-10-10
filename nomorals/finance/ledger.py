"""Expense ledger: integer-kobo transactions, Nigerian amount parsing.

Money is stored and summed as integer kobo — ``0.1 + 0.2`` style float
drift is impossible by construction.
"""

from __future__ import annotations

import csv
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
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

#: Word suffixes: "5 thousand" → ₦5,000 · "2.5 million" → ₦2.5M.
_WORD_MULTIPLIER = {
    "thousand": 1_000, "k": 1_000,
    "million": 1_000_000, "m": 1_000_000,
    "billion": 1_000_000_000, "b": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

_RE_AMOUNT_WORD = re.compile(
    r"^\s*(?:₦|ngn|naira|\$|usd)?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<word>thousand|million|billion|trillion)\s*"
    r"(?:naira|ngn|dollars?|usd)?\s*$",
    re.IGNORECASE,
)


def naira_to_kobo(naira: Decimal | int | str) -> int:
    """Exact naira → kobo conversion. Never floats."""
    return int(Decimal(str(naira)) * 100)


def parse_amount(text: str) -> int | None:
    """Parse Nigerian shorthand into integer kobo.

    "5k" → 500000 · "2.5m" → 250000000 · "₦1,500" → 150000 ·
    "2000 naira" → 200000 · "5 thousand" → 500000. Returns None when
    unparseable.
    """
    if not text:
        return None
    cleaned = text.strip()
    m = _RE_AMOUNT.match(cleaned)
    if not m:
        m = _RE_AMOUNT_WORD.match(cleaned)
        if not m:
            return None
        try:
            num = Decimal(m.group("num").replace(",", ""))
        except InvalidOperation:
            return None
        if num < 0:
            return None
        naira = num * _WORD_MULTIPLIER[m.group("word").lower()]
        return int(naira * 100)
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


def extract_merchant(note: str) -> str:
    """Best-effort merchant name from a free-text note.

    "Netflix, monthly sub" → "Netflix" · "Uber - trip to Lekki" → "Uber".
    """
    note = (note or "").strip()
    if not note:
        return ""
    return note.split(",")[0].split("-")[0].strip()[:40]


class MerchantMemory:
    """Learned merchant → category corrections. Corrections stick.

    Firefly III's "alternative categories" pattern: when the owner (or
    Devon) corrects a transaction's category, the merchant is remembered
    and future transactions from that merchant categorize correctly
    without asking again.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = (Path.home() / ".nomorals" / "finance"
                    / "merchant_memory.json")
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._data is None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._data = {str(k).lower(): str(v)
                              for k, v in raw.items()} \
                    if isinstance(raw, dict) else {}
            except (OSError, ValueError):
                self._data = {}
        return self._data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._load(), indent=1, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(self.path)

    def lookup(self, note: str) -> str | None:
        """Category learned for this note's merchant, or None."""
        merchant = extract_merchant(note).lower()
        if not merchant:
            return None
        with self._lock:
            data = self._load()
            if merchant in data:
                return data[merchant]
            # Substring, then token overlap: "netflix ng" still matches
            # "netflix", and "starlink monthly" matches "starlink".
            for known, cat in data.items():
                if not known:
                    continue
                if known in merchant or merchant in known:
                    return cat
                known_tokens = {t for t in known.split() if len(t) >= 4}
                merch_tokens = {t for t in merchant.split() if len(t) >= 4}
                if known_tokens & merch_tokens:
                    return cat
        return None

    def learn(self, note: str, category: str) -> bool:
        """Remember merchant → category. Returns False when unusable."""
        merchant = extract_merchant(note).lower()
        category = (category or "").strip().lower()
        if not merchant or not category:
            return False
        with self._lock:
            self._load()[merchant] = category
            self._save()
        _log.debug("merchant memory: %r → %s", merchant, category)
        return True

    def forget(self, note: str) -> bool:
        merchant = extract_merchant(note).lower()
        with self._lock:
            if merchant in self._load():
                del self._load()[merchant]
                self._save()
                return True
        return False

    def all(self) -> dict[str, str]:
        with self._lock:
            return dict(self._load())


def categorize(note: str, memory: MerchantMemory | None = None) -> str:
    """Rule-based, offline categorizer. Returns a canonical category.

    Learned merchant mappings (``memory``) win over keyword rules —
    corrections stick.
    """
    if memory is not None:
        learned = memory.lookup(note)
        if learned:
            return learned
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
    id: str = ""  # stable record id ("txn_<hex>"); backfilled on read

    def __post_init__(self) -> None:
        if not self.id:
            self.id = "txn_" + uuid.uuid4().hex[:12]

    @property
    def merchant(self) -> str:
        """Best-effort merchant name from the note."""
        return extract_merchant(self.note)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
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
            id=str(data.get("id") or ""),
            ts=float(data.get("ts", 0)),
            amount_kobo=int(data.get("amount_kobo", 0)),
            category=str(data.get("category", "other")),
            note=str(data.get("note", "")),
            kind=str(data.get("kind", "spend")),
            source=str(data.get("source", "manual")),
        )


class Ledger:
    """Append-only JSON-lines transaction log. Thread-safe."""

    def __init__(self, path: str | Path | None = None,
                 memory: MerchantMemory | None = None) -> None:
        if path is None:
            path = Path.home() / ".nomorals" / "finance" / "ledger.jsonl"
        self.path = Path(path)
        self._lock = threading.RLock()
        self.memory = memory  # merchant learning; None = keyword rules only

    def _memory(self) -> MerchantMemory | None:
        if self.memory is None:
            # Lazily bind to the ledger's own finance dir.
            self.memory = MerchantMemory(
                self.path.parent / "merchant_memory.json")
        return self.memory

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
        category = ((category or "").strip().lower()
                    or categorize(note, self._memory()))
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

    # ── learning / search / import ───────────────────────────────────

    def learn(self, note: str, category: str) -> bool:
        """Teach the ledger: this merchant is always ``category``.

        Future ``log()`` calls with a matching merchant categorize
        automatically — corrections stick (Firefly III pattern).
        """
        return bool(self._memory()) and self._memory().learn(note, category)

    def search(self, text: str, *, limit: int = 50) -> list[Transaction]:
        """Full-text search over notes, merchants, and categories."""
        needle = (text or "").strip().lower()
        if not needle:
            return []
        out = []
        for t in self.transactions():
            hay = f"{t.note} {t.merchant} {t.category}".lower()
            if needle in hay:
                out.append(t)
                if len(out) >= limit:
                    break
        return out

    def duplicates(
        self,
        txns: list[Transaction] | None = None,
        *,
        window_days: float = 3.0,
    ) -> list[list[Transaction]]:
        """Groups of likely-duplicate transactions.

        Same amount + same merchant within ``window_days``. Used on import
        (Actual Budget's intake pattern) and as a ledger hygiene check.
        """
        txns = self.transactions() if txns is None else list(txns)
        buckets: dict[tuple[int, str], list[Transaction]] = {}
        for t in txns:
            key = (t.amount_kobo, t.merchant.lower())
            buckets.setdefault(key, []).append(t)
        groups = []
        for hits in buckets.values():
            if len(hits) < 2:
                continue
            hits.sort(key=lambda t: t.ts)
            cluster = [hits[0]]
            for t in hits[1:]:
                if t.ts - cluster[-1].ts <= window_days * 86400:
                    cluster.append(t)
                else:
                    if len(cluster) > 1:
                        groups.append(cluster)
                    cluster = [t]
            if len(cluster) > 1:
                groups.append(cluster)
        return groups

    def import_csv(
        self,
        path: str | Path,
        *,
        date_col: str = "",
        amount_col: str = "",
        note_col: str = "",
        kind: str = "spend",
        source: str = "csv",
        skip_duplicates: bool = True,
    ) -> dict[str, Any]:
        """Import a bank-statement CSV into the ledger.

        Header names are sniffed (date/amount/description/narration/…);
        explicit ``*_col`` overrides the sniffing. Amounts parse through
        :func:`parse_amount` (Nigerian shorthand ok). Duplicate groups
        (same amount + merchant within 3 days) are skipped when
        ``skip_duplicates`` is set. Returns an import report.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8-sig")
        try:
            dialect = csv.Sniffer().sniff(text[:4096])
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(text.splitlines(), dialect=dialect)
        # Normalize to lowercase once — sniffing and lookups are
        # case-insensitive from here on.
        reader.fieldnames = [(h or "").strip().lower()
                             for h in (reader.fieldnames or [])]
        headers = list(reader.fieldnames)

        def pick(want: str, *cands: str) -> str:
            if want:
                return want
            for c in cands:
                for h in headers:
                    if c in h:
                        return h
            return ""

        dcol = pick(date_col, "date", "transaction date", "value date")
        acol = pick(amount_col, "amount", "debit", "withdrawal", "value")
        ncol = pick(note_col, "narration", "description", "details",
                    "merchant", "note", "particulars")
        if not (dcol and acol and ncol):
            raise ValueError(
                f"couldn't sniff columns from {sorted(headers)} — pass "
                "date_col/amount_col/note_col explicitly")

        report = {"imported": 0, "skipped_duplicates": 0, "errors": []}
        staged: list[Transaction] = []
        existing = {(t.amount_kobo, t.merchant.lower(), int(t.ts // 86400))
                    for t in self.transactions()}
        for lineno, row in enumerate(reader, start=2):
            try:
                raw_amount = (row.get(acol) or "").strip()
                row_kind = kind
                # Sign decides spend vs income in mixed statements:
                # "-1,250.00" / "(1,250.00)" → spend.
                stripped = raw_amount.strip()
                is_negative = (stripped.startswith("-") or
                               (stripped.startswith("(")
                                and stripped.endswith(")")))
                unsigned = stripped.strip("+-").strip()
                if unsigned.startswith("(") and unsigned.endswith(")"):
                    unsigned = unsigned[1:-1]
                kobo = parse_amount(unsigned)
                if kobo is None:
                    # Bare decimal fallback ("1250.00").
                    try:
                        kobo = abs(int(Decimal(
                            unsigned.replace(",", "")) * 100))
                    except InvalidOperation:
                        raise ValueError(
                            f"unparseable amount {raw_amount!r}")
                kobo = abs(kobo)
                # Spend vs income: explicit sign wins, then column-name
                # hints ("Debit"/"Credit"), then the kind fallback.
                if is_negative:
                    row_kind = "spend"
                elif any(w in acol for w in ("credit", "deposit")):
                    row_kind = "income"
                elif any(w in acol for w in ("debit", "withdrawal")):
                    row_kind = "spend"
                elif kobo > 0 and kind == "spend" and "amount" in acol:
                    # Generic "Amount" column, positive value → money in.
                    row_kind = "income"
                note = (row.get(ncol) or "").strip()
                ts = _parse_csv_date((row.get(dcol) or "").strip())
                staged.append(Transaction(
                    ts=ts, amount_kobo=kobo, category="",
                    note=note, kind=row_kind, source=source))
            except Exception as exc:  # noqa: BLE001 - one bad row ≠ dead import
                report["errors"].append(f"line {lineno}: {exc}")
        if skip_duplicates:
            fresh = []
            for t in staged:
                day = int(t.ts // 86400)
                dup = any(
                    e[0] == t.amount_kobo and e[1] == t.merchant.lower()
                    and abs(e[2] - day) <= 3
                    for e in existing
                )
                if dup:
                    report["skipped_duplicates"] += 1
                else:
                    fresh.append(t)
                    existing.add((t.amount_kobo, t.merchant.lower(), day))
            staged = fresh
        for t in staged:
            self.log(t.amount_kobo, category=None, note=t.note, kind=t.kind,
                     ts=t.ts, source=source)
            report["imported"] += 1
        _log.info("csv import %s: %+d imported, %d dupes skipped, %d errors",
                  path.name, report["imported"],
                  report["skipped_duplicates"], len(report["errors"]))
        return report


def _parse_csv_date(text: str) -> float:
    """Best-effort statement date → epoch. Never raises."""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
                "%Y/%m/%d", "%d %b %Y", "%d %B %Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text.strip(), fmt).astimezone()\
                .timestamp()
        except ValueError:
            continue
    _log.debug("unparseable csv date %r — using now", text)
    return time.time()
