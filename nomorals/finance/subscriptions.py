"""Subscription auditor — renewals, cost, unused detection, reminders.

MORE_TO_ADD gap 9. NEW file in ``nomorals/finance/``; no existing finance
file is modified. (``ledger.py`` / ``insights.py`` / ``commands.py`` /
``tools.py`` mention "subscription" only as a category keyword / budget
line — none of them detect recurring subscriptions from transactions.)

Rules this module lives by:

* NEVER INVENTS. Subscriptions are detected ONLY from the finance
  transactions the caller supplies (e.g. ``Ledger`` records). No
  hardcoded merchant list, no assumed subscriptions, no fabricated
  renewals. No transactions in → no subscriptions out.
* NEVER CANCELS. There is deliberately NO function in this module that
  performs a cancellation — no ``cancel_subscription``, no
  ``cancel()``, no API call, no side effect. The module produces
  renewal/cancel REMINDERS and DRAFT cancel messages (text the owner
  can send themselves). Cancelling is the owner's action, taken after
  their explicit confirmation, outside this module.
* UNUSED = NO USAGE RECORDS. A subscription is flagged
  ``suspected_unused`` only when the caller supplies usage records and
  this merchant has none (or none recent). Absence of usage *evidence*
  is reported as exactly that — a suspicion, not a verdict.

Money: integer minor units (kobo/cents), mirroring ``ledger.py``.
Currency is a parameter (Nigeria default ``₦`` — never a boundary).
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "Subscription",
    "RenewalReminder",
    "UsageStore",
    "detect_subscriptions",
    "renewal_reminders",
    "draft_cancel_message",
    "monthly_equivalent",
    "audit",
    "render_audit",
    "render_reminders",
]

DEFAULT_CURRENCY = "₦"


@dataclass
class Subscription:
    """One detected recurring subscription (from supplied transactions)."""

    merchant: str
    occurrences: int
    first_seen: date
    last_seen: date
    cost_minor: int  # cost per period, integer minor units
    period_days: float  # median gap between occurrences
    currency: str = DEFAULT_CURRENCY
    estimated_next_renewal: date | None = None
    suspected_unused: bool = False
    unused_reason: str = ""
    transaction_refs: list[Any] = field(default_factory=list)

    @property
    def yearly_cost_minor(self) -> int:
        """Annualised cost from the detected cadence (never invented)."""
        if self.period_days <= 0:
            return self.cost_minor
        return round(self.cost_minor * (365.0 / self.period_days))


@dataclass(frozen=True)
class RenewalReminder:
    """A reminder — not an action. Nothing here cancels anything."""

    merchant: str
    renewal_date: date
    cost_minor: int
    currency: str
    days_until: int
    suspected_unused: bool
    draft_message: str


def _norm_merchant(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(ltd|limited|inc|llc|ng|nigeria)\b\.?", "", text).strip()
    return text or raw.strip().lower()


def _txn_fields(txn: Any) -> tuple[str, int, date, str]:
    """Duck-typed read of a supplied transaction.

    Accepts dicts (``merchant``/``description``, ``amount_minor``/``amount``,
    ``date``) or ``Ledger.Transaction``-shaped objects (``merchant``,
    ``amount_minor``/``amount``/``amount_kobo``, ``date``/``ts``). Raises
    ``ValueError`` on a malformed row instead of guessing.
    """
    if isinstance(txn, Mapping):
        merchant = txn.get("merchant") or txn.get("description") or txn.get("narration") or ""
        amount = txn.get("amount_minor", txn.get("amount"))
        when = txn.get("date") or txn.get("txn_date")
        currency = txn.get("currency", DEFAULT_CURRENCY)
    else:
        merchant = getattr(txn, "merchant", None) or getattr(txn, "description", None) or getattr(txn, "narration", None) or ""
        amount = getattr(txn, "amount_minor", None)
        if amount is None:
            amount = getattr(txn, "amount", None)
        if amount is None:
            amount = getattr(txn, "amount_kobo", None)
        when = getattr(txn, "date", None) or getattr(txn, "txn_date", None)
        if when is None:
            ts = getattr(txn, "ts", None)
            if ts is not None:
                when = datetime.fromtimestamp(float(ts)).date()
        currency = getattr(txn, "currency", DEFAULT_CURRENCY)
    if not merchant or amount is None or when is None:
        raise ValueError(f"transaction missing merchant/amount/date: {txn!r}")
    if isinstance(when, str):
        when = date.fromisoformat(when[:10])
    return str(merchant), int(amount), when, str(currency)


def detect_subscriptions(
    transactions: Iterable[Any],
    *,
    min_occurrences: int = 2,
    usage_records: Mapping[str, date | None] | None = None,
    unused_after_days: int = 60,
    as_of: date | None = None,
) -> list[Subscription]:
    """Detect recurring subscriptions from SUPPLIED transactions only.

    Groups by normalised merchant; a merchant qualifies when it appears
    ``>= min_occurrences`` times with a stable cost and a roughly
    regular cadence. ``usage_records`` maps merchant (any casing) →
    last-used date (or ``None`` = never used). A subscription with no
    usage record at all is flagged ``suspected_unused`` with reason
    "no usage records supplied for this merchant" — the flag reports
    missing evidence, never a verdict. Empty input → empty output.
    """
    as_of = as_of or date.today()
    groups: dict[str, list[tuple[int, date, str, Any]]] = {}
    display: dict[str, str] = {}
    for txn in transactions:
        merchant, amount, when, currency = _txn_fields(txn)
        key = _norm_merchant(merchant)
        groups.setdefault(key, []).append((amount, when, currency, txn))
        display.setdefault(key, merchant.strip())
    usage_norm = {(_norm_merchant(k)): v for k, v in (usage_records or {}).items()}

    out: list[Subscription] = []
    for key, rows in groups.items():
        if len(rows) < min_occurrences:
            continue
        rows.sort(key=lambda r: r[1])
        amounts = [r[0] for r in rows]
        # Cost must be stable: median amount, and every charge within 25%
        # of it — otherwise this is not one subscription, just a merchant.
        median_amount = int(statistics.median(amounts))
        if median_amount <= 0:
            continue
        if any(abs(a - median_amount) > 0.25 * median_amount for a in amounts):
            continue
        gaps = [(rows[i + 1][1] - rows[i][1]).days for i in range(len(rows) - 1)]
        gaps = [g for g in gaps if g > 0]
        if not gaps:
            continue
        period = float(statistics.median(gaps))
        last_seen = rows[-1][1]
        step = max(1, round(period))
        nxt = last_seen + timedelta(days=step)
        if nxt <= as_of:
            # Roll forward by whole periods (O(1)): an estimate in the
            # past is unreachable for renewal reminders — the real next
            # renewal is the first occurrence after ``as_of``.
            k = (as_of - nxt).days // step + 1
            nxt += timedelta(days=step * k)
        last_used = usage_norm.get(key, "MISSING")
        suspected = False
        reason = ""
        if last_used == "MISSING":
            suspected = True
            reason = "no usage records supplied for this merchant"
        elif last_used is None:
            suspected = True
            reason = "usage records show never used"
        elif (as_of - last_used).days > unused_after_days:
            suspected = True
            reason = f"no usage in {(as_of - last_used).days} days (last used {last_used})"
        out.append(
            Subscription(
                merchant=display[key],
                occurrences=len(rows),
                first_seen=rows[0][1],
                last_seen=last_seen,
                cost_minor=median_amount,
                period_days=period,
                currency=rows[0][2],
                estimated_next_renewal=nxt,
                suspected_unused=suspected,
                unused_reason=reason,
                transaction_refs=[r[3] for r in rows],
            )
        )
    return sorted(out, key=lambda s: s.estimated_next_renewal or date.max)


def monthly_equivalent(sub: Subscription) -> int:
    """Monthly-equivalent minor-unit cost for one subscription."""
    if sub.period_days <= 0:
        return sub.cost_minor
    return round(sub.cost_minor * (30.0 / sub.period_days))


def draft_cancel_message(sub: Subscription) -> str:
    """Draft cancel text for the owner to send THEMSELVES.

    Text only — this function performs no cancellation, makes no API
    call, and changes no state. The owner reviews, confirms, and acts.
    """
    return (
        f"Hello, please cancel my subscription to {sub.merchant} "
        f"(~{sub.currency}{sub.cost_minor / 100:,.2f} per period, next renewal "
        f"{sub.estimated_next_renewal}). Please confirm the cancellation and "
        f"that no further charges will be taken. Thank you."
    )


def renewal_reminders(
    subscriptions: Iterable[Subscription],
    *,
    as_of: date | None = None,
    days_ahead: int = 7,
) -> list[RenewalReminder]:
    """Renewal + cancel reminders for subscriptions renewing within window.

    Reminders only — nothing here cancels, charges, or contacts anyone.
    Each reminder carries a :func:`draft_cancel_message` the owner may
    choose to use after their own confirmation.
    """
    as_of = as_of or date.today()
    out: list[RenewalReminder] = []
    for sub in subscriptions:
        if sub.estimated_next_renewal is None:
            continue
        days_until = (sub.estimated_next_renewal - as_of).days
        if 0 <= days_until <= days_ahead:
            out.append(
                RenewalReminder(
                    merchant=sub.merchant,
                    renewal_date=sub.estimated_next_renewal,
                    cost_minor=sub.cost_minor,
                    currency=sub.currency,
                    days_until=days_until,
                    suspected_unused=sub.suspected_unused,
                    draft_message=draft_cancel_message(sub),
                )
            )
    return sorted(out, key=lambda r: r.renewal_date)


class UsageStore:
    """Durable last-used dates per merchant — the unused flag's evidence.

    JSON: {normalised merchant: ISO date}. Tolerant load — a missing or
    corrupt file reads as empty, bad rows are skipped, never fatal.
    Atomic tmp+replace save after every mutation, mirroring
    ``MerchantMemory``. A merchant with no record here is exactly what
    the auditor reports: "no usage records supplied", never a verdict.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".nomorals" / "finance" / "subscription_usage.json"
        self.path = Path(path)
        self._data = self._load()

    def _load(self) -> dict[str, date]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, date] = {}
        for key, value in raw.items():
            try:
                out[_norm_merchant(str(key))] = date.fromisoformat(str(value)[:10])
            except ValueError:
                continue
        return out

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({k: v.isoformat() for k, v in sorted(self._data.items())},
                       indent=1, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def record(self, merchant: str, when: date | None = None) -> date:
        """Record a last-used date (today when omitted)."""
        when = when or date.today()
        self._data[_norm_merchant(merchant)] = when
        self._save()
        return when

    def last_used(self, merchant: str) -> date | None:
        return self._data.get(_norm_merchant(merchant))

    def all(self) -> dict[str, date]:
        return dict(self._data)


def audit(
    transactions: Iterable[Any],
    *,
    usage_records: Mapping[str, date | None] | None = None,
    as_of: date | None = None,
    days_ahead: int = 7,
) -> tuple[list[Subscription], list[RenewalReminder]]:
    """One shared path for the chat command and the tool: detect + remind.

    Detection runs on the supplied transactions only — never invented.
    Reminders are text only; nothing here cancels, charges, or contacts
    anyone.
    """
    as_of = as_of or date.today()
    subs = detect_subscriptions(transactions, usage_records=usage_records,
                                as_of=as_of)
    reminders = renewal_reminders(subs, as_of=as_of, days_ahead=days_ahead)
    return subs, reminders


def _money(minor: int, currency: str) -> str:
    return f"{currency}{minor / 100:,.2f}"


def render_audit(
    subscriptions: list[Subscription],
    *,
    as_of: date | None = None,
    window_days: int = 365,
) -> str:
    """The audit as text: cost, cadence, next renewal, unused flags."""
    as_of = as_of or date.today()
    if not subscriptions:
        return (f"no recurring subscriptions found in the last {window_days} days — "
                "nothing charged at least twice on a steady cadence. "
                "Log spends with /spend and they show up here once a charge repeats.")
    currency = subscriptions[0].currency
    monthly = sum(monthly_equivalent(s) for s in subscriptions)
    yearly = sum(s.yearly_cost_minor for s in subscriptions)
    lines = [f"{len(subscriptions)} subscription(s) — "
             f"~{_money(monthly, currency)} monthly, ~{_money(yearly, currency)} yearly:"]
    for sub in subscriptions:
        nxt = sub.estimated_next_renewal
        when = (f"next renewal {nxt} ({(nxt - as_of).days}d)"
                if nxt is not None else "next renewal unknown")
        flag = (f" — suspected unused: {sub.unused_reason}"
                if sub.suspected_unused else "")
        lines.append(
            f"• {sub.merchant} — {_money(sub.cost_minor, sub.currency)} "
            f"every ~{sub.period_days:.0f}d, {when}, "
            f"~{_money(sub.yearly_cost_minor, sub.currency)}/yr "
            f"({sub.occurrences} charges){flag}")
    lines.append("Mark one used with: /subscriptions used <merchant>. "
                 "Draft a cancel note with: /subscriptions cancel <merchant>.")
    return "\n".join(lines)


def render_reminders(reminders: list[RenewalReminder], *, days_ahead: int = 7) -> str:
    """Renewal reminders as text — reminders only, never actions."""
    if not reminders:
        return f"nothing renewing in the next {days_ahead} day(s)."
    lines = [f"{len(reminders)} renewal(s) in the next {days_ahead} day(s):"]
    for reminder in reminders:
        flag = " — suspected unused" if reminder.suspected_unused else ""
        lines.append(
            f"• {reminder.merchant} — renews {reminder.renewal_date} "
            f"({reminder.days_until}d), "
            f"{_money(reminder.cost_minor, reminder.currency)}{flag}")
        if reminder.suspected_unused:
            lines.append(f"  draft cancel: {reminder.draft_message}")
    return "\n".join(lines)
