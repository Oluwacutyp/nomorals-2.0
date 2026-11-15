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
* CANCELS FOR REAL (owner order 2026-11-15, verbatim: "Asking it to
  cancel should be able to cancel the subscription I don't care
  how.... through Google play web or app store web or the websites").
  The NEVER CANCELS rule this module shipped with is REVERSED by that
  order. ``/subscriptions cancel`` now drives the real cancel flow on
  the billing page (Google Play web, App Store web, or the merchant's
  own site) through Devon's web-driving seam, and a cancel counts as
  done ONLY when the page state read back afterwards shows it
  cancelled. The draft cancel message survives as the FALLBACK: when
  a real cancel cannot run (no saved login, bot wall, changed page,
  unverified state), the command says so plainly and hands over the
  draft text, clearly labelled as the fallback. Cancelling never
  invents a route and never claims success from a clicked button
  alone.
* UNUSED = NO USAGE RECORDS. A subscription is flagged
  ``suspected_unused`` only when the caller supplies usage records and
  this merchant has none (or none recent). Absence of usage *evidence*
  is reported as exactly that — a suspicion, not a verdict.

Money: integer minor units (kobo/cents), mirroring ``ledger.py``.
Currency is a parameter (Nigeria default ``₦`` — never a boundary).
"""

from __future__ import annotations

import contextlib
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

__all__ = [
    "Subscription",
    "RenewalReminder",
    "UsageStore",
    "ChannelStore",
    "CancelJournal",
    "CancelOutcome",
    "CancelRoute",
    "CancelStep",
    "BillingChannel",
    "detect_subscriptions",
    "renewal_reminders",
    "draft_cancel_message",
    "monthly_equivalent",
    "audit",
    "render_audit",
    "render_reminders",
    "resolve_channel",
    "route_for",
    "run_cancel_flow",
    "verify_cancelled",
    "render_cancel_outcome",
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
                 "Cancel one for real with: /subscriptions cancel <merchant>.")
    lines.append("Set how a merchant bills with: /subscriptions channel "
                 "<merchant> <google_play|app_store|website> [url]")
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


# ── Real cancellation (owner order 2026-11-15) ──────────────────────────

BillingChannel = Literal["google_play", "app_store", "website", "unknown"]

CHANNELS: tuple[str, ...] = ("google_play", "app_store", "website", "unknown")

# Statement-descriptor evidence, read from the transaction text itself
# (merchant / description / narration). Never guessed from a name.
_APPLE_DESCRIPTORS = ("apple.com/bill", "apl*itunes", "itunes.com/bill")
_GOOGLE_DESCRIPTORS = ("google play", "googleplay", "google *", "g.co/play")

# A date is only an end date when it sits on a page that already showed
# a cancellation marker — _extract_ends_on is never called on its own.
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_ISO_DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_LONG_DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|"
    r"Oct|Nov|Dec)\s+(\d{1,2}),\s+(20\d{2})\b")


def _txn_text(txn: Any) -> str:
    """Searchable text of one transaction row (merchant/description/narration)."""
    parts: list[str] = []
    if isinstance(txn, Mapping):
        for key in ("merchant", "description", "narration"):
            value = txn.get(key)
            if value:
                parts.append(str(value))
    else:
        for attr in ("merchant", "description", "narration"):
            value = getattr(txn, attr, None)
            if value:
                parts.append(str(value))
    return " ".join(parts).lower()


class ChannelStore:
    """Durable billing-channel records per merchant (+ optional cancel URL).

    Lives beside ``UsageStore`` under ``<home>/finance/``. Tolerant load:
    a corrupt or partial file behaves as an empty store, never an
    exception. Atomic save (temp file + rename), exactly like UsageStore.
    """

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = (Path.home() / ".nomorals" / "finance"
                    / "subscription_channels.json")
        self.path = Path(path)
        self._data: dict[str, dict[str, str]] = self._load()

    def _load(self) -> dict[str, dict[str, str]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, Mapping):
            return {}
        data: dict[str, dict[str, str]] = {}
        for key, value in raw.items():
            if not isinstance(value, Mapping):
                continue
            channel = value.get("channel")
            if channel not in CHANNELS:
                continue
            row: dict[str, str] = {"channel": str(channel)}
            url = value.get("cancel_url")
            if url:
                row["cancel_url"] = str(url)
            data[str(key).strip().lower()] = row
        return data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(self.path)

    def set_channel(self, merchant: str, channel: str,
                    cancel_url: str | None = None) -> None:
        """Record how a merchant bills (owner-set). Raises ValueError on a
        channel that is not one of ``CHANNELS``."""
        if channel not in CHANNELS:
            raise ValueError(
                f"unknown billing channel {channel!r} "
                f"(one of: {', '.join(CHANNELS)})")
        row: dict[str, str] = {"channel": channel}
        if cancel_url:
            row["cancel_url"] = cancel_url.strip()
        self._data[_norm_merchant(merchant)] = row
        self._save()

    def channel(self, merchant: str) -> str | None:
        row = self._data.get(_norm_merchant(merchant))
        return row["channel"] if row else None

    def cancel_url(self, merchant: str) -> str | None:
        row = self._data.get(_norm_merchant(merchant))
        return row.get("cancel_url") if row else None


def resolve_channel(merchant: str,
                    transactions: Iterable[Any] | None = None,
                    store: ChannelStore | None = None) -> str:
    """How a merchant bills, from evidence only — never guessed.

    (1) a stored channel (owner-set) wins; (2) a billing descriptor in
    the supplied transaction rows (``APPLE.COM/BILL`` / ``APL*ITUNES``
    → App Store; ``GOOGLE *<merchant>`` / ``GOOGLE PLAY`` → Google
    Play); (3) ``"unknown"`` when nothing says so.
    """
    if store is not None:
        stored = store.channel(merchant)
        if stored is not None:
            return stored
    key = _norm_merchant(merchant)
    for txn in transactions or []:
        try:
            name, _amount, _when, _currency = _txn_fields(txn)
        except ValueError:
            continue
        if _norm_merchant(name) != key:
            continue
        text = _txn_text(txn)
        if any(mark in text for mark in _APPLE_DESCRIPTORS):
            return "app_store"
        if any(mark in text for mark in _GOOGLE_DESCRIPTORS):
            return "google_play"
    return "unknown"


@dataclass(frozen=True)
class CancelStep:
    """One click (or wait) inside a cancel flow.

    ``target`` is visible text (or a CSS selector when it looks like
    one) and supports ``{merchant}`` formatting. A required step that
    fails ends the attempt as ``failed``; an optional step that fails
    is skipped and the page state afterwards arbitrates.
    """

    action: Literal["click", "wait_text"]
    target: str
    required: bool = True


@dataclass(frozen=True)
class CancelRoute:
    """Where and how one subscription is cancelled, per channel."""

    channel: str
    service: str          # vault service name for the sign-in (google/apple/…)
    session_name: str     # browser session the drive runs in
    url: str              # the cancel page to open
    login_url: str        # where the sign-in happens when the session is cold
    login_success_text: str  # text that must appear after a fresh sign-in
    steps: tuple[CancelStep, ...]
    cancelled_markers: tuple[str, ...]  # page text proving it is cancelled
    already_markers: tuple[str, ...] = ()


PLAY_SUBSCRIPTIONS = "https://play.google.com/store/account/subscriptions"
APPLE_SUBSCRIPTIONS = ("https://account.apple.com/account/manage/section/"
                       "subscriptions")

KNOWN_ROUTES: dict[str, CancelRoute] = {
    "google_play": CancelRoute(
        channel="google_play",
        service="google",
        session_name="subcancel-google-play",
        url=PLAY_SUBSCRIPTIONS,
        login_url="https://accounts.google.com/",
        login_success_text="Subscriptions",
        steps=(
            CancelStep("click", "{merchant}"),
            CancelStep("click", "Cancel subscription"),
            CancelStep("click", "Prefer not to answer", required=False),
            CancelStep("click", "Continue", required=False),
        ),
        cancelled_markers=("canceled", "cancelled", "ends on"),
    ),
    "app_store": CancelRoute(
        channel="app_store",
        service="apple",
        session_name="subcancel-apple",
        url=APPLE_SUBSCRIPTIONS,
        login_url="https://account.apple.com/sign-in",
        login_success_text="Subscriptions",
        steps=(
            CancelStep("click", "{merchant}"),
            CancelStep("click", "Cancel Subscription"),
            CancelStep("click", "Confirm", required=False),
        ),
        cancelled_markers=("expires on", "already canceled",
                           "already cancelled", "cancelled", "canceled",
                           "subscription ends"),
        already_markers=("already canceled", "already cancelled"),
    ),
    "netflix": CancelRoute(
        channel="website",
        service="netflix",
        session_name="subcancel-netflix",
        url="https://www.netflix.com/cancelplan",
        login_url="https://www.netflix.com/login",
        login_success_text="Cancel",
        steps=(
            CancelStep("click", "Cancel"),
            CancelStep("click", "Finish Cancellation"),
        ),
        cancelled_markers=("will close on", "membership will end",
                           "cancellation confirmed"),
        already_markers=("already canceled", "already cancelled"),
    ),
    "spotify": CancelRoute(
        channel="website",
        service="spotify",
        session_name="subcancel-spotify",
        url="https://www.spotify.com/account",
        login_url="https://accounts.spotify.com/login",
        login_success_text="Manage your plan",
        steps=(
            CancelStep("click", "Cancel subscription"),
            CancelStep("click", "Continue to cancel", required=False),
            CancelStep("click", "Continue to cancel", required=False),
            CancelStep("click", "Yes, cancel"),
        ),
        cancelled_markers=("switch to free", "switching to free",
                           "premium ends", "your premium will end",
                           "cancelled", "canceled"),
    ),
}

# Merchants with a verified website cancel route, keyed by normalised name.
KNOWN_MERCHANT_ROUTES: dict[str, str] = {
    "netflix": "netflix",
    "spotify": "spotify",
}


def _generic_website_route(sub: Subscription, url: str) -> CancelRoute:
    """A merchant site with a stored cancel URL: try the cancel controls
    a typical account page carries, in order. Every step is optional —
    the page state afterwards is the only arbiter."""
    slug = re.sub(r"[^a-z0-9]+", "-", _norm_merchant(sub.merchant)).strip("-")
    return CancelRoute(
        channel="website",
        service=slug or "website",
        session_name=f"subcancel-{slug or 'website'}",
        url=url,
        login_url=url,
        login_success_text="Cancel",
        steps=(
            CancelStep("click", "Cancel subscription", required=False),
            CancelStep("click", "Cancel", required=False),
            CancelStep("click", "Cancel plan", required=False),
            CancelStep("click", "Confirm", required=False),
            CancelStep("click", "Yes, cancel", required=False),
            CancelStep("click", "Finish Cancellation", required=False),
        ),
        cancelled_markers=("cancelled", "canceled", "will close",
                           "ends on", "switch to free",
                           "subscription ends", "membership will end"),
        already_markers=("already canceled", "already cancelled"),
    )


def route_for(sub: Subscription, channel: str,
              store: ChannelStore | None = None) -> CancelRoute | None:
    """The cancel route for a subscription's channel, or None.

    None means NO route was attempted and none is claimed: unknown
    channel, or a website channel whose merchant has neither a known
    route nor a stored cancel URL. The caller answers that case with a
    no-route reply + the draft fallback, never a guess.
    """
    if channel == "google_play":
        return KNOWN_ROUTES["google_play"]
    if channel == "app_store":
        return KNOWN_ROUTES["app_store"]
    if channel == "website":
        key = KNOWN_MERCHANT_ROUTES.get(_norm_merchant(sub.merchant))
        if key is not None:
            return KNOWN_ROUTES[key]
        url = store.cancel_url(sub.merchant) if store is not None else None
        if url:
            return _generic_website_route(sub, url)
    return None


@dataclass
class CancelOutcome:
    """What one cancel attempt did, in plain language."""

    merchant: str
    channel: str
    status: str  # cancelled | already_cancelled | failed | login_missing
                 # | unverified | no_route
    route_url: str
    ends_on: date | None = None
    detail: str = ""
    attempted_at: datetime = field(default_factory=datetime.now)
    draft: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "merchant": self.merchant,
            "channel": self.channel,
            "status": self.status,
            "route_url": self.route_url,
            "ends_on": self.ends_on.isoformat() if self.ends_on else None,
            "detail": self.detail,
            "attempted_at": self.attempted_at.isoformat(),
            "draft": self.draft,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> CancelOutcome:
        ends_on = row.get("ends_on")
        return cls(
            merchant=str(row.get("merchant", "")),
            channel=str(row.get("channel", "unknown")),
            status=str(row.get("status", "failed")),
            route_url=str(row.get("route_url", "")),
            ends_on=date.fromisoformat(ends_on) if ends_on else None,
            detail=str(row.get("detail", "")),
            attempted_at=(datetime.fromisoformat(row["attempted_at"])
                          if row.get("attempted_at") else datetime.now()),
            draft=str(row.get("draft", "")),
        )


class CancelJournal:
    """Durable record of cancel attempts (subscription, channel, route
    attempted, result). Tolerant load, atomic save — mirrors
    UsageStore/ChannelStore. Never stores credentials or page contents."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = (Path.home() / ".nomorals" / "finance"
                    / "subscription_cancels.json")
        self.path = Path(path)
        self._rows: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        return [row for row in raw
                if isinstance(row, Mapping)
                and row.get("merchant") and row.get("status")]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._rows, indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(self.path)

    def record(self, outcome: CancelOutcome) -> None:
        self._rows.append(outcome.to_dict())
        self._save()

    def all(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def last_for(self, merchant: str) -> dict[str, Any] | None:
        key = _norm_merchant(merchant)
        for row in reversed(self._rows):
            if _norm_merchant(str(row.get("merchant", ""))) == key:
                return row
        return None


def _extract_ends_on(text: str) -> date | None:
    """The end date on a page that already showed a cancellation marker
    (``2026-12-01`` or ``Dec 1, 2026``). None when the page names none."""
    match = _ISO_DATE.search(text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)),
                        int(match.group(3)))
        except ValueError:
            pass
    match = _LONG_DATE.search(text)
    if match:
        month = _MONTHS.get(match.group(1).lower())
        if month is not None:
            try:
                return date(int(match.group(3)), month,
                            int(match.group(2)))
            except ValueError:
                pass
    return None


def verify_cancelled(page_text: str, route: CancelRoute) -> tuple[str, date | None]:
    """Decide the outcome from the page text AFTER a flow ran.

    Returns (status, ends_on): ``"already_cancelled"`` when the page
    says it was already cancelled, ``"cancelled"`` when a cancellation
    marker is present (with the end date when the page names one),
    ``"unverified"`` otherwise. Verification is the ONLY arbiter of
    success — no success claim ever comes from a clicked button alone.
    """
    text = (page_text or "").lower()
    if any(mark in text for mark in route.already_markers):
        return "already_cancelled", _extract_ends_on(page_text or "")
    if any(mark in text for mark in route.cancelled_markers):
        return "cancelled", _extract_ends_on(page_text or "")
    return "unverified", None


def run_cancel_flow(driver: Any, sub: Subscription, route: CancelRoute,
                    *, settle_s: float = 2.0) -> CancelOutcome:
    """Drive one cancel flow on a tab and verify the page afterwards.

    ``driver`` is duck-typed — the exact interface a real rendered tab
    and a plain tab both carry: ``navigate(url)`` (raises on failure),
    ``click(target)`` (visible text or selector; raises when the target
    is not on the page), ``page_text() -> str``, ``close()``. The
    caller owns the driver (sign-in happens before this runs); this
    function only closes it, best-effort, at the end.
    """
    draft = draft_cancel_message(sub)

    def outcome(status: str, detail: str,
                ends_on: date | None = None) -> CancelOutcome:
        return CancelOutcome(
            merchant=sub.merchant, channel=route.channel, status=status,
            route_url=route.url, ends_on=ends_on, detail=detail,
            draft=draft if status not in ("cancelled", "already_cancelled")
            else "")

    try:
        driver.navigate(route.url)
    except Exception as exc:
        with contextlib.suppress(Exception):
            driver.close()
        return outcome("failed", f"could not open {route.url}: {exc}")
    for step in route.steps:
        target = step.target.format(merchant=sub.merchant)
        try:
            if step.action == "click":
                driver.click(target)
            else:
                driver.page_text()  # wait_text: re-read below; no-op here
        except Exception as exc:
            if step.required:
                with contextlib.suppress(Exception):
                    driver.close()
                return outcome(
                    "failed",
                    f"the page did not show {target!r} — the cancel flow "
                    f"may have changed: {exc}")
            continue
    if settle_s > 0:
        time.sleep(settle_s)
    text = driver.page_text()
    status, ends_on = verify_cancelled(text, route)
    if status == "unverified" and settle_s > 0:
        # A slow page may still be settling — one re-read, then decide.
        time.sleep(settle_s)
        text = driver.page_text()
        status, ends_on = verify_cancelled(text, route)
    with contextlib.suppress(Exception):
        driver.close()
    if status == "cancelled":
        return outcome("cancelled",
                       "verified on the page just now", ends_on)
    if status == "already_cancelled":
        return outcome("already_cancelled",
                       "the page showed it was already cancelled", ends_on)
    return outcome(
        "unverified",
        "the page after the flow did not show a cancellation — "
        "still subscribed as far as I can tell")


def _fmt_day(day: date) -> str:
    return f"{day:%b} {day.day}, {day:%Y}"


def render_cancel_outcome(outcome: CancelOutcome, sub: Subscription) -> str:
    """The plain-words reply for one cancel attempt.

    Success states carry no draft; every other state names what
    happened, says the subscription is still standing, and hands over
    the draft cancel text, clearly labelled as the fallback.
    """
    name = outcome.merchant or sub.merchant
    if outcome.status == "cancelled":
        tail = (f" Access runs until {_fmt_day(outcome.ends_on)}."
                if outcome.ends_on else " No further charges.")
        return (f"cancelled: {name} — {outcome.channel} route, verified "
                f"on the page just now.{tail}")
    if outcome.status == "already_cancelled":
        return (f"already cancelled: {name} — the page showed it was "
                f"already cancelled, so nothing was charged again.")
    lines = [
        f"couldn't cancel {name} for real: {outcome.detail}",
        "You're still subscribed as far as I can tell.",
    ]
    if outcome.draft:
        lines.append("")
        lines.append("Draft cancel text — the fallback, send it yourself:")
        lines.append("")
        lines.append(outcome.draft)
    return "\n".join(lines)
