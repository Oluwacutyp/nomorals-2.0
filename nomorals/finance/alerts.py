"""Native price/money alert engine — Devon's own watchtower.

Kinds of alerts:
  price_above / price_below  — an instrument crosses a price
                              (target = "BTC" / "BTC/USDT" etc.)
  pct_change                — 24h move magnitude exceeds a % (target = symbol)
  budget_pct                — a category's budget usage crosses a %
                              (target = category name, threshold = 0.8)
  fx_rate                   — USD→NGN (or EUR/GBP→NGN) crosses a rate

Alerts are evaluated by ``evaluate_alerts`` (pure-ish: takes price and
notifier callables so tests run offline) and by the scheduler hook
``ensure_alerts_job`` which runs every 15 minutes and delivers through
the Notifier. One-shot alerts auto-disable after firing; repeating
alerts re-arm only after the condition clears (no spam while the price
sits above the line).

Prices come from ``nomorals.integrations.market_data.quote`` (keyless:
Binance/CoinGecko) — external APIs are the fallback data source, never
the product: the alert state, dedup, and delivery are Devon's own.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .budgets import BudgetStore, budget_status, finance_paths, month_key
from .ledger import Ledger

_log = get_logger("nomorals.finance")

__all__ = [
    "Alert",
    "AlertStore",
    "add_alert",
    "evaluate_alerts",
    "ensure_alerts_job",
    "sync_bill_alerts",
    "render_alerts",
    "ALERTS_CRON",
    "KIND_PRICE_ABOVE",
    "KIND_PRICE_BELOW",
    "KIND_PCT_CHANGE",
    "KIND_BUDGET_PCT",
    "KIND_FX_RATE",
    "KIND_BILL_DUE",
]

KIND_PRICE_ABOVE = "price_above"
KIND_PRICE_BELOW = "price_below"
KIND_PCT_CHANGE = "pct_change"
KIND_BUDGET_PCT = "budget_pct"
KIND_FX_RATE = "fx_rate"
#: A detected recurring charge is due within ``threshold`` days.
#: target = the charge's note pattern (from detect_recurring).
KIND_BILL_DUE = "bill_due"

KINDS = (
    KIND_PRICE_ABOVE, KIND_PRICE_BELOW, KIND_PCT_CHANGE,
    KIND_BUDGET_PCT, KIND_FX_RATE, KIND_BILL_DUE,
)

#: How often the scheduler evaluates alerts.
ALERTS_CRON = "*/15 * * * *"
ALERTS_TASK_ID = "finance.alerts_evaluate"
ALERTS_ACTION = "finance.alerts_evaluate"


@dataclass
class Alert:
    """One watch condition."""

    id: str
    kind: str
    target: str          # symbol, category, or "USD" (fx base)
    threshold: float     # price / % / rate depending on kind
    market: str = "crypto"  # market_data market for price kinds
    note: str = ""
    one_shot: bool = True
    enabled: bool = True
    created_at: float = 0.0
    last_fired_at: float | None = None
    fire_count: int = 0
    # re-arm state: a repeating alert fires again only after the
    # condition was observed clear at least once.
    was_clear: bool = True
    # YNAB-style snooze: skip evaluation until this epoch.
    snooze_until: float | None = None
    # bill_due bookkeeping: the due date this alert last fired for, so a
    # monthly bill doesn't re-fire every 15 minutes until it's paid.
    last_due_ts: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Alert":
        kwargs = {k: data.get(k) for k in asdict(cls(
            id="", kind="", target="", threshold=0.0)).keys()}
        kwargs["id"] = str(kwargs.get("id") or "")
        kwargs["kind"] = str(kwargs.get("kind") or "")
        kwargs["target"] = str(kwargs.get("target") or "")
        try:
            kwargs["threshold"] = float(kwargs.get("threshold") or 0)
        except (TypeError, ValueError):
            kwargs["threshold"] = 0.0
        kwargs["market"] = str(kwargs.get("market") or "crypto")
        kwargs["note"] = str(kwargs.get("note") or "")
        kwargs["one_shot"] = bool(kwargs.get("one_shot", True))
        kwargs["enabled"] = bool(kwargs.get("enabled", True))
        kwargs["created_at"] = float(kwargs.get("created_at") or 0)
        kwargs["last_fired_at"] = kwargs.get("last_fired_at")
        kwargs["fire_count"] = int(kwargs.get("fire_count") or 0)
        kwargs["was_clear"] = bool(kwargs.get("was_clear", True))
        kwargs["snooze_until"] = kwargs.get("snooze_until")
        kwargs["last_due_ts"] = kwargs.get("last_due_ts")
        return cls(**kwargs)

    @property
    def snoozed(self) -> bool:
        return bool(self.snooze_until) and self.snooze_until > time.time()

    def describe(self) -> str:
        if self.kind == KIND_PRICE_ABOVE:
            return f"{self.target} above {self.threshold:,.2f}"
        if self.kind == KIND_PRICE_BELOW:
            return f"{self.target} below {self.threshold:,.2f}"
        if self.kind == KIND_PCT_CHANGE:
            return f"{self.target} moves ±{self.threshold:.1f}% in 24h"
        if self.kind == KIND_BUDGET_PCT:
            return f"{self.target} budget hits {self.threshold:.0%}"
        if self.kind == KIND_FX_RATE:
            return f"{self.target}→NGN crosses {self.threshold:,.0f}"
        if self.kind == KIND_BILL_DUE:
            return (f"{self.target} due within {self.threshold:.0f}d "
                    f"(recurring bill)")
        return f"{self.kind} {self.target} {self.threshold}"


class AlertStore:
    """Persistent alert registry. JSON file. Thread-unsafe on purpose —
    the scheduler is the only writer; chat adds go through the same
    process."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            _, budgets_path = finance_paths(None)
            path = Path(budgets_path).parent / "alerts.json"
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

    def add(self, alert: Alert) -> Alert:
        data = self._load()
        data[alert.id] = alert.to_dict()
        self._save(data)
        return alert

    def remove(self, alert_id: str) -> bool:
        data = self._load()
        if alert_id not in data:
            return False
        del data[alert_id]
        self._save(data)
        return True

    def get(self, alert_id: str) -> Alert | None:
        raw = self._load().get(alert_id or "")
        if not raw:
            return None
        try:
            return Alert.from_dict(raw)
        except Exception:  # noqa: BLE001
            return None

    def list(self, *, enabled_only: bool = False) -> list[Alert]:
        out = []
        for raw in self._load().values():
            try:
                a = Alert.from_dict(raw)
            except Exception:  # noqa: BLE001
                continue
            if enabled_only and not a.enabled:
                continue
            out.append(a)
        return sorted(out, key=lambda a: a.created_at)

    def update(self, alert: Alert) -> None:
        data = self._load()
        data[alert.id] = alert.to_dict()
        self._save(data)

    def snooze(self, alert_id: str, days: float = 7.0) -> bool:
        """Pause an alert without deleting it (YNAB's snooze, generalized)."""
        a = self.get(alert_id)
        if a is None:
            return False
        a.snooze_until = time.time() + float(days) * 86400
        self.update(a)
        return True


def add_alert(
    kind: str,
    target: str,
    threshold: float,
    *,
    market: str = "crypto",
    note: str = "",
    one_shot: bool = True,
    store: AlertStore | None = None,
) -> Alert:
    """Validate and persist a new alert. Raises ValueError on bad input."""
    kind = (kind or "").strip().lower().replace("-", "_")
    if kind not in KINDS:
        raise ValueError(
            f"unknown alert kind {kind!r} — pick from: {', '.join(KINDS)}")
    target = (target or "").strip()
    if not target:
        raise ValueError("alert target is required")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"threshold must be a number: {exc}") from exc
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if kind == KIND_BUDGET_PCT and not 0 < threshold <= 5:
        raise ValueError("budget_pct threshold is a fraction (e.g. 0.8 = 80%)")
    if kind == KIND_BILL_DUE and not 0 < threshold <= 90:
        raise ValueError("bill_due threshold is days ahead (e.g. 3 = 3 days)")
    if kind == KIND_BILL_DUE:
        # Bills recur — a one-shot bill alert would die after the first
        # cycle; default to repeating.
        one_shot = False
    alert = Alert(
        id="alr_" + uuid.uuid4().hex[:10],
        kind=kind, target=target, threshold=threshold,
        market=(market or "crypto").strip().lower() or "crypto",
        note=note.strip(), one_shot=bool(one_shot),
        created_at=time.time(),
    )
    return (store or AlertStore()).add(alert)


# ── evaluation ─────────────────────────────────────────────────────────

def _default_price_fn(symbol: str, market: str) -> dict[str, Any] | None:
    from ..integrations.market_data import quote
    try:
        return quote(symbol, market=market)
    except Exception:  # noqa: BLE001 - a dead source is "no data", not a crash
        _log.debug("quote failed for %s", symbol, exc_info=True)
        return None


def _default_fx_fn(base: str) -> float | None:
    from ..integrations.naija_shopping import get_fx_rate
    try:
        return get_fx_rate(base.upper(), "NGN")
    except Exception:  # noqa: BLE001
        return None


def evaluate_alerts(
    *,
    store: AlertStore | None = None,
    ledger: Ledger | None = None,
    budgets: BudgetStore | None = None,
    price_fn: Callable[[str, str], dict[str, Any] | None] | None = None,
    fx_fn: Callable[[str], float | None] | None = None,
    notify: Callable[[str, str], None] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Check every enabled alert once. Returns {"checked", "fired", "details"}.

    ``price_fn(symbol, market)`` → quote dict with "price" and
    "change_pct_24h" (or None). ``fx_fn(base)`` → NGN rate (or None).
    ``notify(title, body)`` delivers. All three are injectable so tests
    run offline; defaults hit the real (keyless) sources.
    Never raises — a broken source marks the alert unchecked, not dead.
    """
    now = time.time() if now is None else now
    store = store or AlertStore()
    price_fn = price_fn or _default_price_fn
    fx_fn = fx_fn or _default_fx_fn
    fired: list[dict[str, Any]] = []
    checked = 0
    skipped_snoozed = 0

    # Budget usage snapshot, computed once for all budget alerts.
    budget_usage: dict[str, float] = {}
    need_budgets = any(
        a.kind == KIND_BUDGET_PCT for a in store.list(enabled_only=True))
    if need_budgets:
        try:
            settings = None
            ledger_path, budgets_path = finance_paths(settings)
            ledger = ledger or Ledger(ledger_path)
            budgets = budgets or BudgetStore(budgets_path)
            for s in budget_status(ledger, budgets, month_key(now)):
                budget_usage[s.category] = s.pct_used
        except Exception:  # noqa: BLE001
            _log.debug("budget snapshot failed", exc_info=True)

    # Recurring-charge snapshot for bill_due alerts.
    recurring: list[Any] = []
    need_bills = any(
        a.kind == KIND_BILL_DUE for a in store.list(enabled_only=True))
    if need_bills:
        try:
            from .insights import _window, detect_recurring
            ledger_path, _ = finance_paths(None)
            ledger = ledger or Ledger(ledger_path)
            recurring = detect_recurring(_window(ledger, 120, now))
        except Exception:  # noqa: BLE001
            _log.debug("recurring snapshot failed", exc_info=True)

    for alert in store.list(enabled_only=True):
        if alert.snoozed:
            skipped_snoozed += 1
            continue
        checked += 1
        try:
            current, message = _check_one(
                alert, budget_usage, recurring, price_fn, fx_fn, now)
        except Exception:  # noqa: BLE001
            _log.debug("alert %s check failed", alert.id, exc_info=True)
            continue
        if current is None:
            # No data — leave state untouched, try again next round.
            continue
        triggered, clear = current
        if triggered and alert.was_clear:
            alert.fire_count += 1
            alert.last_fired_at = now
            alert.was_clear = False
            if alert.kind == KIND_BILL_DUE:
                # Remember which billing cycle fired — one ping per cycle.
                for r in recurring:
                    if r.note_pattern.lower() == alert.target.lower():
                        alert.last_due_ts = r.next_due_ts
                        break
            if alert.one_shot:
                alert.enabled = False
            store.update(alert)
            fired.append({"id": alert.id, "kind": alert.kind,
                          "target": alert.target, "message": message})
            if notify is not None:
                try:
                    notify(f"🔔 {alert.describe()}", message)
                except Exception:  # noqa: BLE001
                    _log.debug("alert notify failed", exc_info=True)
        elif clear and not alert.was_clear:
            alert.was_clear = True
            store.update(alert)

    return {"checked": checked, "fired": len(fired), "details": fired,
            "skipped_snoozed": skipped_snoozed}


def _check_one(
    alert: Alert,
    budget_usage: dict[str, float],
    recurring: list[Any],
    price_fn: Callable[[str, str], dict[str, Any] | None],
    fx_fn: Callable[[str], float | None],
    now: float,
) -> tuple[tuple[bool, bool] | None, str]:
    """(triggered, clear) + message, or (None, msg) when no data."""
    from .ledger import format_naira
    k = alert.kind
    if k in (KIND_PRICE_ABOVE, KIND_PRICE_BELOW):
        q = price_fn(alert.target, alert.market)
        if not q or q.get("price") is None:
            return None, "no price data"
        price = float(q["price"])
        if k == KIND_PRICE_ABOVE:
            hit, clear = price >= alert.threshold, price < alert.threshold
        else:
            hit, clear = price <= alert.threshold, price > alert.threshold
        return (hit, clear), (
            f"{alert.target} is now {price:,.2f} "
            f"({'≥' if k == KIND_PRICE_ABOVE else '≤'} "
            f"{alert.threshold:,.2f} — alert: {alert.describe()})")
    if k == KIND_PCT_CHANGE:
        q = price_fn(alert.target, alert.market)
        if not q or q.get("change_pct_24h") is None:
            return None, "no change data"
        move = abs(float(q["change_pct_24h"]))
        hit = move >= alert.threshold
        return (hit, not hit), (
            f"{alert.target} moved {float(q['change_pct_24h']):+.1f}% in 24h "
            f"(threshold ±{alert.threshold:.1f}%)")
    if k == KIND_BUDGET_PCT:
        usage = budget_usage.get(alert.target.strip().lower())
        if usage is None:
            return None, f"no budget set for '{alert.target}'"
        hit = usage >= alert.threshold
        return (hit, not hit), (
            f"{alert.target} budget at {usage:.0%} "
            f"(threshold {alert.threshold:.0%})")
    if k == KIND_FX_RATE:
        rate = fx_fn(alert.target)
        if rate is None:
            return None, "no FX data"
        hit = rate >= alert.threshold
        return (hit, not hit), (
            f"{alert.target}/NGN is now {rate:,.0f} "
            f"(threshold {alert.threshold:,.0f})")
    if k == KIND_BILL_DUE:
        charge = next(
            (r for r in recurring
             if r.note_pattern.lower() == alert.target.strip().lower()
             or alert.target.strip().lower() in r.note_pattern.lower()),
            None)
        if charge is None or not charge.next_due_ts:
            return None, f"no recurring charge matching '{alert.target}'"
        days_left = (charge.next_due_ts - now) / 86400
        hit = 0 < days_left <= alert.threshold
        # Clear once the due date passes (bill presumably paid) or a new
        # cycle's due date appears — re-arms for the next cycle.
        clear = days_left <= 0 or (
            alert.last_due_ts is not None
            and charge.next_due_ts != alert.last_due_ts)
        return (hit, clear), (
            f"🧾 {charge.note_pattern} — "
            f"{format_naira(charge.amount_kobo)} due in "
            f"{max(0, days_left):.0f} day(s)")
    return None, f"unknown kind {k}"


def sync_bill_alerts(store: AlertStore | None = None,
                     ledger: Ledger | None = None,
                     *, days_ahead: float = 7.0,
                     now: float | None = None) -> dict[str, int]:
    """Auto-watch detected recurring bills (Rocket Money's core loop).

    Creates/refreshes a repeating ``bill_due`` alert for every recurring
    charge whose next due date falls within ``days_ahead``. Idempotent:
    an alert for the same charge is reused, never duplicated. Returns
    {"created", "refreshed", "total"}.
    """
    from .insights import _window, detect_recurring
    now = time.time() if now is None else now
    store = store or AlertStore()
    ledger_path, _ = finance_paths(None)
    ledger = ledger or Ledger(ledger_path)
    recurring = detect_recurring(_window(ledger, 120, now))
    existing = {a.target.strip().lower(): a
                for a in store.list()
                if a.kind == KIND_BILL_DUE}
    created = refreshed = 0
    for r in recurring:
        if not (0 < (r.next_due_ts - now) / 86400 <= days_ahead):
            continue
        key = r.note_pattern.strip().lower()
        if key in existing:
            a = existing[key]
            if a.threshold != days_ahead:
                a.threshold = days_ahead
                store.update(a)
                refreshed += 1
        else:
            add_alert(KIND_BILL_DUE, r.note_pattern, days_ahead,
                      note=f"auto: {r.note_pattern} "
                           f"{r.amount_kobo // 100:,}/cycle",
                      one_shot=False, store=store)
            created += 1
    return {"created": created, "refreshed": refreshed,
            "total": len(recurring)}


def render_alerts(alerts: list[Alert],
                  theme: str | None = None) -> str:
    """One readable view of the whole watchtower."""
    from .style import current_theme
    th = current_theme(theme)
    if not alerts:
        return "no alerts set — /alert add price_above BTC 90000"
    lines = [th.paint(f"{th.alert} alerts — {len(alerts)} watching",
                      th.bold)]
    for a in alerts:
        state = ("💤 snoozed" if a.snoozed else
                 "on" if a.enabled else "off")
        extra = f" · fired {a.fire_count}×" if a.fire_count else ""
        lines.append(f"  {th.bullet} {a.id} [{state}] "
                     f"{a.describe()}{extra}")
    return "\n".join(lines)


# ── scheduler hook (follows the digest.py pattern) ─────────────────────

async def ensure_alerts_job(scheduler: Any, context: Any) -> bool:
    """Register the 15-minute alert evaluation job. Idempotent."""
    try:
        existing = await scheduler.list_cron_jobs()
    except Exception as exc:  # noqa: BLE001
        _log.warning("alerts: could not list cron jobs: %s", exc)
        return False
    if any(getattr(j, "task_id", "") == ALERTS_TASK_ID for j in existing):
        return True

    async def _run(**params: Any) -> None:
        try:
            def _notify(title: str, body: str) -> None:
                try:
                    from ..agents.notifier import Notifier
                    Notifier(context).publish("price_alert", title, body)
                except Exception as exc:  # noqa: BLE001
                    _log.debug("alert notify failed: %s", exc)

            result = evaluate_alerts(notify=_notify)
            if result["fired"]:
                _log.info("alerts fired: %d", result["fired"])
        except Exception as exc:  # noqa: BLE001
            _log.warning("alert evaluation failed: %s", exc)

    scheduler.register_action(ALERTS_ACTION, _run)
    await scheduler.schedule_cron(ALERTS_TASK_ID, ALERTS_CRON, ALERTS_ACTION,
                                  parameters={})
    _log.info("alert evaluation scheduled (%s)", ALERTS_CRON)
    return True
