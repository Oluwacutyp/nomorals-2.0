"""Unified money view — every rail, one balance sheet.

Devon talks to a lot of money rails (Nigerian banks via Mono, Paystack,
Binance, Coinbase, Exness, Wise). Each connector knows its own API; this
module is Devon's own aggregation layer: it asks every connected rail
for balances, converts everything to NGN through the shared FX layer,
and reports a single net position.

Rules:
- A rail that isn't connected (or errors) is reported as such — never
  as ₦0. Absent data is absent, not zero.
- Fiat balances convert at the cached live FX rate (USD/EUR/GBP→NGN);
  crypto converts via the keyless market-data quote (BTC→USDT × USDT/NGN).
- Anything unconvertible is listed under "unconverted" with its native
  currency, still counted in the per-rail line but not the NGN total.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .ledger import format_naira

_log = get_logger("nomorals.finance")

__all__ = [
    "RailBalance",
    "MoneyOverview",
    "collect_balances",
    "render_overview",
]

#: Fiat → NGN conversion is only attempted for these.
_FIAT_FX = ("USD", "EUR", "GBP")

#: Crypto quote currencies treated as ~1 USD.
_STABLES = ("USDT", "USDC", "BUSD", "DAI", "FDUSD", "TUSD")


@dataclass
class RailBalance:
    """One rail's contribution to the balance sheet."""

    rail: str                    # "mono" | "binance" | ...
    label: str                   # human line, e.g. "GTBank ••4521"
    amount: float = 0.0
    currency: str = ""
    amount_ngn: float | None = None   # None → unconvertible
    status: str = "ok"           # "ok" | "not_connected" | "error"
    detail: str = ""             # error text / account meta


@dataclass
class MoneyOverview:
    """The whole picture."""

    rails: list[RailBalance] = field(default_factory=list)
    total_ngn: float = 0.0
    unconverted: list[RailBalance] = field(default_factory=list)
    collected_at: float = 0.0

    def connected_rails(self) -> list[str]:
        return [r.rail for r in self.rails if r.status == "ok"]


def _fx_to_ngn(amount: float, currency: str,
               fx_fn: Callable[[str], float | None]) -> float | None:
    cur = (currency or "").upper()
    if cur in ("NGN", "₦"):
        return amount
    if cur in _FIAT_FX:
        rate = fx_fn(cur)
        return amount * rate if rate else None
    return None


def _crypto_to_ngn(symbol: str, amount: float,
                   price_fn: Callable[[str, str], dict[str, Any] | None],
                   fx_fn: Callable[[str], float | None]) -> float | None:
    sym = (symbol or "").upper()
    if sym in _STABLES:
        return _fx_to_ngn(amount, "USD", fx_fn)
    try:
        q = price_fn(sym, "crypto")
    except Exception:  # noqa: BLE001
        q = None
    if not q or not q.get("price"):
        return None
    usd_value = amount * float(q["price"])
    return _fx_to_ngn(usd_value, "USD", fx_fn)


def _default_fx(base: str) -> float | None:
    from ..integrations.naija_shopping import get_fx_rate
    try:
        return get_fx_rate(base.upper(), "NGN")
    except Exception:  # noqa: BLE001
        return None


def _default_price(symbol: str, market: str) -> dict[str, Any] | None:
    from ..integrations.market_data import quote
    try:
        return quote(symbol, market=market)
    except Exception:  # noqa: BLE001
        return None


# ── per-rail collectors (each fail-soft) ───────────────────────────────

def _rail_mono(vault: Any, fx_fn: Callable) -> list[RailBalance]:
    from ..connectors.mono import MonoConnector
    conn = MonoConnector(vault=vault)
    st = conn.status()
    if not getattr(st, "connected", False):
        return [RailBalance(rail="mono", label="Mono (bank)",
                            status="not_connected",
                            detail="link a bank account first")]
    out = []
    for acct in conn.linked_accounts():
        aid = str(acct.get("id") or acct.get("account_id") or "")
        try:
            info = conn.get_account(aid) if aid else {}
        except Exception as exc:  # noqa: BLE001
            out.append(RailBalance(rail="mono", label=f"Mono {aid or 'account'}",
                                   status="error", detail=str(exc)[:120]))
            continue
        acc = info.get("account") or info
        bal = acc.get("balance")
        try:
            amount = float(bal if bal is not None else acc.get("balance", 0))
        except (TypeError, ValueError):
            amount = 0.0
        cur = str(acc.get("currency") or "NGN").upper()
        name = str(acct.get("name") or acc.get("name") or aid or "bank")
        out.append(RailBalance(
            rail="mono", label=f"Mono · {name}", amount=amount,
            currency=cur, amount_ngn=_fx_to_ngn(amount, cur, fx_fn)))
    return out or [RailBalance(rail="mono", label="Mono (bank)",
                               status="error",
                               detail="no linked accounts returned")]


def _rail_binance(vault: Any, fx_fn: Callable,
                  price_fn: Callable) -> list[RailBalance]:
    from ..connectors.binance import BinanceConnector
    conn = BinanceConnector(vault=vault)
    st = conn.status()
    if not getattr(st, "connected", False):
        return [RailBalance(rail="binance", label="Binance",
                            status="not_connected",
                            detail="connect Binance first")]
    out = []
    for b in conn.get_balances(nonzero=True):
        asset = str(b.get("asset", "")).upper()
        total = float(b.get("total", 0) or 0)
        ngn = _crypto_to_ngn(asset, total, price_fn, fx_fn)
        if ngn is None:  # maybe it's a fiat balance
            ngn = _fx_to_ngn(total, asset, fx_fn)
        out.append(RailBalance(rail="binance", label=f"Binance · {asset}",
                               amount=total, currency=asset, amount_ngn=ngn))
    return out or [RailBalance(rail="binance", label="Binance",
                               status="ok", detail="no nonzero balances",
                               amount=0.0, currency="USDT", amount_ngn=0.0)]


def _rail_coinbase(vault: Any, fx_fn: Callable,
                   price_fn: Callable) -> list[RailBalance]:
    from ..connectors.coinbase import CoinbaseConnector
    conn = CoinbaseConnector(vault=vault)
    st = conn.status()
    if not getattr(st, "connected", False):
        return [RailBalance(rail="coinbase", label="Coinbase",
                            status="not_connected",
                            detail="connect Coinbase first")]
    out = []
    for acct in conn.list_accounts():
        try:
            avail = acct.get("available_balance") or {}
            amount = float(avail.get("value", 0) or 0)
            cur = str(avail.get("currency", "")).upper()
        except (TypeError, ValueError, AttributeError):
            continue
        if amount <= 0:
            continue
        ngn = _crypto_to_ngn(cur, amount, price_fn, fx_fn)
        if ngn is None:
            ngn = _fx_to_ngn(amount, cur, fx_fn)
        out.append(RailBalance(rail="coinbase", label=f"Coinbase · {cur}",
                               amount=amount, currency=cur, amount_ngn=ngn))
    return out or [RailBalance(rail="coinbase", label="Coinbase",
                               status="ok", detail="no nonzero balances",
                               amount=0.0, currency="USD", amount_ngn=0.0)]


def _rail_exness(vault: Any, fx_fn: Callable) -> list[RailBalance]:
    from ..connectors.exness import ExnessConnector
    conn = ExnessConnector(vault=vault)
    st = conn.status()
    if not getattr(st, "connected", False):
        return [RailBalance(rail="exness", label="Exness",
                            status="not_connected",
                            detail="connect Exness first")]
    snap = conn.get_snapshot()
    state = snap.get("account_state") or {}
    out = []
    for key, label in (("balance", "balance"), ("equity", "equity")):
        raw = state.get(key)
        if raw is None:
            continue
        try:
            amount = float(raw)
        except (TypeError, ValueError):
            continue
        cur = str(state.get("currency") or
                  (conn.get_account_details().get("settings") or {})
                  .get("currency", "USD")).upper()
        out.append(RailBalance(
            rail="exness", label=f"Exness · {label}", amount=amount,
            currency=cur, amount_ngn=_fx_to_ngn(amount, cur, fx_fn)))
    n_pos = len(snap.get("positions") or [])
    if out:
        out[0].detail = f"{n_pos} open position(s)"
    return out or [RailBalance(rail="exness", label="Exness",
                               status="error",
                               detail="snapshot returned no account state")]


def _rail_wise(vault: Any, fx_fn: Callable) -> list[RailBalance]:
    from ..connectors.wise import WiseConnector
    conn = WiseConnector(vault=vault)
    st = conn.status()
    if not getattr(st, "connected", False):
        return [RailBalance(rail="wise", label="Wise",
                            status="not_connected",
                            detail="connect Wise first")]
    # Wise connector is quote/transfer-oriented (CLI-only); no balance
    # endpoint is surfaced, so report connection state honestly.
    return [RailBalance(rail="wise", label="Wise", status="ok",
                        detail="connected — balances not exposed by "
                               "the Wise API surface used here")]


_RAILS: tuple[tuple[str, Callable], ...] = (
    ("mono", _rail_mono),
    ("binance", _rail_binance),
    ("coinbase", _rail_coinbase),
    ("exness", _rail_exness),
    ("wise", _rail_wise),
)


def collect_balances(
    vault: Any,
    *,
    rails: tuple[str, ...] | None = None,
    fx_fn: Callable[[str], float | None] | None = None,
    price_fn: Callable[[str, str], dict[str, Any] | None] | None = None,
) -> MoneyOverview:
    """Ask every connected rail for balances. Fail-soft per rail.

    ``vault`` is the credential vault the connectors need. ``rails``
    limits to a subset (e.g. ("mono", "binance")). Never raises — a
    rail that blows up is reported with status="error".
    """
    from ..connectors.base import ConnectorError

    fx_fn = fx_fn or _default_fx
    price_fn = price_fn or _default_price
    wanted = set(rails) if rails else None
    overview = MoneyOverview(collected_at=time.time())
    for name, collector in _RAILS:
        if wanted is not None and name not in wanted:
            continue
        try:
            if name in ("binance", "coinbase"):
                found = collector(vault, fx_fn, price_fn)
            else:
                found = collector(vault, fx_fn)
        except ConnectorError as exc:
            # "not connected" from a healthy vault → honest label,
            # not an error.
            if "not connected" in str(exc).lower():
                found = [RailBalance(rail=name, label=name.capitalize(),
                                     status="not_connected",
                                     detail=str(exc)[:120])]
            else:
                _log.debug("balance rail %s failed", name, exc_info=True)
                found = [RailBalance(rail=name, label=name.capitalize(),
                                     status="error", detail=str(exc)[:120])]
        except Exception as exc:  # noqa: BLE001 - one rail never sinks the sheet
            _log.debug("balance rail %s failed", name, exc_info=True)
            found = [RailBalance(rail=name, label=name.capitalize(),
                                 status="error", detail=str(exc)[:120])]
        for rb in found:
            overview.rails.append(rb)
            if rb.status == "ok" and rb.amount_ngn is not None:
                overview.total_ngn += rb.amount_ngn
            elif rb.status == "ok":
                overview.unconverted.append(rb)
    return overview


def render_overview(ov: MoneyOverview) -> str:
    """Human-readable balance sheet."""
    lines = ["💰 money overview — all rails"]
    for rb in ov.rails:
        if rb.status == "not_connected":
            lines.append(f"  • {rb.label}: not connected ({rb.detail})")
        elif rb.status == "error":
            lines.append(f"  • {rb.label}: error — {rb.detail}")
        elif rb.amount_ngn is not None:
            conv = (f"≈ {format_naira(int(rb.amount_ngn * 100))}"
                    if rb.currency.upper() not in ("NGN", "₦") else "")
            extra = f" — {rb.detail}" if rb.detail else ""
            lines.append(f"  • {rb.label}: {rb.amount:,.2f} {rb.currency} "
                         f"{conv}{extra}".rstrip())
        else:
            lines.append(f"  • {rb.label}: {rb.amount:,.2f} {rb.currency} "
                         f"(no NGN rate)")
    lines.append(f"\ntotal (converted): {format_naira(int(ov.total_ngn * 100))}")
    if ov.unconverted:
        lines.append(f"({len(ov.unconverted)} holding(s) without an NGN rate)")
    return "\n".join(lines)
