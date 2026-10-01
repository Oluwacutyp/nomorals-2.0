"""Trading tool: the `trading` action surface over the FinancialExpert.

Actions: analyze | backtest | signal | compare | strategies | paper_start |
paper_status | paper_stop | doctor | live_unlock | live_order | kill

Paper trading is simulated (Sentinel's PaperBroker) and always available.
Live trading is INERT unless every gate passes:

1. ``settings.trading.live_enabled`` is true, AND
2. a live unlock grant exists and has not expired (24h TTL), AND
3. today's live loss is under ``settings.trading.max_daily_loss_pct``, AND
4. fewer than 3 consecutive engine errors.

Exchange API keys come ONLY from the credential vault
(``NM_VAULT_PASSPHRASE`` unlocks it) — never from CLI args, env values, or
configs. Every live action is journaled to the append-only ``trade_journal``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ..agents.financial_expert import FinancialExpert
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..core.policy import Capability
import nomorals.integrations.sentinel_bridge as bridge

__all__ = ["register", "LiveTradingDisabled",
           "set_exchange_client_factory", "trading"]

_log = get_logger(__name__)

LiveTradingDisabled = bridge.LiveTradingDisabled

# Paper P&L is marked in the quote currency of the symbol; fees in bps.
_PAPER_FEE_BPS = 5.0
_ERROR_BUDGET = 3  # consecutive engine errors before auto-kill


# ── exchange client (injectable for tests) ──────────────────────────────
def _default_exchange_client(symbol: str, exchange: str, api_key: str,
                             secret: str) -> Any:
    bridge._ensure_path()
    try:
        from sentinel.live.trader import CCXTBroker
    except ImportError as exc:
        raise bridge.SentinelError(
            f"live broker unavailable ({exc})") from exc
    return CCXTBroker(symbol=symbol, exchange=exchange,
                      api_key=api_key, secret=secret)


_EXCHANGE_CLIENT_FACTORY = _default_exchange_client


def set_exchange_client_factory(fn: Any) -> None:
    """Tests: inject a fake exchange client. Pass None to restore default."""
    global _EXCHANGE_CLIENT_FACTORY
    _EXCHANGE_CLIENT_FACTORY = fn or _default_exchange_client


# ── small helpers ───────────────────────────────────────────────────────
def _ts() -> float:
    return time.time()


def _trading_settings(context: Any) -> Any:
    return getattr(getattr(context, "settings", None), "trading", None)


def _vault(context: Any) -> Any:
    from ..accounts.vault import CredentialVault
    passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
    if not passphrase:
        raise bridge.SentinelError(
            "vault is locked: set the NM_VAULT_PASSPHRASE environment "
            "variable to unlock exchange credentials")
    return CredentialVault(context.db, master_passphrase=passphrase)


def _exchange_keys(context: Any, exchange: str) -> tuple[str, str]:
    """API key + secret from the vault ONLY. Never from args/env/config."""
    vault = _vault(context)
    service = f"exchange:{exchange.lower()}"
    try:
        cred = vault.get(service, "api")
    except Exception as exc:  # noqa: BLE001
        raise bridge.SentinelError(
            f"no API credentials in the vault for {exchange!r} "
            f"(expected service={service!r}, username='api'): {exc}") from exc
    api_key = getattr(cred, "password", "") or ""
    secret = (getattr(cred, "metadata", {}) or {}).get("secret", "")
    if not api_key or not secret:
        raise bridge.SentinelError(
            f"vault credential {service!r} is missing the key or secret")
    return api_key, secret


def _journal(db: Any, kind: str, symbol: str, side: str, size: float,
             price: float, order_id: str, reason: str,
             meta: dict[str, Any] | None = None) -> str:
    jid = new_id("tj")
    db.execute(
        "INSERT INTO trade_journal "
        "(id, ts, kind, symbol, side, size, price, order_id, reason, meta_json)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (jid, _ts(), kind, symbol, side, float(size), float(price),
         order_id, reason, json.dumps(meta or {})),
    )
    return jid


def _kv_get(db: Any, key: str, default: str = "") -> str:
    row = db.query_one("SELECT value FROM kv_store WHERE key=?", (key,))
    return row["value"] if row else default


def _kv_set(db: Any, key: str, value: str) -> None:
    db.execute(
        "INSERT INTO kv_store (key, value, kind, updated_at) VALUES (?,?,?,?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
        " updated_at=excluded.updated_at",
        (key, value, "str", _ts()),
    )


def _error_count(db: Any) -> int:
    try:
        return int(_kv_get(db, "trading.live_errors", "0"))
    except ValueError:
        return 0


def _reset_errors(db: Any) -> None:
    _kv_set(db, "trading.live_errors", "0")


def _bump_errors(db: Any) -> int:
    n = _error_count(db) + 1
    _kv_set(db, "trading.live_errors", str(n))
    return n


def _daily_live_loss_pct(db: Any) -> float:
    """Today's live P&L as a fraction of today's starting equity.

    Equity is tracked per journal fill via meta.equity_after; the day's
    start is the last fill before today (or the first fill today).
    """
    day_start = _ts() - (_ts() % 86400)
    rows = db.query(
        "SELECT ts, meta_json FROM trade_journal WHERE kind='live' "
        "AND ts >= ? ORDER BY ts", (day_start - 86400 * 2,))
    equities: list[tuple[float, float]] = []
    for r in rows:
        try:
            eq = float((json.loads(r["meta_json"] or "{}"))
                       .get("equity_after", 0) or 0)
        except (ValueError, TypeError):
            eq = 0.0
        if eq > 0:
            equities.append((float(r["ts"]), eq))
    if len(equities) < 1:
        return 0.0
    start_eq = next((eq for ts, eq in equities if ts < day_start),
                    equities[0][1])
    current_eq = equities[-1][1]
    if start_eq <= 0:
        return 0.0
    return (start_eq - current_eq) / start_eq


def _valid_unlock(db: Any) -> dict[str, Any] | None:
    row = db.query_one(
        "SELECT * FROM live_unlocks WHERE expires_at > ? "
        "ORDER BY expires_at DESC LIMIT 1", (_ts(),))
    return row


def _auto_kill(context: Any, reason: str) -> dict[str, Any]:
    """Revoke all unlocks and journal the kill. Live goes inert."""
    db = context.db
    db.execute("DELETE FROM live_unlocks")
    _kv_set(db, "trading.last_kill", json.dumps(
        {"ts": _ts(), "reason": reason}))
    jid = _journal(db, "live", "", "kill", 0, 0, "",
                   f"kill switch: {reason}")
    _log.warning("trading kill switch: %s", reason)
    return {"ok": True, "reason": reason, "journal_id": jid}


def _check_live_gates(context: Any) -> None:
    """Raise LiveTradingDisabled unless every live gate passes."""
    settings = _trading_settings(context)
    if not getattr(settings, "live_enabled", False):
        raise LiveTradingDisabled(
            "live trading is disabled (settings.trading.live_enabled=false). "
            "Paper trading is always available.")
    if _valid_unlock(context.db) is None:
        raise LiveTradingDisabled(
            "no valid live unlock — run "
            "`nm trade live unlock --confirm \"I understand\"` first")
    max_loss = float(getattr(settings, "max_daily_loss_pct", 3.0)) / 100.0
    loss = _daily_live_loss_pct(context.db)
    if loss >= max_loss:
        _auto_kill(context,
                   f"daily loss {loss:.2%} >= limit {max_loss:.2%}")
        raise LiveTradingDisabled(
            f"daily loss limit hit ({loss:.2%}); kill switch engaged")
    if _error_count(context.db) >= _ERROR_BUDGET:
        _auto_kill(context, "3 consecutive engine errors")
        raise LiveTradingDisabled(
            "3 consecutive engine errors; kill switch engaged")


# ── paper sessions ──────────────────────────────────────────────────────
def _paper_broker_state(db: Any, session_id: str) -> dict[str, Any]:
    row = db.query_one("SELECT state_json FROM paper_sessions WHERE id=?",
                       (session_id,))
    if not row:
        raise bridge.SentinelError(f"no paper session {session_id}")
    try:
        return json.loads(row["state_json"] or "{}")
    except ValueError:
        return {}


def _paper_save(db: Any, session_id: str, state: dict[str, Any]) -> None:
    db.execute("UPDATE paper_sessions SET state_json=?, updated_at=? "
               "WHERE id=?", (json.dumps(state), _ts(), session_id))


def paper_start(context: Any, symbol: str, market: str = "crypto",
                capital: float = 10000.0) -> dict[str, Any]:
    db = context.db
    existing = db.query_one(
        "SELECT * FROM paper_sessions WHERE symbol=? AND market=? "
        "AND status='open' ORDER BY created_at DESC LIMIT 1",
        (symbol, market))
    if existing:
        return {"ok": True, "session_id": existing["id"],
                "resumed": True, "symbol": symbol, "market": market}
    sid = new_id("paper")
    state = {"cash": float(capital), "units": 0.0, "orders": [],
             "fee_bps": _PAPER_FEE_BPS, "capital": float(capital)}
    db.execute(
        "INSERT INTO paper_sessions (id, symbol, market, capital, status,"
        " state_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        (sid, symbol, market, float(capital), "open",
         json.dumps(state), _ts(), _ts()),
    )
    _journal(db, "paper", symbol, "start", 0, 0, sid,
             f"paper session opened with {capital:,.2f}")
    _log.info("paper session %s opened for %s", sid, symbol)
    return {"ok": True, "session_id": sid, "resumed": False,
            "symbol": symbol, "market": market, "capital": float(capital)}


def _restore_broker(state: dict[str, Any]) -> Any:
    bridge._ensure_path()
    from sentinel.live.trader import PaperBroker
    broker = PaperBroker(cash=float(state.get("cash", 0.0)),
                         fee_bps=float(state.get("fee_bps", _PAPER_FEE_BPS)))
    broker.units = float(state.get("units", 0.0))
    broker.orders = list(state.get("orders", []))
    return broker


def _snapshot_broker(broker: Any, capital: float) -> dict[str, Any]:
    return {"cash": float(broker.cash), "units": float(broker.units),
            "orders": list(getattr(broker, "orders", [])),
            "fee_bps": float(getattr(broker, "fee", 0.0)) * 1e4,
            "capital": float(capital)}


def paper_status(context: Any, session_id: str = "",
                 symbol: str = "") -> dict[str, Any]:
    """Mark one paper session to market through one live-loop step
    (Sentinel LiveTrader + PaperBroker = simulated fills, never real)."""
    db = context.db
    if session_id:
        row = db.query_one("SELECT * FROM paper_sessions WHERE id=?",
                           (session_id,))
    else:
        row = db.query_one(
            "SELECT * FROM paper_sessions WHERE symbol=? AND status='open'"
            " ORDER BY created_at DESC LIMIT 1", (symbol,))
    if not row:
        raise bridge.SentinelError("no open paper session found")
    state = _paper_broker_state(db, row["id"])
    broker = _restore_broker(state)
    bridge._ensure_path()
    from sentinel.live.trader import LiveTrader
    engine = bridge.get_engine(row["market"])
    df = bridge.load_data(row["symbol"], row["market"], "1h", 600)
    trader = LiveTrader(engine, broker=broker, dry_run=False)
    action = trader.step(df, symbol=row["symbol"])
    _paper_save(db, row["id"],
                _snapshot_broker(broker, float(row["capital"])))
    equity = float(broker.equity(float(df["close"].iloc[-1])))
    pnl = equity - float(row["capital"])
    return {
        "ok": True, "session_id": row["id"], "symbol": row["symbol"],
        "market": row["market"], "status": row["status"],
        "cash": round(broker.cash, 2), "units": round(broker.units, 6),
        "equity": round(equity, 2),
        "pnl": round(pnl, 2), "pnl_pct": round(pnl / row["capital"], 4),
        "orders": len(broker.orders),
        "last_action": {k: action.get(k) for k in (
            "regime", "bias", "position", "approved", "target_frac")},
    }


def paper_stop(context: Any, session_id: str = "") -> dict[str, Any]:
    db = context.db
    row = db.query_one("SELECT * FROM paper_sessions WHERE id=? "
                       "AND status='open'", (session_id,)) if session_id else None
    if row is None:
        row = db.query_one("SELECT * FROM paper_sessions WHERE status='open'"
                           " ORDER BY created_at DESC LIMIT 1")
    if not row:
        raise bridge.SentinelError("no open paper session to stop")
    db.execute("UPDATE paper_sessions SET status='closed', updated_at=? "
               "WHERE id=?", (_ts(), row["id"]))
    _journal(db, "paper", row["symbol"], "stop", 0, 0, row["id"],
             "paper session closed")
    return {"ok": True, "session_id": row["id"], "symbol": row["symbol"]}


# ── live ────────────────────────────────────────────────────────────────
def live_unlock(context: Any, confirm: str = "") -> dict[str, Any]:
    settings = _trading_settings(context)
    if not getattr(settings, "live_enabled", False):
        raise LiveTradingDisabled(
            "settings.trading.live_enabled=false — enable it in config "
            "first, then unlock")
    if (confirm or "").strip().lower() not in ("i understand",):
        raise bridge.SentinelError(
            "refusing to unlock: pass --confirm \"I understand\"")
    ttl = float(getattr(settings, "unlock_ttl_hours", 24.0)) * 3600
    now = _ts()
    uid = new_id("unlock")
    context.db.execute(
        "INSERT INTO live_unlocks (id, granted_at, expires_at, note)"
        " VALUES (?,?,?,?)", (uid, now, now + ttl,
                               "operator confirmed understanding"))
    _journal(context.db, "live", "", "unlock", 0, 0, uid,
             "live trading unlocked for 24h")
    return {"ok": True, "unlock_id": uid,
            "expires_at": now + ttl,
            "expires_in_hours": round(ttl / 3600, 1)}


def live_order(context: Any, symbol: str, side: str, size: float,
               exchange: str = "binance") -> dict[str, Any]:
    """Place ONE live market order. Every gate must pass first."""
    _check_live_gates(context)
    side = (side or "").strip().lower()
    if side not in ("buy", "sell"):
        raise bridge.SentinelError("side must be 'buy' or 'sell'")
    size = float(size)
    if size <= 0:
        raise bridge.SentinelError("size must be positive")
    api_key, secret = _exchange_keys(context, exchange)
    try:
        client = _EXCHANGE_CLIENT_FACTORY(symbol, exchange, api_key, secret)
        price = 0.0
        try:  # mark price for the journal; best effort
            ticker = client._ex.fetch_ticker(symbol) \
                if getattr(client, "_ex", None) else None
            price = float((ticker or {}).get("last", 0) or 0)
        except Exception:  # noqa: BLE001
            pass
        units = size if side == "buy" else -size
        fill = client.market(units, price, tag="devon:manual")
        order_id = str(fill.get("ccxt_id") or fill.get("order_id")
                       or new_id("ord"))
        _reset_errors(context.db)
        jid = _journal(context.db, "live", symbol, side, abs(units), price,
                       order_id, "manual live order",
                       {"exchange": exchange, "fill": str(fill)[:500]})
        _log.warning("LIVE order placed: %s %s %s (%s)", side, units,
                     symbol, order_id)
        return {"ok": True, "order_id": order_id, "journal_id": jid,
                "symbol": symbol, "side": side, "size": abs(units),
                "exchange": exchange}
    except LiveTradingDisabled:
        raise
    except Exception as exc:  # noqa: BLE001 - engine/exchange failure
        n = _bump_errors(context.db)
        _log.error("live order failed (%d/%d): %s", n, _ERROR_BUDGET, exc)
        if n >= _ERROR_BUDGET:
            _auto_kill(context, f"{n} consecutive engine errors")
        raise bridge.SentinelError(f"live order failed: {exc}") from exc


def kill_switch(context: Any, reason: str = "operator kill") -> dict[str, Any]:
    return _auto_kill(context, reason)


# ── registry ────────────────────────────────────────────────────────────
def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "trading",
        description=(
            "FinancialExpert trading brain (Sentinel.py engine). "
            "analyze/backtest/signal/compare/strategies are research tools; "
            "paper_* run simulated sessions; live_order is inert unless "
            "every live gate passes (settings + 24h unlock + loss limit); "
            "kill revokes live immediately. action=..."),
        capability=Capability.NET_OUT,
        parameters={
            "action": "str — analyze|backtest|signal|compare|strategies|"
                      "paper_start|paper_status|paper_stop|doctor|"
                      "live_unlock|live_order|kill",
            "symbol": "str — e.g. BTC/USDT, XAUUSD",
            "symbols": "str — comma-separated, for compare",
            "market": "str — crypto|forex|stocks",
            "timeframe": "str — 1h, 4h, 1d",
            "bars": "int — bars to load",
            "profile": "str — default|aggressive|conservative",
            "capital": "float — paper starting capital",
            "session_id": "str — paper session",
            "side": "str — buy|sell (live_order)",
            "size": "float — order size in units (live_order)",
            "exchange": "str — exchange id (live_order)",
            "confirm": "str — 'I understand' (live_unlock)",
        },
    )
    def trading(action: str = "doctor", **kwargs: Any) -> dict[str, Any]:
        expert = FinancialExpert(context)
        action = (action or "doctor").strip().lower()
        symbol = str(kwargs.get("symbol") or "")
        market = str(kwargs.get("market") or "crypto")

        if action == "analyze":
            return expert.analyze(
                symbol, market,
                timeframe=str(kwargs.get("timeframe") or "1h"),
                bars=int(kwargs.get("bars") or 2000)).to_dict()
        if action == "backtest":
            return expert.backtest(
                symbol, market,
                profile=str(kwargs.get("profile") or "default")).to_dict()
        if action == "signal":
            return expert.signal(symbol, market).to_dict()
        if action == "compare":
            symbols = [s.strip() for s in
                       str(kwargs.get("symbols") or symbol).split(",")
                       if s.strip()]
            return expert.compare(symbols, market).to_dict()
        if action == "strategies":
            names = bridge.list_strategies()
            return {"ok": True, "market": market, "count": len(names),
                    "strategies": names[:200]}
        if action == "paper_start":
            return paper_start(context, symbol, market,
                               float(kwargs.get("capital") or 10000.0))
        if action == "paper_status":
            return paper_status(context,
                                session_id=str(kwargs.get("session_id") or ""),
                                symbol=symbol)
        if action == "paper_stop":
            return paper_stop(context,
                              session_id=str(kwargs.get("session_id") or ""))
        if action == "doctor":
            return bridge.doctor().to_dict()
        if action == "live_unlock":
            return live_unlock(context, str(kwargs.get("confirm") or ""))
        if action == "live_order":
            return live_order(context, symbol,
                              str(kwargs.get("side") or ""),
                              float(kwargs.get("size") or 0),
                              str(kwargs.get("exchange") or "binance"))
        if action == "kill":
            return kill_switch(
                context, str(kwargs.get("reason") or "operator kill"))
        raise bridge.SentinelError(f"unknown trading action {action!r}")
