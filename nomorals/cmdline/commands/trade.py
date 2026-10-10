"""``nm trade`` — trading surfaces."""

from __future__ import annotations

import argparse
import sys
from typing import Any
from ..emit import _emit



def _cmd_trade(args: argparse.Namespace, context: Any) -> int:
    """Route `nm trade` to the FinancialExpert / trading tool actions."""
    from ...agents.financial_expert import FinancialExpert
    from ...integrations import sentinel_bridge as bridge
    from ...tools import trading as trading_tool

    as_json = getattr(args, "json", False)
    action = args.trade_action
    try:
        if action == "analyze":
            rep = FinancialExpert(context).analyze(
                args.symbol, market=args.market,
                timeframe=args.timeframe, bars=args.bars)
            _emit(args, rep.to_dict(), rep.summary_text())
            return 0
        if action == "backtest":
            rep = FinancialExpert(context).backtest(
                args.symbol, market=args.market, profile=args.profile)
            _emit(args, rep.to_dict(), rep.summary_text())
            return 0
        if action == "signal":
            rep = FinancialExpert(context).signal(args.symbol,
                                                  market=args.market)
            _emit(args, rep.to_dict(), rep.summary_text())
            return 0
        if action == "compare":
            symbols = [s.strip() for s in args.symbols.split(",")
                       if s.strip()]
            rep = FinancialExpert(context).compare(symbols,
                                                   market=args.market)
            _emit(args, rep.to_dict(), rep.summary_text())
            return 0
        if action == "strategies":
            names = bridge.list_strategies()[: args.limit]
            _emit(args, {"market": args.market, "count": len(names),
                         "strategies": names},
                  "\n".join(f"  {n}" for n in names) or "no strategies")
            return 0
        if action == "paper":
            paction = args.paper_action
            if paction == "start":
                out = trading_tool.paper_start(context, args.symbol,
                                               market=args.market,
                                               capital=args.capital)
                _emit(args, out,
                      f"paper session {out['session_id']} "
                      f"({'resumed' if out.get('resumed') else 'opened'}): "
                      f"{args.symbol} capital={out.get('capital')}")
                return 0
            if paction == "status":
                out = trading_tool.paper_status(
                    context, session_id=args.session, symbol=args.symbol)
                _emit(args, out,
                      f"{out['symbol']} equity={out['equity']:,.2f} "
                      f"pnl={out['pnl']:+,.2f} ({out['pnl_pct']:+.2%}) "
                      f"units={out['units']} orders={out['orders']} "
                      f"regime={out['last_action'].get('regime')}")
                return 0
            if paction == "stop":
                out = trading_tool.paper_stop(context,
                                              session_id=args.session)
                _emit(args, out,
                      f"paper session {out['session_id']} closed "
                      f"({out['symbol']})")
                return 0
        if action == "live":
            laction = args.live_action
            if laction == "unlock":
                out = trading_tool.live_unlock(context,
                                               confirm=args.confirm)
                _emit(args, out,
                      f"live unlocked for {out['expires_in_hours']}h "
                      f"(id {out['unlock_id']})")
                return 0
            if laction == "order":
                out = trading_tool.live_order(context, args.symbol,
                                              args.side, args.size,
                                              exchange=args.exchange)
                _emit(args, out,
                      f"LIVE {out['side']} {out['size']} {out['symbol']} "
                      f"order={out['order_id']}")
                return 0
        if action == "kill":
            out = trading_tool.kill_switch(context, "nm trade kill")
            _emit(args, out,
                  f"kill switch engaged: {out['reason']} "
                  f"(journal {out['journal_id']})")
            return 0
        if action == "doctor":
            rep = bridge.doctor()
            _emit(args, rep.to_dict(), rep.summary_text())
            return 0 if rep.ok else 1
        if action == "exness":
            return _cmd_trade_exness(args, context)
    except bridge.LiveTradingDisabled as exc:
        print(f"trade: live trading disabled: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"trade: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"unknown trade action: {action}", file=sys.stderr)
    return 2


def _cmd_trade_exness(args: argparse.Namespace, context: Any) -> int:
    """Route `nm trade exness <action>` to the Exness desk."""
    from ...tools import trading as trading_tool

    xaction = args.exness_action
    kw: dict[str, Any] = {"mode": getattr(args, "mode", "paper") or "paper"}
    for name in ("instrument", "side", "volume", "entry", "stop_loss",
                 "take_profit", "risk_pct", "position_id", "price",
                 "exit_price", "comment", "limit", "confirm"):
        val = getattr(args, name, None)
        if val is not None and val != "":
            kw[name] = val
    # argparse gives `instrument` positionally; the desk also accepts it
    # as `symbol`.
    if "instrument" in kw:
        kw["symbol"] = kw["instrument"]
    if getattr(args, "confirmed", False):
        kw["confirmed"] = True
    out = trading_tool.exness_action(context, f"exness_{xaction}", **kw)
    as_json = getattr(args, "json", False)
    if not out.get("ok"):
        _emit(args, out, f"exness {xaction} failed: {out.get('error')}")
        return 1
    _emit(args, out, _render_exness(xaction, out))
    return 0


def _render_exness(action: str, out: dict[str, Any]) -> str:
    """One-line-ish human rendering per exness action."""
    if action == "snapshot":
        n = len(out.get("positions") or [])
        return (f"balance={out.get('balance')} equity={out.get('equity')} "
                f"{out.get('currency')} · {n} open position(s)")
    if action == "positions":
        poss = out.get("positions") or []
        if not poss:
            return "no open positions"
        return "\n".join(
            f"  {p.get('id')} {p.get('side')} {p.get('volume')} "
            f"{p.get('instrument')} @ {p.get('entry')}"
            for p in poss)
    if action == "size":
        return (f"{out.get('instrument')}: volume={out.get('volume')} "
                f"(risk {out.get('actual_risk')} = "
                f"{out.get('actual_risk_pct')}%)")
    if action == "open":
        return (f"opened {out.get('position_id', '')} "
                f"{out.get('side')} {out.get('volume')} "
                f"{out.get('instrument')} @ {out.get('entry')}")
    if action == "close":
        return (f"closed {out.get('position_id')}: "
                f"pnl={out.get('pnl', 0):+.2f} ({out.get('reason')})")
    if action == "mark":
        closes = out.get("closes") or []
        lines = [f"marked — {out.get('open')} still open"]
        for c in closes:
            lines.append(f"  closed {c['position_id']}: "
                         f"pnl={c['pnl']:+.2f} ({c['reason']})")
        return "\n".join(lines)
    if action == "stats":
        return (f"trades={out.get('total_trades')} "
                f"win_rate={out.get('win_rate', 0):.0%} "
                f"realized_pnl={out.get('realized_pnl', 0):+.2f} "
                f"daily_pnl={out.get('daily_pnl', 0):+.2f}")
    if action == "journal":
        entries = out.get("entries") or []
        if not entries:
            return "journal is empty"
        return "\n".join(
            f"  {e.get('event')} {e.get('instrument', '')} "
            f"{e.get('side', '')} pnl={e.get('pnl', '')}"
            for e in entries)
    if action == "unlock":
        return f"live unlocked: {out.get('note', '')}"
    if action == "risk":
        pol = out.get("policy") or {}
        return ("risk policy: "
                f"{pol.get('max_risk_pct_per_trade')}%/trade, "
                f"max {pol.get('max_open_positions')} positions, "
                f"daily loss kill at {pol.get('max_daily_loss_pct')}%, "
                f"SL required={pol.get('require_stop_loss')}, "
                f"live_unlocked={pol.get('live_unlocked')} · "
                f"daily pnl={out.get('daily_pnl', 0):+.2f}")
    return str(out)
