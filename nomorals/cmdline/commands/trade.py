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
    except bridge.LiveTradingDisabled as exc:
        print(f"trade: live trading disabled: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"trade: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"unknown trade action: {action}", file=sys.stderr)
    return 2
