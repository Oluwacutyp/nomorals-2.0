"""``nm markets`` — the markets desk (BTC/stocks lab + signal calls)."""

from __future__ import annotations

import sys
from typing import Any


def _cmd_markets(args: Any, context: Any) -> int:
    """Route `nm markets` subcommands.  Analysis only — never trades."""
    from ...agents.marketdesk import MarketsDesk, render_collect, render_library, render_status

    desk = MarketsDesk(context)
    action = args.markets_action

    if action == "status":
        print(render_status(desk.status()))
        return 0

    if action == "library":
        print(render_library(desk.library()))
        return 0

    if action == "collect":
        print(render_collect(desk.collect()))
        return 0

    if action == "study":
        rep = desk.study(deliver_signals=False)
        for symbol, v in (rep.get("verdicts") or {}).items():
            print(f"{symbol}: {v.get('verdict')} "
                  f"({v.get('n_daily', 0)} daily, "
                  f"{v.get('n_hourly', 0)} hourly bars)")
            if v.get("evidence"):
                print(f"  {v['evidence'][:300]}")
        coll = rep.get("collect") or {}
        print(f"collect: +{coll.get('added', 0)} bars")
        return 0

    if action == "signals":
        desk.analyze_all()
        rep = desk.run_signals(deliver=not args.no_deliver)
        for pub in rep["published"]:
            c = pub["call"]
            print(f"CALL {c['symbol']} window "
                  f"{(c['window_start'] - c['published_at']) / 60:.0f}m lead "
                  f"— {pub.get('detail', '')}")
        for reason in rep["no_call_reasons"]:
            print(f"no call: {reason}")
        sc = rep["scoring"]
        print(f"scoring: {sc['calls_scored']}/{sc['calls_total']} calls "
              f"graded, hit rate {sc.get('hit_rate')}")
        return 0

    if action == "sources":
        from ...agents.marketdesk.sources import keyed_feed_state, market_status
        st = market_status(desk.registry)
        for s in st["sources"]:
            print(f"{s['domain']} — {s['health']} ({s['origin']}) "
                  f"active={s['active']}")
        for f in keyed_feed_state():
            print(f"key {f['env_key']}: "
                  f"{'present' if f['key_present'] else f['status']}")
        return 0

    print(f"unknown markets action: {action}", file=sys.stderr)
    return 2
