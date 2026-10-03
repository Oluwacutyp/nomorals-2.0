"""``nm bet`` — paper sports betting."""

from __future__ import annotations

import argparse
import sys
from typing import Any



def _cmd_bet(args: argparse.Namespace, context: Any) -> int:
    """Route `nm bet` subcommands.  Analysis only — never places bets."""
    from ...agents.sports_bet import (BetStore, Fixture, OddsSnapshot, backtest,
                                    fixture_digest, render_analysis,
                                    render_backtest, synthetic_history)

    store = BetStore()
    action = args.bet_action

    if action == "bankroll":
        if args.set is not None:
            store.bankroll = args.set
            store.save()
            print(f"bankroll set to {args.set:.2f}")
        else:
            print(f"paper bankroll: {store.bankroll:.2f}")
        return 0

    if action == "record":
        try:
            hg_s, ag_s = args.score.split("-")
            hg, ag = int(hg_s), int(ag_s)
        except ValueError:
            print("bad --score, want HG-AG like 2-1", file=sys.stderr)
            return 2
        store.record(Fixture(home=args.home, away=args.away,
                             league=args.league, home_goals=hg,
                             away_goals=ag))
        print(f"recorded: {args.home} {hg}-{ag} {args.away}")
        return 0

    if action == "backtest":
        entries = synthetic_history(n=args.n, seed=args.seed)
        r = backtest(entries, bankroll=store.bankroll, seed=args.seed)
        print(render_backtest(r))
        return 0

    if action == "analyze":
        if not args.home or not args.away:
            # fixture mode: fetch upcoming fixtures & analyze the best ones
            print(fixture_digest(store, league_text=args.league or "",
                                 top_n=args.top, min_edge=args.min_edge))
            return 0
        snaps = []
        if args.odds:
            h, d, a = args.odds
            snaps = [OddsSnapshot(bookmaker=args.bookmaker, home=h,
                                  draw=d, away=a)]
        an = store.analyst.analyze(args.home, args.away,
                                   league=args.league, odds=snaps,
                                   fixtures=store.fixtures(),
                                   min_edge=args.min_edge)
        print(render_analysis(an))
        return 0

    print(f"unknown bet action: {action}", file=sys.stderr)
    return 2
