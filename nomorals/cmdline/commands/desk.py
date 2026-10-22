"""``nm desk`` — the betting desk (crash lab + signal calls)."""

from __future__ import annotations

import sys
from typing import Any


def _cmd_desk(args: Any, context: Any) -> int:
    """Route `nm desk` subcommands.  Analysis only — never places bets."""
    from ...agents.betdesk import (BettingDesk, render_collect,
                                   render_library, render_status)

    desk = BettingDesk(context)
    action = args.desk_action

    if action == "status":
        print(render_status(desk.status()))
        return 0

    if action == "library":
        print(render_library(desk.library()))
        return 0

    if action == "collect":
        print(render_collect(desk.collect()))
        return 0

    if action == "import":
        from ...agents.betdesk.collectors import import_rounds_file
        res = import_rounds_file(desk.store, args.path, args.provider,
                                 args.game)
        print(res)
        return 0 if res.get("ok") else 2

    if action == "study":
        rep = desk.study(deliver_signals=False)
        for key, v in (rep.get("verdicts") or {}).items():
            print(f"{key}: {v.get('verdict')} "
                  f"({v.get('n_rounds', 0)} rounds)")
            if v.get("evidence"):
                print(f"  {v['evidence'][:300]}")
        sc = rep.get("sports_collect") or {}
        if sc.get("error"):
            print(f"sports collect: {sc['error']}")
        else:
            print(f"sports collect: {sc.get('landed', 0)} new fixtures "
                  f"landed")
        sports = rep.get("sports") or {}
        print(f"sports scan: "
              f"{'ok' if sports.get('ok') else sports.get('detail')}")
        return 0

    if action == "signals":
        desk.analyze_all()
        rep = desk.run_signals(deliver=not args.no_deliver)
        for pub in rep["published"]:
            c = pub["call"]
            print(f"CALL {c['provider']}/{c['game']} target "
                  f"{c['target']:.2f}x lead "
                  f"{(c['window_start'] - c['published_at']) / 60:.0f}m "
                  f"— {pub.get('detail', '')}")
        for reason in rep["no_call_reasons"]:
            print(f"no call: {reason}")
        sc = rep["scoring"]
        print(f"scoring: {sc['calls_scored']}/{sc['calls_total']} calls "
              f"graded, hit rate {sc.get('hit_rate')}")
        return 0

    if action == "sources":
        from ...agents.betdesk.sources import betting_status
        st = betting_status(desk.registry)
        for s in st["sources"]:
            print(f"{s['domain']} — {s['health']} ({s['origin']}) "
                  f"active={s['active']}")
        return 0

    if action == "provision":
        from ...agents.betdesk.sources import ensure_provisioning
        ids = ensure_provisioning()
        print("registered with self-provisioning: " + ", ".join(ids))
        for p in desk.status()["provisioning"]:
            if not p["key_present"]:
                state = "MISSING"
            elif p.get("source") == "vault":
                state = ("present — vaulted"
                         + (f", validated: {p['validation']}"
                            if p.get("validation") else ""))
            else:
                state = "present"
            print(f"  {p['service']} ({p['env_key']}): {state}")
        print("drive a provisioning run from chat: "
              "/desk provision <service> (or /provision <service>) — "
              "the drive needs the chat browser session")
        return 0

    print(f"unknown desk action: {action}", file=sys.stderr)
    return 2
