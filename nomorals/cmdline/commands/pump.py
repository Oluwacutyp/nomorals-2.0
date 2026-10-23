"""``nm pump`` — the pump.fun intelligence desk (signals only)."""

from __future__ import annotations

from typing import Any


def _cmd_pump(args: Any, context: Any) -> int:
    """Route `nm pump` subcommands.  Intelligence only — the desk
    never trades, holds no keys, signs nothing."""
    from ...agents.pumpdesk import (PumpDesk, render_calls,
                                    render_dossier, render_status)

    desk = PumpDesk(context)
    action = args.pump_action

    if action == "status":
        print(render_status(desk.status()))
        return 0

    if action == "watch":
        rep = desk.watch_cycle(deliver=not args.no_deliver)
        for pub in rep.get("calls", []):
            c = pub["call"]
            print(f"CALL {c['symbol'] or c['mint'][:8]}: entry "
                  f"{c['entry_mc_sol']:.1f} SOL mcap → target "
                  f"{c['target_mc_sol']:.1f} — {pub.get('detail', '')}")
        for r in rep.get("no_call_reasons", []):
            print(f"  no call: {r}")
        fe = rep.get("frontend") or {}
        print(f"frontend: {fe.get('promoted', 0)} candidates promoted "
              f"of {fe.get('rows', 0)} rows"
              + (f" (error: {fe['error']})" if fe.get("error") else ""))
        return 0

    if action == "study":
        rep = desk.study_cycle()
        sc = rep.get("scoring") or {}
        g = rep.get("graph") or {}
        ln = rep.get("learning") or {}
        sh = rep.get("shelved") or {}
        print(f"study: {sc.get('scored_now', 0)} calls graded "
              f"({sc.get('wins_now', 0)} wins), "
              f"{rep.get('dev_profiles', 0)} dev profiles, "
              f"{rep.get('serial_ruggers', 0)} serial ruggers flagged; "
              f"graph: {g.get('wallets', 0)} wallets "
              f"({g.get('graded', 0)} graded, {g.get('clusters', 0)} "
              f"cohorts); learning: {ln.get('status', 'n/a')}; "
              f"shelved {sh.get('dossiers', 0)} dossiers + "
              f"{sh.get('wallets', 0)} wallets")
        return 0

    if action == "wallet":
        print(desk.wallet_view(args.wallet)["rendered"])
        return 0

    if action == "graph":
        print(desk.graph_view()["rendered"])
        return 0

    if action == "attention":
        print(desk.attention_view(args.mint)["rendered"])
        return 0

    if action == "learn":
        print(desk.learn_view()["rendered"])
        return 0

    if action == "shelf":
        print(desk.shelf_view(" ".join(args.query))["rendered"])
        return 0

    if action == "dossier":
        d = desk.refresh_dossier(args.mint)
        print(render_dossier(d))
        from ...agents.pumpdesk.score import evaluate
        v = evaluate(d, smart_wallets=desk.store.watched_smart())
        print(f"verdict: {v.verdict} (composite {v.composite:.2f}, "
              f"rug {v.rug_risk:.2f}, manipulation {v.manipulation:.2f}, "
              f"momentum {v.momentum:.2f})")
        print(f"  {v.evidence}")
        return 0

    if action == "calls":
        calls = [c.to_dict() for c in desk.calls.log.calls()]
        print(render_calls(calls))
        return 0

    if action == "sources":
        from ...agents.pumpdesk.sources import pump_status
        st = pump_status(desk.registry)
        if not st["sources"]:
            print("no memecoin sources registered yet")
            return 0
        for s in st["sources"]:
            print(f"{s['domain']} — {s['health']} ({s['origin']})")
        return 0

    if action == "provision":
        from ...agents.pumpdesk.sources import ensure_provisioning
        ids = ensure_provisioning()
        import os
        print("pump desk keyed services in self-provisioning: "
              + ", ".join(ids))
        print(f"  helius (HELIUS_API_KEY): "
              f"{'present' if os.environ.get('HELIUS_API_KEY', '').strip() else 'MISSING'}")
        print(f"  pumpportal (PUMPPORTAL_API_KEY): "
              f"{'present' if os.environ.get('PUMPPORTAL_API_KEY', '').strip() else 'MISSING'}")
        print("drive it with /pump provision helius or /pump "
              "provision pumpportal in chat (owner's own account; "
              "then vaulted — Helius validated live, PumpPortal "
              "has no GET key check so its validation is "
              "inconclusive by design)")
        return 0

    print(f"unknown pump action {action!r}")
    return 2
