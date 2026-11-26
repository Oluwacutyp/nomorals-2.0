"""The pump desk — pump.fun intelligence: intake, dossiers, the
de-rug gate, and calls that grade themselves in the open.

An evidence engine with an auditable ledger, not a call-shill bot:
launches stream in over free feeds, candidates earn dossiers built
from chain evidence, a computed gate (never the brain's arithmetic)
decides what is callable, and every published call is scored
against what the market actually did.  Signals only — the desk
contains no buying/selling machinery, no wallet key handling,
nothing that signs a transaction.
"""

from __future__ import annotations

from typing import Any

from .desk import PumpDesk, render_calls, render_status
from .dossier import render_dossier
from .jobs import ensure_pumpdesk_jobs
from .store import PumpStore, pumpdesk_dir

__all__ = [
    "PumpDesk", "PumpStore", "pumpdesk_dir", "ensure_pumpdesk_jobs",
    "render_calls", "render_dossier", "render_status",
    "register",
]


def register(registry: Any) -> None:
    """Register the ``pumpdesk`` agent tool (watch / study / status
    / dossier / calls / watchlist / provision state)."""
    from ...core.policy import Capability

    context = registry.context

    @registry.register(
        "pumpdesk",
        description=(
            "Pump.fun intelligence desk: action=board (the ranked "
            "board of live candidates: every dossiered mint by its "
            "current verdict, filtered — min_composite / max_rug / "
            "min_momentum / only_candidates / only_watched / limit; "
            "watched mints pin first) | open (open calls with live "
            "progress: current SOL mcap from the observations "
            "ledger, progress to target, distance to invalidation, "
            "time to expiry) | watch (one intake + evaluation "
            "cycle: discovery candidates (DexScreener profiles "
            "first, pump.fun frontend as fallback), dossier "
            "refresh, attention verification, de-rug gate, publish "
            "calls) | "
            "study (grade due calls, recompute dev profiles, rebuild "
            "the smart-money graph, recalibrate score parameters "
            "from the graded journal, shelve dossiers into the "
            "Library) | status (launches, dossiers, call "
            "calibration vs the graduation base rate, wallet graph "
            "+ learning state, feed health) | dossier (mint — full "
            "evidence record for one token) | calls (the call "
            "ledger) | wallet (wallet — the graph's realized-PnL "
            "record for one wallet) | graph (the smart-money graph: "
            "graded wallets + cohorts) | attention (mint — the "
            "attention verifier's verdict: organic vs manufactured) "
            "| learn (scoring parameter version + recalibration "
            "journal) | shelf (query — search the Library shelf of "
            "dossiers/wallets) | watch_dev / watch_smart / "
            "watch_mint (wallet/mint — the watchlists; a watched "
            "mint pins first on the board) | unwatch "
            "(wallet/mint — stop tracking it) | provision (the "
            "desk's keyed RPC tier state). Signals only — never trades."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — board|open|watch|study|status|dossier|"
                      "calls|wallet|graph|attention|learn|shelf|"
                      "watch_dev|watch_smart|watch_mint|unwatch|"
                      "provision",
            "mint": "str — token mint for action=dossier/attention/"
                    "watch_mint",
            "wallet": "str — wallet for watch/wallet actions",
            "query": "str — search text for action=shelf",
            "note": "str — note attached to a watch action",
            "deliver": "bool — publish calls when action=watch",
            "min_composite": "float — board filter, default 0",
            "max_rug": "float — board filter, default 1",
            "min_momentum": "float — board filter, default 0",
            "only_candidates": "bool — board filter",
            "only_watched": "bool — board filter",
            "limit": "int — board rows, default 10",
        },
    )
    def pumpdesk(*, action: str = "status", mint: str = "",
                 wallet: str = "", query: str = "", note: str = "",
                 deliver: bool = False,
                 min_composite: float = 0.0, max_rug: float = 1.0,
                 min_momentum: float = 0.0,
                 only_candidates: bool = False,
                 only_watched: bool = False, limit: int = 10
                 ) -> dict[str, Any]:
        act = (action or "status").strip().lower()
        desk = PumpDesk(context)
        if act == "status":
            return desk.status()
        if act == "board":
            return {"ok": True, **desk.board_view(
                min_composite=float(min_composite or 0.0),
                max_rug=float(max_rug if max_rug is not None else 1.0),
                min_momentum=float(min_momentum or 0.0),
                only_candidates=bool(only_candidates),
                only_watched=bool(only_watched),
                limit=int(limit or 10))}
        if act == "open":
            return {"ok": True, **desk.open_view()}
        if act == "watch":
            return desk.watch_cycle(deliver=bool(deliver))
        if act == "study":
            return desk.study_cycle()
        if act == "dossier":
            if not mint:
                return {"error": "action=dossier needs a mint"}
            d = desk.refresh_dossier(mint.strip())
            verdict = None
            from .score import evaluate
            verdict = evaluate(d, smart_wallets=desk.store.watched_smart())
            return {"dossier": d, "verdict": verdict.to_dict(),
                    "rendered": render_dossier(d)}
        if act == "calls":
            return {"calls": [c.to_dict()
                              for c in desk.calls.log.calls()],
                    "calibration": desk.status()["calls"]}
        if act == "wallet":
            if not wallet:
                return {"error": "action=wallet needs a wallet"}
            return desk.wallet_view(wallet.strip())
        if act == "graph":
            return desk.graph_view()
        if act == "attention":
            if not mint:
                return {"error": "action=attention needs a mint"}
            return desk.attention_view(mint.strip())
        if act == "learn":
            return desk.learn_view()
        if act == "shelf":
            if not query:
                return {"error": "action=shelf needs a query"}
            return desk.shelf_view(query.strip())
        if act in ("watch_dev", "watch_smart"):
            if not wallet:
                return {"error": f"action={act} needs a wallet"}
            if act == "watch_dev":
                desk.store.watch_dev(wallet.strip(), note=note)
            else:
                desk.store.watch_smart(wallet.strip(), note=note)
            return {"watched": wallet.strip(),
                    "watchlists": desk.store.watchlists()}
        if act == "watch_mint":
            from .board import valid_mint
            m = (mint or "").strip()
            if not m:
                return {"error": "action=watch_mint needs a mint"}
            if not valid_mint(m):
                return {"error": f"{m!r} is not a valid token mint"}
            desk.store.watch_mint(m, note=note)
            return {"watched": m,
                    "watchlists": desk.store.watchlists()}
        if act == "unwatch":
            target = (wallet or mint or "").strip()
            if not target:
                return {"error":
                        "action=unwatch needs a wallet or mint"}
            return {"removed": desk.store.unwatch(target),
                    "watchlists": desk.store.watchlists()}
        if act == "provision":
            from .sources import ensure_provisioning
            ids = ensure_provisioning()
            import os
            return {"services": ids,
                    "helius_key_present": bool(
                        os.environ.get("HELIUS_API_KEY", "").strip()),
                    "pumpportal_key_present": bool(
                        os.environ.get("PUMPPORTAL_API_KEY",
                                       "").strip()),
                    "drive": "run /pump provision helius or "
                             "/pump provision pumpportal in chat — "
                             "the drive needs the chat browser "
                             "session; the key is vaulted (Helius is "
                             "validated live; PumpPortal exposes no "
                             "GET key check, so its validation is "
                             "inconclusive by design)"}
        return {"error": f"unknown action {act!r} — board|open|watch|"
                         "study|status|dossier|calls|wallet|graph|"
                         "attention|learn|shelf|watch_dev|watch_smart|"
                         "watch_mint|unwatch|provision"}
