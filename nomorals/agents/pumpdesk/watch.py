"""Watchlists — the owner's eyes on the actors, not just tokens.

The lists themselves, their use as intake priority (a watched
dev's launch earns a dossier), the serial-rugger auto-flag
feeding the dev list, and watched-wallet activity checks the
watch cycle reports.  ``smart_confluence`` below is the P1
set-intersection count, kept as the fallback for dossiers the
wallet graph has not covered; P2's track-record confluence —
wallets weighted by earned, realized-PnL records, cohorts
counted once — lives in ``graph.py`` and supersedes it in the
desk's refresh path whenever the graph covers a mint's buyers.
"""

from __future__ import annotations

from typing import Any

__all__ = ["watch_activity", "smart_confluence"]


def watch_activity(store: Any, since: float) -> list[dict[str, Any]]:
    """What watched actors did since ``since``: launches by watched
    devs, and graduations of anything they launched.  These are
    DIGEST-grade events (Pass-7: noteworthy, not pageable) — the
    watch cycle reports them, the owner reads them batched."""
    out: list[dict[str, Any]] = []
    devs = store.watched_devs()
    if not devs:
        return out
    for launch in store.launches():
        if launch.get("creator") not in devs:
            continue
        ts = float(launch.get("ts") or 0)
        if ts >= since:
            out.append({"kind": "watched-dev-launch",
                        "mint": launch.get("mint", ""),
                        "creator": launch.get("creator", ""),
                        "symbol": launch.get("symbol", ""), "ts": ts})
        elif launch.get("graduated"):
            out.append({"kind": "watched-dev-graduated",
                        "mint": launch.get("mint", ""),
                        "creator": launch.get("creator", ""),
                        "symbol": launch.get("symbol", ""), "ts": ts})
    return out


def smart_confluence(store: Any, mint: str,
                     smart_wallets: set[str]) -> int:
    """How many watched smart wallets appear as buyers in this
    mint's decoded trade ledger — the P1 confluence input to
    scoring.  Funding-graph clustering (one entity behind many
    wallets) is P2; this count is labelled simple for a reason."""
    if not smart_wallets:
        return 0
    buyers = {t.get("user") for t in store.trades(mint)
              if t.get("is_buy")}
    return len(buyers & smart_wallets)
