"""Watchlists — the owner's eyes on the actors, not just tokens.

P1 scope (design §5 puts the full smart-money graph in P2): the
lists themselves, their use as intake priority (a watched dev's
launch earns a dossier), the serial-rugger auto-flag feeding the
dev list, and watched-wallet activity checks the watch cycle
reports.  The smart list already feeds scoring today — as simple
confluence (a watched smart wallet among a dossier's buyers),
honestly labelled; the realized-PnL ledger that would *earn* a
wallet its place on the list is P2's graph work, and nothing here
pretends otherwise.
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
