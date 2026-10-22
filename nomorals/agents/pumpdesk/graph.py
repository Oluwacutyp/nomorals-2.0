"""The smart-money graph — wallets graded by what they DID.

P1's confluence was set intersection: a wallet on the owner's
smart list appearing among a mint's buyers counted, on the list's
authority alone.  P2 replaces the label with a ledger (design
§3.1/§3.2, §5-P2):

* **Realized PnL, FIFO-attributed.**  Every wallet's decoded
  trades are matched buy-lot → sell, per mint, in time order.
  Realized PnL is the SOL that actually came back minus the SOL
  basis of the tokens sold.  Sells whose tokens were never bought
  in the observed stream (transferred in, or bought before the
  desk watched) have no attributable basis — their proceeds are
  NOT counted as profit; they are counted and named as
  ``basis_unknown_sells``.  A wallet earns "smart" only from
  closed, attributable, recent evidence.
* **Recent over lifetime.**  The record splits realized PnL into
  a trailing-30-day window and lifetime: lifetime leaders can be
  washers (design §3.2); the verdict reads the recent window
  first.
* **Cohorts, not vibes.**  Wallets are clustered only on observed
  co-entry evidence: buying the same mint in the same slot, or
  sharing block-0 absorption on the same launch, repeated across
  launches — the persistent-cohort pattern (design §9.3) and the
  substrate of the KOL copy-trap (§3.1): linked wallets trading
  ahead of attention are one entity wearing several addresses.
  Transfer/funding links are NOT claimed: the desk's feeds do not
  observe plain transfers, and every wallet record says so
  (``funding: unknown``) instead of pretending otherwise.
* **Momentum-lead.**  A wallet whose first buy on a mint precedes
  a ≥2x price rise inside 30 minutes, repeatedly, is leading
  momentum the desk can see — the observable residue of the
  front-run pattern.  Counted per wallet, from the desk's own
  observation series.

Everything here is computed from the desk's ledgers; a wallet the
desk has never observed has no record, and "no record" is the
record — confluence reads track records, never labels.
"""

from __future__ import annotations

import time
from typing import Any, Optional

__all__ = [
    "RECENT_WINDOW_S", "MIN_CLOSED_TRADES", "build_wallet_graph",
    "wallet_record", "graph_confluence", "render_wallet",
    "render_graph",
]

#: "Recent" PnL window — the verdict reads this first (§3.2).
RECENT_WINDOW_S = 30 * 86400.0
#: Floors: below this many closed (fully-exited) positions a
#: wallet's grade is "unknown" — no assertion from thin evidence.
MIN_CLOSED_TRADES = 3
#: A momentum-lead: first buy, then >= this multiple within the
#: window, from the observation series.
LEAD_MULTIPLE = 2.0
LEAD_WINDOW_S = 30 * 60.0

LAMPORTS = 1_000_000_000


def _fifo_wallet(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """FIFO-match one wallet's trades (all mints) into realized
    PnL.  Returns the raw aggregates the record is built from."""
    lots: dict[str, list[list[float]]] = {}   # mint -> [[tokens, sol_per_token]]
    realized = 0.0
    realized_recent = 0.0
    closed: list[tuple[float, float]] = []    # (exit_ts, pnl_sol)
    basis_unknown_sells = 0
    open_cost = 0.0
    buys = sells = 0
    now_ref = max((float(t.get("timestamp") or 0) for t in trades),
                  default=0.0)
    for t in sorted(trades, key=lambda r: float(r.get("timestamp") or 0)):
        mint = t.get("mint", "")
        toks = float(int(t.get("token_amount", 0)))
        sol = float(int(t.get("sol_amount", 0))) / LAMPORTS
        ts = float(t.get("timestamp") or 0)
        if toks <= 0:
            continue
        if t.get("is_buy"):
            buys += 1
            lots.setdefault(mint, []).append([toks, sol / toks])
            open_cost += sol
            continue
        sells += 1
        remaining = toks
        pnl = 0.0
        matched = 0.0
        queue = lots.get(mint) or []
        while remaining > 0 and queue:
            lot = queue[0]
            take = min(lot[0], remaining)
            pnl += take * (sol / toks - lot[1])
            matched += take
            lot[0] -= take
            remaining -= take
            open_cost -= take * lot[1]
            if lot[0] <= 0:
                queue.pop(0)
        if matched < toks:
            basis_unknown_sells += 1
        if matched > 0:
            realized += pnl
            if now_ref and ts >= now_ref - RECENT_WINDOW_S:
                realized_recent += pnl
            if not queue and matched >= toks:
                closed.append((ts, pnl))
    return {
        "realized_sol": round(realized, 4),
        "realized_recent_sol": round(realized_recent, 4),
        "closed_trades": len(closed),
        "closed_wins": sum(1 for _ts, p in closed if p > 0),
        "basis_unknown_sells": basis_unknown_sells,
        "open_cost_sol": round(max(open_cost, 0.0), 4),
        "buys": buys, "sells": sells,
    }


def _grade(agg: dict[str, Any]) -> str:
    """The verdict a wallet's record earns — floors first."""
    if agg["closed_trades"] < MIN_CLOSED_TRADES:
        return "unknown"
    recent = agg["realized_recent_sol"]
    win_rate = (agg["closed_wins"] / agg["closed_trades"]
                if agg["closed_trades"] else 0.0)
    if recent > 0 and win_rate >= 0.5:
        return "smart"
    if recent < 0:
        return "losing"
    if agg["realized_sol"] > 0 and win_rate >= 0.5:
        return "smart"          # proven, just not recently active
    return "mixed"


def _wallet_weight(rec: dict[str, Any]) -> float:
    """Confluence weight 0..1 from the record alone: recent
    realized PnL dominates, win rate supports, sample size gates
    confidence.  An unknown wallet weighs 0 — never smart by
    assertion — and so does a losing one: a wallet whose recent
    record is net-negative is evidence AGAINST, and confluence
    counts it as zero rather than letting the formula's floor
    leak it a positive weight."""
    if rec.get("verdict") in (None, "unknown", "losing"):
        return 0.0
    if float(rec.get("realized_recent_sol", 0.0)) <= 0 and \
            float(rec.get("realized_sol", 0.0)) <= 0:
        return 0.0
    closed = int(rec.get("closed_trades", 0))
    confidence = min(1.0, closed / 5.0)
    pnl_term = max(-0.35, min(0.35, float(
        rec.get("realized_recent_sol", 0.0)) / 100.0))
    wr = float(rec.get("win_rate", 0.0))
    wr_term = max(-0.2, min(0.2, (wr - 0.5))) * confidence
    return round(max(0.0, min(1.0, 0.5 + pnl_term + wr_term)), 4)


def _cohorts(store: Any, wallets: dict[str, Any]) -> list[dict[str, Any]]:
    """Cluster wallets on repeated co-entry evidence: same mint +
    same slot buys, or shared block-0 absorption, across >= 2
    launches.  Union-find over the evidence pairs."""
    pair_evidence: dict[tuple[str, str], dict[str, Any]] = {}
    by_mint: dict[str, list[dict[str, Any]]] = {}
    for mint in {t.get("mint") for t in store.trades()}:
        if mint:
            by_mint[mint] = store.trades(mint)
    launches = {l.get("mint"): l for l in store.launches()}
    for mint, rows in by_mint.items():
        create_slot = int((launches.get(mint) or {}).get("slot") or 0)
        slot_buyers: dict[int, set[str]] = {}
        block0: set[str] = set()
        for t in rows:
            if not t.get("is_buy"):
                continue
            slot = int(t.get("slot") or 0)
            if slot:
                slot_buyers.setdefault(slot, set()).add(t.get("user", ""))
            if create_slot and slot == create_slot:
                block0.add(t.get("user", ""))
        groups = [s for s in slot_buyers.values() if len(s) >= 2]
        if len(block0) >= 2:
            groups.append(block0)
        for group in groups:
            members = sorted(w for w in group if w)
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    ev = pair_evidence.setdefault(
                        (a, b), {"mints": set(), "co_entries": 0})
                    ev["mints"].add(mint)
                    ev["co_entries"] += 1
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    linked: dict[tuple[str, str], dict[str, Any]] = {}
    for (a, b), ev in pair_evidence.items():
        if len(ev["mints"]) >= 2 or ev["co_entries"] >= 3:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
            linked[(a, b)] = ev
    clusters: dict[str, set[str]] = {}
    for a, b in linked:
        clusters.setdefault(find(a), set()).update({a, b})
    out = []
    for root, members in clusters.items():
        ev_mints: set[str] = set()
        co = 0
        for (a, b), ev in linked.items():
            if a in members and b in members:
                ev_mints |= ev["mints"]
                co += ev["co_entries"]
        out.append({
            "id": f"cohort-{root[:8]}",
            "members": sorted(members),
            "evidence": {
                "shared_launches": len(ev_mints),
                "co_entry_events": co,
                "basis": "repeated same-slot co-entry / shared "
                         "block-0 absorption across launches",
                "funding": "unknown — the desk's feeds do not "
                           "observe plain transfers; no funding "
                           "link is claimed",
            },
        })
    return out


def _momentum_leads(store: Any, wallet: str) -> int:
    """Count mints where this wallet's first observed buy precedes
    a >= LEAD_MULTIPLE rise inside LEAD_WINDOW_S (observation
    series) — the front-run pattern's observable residue."""
    leads = 0
    mints = {t.get("mint") for t in store.trades()
             if t.get("user") == wallet and t.get("is_buy")}
    for mint in mints:
        if not mint:
            continue
        first_buy = min((float(t.get("timestamp") or 0)
                         for t in store.trades(mint)
                         if t.get("user") == wallet and t.get("is_buy")),
                        default=0.0)
        if not first_buy:
            continue
        obs = store.observations(mint, since=first_buy,
                                 until=first_buy + LEAD_WINDOW_S)
        if len(obs) < 2:
            continue
        base = obs[0]["price"]
        if base > 0 and max(o["price"] for o in obs) >= LEAD_MULTIPLE * base:
            leads += 1
    return leads


def build_wallet_graph(store: Any,
                       now: Optional[float] = None) -> dict[str, Any]:
    """Rebuild the whole graph from the trade ledger (the study
    cycle's job).  Deterministic over the ledger rows."""
    now = time.time() if now is None else now
    all_trades = store.trades()
    by_wallet: dict[str, list[dict[str, Any]]] = {}
    for t in all_trades:
        user = t.get("user", "")
        if user:
            by_wallet.setdefault(user, []).append(t)
    wallets: dict[str, Any] = {}
    for wallet, rows in by_wallet.items():
        agg = _fifo_wallet(rows)
        mints = {t.get("mint") for t in rows if t.get("mint")}
        rec = {
            "wallet": wallet,
            **agg,
            "win_rate": round(agg["closed_wins"] / agg["closed_trades"], 4)
            if agg["closed_trades"] else 0.0,
            "tokens_traded": len(mints),
            "first_seen": min((float(t.get("timestamp") or 0)
                               for t in rows), default=0.0),
            "last_seen": max((float(t.get("timestamp") or 0)
                              for t in rows), default=0.0),
            "momentum_leads": _momentum_leads(store, wallet),
            "funding": "unknown — transfers are not observed by "
                       "the desk's feeds",
            "cohorts": [],
        }
        rec["verdict"] = _grade(rec)
        rec["weight"] = _wallet_weight(rec)
        wallets[wallet] = rec
    clusters = _cohorts(store, wallets)
    for cl in clusters:
        for member in cl["members"]:
            if member in wallets:
                wallets[member]["cohorts"].append(cl["id"])
    graph = {"built_at": now, "wallets": wallets, "clusters": clusters}
    store.save_graph(graph)
    return graph


def wallet_record(store: Any, wallet: str) -> Optional[dict[str, Any]]:
    return store.graph().get("wallets", {}).get(wallet)


def graph_confluence(store: Any, mint: str,
                     graph: Optional[dict[str, Any]] = None
                     ) -> dict[str, Any]:
    """Confluence for one mint, read from track records: which of
    the mint's buyers have EARNED weight in the graph, and how
    much.  Cohort members count once per cohort (linked wallets
    are one entity — §3.1), using the cohort's best record."""
    graph = graph if graph is not None else store.graph()
    wallets = graph.get("wallets", {})
    buyers = {t.get("user") for t in store.trades(mint)
              if t.get("is_buy") and t.get("user")}
    entries = []
    counted_cohorts: set[str] = set()
    weighted = 0.0
    hits = 0
    for w in sorted(buyers):
        rec = wallets.get(w)
        if rec is None:
            entries.append({"wallet": w, "verdict": "no-record",
                            "weight": 0.0})
            continue
        cohort_ids = set(rec.get("cohorts") or [])
        if cohort_ids & counted_cohorts:
            entries.append({"wallet": w, "verdict": rec["verdict"],
                            "weight": 0.0,
                            "note": "cohort already counted — linked "
                                    "wallets are one entity"})
            continue
        counted_cohorts |= cohort_ids
        weight = float(rec.get("weight", 0.0))
        if weight > 0:
            hits += 1
            weighted += weight
        entries.append({"wallet": w, "verdict": rec["verdict"],
                        "weight": weight,
                        "realized_recent_sol":
                            rec.get("realized_recent_sol", 0.0),
                        "closed_trades": rec.get("closed_trades", 0)})
    return {
        "mint": mint,
        "hits": hits,
        "weighted": round(min(1.0, weighted / 1.5), 4),
        "buyers_seen": len(buyers),
        "entries": entries,
    }


def render_wallet(rec: Optional[dict[str, Any]], wallet: str = "") -> str:
    if rec is None:
        return (f"wallet {wallet or '?'}: no record — the desk has "
                f"never observed it trade a watched mint. No record "
                f"is the record: it is not smart money here, it is "
                f"unknown.")
    L = [f"👛 {rec['wallet'][:16]}… — verdict: {rec['verdict']} "
         f"(weight {rec.get('weight', 0):.2f})"]
    L.append(f"realized PnL: {rec['realized_sol']:+.2f} SOL lifetime, "
             f"{rec['realized_recent_sol']:+.2f} SOL last 30d "
             f"(attributable, FIFO from observed trades)")
    L.append(f"closed positions: {rec['closed_trades']} "
             f"({rec['closed_wins']} wins, win rate "
             f"{rec['win_rate']:.0%}); open cost basis "
             f"{rec['open_cost_sol']:.2f} SOL; "
             f"{rec['basis_unknown_sells']} sell(s) with unattributable "
             f"basis (excluded from PnL)")
    L.append(f"activity: {rec['buys']} buys / {rec['sells']} sells "
             f"across {rec['tokens_traded']} tokens; momentum leads: "
             f"{rec['momentum_leads']}")
    if rec.get("cohorts"):
        L.append(f"cohorts: {', '.join(rec['cohorts'])} (linked "
                 f"wallets trade as one entity in confluence)")
    L.append(f"funding: {rec['funding']}")
    return "\n".join(L)


def render_graph(graph: dict[str, Any]) -> str:
    wallets = graph.get("wallets", {})
    graded = [w for w in wallets.values()
              if w.get("verdict") != "unknown"]
    smart = [w for w in graded if w["verdict"] == "smart"]
    losing = [w for w in graded if w["verdict"] == "losing"]
    L = [f"🕸️ smart-money graph: {len(wallets)} wallets observed, "
         f"{len(graded)} graded ({len(smart)} smart, {len(losing)} "
         f"losing), {len(graph.get('clusters', []))} cohort(s)"]
    for w in sorted(smart, key=lambda r: -r["realized_recent_sol"])[:8]:
        L.append(f"  smart {w['wallet'][:12]}… "
                 f"{w['realized_recent_sol']:+.1f} SOL/30d, "
                 f"{w['closed_trades']} closed, "
                 f"{w['win_rate']:.0%} wins")
    for cl in graph.get("clusters", [])[:5]:
        L.append(f"  {cl['id']}: {len(cl['members'])} wallets, "
                 f"{cl['evidence']['shared_launches']} shared "
                 f"launches, {cl['evidence']['co_entry_events']} "
                 f"co-entry events")
    if not graded:
        L.append("  no wallet has the closed-trade sample to be "
                 "graded yet — confluence runs on records, and the "
                 "records are still thin")
    return "\n".join(L)
