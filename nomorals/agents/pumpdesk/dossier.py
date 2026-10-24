"""The token dossier — evidence, traced, per watched mint.

One living record per candidate token, assembled from chain reads
and the desk's own ledgers.  Every field carries its source; a
field the desk could not measure says so (``missing``) instead of
wearing an estimate as a fact.  All money figures are in the curve's
quote asset (SOL for standard pairs) — never USD (design §7.3).

Dev profiles are the desk's memory of creators: every launch the
firehose saw, classified by outcome (graduated / rugged / dead /
alive / unknown) from observation evidence, with the rug ratio and
graduation rate the de-rug gate reads.  The serial-rugger headline
(>= 3 launches, >= 67% rugged — the G-BOT rule the mining adopted)
is computed here, in the open.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from . import chain
from .failures import record_failure

__all__ = [
    "SERIAL_MIN_LAUNCHES", "SERIAL_RUG_RATIO",
    "HOLDER_DERIVE_MAX_WALLETS",
    "classify_launch_outcome", "build_dev_profile",
    "refresh_dev_profiles", "build_dossier", "render_dossier",
]

#: Serial-rugger headline rule (design §8.1, G-BOT blueprint).
SERIAL_MIN_LAUNCHES = 3
SERIAL_RUG_RATIO = 0.67

#: Cap on the ledger-derived holder fallback: distinct ledger
#: wallets whose live balances are read when largest-accounts is
#: unreachable.  The reads are per-wallet RPC calls; the cap keeps
#: a dossier build bounded, and the picture reports how many it
#: checked — coverage is stated, never implied complete.
HOLDER_DERIVE_MAX_WALLETS = 40

LAMPORTS_PER_SOL = 1_000_000_000


def _sol(lamports: float) -> float:
    return lamports / LAMPORTS_PER_SOL


# ── launch outcome classification ─────────────────────────────────────

def classify_launch_outcome(store: Any, launch: dict[str, Any],
                              now: Optional[float] = None) -> str:
    """Classify one launch from the desk's observation evidence.

    graduated — the migration/pool evidence landed;
    rugged    — it ran (curve >= 8% or price >= 3x birth) and then
                collapsed to <= 25% of its peak price: the dump is
                in the observations, not in a vibe;
    dead      — never reached 5% curve and silent for 12h+;
    alive     — observed recently and none of the above;
    unknown   — not enough observation evidence to say."""
    now = time.time() if now is None else now
    if launch.get("graduated"):
        return "graduated"
    mint = launch.get("mint", "")
    obs = store.observations(mint)
    if not obs:
        return "unknown"
    birth = float(launch.get("birth_price") or obs[0]["price"] or 0.0)
    peak_price = max(o["price"] for o in obs)
    peak_pct = max(o["curve_pct"] for o in obs)
    last = obs[-1]
    ran = peak_pct >= 0.08 or (birth > 0 and peak_price >= 3.0 * birth)
    if ran and peak_price > 0 and last["price"] <= 0.25 * peak_price:
        return "rugged"
    if peak_pct < 0.05 and now - last["ts"] > 12 * 3600:
        return "dead"
    if now - last["ts"] <= 6 * 3600:
        return "alive"
    return "unknown"


def build_dev_profile(store: Any, creator: str,
                      now: Optional[float] = None) -> dict[str, Any]:
    """A creator's record from the launch ledger + outcomes."""
    now = time.time() if now is None else now
    launches = store.launches_by_creator(creator)
    outcomes: dict[str, int] = {}
    for launch in launches:
        oc = classify_launch_outcome(store, launch, now=now)
        outcomes[oc] = outcomes.get(oc, 0) + 1
    known = sum(outcomes.get(k, 0)
                for k in ("graduated", "rugged", "dead"))
    rugged = outcomes.get("rugged", 0)
    graduated = outcomes.get("graduated", 0)
    rug_ratio = (rugged / known) if known else 0.0
    return {
        "creator": creator,
        "launches": len(launches),
        "outcomes": outcomes,
        "known_outcomes": known,
        "rug_ratio": round(rug_ratio, 4),
        "graduation_rate": round(graduated / known, 4) if known else 0.0,
        "serial_rugger": bool(len(launches) >= SERIAL_MIN_LAUNCHES
                              and known >= SERIAL_MIN_LAUNCHES
                              and rug_ratio >= SERIAL_RUG_RATIO),
        "first_seen": min((l.get("ts", 0) for l in launches),
                          default=0.0),
        "updated_at": now,
    }


def refresh_dev_profiles(store: Any,
                         now: Optional[float] = None) -> dict[str, Any]:
    """Recompute every known creator's profile (the study cycle)."""
    now = time.time() if now is None else now
    creators = {l.get("creator") for l in store.launches()
                if l.get("creator")}
    profiles = {c: build_dev_profile(store, c, now=now) for c in creators}
    store.save_dev_profiles(profiles)
    return profiles


# ── trade-derived facts ───────────────────────────────────────────────

def trade_facts(trades: list[dict[str, Any]], creator: str,
                create_slot: int = 0,
                sellable_raw: int = 0) -> dict[str, Any]:
    """Block-0 absorption, unique buyers, bump score, dev selling —
    computed from decoded TradeEvents, every number reproducible
    from the ledger rows it came from."""
    buyers: set[str] = set()
    sellers: set[str] = set()
    buy_sol = sell_sol = 0
    slot0_tokens = 0        # non-creator tokens absorbed in create slot
    slot0_total = 0         # all tokens absorbed in create slot
    dev_sold_sol = 0.0
    per_wallet: dict[str, dict[str, float]] = {}
    first_ts = min((t.get("timestamp", 0) for t in trades), default=0)
    last_ts = max((t.get("timestamp", 0) for t in trades), default=0)
    for t in trades:
        user = t.get("user", "")
        sol = int(t.get("sol_amount", 0))
        toks = int(t.get("token_amount", 0))
        w = per_wallet.setdefault(user, {"buys": 0, "sells": 0,
                                         "net_tokens": 0.0, "flips": 0,
                                         "last_side": ""})
        if t.get("is_buy"):
            buyers.add(user)
            buy_sol += sol
            w["buys"] += 1
            w["net_tokens"] += toks
            if w["last_side"] == "sell":
                w["flips"] += 1
            w["last_side"] = "buy"
            if create_slot and int(t.get("slot", -1)) == int(create_slot):
                slot0_total += toks
                if user != creator:
                    slot0_tokens += toks
        else:
            sellers.add(user)
            sell_sol += sol
            w["sells"] += 1
            w["net_tokens"] -= toks
            if w["last_side"] == "buy":
                w["flips"] += 1
            w["last_side"] = "sell"
            if user == creator:
                dev_sold_sol += sol / LAMPORTS_PER_SOL
    # bump score alpha = F / (DeltaP + eps) per wallet (arXiv
    # 2601.08641): repetitive flips with ~zero net position change
    # is the wash signature.  Report the worst wallet's alpha in
    # whole-token terms.
    worst_alpha = 0.0
    bump_wallets = 0
    for w in per_wallet.values():
        if w["flips"] >= 2:
            alpha = w["flips"] / (abs(w["net_tokens"]) / 1e6 + 1.0)
            worst_alpha = max(worst_alpha, alpha)
            if alpha >= 2.0:
                bump_wallets += 1
    span_min = max(1.0, (last_ts - first_ts) / 60.0) if trades else 0.0
    return {
        "trades": len(trades),
        "unique_buyers": len(buyers),
        "unique_sellers": len(sellers),
        "buy_sol": round(_sol(buy_sol), 4),
        "sell_sol": round(_sol(sell_sol), 4),
        "buy_sell_ratio": round(buy_sol / sell_sol, 3) if sell_sol else None,
        "block0_absorption_share": (
            round(slot0_tokens / sellable_raw, 4)
            if sellable_raw > 0 and create_slot else None),
        "block0_total_share": (
            round(slot0_total / sellable_raw, 4)
            if sellable_raw > 0 and create_slot else None),
        "dev_sold_sol": round(dev_sold_sol, 4),
        "dev_sold": bool(dev_sold_sol > 0),
        "bump_alpha_max": round(worst_alpha, 3),
        "bump_wallets": bump_wallets,
        "velocity_sol_per_min": (round(_sol(buy_sol) / span_min, 3)
                                 if trades else 0.0),
        "sol_per_trade": (round(_sol(buy_sol) / max(1, len(trades)), 4)
                          if trades else 0.0),
    }


# ── holder concentration ──────────────────────────────────────────────

def _route_of(rpc: Any, method: str) -> str:
    """Which endpoint served a read (RoutedRPC provenance), or
    the single client's endpoint for a plain SolanaRPC."""
    served = getattr(rpc, "served_by", None)
    if isinstance(served, dict) and served.get(method):
        return str(served[method])
    return str(getattr(rpc, "endpoint", "") or "")


def _holder_picture(store: Any, rpc: Any, mint: str, supply: int,
                    creator: str = "") -> tuple[Optional[dict], str]:
    """(holders, gap) — holder concentration by two real routes.

    Route 1: ``getTokenLargestAccounts`` through the desk's routed
    RPC (a provisioned Helius route first when keyed, then the
    registry's public routes in health order — chain.RoutedRPC
    fails over per call, so one endpoint's throttle is no longer
    the whole story).  Route 2, when route 1 fails on EVERY route
    or answers empty: a PARTIAL picture derived from the desk's
    own ledger — the live balances of the wallets the trade ledger
    has actually seen on this mint (plus its creator), read via
    getTokenAccountsByOwner, the read that stays up on the free
    floor when largest-accounts is throttled.  Partial shares are
    computed against TOTAL supply, so they are lower bounds on the
    true concentration (unobserved holders can only add), and the
    picture says so: ``partial`` + how many wallets were checked
    and what share of supply they account for.  Both routes dead
    → (None, a gap naming both)."""
    if not supply:
        return None, "holder concentration unavailable (supply unknown)"
    largest_error = ""
    try:
        largest = rpc.get_token_largest(mint)
    except Exception as exc:  # noqa: BLE001 - route 2 exists for this
        largest, largest_error = [], f"{type(exc).__name__}: {exc}"
    if largest:
        amounts = sorted((int(a.get("amount", 0)) for a in largest),
                         reverse=True)
        if amounts and amounts[0] > 0:
            return {
                "top1_share": round(amounts[0] / supply, 4),
                "top3_share": round(sum(amounts[:3]) / supply, 4),
                "top10_share": round(sum(amounts[:10]) / supply, 4),
                "accounts_seen": len(amounts),
                "partial": False,
                "source": "largest-accounts",
                "route": _route_of(rpc, "get_token_largest"),
            }, ""
    # route 2: derive from the desk's own observed wallets
    by_owner = getattr(rpc, "get_token_accounts_by_owner", None)
    if by_owner is None:
        return None, ("holder concentration unavailable "
                      "(largest-accounts unreachable"
                      + (f": {largest_error}" if largest_error else "")
                      + "; this RPC exposes no by-owner read)")
    wallets: list[str] = []
    seen: set[str] = set()
    for t in store.trades(mint):
        user = t.get("user", "")
        if user and user not in seen:
            seen.add(user)
            wallets.append(user)
    if creator and creator not in seen:
        wallets.append(creator)
    checked = wallets[:HOLDER_DERIVE_MAX_WALLETS]
    balances: list[int] = []
    read_error = ""
    for wallet in checked:
        try:
            accounts = by_owner(wallet, mint)
        except Exception as exc:  # noqa: BLE001
            read_error = f"{type(exc).__name__}: {exc}"
            continue
        bal = sum(int(a.get("amount", 0)) for a in accounts or [])
        if bal > 0:
            balances.append(bal)
    if not balances:
        detail = "largest-accounts unreachable"
        if largest_error:
            detail += f" ({largest_error})"
        detail += (f"; ledger-derived read found no live balances "
                   f"across {len(checked)} observed wallet(s)")
        if read_error:
            detail += f" (last read error: {read_error})"
        return None, f"holder concentration unavailable ({detail})"
    balances.sort(reverse=True)
    observed = sum(balances)
    return {
        "top1_share": round(balances[0] / supply, 4),
        "top3_share": round(sum(balances[:3]) / supply, 4),
        "top10_share": round(sum(balances[:10]) / supply, 4),
        "accounts_seen": len(balances),
        "partial": True,
        "source": "ledger-derived",
        "route": _route_of(rpc, "get_token_accounts_by_owner"),
        "wallets_checked": len(checked),
        "wallets_holding": len(balances),
        "observed_supply_share": round(observed / supply, 4),
        "note": ("partial picture: shares are lower bounds from "
                 "the wallets the desk's ledger has observed; "
                 "unobserved holders are not counted"),
    }, ""


# ── dossier assembly ──────────────────────────────────────────────────

def build_dossier(store: Any, rpc: Any, mint: str,
                  http_get: Any = None,
                  now: Optional[float] = None) -> dict[str, Any]:
    """Assemble/refresh the dossier for one mint.

    Chain reads are the spine (curve account, mint account, largest
    holders); the desk's ledgers supply history (launch record,
    trades, dev profile).  ``rpc`` is duck-typed (SolanaRPC or a
    test double).  Fields that cannot be measured are marked
    missing in ``gaps`` — the score reads the gaps, not hopes."""
    now = time.time() if now is None else now
    launch = store.launch(mint) or {}
    gaps: list[str] = []

    state = None
    try:
        state = rpc.get_curve(mint)
    except Exception as exc:  # noqa: BLE001 - a failed read is a gap
        gaps.append(f"curve read failed: {exc}")
    if state is None:
        gaps.append("curve account not found (not a pump token, "
                    "or the account is gone)")

    mint_info = None
    try:
        mint_info = rpc.get_mint_info(mint)
    except Exception:  # noqa: BLE001
        gaps.append("mint account unreadable")
    decimals = mint_info.decimals if mint_info else 6

    dossier: dict[str, Any] = {
        "mint": mint,
        "name": launch.get("name", ""), "symbol": launch.get("symbol", ""),
        "uri": launch.get("uri", ""),
        "creator": (state.creator if state else launch.get("creator", "")),
        "created_slot": launch.get("slot", 0),
        "created_at": launch.get("ts", 0.0),
        "source": launch.get("source", ""),
        "updated_at": now,
        "gaps": gaps,
    }

    if state is not None:
        pct, basis = chain.curve_pct(
            state, int(launch.get("initial_real_token_reserves") or 0))
        price = chain.price_quote_per_token(state, decimals)
        mc = chain.market_cap_quote(state)
        dossier["curve"] = {
            "state": state.to_dict(),
            "curve_pct": round(pct, 4), "curve_pct_basis": basis,
            "price_sol": price / LAMPORTS_PER_SOL,
            "market_cap_sol": round(mc / LAMPORTS_PER_SOL, 3),
            "real_sol": round(_sol(state.real_sol_reserves), 3),
            # virtual/real reserve divergence (P2): the share of
            # the curve's apparent SOL depth that is neither the
            # fixed 30 SOL virtual advance nor net real money —
            # churn that washed through and out.  The score's
            # manipulation component reads it.
            "virtual_sol": round(_sol(state.virtual_sol_reserves), 3),
            "reserve_divergence": round(max(
                0.0, (state.virtual_sol_reserves
                      - 30 * LAMPORTS_PER_SOL
                      - state.real_sol_reserves)
                / state.virtual_sol_reserves), 4)
            if state.virtual_sol_reserves > 0 else None,
            "complete_flag": state.complete,
            "quote_mint": state.quote_mint,
        }
        # an observation is the scoring series' raw material
        store.record_observation(mint, now, price, mc, pct,
                                 source="rpc-read")
        dossier["curve"]["graduated"] = bool(
            launch.get("graduated")) or _pool_exists(rpc, state, mint)
    else:
        dossier["curve"] = None

    if mint_info is not None:
        dossier["authorities"] = {
            "mint_authority_live": mint_info.mint_authority is not None,
            "freeze_authority_live": mint_info.freeze_authority is not None,
            "supply": mint_info.supply, "decimals": mint_info.decimals,
        }
    else:
        dossier["authorities"] = None
        gaps.append("authority flags unknown")

    # holder concentration (top accounts vs total supply)
    supply_hint = (mint_info.supply if mint_info else
                   (state.token_total_supply if state else 0))
    holders, holder_gap = _holder_picture(
        store, rpc, mint, supply_hint, dossier["creator"])
    dossier["holders"] = holders
    if holder_gap:
        gaps.append(holder_gap)

    # dev profile (the desk's memory of this creator)
    creator = dossier["creator"]
    profiles = store.dev_profiles()
    profile = profiles.get(creator)
    if profile is None and creator:
        profile = build_dev_profile(store, creator, now=now)
    dossier["dev"] = profile or {"creator": creator, "launches": 0,
                                 "known_outcomes": 0, "rug_ratio": 0.0,
                                 "graduation_rate": 0.0,
                                 "serial_rugger": False}
    if not creator:
        gaps.append("creator unknown")

    # trade facts from the desk's own decoded ledger
    trades = store.trades(mint)
    sellable = 0
    if state is not None:
        sellable = int(state.token_total_supply
                       * chain.SELLABLE_FRACTION)
    dossier["trades"] = trade_facts(
        trades, creator, create_slot=int(launch.get("slot") or 0),
        sellable_raw=sellable)
    if not trades:
        gaps.append("no trades decoded yet (watched flow only)")

    store.save_dossier(dossier)
    return dossier


def _pool_exists(rpc: Any, state: chain.BondingCurveState,
                 mint: str) -> bool:
    """Graduation evidence: the canonical PumpSwap pool account
    exists (§7.5 — the CompleteEvent alone proves nothing)."""
    try:
        pool = chain.pumpswap_pool_pda(state.creator, mint,
                                       state.quote_mint)
        return rpc.get_account(pool) is not None
    except Exception as exc:  # noqa: BLE001 - unprovable is not graduated
        record_failure("dossier._pool_exists", exc)
        return False


def render_dossier(d: dict[str, Any]) -> str:
    L = [f"🪙 {d.get('symbol') or '?'} — {d.get('name') or 'unnamed'}"]
    L.append(f"mint: {d.get('mint')}")
    c = d.get("curve")
    if c:
        L.append(f"curve: {c['curve_pct']:.1%} sold "
                 f"({c['curve_pct_basis']}), {c['real_sol']:.1f} SOL in, "
                 f"mcap {c['market_cap_sol']:.1f} SOL, "
                 f"price {c['price_sol']:.8f} SOL")
        if c.get("virtual_sol") is not None:
            L.append(f"reserves: {c['virtual_sol']:.0f} SOL virtual / "
                     f"{c['real_sol']:.1f} SOL real (divergence "
                     f"{(c.get('reserve_divergence') or 0):.0%})")
        L.append(f"graduated: {'YES' if c.get('graduated') else 'no'}"
                 f" (curve complete flag: {c['complete_flag']})")
    else:
        L.append("curve: unreadable")
    a = d.get("authorities")
    if a:
        L.append(f"authorities: mint {'LIVE' if a['mint_authority_live'] else 'revoked'}, "
                 f"freeze {'LIVE' if a['freeze_authority_live'] else 'revoked'}")
    h = d.get("holders")
    if h:
        if h.get("partial"):
            L.append(f"holders (partial, ledger-derived): top1 "
                     f"{h['top1_share']:.0%}, top10 "
                     f"{h['top10_share']:.0%} of supply across "
                     f"{h.get('wallets_holding', 0)} holding "
                     f"wallet(s) of {h.get('wallets_checked', 0)} "
                     f"observed — lower bounds; observed wallets "
                     f"account for "
                     f"{(h.get('observed_supply_share') or 0):.0%} "
                     f"of supply")
        else:
            L.append(f"holders: top1 {h['top1_share']:.0%}, "
                     f"top10 {h['top10_share']:.0%} "
                     f"({h['accounts_seen']} largest accounts)")
    dev = d.get("dev") or {}
    if dev.get("serial_rugger"):
        L.append(f"dev: ⚠️ SERIAL RUGGER — {dev.get('launches', 0)} "
                 f"launches, rug ratio {dev.get('rug_ratio', 0):.0%}")
    else:
        L.append(f"dev: {dev.get('launches', 0)} launches, "
                 f"rug ratio {dev.get('rug_ratio', 0):.0%} "
                 f"({dev.get('known_outcomes', 0)} known outcomes), "
                 f"graduation {dev.get('graduation_rate', 0):.0%}")
    t = d.get("trades") or {}
    L.append(f"trades decoded: {t.get('trades', 0)} "
             f"({t.get('unique_buyers', 0)} unique buyers, "
             f"buy {t.get('buy_sol', 0)} SOL / sell {t.get('sell_sol', 0)} SOL)")
    if t.get("block0_absorption_share") is not None:
        L.append(f"block-0 absorption (non-dev): "
                 f"{t['block0_absorption_share']:.0%} of sellable supply")
    if t.get("bump_alpha_max"):
        L.append(f"bump score: worst wallet alpha "
                 f"{t['bump_alpha_max']} ({t.get('bump_wallets', 0)} "
                 f"wash-flagged)")
    if d.get("gaps"):
        L.append("gaps: " + "; ".join(d["gaps"]))
    return "\n".join(L)
