"""Calls — published ahead, logged append-only, graded in the open.

A port of the betdesk signal ledger's shape (design §2.1/§4.1) to
memecoin dossiers: a call exists only when the de-rug gate returns
CALL_CANDIDATE, it is written to the ledger BEFORE delivery is
attempted (an undelivered call still happened and still scores),
and ``score_due`` grades every expired call against outcome
evidence — the desk's own observation series first (first-touch
ordering of target vs invalidation), a DexScreener snapshot when
the series is empty.  A call whose window produced NO outcome data
scores as a MISS: the desk does not get to leave its calls
ungraded because collection was thin (the betdesk rule).

Calls are owner-wanted standing content: they publish through the
Notifier un-tiered, exactly like betdesk signals and motivation
(Pass-7 discipline — the module's HEALTH alerts carry hints; its
calls are the product, not an interruption).

Calibration is part of the product: status reports calls / wins /
hit rate against the graduation base rate (~0.6–1.4% of launches
ever graduate — design §8) broken down by call class, so the owner
sees the desk's edge, or its absence, in numbers.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

__all__ = [
    "BASE_RATE_GRADUATION", "TARGET_MULTIPLE", "INVALIDATION_FRACTION",
    "CALL_TTL_SECONDS", "PumpCall", "CallLog", "CallEngine",
    "render_call",
]

#: The honesty anchor: the share of ALL pump.fun launches that ever
#: graduate (design §8: 0.63% arXiv sample; ~1.4% Messari).  The
#: desk's hit rate is reported against the top of that range.
BASE_RATE_GRADUATION = 0.014

#: A call targets a doubling of the SOL market cap from entry...
TARGET_MULTIPLE = 2.0
#: ...and is invalidated when the cap falls to this fraction of
#: entry (or the dev sells — scored from the trade ledger).
INVALIDATION_FRACTION = 0.55
#: Calls expire: memecoin windows are short (median hold ~100s on
#: the platform; a desk call gets hours, not days).
CALL_TTL_SECONDS = 6 * 3600


@dataclass
class PumpCall:
    mint: str
    symbol: str = ""
    name: str = ""
    call_class: str = "dossier"        # dossier | watch | convergence
    published_at: float = 0.0
    expires_at: float = 0.0
    entry_price_sol: float = 0.0       # per whole token, SOL
    entry_mc_sol: float = 0.0
    entry_curve_pct: float = 0.0
    target_mc_sol: float = 0.0
    invalidation_mc_sol: float = 0.0
    composite: float = 0.0
    catalyst: str = ""
    risks: str = ""
    narrative: str = ""
    delivered: bool = False
    # scoring (None until graded)
    scored: Optional[bool] = None      # True = win
    outcome: str = ""                  # win|loss|miss-no-data
    peak_mc_sol: float = 0.0
    peak_at: float = 0.0
    max_favorable: float = 0.0         # peak / entry (multiple)
    max_adverse: float = 0.0           # entry / trough (drawdown frac)
    graduated: bool = False
    graded_via: str = ""               # observations|dexscreener|none
    call_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PumpCall":
        return cls(**{k: v for k, v in (d or {}).items()
                      if k in cls.__dataclass_fields__})


class CallLog:
    """Append-only call ledger (JSONL) with score-back rewrite —
    the SignalLog shape."""

    def __init__(self, desk_dir: str):
        self.path = os.path.join(desk_dir, "pump_calls.jsonl")
        os.makedirs(desk_dir, exist_ok=True)

    def add(self, call: PumpCall) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(call.to_dict()) + "\n")

    def calls(self) -> list[PumpCall]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(PumpCall.from_dict(json.loads(line)))
                    except (ValueError, TypeError):
                        continue
        return out

    def rewrite(self, calls: list[PumpCall]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for c in calls:
                fh.write(json.dumps(c.to_dict()) + "\n")
        os.replace(tmp, self.path)


class CallEngine:
    def __init__(self, desk_dir: str, notifier: Any = None,
                 ttl_seconds: int = CALL_TTL_SECONDS):
        self.log = CallLog(desk_dir)
        self.notifier = notifier
        self.ttl = int(ttl_seconds)

    # ── making a call ─────────────────────────────────────────────

    def build_call(self, dossier: dict[str, Any], verdict: Any,
                   now: Optional[float] = None,
                   call_class: str = "dossier") -> Optional[PumpCall]:
        """A call exists only off a CALL_CANDIDATE verdict with a
        live curve read.  Anything else returns None — the caller
        reports the no-call reason from the verdict itself."""
        now = time.time() if now is None else now
        if getattr(verdict, "verdict", "") != "CALL_CANDIDATE":
            return None
        curve = dossier.get("curve") or {}
        entry_mc = float(curve.get("market_cap_sol") or 0)
        if entry_mc <= 0:
            return None
        risks = []
        if verdict.gates:
            risks.extend(verdict.gates)
        dev = dossier.get("dev") or {}
        if dev.get("launches"):
            risks.append(f"dev history: {dev['launches']} launches, "
                         f"rug ratio {dev.get('rug_ratio', 0):.0%}")
        trades = dossier.get("trades") or {}
        if trades.get("block0_absorption_share"):
            risks.append(f"block-0 absorption "
                         f"{trades['block0_absorption_share']:.0%}")
        call = PumpCall(
            mint=dossier.get("mint", ""),
            symbol=dossier.get("symbol", ""), name=dossier.get("name", ""),
            call_class=call_class,
            published_at=now, expires_at=now + self.ttl,
            entry_price_sol=float(curve.get("price_sol") or 0),
            entry_mc_sol=entry_mc,
            entry_curve_pct=float(curve.get("curve_pct") or 0),
            target_mc_sol=entry_mc * TARGET_MULTIPLE,
            invalidation_mc_sol=entry_mc * INVALIDATION_FRACTION,
            composite=float(getattr(verdict, "composite", 0.0)),
            catalyst=getattr(verdict, "evidence", "")[:400],
            risks="; ".join(risks)[:400],
            call_id=f"pump-{dossier.get('mint', '')[:8]}-{int(now)}")
        return call

    def publish(self, call: PumpCall, deliver: bool = True) -> dict[str, Any]:
        """Ledger first, delivery second (the betdesk order)."""
        self.log.add(call)
        out: dict[str, Any] = {"call": call.to_dict(), "delivered": False}
        if not deliver:
            out["detail"] = "logged, not delivered (deliver=False)"
            return out
        if self.notifier is None:
            out["detail"] = ("logged; no notifier bound — the call "
                             "exists but reached no surface")
            return out
        try:
            res = self.notifier.publish(
                "pumpdesk", f"🚀 Pump call: {call.symbol or call.mint[:8]}",
                render_call(call))
            call.delivered = bool(res.get("sent") or res.get("delivered")
                                  or res.get("ok"))
            out["delivered"] = call.delivered
            out["detail"] = f"notifier: {res}"
        except Exception as exc:  # noqa: BLE001 - delivery failure is data
            out["detail"] = f"notifier failed: {exc}"
        if out["delivered"]:
            calls = self.log.calls()
            for c in calls:
                if c.call_id == call.call_id:
                    c.delivered = True
            self.log.rewrite(calls)
        return out

    # ── grading ───────────────────────────────────────────────────

    def score_due(self, store: Any, snapshot_fn: Any = None,
                  now: Optional[float] = None) -> dict[str, Any]:
        """Grade every expired call.  Evidence order:

        1. the desk's own observation series inside the window —
           first touch of target vs invalidation decides, peak and
           trough give the excursions;
        2. a DexScreener snapshot (graduated venue or current price
           vs entry) — coarser, and graded_via says so;
        3. nothing at all -> a MISS (never left ungraded)."""
        now = time.time() if now is None else now
        calls = self.log.calls()
        scored_now = wins_now = 0
        changed = False
        for c in calls:
            if c.scored is not None or c.expires_at > now:
                continue
            self._grade(c, store, snapshot_fn)
            scored_now += 1
            wins_now += 1 if c.scored else 0
            changed = True
        if changed:
            self.log.rewrite(calls)
        return calibration(calls, scored_now=scored_now, wins_now=wins_now)

    def _grade(self, c: PumpCall, store: Any, snapshot_fn: Any) -> None:
        obs = store.observations(c.mint, since=c.published_at,
                                 until=c.expires_at)
        launch = store.launch(c.mint) or {}
        c.graduated = bool(launch.get("graduated"))
        if obs:
            c.graded_via = "observations"
            peak = max(obs, key=lambda o: o["mc"])
            trough = min(obs, key=lambda o: o["mc"])
            c.peak_mc_sol = round(peak["mc"] / 1e9, 3)
            c.peak_at = peak["ts"]
            c.max_favorable = round(
                (peak["mc"] / 1e9) / c.entry_mc_sol, 3) \
                if c.entry_mc_sol else 0.0
            c.max_adverse = round(
                1.0 - (trough["mc"] / 1e9) / c.entry_mc_sol, 3) \
                if c.entry_mc_sol else 0.0
            first_target = next(
                (o for o in obs if o["mc"] / 1e9 >= c.target_mc_sol), None)
            first_inv = next(
                (o for o in obs
                 if o["mc"] / 1e9 <= c.invalidation_mc_sol), None)
            if c.graduated:
                c.scored, c.outcome = True, "win"
            elif first_target is not None and (
                    first_inv is None
                    or first_target["ts"] <= first_inv["ts"]):
                c.scored, c.outcome = True, "win"
            elif first_inv is not None:
                c.scored, c.outcome = False, "loss"
            else:
                c.scored, c.outcome = False, "loss"
            return
        snap = None
        if snapshot_fn is not None:
            try:
                snap = snapshot_fn(c.mint)
            except Exception:  # noqa: BLE001 - a dead grader is no data
                snap = None
        if snap:
            c.graded_via = "dexscreener"
            c.graduated = c.graduated or snap.get("dex") in (
                "pumpswap", "pumpfun")
            ratio = (snap.get("price_native", 0.0) / c.entry_price_sol
                     if c.entry_price_sol else 0.0)
            c.max_favorable = round(max(ratio, 0.0), 3)
            c.peak_mc_sol = round(c.entry_mc_sol * ratio, 3) if ratio else 0.0
            if c.graduated or ratio >= TARGET_MULTIPLE:
                c.scored, c.outcome = True, "win"
            else:
                c.scored, c.outcome = False, "loss"
            return
        c.graded_via = "none"
        c.scored, c.outcome = False, "miss-no-data"


def calibration(calls: list[PumpCall], scored_now: int = 0,
                wins_now: int = 0) -> dict[str, Any]:
    """The desk's track record, by call class, against the base
    rate — the accountability surface (design §4.1/§9.5)."""
    done = [c for c in calls if c.scored is not None]
    by_class: dict[str, dict[str, int]] = {}
    for c in done:
        row = by_class.setdefault(c.call_class, {"calls": 0, "wins": 0})
        row["calls"] += 1
        row["wins"] += 1 if c.scored else 0
    for row in by_class.values():
        row["hit_rate"] = round(row["wins"] / row["calls"], 4) \
            if row["calls"] else None
    return {
        "scored_now": scored_now, "wins_now": wins_now,
        "calls_total": len(calls), "calls_scored": len(done),
        "wins_total": sum(1 for c in done if c.scored),
        "hit_rate": (round(sum(1 for c in done if c.scored) / len(done), 4)
                     if done else None),
        "base_rate_graduation": BASE_RATE_GRADUATION,
        "misses_no_data": sum(1 for c in done
                              if c.outcome == "miss-no-data"),
        "by_class": by_class,
    }


def render_call(c: PumpCall) -> str:
    from datetime import datetime, timezone
    expiry = datetime.fromtimestamp(c.expires_at,
                                    tz=timezone.utc).strftime("%H:%M UTC")
    L = [f"**{c.symbol or '?'}** ({c.name or 'unnamed'}) — dossier call, "
         f"score {c.composite:.2f}"]
    L.append(f"entry: {c.entry_mc_sol:.1f} SOL mcap at "
             f"{c.entry_curve_pct:.0%} of the curve")
    L.append(f"target: {c.target_mc_sol:.1f} SOL mcap (2.0x) — "
             f"graduation is the stretch outcome (~85 SOL in the curve)")
    L.append(f"invalidation: {c.invalidation_mc_sol:.1f} SOL mcap "
             f"or the dev selling")
    L.append(f"expires: {expiry}")
    if c.catalyst:
        L.append(f"why: {c.catalyst}")
    if c.risks:
        L.append(f"risks: {c.risks}")
    if c.narrative:
        L.append(c.narrative)
    L.append(f"mint: {c.mint}")
    L.append("analysis only — the desk never trades; every call is "
             "graded in the open (/pump calls)")
    return "\n".join(L)
