"""Demo execution + position management — Devon trading system, Pass 4.

The chain this pass builds, end to end:

    setup TRIGGERED (Pass 2 lifecycle) + judgment ACT (Pass 3)
        -> hard invariants (here, outside and above judgment)
        -> doctrine sizing
        -> order on the DEMO rung (paper simulator, or the Exness
           demo account through the desk's live path)
        -> per-playbook management (poll-driven, on fresh quotes)
        -> every intent, refusal, fill, management action and
           close journaled to the Pass 1 trading journal.

DEMO ONLY. There is no real-money routing in this pass: the live
desk path is entered only after the account is verified DEMO from
the connector's own account details, and a verification failure
fails closed. The paper rung never touches a broker at all.

THE ENGINE (consolidation verdict, from the Pass 4 scoping): three
paper-broker paths existed — the desk's paper engine, the Sentinel
PaperBroker sessions behind ``tools/trading``'s ``paper_*`` actions,
and the desk's live path. The system's engine is the **TradingDesk**
(``nomorals/finance/trading_desk.py``): it is the only path that
reaches the real Exness connector, owns the kill switch (the HALT
file) and already journals orders/fills. The Sentinel paper
sessions stay what they always were — a manual research sandbox
driven by the FinancialExpert strategy engine over CCXT data — and
are NOT a system execution path: nothing here reads them, and the
system never executes through them. The system's desk instance
carries the system policy (see :func:`system_policy`): the
doctrine invariants below are enforced HERE, above the desk, and
the desk's own mined gates (daily/weekly loss kills, streak
adaptation) are configured off for the system desk so the
drawdown halt is the sole account halt, as doctrine settles. The
manual desk surfaces keep their own policy untouched. The system
desk also gets its own state file (``trading_desk_system.json``) so
the two desks never overwrite each other's policy/positions; the
HALT kill-switch file is shared (same finance home — one switch).

HARD INVARIANTS (outside and above judgment — judgment cannot
waive any of them, and an abstention or a judgment failure refuses
execution outright):

- account mode: DEMO verified structurally, fail closed;
- kill switch: the desk's HALT file;
- friction gate: measured spread / stop distance > 0.10 refuses,
  journaled — the setup stays ALIVE (the lifecycle owns its death;
  a later pass may retry when the spread narrows);
- drawdown: 15% from the peak observed equity is the sole account
  halt (equity observations are journaled, so the peak survives
  restarts);
- at most 3 open system positions; at most 6 consecutive losses;
- the pre-order check: every system order carries a TradeIdea
  built from the setup plan + judgment verdict through the desk's
  order path, where ``ta/analyst.executor_check`` finally runs for
  real (its docstring claimed this call for years; Pass 4 wired
  it — a failing check blocks the order).

SIZING (doctrine): the risk rule is the owner's lot-step rule —
0.01 lot per step of equity, scaled UP by judgment confidence
inside the variant's band — converted by the stop distance and the
instrument's real contract specs into the journaled monetary risk,
rounded DOWN to the broker step, with a hard FLOOR at the minimum
lot: a setup is never abandoned over size. The A/B constants live
in :class:`SizingConfig` — variant A (0.01 per $1,000, confidence
x1–2) is the default because it is the conservative one; variant B
(0.01 per $500, x1–3) is one config flip when the owner picks. His
pick is still open; nothing here blocks on it.

EXECUTION HOME = BOTH RUNGS: Exness REST (working — paper sim +
demo live) and the MT5 bridge adapter shape
(``nomorals/connectors/mt5_bridge.py``), which reports honestly
unavailable where no terminal exists; an execution attempt on
that rung is a journaled refusal, never a fake fill.

MANAGEMENT is poll-driven on the freshest quotes the data organs
provide — there is no tick stream in Devon's Exness REST path, so
excursion R is the CURRENT excursion at poll time, named as such
in the journal (``current_r``), never a fabricated tick high.
Profiles (declared by the playbooks in Pass 2, executed here):

- ``qm_fixed_bracket``: the placement bracket ONLY — no profit
  lock, no breakeven move, no trail, no partials. The desk's own
  SL/TP marking is the whole management.
- ``staged_runner`` (hot-scalp analogue): staged partials at the
  plan's targets (50% at T1, 25% at T2, the rest is the runner)
  and an EARNED profit-lock: it arms only on an M5 acceptance
  close beyond T1 or the 2.0R backstop, at
  entry + max(0.50R, 0.75 ATR), capped at the entry zone's
  profit-side edge. An unproven push arms nothing.
- generic (``zone_retest`` / ``sweep_retest`` / ``range_mid``):
  profit-lock at best R >= 1.0 or >= 50% progress toward T1
  (same level formula, one-shot), and partial tiers 1R 50% /
  2R 25% / 3R 25% of the original volume — ONLY when the original
  volume is >= 0.03 lots.

Live-rung honesty: the Exness connector supports partial closes
(it carries a volume) but exposes NO modify for an open position,
so on the demo-live rung a profit-lock is journaled as designed
but marked ``applied: False`` with the reason — never pretended.
Broker closes are priced from the broker's closing record where
one exists (the desk journal's own close record on the paper
rung; the deals history on the live rung), else the close is
marked ``snapshot``-priced.

Placement: ``agents/`` beside judgment/setups — the layer from
which the desk, the TA library and the market-data integrations
are all legal static imports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..finance import trading_journal
from ..finance.budgets import finance_paths
from ..finance.trading_desk import DeskError, RiskPolicy
from ..integrations import market_data
from ..ta import indicators
from ..ta.analyst import TradeIdea, executor_check
from . import forecast as forecast_mod
from . import judgment as judgment_mod
from .playbooks import get_playbook, ts_epoch
from .setups import load_setups

_log = logging.getLogger(__name__)

# ── doctrine constants (the invariant numbers) ───────────────────────

FRICTION_LIMIT = 0.10          # spread / stop distance
MAX_DRAWDOWN_PCT = 15.0        # sole account halt
MAX_OPEN_POSITIONS = 3
MAX_CONSECUTIVE_LOSSES = 6
MIN_LOT = 0.01
PARTIAL_MIN_VOLUME = 0.03      # partial tiers need >= this to start
GENERIC_TIERS: tuple[tuple[float, float], ...] = (
    (1.0, 0.50), (2.0, 0.25), (3.0, 0.25))
SCALP_TIERS: tuple[tuple[int, float], ...] = ((0, 0.50), (1, 0.25))

REFUSAL_CATEGORIES: tuple[str, ...] = (
    "rung_unavailable", "judgment_missing", "judgment_error",
    "judgment_abstain", "kill_switch", "invalid_geometry",
    "pre_order_check", "account_not_demo", "friction_gate",
    "risk_unverifiable", "drawdown_halt", "max_positions",
    "loss_streak_halt", "contract_unknown", "desk_gate")


# ── sizing ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SizingConfig:
    """The lot-step rule's constants — ONE place, both variants.

    The owner's A/B pick is still open (it is his, on the
    GoldenSwing side too). Default = variant A, the conservative
    one. ``confidence`` scales the size UP inside the variant's
    band only; it never shrinks a trade below the base step.
    """

    variant: str = "A"  # pending the owner's pick — see docstring
    step_equity_a: float = 1000.0   # 0.01 lot per $1,000 equity
    scale_max_a: float = 2.0        # confidence scales x1–x2
    step_equity_b: float = 500.0    # 0.01 lot per $500 equity
    scale_max_b: float = 3.0        # confidence scales x1–x3

    def step_equity(self) -> float:
        return self.step_equity_b if self.variant == "B" \
            else self.step_equity_a

    def scale_max(self) -> float:
        return self.scale_max_b if self.variant == "B" \
            else self.scale_max_a


def doctrine_size(*, equity: float, entry: float, stop_loss: float,
                  contract_value: float, volume_min: float = MIN_LOT,
                  volume_max: float = 100.0, volume_step: float = MIN_LOT,
                  confidence: float | None = None,
                  config: SizingConfig | None = None
                  ) -> dict[str, Any]:
    """Doctrine volume for one setup. Never refuses over size.

    base = (equity / step_equity) x 0.01 lot, scaled up by the
    judgment confidence inside the variant band; rounded DOWN to
    the broker step; floored at the broker minimum (the setup is
    traded at the minimum, never abandoned); capped at the broker
    maximum. The stop distance x contract specs give the monetary
    risk, journaled with the order.
    """
    cfg = config or SizingConfig()
    conf = 0.0 if confidence is None else max(0.0, min(1.0,
                                                       float(confidence)))
    factor = 1.0 + (cfg.scale_max() - 1.0) * conf
    raw = (float(equity) / cfg.step_equity()) * MIN_LOT * factor
    steps = int(raw // volume_step) if volume_step > 0 else 0
    volume = round(steps * volume_step, 8)
    floored = False
    if volume < volume_min:
        volume = float(volume_min)
        floored = True
    capped = False
    if volume > volume_max:
        volume = float(volume_max)
        capped = True
    stop_distance = abs(float(entry) - float(stop_loss))
    risk_amount = stop_distance * float(contract_value) * volume \
        if contract_value and contract_value > 0 else None
    return {"volume": volume, "raw_volume": round(raw, 6),
            "confidence_factor": round(factor, 4),
            "variant": cfg.variant, "step_equity": cfg.step_equity(),
            "floored_to_min": floored, "capped_at_max": capped,
            "stop_distance": stop_distance,
            "contract_value": contract_value,
            "risk_amount": (round(risk_amount, 2)
                            if risk_amount is not None else None)}


def system_policy() -> RiskPolicy:
    """The desk policy for the SYSTEM desk instance.

    The doctrine invariants live in this module, above the desk;
    the desk's own daily/weekly loss kills and streak adaptation
    are configured off here so the 15% drawdown halt is the sole
    account halt on system trades (the manual desk keeps its mined
    policy — separate instance, separate state file). The desk's
    structural gates stay: stop-loss required, position count,
    kill-switch file.
    """
    return RiskPolicy(
        max_risk_pct_per_trade=1.0,
        max_open_positions=MAX_OPEN_POSITIONS,
        max_daily_loss_pct=100.0,
        max_weekly_loss_pct=100.0,
        require_stop_loss=True,
        live_unlocked=False,
        adapt_on_streaks=False,
        consec_loss_halt_n=0)


def build_system_desk(connector: Any, mode: str, *,
                      settings: Any = None) -> Any:
    """A TradingDesk shaped for the trading system (see module doc)."""
    from ..finance.trading_desk import TradingDesk

    _ledger, budgets_path = finance_paths(settings)
    state = Path(budgets_path).parent / "trading_desk_system.json"
    return TradingDesk(connector, policy=system_policy(), mode=mode,
                       state_path=state)


# ── small helpers ────────────────────────────────────────────────────

def _spread_from_quote(quote: dict[str, Any] | None) -> float | None:
    """Measured spread from a quote dict, or None when unmeasurable."""
    if not quote:
        return None
    bid, ask = quote.get("bid"), quote.get("ask")
    if bid and ask:
        try:
            return float(ask) - float(bid)
        except (TypeError, ValueError):
            return None
    bps, price = quote.get("spread_bps"), quote.get("price")
    if bps and price:
        try:
            return float(price) * float(bps) / 10000.0
        except (TypeError, ValueError):
            return None
    return None


def _floor_step(value: float, step: float) -> float:
    if step <= 0:
        return round(float(value), 8)
    return round(int(float(value) // step) * step, 8)


# ── the engine ───────────────────────────────────────────────────────

class ExecutionEngine:
    """Executes judged setups on the demo rung and manages them.

    ``desk`` is the system TradingDesk (paper mode = the simulator
    rung; live mode = the Exness demo rung — demo is verified per
    order). With ``desk=None`` + an ``mt5_adapter``, every
    execution is a journaled rung-unavailable refusal. Quotes and
    bars are injectable seams (tests, and Pass 6's scheduler will
    drive the same passes); the defaults are the Pass 1 data
    organs.
    """

    def __init__(self, desk: Any = None, *,
                 settings: Any = None, journal_path: Any = None,
                 sizing: SizingConfig | None = None,
                 now: Any = None,
                 quote_fn: Any = None, bars_fn: Any = None,
                 mt5_adapter: Any = None) -> None:
        self.desk = desk
        self.mt5_adapter = mt5_adapter
        self.settings = settings
        self.journal_path = journal_path
        self.sizing = sizing or SizingConfig()
        self._now = now or time.time
        self._quote_fn = quote_fn
        self._bars_fn = bars_fn

    # ── plumbing ──

    @property
    def rung(self) -> str:
        if self.desk is None:
            return "mt5_bridge"
        return "paper" if self.desk.mode == "paper" else "exness_demo"

    def _journal(self, event: str, **fields: Any) -> dict[str, Any]:
        return trading_journal.append_event(
            event, settings=self.settings, path=self.journal_path,
            **fields)

    def _events(self, event: str | None = None) -> list[dict[str, Any]]:
        return trading_journal.read_events(
            limit=0, event=event, settings=self.settings,
            path=self.journal_path)

    def _quote(self, symbol: str, market: str) -> dict[str, Any]:
        try:
            if self._quote_fn is not None:
                return self._quote_fn(symbol, market) or {}
            return market_data.quote(symbol, market=market) or {}
        except Exception as exc:  # noqa: BLE001 - marked, not fatal
            _log.info("execution: quote failed for %s: %s", symbol, exc)
            return {}

    def _bars(self, symbol: str, market: str, timeframe: str) -> Any:
        try:
            if self._bars_fn is not None:
                return self._bars_fn(symbol, market, timeframe)
            return market_data.get_ohlcv(
                symbol, market=market, timeframe=timeframe, bars=160)
        except Exception as exc:  # noqa: BLE001 - management degrades
            _log.info("execution: bars failed for %s %s: %s",
                      symbol, timeframe, exc)
            return None

    def _atr(self, pos: dict[str, Any]) -> float:
        """Fresh entry-timeframe ATR, else the ATR stored at open."""
        frame = self._bars(pos["symbol"], pos.get("market") or "forex",
                           pos.get("entry_timeframe") or "1h")
        if frame is not None and len(frame) >= 15:
            try:
                value = float(indicators.atr(frame).iloc[-1])
                if value > 0:
                    return value
            except Exception as exc:  # noqa: BLE001 - stored fallback
                _log.debug("execution: ATR failed: %s", exc)
        try:
            return float(pos.get("atr") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # ── the position book: a fold of the journal ──

    def load_positions(self) -> list[dict[str, Any]]:
        """All system positions (open + closed), creation order."""
        positions: dict[str, dict[str, Any]] = {}
        for rec in reversed(self._events()):  # read_events: newest 1st
            event = rec.get("event") or ""
            pid = rec.get("position_id")
            if event == "position_opened" and pid:
                positions[pid] = {
                    **{k: v for k, v in rec.items()
                       if k not in ("event", "ts")},
                    "position_id": pid, "state": "open",
                    "opened_at": rec.get("ts"),
                    "realized_pnl": 0.0, "tiers_done": [],
                    "locked": False, "history": []}
                continue
            pos = positions.get(pid) if pid else None
            if pos is None:
                continue
            if event == "position_managed":
                action = rec.get("action")
                if action == "partial_close":
                    pos["volume"] = rec.get("remaining_volume",
                                            pos["volume"])
                    pos["realized_pnl"] = round(
                        pos["realized_pnl"] + float(rec.get("pnl") or 0),
                        2)
                    if rec.get("tier") is not None:
                        pos["tiers_done"].append(rec["tier"])
                elif action == "profit_lock" and rec.get("applied"):
                    pos["stop_loss"] = rec.get("level")
                    pos["locked"] = True
                elif action == "profit_lock":
                    pos["lock_unapplied"] = rec.get("reason")
                pos["history"].append(
                    {"ts": rec.get("ts"), "action": action})
            elif event == "position_closed":
                pos["state"] = "closed"
                pos["closed_at"] = rec.get("ts")
                pos["exit_price"] = rec.get("exit_price")
                pos["close_reason"] = rec.get("reason")
                pos["pnl"] = rec.get("pnl")
                pos["close_price_source"] = rec.get("price_source")
        return list(positions.values())

    def open_positions(self) -> list[dict[str, Any]]:
        return [p for p in self.load_positions() if p["state"] == "open"]

    def _loss_streak(self) -> int:
        closed = [p for p in self.load_positions()
                  if p["state"] == "closed"]
        closed.sort(key=lambda p: p.get("closed_at") or 0, reverse=True)
        streak = 0
        for pos in closed:
            pnl = pos.get("pnl")
            if pnl is None:
                pnl = pos.get("realized_pnl") or 0.0
            if pnl < 0:
                streak += 1
            elif pnl > 0:
                break
            # breakeven closes neither build nor break the streak
        return streak

    def _observe_equity(self, equity: float) -> float:
        """Journal an equity observation; return the running peak."""
        peak = float(equity)
        for rec in self._events("equity_observation"):
            try:
                peak = max(peak, float(rec.get("peak") or 0),
                           float(rec.get("equity") or 0))
            except (TypeError, ValueError):
                continue
        self._journal("equity_observation", equity=float(equity),
                      peak=peak, rung=self.rung)
        return peak

    # ── refusals ──

    def _refuse(self, setup: dict[str, Any], category: str,
                reason: str, *,
                judgment_rec: dict[str, Any] | None = None,
                extra: dict[str, Any] | None = None) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "setup_id": setup.get("setup_id"),
            "symbol": setup.get("symbol"),
            "playbook": setup.get("playbook"),
            "route": setup.get("route"),
            "reason_category": category, "reason": reason,
            "rung": self.rung}
        if judgment_rec is not None:
            fields["judgment"] = {
                "decision": judgment_rec.get("decision"),
                "reason_category": judgment_rec.get("reason_category"),
                "chosen_scenario": judgment_rec.get("chosen_scenario")}
        if extra:
            fields.update(extra)
        self._journal("execution_refused", **fields)
        return {"ok": False, "refused": True,
                "setup_id": setup.get("setup_id"),
                "reason_category": category, "reason": reason}

    # ── execution ──

    def _demo_check(self) -> tuple[bool, str]:
        """Structural demo verification for the desk's rung."""
        if self.desk.mode == "paper":
            return True, ("paper simulator rung — fills are simulated "
                          "against real candles; no account is touched")
        try:
            demo = bool(self.desk._is_demo_account())
        except Exception as exc:  # noqa: BLE001 - fail closed
            return False, (f"demo verification failed ({exc}) — "
                           f"failing closed")
        if demo:
            return True, "account verified DEMO from its own details"
        return False, ("account is NOT a demo account — this system "
                       "executes on demo only, and no real-money "
                       "routing exists in it")

    def build_idea(self, setup: dict[str, Any],
                   judgment_rec: dict[str, Any], entry: float,
                   stop: float) -> TradeIdea:
        """The order intent as the boundary law's TradeIdea.

        Built from the setup plan (geometry) and the judgment
        verdict (confidence, why) — the playbook/judgment pipeline
        is the analyst-grade discipline behind it, and the idea
        says so in its reasons rather than borrowing a signature
        silently (the TradeIdea format carries its v1 marker; the
        provenance is recorded here and in the journal).
        """
        plan = setup.get("plan") or {}
        targets = [float(t) for t in plan.get("targets") or []]
        weights = (0.5, 0.3, 0.2)
        idea_targets = [(t, weights[i] if i < len(weights) else 0.1)
                        for i, t in enumerate(targets)]
        conf = judgment_rec.get("confidence")
        confidence = round(float(conf) * 100.0, 1) \
            if isinstance(conf, (int, float)) else 0.0
        return TradeIdea(
            symbol=str(setup.get("symbol") or ""),
            side=int(setup.get("direction") or 0),
            confidence=confidence,
            entry=float(entry), invalidation=float(stop),
            targets=idea_targets,
            reasons=[str(judgment_rec.get("why") or "")],
            timeframe=str(setup.get("entry_timeframe") or ""),
            regime=str((judgment_rec.get("market_state") or {})
                       .get("structure_class") or ""))

    def execute_setup(self, setup: dict[str, Any],
                      judgment_rec: dict[str, Any] | None, *,
                      read: dict[str, Any] | None = None
                      ) -> dict[str, Any]:
        """Execute one triggered setup under one judgment verdict.

        Returns ``{"ok": True, "position": {...}}`` or a refusal
        dict (``ok`` False, ``reason_category`` from the invariant
        vocabulary). Every outcome is journaled.
        """
        # Rung availability first: an MT5 attempt is a refusal,
        # never a fake fill.
        if self.desk is None:
            status = self.mt5_adapter.status() \
                if self.mt5_adapter is not None else {}
            return self._refuse(
                setup, "rung_unavailable",
                "MT5 bridge rung is unavailable on this machine: "
                + str(status.get("reason") or "no terminal"),
                judgment_rec=judgment_rec)
        # Judgment gate — fail closed on absence and on error.
        if judgment_rec is None:
            return self._refuse(
                setup, "judgment_missing",
                "no judgment record for this setup — execution "
                "requires a Pass 3 verdict, so it fails closed")
        if judgment_rec.get("decision") == "error":
            return self._refuse(
                setup, "judgment_error",
                "judgment evaluation failed ("
                + str(judgment_rec.get("why") or "unknown error")
                + ") — failing closed", judgment_rec=judgment_rec)
        if judgment_rec.get("decision") != "act":
            return self._refuse(
                setup, "judgment_abstain",
                "judgment abstained ("
                + str(judgment_rec.get("reason_category") or "?")
                + "): " + str(judgment_rec.get("why") or ""),
                judgment_rec=judgment_rec)
        # Kill switch — the desk's HALT file, checked above all
        # order work (the desk re-checks it in its own gates too).
        if self.desk.kill_switch_engaged():
            return self._refuse(
                setup, "kill_switch",
                "kill switch engaged (HALT file present) — "
                "remove it to resume", judgment_rec=judgment_rec)
        # Geometry.
        plan = setup.get("plan") or {}
        direction = int(setup.get("direction") or 0)
        trigger = setup.get("trigger") or {}
        entry = trigger.get("price") or plan.get("entry_ref")
        stop = plan.get("invalidation")
        targets = plan.get("targets") or []
        if direction not in (1, -1) or entry is None or stop is None \
                or not targets:
            return self._refuse(
                setup, "invalid_geometry",
                "setup plan lacks direction, entry, invalidation or "
                "targets — there is no order to place",
                judgment_rec=judgment_rec)
        entry, stop = float(entry), float(stop)
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return self._refuse(
                setup, "invalid_geometry",
                "stop distance is zero — no definable risk",
                judgment_rec=judgment_rec)
        # The pre-order check, genuinely in the path (the desk
        # re-runs it at the order; here it categorizes the refusal).
        idea = self.build_idea(setup, judgment_rec, entry, stop)
        verdict = executor_check(idea)
        if not verdict.get("ok"):
            return self._refuse(
                setup, "pre_order_check",
                f"executor_check refused: {verdict.get('reason')}",
                judgment_rec=judgment_rec)
        # Demo-only, structurally.
        demo_ok, demo_detail = self._demo_check()
        if not demo_ok:
            return self._refuse(setup, "account_not_demo", demo_detail,
                                judgment_rec=judgment_rec)
        # Friction gate on the freshest quote. An unmeasurable
        # spread is recorded on the intent, not invented.
        symbol = str(setup.get("symbol") or "")
        market = str(setup.get("market") or "forex")
        quote = self._quote(symbol, market)
        spread = _spread_from_quote(quote)
        if spread is not None and spread / stop_distance > FRICTION_LIMIT:
            return self._refuse(
                setup, "friction_gate",
                f"friction gate: spread {spread:.2f} / stop "
                f"{stop_distance:.2f} = {spread / stop_distance:.3f} "
                f"> {FRICTION_LIMIT} — refused; the setup stays "
                f"alive for a later pass",
                judgment_rec=judgment_rec,
                extra={"spread": spread, "stop_distance": stop_distance})
        # Equity, drawdown (sole account halt), caps, streak.
        try:
            equity = float(self.desk.equity())
        except Exception as exc:  # noqa: BLE001 - fail closed
            return self._refuse(
                setup, "risk_unverifiable",
                f"account equity unverifiable ({exc}) — failing "
                f"closed", judgment_rec=judgment_rec)
        peak = self._observe_equity(equity)
        if peak > 0 and (peak - equity) / peak * 100.0 >= MAX_DRAWDOWN_PCT:
            return self._refuse(
                setup, "drawdown_halt",
                f"drawdown {(peak - equity) / peak * 100.0:.1f}% "
                f"from the peak equity {peak:,.2f} >= "
                f"{MAX_DRAWDOWN_PCT:.0f}% — the account halt",
                judgment_rec=judgment_rec,
                extra={"equity": equity, "peak_equity": peak})
        open_now = self.open_positions()
        if len(open_now) >= MAX_OPEN_POSITIONS:
            return self._refuse(
                setup, "max_positions",
                f"{len(open_now)} system positions already open "
                f"(max {MAX_OPEN_POSITIONS})",
                judgment_rec=judgment_rec)
        streak = self._loss_streak()
        if streak >= MAX_CONSECUTIVE_LOSSES:
            return self._refuse(
                setup, "loss_streak_halt",
                f"{streak} consecutive losses — halted until the "
                f"streak breaks", judgment_rec=judgment_rec)
        # Contract specs: sizing refuses to guess them.
        try:
            cond = self.desk.contract_specs(symbol)
            contract_value = float(self.desk._contract_value(symbol))
        except Exception as exc:  # noqa: BLE001 - fail closed
            return self._refuse(
                setup, "contract_unknown",
                f"contract specs unverifiable ({exc}) — sizing "
                f"refuses to guess", judgment_rec=judgment_rec)
        sizing = doctrine_size(
            equity=equity, entry=entry, stop_loss=stop,
            contract_value=contract_value,
            volume_min=float(cond.get("volume_min") or MIN_LOT),
            volume_max=float(cond.get("volume_max") or 100.0),
            volume_step=float(cond.get("volume_step") or MIN_LOT),
            confidence=judgment_rec.get("confidence"),
            config=self.sizing)
        # The intent lands BEFORE the order (the desk's own
        # live_open_attempt pattern): if the broker call raises
        # after accepting, the intent is the record to reconcile.
        self._journal(
            "order_intent", setup_id=setup.get("setup_id"),
            symbol=symbol, market=market,
            playbook=setup.get("playbook"), route=setup.get("route"),
            direction=direction, entry=entry, stop_loss=stop,
            targets=[float(t) for t in targets], sizing=sizing,
            spread=spread, rung=self.rung, demo_check=demo_detail,
            confirmation=("system_demo_authority"
                          if self.rung == "exness_demo" else "n/a"),
            judgment={"decision": judgment_rec.get("decision"),
                      "chosen_scenario":
                          judgment_rec.get("chosen_scenario"),
                      "expected_r": judgment_rec.get("expected_r"),
                      "confidence": judgment_rec.get("confidence")})
        side = "buy" if direction == 1 else "sell"
        outer_target = float(targets[-1])
        try:
            if self.rung == "paper":
                pos = self.desk.paper_open(
                    symbol, side, sizing["volume"], entry=entry,
                    stop_loss=stop, take_profit=outer_target, idea=idea)
                broker_id = pos.id
                fill_entry = float(pos.entry)
            else:
                res = self.desk.live_open(
                    symbol, side, sizing["volume"], stop_loss=stop,
                    take_profit=outer_target, entry=entry,
                    comment=f"devon-system:{setup.get('playbook')}",
                    confirmed=True, idea=idea)
                broker_id, fill_entry = self._live_fill(res, entry)
        except DeskError as exc:
            return self._refuse(setup, "desk_gate",
                                f"the desk refused: {exc}",
                                judgment_rec=judgment_rec)
        pb = get_playbook(str(setup.get("playbook") or ""))
        profile = pb.management_profile if pb is not None else (
            "staged_runner" if setup.get("family") == "scalp"
            else "qm_fixed_bracket"
            if "qm_" in str(setup.get("playbook") or "")
            else "zone_retest")
        zone = plan.get("entry_zone") or {}
        record = self._journal(
            "position_opened", position_id=broker_id,
            broker_position_id=broker_id,
            setup_id=setup.get("setup_id"), symbol=symbol,
            market=market, playbook=setup.get("playbook"),
            route=setup.get("route"), family=setup.get("family"),
            profile=profile, direction=direction, side=side,
            entry=fill_entry, stop_loss=stop,
            initial_stop=stop, targets=[float(t) for t in targets],
            volume=sizing["volume"], original_volume=sizing["volume"],
            volume_step=float(cond.get("volume_step") or MIN_LOT),
            contract_value=contract_value, sizing=sizing,
            entry_timeframe=setup.get("entry_timeframe"),
            entry_zone={"top": zone.get("top"),
                        "bottom": zone.get("bottom")},
            atr=(setup.get("evidence") or {}).get("atr"),
            spread=spread, rung=self.rung,
            ts=float(self._now()),
            judgment={"decision": judgment_rec.get("decision"),
                      "chosen_scenario":
                          judgment_rec.get("chosen_scenario"),
                      "expected_r": judgment_rec.get("expected_r"),
                      "confidence": judgment_rec.get("confidence")})
        position = {k: v for k, v in record.items()
                    if k not in ("event", "ts")}
        position["state"] = "open"
        return {"ok": True, "position": position}

    @staticmethod
    def _live_fill(res: dict[str, Any],
                   planned_entry: float) -> tuple[str, float]:
        """Broker position id + fill price from a live_open result.

        The Exness response shape varies by account; the id is read
        from the keys the API/broker layer uses, and the fill price
        falls back to the planned entry (marked as such in the
        journal by the caller's response record) when the response
        carries no price of its own.
        """
        resp = res.get("response") if isinstance(res, dict) else None
        broker_id = ""
        fill = planned_entry
        if isinstance(resp, dict):
            for key in ("position_id", "positionId", "id"):
                if resp.get(key):
                    broker_id = str(resp[key])
                    break
            inner = resp.get("position")
            if not broker_id and isinstance(inner, dict) \
                    and inner.get("id"):
                broker_id = str(inner["id"])
            for key in ("price", "entry", "open_price", "fill_price"):
                val = resp.get(key)
                if val:
                    try:
                        fill = float(val)
                        break
                    except (TypeError, ValueError):
                        continue
        if not broker_id:
            broker_id = f"unreconciled:{int(time.time() * 1000)}"
        return broker_id, fill

    # ── the execution pass ──

    def _latest_judgments(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for rec in self._events("judgment"):  # newest first
            subj = rec.get("subject") or {}
            sid = subj.get("setup_id")
            if subj.get("type") == "setup" and sid \
                    and subj.get("state") == "triggered" \
                    and sid not in out:
                out[sid] = rec
        return out

    def _prior_refusals(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for rec in self._events("execution_refused"):
            sid = rec.get("setup_id")
            if sid:
                out.setdefault(sid, set()).add(
                    str(rec.get("reason_category") or ""))
        return out

    def execution_pass(self, symbol: str, market: str = "forex", *,
                       read: dict[str, Any] | None = None,
                       forecasts: dict[str, dict[str, Any]] | None = None
                       ) -> dict[str, Any]:
        """One execution pass over the symbol's TRIGGERED setups.

        Each triggered setup with no open position and no prior
        terminal refusal is judged (the journaled Pass 3 verdict at
        its triggered state, or a fresh evaluation journaled now)
        and executed under the invariants. Friction refusals are
        the one retryable outcome: a narrowed spread on a later
        pass may still take the setup while it lives.
        """
        taken = float(self._now())
        sym = (symbol or "").strip().upper()
        if read is None:
            from . import market_read
            read = market_read.whole_market_read(
                sym, market=market, journal=True,
                settings=self.settings, journal_path=self.journal_path,
                now=taken)
        if forecasts is None:
            forecasts = forecast_mod.latest_forecasts(
                sym, settings=self.settings,
                journal_path=self.journal_path)
        book = forecast_mod.calibration_book(
            symbol=sym, settings=self.settings,
            journal_path=self.journal_path)
        judged = self._latest_judgments()
        refusals = self._prior_refusals()
        open_setup_ids = {p["setup_id"] for p in self.open_positions()}
        executed: list[dict[str, Any]] = []
        refused: list[dict[str, Any]] = []
        skipped = 0
        for setup in load_setups(settings=self.settings,
                                 journal_path=self.journal_path,
                                 symbol=sym):
            if setup["state"] != "triggered":
                continue
            sid = setup["setup_id"]
            if sid in open_setup_ids:
                skipped += 1
                continue
            prior = refusals.get(sid) or set()
            if prior - {"friction_gate"}:
                skipped += 1
                continue
            verdict = judged.get(sid)
            if verdict is None:
                try:
                    verdict = judgment_mod.evaluate_setup(
                        read, setup, forecasts=forecasts, book=book)
                except Exception as exc:  # noqa: BLE001 - fail closed
                    verdict = {"decision": "error",
                               "reason_category": "judgment_error",
                               "why": f"{type(exc).__name__}: {exc}"}
                else:
                    self._journal("judgment", judged_at=taken, **verdict)
            result = self.execute_setup(setup, verdict, read=read)
            (executed if result.get("ok") else refused).append(result)
        summary: dict[str, Any] = {
            "kind": "execution_pass", "symbol": sym, "market": market,
            "taken_at": taken, "rung": self.rung,
            "executed": executed, "refused": refused,
            "skipped": skipped,
            "counts": {"executed": len(executed),
                       "refused": len(refused), "skipped": skipped}}
        self._journal("execution_pass", symbol=sym, market=market,
                      rung=self.rung, counts=summary["counts"])
        return summary

    # ── management ──

    def manage_pass(self, *, symbol: str | None = None
                    ) -> dict[str, Any]:
        """One management sweep over every open system position."""
        taken = float(self._now())
        actions: list[dict[str, Any]] = []
        managed = 0
        for pos in self.open_positions():
            if symbol is not None and pos["symbol"] != \
                    symbol.strip().upper():
                continue
            managed += 1
            actions.extend(self._manage_position(pos))
        summary: dict[str, Any] = {
            "kind": "manage_pass", "taken_at": taken, "rung": self.rung,
            "managed": managed, "actions": actions,
            "counts": {"positions": managed, "actions": len(actions)}}
        self._journal("manage_pass", rung=self.rung,
                      counts=summary["counts"])
        return summary

    def _manage_position(self, pos: dict[str, Any]) -> list[dict[str, Any]]:
        acts: list[dict[str, Any]] = []
        quote = self._quote(pos["symbol"], pos.get("market") or "forex")
        price = quote.get("price")
        if not price:
            return [{"position_id": pos["position_id"],
                     "action": "no_price",
                     "detail": "no fresh quote — position left as-is"}]
        price = float(price)
        # 1. The placement bracket fires first (both QM and every
        # other profile carry the desk-level SL/outer-TP).
        if self.rung == "paper":
            # A position already gone from the desk's book was
            # closed by an earlier mark in this sweep (paper_mark
            # marks the whole instrument): reconcile it from the
            # desk journal's own close record.
            live_ids = {p.id for p in self.desk.paper_positions()}
            if pos["position_id"] not in live_ids:
                rec = self._desk_close_record(pos["position_id"])
                if rec is not None:
                    self._close_from_broker_record(
                        pos, rec, source="simulator_record")
                    acts.append({
                        "position_id": pos["position_id"],
                        "action": "bracket_close",
                        "reason": rec.get("reason"),
                        "pnl": rec.get("pnl")})
                return acts
            try:
                closes = self.desk.paper_mark(pos["symbol"], price)
            except DeskError:
                closes = []
            for close in closes:
                if close.get("position_id") == pos["position_id"]:
                    self._close_from_broker_record(pos, close,
                                                   source="simulator_record")
                    acts.append({
                        "position_id": pos["position_id"],
                        "action": "bracket_close",
                        "reason": close.get("reason"),
                        "pnl": close.get("pnl")})
                    return acts
        elif self.rung == "exness_demo":
            if self._broker_position_gone(pos):
                self._close_from_snapshot(pos, price)
                acts.append({"position_id": pos["position_id"],
                             "action": "reconciled_close",
                             "detail": "broker no longer lists the "
                                       "position — closed from the "
                                       "snapshot price"})
                return acts
        # 2. Excursion facts (poll rung: current, never tick highs).
        direction = int(pos["direction"])
        entry = float(pos["entry"])
        risk_dist = abs(entry - float(pos["initial_stop"]))
        if risk_dist <= 0:
            return acts
        fav = (price - entry) * direction
        current_r = fav / risk_dist
        targets = [float(t) for t in pos.get("targets") or []]
        progress = fav / abs(targets[0] - entry) \
            if targets and abs(targets[0] - entry) > 0 else 0.0
        # 3. The profile. qm_fixed_bracket: NOTHING beyond the
        # bracket — no lock, no partials, no trail, by doctrine.
        if pos.get("profile") == "qm_fixed_bracket":
            return acts
        if pos.get("profile") == "staged_runner":
            acts.extend(self._manage_scalp(pos, price, current_r,
                                           targets))
        else:
            acts.extend(self._manage_generic(pos, price, current_r,
                                             progress, targets))
        return acts

    def _manage_generic(self, pos: dict[str, Any], price: float,
                        current_r: float, progress: float,
                        targets: list[float]) -> list[dict[str, Any]]:
        acts: list[dict[str, Any]] = []
        # Partial tiers — only at >= 0.03 original lots.
        if float(pos["original_volume"]) >= PARTIAL_MIN_VOLUME - 1e-9:
            for tier_r, frac in GENERIC_TIERS:
                tier = f"{tier_r:g}R"
                if tier in (pos.get("tiers_done") or []) \
                        or current_r < tier_r:
                    continue
                vol = min(_floor_step(
                    float(pos["original_volume"]) * frac,
                    float(pos.get("volume_step") or MIN_LOT)),
                    float(pos["volume"]))
                if vol <= 0:
                    continue
                act = self._partial_close(
                    pos, vol, price, tier=tier,
                    reason=f"partial_tier_{tier}")
                if act is not None:
                    acts.append(act)
                    pos = self._refresh(pos)
                    if pos is None or pos["state"] != "open":
                        return acts
        # Profit lock — one-shot, at 1R or half-way to T1.
        if not pos.get("locked") and not pos.get("lock_unapplied") \
                and (current_r >= 1.0 or progress >= 0.5):
            trigger = "1R" if current_r >= 1.0 else "progress_50pct"
            act = self._apply_lock(pos, price, trigger=trigger)
            if act is not None:
                acts.append(act)
        return acts

    def _manage_scalp(self, pos: dict[str, Any], price: float,
                      current_r: float,
                      targets: list[float]) -> list[dict[str, Any]]:
        acts: list[dict[str, Any]] = []
        direction = int(pos["direction"])
        entry = float(pos["entry"])
        # Staged partials at the plan's own targets.
        for idx, frac in SCALP_TIERS:
            if idx >= len(targets):
                break
            tier = f"target_{idx + 1}"
            if tier in (pos.get("tiers_done") or []):
                continue
            if (targets[idx] - entry) * direction <= 0:
                continue
            if (price - targets[idx]) * direction < 0:
                continue  # target not reached yet
            vol = min(_floor_step(
                float(pos["original_volume"]) * frac,
                float(pos.get("volume_step") or MIN_LOT)),
                float(pos["volume"]))
            if vol <= 0:
                continue
            act = self._partial_close(pos, vol, price, tier=tier,
                                      reason=f"scalp_{tier}")
            if act is not None:
                acts.append(act)
                pos = self._refresh(pos)
                if pos is None or pos["state"] != "open":
                    return acts
        # The EARNED lock: M5 acceptance beyond T1, or the 2.0R
        # backstop. An unproven push arms nothing.
        if not pos.get("locked") and not pos.get("lock_unapplied") \
                and targets:
            accepted = self._m5_acceptance(pos, targets[0])
            backstop = current_r >= 2.0
            if accepted or backstop:
                zone = pos.get("entry_zone") or {}
                cap = zone.get("top") if direction == 1 \
                    else zone.get("bottom")
                act = self._apply_lock(
                    pos, price,
                    trigger=("m5_acceptance" if accepted
                             else "backstop_2r"),
                    cap=(float(cap) if cap is not None else None))
                if act is not None:
                    acts.append(act)
        return acts

    def _m5_acceptance(self, pos: dict[str, Any],
                       target: float) -> bool:
        """A closed M5 bar beyond the first target, since the open.

        Acceptance is by CLOSES on the entry timeframe's own bars
        (Pass 2/3 doctrine: acceptance-by-closes, never a wick).
        """
        frame = self._bars(pos["symbol"], pos.get("market") or "forex",
                           "5m")
        if frame is None or len(frame) == 0:
            return False
        direction = int(pos["direction"])
        opened = float(pos.get("opened_at") or 0)
        try:
            for ts, row in frame.iterrows():
                if ts_epoch(ts) <= opened:
                    continue
                close = float(row["close"])
                if (close - target) * direction > 0:
                    return True
        except Exception as exc:  # noqa: BLE001 - no acceptance claim
            _log.debug("execution: M5 acceptance scan failed: %s", exc)
            return False
        return False

    def _apply_lock(self, pos: dict[str, Any], price: float, *,
                    trigger: str,
                    cap: float | None = None) -> dict[str, Any] | None:
        """Move the stop to the earned lock level (or journal why not)."""
        direction = int(pos["direction"])
        entry = float(pos["entry"])
        risk_dist = abs(entry - float(pos["initial_stop"]))
        atr = self._atr(pos)
        level = entry + direction * max(0.5 * risk_dist, 0.75 * atr)
        capped = False
        if cap is not None:
            capped_level = min(level, cap) if direction == 1 \
                else max(level, cap)
            capped = capped_level != level
            level = capped_level
        current_stop = float(pos["stop_loss"])
        # The lock must improve the stop and sit at/past entry on
        # the profit side; otherwise it is not a lock at all.
        if (level - current_stop) * direction <= 0 \
                or (level - entry) * direction < 0:
            return None
        applied = False
        note = ""
        if self.rung == "paper":
            try:
                self.desk.paper_modify(pos["position_id"],
                                       stop_loss=level, price=price)
                applied = True
            except DeskError as exc:
                note = f"desk refused the stop move: {exc}"
        else:
            note = ("the Exness connector exposes no modify for an "
                    "open position — the lock is designed and "
                    "journaled, not applied")
        self._journal(
            "position_managed", position_id=pos["position_id"],
            setup_id=pos.get("setup_id"), action="profit_lock",
            level=round(level, 5), trigger=trigger, capped=capped,
            atr=atr, price=price, applied=applied,
            reason=note or "stop moved to the earned lock level")
        return {"position_id": pos["position_id"],
                "action": "profit_lock", "level": round(level, 5),
                "trigger": trigger, "capped": capped,
                "applied": applied}

    def _partial_close(self, pos: dict[str, Any], volume: float,
                       price: float, *, tier: str,
                       reason: str) -> dict[str, Any] | None:
        """Close ``volume`` of a position at ``price``; journal it."""
        pid = pos["position_id"]
        direction = int(pos["direction"])
        if self.rung == "paper":
            try:
                res = self.desk.paper_close_partial(
                    pid, volume, price=price, reason=reason)
            except DeskError as exc:
                _log.info("execution: partial close failed: %s", exc)
                return None
            pnl = float(res.get("pnl") or 0)
            remaining = float(res.get("remaining_volume") or 0)
            source = "simulator_record"
        else:
            try:
                self.desk.live_close_partial(pid, volume,
                                             confirmed=True)
            except DeskError as exc:
                _log.info("execution: live partial failed: %s", exc)
                return None
            # No broker deal record in hand at this seam: the P&L
            # is computed from the poll price and marked estimated
            # (Pass 5 reconciliation grades it against the deals).
            pnl = round((price - float(pos["entry"])) * direction
                        * float(pos.get("contract_value") or 0)
                        * volume, 2)
            remaining = round(float(pos["volume"]) - volume, 8)
            source = "estimated"
        self._journal(
            "position_managed", position_id=pid,
            setup_id=pos.get("setup_id"), action="partial_close",
            tier=tier, volume=volume, price=price, pnl=pnl,
            remaining_volume=remaining, reason=reason,
            price_source=source)
        act = {"position_id": pid, "action": "partial_close",
               "tier": tier, "volume": volume, "price": price,
               "pnl": pnl, "remaining_volume": remaining}
        if remaining <= 1e-9:
            total = round(float(pos.get("realized_pnl") or 0) + pnl, 2)
            self._journal(
                "position_closed", position_id=pid,
                setup_id=pos.get("setup_id"), exit_price=price,
                pnl=total, reason=f"{reason}_complete",
                price_source=source)
            act["closed"] = True
        return act

    def _refresh(self, pos: dict[str, Any]) -> dict[str, Any] | None:
        for fresh in self.load_positions():
            if fresh["position_id"] == pos["position_id"]:
                return fresh
        return None

    def _desk_close_record(self, position_id: str) -> dict[str, Any] | None:
        """The desk journal's close record for one paper position."""
        for rec in self.desk.journal(limit=10000):
            if rec.get("event") == "paper_close" \
                    and rec.get("position_id") == position_id:
                return {"exit": rec.get("exit_price"),
                        "pnl": rec.get("pnl"),
                        "reason": rec.get("reason")}
        return None

    def _broker_position_gone(self, pos: dict[str, Any]) -> bool:
        try:
            snap = self.desk.account()
        except Exception:  # noqa: BLE001 - cannot verify ≠ gone
            return False
        ids = {str(p.get("id") or p.get("position_id") or "")
               for p in snap.get("positions") or []}
        return pos["position_id"] not in ids

    def _close_from_broker_record(self, pos: dict[str, Any],
                                  close: dict[str, Any], *,
                                  source: str) -> None:
        self._journal(
            "position_closed", position_id=pos["position_id"],
            setup_id=pos.get("setup_id"),
            exit_price=close.get("exit"),
            pnl=round(float(pos.get("realized_pnl") or 0)
                      + float(close.get("pnl") or 0), 2),
            reason=close.get("reason"), price_source=source)

    def _close_from_snapshot(self, pos: dict[str, Any],
                             price: float) -> None:
        # Try the broker's deals history for the closing price
        # first; fall back to the snapshot (poll) price, marked.
        exit_price, source = price, "snapshot"
        try:
            deals = self.desk.connector.deals_history(limit=50)
            rows: list[Any] = []
            if isinstance(deals, dict):
                rows = deals.get("deals") or deals.get("history") or []
            for row in rows:
                if str(row.get("position_id") or row.get("positionId")
                       or "") == pos["position_id"] \
                        and row.get("price"):
                    exit_price = float(row["price"])
                    source = "broker_deal"
                    break
        except Exception as exc:  # noqa: BLE001 - snapshot fallback
            _log.debug("execution: deals lookup failed: %s", exc)
        direction = int(pos["direction"])
        pnl = round(float(pos.get("realized_pnl") or 0)
                    + (exit_price - float(pos["entry"])) * direction
                    * float(pos.get("contract_value") or 0)
                    * float(pos["volume"]), 2)
        self._journal(
            "position_closed", position_id=pos["position_id"],
            setup_id=pos.get("setup_id"), exit_price=exit_price,
            pnl=pnl, reason="broker_closed", price_source=source)

    def close_position(self, position_id: str, *,
                       price: float | None = None,
                       reason: str = "manual") -> dict[str, Any]:
        """Close one open system position in full, journaled."""
        pos = self._refresh({"position_id": position_id})
        if pos is None or pos["state"] != "open":
            return {"ok": False,
                    "error": f"no open system position {position_id}"}
        if price is None:
            quote = self._quote(pos["symbol"],
                                pos.get("market") or "forex")
            price = float(quote.get("price") or 0)
            if not price:
                return {"ok": False,
                        "error": "no fresh quote to close at"}
        if self.rung == "paper":
            res = self.desk.paper_close(position_id, exit_price=price)
            self._close_from_broker_record(
                pos, {"exit": res.get("exit"), "pnl": res.get("pnl"),
                      "reason": reason}, source="simulator_record")
            return {"ok": True, "position_id": position_id,
                    "pnl": res.get("pnl"), "exit": res.get("exit")}
        self.desk.live_close(position_id, confirmed=True)
        self._close_from_snapshot(pos, price)
        return {"ok": True, "position_id": position_id,
                "exit": price, "price_source": "snapshot"}


# ── surfaces ─────────────────────────────────────────────────────────

def _fmt(value: Any) -> str:
    if value is None:
        return "?"
    return f"{float(value):,.2f}"


def render_execution_pass(summary: dict[str, Any]) -> str:
    counts = summary.get("counts") or {}
    lines = [f"execution pass {summary.get('symbol')} "
             f"[{summary.get('rung')}] — executed "
             f"{counts.get('executed', 0)}, refused "
             f"{counts.get('refused', 0)}, skipped "
             f"{counts.get('skipped', 0)}"]
    for res in summary.get("executed") or []:
        pos = res.get("position") or {}
        lines.append(
            f"  OPEN {pos.get('playbook')} "
            f"{'LONG' if pos.get('direction') == 1 else 'SHORT'} "
            f"{pos.get('volume')} @ {_fmt(pos.get('entry'))} "
            f"(stop {_fmt(pos.get('stop_loss'))}, "
            f"profile {pos.get('profile')})")
    for res in summary.get("refused") or []:
        lines.append(f"  REFUSED [{res.get('reason_category')}] "
                     f"{res.get('reason')}")
    return "\n".join(lines)


def render_positions(positions: list[dict[str, Any]], *,
                     symbol: str | None = None) -> str:
    open_now = [p for p in positions if p["state"] == "open"]
    closed = [p for p in positions if p["state"] == "closed"]
    lines = [f"{symbol or 'all symbols'} system positions — "
             f"{len(open_now)} open, {len(closed)} closed"]
    for p in open_now:
        lines.append(
            f"  {p['playbook']} "
            f"{'LONG' if p['direction'] == 1 else 'SHORT'} "
            f"{p['volume']} @ {_fmt(p['entry'])} · stop "
            f"{_fmt(p['stop_loss'])}{' (locked)' if p.get('locked') else ''}"
            f" · {p['profile']} [{p['rung']}]")
    for p in closed[-10:]:
        lines.append(
            f"  closed {p['playbook']} pnl {_fmt(p.get('pnl'))} "
            f"({p.get('close_reason')}, priced from "
            f"{p.get('close_price_source')})")
    if not positions:
        lines.append("no system positions yet")
    return "\n".join(lines)


def render_manage_pass(summary: dict[str, Any]) -> str:
    lines = [f"manage pass [{summary.get('rung')}] — "
             f"{(summary.get('counts') or {}).get('positions', 0)} "
             f"positions, {(summary.get('counts') or {}).get('actions', 0)} "
             f"actions"]
    for act in summary.get("actions") or []:
        lines.append(f"  {act.get('position_id')}: "
                     f"{act.get('action')}"
                     f"{' ' + str(act.get('tier')) if act.get('tier') else ''}"
                     f"{' @ ' + _fmt(act.get('level')) if act.get('level') else ''}"
                     f"{' pnl ' + _fmt(act.get('pnl')) if act.get('pnl') is not None else ''}")
    return "\n".join(lines)
