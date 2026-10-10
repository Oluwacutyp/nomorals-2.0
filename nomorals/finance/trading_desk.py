"""Devon's own trading desk over the Exness connector.

The connector (``nomorals/connectors/exness.py``) owns the API surface:
signing, endpoints, order placement. The desk owns the *trading
discipline* — the part that keeps an account alive:

- **Risk policy**: max risk per trade (% of equity), mandatory
  stop-loss on every market order, max open positions, max daily loss
  (kill switch), per-trade confirmation.
- **Position sizing**: volume derived from the stop-loss distance and
  the instrument's real contract specs — never a guessed lot size.
  When the contract value can't be determined, sizing refuses and the
  caller must pass an explicit volume.
- **Paper mode** (default): fills simulated against real candles from
  the connector. Demo-first — live money requires an explicit unlock
  *and* the usual confirmation gates.
- **P&L journal**: every open/close, paper or live, lands in an
  append-only journal; ``stats()`` reports per-instrument win rate and
  realized P&L.

Money safety: live orders go through the connector's own
confirmation/checkpoint gate *in addition* to the desk's risk gates.
Paper mode never touches the connector's money-moving calls.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .budgets import finance_paths

_log = get_logger("nomorals.finance.trading")

__all__ = [
    "RiskPolicy",
    "PaperPosition",
    "DeskError",
    "TradingDesk",
    "size_position",
    "expected_value_r",
]


def expected_value_r(win_rate: float, avg_win_r: float,
                     avg_loss_r: float) -> float:
    """Expectancy in R: W × avg_win − (1−W) × |avg_loss|.

    The single number that says whether a system has an edge
    (Van Tharp: expectancy is what matters, not win rate).
    """
    return win_rate * avg_win_r - (1.0 - win_rate) * abs(avg_loss_r)


class DeskError(Exception):
    """The desk refused a trade — risk gate, bad input, or locked live."""


@dataclass
class RiskPolicy:
    """Trading discipline. All values enforced, not suggested."""

    max_risk_pct_per_trade: float = 1.0   # % of equity risked per trade
    max_open_positions: int = 3
    max_daily_loss_pct: float = 3.0       # kill switch on realized+unrealized
    max_weekly_loss_pct: float = 7.0      # weekly kill switch
    require_stop_loss: bool = True        # market orders without SL are refused
    allowed_instruments: tuple[str, ...] = ()  # empty = any
    live_unlocked: bool = False           # live money needs an explicit unlock
    min_rr: float = 0.0                   # min reward:risk; 0 = unenforced
    # Adaptive risk scaling (mined from real algo traders):
    adapt_on_streaks: bool = True         # shrink/grow risk by win/loss streaks
    streak_win_bonus_pct: float = 0.25    # +risk per win after streak_win_n
    streak_win_n: int = 3                 # wins before bonus applies
    streak_loss_cut_pct: float = 0.25     # −risk per loss after streak_loss_n
    streak_loss_n: int = 2                # losses before cut applies
    streak_risk_floor_pct: float = 0.5    # never go below this
    streak_risk_cap_pct: float = 2.0      # never go above this
    consec_loss_halt_n: int = 5           # halt after N consecutive losses

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperPosition:
    """A simulated position."""

    id: str
    instrument: str
    side: str            # "buy" | "sell"
    volume: float
    entry: float
    stop_loss: float | None = None
    take_profit: float | None = None
    opened_at: float = 0.0
    contract_value: float = 0.0  # account-currency value of a 1.0 price move
                                 # per 1.0 volume — from instrument conditions
    risk_amount: float = 0.0  # account currency risked if SL hits (for R)


def _r_multiple(pnl: float, risk_amount: float) -> float | None:
    """Outcome in units of initial risk. −1R = stopped, +2R = 2× risk."""
    if risk_amount and risk_amount > 0:
        return round(pnl / risk_amount, 2)
    return None


def size_position(
    *,
    equity: float,
    entry: float,
    stop_loss: float,
    risk_pct: float,
    contract_value: float,
    volume_min: float = 0.01,
    volume_max: float = 100.0,
    volume_step: float = 0.01,
) -> dict[str, Any]:
    """Volume for a fixed-fractional risk trade.

    risk_amount = equity × risk_pct/100; the stop distance in price ×
    contract_value is the loss per 1.0 volume; volume = risk/loss_per_lot,
    floored to the volume step and clamped to [min, max].

    Raises DeskError when the numbers don't admit a valid size (e.g. the
    stop is so tight the minimum lot risks too much) — sizing never
    silently rounds into a bigger risk than asked.
    """
    if equity <= 0:
        raise DeskError("equity must be positive to size a position")
    if stop_loss == entry:
        raise DeskError("stop-loss equals entry — no definable risk")
    if contract_value <= 0:
        raise DeskError(
            "contract value unknown for this instrument — pass volume "
            "explicitly instead of sizing")
    if not 0 < risk_pct <= 100:
        raise DeskError(f"risk_pct must be in (0, 100], got {risk_pct}")
    risk_amount = equity * risk_pct / 100.0
    stop_distance = abs(entry - stop_loss)
    loss_per_unit = stop_distance * contract_value
    if loss_per_unit <= 0:
        raise DeskError("stop distance × contract value is not positive")
    raw = risk_amount / loss_per_unit
    # Floor to the broker's step (never round up into extra risk).
    steps = int(raw // volume_step)
    volume = round(steps * volume_step, 8)
    if volume < volume_min:
        raise DeskError(
            f"sized volume {volume} is below the minimum {volume_min} — "
            f"the stop is too tight for {risk_pct}% risk at this equity; "
            "widen the stop, lower the risk, or pass volume explicitly")
    if volume > volume_max:
        raise DeskError(
            f"sized volume {volume} exceeds the maximum {volume_max}")
    actual_risk = volume * loss_per_unit
    return {
        "volume": volume,
        "risk_amount": round(risk_amount, 2),
        "actual_risk": round(actual_risk, 2),
        "actual_risk_pct": round(actual_risk / equity * 100, 3),
        "stop_distance": stop_distance,
    }


def _journal_path(settings: Any = None) -> Path:
    _, budgets_path = finance_paths(settings)
    return Path(budgets_path).parent / "trading_journal.jsonl"


def _desk_state_path(settings: Any = None) -> Path:
    _, budgets_path = finance_paths(settings)
    return Path(budgets_path).parent / "trading_desk.json"


class TradingDesk:
    """Risk-managed trading over an ExnessConnector.

    ``mode="paper"`` (default): simulated fills against the connector's
    real candles. ``mode="live"``: real orders through the connector —
    still gated by the risk policy, the live unlock, and the connector's
    own confirmation/checkpoint.
    """

    def __init__(
        self,
        connector: Any,
        *,
        policy: RiskPolicy | None = None,
        mode: str = "paper",
        journal_path: str | Path | None = None,
        state_path: str | Path | None = None,
        now: Any = None,
    ) -> None:
        if mode not in ("paper", "live"):
            raise DeskError(f"mode must be paper|live, got {mode!r}")
        self.connector = connector
        self.policy = policy or RiskPolicy()
        self.mode = mode
        self.journal_path = Path(journal_path) if journal_path else _journal_path()
        self.state_path = Path(state_path) if state_path else _desk_state_path()
        self._now = now or time.time
        self._paper_positions: dict[str, PaperPosition] = {}
        self._load_state()

    # ── persistence ──────────────────────────────────────────────

    def _load_state(self) -> None:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        pol = raw.get("policy") or {}
        for k, v in pol.items():
            if hasattr(self.policy, k):
                setattr(self.policy, k, v)
        for p in raw.get("paper_positions") or []:
            try:
                pos = PaperPosition(**p)
                self._paper_positions[pos.id] = pos
            except TypeError:
                continue

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "policy": self.policy.to_dict(),
            "paper_positions": [asdict(p) for p in
                                self._paper_positions.values()],
        }, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    def _journal(self, event: str, **fields: Any) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": self._now(), "event": event, "mode": self.mode,
               **fields}
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def journal(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Last N journal records, newest first."""
        if not self.journal_path.exists():
            return []
        recs = []
        for line in self.journal_path.read_text(
                encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except ValueError:
                continue
        return recs[-limit:][::-1]

    # ── account / market data (read-only, shared) ─────────────────

    def account(self) -> dict[str, Any]:
        """Live account state + open positions + pending orders."""
        return self.connector.get_snapshot()

    def equity(self) -> float:
        state = (self.account().get("account_state") or {})
        for key in ("equity", "balance"):
            try:
                val = float(state.get(key) or 0)
            except (TypeError, ValueError):
                continue
            if val > 0:
                return val
        raise DeskError("could not read equity from the Exness snapshot")

    def candles(self, instrument: str, timeframe: str = "H1",
                count: int = 200) -> list[dict[str, Any]]:
        return self.connector.get_candles(instrument, timeframe,
                                          count=count)

    def contract_specs(self, instrument: str) -> dict[str, Any]:
        """Real contract terms for sizing. Never guessed."""
        cond = self.connector.get_instrument_condition(instrument)
        if not isinstance(cond, dict):
            raise DeskError(
                f"no contract conditions for {instrument} — cannot size")
        return cond

    def _contract_value(self, instrument: str) -> float:
        """Account-currency value of a 1.0 price move per 1.0 volume.

        Derived from the instrument conditions. Raises DeskError when the
        conditions don't carry enough information — sizing then refuses
        rather than inventing a number.
        """
        cond = self.contract_specs(instrument)
        # Exness conditions expose contract_size (units per 1.0 lot) and
        # the quote currency; for XXXUSD-style symbols the quote is USD
        # and 1 lot × 1.0 price move = contract_size USD. For cross pairs
        # the honest answer is "unknown" — refuse instead of guessing.
        contract_size = cond.get("contract_size")
        quote_cur = str(cond.get("quote_currency") or "").upper()
        try:
            size = float(contract_size) if contract_size else 0
        except (TypeError, ValueError):
            size = 0
        if size > 0 and quote_cur in ("USD", "USDT", ""):
            return size
        raise DeskError(
            f"cannot derive contract value for {instrument} "
            f"(contract_size={contract_size!r}, quote={quote_cur!r}) — "
            "pass volume explicitly")

    # ── risk gates (live AND paper) ───────────────────────────────

    def _check_risk(self, instrument: str, side: str, volume: float,
                    stop_loss: float | None, take_profit: float | None,
                    entry: float) -> dict[str, Any]:
        """Enforce the policy. Returns the sizing/RR facts. Raises DeskError."""
        p = self.policy
        instrument = (instrument or "").strip().upper()
        # Manual kill switch — operator touched the HALT file.
        if self.kill_switch_engaged():
            raise DeskError(
                "kill switch engaged (HALT file present) — remove it to resume")
        # Consecutive-loss halt.
        if p.consec_loss_halt_n > 0:
            losses = self.streaks()["losses"]
            if losses >= p.consec_loss_halt_n:
                self._journal("kill", reason="consecutive loss halt",
                              consec_losses=losses)
                raise DeskError(
                    f"{losses} consecutive losses — desk halted, review "
                    "before resuming")
        if not instrument:
            raise DeskError("instrument is required")
        if side not in ("buy", "sell"):
            raise DeskError(f"side must be buy|sell, got {side!r}")
        if p.allowed_instruments and instrument not in {
                s.upper() for s in p.allowed_instruments}:
            raise DeskError(f"{instrument} is not in the allowed list")
        if not volume or volume <= 0:
            raise DeskError("volume must be positive")
        if p.require_stop_loss and stop_loss is None:
            raise DeskError(
                "stop-loss is required by the risk policy — "
                "no naked market orders")
        if stop_loss is not None:
            if side == "buy" and stop_loss >= entry:
                raise DeskError("buy stop-loss must be below entry")
            if side == "sell" and stop_loss <= entry:
                raise DeskError("sell stop-loss must be above entry")
        if take_profit is not None:
            if side == "buy" and take_profit <= entry:
                raise DeskError("buy take-profit must be above entry")
            if side == "sell" and take_profit >= entry:
                raise DeskError("sell take-profit must be below entry")
        if p.min_rr > 0 and stop_loss is not None and take_profit is not None:
            risk_dist = abs(entry - stop_loss)
            reward_dist = abs(take_profit - entry)
            rr = reward_dist / risk_dist if risk_dist > 0 else 0
            if rr < p.min_rr:
                raise DeskError(
                    f"reward:risk {rr:.2f} is below the minimum {p.min_rr}")
        # Position count gate (paper + live-open paper legs).
        open_count = len(self._paper_positions)
        if self.mode == "live":
            try:
                open_count += len(self.account().get("positions") or [])
            except Exception:  # noqa: BLE001 - snapshot failure ≠ free pass
                raise DeskError(
                    "cannot verify open position count — refusing to open")
        if open_count >= p.max_open_positions:
            raise DeskError(
                f"already at max open positions ({p.max_open_positions})")
        # Daily loss kill switch.
        day_pnl = self.daily_pnl()
        if day_pnl is not None:
            try:
                eq = self.equity()
            except DeskError:
                eq = 0.0
            if eq > 0 and day_pnl <= -eq * p.max_daily_loss_pct / 100:
                self._journal("kill", reason="daily loss limit",
                              day_pnl=day_pnl)
                raise DeskError(
                    f"daily loss limit hit ({day_pnl:+.2f}) — desk is "
                    "killed for today")
        # Weekly loss kill switch.
        week_pnl = self.weekly_pnl()
        try:
            eq = self.equity()
        except DeskError:
            eq = 0.0
        if eq > 0 and week_pnl <= -eq * p.max_weekly_loss_pct / 100:
            self._journal("kill", reason="weekly loss limit",
                          week_pnl=week_pnl)
            raise DeskError(
                f"weekly loss limit hit ({week_pnl:+.2f}) — desk is "
                "killed for the week")
        return {"instrument": instrument, "side": side, "volume": volume}

    def daily_pnl(self) -> float | None:
        """Realized P&L today (account currency) + open unrealized.

        Journal-backed for paper; for live, the snapshot's position
        profits are added when the API exposes them.
        """
        day_start = self._now() - (self._now() % 86400)
        total = 0.0
        seen = False
        for rec in self.journal(limit=10000):
            if rec.get("ts", 0) < day_start:
                break
            if rec.get("event") in ("paper_close", "live_close"):
                try:
                    total += float(rec.get("pnl") or 0)
                    seen = True
                except (TypeError, ValueError):
                    continue
        if self.mode == "live":
            try:
                for pos in self.account().get("positions") or []:
                    try:
                        total += float(pos.get("profit") or 0)
                        seen = True
                    except (TypeError, ValueError):
                        continue
            except Exception:  # noqa: BLE001
                pass
        return total if seen else 0.0

    def weekly_pnl(self) -> float:
        """Realized P&L over the trailing 7 days (account currency)."""
        week_start = self._now() - 7 * 86400
        total = 0.0
        for rec in self.journal(limit=10000):
            if rec.get("ts", 0) < week_start:
                break
            if rec.get("event") in ("paper_close", "live_close"):
                try:
                    total += float(rec.get("pnl") or 0)
                except (TypeError, ValueError):
                    continue
        return total

    def streaks(self) -> dict[str, int]:
        """Current win/loss streaks from closed trades (most recent first).

        Returns ``{"wins": n, "losses": n}`` — only one is non-zero
        (the active streak); both zero when no closed trades yet.
        """
        wins = losses = 0
        for rec in self.journal(limit=10000):
            if rec.get("event") not in ("paper_close", "live_close"):
                continue
            try:
                pnl = float(rec.get("pnl") or 0)
            except (TypeError, ValueError):
                continue
            if pnl > 0:
                if losses:
                    break
                wins += 1
            elif pnl < 0:
                if wins:
                    break
                losses += 1
            # breakeven (pnl == 0) doesn't break either streak
        return {"wins": wins, "losses": losses}

    def effective_risk_pct(self) -> float:
        """Risk % per trade after streak adaptation.

        Win streak ≥ streak_win_n → +streak_win_bonus_pct (capped).
        Loss streak ≥ streak_loss_n → −streak_loss_cut_pct (floored).
        Returns the base when adaptation is disabled.
        """
        p = self.policy
        base = p.max_risk_pct_per_trade
        if not p.adapt_on_streaks:
            return base
        s = self.streaks()
        risk = base
        if s["wins"] >= p.streak_win_n:
            risk += p.streak_win_bonus_pct
        if s["losses"] >= p.streak_loss_n:
            risk -= p.streak_loss_cut_pct * (s["losses"] - p.streak_loss_n + 1)
        return max(p.streak_risk_floor_pct,
                   min(p.streak_risk_cap_pct, risk))

    def kill_switch_path(self) -> Path:
        """Path of the HALT file — touch it to stop all trading immediately."""
        return self.state_path.parent / "HALT"

    def kill_switch_engaged(self) -> bool:
        """True when the operator has engaged the manual kill switch."""
        return self.kill_switch_path().exists()

    # ── paper trading ─────────────────────────────────────────────

    def paper_open(self, instrument: str, side: str, volume: float, *,
                   entry: float | None = None,
                   stop_loss: float | None = None,
                   take_profit: float | None = None) -> PaperPosition:
        """Simulated fill at the latest candle close (or given entry)."""
        if self.mode != "paper":
            raise DeskError("desk is in live mode — paper_open is disabled")
        if entry is None:
            bars = self.candles(instrument, count=2)
            if not bars:
                raise DeskError(f"no candles for {instrument}")
            entry = float(bars[-1].get("close") or bars[-1].get("c") or 0)
            if entry <= 0:
                raise DeskError(f"no usable close price for {instrument}")
        self._check_risk(instrument, side, volume, stop_loss, take_profit,
                         entry)
        contract_value = 0.0
        try:
            contract_value = self._contract_value(instrument)
        except DeskError:
            pass  # paper can still track in price terms
        risk_amount = 0.0
        if stop_loss is not None:
            stop_dist = abs(float(entry) - float(stop_loss))
            unit = contract_value if contract_value > 0 else 1.0
            risk_amount = round(stop_dist * unit * float(volume), 2)
        pos = PaperPosition(
            id="px_" + uuid.uuid4().hex[:10],
            instrument=instrument.upper(), side=side, volume=float(volume),
            entry=float(entry),
            stop_loss=float(stop_loss) if stop_loss is not None else None,
            take_profit=float(take_profit) if take_profit is not None else None,
            opened_at=self._now(), contract_value=contract_value,
            risk_amount=risk_amount,
        )
        self._paper_positions[pos.id] = pos
        self._save_state()
        self._journal("paper_open", position_id=pos.id,
                      instrument=pos.instrument, side=side, volume=volume,
                      entry=entry, stop_loss=stop_loss,
                      take_profit=take_profit)
        _log.info("paper opened %s %s %s @ %s", pos.id, side, instrument,
                  entry)
        return pos

    def paper_positions(self) -> list[PaperPosition]:
        return list(self._paper_positions.values())

    def _paper_pnl(self, pos: PaperPosition, price: float) -> float:
        move = (price - pos.entry) if pos.side == "buy" else (pos.entry - price)
        if pos.contract_value > 0:
            return move * pos.contract_value * pos.volume
        return move * pos.volume  # price-points × volume fallback

    def paper_mark(self, instrument: str,
                   price: float) -> list[dict[str, Any]]:
        """Mark paper positions to market; close any that hit SL/TP.

        Returns the closes that happened."""
        closes = []
        for pid, pos in list(self._paper_positions.items()):
            if pos.instrument != instrument.upper():
                continue
            hit_sl = (pos.stop_loss is not None and
                      ((pos.side == "buy" and price <= pos.stop_loss) or
                       (pos.side == "sell" and price >= pos.stop_loss)))
            hit_tp = (pos.take_profit is not None and
                      ((pos.side == "buy" and price >= pos.take_profit) or
                       (pos.side == "sell" and price <= pos.take_profit)))
            if hit_sl or hit_tp:
                pnl = self._paper_pnl(
                    pos, pos.stop_loss if hit_sl else pos.take_profit)
                closes.append(self._paper_close(
                    pid, pos.stop_loss if hit_sl else pos.take_profit,
                    reason="stop_loss" if hit_sl else "take_profit"))
        return closes

    def _paper_close(self, position_id: str, exit_price: float,
                     reason: str) -> dict[str, Any]:
        pos = self._paper_positions.pop(position_id, None)
        if pos is None:
            raise DeskError(f"unknown paper position {position_id}")
        pnl = self._paper_pnl(pos, exit_price)
        r_mult = _r_multiple(pnl, pos.risk_amount)
        self._save_state()
        self._journal("paper_close", position_id=position_id,
                      instrument=pos.instrument, side=pos.side,
                      volume=pos.volume, entry=pos.entry,
                      exit_price=exit_price, pnl=round(pnl, 2), reason=reason,
                      risk_amount=pos.risk_amount, r_multiple=r_mult)
        return {"position_id": position_id, "instrument": pos.instrument,
                "side": pos.side, "volume": pos.volume, "entry": pos.entry,
                "exit": exit_price, "pnl": round(pnl, 2), "reason": reason,
                "r_multiple": r_mult}

    def paper_close(self, position_id: str, *,
                    exit_price: float | None = None) -> dict[str, Any]:
        """Close a paper position at the latest close (or given price)."""
        pos = self._paper_positions.get(position_id)
        if pos is None:
            raise DeskError(f"unknown paper position {position_id}")
        if exit_price is None:
            bars = self.candles(pos.instrument, count=2)
            if not bars:
                raise DeskError(f"no candles for {pos.instrument}")
            exit_price = float(bars[-1].get("close") or bars[-1].get("c") or 0)
        return self._paper_close(position_id, float(exit_price),
                                 reason="manual")

    # ── live trading (demo-first, confirmation-gated) ──────────────

    def _is_demo_account(self) -> bool:
        """Best-effort demo detection from the account details/settings."""
        try:
            details = self.connector.get_account_details()
        except Exception:  # noqa: BLE001
            return False
        blob = json.dumps(details).lower()
        return "demo" in blob

    def unlock_live(self, confirm: str = "") -> dict[str, Any]:
        """Enable live money on a NON-demo account. Explicit by design."""
        if (confirm or "").strip().lower() not in (
                "i understand the risks", "i understand"):
            raise DeskError(
                'refusing: pass confirm="I understand the risks"')
        if self._is_demo_account():
            return {"ok": True, "mode": "demo",
                    "note": "demo account — confirmation gates still apply, "
                            "no unlock needed"}
        self.policy.live_unlocked = True
        self._save_state()
        self._journal("live_unlocked", note="operator confirmed risks")
        _log.warning("trading desk live unlocked (non-demo account)")
        return {"ok": True, "mode": "live",
                "note": "live unlocked — every order still needs explicit "
                        "confirmation at call time"}

    def live_open(self, instrument: str, side: str, volume: float, *,
                  stop_loss: float | None = None,
                  take_profit: float | None = None,
                  entry: float | None = None,
                  comment: str = "",
                  confirmed: bool = False,
                  db: Any = None,
                  context: Any = None) -> dict[str, Any]:
        """Open a REAL position. Risk gates + live unlock + the connector's
        own confirmation/checkpoint gate. Demo accounts skip the unlock
        but never the confirmation."""
        if self.mode != "live":
            raise DeskError(
                "desk is in paper mode — switch to mode='live' to trade "
                "real money")
        if not self.policy.live_unlocked and not self._is_demo_account():
            raise DeskError(
                "live trading is locked — run unlock_live("
                '"I understand the risks") first (demo accounts exempt)')
        # Entry estimate for the risk checks (the fill price may differ;
        # the connector reports the real fill).
        if entry is None:
            bars = self.candles(instrument, count=2)
            if not bars:
                raise DeskError(f"no candles for {instrument}")
            entry = float(bars[-1].get("close") or bars[-1].get("c") or 0)
        self._check_risk(instrument, side, volume, stop_loss, take_profit,
                         entry)
        self._journal("live_open_attempt", instrument=instrument.upper(),
                      side=side, volume=volume, entry=entry,
                      stop_loss=stop_loss, take_profit=take_profit)
        result = self.connector.open_position(
            instrument.upper(), side, volume,
            stop_loss=stop_loss, take_profit=take_profit,
            comment=comment or "devon-desk",
            confirmed=confirmed, db=db, context=context)
        self._journal("live_open", instrument=instrument.upper(), side=side,
                      volume=volume, stop_loss=stop_loss,
                      take_profit=take_profit,
                      response=str(result)[:500])
        return {"ok": True, "response": result}

    def live_close(self, position_id: str, *,
                   confirmed: bool = False,
                   db: Any = None,
                   context: Any = None) -> dict[str, Any]:
        """Close a REAL position. Confirmation-gated like live_open."""
        if self.mode != "live":
            raise DeskError("desk is in paper mode")
        result = self.connector.close_position(
            position_id, confirmed=confirmed, db=db, context=context)
        self._journal("live_close", position_id=position_id,
                      response=str(result)[:500])
        return {"ok": True, "response": result}

    # ── reporting ─────────────────────────────────────────────────

    def _closed_trades(self) -> list[dict[str, Any]]:
        """Closed trades from the journal, oldest → newest."""
        out = []
        for rec in self.journal(limit=10000):
            if rec.get("event") not in ("paper_close", "live_close"):
                continue
            try:
                pnl = float(rec.get("pnl") or 0)
            except (TypeError, ValueError):
                continue
            r = rec.get("r_multiple")
            try:
                r = float(r) if r is not None else None
            except (TypeError, ValueError):
                r = None
            out.append({
                "ts": rec.get("ts", 0),
                "instrument": str(rec.get("instrument", "")),
                "side": str(rec.get("side", "")),
                "pnl": pnl,
                "r_multiple": r,
            })
        return out[::-1]

    def equity_curve(self) -> list[float]:
        """Cumulative P&L after each closed trade — for sparklines."""
        curve = []
        running = 0.0
        for t in self._closed_trades():
            running += t["pnl"]
            curve.append(round(running, 2))
        return curve

    def max_drawdown(self) -> dict[str, float]:
        """Peak-to-trough of the equity curve (Edgewonk's key number)."""
        curve = self.equity_curve()
        peak = trough_dd = 0.0
        running_peak = 0.0
        for v in curve:
            running_peak = max(running_peak, v)
            trough_dd = max(trough_dd, running_peak - v)
        return {"max_drawdown": round(trough_dd, 2),
                "peak": round(running_peak, 2)}

    def portfolio_heat(self) -> dict[str, Any]:
        """Total open risk as % of equity — Van Tharp's <6% rule.

        Sums each open paper position's planned risk (stop distance ×
        contract value × volume). Live positions contribute unrealized
        exposure when the snapshot exposes it.
        """
        heat_amt = sum(p.risk_amount for p in self._paper_positions.values()
                       if p.risk_amount > 0)
        try:
            eq = self.equity()
        except DeskError:
            eq = 0.0
        heat_pct = (heat_amt / eq * 100) if eq > 0 else 0.0
        return {
            "heat_pct": round(heat_pct, 2),
            "heat_amount": round(heat_amt, 2),
            "equity": round(eq, 2),
            "open_positions": len(self._paper_positions),
            "within_limits": heat_pct < 6.0,
        }

    def kelly_fraction(self) -> dict[str, Any]:
        """Kelly criterion from journal stats — sizing guidance.

        f* = W − (1−W) / R̄. Reported raw and quarter-Kelly (the sane
        default — full Kelly is notoriously aggressive).
        """
        s = self.stats()
        wr = s["win_rate"]
        avg_r = s["avg_r_multiple"]
        if s["total_trades"] < 10 or avg_r <= 0:
            return {"ok": False,
                    "reason": "need ≥10 R-tracked trades for Kelly"}
        kelly = wr - (1 - wr) / avg_r
        return {
            "ok": True,
            "kelly_pct": round(kelly * 100, 2),
            "quarter_kelly_pct": round(kelly * 25, 2),
            "note": "quarter-Kelly is the sane default; full Kelly "
                    "assumes your edge estimate is exact (it isn't)",
        }

    def stats(self) -> dict[str, Any]:
        """Edgewonk-style analytics from the journal.

        Win rate, profit factor, expectancy (R + currency), average R,
        max drawdown, streaks, best/worst, long-vs-short — per instrument
        and overall.
        """
        trades = self._closed_trades()
        per: dict[str, dict[str, Any]] = {}
        total_pnl = 0.0
        wins = losses = 0
        gross_win = gross_loss = 0.0
        r_vals: list[float] = []
        wins_r: list[float] = []
        losses_r: list[float] = []
        longs = {"trades": 0, "pnl": 0.0}
        shorts = {"trades": 0, "pnl": 0.0}
        best = worst = 0.0
        cur_streak = best_streak = worst_streak = 0
        cur_sign = 0

        for t in trades:
            pnl = t["pnl"]
            inst = t["instrument"]
            bucket = per.setdefault(inst, {"trades": 0, "wins": 0,
                                           "pnl": 0.0, "r_sum": 0.0,
                                           "r_n": 0})
            bucket["trades"] += 1
            bucket["pnl"] += pnl
            total_pnl += pnl
            best = max(best, pnl)
            worst = min(worst, pnl)
            if pnl > 0:
                wins += 1
                bucket["wins"] += 1
                gross_win += pnl
                sign = 1
            elif pnl < 0:
                losses += 1
                gross_loss += abs(pnl)
                sign = -1
            else:
                sign = 0
            if sign == cur_sign and sign != 0:
                cur_streak += 1
            else:
                cur_streak, cur_sign = (1 if sign else 0), sign
            if sign > 0:
                best_streak = max(best_streak, cur_streak)
            elif sign < 0:
                worst_streak = max(worst_streak, -cur_streak)
            r = t["r_multiple"]
            if r is not None:
                r_vals.append(r)
                bucket["r_sum"] += r
                bucket["r_n"] += 1
                (wins_r if r > 0 else losses_r if r < 0 else []).append(r)
            if t["side"] == "buy":
                longs["trades"] += 1
                longs["pnl"] += pnl
            elif t["side"] == "sell":
                shorts["trades"] += 1
                shorts["pnl"] += pnl

        n = wins + losses
        win_rate = round(wins / n, 3) if n else 0.0
        profit_factor = (round(gross_win / gross_loss, 2)
                         if gross_loss > 0 else
                         (float("inf") if gross_win > 0 else 0.0))
        avg_r = round(sum(r_vals) / len(r_vals), 2) if r_vals else 0.0
        expectancy_r = (round(expected_value_r(
            win_rate,
            sum(wins_r) / len(wins_r) if wins_r else 0.0,
            sum(losses_r) / len(losses_r) if losses_r else 0.0), 3)
            if r_vals else 0.0)
        expectancy_ccy = round(total_pnl / n, 2) if n else 0.0

        for bucket in per.values():
            bucket["pnl"] = round(bucket["pnl"], 2)
            bucket["win_rate"] = (round(bucket["wins"] / bucket["trades"], 3)
                                  if bucket["trades"] else 0.0)
            bucket["avg_r"] = (round(bucket["r_sum"] / bucket["r_n"], 2)
                               if bucket["r_n"] else 0.0)
            del bucket["r_sum"], bucket["r_n"]

        dd = self.max_drawdown()
        return {
            "mode": self.mode,
            "total_trades": n,
            "wins": wins, "losses": losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy_r": expectancy_r,
            "expectancy_ccy": expectancy_ccy,
            "avg_r_multiple": avg_r,
            "realized_pnl": round(total_pnl, 2),
            "best_trade": round(best, 2),
            "worst_trade": round(worst, 2),
            "max_drawdown": dd["max_drawdown"],
            "best_win_streak": best_streak,
            "worst_loss_streak": worst_streak,
            "longs": {"trades": longs["trades"],
                      "pnl": round(longs["pnl"], 2)},
            "shorts": {"trades": shorts["trades"],
                       "pnl": round(shorts["pnl"], 2)},
            "r_tracked_trades": len(r_vals),
            "open_paper_positions": len(self._paper_positions),
            "per_instrument": per,
        }

    def render_stats(self, theme: str | None = None) -> str:
        """Styled performance report — the desk's dashboard line."""
        from .style import current_theme, sparkline
        th = current_theme(theme)
        s = self.stats()
        lines = [th.paint(f"📈 trading desk — {s['mode']} · "
                          f"{s['total_trades']} closed trades", th.bold)]
        if not s["total_trades"]:
            return "\n".join(lines + ["  no closed trades yet."])
        wr_color = th.good if s["win_rate"] >= 0.5 else th.warn
        lines.append(
            f"  win rate {th.paint(f'{s['win_rate']:.0%}', wr_color)} · "
            f"profit factor {s['profit_factor']} · "
            f"expectancy {s['expectancy_r']:+.2f}R "
            f"({s['expectancy_ccy']:+.2f}/trade)")
        lines.append(
            f"  realized {s['realized_pnl']:+.2f} · max DD "
            f"{s['max_drawdown']:.2f} · avg R {s['avg_r_multiple']:+.2f}")
        lines.append(
            f"  streaks: best {s['best_win_streak']}W / worst "
            f"{s['worst_loss_streak']}L · longs {s['longs']['pnl']:+.2f} / "
            f"shorts {s['shorts']['pnl']:+.2f}")
        curve = self.equity_curve()
        if len(curve) >= 2:
            lines.append(f"  equity {sparkline(curve, th)}")
        heat = self.portfolio_heat()
        heat_txt = (f"{heat['heat_pct']:.1f}% heat"
                    f"{'' if heat['within_limits'] else ' — OVER 6%!'}")
        lines.append(f"  open: {heat['open_positions']} · "
                     f"{th.paint(heat_txt, th.good if heat['within_limits'] else th.bad)}")
        return "\n".join(lines)
