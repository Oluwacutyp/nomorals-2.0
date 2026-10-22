"""Signal calls — timed alerts, published ahead, or not at all.

The owner's terms (2026-10-22): the desk calls signals with **at
least 2 minutes of lead** before the called moment.  A call is a
scheduled prediction published BEFORE its window opens, through the
owner's existing delivery surface (the Notifier, same as finance
alerts and news pushes) — an after-the-fact "call" is not a call.

Eligibility is deliberately narrow, and the narrowness is the point:

* the game's lab verdict is EDGE_CANDIDATE,
* the confirmed structure is ``time_of_day`` (the only structure a
  timed alert can point at), and
* the flagged hours show an effect at least ``MIN_EFFECT`` over the
  base rate on confirmed data.

Anything else produces an explicit no-call report with the reason —
the desk says "nothing here is callable yet, and here is the
evidence" instead of dressing noise up as a signal.  If fresh data
ever validates only a shorter lead for a game, ``min_lead_seconds``
is per-desk configuration and the status surface states the lead the
desk actually stands behind.

Every published call is logged and later SCORED against the rounds
that actually landed in its window (hit = some round in the window
reached the target).  The calibration record — calls made, hit rate
vs the base rate the lab measured — is part of the desk's status, so
the calls grade themselves in the open.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

__all__ = [
    "MIN_LEAD_SECONDS", "MIN_EFFECT", "SignalCall", "SignalEngine",
    "SignalLog",
]

#: The owner's floor: a signal call leads its window by >= 2 minutes.
MIN_LEAD_SECONDS = 120

#: Minimum confirmed advantage (flagged-hour rate minus base rate)
#: before a time structure is worth interrupting the owner for.
MIN_EFFECT = 0.05


@dataclass
class SignalCall:
    provider: str
    game: str
    target: float
    window_start: float          # epoch — the called moment
    window_end: float
    published_at: float          # epoch — when the alert went out
    flagged_hours: list = field(default_factory=list)
    expected_rate: float = 0.0   # lab's flagged-hour success rate
    base_rate: float = 0.0
    evidence: str = ""
    scored: Optional[bool] = None   # None until its window has passed
    call_id: str = ""

    @property
    def lead_seconds(self) -> float:
        return self.window_start - self.published_at

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SignalCall":
        return cls(**{k: v for k, v in (d or {}).items()
                      if k in cls.__dataclass_fields__})


class SignalLog:
    """Append-only call ledger (JSONL) with score-back updates."""

    def __init__(self, desk_dir: str):
        self.path = os.path.join(desk_dir, "signal_calls.jsonl")
        os.makedirs(desk_dir, exist_ok=True)

    def add(self, call: SignalCall) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(call.to_dict()) + "\n")

    def calls(self) -> list[SignalCall]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(SignalCall.from_dict(json.loads(line)))
                    except (ValueError, TypeError):
                        continue
        return out

    def rewrite(self, calls: list[SignalCall]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for c in calls:
                fh.write(json.dumps(c.to_dict()) + "\n")
        os.replace(tmp, self.path)


class SignalEngine:
    """Turns confirmed time structures into ahead-of-time calls."""

    def __init__(self, desk_dir: str, notifier: Any = None,
                 min_lead_seconds: int = MIN_LEAD_SECONDS,
                 min_effect: float = MIN_EFFECT):
        self.log = SignalLog(desk_dir)
        self.notifier = notifier
        self.min_lead = max(0, int(min_lead_seconds))
        self.min_effect = float(min_effect)

    # ── call generation ───────────────────────────────────────────────

    def evaluate(self, verdict: Any, now: Optional[float] = None) -> dict[str, Any]:
        """Decide whether a verdict yields a call right now.

        Returns {"call": SignalCall|None, "reason": str}.  Pure
        decision — publishing happens in :meth:`publish`.
        """
        now = time.time() if now is None else now
        if verdict.verdict != "EDGE_CANDIDATE":
            return {"call": None,
                    "reason": f"{verdict.provider}/{verdict.game}: verdict "
                              f"is {verdict.verdict} — nothing callable"}
        if verdict.callable_structure != "time_of_day":
            return {"call": None,
                    "reason": f"{verdict.provider}/{verdict.game}: confirmed "
                              f"structure is not time-based — a timed alert "
                              f"needs a clock to point at"}
        if not verdict.best_hours:
            return {"call": None,
                    "reason": f"{verdict.provider}/{verdict.game}: no "
                              f"confirmed flagged hours"}
        flagged_rates = [verdict.hour_rates.get(h, 0.0)
                         for h in verdict.best_hours]
        expected = sum(flagged_rates) / len(flagged_rates) if flagged_rates else 0.0
        if expected - verdict.base_success_rate < self.min_effect:
            return {"call": None,
                    "reason": f"{verdict.provider}/{verdict.game}: flagged "
                              f"hours beat base by "
                              f"{expected - verdict.base_success_rate:.1%} "
                              f"< the {self.min_effect:.0%} floor"}
        # next flagged hour boundary at least min_lead ahead
        start = self._next_window_start(verdict.best_hours, now)
        if start is None:
            return {"call": None,
                    "reason": f"{verdict.provider}/{verdict.game}: no "
                              f"flagged hour opens within the next day"}
        call = SignalCall(
            provider=verdict.provider, game=verdict.game,
            target=verdict.target,
            window_start=start, window_end=start + 3600,
            published_at=now,
            flagged_hours=list(verdict.best_hours),
            expected_rate=expected, base_rate=verdict.base_success_rate,
            evidence=verdict.evidence[:400],
            call_id=f"{verdict.provider}-{verdict.game}-{int(start)}")
        if call.lead_seconds < self.min_lead:
            return {"call": None,
                    "reason": f"{verdict.provider}/{verdict.game}: next "
                              f"flagged window opens in "
                              f"{call.lead_seconds:.0f}s < the "
                              f"{self.min_lead}s lead floor — not called "
                              f"rather than called late"}
        return {"call": call, "reason": "callable"}

    def _next_window_start(self, hours: list[int],
                           now: float) -> Optional[float]:
        """Epoch of the next flagged-hour boundary >= now + lead."""
        hour_floor = int(now // 3600) * 3600
        for ahead in range(0, 26):
            boundary = hour_floor + ahead * 3600
            h = int((boundary // 3600) % 24)
            if h in hours and boundary >= now + self.min_lead:
                return float(boundary)
            # also: we may already be inside a flagged hour whose
            # NEXT hour is flagged too — the boundary rule above
            # covers it (the call points at the next hour's window).
        return None

    # ── publishing ────────────────────────────────────────────────────

    def publish(self, call: SignalCall, deliver: bool = True) -> dict[str, Any]:
        """Log the call and (when delivering) push it via the Notifier.

        The call is logged either way — an undelivered call still
        happened as a prediction and must be scored; delivery state
        is reported, never hidden.
        """
        self.log.add(call)
        out: dict[str, Any] = {"call": call.to_dict(), "delivered": False}
        if not deliver:
            out["detail"] = "logged, not delivered (deliver=False)"
            return out
        if self.notifier is None:
            out["detail"] = ("logged; no notifier bound — the call exists "
                            "but reached no surface")
            return out
        from datetime import datetime, timezone
        start = datetime.fromtimestamp(call.window_start,
                                       tz=timezone.utc).strftime("%H:%M UTC")
        body = (f"{call.provider} {call.game}: flagged window opens "
                f"{start} (lead {call.lead_seconds / 60:.0f}m). "
                f"Target {call.target:.2f}x — lab expects "
                f"{call.expected_rate:.0%} of rounds to reach it in "
                f"flagged hours vs {call.base_rate:.0%} base. "
                f"Analysis only — the desk never places bets.")
        try:
            res = self.notifier.publish(
                "betdesk", f"📈 Bet desk signal: {call.game} window", body)
            out["delivered"] = bool(res.get("sent") or res.get("delivered")
                                     or res.get("ok"))
            out["detail"] = f"notifier: {res}"
        except Exception as exc:  # noqa: BLE001 - delivery failure is data
            out["detail"] = f"notifier failed: {exc}"
        return out

    # ── scoring ───────────────────────────────────────────────────────

    def score_due(self, store: Any, now: Optional[float] = None) -> dict[str, Any]:
        """Score every call whose window has fully passed.

        Hit = at least one round of that game landed in the window
        and reached the call's target.  A window with NO collected
        rounds scores as a miss with a note — the desk does not get
        to leave its calls ungraded because collection was thin.
        """
        now = time.time() if now is None else now
        calls = self.log.calls()
        scored = hits = 0
        changed = False
        for c in calls:
            if c.scored is not None or c.window_end > now:
                continue
            rounds = [r for r in store.rounds(c.provider, c.game)
                      if c.window_start <= r.ts < c.window_end]
            c.scored = any(r.crash >= c.target for r in rounds)
            scored += 1
            hits += 1 if c.scored else 0
            changed = True
        if changed:
            self.log.rewrite(calls)
        done = [c for c in calls if c.scored is not None]
        return {
            "scored_now": scored, "hits_now": hits,
            "calls_total": len(calls),
            "calls_scored": len(done),
            "hits_total": sum(1 for c in done if c.scored),
            "hit_rate": (sum(1 for c in done if c.scored) / len(done)
                         if done else None),
        }
