"""Playbooks — Devon trading system, Pass 2.

The playbook FORM and the seeded registry. A playbook is a
declarative definition plus a detector: it declares its family, its
entry timeframe and entry clock (each playbook its own — doctrine),
the timeframe above it in the Swing/ Intraday/ Scalp hierarchy whose
bias directs it, the read conditions it requires, its arming rule,
trigger rule, invalidation and target logic, and the name of the
management profile Pass 4 will execute. The detector is a stateless
function of one snapshot — the Pass 1 whole-market read plus the
timeframe bars — returning zero or more :class:`Candidate`s at a
declared maturity (``detected``: premise present, plan provisional;
``armed``: plan complete, awaiting trigger; ``rejected``: premise
completed but its own geometry floor failed — the setup is skipped,
never re-targeted).

Doctrine shapes (settled): entries are at zones OF THE ENTRY
TIMEFRAME; the hierarchy directs (HTF bias / LTF trigger), it never
votes — a playbook's required alignment is the bias of the level
above its entry timeframe, and a counter-trend swing bias is
recorded as evidence, never a veto (hard veto OFF); phase comes from
structure. Everything here is SHADOW: detectors find and arm setups,
nothing executes (Pass 4 owns execution).

The seeded tranche spans the families: trend/structure (FVG and
order-block retests on H1), range (range-edge reversion on H1),
reversal (Quasimodo on M5 and D1 sharing one engine with per-
timeframe params, liquidity-sweep reversal on H1) and scalp (an M5
hot-scalp analogue: H1-directed momentum pullbacks with staged
targets). Every registered playbook detects from Devon's own
organs — ``ta/structure``, ``ta/smc``, ``ta/indicators`` and the
Pass 1 read — or honestly never fires; there are no placeholders.

Placement: ``agents/`` beside :mod:`nomorals.agents.market_read` —
the layer from which the TA library, the finance journal and the
market-data integrations are all legal static imports (Pass 1's
placement note applies unchanged).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..ta import indicators, smc, structure

_log = logging.getLogger(__name__)

#: Seconds per provider timeframe spelling.
TF_SECONDS = {"1d": 86400, "4h": 14400, "1h": 3600,
              "30m": 1800, "15m": 900, "5m": 300}
TF_LABELS = {"1d": "D1", "4h": "H4", "1h": "H1",
             "30m": "M30", "15m": "M15", "5m": "M5"}

#: The lifecycle reason vocabulary (see agents/setups.py).
REASON_PATTERN_FORMED = "pattern_formed"
REASON_TARGET_GEOMETRY = "target_invalid_geometry"


def ts_epoch(ts: Any) -> float:
    """Epoch seconds for a bar-index label (naive reads as UTC)."""
    import calendar

    if hasattr(ts, "timestamp") and getattr(ts, "tzinfo", None) is not None:
        return float(ts.timestamp())
    if hasattr(ts, "timetuple"):
        return float(calendar.timegm(ts.timetuple()))
    return float(ts)


@dataclass(frozen=True)
class QMParams:
    """One QM timeframe's numbers (GS qm_m5/qm_d1 shape).

    The ATR-relative thresholds are scale-free and identical on
    every timeframe; only the bar-count windows are timeframe-
    specific calendar-structural equivalents, never blind rescalings.
    """

    swing_n: int = 3
    min_leg_atr: float = 0.5
    min_fail_atr: float = 0.25
    stop_buffer_atr: float = 0.25
    min_target_r: float = 1.0
    max_pattern_bars: int = 120
    max_retrace_bars: int = 48


@dataclass
class DetectContext:
    """One snapshot handed to every detector in a shadow pass."""

    symbol: str
    market: str
    read: dict[str, Any]
    bars: dict[str, Any]
    now: float
    _atr_cache: dict[str, float] = field(default_factory=dict)

    def tf_block(self, timeframe: str) -> dict[str, Any]:
        return (self.read.get("timeframes") or {}).get(timeframe) or {}

    def frame(self, timeframe: str) -> Any:
        return self.bars.get(timeframe)

    def atr(self, timeframe: str) -> float:
        """Latest ATR of the timeframe's own bars (0 when unknown)."""
        if timeframe not in self._atr_cache:
            value = 0.0
            df = self.frame(timeframe)
            if df is not None and len(df) >= 15:
                try:
                    value = float(indicators.atr(df).iloc[-1])
                except Exception as exc:  # noqa: BLE001 - evidence only
                    _log.debug("playbooks: ATR failed for %s: %s",
                               timeframe, exc)
            self._atr_cache[timeframe] = value if value > 0 else 0.0
        return self._atr_cache[timeframe]

    def price(self) -> float | None:
        block = self.read.get("price") or {}
        value = block.get("price")
        return float(value) if value is not None else None

    def hierarchy_bias(self, level: str) -> int | None:
        entry = (self.read.get("hierarchy") or {}).get(level) or {}
        bias = entry.get("bias")
        return int(bias) if bias is not None else None


@dataclass
class Candidate:
    """One detector finding, at a declared maturity."""

    playbook: str
    pattern_key: str
    direction: int  # +1 long / -1 short
    maturity: str  # "detected" | "armed" | "rejected"
    entry_zone_top: float
    entry_zone_bottom: float
    entry_ref: float
    invalidation: float
    targets: tuple[float, ...]
    anchor_ts: float
    reason: str
    reason_category: str = REASON_PATTERN_FORMED
    evidence: dict[str, Any] = field(default_factory=dict)

    def plan(self) -> dict[str, Any]:
        return {"entry_zone": {"top": self.entry_zone_top,
                               "bottom": self.entry_zone_bottom},
                "entry_ref": self.entry_ref,
                "invalidation": self.invalidation,
                "targets": list(self.targets)}


@dataclass(frozen=True)
class Playbook:
    """The declarative playbook definition (the FORM)."""

    name: str  # the route name, used in the journal and surfaces
    family: str  # trend | range | reversal | scalp
    description: str
    entry_timeframe: str  # provider spelling: "5m" | "1h" | "1d"
    entry_clock: str  # the bar close this playbook trades on
    bias_level: str | None  # hierarchy level that directs it
    required_read: tuple[str, ...]
    arming_rule: str
    trigger_kind: str  # "zone_touch" (Pass 4 may add kinds)
    trigger_rule: str
    invalidation_rule: str
    target_logic: str
    management_profile: str  # name only this pass; Pass 4 executes it
    detected_ttl_bars: int
    armed_ttl_bars: int
    detector: Callable[[DetectContext, Playbook], list[Candidate]]
    qm: QMParams | None = None
    recency_bars: int = 0  # bar-scanning forms: freshness window

    @property
    def entry_label(self) -> str:
        return TF_LABELS[self.entry_timeframe]


# ── Quasimodo (one engine, per-timeframe params) ─────────────────────

def _zigzag(pts: Any) -> list[dict[str, Any]]:
    """Alternating pivot list; same-side repeats keep the extreme."""
    out: list[dict[str, Any]] = []
    for _, row in pts.iterrows():
        piv = {"bar": row["bar"], "kind": int(row["kind"]),
               "price": float(row["price"])}
        if out and out[-1]["kind"] == piv["kind"]:
            last = out[-1]
            more_extreme = (piv["price"] > last["price"]
                            if piv["kind"] == 1
                            else piv["price"] < last["price"])
            if more_extreme:
                out[-1] = piv
            continue
        out.append(piv)
    return out


def _detect_qm(ctx: DetectContext, pb: Playbook) -> list[Candidate]:
    """Quasimodo reversal on the playbook's entry timeframe.

    Head -> neckline pullback -> failed rally/retest (the QM level)
    -> close through the neckline -> retracement to the QM level.
    Detection runs on the timeframe's own bars via ta/structure
    fractal pivots; the QM level freezes at the break bar; a pattern
    whose neckline target pays < min_target_r at the break is
    rejected, never re-targeted.
    """
    p = pb.qm
    assert p is not None  # noqa: S101 - registry invariant
    df = ctx.frame(pb.entry_timeframe)
    if df is None or len(df) < 40:
        return []
    atr = ctx.atr(pb.entry_timeframe)
    if atr <= 0:
        return []
    try:
        pts = structure.swing_points(df, left=p.swing_n, right=p.swing_n)
    except Exception as exc:  # noqa: BLE001 - one form down ≠ pass down
        _log.warning("playbooks: %s swing scan failed: %s", pb.name, exc)
        return []
    if len(pts) < 3:
        return []
    pivots = _zigzag(pts)
    if len(pivots) < 3:
        return []
    pos = {ts: i for i, ts in enumerate(df.index)}
    last_pos = len(df) - 1
    # Scan triples newest -> oldest: post-break bars form new pivots,
    # so the freshest triple is often a fragment of the live
    # pattern's aftermath rather than the pattern itself. The first
    # triple that evaluates to a candidate wins; invalid or dead
    # triples fall through to older ones.
    for ti in range(len(pivots) - 3, max(len(pivots) - 9, -1), -1):
        cand = _qm_eval(pb, p, pivots, ti, df, pos, last_pos, atr)
        if cand is not None:
            return [cand]
    return []


def _qm_eval(pb: Playbook, p: QMParams, pivots: list[dict[str, Any]],
             ti: int, df: Any, pos: dict[Any, int], last_pos: int,
             atr: float) -> Candidate | None:
    """Evaluate one pivot triple as a QM; None = not a live pattern."""
    a, b, c = pivots[ti], pivots[ti + 1], pivots[ti + 2]
    if not (a["kind"] == -b["kind"] == c["kind"]):
        return None
    direction = -a["kind"]  # H-L-H -> short; L-H-L -> long
    if direction == -1:
        leg1 = a["price"] - b["price"]
        leg2 = c["price"] - b["price"]
        fail = a["price"] - c["price"]
    else:
        leg1 = b["price"] - a["price"]
        leg2 = b["price"] - c["price"]
        fail = c["price"] - a["price"]
    if leg1 < p.min_leg_atr * atr or leg2 < p.min_leg_atr * atr \
            or fail < p.min_fail_atr * atr:
        return None
    a_pos, c_pos = pos[a["bar"]], pos[c["bar"]]
    if last_pos - a_pos > p.max_pattern_bars:
        return None  # stale formation: honestly not a live premise
    head, neckline, qm_level = a["price"], b["price"], c["price"]
    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    key = (f"qm:{ts_epoch(a['bar']):.0f}:{head:.2f}:{neckline:.2f}")
    evidence: dict[str, Any] = {
        "head": head, "neckline": neckline, "qm_level": qm_level,
        "head_bar": str(a["bar"]), "neckline_bar": str(b["bar"]),
        "qm_bar": str(c["bar"]), "atr": round(atr, 4)}
    if ti >= 2 and pivots[ti - 2]["kind"] == a["kind"]:
        prior = pivots[ti - 2]["price"]
        if (direction == -1 and head > prior) or \
                (direction == 1 and head < prior):
            evidence["tags"] = ["sweep_head"]
    # Pre-break death: a close beyond the head un-forms the pattern.
    break_pos: int | None = None
    for j in range(c_pos + 1, last_pos + 1):
        close = float(closes.iloc[j])
        if (direction == -1 and close > head) or \
                (direction == 1 and close < head):
            return None
        if (direction == -1 and close < neckline) or \
                (direction == 1 and close > neckline):
            break_pos = j
            break
    if break_pos is None:
        return Candidate(
            playbook=pb.name, pattern_key=key, direction=direction,
            maturity="detected",
            entry_zone_top=qm_level, entry_zone_bottom=qm_level,
            entry_ref=qm_level, invalidation=head,
            targets=(neckline,), anchor_ts=ts_epoch(a["bar"]),
            reason=(f"QM triple formed: head {head:.2f} -> neckline "
                    f"{neckline:.2f} -> failed retest {qm_level:.2f}; "
                    f"awaiting neckline break"),
            evidence=evidence)
    # Broken: expiry, invalidation and a spent touch all kill it.
    if last_pos - break_pos > p.max_retrace_bars:
        return None
    for j in range(break_pos + 1, last_pos + 1):
        close = float(closes.iloc[j])
        if (direction == -1 and close > head) or \
                (direction == 1 and close < head):
            return None  # close beyond the head after the break: dead
        touched = (float(highs.iloc[j]) >= qm_level if direction == -1
                   else float(lows.iloc[j]) <= qm_level)
        if touched:
            return None  # retracement already happened: pattern spent
    # Stop sits beyond the head by the standard 0.25 ATR buffer;
    # the neckline target must pay >= min_target_r against it.
    stop = head - direction * p.stop_buffer_atr * atr
    risk = abs(qm_level - stop)
    target_r = abs(neckline - qm_level) / risk if risk > 0 else 0.0
    break_bar = df.index[break_pos]
    body = abs(float(closes.iloc[break_pos])
               - float(df["open"].iloc[break_pos]))
    if body >= 1.0 * atr:
        evidence.setdefault("tags", []).append("displacement_break")
    evidence["break_bar"] = str(break_bar)
    if target_r < p.min_target_r:
        return Candidate(
            playbook=pb.name, pattern_key=key, direction=direction,
            maturity="rejected",
            entry_zone_top=qm_level, entry_zone_bottom=qm_level,
            entry_ref=qm_level, invalidation=stop,
            targets=(neckline,), anchor_ts=ts_epoch(a["bar"]),
            reason=(f"neckline target pays {target_r:.2f}R < "
                    f"{p.min_target_r:.1f}R floor — skipped, "
                    f"not re-targeted"),
            reason_category=REASON_TARGET_GEOMETRY,
            evidence=evidence)
    return Candidate(
        playbook=pb.name, pattern_key=key, direction=direction,
        maturity="armed",
        entry_zone_top=qm_level, entry_zone_bottom=qm_level,
        entry_ref=qm_level, invalidation=stop,
        targets=(neckline,), anchor_ts=ts_epoch(a["bar"]),
        reason=(f"neckline broke at {str(break_bar)[:16]}; resting "
                f"at the QM level {qm_level:.2f}, stop beyond the "
                f"head, target the broken neckline "
                f"({target_r:.2f}R)"),
        evidence=evidence)


# ── liquidity-sweep reversal (bars) ──────────────────────────────────

def _detect_sweep(ctx: DetectContext, pb: Playbook) -> list[Candidate]:
    """Sweep of a confirmed swing + close back inside (ta/smc).

    The reclaim bar completes the premise and the plan: retest zone
    around the swept level, invalidation beyond the sweep extreme,
    target the nearest opposing structure zone (or swing).
    """
    df = ctx.frame(pb.entry_timeframe)
    if df is None or len(df) < 40:
        return []
    atr = ctx.atr(pb.entry_timeframe)
    if atr <= 0:
        return []
    try:
        sweeps = smc.liquidity_sweeps(df)
        zones = structure.sr_zones(df)
        pts = structure.swing_points(df)
    except Exception as exc:  # noqa: BLE001 - one form down ≠ pass down
        _log.warning("playbooks: %s sweep scan failed: %s", pb.name, exc)
        return []
    if sweeps.empty:
        return []
    last = sweeps.iloc[-1]
    reclaim_pos = df.index.get_loc(last["bar"])
    if len(df) - 1 - reclaim_pos > pb.recency_bars:
        return []  # the reclaim is history; entering now is chasing
    direction = int(last["kind"])
    level = float(last["level"])
    # Sweep extreme: the furthest penetration among bars that
    # actually crossed the level inside the sweep window.
    extreme = level
    for j in range(max(0, reclaim_pos - 23), reclaim_pos + 1):
        if direction == 1 and float(df["low"].iloc[j]) < level:
            extreme = min(extreme, float(df["low"].iloc[j]))
        elif direction == -1 and float(df["high"].iloc[j]) > level:
            extreme = max(extreme, float(df["high"].iloc[j]))
    price = ctx.price()
    if price is None:
        price = float(df["close"].iloc[-1])
    levels = structure.nearest_levels(zones, price) if not zones.empty \
        else {"support": None, "resistance": None}
    targets: list[float] = []
    opp = levels["resistance"] if direction == 1 else levels["support"]
    if opp is not None:
        targets.append(float(opp["mid"]))
    else:
        # Opposing swings: highs above for a long, lows below for a
        # short — the swing kind equals the trade direction.
        swings = pts[pts["kind"] == direction]
        beyond = [float(r["price"]) for _, r in swings.iterrows()
                  if (float(r["price"]) - price) * direction > 0]
        if beyond:
            targets.append(min(beyond, key=lambda v: abs(v - price)))
    if not targets:
        return []  # no structural target: honestly no setup
    invalidation = extreme - direction * 0.25 * atr
    return [Candidate(
        playbook=pb.name,
        pattern_key=f"sweep:{level:.2f}:{ts_epoch(last['bar']):.0f}",
        direction=direction, maturity="armed",
        entry_zone_top=level + 0.25 * atr,
        entry_zone_bottom=level - 0.25 * atr,
        entry_ref=level, invalidation=invalidation,
        targets=tuple(targets), anchor_ts=ts_epoch(last["bar"]),
        reason=(f"liquidity sweep of {level:.2f} "
                f"({float(last['penetration_atr']):.2f} ATR penetration) "
                f"reclaimed; retest zone around the swept level, "
                f"invalidation beyond the sweep extreme "
                f"{extreme:.2f}"),
        evidence={"level": level, "extreme": extreme,
                  "penetration_atr": float(last["penetration_atr"]),
                  "reclaim_bar": str(last["bar"]),
                  "atr": round(atr, 4)})]


# ── read-consuming forms (FVG / order block / range / hot scalp) ────

def _above_bias(ctx: DetectContext, pb: Playbook) -> int | None:
    if pb.bias_level is None:
        return None
    return ctx.hierarchy_bias(pb.bias_level)


def _committee_evidence(ctx: DetectContext, pb: Playbook,
                        direction: int) -> dict[str, Any]:
    """Committee position as EVIDENCE for trend forms.

    The committee is a vote; the hierarchy directs and nothing here
    gates on it (hard veto OFF). Recorded so Pass 5 can grade
    whether agreement meant anything.
    """
    try:
        from ..ta import pipeline

        df = ctx.frame(pb.entry_timeframe)
        if df is None or len(df) < 30:
            return {"committee": "unavailable"}
        pos = pipeline.committee_position(df)
        value = float(pos.iloc[-1])
        stance = ("aligned" if value * direction > 0.05 else
                  "opposed" if value * direction < -0.05 else "flat")
        return {"committee_position": round(value, 3),
                "committee_stance": stance}
    except Exception as exc:  # noqa: BLE001 - evidence, never a gate
        _log.debug("playbooks: committee evidence failed: %s", exc)
        return {"committee": "unavailable"}


def _zone_targets(block: dict[str, Any], direction: int,
                  entry_ref: float) -> list[float]:
    """Opposing structure from the read: S/R zone, then opposing gap."""
    zones = block.get("zones") or {}
    fvgs = block.get("fvgs") or {}
    out: list[float] = []
    opp_zone = zones.get("resistance") if direction == 1 \
        else zones.get("support")
    if opp_zone and (float(opp_zone["mid"]) - entry_ref) * direction > 0:
        out.append(float(opp_zone["mid"]))
    opp_gap = fvgs.get("above") if direction == 1 else fvgs.get("below")
    if opp_gap and (float(opp_gap["mid"]) - entry_ref) * direction > 0 \
            and float(opp_gap["mid"]) not in out:
        out.append(float(opp_gap["mid"]))
    # Staged in the order price would meet them.
    out.sort(key=lambda t: (t - entry_ref) * direction)
    return out


def _detect_gap_form(ctx: DetectContext, pb: Playbook,
                     block_key: str, key_prefix: str,
                     noun: str) -> list[Candidate]:
    """Shared FVG / order-block retest form over the Pass 1 read.

    Requires an unmitigated gap/block on the entry timeframe in the
    direction the hierarchy level above is biased; the swing bias is
    recorded as context (a counter-swing tag), never a veto.
    """
    block = ctx.tf_block(pb.entry_timeframe)
    if block.get("status") in (None, "missing"):
        return []
    gaps = block.get(block_key) or {}
    bias = _above_bias(ctx, pb)
    if bias in (None, 0):
        return []
    direction = bias
    gap = gaps.get("below") if direction == 1 else gaps.get("above")
    want = "bullish" if direction == 1 else "bearish"
    if not gap or gap.get("kind") != want:
        return []
    atr = ctx.atr(pb.entry_timeframe)
    top, bottom = float(gap["top"]), float(gap["bottom"])
    entry_ref = top if direction == 1 else bottom
    invalidation = bottom - 0.25 * atr if direction == 1 \
        else top + 0.25 * atr
    targets = _zone_targets(block, direction, entry_ref)
    if not targets:
        return []
    evidence: dict[str, Any] = {
        "gap": {"top": top, "bottom": bottom, "mid": float(gap["mid"])},
        "bias_level": pb.bias_level, "bias": bias, "atr": round(atr, 4)}
    swing = ctx.hierarchy_bias("swing")
    if swing is not None and swing * direction < 0:
        evidence["tags"] = ["counter_swing"]
    evidence.update(_committee_evidence(ctx, pb, direction))
    side = "long" if direction == 1 else "short"
    return [Candidate(
        playbook=pb.name,
        pattern_key=f"{key_prefix}:{want}:{bottom:.2f}:{top:.2f}",
        direction=direction, maturity="armed",
        entry_zone_top=top, entry_zone_bottom=bottom,
        entry_ref=entry_ref, invalidation=invalidation,
        targets=tuple(targets), anchor_ts=ctx.now,
        reason=(f"unmitigated {noun} {bottom:.2f}–{top:.2f} on "
                f"{pb.entry_label} with {pb.bias_level} bias {side}; "
                f"resting at the near edge, invalidation beyond the "
                f"{'low' if direction == 1 else 'high'}"),
        evidence=evidence)]


def _detect_fvg(ctx: DetectContext, pb: Playbook) -> list[Candidate]:
    return _detect_gap_form(ctx, pb, "fvgs", "fvg", "fair-value gap")


def _detect_order_block(ctx: DetectContext,
                        pb: Playbook) -> list[Candidate]:
    return _detect_gap_form(ctx, pb, "order_blocks", "ob", "order block")


def _detect_range_edge(ctx: DetectContext,
                       pb: Playbook) -> list[Candidate]:
    """Range-edge reversion: trade the edge back to the mid.

    Requires the entry timeframe's structure phase to be RANGE with
    both edges defined in the read, and price in the discount
    (long) or premium (short) third of the dealing range. Mid-range
    is honestly no setup.
    """
    block = ctx.tf_block(pb.entry_timeframe)
    if block.get("status") in (None, "missing"):
        return []
    if block.get("phase") != "range":
        return []
    zones = block.get("zones") or {}
    support, resistance = zones.get("support"), zones.get("resistance")
    pd_block = block.get("premium_discount") or {}
    price = ctx.price()
    if not support or not resistance or price is None:
        return []
    zone_name = pd_block.get("zone")
    atr = ctx.atr(pb.entry_timeframe)
    range_mid = (float(pd_block["range_high"])
                 + float(pd_block["range_low"])) / 2 \
        if pd_block.get("range_high") is not None else None
    if zone_name == "discount":
        direction = 1
        edge, opp = support, resistance
    elif zone_name == "premium":
        direction = -1
        edge, opp = resistance, support
    else:
        return []  # equilibrium: mid-range is no-trade
    top, bottom = float(edge["top"]), float(edge["bottom"])
    entry_ref = top if direction == 1 else bottom
    invalidation = bottom - 0.25 * atr if direction == 1 \
        else top + 0.25 * atr
    targets = [t for t in (range_mid, float(opp["mid"]))
               if t is not None and (t - entry_ref) * direction > 0]
    if not targets:
        return []
    key = (f"range:{direction}:{float(support['mid']):.2f}:"
           f"{float(resistance['mid']):.2f}")
    return [Candidate(
        playbook=pb.name, pattern_key=key, direction=direction,
        maturity="armed",
        entry_zone_top=top, entry_zone_bottom=bottom,
        entry_ref=entry_ref, invalidation=invalidation,
        targets=tuple(targets), anchor_ts=ctx.now,
        reason=(f"{pb.entry_label} structure is a range "
                f"{float(support['mid']):.2f}–"
                f"{float(resistance['mid']):.2f} and price sits in "
                f"{zone_name}; edge reversion toward the mid"),
        evidence={"support_mid": float(support["mid"]),
                  "resistance_mid": float(resistance["mid"]),
                  "premium": pd_block.get("premium"),
                  "atr": round(atr, 4)})]


def _detect_hot_scalp(ctx: DetectContext,
                      pb: Playbook) -> list[Candidate]:
    """Hot-scalp analogue: H1-directed M5 momentum pullbacks.

    The scalp (H1) bias directs; the M5 block must show a structure
    break in that direction (fresh momentum, from structure — never
    an indicator cross); the entry is the pullback to the nearest
    M5 zone in the bias direction, with staged targets (M5 opposing
    zone, then the H1 opposing zone) and a runner beyond.
    """
    block = ctx.tf_block(pb.entry_timeframe)
    if block.get("status") in (None, "missing"):
        return []
    direction = ctx.hierarchy_bias("scalp")
    if direction in (None, 0):
        return []
    brk = block.get("last_break") or {}
    want = "BOS_BULL" if direction == 1 else "BOS_BEAR"
    if brk.get("kind") != want:
        return []
    zones = block.get("zones") or {}
    zone = zones.get("support") if direction == 1 \
        else zones.get("resistance")
    price = ctx.price()
    if not zone or price is None:
        return []
    top, bottom = float(zone["top"]), float(zone["bottom"])
    if direction == 1 and price <= top:
        return []  # already in/below the zone: the pullback happened
    if direction == -1 and price >= bottom:
        return []
    atr = ctx.atr(pb.entry_timeframe)
    entry_ref = top if direction == 1 else bottom
    invalidation = bottom - 0.25 * atr if direction == 1 \
        else top + 0.25 * atr
    targets = _zone_targets(block, direction, entry_ref)
    h1_block = ctx.tf_block("1h")
    h1_targets = _zone_targets(h1_block, direction, entry_ref)
    for t in h1_targets:
        if all(abs(t - x) > 1e-9 for x in targets):
            targets.append(t)
    targets.sort(key=lambda t: (t - entry_ref) * direction)
    if not targets:
        return []
    key = (f"scalp:{direction}:{float(brk['level']):.2f}:"
           f"{float(zone['mid']):.2f}")
    return [Candidate(
        playbook=pb.name, pattern_key=key, direction=direction,
        maturity="armed",
        entry_zone_top=top, entry_zone_bottom=bottom,
        entry_ref=entry_ref, invalidation=invalidation,
        targets=tuple(targets), anchor_ts=ctx.now,
        reason=(f"H1 scalp bias {'long' if direction == 1 else 'short'}"
                f" with an M5 {want} at {float(brk['level']):.2f}; "
                f"pullback entry at the M5 zone "
                f"{bottom:.2f}–{top:.2f}, staged targets"),
        evidence={"break_level": float(brk["level"]),
                  "zone_mid": float(zone["mid"]), "atr": round(atr, 4),
                  "staged_targets": len(targets)})]


# ── the registry (seeded tranche) ────────────────────────────────────

_M5_QM = QMParams()  # 120-bar formation, 48-bar retracement (M5)
_D1_QM = QMParams(max_pattern_bars=60, max_retrace_bars=20)

PLAYBOOKS: tuple[Playbook, ...] = (
    Playbook(
        name="qm_m5_native", family="reversal",
        description=("Quasimodo reversal on M5: head -> neckline "
                     " pullback -> failed retest (QM level) -> close"
                     " through the neckline -> resting entry at the"
                     " QM level, stop beyond the head, target the"
                     " broken neckline"),
        entry_timeframe="5m", entry_clock="M5 close",
        bias_level=None,
        required_read=("entry-timeframe swing structure only — a "
                       "reversal form; no HTF alignment required",),
        arming_rule="close through the neckline after the failed retest",
        trigger_kind="zone_touch",
        trigger_rule="first touch of the QM level after the break",
        invalidation_rule=("close beyond the head; stop = head + "
                           "0.25 ATR buffer"),
        target_logic="the broken neckline; must pay >= 1.0R or skip",
        management_profile="qm_fixed_bracket",
        detected_ttl_bars=120, armed_ttl_bars=48,
        detector=_detect_qm, qm=_M5_QM),
    Playbook(
        name="qm_d1_native", family="reversal",
        description=("Quasimodo reversal on D1 — the same engine as"
                     " qm_m5_native with the D1 parameter row "
                     "(60-bar formation, 20-bar retracement)"),
        entry_timeframe="1d", entry_clock="D1 close",
        bias_level=None,
        required_read=("entry-timeframe swing structure only — a "
                       "reversal form; no HTF alignment required",),
        arming_rule="close through the neckline after the failed retest",
        trigger_kind="zone_touch",
        trigger_rule="first touch of the QM level after the break",
        invalidation_rule=("close beyond the head; stop = head + "
                           "0.25 ATR buffer"),
        target_logic="the broken neckline; must pay >= 1.0R or skip",
        management_profile="qm_fixed_bracket",
        detected_ttl_bars=60, armed_ttl_bars=20,
        detector=_detect_qm, qm=_D1_QM),
    Playbook(
        name="sweep_reversal_h1", family="reversal",
        description=("Liquidity sweep of a confirmed H1 swing with a"
                     " close back inside (ta/smc liquidity_sweeps);"
                     " entry on the retest of the swept level, "
                     "invalidation beyond the sweep extreme, target"
                     " the nearest opposing structure"),
        entry_timeframe="1h", entry_clock="H1 close",
        bias_level=None,
        required_read=("a confirmed sweep + reclaim on the entry "
                       "timeframe within the recency window",),
        arming_rule="the reclaim close completes the plan",
        trigger_kind="zone_touch",
        trigger_rule="retest of the band around the swept level",
        invalidation_rule="close beyond the sweep extreme + 0.25 ATR",
        target_logic="nearest opposing S/R zone, else opposing swing",
        management_profile="sweep_retest",
        detected_ttl_bars=12, armed_ttl_bars=24,
        detector=_detect_sweep, recency_bars=12),
    Playbook(
        name="fvg_retest_h1", family="trend",
        description=("Retest of an unmitigated H1 fair-value gap in"
                     " the direction the intraday (H4) bias points;"
                     " invalidation beyond the gap, targets the "
                     "opposing structure"),
        entry_timeframe="1h", entry_clock="H1 close",
        bias_level="intraday",
        required_read=("unmitigated entry-timeframe FVG present",
                       "intraday (H4) bias aligned with the gap"),
        arming_rule="gap present and aligned: the resting plan is complete",
        trigger_kind="zone_touch",
        trigger_rule="first touch of the gap after arming",
        invalidation_rule="close beyond the gap far edge + 0.25 ATR",
        target_logic="nearest opposing S/R zone, then opposing gap",
        management_profile="zone_retest",
        detected_ttl_bars=48, armed_ttl_bars=48,
        detector=_detect_fvg),
    Playbook(
        name="order_block_h1", family="trend",
        description=("Retest of an unmitigated H1 order block in the"
                     " direction the intraday (H4) bias points;"
                     " invalidation beyond the block, targets the"
                     " opposing structure"),
        entry_timeframe="1h", entry_clock="H1 close",
        bias_level="intraday",
        required_read=("unmitigated entry-timeframe order block present",
                       "intraday (H4) bias aligned with the block"),
        arming_rule="block present and aligned: the resting plan is complete",
        trigger_kind="zone_touch",
        trigger_rule="first touch of the block after arming",
        invalidation_rule="close beyond the block far edge + 0.25 ATR",
        target_logic="nearest opposing S/R zone, then opposing gap",
        management_profile="zone_retest",
        detected_ttl_bars=48, armed_ttl_bars=48,
        detector=_detect_order_block),
    Playbook(
        name="range_edge_h1", family="range",
        description=("Range-edge reversion on H1: structure phase is"
                     " RANGE with both edges defined; buy the "
                     "discount edge / sell the premium edge back "
                     "toward the range mid, invalidation beyond the "
                     "edge"),
        entry_timeframe="1h", entry_clock="H1 close",
        bias_level=None,
        required_read=("entry-timeframe structure phase is RANGE",
                       "both range edges defined in the read",
                       "price in the discount or premium third"),
        arming_rule="edge + regime present: the resting plan is complete",
        trigger_kind="zone_touch",
        trigger_rule="first touch of the edge zone after arming",
        invalidation_rule="close beyond the edge + 0.25 ATR",
        target_logic="range mid, then the opposite edge",
        management_profile="range_mid",
        detected_ttl_bars=72, armed_ttl_bars=72,
        detector=_detect_range_edge),
    Playbook(
        name="hot_scalp_m5", family="scalp",
        description=("Hot-scalp analogue: the scalp (H1) bias "
                     "directs, an M5 structure break in that "
                     "direction confirms momentum, and the entry is "
                     "the pullback to the nearest M5 zone with "
                     "staged targets and a runner"),
        entry_timeframe="5m", entry_clock="M5 close",
        bias_level="scalp",
        required_read=("scalp (H1) bias non-flat",
                       "M5 last structure break in the bias direction",
                       "an M5 pullback zone on the entry side"),
        arming_rule="break + zone present: the resting plan is complete",
        trigger_kind="zone_touch",
        trigger_rule="first touch of the M5 pullback zone after arming",
        invalidation_rule="close beyond the zone far edge + 0.25 ATR",
        target_logic=("staged: M5 opposing zone, then H1 opposing "
                      "zone, runner beyond (management profile)"),
        management_profile="staged_runner",
        detected_ttl_bars=36, armed_ttl_bars=36,
        detector=_detect_hot_scalp),
)

REGISTRY: dict[str, Playbook] = {pb.name: pb for pb in PLAYBOOKS}


def get_playbook(name: str) -> Playbook | None:
    return REGISTRY.get(name)


def entry_timeframes() -> tuple[str, ...]:
    """Entry timeframes the registry needs bars for, registry order."""
    out: list[str] = []
    for pb in PLAYBOOKS:
        if pb.entry_timeframe not in out:
            out.append(pb.entry_timeframe)
    return tuple(out)
