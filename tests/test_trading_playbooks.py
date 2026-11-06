"""Pass 2 of Devon's trading system: playbooks + setup lifecycle.

The playbook FORM and registry (agents/playbooks.py), the detectors
running on Devon's own TA organs and the Pass 1 read, and the
event-sourced setup lifecycle + shadow driver (agents/setups.py)
over the Pass 1 trading journal.

Every test here is a NEW-BEHAVIOR test: on pristine 5512501 the
organs do not exist and the file fails at import. Preservation of
Pass 1 behavior is carried by test_market_read.py and the
neighborhood suites.
"""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from nomorals.agents import playbooks, setups
from nomorals.agents.playbooks import PLAYBOOKS, REGISTRY, Candidate, DetectContext
from nomorals.finance import trading_journal
from nomorals.integrations import market_data

BASE_TS = 1762300000.0  # 2025-11-05 ~04:26 UTC
_STEP = {"1d": 86400, "4h": 14400, "1h": 3600,
         "30m": 1800, "15m": 900, "5m": 300}


# ── bar builders ─────────────────────────────────────────────────────

def path_frame(points, *, tf="5m", start_ts=BASE_TS):
    """Piecewise-linear closes through (bars, price) legs.

    Turning bars get a wick spike so Devon's fractal swing detector
    sees strict, unique extremes — pivot prices in assertions are
    read back from the candidate evidence, not assumed.
    """
    closes = [points[0][1]]
    for n, price in points[1:]:
        prev = closes[-1]
        for i in range(1, n + 1):
            closes.append(prev + (price - prev) * i / n)
    pivot_at, acc = set(), 0
    for n, _price in points[1:]:
        acc += n
        pivot_at.add(acc)
    rows, idx = [], []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c
        hi, lo = max(o, c) + 0.05, min(o, c) - 0.05
        if i in pivot_at:
            if c >= o:
                hi = c + 0.45
            if c <= o:
                lo = c - 0.45
        rows.append((o, hi, lo, c, 100.0))
        idx.append(pd.Timestamp(start_ts + i * _STEP[tf],
                                unit="s", tz="UTC"))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close",
                                     "volume"], index=idx)
    df.attrs["source"] = "fake_src"
    df.attrs["freshness"] = "fresh"
    df.attrs["last_bar_ts"] = start_ts + (len(closes) - 1) * _STEP[tf]
    return df


def drift_frame(*, n=220, base=2600.0, drift=0.35, tf="1h"):
    closes = [base + drift * i + 6.0 * math.sin(i * 2 * math.pi / 24)
              for i in range(n)]
    rows, idx = [], []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c - drift
        rows.append((o, max(o, c) + 0.9, min(o, c) - 0.9, c, 100.0))
        idx.append(pd.Timestamp(BASE_TS + i * _STEP[tf],
                                unit="s", tz="UTC"))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close",
                                     "volume"], index=idx)
    df.attrs["source"] = "fake_src"
    df.attrs["freshness"] = "fresh"
    df.attrs["last_bar_ts"] = BASE_TS + (n - 1) * _STEP[tf]
    return df


def flat_frame(*, n=200, price=2600.0, tf="1h"):
    return path_frame([(1, price), (n - 1, price)], tf=tf)


# ── read stubs (the Pass 1 contract, hand-built) ─────────────────────

def block(**kw):
    base = {"status": "ok", "bias": None, "phase": None, "zones": None,
            "fvgs": None, "order_blocks": None,
            "premium_discount": None, "last_break": None}
    base.update(kw)
    return base


def zone_row(top, bottom, kind, state):
    return {"top": top, "bottom": bottom, "mid": (top + bottom) / 2,
            "touches": 3, "strength": 55.0, "kind": kind, "state": state}


def gap_row(kind, top, bottom):
    return {"kind": kind, "top": top, "bottom": bottom,
            "mid": (top + bottom) / 2}


def read_stub(price, tfs, hierarchy, *, taken_at=BASE_TS):
    return {"kind": "market_read", "taken_at": taken_at,
            "price": {"price": price}, "hierarchy": hierarchy,
            "timeframes": tfs}


def dctx(read, bars, *, now=BASE_TS):
    return DetectContext(symbol="XAUUSD", market="forex", read=read,
                         bars=bars, now=now)


# ── the form + registry ──────────────────────────────────────────────

def test_registry_form_is_complete_and_covers_the_families():
    families = {pb.family for pb in PLAYBOOKS}
    assert families == {"trend", "range", "reversal", "scalp"}
    names = [pb.name for pb in PLAYBOOKS]
    assert len(names) == len(set(names)) == len(REGISTRY)
    for pb in PLAYBOOKS:
        assert pb.entry_timeframe in playbooks.TF_SECONDS
        assert pb.entry_clock and pb.arming_rule and pb.trigger_rule
        assert pb.invalidation_rule and pb.target_logic
        assert pb.management_profile and pb.required_read
        assert pb.trigger_kind == "zone_touch"
        assert pb.detected_ttl_bars > 0 and pb.armed_ttl_bars > 0
    # The seeded tranche: QM on both clocks, sweep, FVG, OB, range,
    # the M5 hot-scalp analogue.
    assert set(names) == {"qm_m5_native", "qm_d1_native",
                          "sweep_reversal_h1", "fvg_retest_h1",
                          "order_block_h1", "range_edge_h1",
                          "hot_scalp_m5"}
    assert playbooks.entry_timeframes() == ("5m", "1d", "1h")


# ── QM detector ──────────────────────────────────────────────────────

_QM_BEAR = [(1, 2600.0), (12, 2650.0), (12, 2630.0), (12, 2642.0),
            (10, 2633.0), (3, 2628.0), (6, 2632.0)]


def test_qm_m5_arms_after_neckline_break():
    pb = REGISTRY["qm_m5_native"]
    df = path_frame(_QM_BEAR, tf="5m")
    read = read_stub(2632.0, {}, {})
    cands = pb.detector(dctx(read, {"5m": df}), pb)
    assert len(cands) == 1
    c = cands[0]
    ev = c.evidence
    assert c.maturity == "armed" and c.direction == -1
    # Zone/invalidation/targets all from the ENTRY timeframe (M5):
    # entry at the QM level, stop beyond the head by 0.25 ATR,
    # target the broken neckline.
    assert c.entry_zone_top == c.entry_zone_bottom == ev["qm_level"]
    assert c.invalidation == pytest.approx(
        ev["head"] + 0.25 * ev["atr"], abs=0.01)
    assert c.targets == (ev["neckline"],)
    assert abs(ev["head"] - 2650.45) < 0.01
    assert abs(ev["neckline"] - 2629.55) < 0.01


def test_qm_detected_before_the_break():
    pb = REGISTRY["qm_m5_native"]
    df = path_frame(_QM_BEAR[:4] + [(10, 2635.0)], tf="5m")
    cands = pb.detector(dctx(read_stub(2635.0, {}, {}), {"5m": df}), pb)
    assert len(cands) == 1
    c = cands[0]
    assert c.maturity == "detected" and c.direction == -1
    # Pre-arm the invalidation is the raw head (a close beyond it
    # un-forms the pattern); the buffered stop arrives at arming.
    assert c.invalidation == c.evidence["head"]
    assert c.targets == (c.evidence["neckline"],)


def test_qm_bullish_mirror_arms():
    pb = REGISTRY["qm_m5_native"]
    legs = [(1, 2700.0), (12, 2650.0), (12, 2670.0), (12, 2658.0),
            (10, 2667.0), (3, 2672.0), (6, 2668.0)]
    df = path_frame(legs, tf="5m")
    cands = pb.detector(dctx(read_stub(2668.0, {}, {}), {"5m": df}), pb)
    assert len(cands) == 1
    c = cands[0]
    ev = c.evidence
    assert c.maturity == "armed" and c.direction == 1
    assert c.entry_ref == ev["qm_level"]
    assert c.invalidation == pytest.approx(
        ev["head"] - 0.25 * ev["atr"], abs=0.01)
    assert c.targets == (ev["neckline"],)


def test_qm_compressed_pattern_is_rejected_not_retargeted():
    pb = REGISTRY["qm_m5_native"]
    legs = [(1, 2600.0), (12, 2650.0), (12, 2645.0), (12, 2647.0),
            (8, 2643.0)]
    df = path_frame(legs, tf="5m")
    cands = pb.detector(dctx(read_stub(2643.0, {}, {}), {"5m": df}), pb)
    assert len(cands) == 1
    c = cands[0]
    assert c.maturity == "rejected"
    assert c.reason_category == "target_invalid_geometry"
    assert "< 1.0R floor" in c.reason or "floor" in c.reason


def test_qm_d1_uses_the_d1_parameter_row():
    pb = REGISTRY["qm_d1_native"]
    assert pb.qm is not None and pb.qm.max_pattern_bars == 60
    assert pb.qm.max_retrace_bars == 20
    df = path_frame(_QM_BEAR, tf="1d")
    cands = pb.detector(dctx(read_stub(2632.0, {}, {}), {"1d": df}), pb)
    assert len(cands) == 1 and cands[0].maturity == "armed"


# ── sweep detector ───────────────────────────────────────────────────

def test_sweep_reversal_arms_from_bars():
    pb = REGISTRY["sweep_reversal_h1"]
    legs = [(1, 2620.0), (12, 2590.0), (12, 2580.0), (8, 2605.0),
            (6, 2590.0), (3, 2572.0), (3, 2586.0), (5, 2592.0)]
    df = path_frame(legs, tf="1h")
    cands = pb.detector(dctx(read_stub(2592.0, {}, {}), {"1h": df}), pb)
    assert len(cands) == 1
    c = cands[0]
    ev = c.evidence
    assert c.maturity == "armed" and c.direction == 1
    assert c.entry_ref == ev["level"]
    # Invalidation sits beyond the sweep extreme, not at it.
    assert c.invalidation < ev["extreme"] < ev["level"]
    # Target is real opposing structure (the 2605 swing high area).
    assert c.targets and c.targets[0] > 2592.0


# ── read-consuming detectors ─────────────────────────────────────────

def _fvg_read(*, intraday=1, swing=1, gap=True):
    h1 = block(
        zones={"support": None,
               "resistance": zone_row(2645.0, 2635.0, "resistance",
                                      "resistance"),
               "inside": []},
        fvgs={"above": None,
              "below": gap_row("bullish", 2610.0, 2600.0) if gap
              else None,
              "open_count": 1 if gap else 0})
    return read_stub(2620.0, {"1h": h1},
                     {"swing": {"bias": swing},
                      "intraday": {"bias": intraday},
                      "scalp": {"bias": 1}})


def test_fvg_arms_from_the_read_with_entry_tf_geometry():
    pb = REGISTRY["fvg_retest_h1"]
    bars = {"1h": drift_frame(tf="1h")}
    cands = pb.detector(dctx(_fvg_read(), bars), pb)
    assert len(cands) == 1
    c = cands[0]
    assert c.maturity == "armed" and c.direction == 1
    assert (c.entry_zone_bottom, c.entry_zone_top) == (2600.0, 2610.0)
    atr = c.evidence["atr"]
    assert atr > 0
    assert c.invalidation == pytest.approx(2600.0 - 0.25 * atr)
    assert c.targets == (2640.0,)
    # Committee stance rides as evidence — recorded, never a gate.
    assert "committee_position" in c.evidence


def test_fvg_needs_the_hierarchy_level_above_aligned():
    pb = REGISTRY["fvg_retest_h1"]
    bars = {"1h": drift_frame(tf="1h")}
    # H4 (intraday) against the gap: the playbook honestly never fires.
    assert pb.detector(dctx(_fvg_read(intraday=-1), bars), pb) == []
    # No gap at all: no fire.
    assert pb.detector(dctx(_fvg_read(gap=False), bars), pb) == []


def test_fvg_counter_swing_is_evidence_not_a_veto():
    pb = REGISTRY["fvg_retest_h1"]
    bars = {"1h": drift_frame(tf="1h")}
    cands = pb.detector(dctx(_fvg_read(swing=-1), bars), pb)
    assert len(cands) == 1  # hard veto OFF: D1 disagreement never blocks
    assert cands[0].evidence.get("tags") == ["counter_swing"]


def test_order_block_arms_from_the_read():
    pb = REGISTRY["order_block_h1"]
    h1 = block(
        zones={"support": None,
               "resistance": zone_row(2645.0, 2635.0, "resistance",
                                      "resistance"),
               "inside": []},
        order_blocks={"above": None,
                      "below": gap_row("bullish", 2612.0, 2602.0),
                      "open_count": 1})
    read = read_stub(2620.0, {"1h": h1},
                     {"swing": {"bias": 1}, "intraday": {"bias": 1},
                      "scalp": {"bias": 1}})
    bars = {"1h": drift_frame(tf="1h")}
    cands = pb.detector(dctx(read, bars), pb)
    assert len(cands) == 1
    c = cands[0]
    assert c.direction == 1
    assert (c.entry_zone_bottom, c.entry_zone_top) == (2602.0, 2612.0)
    assert c.targets == (2640.0,)


def _range_read(*, pd_zone="discount", price=2595.0, phase="range"):
    h1 = block(
        phase=phase,
        zones={"support": zone_row(2585.0, 2575.0, "support", "support"),
               "resistance": zone_row(2685.0, 2675.0, "resistance",
                                      "resistance"),
               "inside": []},
        premium_discount={"premium": 0.2 if pd_zone == "discount"
                          else 0.5 if pd_zone == "equilibrium" else 0.85,
                          "zone": pd_zone, "range_high": 2680.0,
                          "range_low": 2580.0})
    return read_stub(price, {"1h": h1},
                     {"swing": {"bias": 0}, "intraday": {"bias": 0},
                      "scalp": {"bias": 0}})


def test_range_edge_arms_at_the_discount_edge():
    pb = REGISTRY["range_edge_h1"]
    bars = {"1h": drift_frame(tf="1h", drift=0.0)}
    cands = pb.detector(dctx(_range_read(), bars), pb)
    assert len(cands) == 1
    c = cands[0]
    assert c.direction == 1
    assert (c.entry_zone_bottom, c.entry_zone_top) == (2575.0, 2585.0)
    # Targets staged: range mid first, then the opposite edge.
    assert c.targets == (2630.0, 2680.0)


def test_range_edge_silent_mid_range_and_in_trend():
    pb = REGISTRY["range_edge_h1"]
    bars = {"1h": drift_frame(tf="1h", drift=0.0)}
    assert pb.detector(
        dctx(_range_read(pd_zone="equilibrium", price=2630.0),
             bars), pb) == []
    assert pb.detector(
        dctx(_range_read(phase="uptrend"), bars), pb) == []


def test_hot_scalp_arms_with_staged_targets():
    pb = REGISTRY["hot_scalp_m5"]
    m5 = block(
        zones={"support": zone_row(2610.0, 2600.0, "support", "support"),
               "resistance": zone_row(2650.0, 2640.0, "resistance",
                                      "resistance"),
               "inside": []},
        fvgs={"above": None, "below": None, "open_count": 0},
        last_break={"kind": "BOS_BULL", "level": 2630.0,
                    "bias_before": 0})
    h1 = block(zones={"support": None,
                      "resistance": zone_row(2642.0, 2638.0,
                                             "resistance", "resistance"),
                      "inside": []},
               fvgs={"above": None, "below": None, "open_count": 0})
    read = read_stub(2620.0, {"5m": m5, "1h": h1},
                     {"swing": {"bias": 1}, "intraday": {"bias": 1},
                      "scalp": {"bias": 1}})
    bars = {"5m": drift_frame(tf="5m"), "1h": drift_frame(tf="1h")}
    cands = pb.detector(dctx(read, bars), pb)
    assert len(cands) == 1
    c = cands[0]
    assert c.direction == 1
    assert (c.entry_zone_bottom, c.entry_zone_top) == (2600.0, 2610.0)
    # Staged: the nearer H1 resistance (2640) before the M5 one (2645).
    assert c.targets == (2640.0, 2645.0)
    # No M5 break in the bias direction: no fire.
    m5["last_break"] = {"kind": "BOS_BEAR", "level": 2590.0,
                        "bias_before": 0}
    assert pb.detector(dctx(read, bars), pb) == []


def test_no_phantom_candidates_on_a_flat_market():
    flat = flat_frame(n=200)
    read = read_stub(2600.0,
                     {tf: block(phase="range") for tf in
                      ("1d", "4h", "1h", "5m")},
                     {"swing": {"bias": 0}, "intraday": {"bias": 0},
                      "scalp": {"bias": 0}})
    ctx = dctx(read, {"5m": flat, "1h": flat, "1d": flat})
    for pb in PLAYBOOKS:
        assert pb.detector(ctx, pb) == [], pb.name


# ── lifecycle machinery (scripted playbooks) ─────────────────────────

def _scripted(script, *, name="fake_h1", detected_ttl=48, armed_ttl=48):
    queue = list(script)

    def detector(ctx, pb):
        if queue:
            return queue.pop(0)
        return []

    return playbooks.Playbook(
        name=name, family="trend", description="scripted test double",
        entry_timeframe="1h", entry_clock="H1 close", bias_level=None,
        required_read=("scripted",), arming_rule="scripted",
        trigger_kind="zone_touch", trigger_rule="scripted",
        invalidation_rule="scripted", target_logic="scripted",
        management_profile="test_profile",
        detected_ttl_bars=detected_ttl, armed_ttl_bars=armed_ttl,
        detector=detector)


def _cand(name, key, *, maturity="armed", direction=1, top=102.0,
          bottom=100.0, inv=95.0, targets=(110.0,)):
    return Candidate(
        playbook=name, pattern_key=key, direction=direction,
        maturity=maturity, entry_zone_top=top, entry_zone_bottom=bottom,
        entry_ref=top if direction == 1 else bottom, invalidation=inv,
        targets=tuple(targets), anchor_ts=BASE_TS,
        reason=f"scripted {maturity} {key}", evidence={})


def _driver_read(*, taken_at, tf_status="ok"):
    return {"kind": "market_read", "taken_at": taken_at,
            "symbol": "XAUUSD", "price": {"price": 105.0},
            "hierarchy": {},
            "timeframes": {"1h": {"status": tf_status}}}


def _pass(journal, pb, *, now, read, bars):
    return setups.shadow_pass(
        "XAUUSD", market="forex", journal_path=journal, now=now,
        read=read, bars=bars, playbooks=(pb,))


def _events(journal, sid_part=None):
    out = []
    for rec in trading_journal.read_events(limit=0, path=journal):
        if rec.get("event", "").startswith("setup_") and (
                sid_part is None
                or sid_part in rec.get("setup_id", "")):
            out.append(rec)
    return list(reversed(out))  # chronological


_BARS_A = [(1, 100.0), (30, 105.0)]  # last bar ts = BASE + 30h


def test_lifecycle_create_arm_trigger(tmp_path):
    journal = tmp_path / "j.jsonl"
    pb = _scripted([[_cand("fake_h1", "k1")], []])
    t0 = BASE_TS + 31 * 3600
    frame_a = path_frame(_BARS_A, tf="1h")
    s1 = _pass(journal, pb, now=t0, read=_driver_read(taken_at=t0),
               bars={"1h": frame_a})
    assert len(s1["created"]) == 1 and len(s1["armed"]) == 1
    sid = s1["created"][0]
    # Pass 2: a later bar trades into the zone -> triggered.
    frame_b = path_frame(_BARS_A + [(3, 100.5), (2, 104.0)], tf="1h")
    t1 = t0 + 5 * 3600
    s2 = _pass(journal, pb, now=t1, read=_driver_read(taken_at=t1),
               bars={"1h": frame_b})
    assert s2["triggered"] == [sid]
    assert s2["counts"]["zone_touched"] == 1
    state = setups.load_setups(journal_path=journal, symbol="XAUUSD")
    assert len(state) == 1
    st = state[0]
    assert st["state"] == "triggered"
    assert st["trigger"]["price"] == 102.0  # zone top for a long
    assert st["read_taken_at"] == t0  # graded against its own read
    seq = [(e["event"], e["reason_category"]) for e in _events(journal)]
    assert seq == [("setup_detected", "pattern_formed"),
                   ("setup_armed", "premise_confirmed"),
                   ("setup_triggered", "zone_touched")]


def test_lifecycle_invalidation_beats_a_late_touch(tmp_path):
    journal = tmp_path / "j.jsonl"
    pb = _scripted([[_cand("fake_h1", "k1")], []])
    t0 = BASE_TS + 31 * 3600
    _pass(journal, pb, now=t0, read=_driver_read(taken_at=t0),
          bars={"1h": path_frame(_BARS_A, tf="1h")})
    # Price falls straight through the zone AND closes beyond the
    # invalidation on ONE bar: the premise died on that bar.
    frame_b = path_frame(_BARS_A + [(1, 94.0)], tf="1h")
    t1 = t0 + 2 * 3600
    s2 = _pass(journal, pb, now=t1, read=_driver_read(taken_at=t1),
               bars={"1h": frame_b})
    assert len(s2["invalidated"]) == 1 and s2["triggered"] == []
    assert s2["counts"]["invalidation_hit"] == 1
    st = setups.load_setups(journal_path=journal)[0]
    assert st["state"] == "invalidated"


def test_lifecycle_expiry_by_ttl(tmp_path):
    journal = tmp_path / "j.jsonl"
    pb = _scripted([[_cand("fake_h1", "k1")], []], armed_ttl=2)
    t0 = BASE_TS + 31 * 3600
    frame = path_frame(_BARS_A, tf="1h")
    _pass(journal, pb, now=t0, read=_driver_read(taken_at=t0),
          bars={"1h": frame})
    t1 = t0 + 2 * 3600 + 1  # armed_ttl = 2 H1 bars
    s2 = _pass(journal, pb, now=t1, read=_driver_read(taken_at=t1),
               bars={"1h": frame})
    assert len(s2["expired"]) == 1
    assert s2["counts"]["ttl_expired"] == 1
    assert setups.load_setups(journal_path=journal)[0]["state"] \
        == "expired"


def test_lifecycle_detected_then_premise_lost(tmp_path):
    journal = tmp_path / "j.jsonl"
    pb = _scripted([[_cand("fake_h1", "k1", maturity="detected")], []])
    t0 = BASE_TS + 31 * 3600
    frame = path_frame(_BARS_A, tf="1h")
    s1 = _pass(journal, pb, now=t0, read=_driver_read(taken_at=t0),
               bars={"1h": frame})
    assert len(s1["created"]) == 1 and s1["armed"] == []
    t1 = t0 + 3600
    s2 = _pass(journal, pb, now=t1, read=_driver_read(taken_at=t1),
               bars={"1h": frame})
    assert len(s2["invalidated"]) == 1
    assert s2["counts"]["premise_lost"] == 1
    assert setups.load_setups(journal_path=journal)[0]["state"] \
        == "invalidated"


def test_lifecycle_detected_then_armed_updates_the_plan(tmp_path):
    journal = tmp_path / "j.jsonl"
    detected = _cand("fake_h1", "k1", maturity="detected", inv=97.0)
    armed = _cand("fake_h1", "k1", maturity="armed", inv=95.0)
    pb = _scripted([[detected], [armed]])
    t0 = BASE_TS + 31 * 3600
    frame = path_frame(_BARS_A, tf="1h")
    _pass(journal, pb, now=t0, read=_driver_read(taken_at=t0),
          bars={"1h": frame})
    t1 = t0 + 3600
    s2 = _pass(journal, pb, now=t1, read=_driver_read(taken_at=t1),
               bars={"1h": frame})
    assert len(s2["armed"]) == 1
    st = setups.load_setups(journal_path=journal)[0]
    assert st["state"] == "armed"
    # The armed plan (buffered stop) replaced the detected plan.
    assert st["plan"]["invalidation"] == 95.0


def test_lifecycle_dedupes_a_repeated_candidate(tmp_path):
    journal = tmp_path / "j.jsonl"
    cand = _cand("fake_h1", "k1")
    pb = _scripted([[cand], [cand]])
    t0 = BASE_TS + 31 * 3600
    frame = path_frame(_BARS_A, tf="1h")
    _pass(journal, pb, now=t0, read=_driver_read(taken_at=t0),
          bars={"1h": frame})
    t1 = t0 + 3600
    s2 = _pass(journal, pb, now=t1, read=_driver_read(taken_at=t1),
               bars={"1h": frame})
    assert s2["created"] == []
    assert s2["counts"]["already_live"] == 1
    assert len(_events(journal)) == 2  # detected + armed, once each


def test_lifecycle_supersede_replaces_a_live_setup(tmp_path):
    journal = tmp_path / "j.jsonl"
    pb = _scripted([[_cand("fake_h1", "k1")],
                    [_cand("fake_h1", "k2", top=112.0, bottom=110.0,
                           inv=105.0, targets=(120.0,))]])
    t0 = BASE_TS + 31 * 3600
    frame = path_frame(_BARS_A, tf="1h")
    s1 = _pass(journal, pb, now=t0, read=_driver_read(taken_at=t0),
               bars={"1h": frame})
    t1 = t0 + 3600
    s2 = _pass(journal, pb, now=t1, read=_driver_read(taken_at=t1),
               bars={"1h": frame})
    assert len(s2["expired"]) == 1 and len(s2["created"]) == 1
    assert s2["counts"]["superseded"] == 1
    states = {s["pattern_key"]: s["state"]
              for s in setups.load_setups(journal_path=journal)}
    assert states == {"k1": "expired", "k2": "armed"}
    assert s1["created"][0] != s2["created"][0]


def test_data_gap_pauses_a_setup_never_kills_it(tmp_path):
    journal = tmp_path / "j.jsonl"
    pb = _scripted([[_cand("fake_h1", "k1")], []])
    t0 = BASE_TS + 31 * 3600
    frame = path_frame(_BARS_A, tf="1h")
    _pass(journal, pb, now=t0, read=_driver_read(taken_at=t0),
          bars={"1h": frame})
    t1 = t0 + 3600
    s2 = _pass(journal, pb, now=t1,
               read=_driver_read(taken_at=t1, tf_status="missing"),
               bars={"1h": None})
    assert s2["counts"]["data_gap"] == 1
    assert s2["invalidated"] == [] and s2["expired"] == []
    assert setups.load_setups(journal_path=journal)[0]["state"] \
        == "armed"


def test_rejected_candidate_kills_its_detected_setup(tmp_path):
    journal = tmp_path / "j.jsonl"
    detected = _cand("fake_h1", "k1", maturity="detected")
    rejected = _cand("fake_h1", "k1", maturity="rejected")
    rejected.reason_category = "target_invalid_geometry"
    pb = _scripted([[detected], [rejected]])
    t0 = BASE_TS + 31 * 3600
    frame = path_frame(_BARS_A, tf="1h")
    _pass(journal, pb, now=t0, read=_driver_read(taken_at=t0),
          bars={"1h": frame})
    t1 = t0 + 3600
    s2 = _pass(journal, pb, now=t1, read=_driver_read(taken_at=t1),
               bars={"1h": frame})
    assert s2["counts"]["target_invalid_geometry"] == 1
    assert setups.load_setups(journal_path=journal)[0]["state"] \
        == "expired"


# ── end-to-end over the provider seam ────────────────────────────────

class FakeProviders:
    """Per-timeframe frames over the market_data seam (Pass 1 style)."""

    def __init__(self, frames):
        self.frames = frames

    def ohlcv(self, symbol, market="crypto", timeframe="1h",
              bars=500, source="auto"):
        return self.frames[timeframe]

    def quote(self, symbol, market="crypto"):
        return {"symbol": symbol, "price": 2632.0,
                "change_pct_24h": 0.8, "currency": "USD",
                "source": "fake_quote", "quote_ts": BASE_TS,
                "quote_age_s": 12.0, "freshness": "fresh",
                "bid": 2631.9, "ask": 2632.1, "spread_bps": 0.7}


@pytest.fixture
def qm_providers(monkeypatch):
    frames = {"5m": path_frame(_QM_BEAR, tf="5m"),
              "1d": drift_frame(tf="1d"),
              "4h": drift_frame(tf="4h"),
              "1h": drift_frame(tf="1h"),
              "30m": drift_frame(tf="30m"),
              "15m": drift_frame(tf="15m")}
    fake = FakeProviders(frames)
    monkeypatch.setattr(market_data, "get_ohlcv", fake.ohlcv)
    monkeypatch.setattr(market_data, "quote", fake.quote)
    return fake


def test_shadow_pass_end_to_end_arms_the_qm(qm_providers, tmp_path):
    journal = tmp_path / "trading_system_journal.jsonl"
    summary = setups.shadow_pass(
        "XAUUSD", market="forex", journal_path=journal,
        now=BASE_TS + 3600)
    qm = [s for s in summary["setups"] if s["playbook"] == "qm_m5_native"]
    assert len(qm) == 1
    st = qm[0]
    assert st["state"] == "armed" and st["direction"] == -1
    assert st["plan"]["targets"] == [st["evidence"]["neckline"]]
    assert st["read_taken_at"] == summary["read_taken_at"]
    events = [e["event"] for e in _events(journal)]
    assert events == ["setup_detected", "setup_armed"]
    reads = trading_journal.read_events(event="market_read",
                                        path=journal)
    assert len(reads) == 1  # the pass journaled its own read


def test_shadow_pass_flat_market_creates_nothing(monkeypatch, tmp_path):
    frames = {tf: flat_frame(n=200, tf=tf)
              for tf in ("1d", "4h", "1h", "30m", "15m", "5m")}
    fake = FakeProviders(frames)
    monkeypatch.setattr(market_data, "get_ohlcv", fake.ohlcv)
    monkeypatch.setattr(market_data, "quote", fake.quote)
    journal = tmp_path / "j.jsonl"
    summary = setups.shadow_pass(
        "XAUUSD", market="forex", journal_path=journal,
        now=BASE_TS + 3600)
    assert summary["created"] == [] and summary["live"] == []
    assert _events(journal) == []
    passes = trading_journal.read_events(event="shadow_pass",
                                         path=journal)
    assert len(passes) == 1  # the pass itself is still accounted


# ── surfaces ─────────────────────────────────────────────────────────

def test_recent_setups_and_render(tmp_path):
    journal = tmp_path / "j.jsonl"
    pb = _scripted([[_cand("fake_h1", "k1")], []])
    t0 = BASE_TS + 31 * 3600
    _pass(journal, pb, now=t0, read=_driver_read(taken_at=t0),
          bars={"1h": path_frame(_BARS_A, tf="1h")})
    found = setups.recent_setups("XAUUSD", journal_path=journal, now=t0)
    assert len(found) == 1 and found[0]["state"] == "armed"
    text = setups.render_setups(found, symbol="XAUUSD", now=t0)
    assert "fake_h1" in text and "armed" in text
    assert "inv 95.00" in text and "110.00" in text
    assert "premise_confirmed" in text


def test_render_pass_summarizes_counts(tmp_path):
    journal = tmp_path / "j.jsonl"
    pb = _scripted([[_cand("fake_h1", "k1")]])
    t0 = BASE_TS + 31 * 3600
    summary = _pass(journal, pb, now=t0, read=_driver_read(taken_at=t0),
                    bars={"1h": path_frame(_BARS_A, tf="1h")})
    text = setups.render_pass(summary)
    assert "created 1" in text and "armed 1" in text
    assert "pattern_formed=1" in text


def test_cmdline_scan_and_setups(qm_providers, tmp_path, capsys):
    from nomorals.cmdline.commands.trade import _cmd_trade
    settings = SimpleNamespace(home_path=tmp_path)
    context = SimpleNamespace(settings=settings)
    args = argparse.Namespace(trade_action="scan", symbol="XAUUSD",
                              market="forex", json=False)
    assert _cmd_trade(args, context) == 0
    out = capsys.readouterr().out
    assert "shadow pass XAUUSD" in out
    args2 = argparse.Namespace(trade_action="setups", symbol="XAUUSD",
                               market="forex", json=False)
    assert _cmd_trade(args2, context) == 0
    out2 = capsys.readouterr().out
    assert "qm_m5_native" in out2 and "armed" in out2


def test_trading_tool_scan_and_setups(qm_providers, tmp_path):
    from nomorals.tools import trading as trading_tool

    class FakeRegistry:
        def __init__(self, context):
            self.context = context
            self.tools = {}

        def register(self, name, **kwargs):
            def deco(fn):
                self.tools[name] = fn
                return fn
            return deco

    settings = SimpleNamespace(
        home_path=tmp_path,
        trading=SimpleNamespace(enabled=True))
    context = SimpleNamespace(settings=settings)
    reg = FakeRegistry(context)
    trading_tool.register(reg)
    out = reg.tools["trading"](action="scan", symbol="XAUUSD",
                               market="forex")
    assert out["kind"] == "shadow_pass"
    assert any(s["playbook"] == "qm_m5_native" and s["state"] == "armed"
               for s in out["setups"])
    assert "shadow pass" in out["rendered"]
    view = reg.tools["trading"](action="setups", symbol="XAUUSD",
                                market="forex")
    assert view["setups"] and "qm_m5_native" in view["rendered"]
