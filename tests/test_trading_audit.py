"""Pass 5 of Devon's trading system: validation + self-audit.

The audit organ (agents/trade_audit.py) folds the Pass 1-4 journal
against itself: forecast/outcome pairs graded per horizon,
judgment decisions graded on discrimination (medians + hit rates,
never means), setups graded per playbook through their lifecycle
plus realized R where an outcome exists. A locked holdout — the
newest 25% of graded outcomes — is the only slice a verdict may
come from; the fit slice is graded walk-forward and is all the
trust re-measurement may see.

Every test here is a NEW-BEHAVIOR test: on pristine 15c43bd the
organ does not exist and the file fails at import. Preservation
of Pass 1-4 behavior is carried by their own suites.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from nomorals.agents import trade_audit as audit
from nomorals.finance import trading_journal

BASE = 1762362000.0  # 2025-11-05 17:00 UTC


# ── journal fixture writers (the shapes Passes 1-4 write) ────────────

def _fc_pair(journal, fid, *, created, graded, horizon="1d",
             symbol="XAUUSD", direction="UP", realized="up",
             point=2000.0, end=2020.0, lo=1950.0, hi=2050.0,
             coverage=0.8, fav=30.0, adv=20.0, run_up=25.0,
             run_dn=10.0, z=0.2):
    trading_journal.append_event(
        "forecast", path=journal, ts=created, forecast_id=fid,
        horizon=horizon, symbol=symbol, direction=direction,
        point_price=point,
        interval={"low": lo, "high": hi, "coverage": coverage},
        fav_excursion=fav, adv_excursion=adv, conviction_z=z,
        deadband_pct=0.1, created=created, base_price=2000.0)
    trading_journal.append_event(
        "forecast_outcome", path=journal, ts=graded,
        forecast_id=fid, horizon=horizon, symbol=symbol,
        realized_direction=realized,
        realized={"ret_pct": 1.0, "run_up": run_up,
                  "run_dn": run_dn, "end_price": end},
        graded_at=graded)


def _setup(journal, sid, *, playbook="qm_m5_native", t0,
           states=("detected", "armed", "triggered"), step=600.0):
    plan = {"entry_ref": 2000.0, "invalidation": 1990.0,
            "targets": [2020.0],
            "entry_zone": {"top": 2002.0, "bottom": 1998.0}}
    trading_journal.append_event(
        "setup_detected", path=journal, ts=t0, setup_id=sid,
        symbol="XAUUSD", market="forex", playbook=playbook,
        route=playbook, family="reversal", entry_timeframe="5m",
        direction=1, pattern_key=sid, from_state=None,
        to_state="detected", reason_category="pattern_formed",
        plan=plan, evidence={}, detected_bar_ts=t0,
        expires_at=t0 + 10 * step)
    for i, state in enumerate(states[1:], start=1):
        trading_journal.append_event(
            f"setup_{state}", path=journal, ts=t0 + i * step,
            setup_id=sid, symbol="XAUUSD", playbook=playbook,
            from_state=states[i - 1], to_state=state,
            reason_category="zone_touched",
            armed_at=t0 + i * step if state == "armed" else None)


def _judgment(journal, sid, *, judged, decision="act",
              route="qm_m5_native", state="triggered"):
    trading_journal.append_event(
        "judgment", path=journal, ts=judged, judged_at=judged,
        kind="judgment", mode="shadow", symbol="XAUUSD",
        subject={"type": "setup", "setup_id": sid,
                 "playbook": route, "route": route, "state": state},
        decision=decision, expected_r=0.5,
        chosen_scenario="continuation", reason_category="quality")


def _position(journal, pid, sid, *, opened, closed, pnl,
              playbook="qm_m5_native", entry=2000.0, stop=1990.0):
    trading_journal.append_event(
        "position_opened", path=journal, ts=opened,
        position_id=pid, setup_id=sid, symbol="XAUUSD",
        playbook=playbook, direction=1, entry=entry,
        stop_loss=stop, initial_stop=stop, targets=[2020.0],
        volume=1.0, original_volume=1.0, contract_value=1.0)
    trading_journal.append_event(
        "position_closed", path=journal, ts=closed,
        position_id=pid, setup_id=sid, exit_price=entry + pnl,
        pnl=pnl, reason="target", price_source="broker_deal")


def _records(journal):
    return list(reversed(trading_journal.read_events(
        limit=0, path=journal)))


# ── forecast grading: exact math on known outcomes ───────────────────

def test_forecast_stats_exact_against_hand_computed(tmp_path):
    journal = tmp_path / "j.jsonl"
    rows = [
        # fid, direction, realized, end, run_up, run_dn
        ("f1", "UP", "up", 2020.0, 25.0, 10.0),      # hit, err +1%
        ("f2", "UP", "dn", 1960.0, 5.0, 40.0),       # miss, err -2%
        ("f3", "DOWN", "dn", 1990.0, 8.0, 15.0),     # hit, err -.5%
        ("f4", "UP", "up", 2060.0, 70.0, 2.0),       # hit, err +3%
        ("f5", "NEUTRAL", "neutral", 1990.0, 3.0, 3.0),  # hit
    ]
    for i, (fid, d, r, end, ru, rd) in enumerate(rows):
        _fc_pair(journal, fid, created=BASE + i * 3600,
                 graded=BASE + (i + 1) * 3600, direction=d,
                 realized=r, end=end, run_up=ru, run_dn=rd)
    pairs = audit._forecast_pairs(_records(journal), "XAUUSD")
    assert [p["covered"] for p in pairs] == \
        [True, True, True, False, True]
    stats = audit.forecast_stats(pairs)
    assert stats["n"] == 5 and stats["hits"] == 4
    assert stats["hit_rate"] == 0.8
    assert stats["realized_mix"] == {"up": 2, "dn": 2, "neutral": 1}
    assert stats["naive_rate"] == 0.4
    assert stats["lift_vs_naive"] == 2.0
    # point errors: +1, -2, -0.5, +3, -0.5 (%)
    assert stats["median_abs_point_err_pct"] == 1.0
    assert stats["median_signed_point_err_pct"] == -0.5
    assert stats["coverage_n"] == 5 and stats["covered"] == 4
    assert stats["coverage_observed"] == 0.8
    assert stats["coverage_stated"] == 0.8
    # fav capture (directional only): 25/30, 5/30, 15/30, 70/30
    assert stats["median_fav_capture"] == round((0.5 + 25 / 30) / 2, 4)
    # adv capture: 10/20, 40/20, 8/20, 2/20 -> median (0.4+0.5)/2
    assert stats["median_adv_capture"] == 0.45
    # z=0.2 everywhere: the trust pool is empty, verdict unproven
    assert stats["trust_remeasured"]["pool_n"] == 0
    assert stats["trust_remeasured"]["state"] == "unproven"


# ── judgment discrimination: medians survive a degenerate record ─────

def _judgment_fixture(journal, *, degenerate=False):
    pnls = [5.0, 10.0, -5.0, 20.0, 2.5]  # R: .5, 1, -.5, 2, .25
    if degenerate:
        pnls.append(2260.0)               # R = +226: the GS poison
    for i, pnl in enumerate(pnls):
        t = BASE + i * 86400
        sid = f"s{i}"
        _judgment(journal, sid, judged=t)
        _position(journal, f"p{i}", sid, opened=t + 60,
                  closed=t + 3600, pnl=pnl)
    # two abstentions graded by the lifecycle alone
    _setup(journal, "abs_missed", t0=BASE + 10 * 86400,
           states=("detected", "armed", "triggered"))
    _judgment(journal, "abs_missed", judged=BASE + 10 * 86400 + 700,
              decision="abstain", state="armed")
    _setup(journal, "abs_right", t0=BASE + 11 * 86400,
           states=("detected", "armed", "invalidated"))
    _judgment(journal, "abs_right", judged=BASE + 11 * 86400 + 700,
              decision="abstain", state="armed")


def test_judgment_discrimination_exact(tmp_path):
    journal = tmp_path / "j.jsonl"
    _judgment_fixture(journal)
    records = _records(journal)
    from nomorals.agents.setups import load_setups
    terminals = audit._setup_terminals(
        load_setups(journal_path=journal))
    positions = audit._position_outcomes(records, "XAUUSD")
    pairs = audit._judgment_pairs(records, terminals, positions,
                                  "XAUUSD")
    stats = audit.judgment_stats(pairs)
    assert stats["act"]["r_n"] == 5
    assert stats["act"]["median_r"] == 0.5
    assert stats["act"]["hit_rate"] == 0.8
    assert stats["abstain"]["r_n"] == 0
    assert stats["abstain"]["median_r"] is None
    assert stats["missed"] == 1
    assert stats["stood_aside_right"] == 1
    assert stats["called_wrong"] == 0
    assert stats["categories"]["r_outcome"] == 5


def test_judgment_median_survives_degenerate_record(tmp_path):
    journal = tmp_path / "j.jsonl"
    _judgment_fixture(journal, degenerate=True)
    records = _records(journal)
    from nomorals.agents.setups import load_setups
    terminals = audit._setup_terminals(
        load_setups(journal_path=journal))
    positions = audit._position_outcomes(records, "XAUUSD")
    pairs = audit._judgment_pairs(records, terminals, positions,
                                  "XAUUSD")
    stats = audit.judgment_stats(pairs)
    rs = [p["realized_r"] for p in pairs
          if p["decision"] == "act" and p["realized_r"] is not None]
    mean_r = sum(rs) / len(rs)
    assert mean_r > 30  # the mean IS poisoned by the +226R record
    # the audit's central figure is the median: (0.5 + 1.0) / 2
    assert stats["act"]["median_r"] == 0.75
    assert stats["act"]["median_r"] < 2.0
    assert "mean_r" not in stats["act"]  # means are never computed


# ── the locked holdout: fit is blind, verdicts come from holdout ─────

def _trust_fixture(journal, *, holdout_hits):
    """40 '1d' pairs: fit (30) re-measures trust at lift 1.5;
    the holdout (10) is all high-conviction — its content must not
    move the fit numbers one bit."""
    for i in range(40):
        graded = BASE + (i + 1) * 3600
        if i < 20:      # fit pool: z=1.0, 15 of 20 hit
            hit = i < 15
            _fc_pair(journal, f"f{i}", created=graded - 3600,
                     graded=graded, direction="UP",
                     realized="up" if hit else "dn", z=1.0,
                     end=2010.0 if hit else 1990.0)
        elif i < 30:    # fit non-pool: z=0.1, all miss
            _fc_pair(journal, f"f{i}", created=graded - 3600,
                     graded=graded, direction="UP", realized="dn",
                     z=0.1, end=1990.0)
        else:           # holdout pool pairs
            hit = i - 30 < holdout_hits
            _fc_pair(journal, f"f{i}", created=graded - 3600,
                     graded=graded, direction="UP",
                     realized="up" if hit else "dn", z=1.0,
                     end=2010.0 if hit else 1990.0)


def test_holdout_split_and_fit_blindness(tmp_path):
    journal = tmp_path / "j.jsonl"
    _trust_fixture(journal, holdout_hits=0)
    report = audit.run_audit("XAUUSD", journal_path=journal,
                             now=BASE + 100 * 3600)
    assert report["split"]["outcome_n"] == 40
    assert report["split"]["fit_n"] == 30
    assert report["split"]["holdout_n"] == 10
    assert report["split"]["cutoff_ts"] == BASE + 31 * 3600
    ent = report["forecasts"]["1d"]
    fit = ent["fit"]["stats"]
    # hand-computed from the 30 fit pairs ALONE: hits 15/30,
    # pool 15/20 -> lift 0.75 / 0.50 = 1.5 -> trusted
    assert fit["hit_rate"] == 0.5
    assert fit["trust_remeasured"]["pool_n"] == 20
    assert fit["trust_remeasured"]["pool_rate"] == 0.75
    assert fit["trust_remeasured"]["lift"] == 1.5
    assert fit["trust_remeasured"]["state"] == "trusted"
    assert fit["trust_remeasured"]["by_side"]["up"]["trust"] \
        == "trusted"
    # the holdout's own grades see its (bad) content
    assert ent["holdout"]["stats"]["hit_rate"] == 0.0
    # holdout n=10 < 20: no verdict is earned, whatever fit says
    assert ent["verdict"] == "insufficient_data"
    # walk-forward: 4 chronological windows summing to the fit n
    windows = ent["fit"]["windows"]
    assert len(windows) == 4
    assert sum(w["n"] for w in windows) == 30
    assert [w["from_ts"] for w in windows] == \
        sorted(w["from_ts"] for w in windows)


def test_holdout_content_cannot_move_fit(tmp_path):
    ja = tmp_path / "a.jsonl"
    jb = tmp_path / "b.jsonl"
    _trust_fixture(ja, holdout_hits=0)
    _trust_fixture(jb, holdout_hits=10)  # holdout now perfect
    ra = audit.run_audit("XAUUSD", journal_path=ja, journal=False)
    rb = audit.run_audit("XAUUSD", journal_path=jb, journal=False)
    assert ra["forecasts"]["1d"]["fit"] == \
        rb["forecasts"]["1d"]["fit"]
    assert ra["forecasts"]["1d"]["holdout"]["stats"]["hit_rate"] \
        == 0.0
    assert rb["forecasts"]["1d"]["holdout"]["stats"]["hit_rate"] \
        == 1.0


# ── verdicts ─────────────────────────────────────────────────────────

def test_forecast_verdict_rules_exact():
    fit = {"trust_remeasured": {"state": "unproven"}}
    hold = {"n": 25, "coverage_n": 25, "coverage_stated": 0.8,
            "coverage_observed": 0.8, "lift_vs_naive": 2.0,
            "hit_rate": 0.8, "naive_rate": 0.4}
    verdict, _ = audit._forecast_verdict(hold, fit)
    assert verdict == "promotion_ready"
    thin = dict(hold, n=19)
    assert audit._forecast_verdict(thin, fit)[0] \
        == "insufficient_data"
    weak = dict(hold, lift_vs_naive=1.1)
    verdict, reasons = audit._forecast_verdict(weak, fit)
    assert verdict == "not_ready" and "lift" in " ".join(reasons)
    drift = dict(hold, coverage_observed=0.55)
    verdict, reasons = audit._forecast_verdict(drift, fit)
    assert verdict == "not_ready" and "coverage" in " ".join(reasons)
    anti_fit = {"trust_remeasured": {"state": "anti_calibrated"}}
    verdict, reasons = audit._forecast_verdict(hold, anti_fit)
    assert verdict == "not_ready" and "ANTI" in " ".join(reasons)


def test_r_verdict_rules_exact():
    hold = {"r_n": 25, "median_r": 0.75, "hit_rate": 0.6}
    assert audit._r_verdict(hold)[0] == "promotion_ready"
    assert audit._r_verdict(dict(hold, r_n=19))[0] \
        == "insufficient_data"
    assert audit._r_verdict(dict(hold, median_r=-0.1))[0] \
        == "not_ready"
    assert audit._r_verdict(dict(hold, hit_rate=0.4))[0] \
        == "not_ready"
    # discrimination clause: act must beat abstain where both exist
    jhold = {"act": hold,
             "abstain": {"r_n": 25, "median_r": 0.9,
                         "hit_rate": 0.7}}
    verdict, reasons = audit._r_verdict(
        jhold, act=True, abstain_hold=jhold["abstain"])
    assert verdict == "not_ready"
    assert "discrimination" in " ".join(reasons)


def test_forecast_promotion_ready_end_to_end(tmp_path):
    journal = tmp_path / "j.jsonl"
    for i in range(100):
        graded = BASE + (i + 1) * 3600
        if i < 75:  # fit: non-pool, half hit — trust stays unproven
            hit = i % 2 == 0
            _fc_pair(journal, f"f{i}", created=graded - 3600,
                     graded=graded, direction="UP",
                     realized="up" if hit else "dn", z=0.1,
                     end=2010.0 if hit else 1990.0)
            continue
        j = i - 75  # holdout: all hit, coverage 20/25 = stated .8
        direction, realized = [("UP", "up"), ("DOWN", "dn"),
                               ("NEUTRAL", "neutral")][j % 3]
        end = 2200.0 if j >= 20 else 2010.0
        _fc_pair(journal, f"f{i}", created=graded - 3600,
                 graded=graded, direction=direction,
                 realized=realized, z=0.1, end=end)
    report = audit.run_audit("XAUUSD", journal_path=journal,
                             journal=False)
    ent = report["forecasts"]["1d"]
    hold = ent["holdout"]["stats"]
    assert ent["holdout"]["n"] == 25
    assert hold["hit_rate"] == 1.0
    assert hold["coverage_observed"] == 0.8
    assert hold["lift_vs_naive"] == round(1.0 / 0.36, 4)
    assert ent["fit"]["stats"]["trust_remeasured"]["state"] \
        == "unproven"
    assert ent["verdict"] == "promotion_ready"


def test_full_stack_promotion_ready_end_to_end(tmp_path):
    journal = tmp_path / "j.jsonl"
    for i in range(100):
        t = BASE + i * 86400
        sid = f"setup{i}"
        _setup(journal, sid, t0=t - 7200,
               states=("detected", "armed", "triggered"), step=600.0)
        _judgment(journal, sid, judged=t - 300)
        # fit alternates win/loss; the holdout runs 15 x +1R then
        # 10 x -1R -> median +1R, hit rate 0.6
        win = (i % 2 == 0) if i < 75 else ((i - 75) < 15)
        pnl = 10.0 if win else -10.0
        _position(journal, f"pos{i}", sid, opened=t - 240,
                  closed=t, pnl=pnl)
    report = audit.run_audit("XAUUSD", journal_path=journal,
                             journal=False)
    route = report["judgment"]["routes"]["qm_m5_native"]
    assert route["holdout"]["stats"]["act"]["r_n"] == 25
    assert route["holdout"]["stats"]["act"]["median_r"] == 1.0
    assert route["verdict"] == "promotion_ready"
    book = report["playbooks"]["qm_m5_native"]
    assert book["holdout"]["stats"]["r_n"] == 25
    assert book["holdout"]["stats"]["funnel"]["triggered"] == 25
    assert book["verdict"] == "promotion_ready"


# ── playbook funnel grading ──────────────────────────────────────────

def test_playbook_funnel_counts_across_slices(tmp_path):
    journal = tmp_path / "j.jsonl"
    _setup(journal, "s1", t0=BASE, states=("detected", "armed",
                                           "triggered"))
    _setup(journal, "s2", t0=BASE + 10000,
           states=("detected", "armed", "invalidated"))
    _setup(journal, "s3", t0=BASE + 20000, states=("detected",
                                                   "expired"))
    _setup(journal, "s4", t0=BASE + 30000,
           states=("detected", "armed", "triggered"))
    report = audit.run_audit("XAUUSD", journal_path=journal,
                             journal=False)
    ent = report["playbooks"]["qm_m5_native"]
    assert ent["total_n"] == 4
    fit_fn = ent["fit"]["stats"]["funnel"]
    hold_fn = ent["holdout"]["stats"]["funnel"]
    total = {k: fit_fn[k] + hold_fn[k] for k in fit_fn}
    assert total == {"detected": 4, "armed": 3, "triggered": 2,
                     "invalidated": 1, "expired": 1}
    assert ent["fit"]["stats"]["armed_rate"] == round(2 / 3, 4)
    assert ent["fit"]["stats"]["trigger_rate"] == 0.5
    # no position ever closed: lifecycle motion earns no verdict
    assert ent["verdict"] == "insufficient_data"


# ── thin journals read honestly; the audit journals itself ───────────

def test_thin_and_empty_journals(tmp_path):
    journal = tmp_path / "j.jsonl"
    report = audit.run_audit("XAUUSD", journal_path=journal)
    assert report["split"]["outcome_n"] == 0
    assert "no graded outcomes" in audit.render_audit(report)
    events = trading_journal.read_events(limit=0, path=journal,
                                         event="trade_audit")
    assert len(events) == 1 and events[0]["verdicts"] == {
        "forecasts": {}, "judgment_routes": {}, "playbooks": {}}

    for i in range(10):
        graded = BASE + (i + 1) * 3600
        _fc_pair(journal, f"f{i}", created=graded - 3600,
                 graded=graded, realized="up" if i < 7 else "dn")
    before = len(trading_journal.read_events(limit=0, path=journal))
    report = audit.run_audit("XAUUSD", journal_path=journal)
    after = trading_journal.read_events(limit=0, path=journal)
    assert len(after) == before + 1  # exactly one audit event added
    assert report["split"]["outcome_n"] == 10
    assert report["split"]["holdout_n"] == 3
    ent = report["forecasts"]["1d"]
    assert ent["verdict"] == "insufficient_data"
    text = audit.render_audit(report)
    assert "insufficient_data" in text
    assert "10 graded outcomes" in text and "fit n=7" in text


def test_symbol_filter(tmp_path):
    journal = tmp_path / "j.jsonl"
    for i in range(6):
        graded = BASE + (i + 1) * 3600
        _fc_pair(journal, f"x{i}", created=graded - 3600,
                 graded=graded, symbol="XAUUSD")
        _fc_pair(journal, f"b{i}", created=graded - 3600,
                 graded=graded, symbol="BTCUSD")
    one = audit.run_audit("XAUUSD", journal_path=journal,
                          journal=False)
    assert one["split"]["outcome_n"] == 6
    both = audit.run_audit(journal_path=journal, journal=False)
    assert both["split"]["outcome_n"] == 12


# ── surfaces: CLI + the trading tool ─────────────────────────────────

def test_cmdline_audit(tmp_path, capsys):
    from nomorals.cmdline.commands.trade import _cmd_trade
    settings = SimpleNamespace(home_path=tmp_path)
    journal = trading_journal.journal_path(settings)
    for i in range(6):
        graded = BASE + (i + 1) * 3600
        _fc_pair(journal, f"f{i}", created=graded - 3600,
                 graded=graded)
    context = SimpleNamespace(settings=settings)
    args = argparse.Namespace(trade_action="audit", symbol="",
                              json=False)
    assert _cmd_trade(args, context) == 0
    out = capsys.readouterr().out
    assert "trade audit" in out and "insufficient_data" in out
    args = argparse.Namespace(trade_action="audit",
                              symbol="XAUUSD", json=True)
    assert _cmd_trade(args, context) == 0
    assert '"kind": "trade_audit"' in capsys.readouterr().out


def test_parser_audit_action():
    from nomorals.cmdline.parser import _parser
    args = _parser().parse_args(["trade", "audit", "XAUUSD"])
    assert args.trade_action == "audit" and args.symbol == "XAUUSD"
    args = _parser().parse_args(["trade", "audit"])
    assert args.trade_action == "audit" and args.symbol == ""


def test_trading_tool_audit(tmp_path):
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
    journal = trading_journal.journal_path(settings)
    for i in range(6):
        graded = BASE + (i + 1) * 3600
        _fc_pair(journal, f"f{i}", created=graded - 3600,
                 graded=graded)
    reg = FakeRegistry(context)
    trading_tool.register(reg)
    out = reg.tools["trading"](action="audit", symbol="XAUUSD")
    assert out["kind"] == "trade_audit"
    assert out["verdicts"]["forecasts"] == {"1d": "insufficient_data"}
    assert "trade audit" in out["rendered"]
