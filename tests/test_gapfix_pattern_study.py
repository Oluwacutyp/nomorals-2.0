"""Pattern study — plain-words rules, backtests, live watches.

Every history-dependent check uses LABELED fixture candles (logic
proof, not history): on this box the real XAUUSD ledger is empty and
the production fetch is SSRF-blocked, so the organ honestly reports
UNPROVEN there — see prove.py for the real-history demonstration
(Yahoo GC=F gold futures, labeled as the proxy).
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS

import pytest

from nomorals.agents.marketdesk.patterns import (
    BUILTIN_ID,
    BUILTIN_RULE,
    Pattern,
    PatternStore,
    backtest,
    check_watches,
    list_patterns,
    parse_rule,
    pattern_status,
    remove_pattern,
    render_check,
    render_study,
    set_watching,
    study_rule,
)
from nomorals.agents.marketdesk.store import Candle, CandleStore

MONDAY = datetime(2026, 10, 12, 12, tzinfo=UTC)


def _bar(day_offset: int, high: float, low: float,
         open_: float | None = None, close: float | None = None) -> Candle:
    dt = MONDAY + timedelta(days=day_offset)
    return Candle(symbol="XAUUSD", timeframe="1d",
                  ts=dt.timestamp(),
                  open=open_ if open_ is not None else low + 1.0,
                  high=high, low=low,
                  close=close if close is not None else high - 1.0,
                  volume=0.0, source="fixture")


def _fixture_bars() -> list[Candle]:
    """Labeled fixture weeks (Mon 2026-10-12 is a Monday — verified).

    W1 (10-12): Thu high 4000, Fri high 3990, Fri low 3995; Mon 10-19
        low 3980 -> HIT.  W2 (10-19): Fri high ABOVE Thu high ->
        evaluated, not triggered.  W3 (10-26): Fri high below Thu high,
        Mon 11-02 low 4035 above Fri low 4015 -> MISS.  W4 (11-02): Fri
        high below Thu high, no Monday bar yet -> pending.  W5 (11-09):
        Tue-Wed only (Monday a holiday, no Thursday) -> skipped, never
        evaluated.
    """
    bars = [
        # W1: Mon 10-12 .. Fri 10-16
        _bar(0, 3980, 3960), _bar(1, 3985, 3965), _bar(2, 3990, 3970),
        _bar(3, 4000, 3975), _bar(4, 3990, 3995),
        # W2: Mon 10-19 .. Fri 10-23 (Mon 10-19 is also W1's outcome day)
        _bar(7, 3980, 3970), _bar(8, 3995, 3975), _bar(9, 4005, 3980),
        _bar(10, 4010, 3985), _bar(11, 4020, 3990),
        # W3: Mon 10-26 .. Fri 10-30
        _bar(14, 4025, 4000), _bar(15, 4028, 4005), _bar(16, 4030, 4010),
        _bar(17, 4030, 4012), _bar(18, 4020, 4015),
        # W4: Mon 11-02 .. Fri 11-06
        _bar(21, 4035, 4020), _bar(22, 4038, 4022), _bar(23, 4039, 4023),
        _bar(24, 4040, 4024), _bar(25, 4030, 4025),
        # W5: Tue 11-10 + Wed 11-11 only (Monday was a holiday, no
        # Thursday bar) -> skipped, never evaluated; and W4's outcome
        # Monday (11-09) is absent, so W4 stays pending.
        _bar(29, 4046, 4031), _bar(30, 4047, 4032),
    ]
    # W1 outcome Monday (10-19) low 3980: _bar(7) already covers it.
    return bars


def _builtin() -> Pattern:
    parsed = parse_rule(BUILTIN_RULE)
    assert parsed["ok"]
    return Pattern(
        id=BUILTIN_ID, name="builtin", symbol="XAUUSD", market="forex",
        condition=parsed["condition"], expectation=parsed["expectation"],
        rule_text=BUILTIN_RULE, created_at=0.0, watching=True)


def _settings(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return NS(home_path=str(home)), home


def _stores(tmp_path):
    settings, home = _settings(tmp_path)
    pstore = PatternStore(settings=settings)
    cstore = CandleStore(desk_dir=str(tmp_path / "desk"))
    return settings, home, pstore, cstore


# ── parser ───────────────────────────────────────────────────────────

def test_parse_builtin_rule():
    parsed = parse_rule(BUILTIN_RULE)
    assert parsed["ok"]
    assert parsed["symbol"] == "XAUUSD"
    assert parsed["market"] == "forex"
    assert parsed["condition"] == {
        "weekday": 4, "field": "high", "op": "<",
        "other_weekday": 3, "other_field": "high"}
    assert parsed["expectation"] == {
        "weekday": 0, "field": "low", "op": "<=",
        "other_weekday": 4, "other_field": "low"}


def test_parse_video_sentence_word_for_word():
    video = (
        "If the high during the Friday is not as high as the high made "
        "on Thursday the day before, then the odds are overwhelmingly "
        "high that whatever low you made on Friday will be visited on "
        "Monday.")
    parsed = parse_rule(video)
    assert parsed["ok"]
    builtin = parse_rule(BUILTIN_RULE)
    assert parsed["condition"] == builtin["condition"]
    assert parsed["expectation"] == builtin["expectation"]


def test_parse_compact_variants():
    parsed = parse_rule(
        "Fri high > Thu high -> Mon high >= Fri high for BTCUSD")
    assert parsed["ok"]
    assert parsed["symbol"] == "BTCUSD"
    assert parsed["condition"]["op"] == ">"
    assert parsed["expectation"]["op"] == ">="
    parsed = parse_rule(
        "Tuesday close < Monday close => Wednesday open > Tuesday open")
    assert parsed["ok"]
    assert parsed["symbol"] == "XAUUSD"  # default symbol
    assert parsed["condition"]["weekday"] == 1
    assert parsed["expectation"]["weekday"] == 2


def test_parse_refusals_name_the_syntax():
    for bad in ["", "gold goes up on fridays", "Friday high",
                "Monday low <= Friday low"]:
        out = parse_rule(bad)
        assert not out["ok"]
        assert "rule syntax" in out["error"]


# ── backtest on labeled fixtures ─────────────────────────────────────

def test_backtest_counts_hits_misses_pending_and_skips():
    score = backtest(_fixture_bars(), _builtin())
    assert score["evaluated"] == 4  # W1..W4 (W5 has no Thursday)
    assert score["triggered"] == 3  # W1, W3, W4
    assert score["resolved"] == 2  # W1 hit, W3 miss
    assert score["hits"] == 1
    assert score["pending"] == 1  # W4, no Monday bar yet
    assert score["hit_rate"] == pytest.approx(0.5)
    assert score["span_start"] == "2026-10-12"
    assert score["span_end"] == "2026-11-02"
    assert score["verdict"] == "proven"


def test_backtest_empty_history_is_unproven():
    score = backtest([], _builtin())
    assert score["verdict"] == "unproven"
    assert score["hit_rate"] is None
    assert score["resolved"] == 0


# ── organ operations ─────────────────────────────────────────────────

def test_study_stores_pattern_and_score(tmp_path):
    settings, _home, pstore, cstore = _stores(tmp_path)
    cstore.append(_fixture_bars())
    out = study_rule(BUILTIN_RULE, settings=settings, store=pstore,
                     candle_store=cstore)
    assert out["ok"]
    assert out["score"]["hits"] == 1
    assert out["score"]["resolved"] == 2
    assert out["score"]["source"] == "ledger"
    assert pstore.get(BUILTIN_ID) is not None


def test_study_unparseable_is_a_refusal_not_a_guess(tmp_path):
    settings, _home, pstore, cstore = _stores(tmp_path)
    out = study_rule("gold goes up on fridays", settings=settings,
                     store=pstore, candle_store=cstore)
    assert not out["ok"]
    assert "rule syntax" in out["error"]


def test_study_with_no_history_is_unproven_with_reason(tmp_path):
    settings, _home, pstore, cstore = _stores(tmp_path)
    out = study_rule(
        BUILTIN_RULE, settings=settings, store=pstore,
        candle_store=cstore,
        fetch=lambda *a: (_ for _ in ()).throw(
            RuntimeError("feeds down")))
    assert out["ok"]
    assert out["score"]["verdict"] == "unproven"
    assert out["score"]["source"] == "failed"
    assert "feeds down" in (out["score"].get("fetch_error") or "")
    assert render_study(out).startswith("pattern fri_thu_low_revisit")
    assert "UNPROVEN" in render_study(out)


def test_list_seeds_builtin_and_watch_toggles(tmp_path):
    settings, _home, pstore, _cstore = _stores(tmp_path)
    listed = list_patterns(settings=settings, store=pstore)
    assert any(p["id"] == BUILTIN_ID for p in listed["patterns"])
    out = set_watching(BUILTIN_ID, False, settings=settings, store=pstore)
    assert out["ok"] and out["pattern"]["watching"] is False
    out = set_watching(BUILTIN_ID, True, settings=settings, store=pstore)
    assert out["pattern"]["watching"] is True
    out = set_watching("nope", True, settings=settings, store=pstore)
    assert not out["ok"]


def test_remove_refuses_builtin_but_forgets_a_custom_rule(tmp_path):
    settings, _home, pstore, cstore = _stores(tmp_path)
    study_rule("Tuesday close < Monday close => Wednesday open > "
               "Tuesday open", settings=settings, store=pstore,
               candle_store=cstore,
               fetch=lambda *a: (_ for _ in ()).throw(
                   RuntimeError("no feed")))
    custom = [p for p in pstore.patterns() if p.id != BUILTIN_ID]
    assert len(custom) == 1
    assert not remove_pattern(BUILTIN_ID, settings=settings,
                              store=pstore)["ok"]
    out = remove_pattern(custom[0].id, settings=settings, store=pstore)
    assert out["ok"] and out["removed"] == custom[0].id
    assert pstore.get(custom[0].id) is None


# ── live watch ───────────────────────────────────────────────────────

def test_watch_triggers_then_resolves_on_monday_bar(tmp_path):
    settings, _home, pstore, cstore = _stores(tmp_path)
    cstore.append(_fixture_bars())
    first = check_watches(settings=settings, store=pstore,
                          candle_store=cstore)
    assert first["ok"] and not first["errors"]
    triggers = [n for n in first["notices"] if n["kind"] == "trigger"]
    assert len(triggers) == 1
    assert triggers[0]["week"] == "2026-11-02"
    assert triggers[0]["expectation"] == "Monday low <= Friday low"
    assert triggers[0]["hit_rate"] == pytest.approx(0.5)
    assert len(pstore.pending_triggers(BUILTIN_ID)) == 1
    # Monday 11-09 lands below Friday's low -> HIT.
    cstore.append([_bar(28, 4045, 4020)])
    second = check_watches(settings=settings, store=pstore,
                           candle_store=cstore)
    outcomes = [n for n in second["notices"] if n["kind"] == "outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["hit"] is True
    assert outcomes[0]["week"] == "2026-11-02"
    assert not pstore.pending_triggers(BUILTIN_ID)
    st = pattern_status(BUILTIN_ID, settings=settings, store=pstore,
                        candle_store=cstore)
    assert st["score"]["resolved"] == 3
    assert st["score"]["hits"] == 2
    assert "HIT" in render_check(second)


def test_watch_resolves_trigger_and_outcome_in_one_sweep(tmp_path):
    settings, _home, pstore, cstore = _stores(tmp_path)
    bars = _fixture_bars() + [_bar(28, 4045, 4020)]
    cstore.append(bars)
    out = check_watches(settings=settings, store=pstore,
                        candle_store=cstore)
    kinds = [n["kind"] for n in out["notices"]]
    assert kinds == ["trigger", "outcome"]
    assert out["notices"][1]["hit"] is True


def test_watch_is_idempotent_per_week(tmp_path):
    settings, _home, pstore, cstore = _stores(tmp_path)
    cstore.append(_fixture_bars())
    check_watches(settings=settings, store=pstore, candle_store=cstore)
    second = check_watches(settings=settings, store=pstore,
                           candle_store=cstore)
    assert second["notices"] == []
    assert len(pstore.events(BUILTIN_ID, kind="pattern_trigger")) == 1


# ── portable state: a fresh home resumes the watch ───────────────────

def test_state_round_trips_to_a_fresh_home_and_watch_resumes(tmp_path):
    settings, home, pstore, cstore = _stores(tmp_path)
    cstore.append(_fixture_bars())
    study_rule(BUILTIN_RULE, settings=settings, store=pstore,
               candle_store=cstore)
    check_watches(settings=settings, store=pstore, candle_store=cstore)
    assert len(pstore.pending_triggers(BUILTIN_ID)) == 1
    # Move the home: copy the patterns dir to a fresh home, open a
    # brand-new store there. No host path may survive the move.
    home_b = tmp_path / "homeB"
    shutil.copytree(home / "patterns", home_b / "patterns")
    raw = (home_b / "patterns" / "patterns.json").read_text()
    assert str(home) not in raw
    settings_b = NS(home_path=str(home_b))
    pstore_b = PatternStore(settings=settings_b)
    reloaded = pstore_b.get(BUILTIN_ID)
    assert reloaded is not None
    assert reloaded.score["hits"] == 1
    assert reloaded.score["resolved"] == 2
    assert len(pstore_b.pending_triggers(BUILTIN_ID)) == 1
    # The watch resumes on the fresh home: Monday's bar resolves it.
    cstore_b = CandleStore(desk_dir=str(tmp_path / "deskB"))
    cstore_b.append(_fixture_bars() + [_bar(28, 4045, 4020)])
    out = check_watches(settings=settings_b, store=pstore_b,
                        candle_store=cstore_b)
    outcomes = [n for n in out["notices"] if n["kind"] == "outcome"]
    assert len(outcomes) == 1 and outcomes[0]["hit"] is True
    assert not pstore_b.pending_triggers(BUILTIN_ID)


# ── chat + CLI + tool wiring ─────────────────────────────────────────

def test_chat_control_pattern_study_list_check(tmp_path, monkeypatch):
    from nomorals.agents.partner.runtime_games import RuntimeGamesMixin

    settings, _home, _pstore, cstore = _stores(tmp_path)
    cstore.append(_fixture_bars())
    rows = [{
        "symbol": "XAUUSD", "timeframe": "1d", "ts": b.ts,
        "open": b.open, "high": b.high, "low": b.low,
        "close": b.close, "volume": 0.0} for b in _fixture_bars()]
    monkeypatch.setattr(
        "nomorals.agents.marketdesk.patterns.fetch_rows",
        lambda *a: rows)
    runtime = object.__new__(RuntimeGamesMixin)
    runtime.context = NS(settings=settings)
    out = runtime._control_pattern(
        "study Friday high < Thursday high => Monday low <= Friday low "
        "for XAUUSD", "test-chat")
    assert "fri_thu_low_revisit" in out
    assert "50%" in out or "1/2" in out
    out = runtime._control_pattern("list", "test-chat")
    assert BUILTIN_ID in out
    out = runtime._control_pattern("check", "test-chat")
    assert "TRIGGER" in out and "2026-11-02" in out
    out = runtime._control_pattern("", "test-chat")
    assert "/pattern study" in out
    out = runtime._control_pattern("frobnicate", "test-chat")
    assert "unknown /pattern verb" in out


def test_cli_pattern_study_and_list(tmp_path, monkeypatch, capsys):
    from nomorals.cmdline.commands.pattern import _cmd_pattern

    settings, _home, _pstore, cstore = _stores(tmp_path)
    cstore.append(_fixture_bars())
    rows = [{
        "symbol": "XAUUSD", "timeframe": "1d", "ts": b.ts,
        "open": b.open, "high": b.high, "low": b.low,
        "close": b.close, "volume": 0.0} for b in _fixture_bars()]
    monkeypatch.setattr(
        "nomorals.agents.marketdesk.patterns.fetch_rows",
        lambda *a: rows)
    ctx = NS(settings=settings)
    args = NS(pattern_action="study",
              rule=["Friday", "high", "<", "Thursday", "high", "=>",
                    "Monday", "low", "<=", "Friday", "low", "for",
                    "XAUUSD"],
              json=False)
    assert _cmd_pattern(args, ctx) == 0
    assert BUILTIN_ID in capsys.readouterr().out
    args = NS(pattern_action="list", json=False)
    assert _cmd_pattern(args, ctx) == 0
    assert BUILTIN_ID in capsys.readouterr().out
    args = NS(pattern_action="study", rule=["gold", "goes", "up"],
              json=False)
    assert _cmd_pattern(args, ctx) == 1


def test_tool_patterns_action(tmp_path):
    from types import SimpleNamespace as NS2

    from nomorals.tools.registry import ToolRegistry

    settings, _home, _pstore, _cstore = _stores(tmp_path)
    registry = ToolRegistry(NS2(db=None, settings=settings))
    registry.register_builtins()
    assert "marketdesk" in registry._tools
    out = registry.call("marketdesk", action="patterns", verb="list")
    assert out.ok, getattr(out.error, "message", out.error)
    assert any(p["id"] == BUILTIN_ID
               for p in out.value["patterns"])


def test_marketdesk_jobs_include_pattern_watch():
    from nomorals.agents.marketdesk.jobs import MARKETDESK_PATTERN_WATCH_JOB, ensure_marketdesk_jobs

    class _FakeScheduler:
        def __init__(self):
            self.jobs = []

        def list_jobs(self):
            return [{"name": j["name"]} for j in self.jobs]

        def add(self, name, spec, kind, payload, routine=False):
            self.jobs.append({"name": name, "spec": spec, "kind": kind,
                              "payload": payload, "routine": routine})
            return {"id": len(self.jobs)}

    sched = _FakeScheduler()
    out = ensure_marketdesk_jobs(sched)
    assert len(out) == 3  # study + signals + pattern watch
    watch = [j for j in sched.jobs
             if j["name"] == MARKETDESK_PATTERN_WATCH_JOB]
    assert len(watch) == 1
    assert watch[0]["payload"] == {
        "tool": "marketdesk",
        "args": {"action": "patterns", "verb": "check"}}
    assert watch[0]["routine"] is True
    out2 = ensure_marketdesk_jobs(sched)
    assert all(o.get("already_scheduled") for o in out2)
