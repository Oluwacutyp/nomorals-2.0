"""Tests for the betting desk (nomorals/agents/betdesk/).

The lab tests run on synthetic crash data with KNOWN ground truth:
an iid stream drawn from the standard crash law must come back
NO_EDGE; a stream with a planted time-of-day effect must come back
EDGE_CANDIDATE with a callable time structure; a stream with a
planted streak effect (and no time structure) may flag, but the
signal engine must refuse to call it — a timed alert needs a clock.
"""

import hashlib
import json
import math
import os
import random
import time
from types import SimpleNamespace

import pytest

from nomorals.agents.betdesk import BettingDesk
from nomorals.agents.betdesk import lab
from nomorals.agents.betdesk.collectors import (
    ProviderSpec, collect_provider, import_rounds_file)
from nomorals.agents.betdesk.crash import CrashRound, CrashStore
from nomorals.agents.betdesk.jobs import ensure_betdesk_jobs
from nomorals.agents.betdesk.signals import SignalEngine
from nomorals.agents.betdesk.sources import (
    betting_status, ensure_betting_sources, ensure_provisioning)


# ── synthetic round streams ────────────────────────────────────────────

def _sample_crash(rng: random.Random, edge: float = 0.01) -> float:
    u = max(rng.random(), 1e-9)
    x = (1.0 - edge) / u
    return max(1.00, math.floor(x * 100) / 100.0)


def _stream(n: int = 6000, seed: int = 11, boost_hours: tuple = (),
            boost: float = 0.0, streaky: bool = False,
            start: float = 1_700_000_000.0, gap: float = 47.0,
            provider: str = "synthetic", game: str = "crash"):
    rng = random.Random(seed)
    rounds = []
    prev_success = False
    for i in range(n):
        ts = start + i * gap
        hour = int((ts // 3600) % 24)
        crash = _sample_crash(rng)
        if streaky:
            p = 0.62 if prev_success else 0.38
            crash = 2.5 if rng.random() < p else 1.20
        if hour in boost_hours and rng.random() < boost:
            crash = 2.50
        prev_success = crash >= 2.0
        rounds.append(CrashRound(provider=provider, game=game,
                                 round_id=str(1_000_000 + i), ts=ts,
                                 crash=crash, source="test"))
    return rounds


# ── crash store ────────────────────────────────────────────────────────

def test_store_append_dedupes_and_counts(tmp_path):
    store = CrashStore(str(tmp_path))
    rounds = _stream(50)
    assert store.append(rounds)["added"] == 50
    assert store.append(rounds)["added"] == 0
    assert store.append(rounds[:10])["duplicates"] == 10
    assert store.count() == 50
    assert store.games() == [("synthetic", "crash", 50)]


def test_store_roundtrip_fields(tmp_path):
    store = CrashStore(str(tmp_path))
    store.append([CrashRound("stake", "crash", "r1", ts=123.0, crash=3.25,
                             game_hash="ab", prev_hash="cd")])
    got = store.rounds("stake", "crash")
    assert len(got) == 1 and got[0].crash == 3.25
    assert got[0].game_hash == "ab"


# ── stats primitives ───────────────────────────────────────────────────

def test_chi2_sf_known_values():
    # chi2 with dof=1: sf(3.841) ~= 0.05 ; dof=2: sf(5.991) ~= 0.05
    assert abs(lab.chi2_sf(3.841, 1) - 0.05) < 0.002
    assert abs(lab.chi2_sf(5.991, 2) - 0.05) < 0.002
    assert lab.chi2_sf(0.0, 5) == 1.0


def test_ks_uniform_detects_skew():
    rng = random.Random(3)
    good = [rng.random() for _ in range(500)]
    _, p_good = lab.ks_uniform(good)
    skewed = [rng.random() ** 2 for _ in range(500)]
    _, p_bad = lab.ks_uniform(skewed)
    assert p_good > 0.01
    assert p_bad < 0.001


# ── the lab's verdicts vs ground truth ─────────────────────────────────

def test_lab_insufficient_data():
    v = lab.analyze_game(_stream(120), provider="synthetic", game="crash")
    assert v.verdict == "INSUFFICIENT_DATA"
    assert "300" in v.what_would_change


def test_lab_iid_stream_is_no_edge():
    v = lab.analyze_game(_stream(6000, seed=11),
                         provider="synthetic", game="crash")
    assert v.verdict == "NO_EDGE"
    assert v.callable_structure == ""
    assert v.n_timed == 6000


def test_lab_finds_planted_time_effect():
    v = lab.analyze_game(_stream(6000, seed=5, boost_hours=(3, 4),
                                 boost=0.25),
                         provider="synthetic", game="crash")
    assert v.verdict == "EDGE_CANDIDATE"
    assert "time_of_day" in v.confirmed
    assert v.callable_structure == "time_of_day"
    assert 3 in v.best_hours and 4 in v.best_hours


def test_lab_streak_effect_is_not_callable():
    v = lab.analyze_game(_stream(6000, seed=9, streaky=True),
                         provider="synthetic", game="crash")
    # whatever it flags, a non-time structure must never become callable
    assert v.callable_structure == ""
    assert "time_of_day" not in v.confirmed


# ── provably-fair verification ─────────────────────────────────────────

def test_hash_chain_verifies_and_catches_tamper():
    rounds = []
    for i in range(5):
        gh = f"hash-{i}-secret-material"
        prev = hashlib.sha256(gh.encode()).hexdigest()
        rounds.append(CrashRound("bustabit", "crash", str(i),
                                 game_hash=gh, prev_hash=prev))
    res = lab.verify_hash_chain(rounds)
    assert res["checked"] == 5 and res["ok"] == 5
    rounds[2].prev_hash = "0" * 64
    res = lab.verify_hash_chain(rounds)
    assert res["ok"] == 4 and res["failed"] == 1


def test_bustabit_derivation_deterministic_and_has_bust_branch():
    h = "a" * 64
    assert lab.bustabit_crash_from_hash(h, "salt") == \
        lab.bustabit_crash_from_hash(h, "salt")
    # the instant-bust branch (h % 33 == 0) exists and returns 1.00
    found = None
    for i in range(500):
        salt = f"s{i}"
        import hmac as _h
        mac = _h.new(salt.encode(), h.encode(), hashlib.sha256).hexdigest()
        if int(mac[:13], 16) % 33 == 0:
            found = salt
            break
    assert found is not None
    assert lab.bustabit_crash_from_hash(h, found) == 1.00


# ── collectors ─────────────────────────────────────────────────────────

def test_collect_provider_parses_mapped_json(tmp_path):
    spec = ProviderSpec(
        id="fakebook", display="Fake", game="crash", domain="fake.example",
        history_url="https://fake.example/history?limit={limit}",
        rounds_path="data.rounds",
        field_map=(("round_id", "id"), ("crash", "busted"),
                   ("ts", "ended_at"), ("game_hash", "hash")))
    payload = {"data": {"rounds": [
        {"id": 7, "busted": 2.34, "ended_at": 1700000000, "hash": "x"},
        {"id": 8, "busted": 1.0, "ended_at": 1700000100, "hash": "y"},
        {"id": 9, "busted": None, "ended_at": 1700000200},  # dropped
    ]}}

    def fake_get(url, headers):
        assert "limit=500" in url
        return 200, json.dumps(payload).encode()

    rep = collect_provider(spec, http_get=fake_get)
    assert rep["ok"] and len(rep["rounds"]) == 2
    store = CrashStore(str(tmp_path))
    store.append(rep["rounds"])
    assert store.count() == 2


def test_collect_provider_without_endpoint_is_honest():
    spec = ProviderSpec(id="stake", display="Stake", game="crash")
    rep = collect_provider(spec)
    assert rep["ok"] is False
    assert "no keyless history endpoint" in rep["detail"]


def test_import_csv_rounds(tmp_path):
    path = tmp_path / "hist.csv"
    path.write_text(
        "id,busted_at,time,hash\n"
        "1,2.50,1700000000,aa\n"
        "2,1.00,1700000060,bb\n"
        "3,4.25,1700000120,cc\n", encoding="utf-8")
    store = CrashStore(str(tmp_path / "desk"))
    res = import_rounds_file(store, str(path), "bustabit", "crash")
    assert res["ok"] and res["added"] == 3
    assert store.rounds("bustabit", "crash")[2].crash == 4.25


# ── sources + provisioning ─────────────────────────────────────────────

def test_betting_sources_register_idempotently():
    from nomorals.media.source_registry import SourceRegistry
    reg = SourceRegistry()
    out = ensure_betting_sources(reg)
    assert len(out) == 8
    assert all(r.get("registered") for r in out)
    out2 = ensure_betting_sources(reg)
    assert all(r.get("already_registered") for r in out2)
    st = betting_status(reg)
    assert st["count"] == 8
    domains = {s["domain"] for s in st["sources"]}
    assert {"stake.com", "sportybet.com", "bustabit.com"} <= domains


def test_provisioning_registers_desk_services():
    from nomorals.accounts.provisioning import (
        get_provision_service, validate_service_token)
    ids = ensure_provisioning()
    assert ids == ["odds-api", "football-data"]
    svc = get_provision_service("oddsapi")  # alias resolves
    assert svc is not None and svc.env_keys == ("ODDS_API_KEY",)
    ok = validate_service_token(svc, "k", http_get=lambda u, h: (200, "{}"))
    assert ok.status == "ok"
    dead = validate_service_token(svc, "k", http_get=lambda u, h: (401, ""))
    assert dead.status == "rejected"
    svc2 = get_provision_service("football-data")
    assert svc2.env_keys == ("FOOTBALL_DATA_API_KEY",)
    ok2 = validate_service_token(
        svc2, "k", http_get=lambda u, h: (200, "{}"))
    assert ok2.status == "ok"


# ── signal engine ──────────────────────────────────────────────────────

class _FakeNotifier:
    def __init__(self):
        self.sent = []

    def publish(self, kind, title, body, **kw):
        self.sent.append((kind, title, body))
        return {"sent": True}


def _edge_verdict(now_hour_base: float):
    v = SimpleNamespace(
        provider="synthetic", game="crash", verdict="EDGE_CANDIDATE",
        callable_structure="time_of_day", best_hours=[3, 4],
        hour_rates={3: 0.74, 4: 0.72}, base_success_rate=0.49,
        target=2.0, evidence="planted test verdict")
    return v


def test_signal_engine_calls_with_two_minute_lead(tmp_path):
    notifier = _FakeNotifier()
    eng = SignalEngine(str(tmp_path), notifier=notifier)
    # 02:30 UTC on a known day: next flagged boundary (03:00) is 30m out
    day = 1_700_000_000.0
    now = day - (day % 86400) + 2.5 * 3600
    decision = eng.evaluate(_edge_verdict(now), now=now)
    call = decision["call"]
    assert call is not None, decision["reason"]
    assert call.lead_seconds >= 120
    out = eng.publish(call, deliver=True)
    assert out["delivered"] is True and len(notifier.sent) == 1
    assert "2 minutes" not in out["detail"]  # detail carries notifier res


def test_signal_engine_refuses_non_edge_and_non_time(tmp_path):
    eng = SignalEngine(str(tmp_path), notifier=_FakeNotifier())
    no_edge = SimpleNamespace(
        provider="p", game="g", verdict="NO_EDGE", callable_structure="",
        best_hours=[], hour_rates={}, base_success_rate=0.5, target=2.0,
        evidence="")
    d = eng.evaluate(no_edge, now=1000.0)
    assert d["call"] is None and "NO_EDGE" in d["reason"]
    streak = SimpleNamespace(
        provider="p", game="g", verdict="EDGE_CANDIDATE",
        callable_structure="", best_hours=[], hour_rates={},
        base_success_rate=0.5, target=2.0, evidence="")
    d = eng.evaluate(streak, now=1000.0)
    assert d["call"] is None and "not time-based" in d["reason"]


def test_signal_scoring_grades_calls(tmp_path):
    store = CrashStore(str(tmp_path))
    eng = SignalEngine(str(tmp_path), notifier=_FakeNotifier())
    day = 1_700_000_000.0
    start = day - (day % 86400) + 3 * 3600
    # one round inside the window reaching the target -> hit
    store.append([CrashRound("synthetic", "crash", "w1", ts=start + 60,
                             crash=2.75)])
    decision = eng.evaluate(_edge_verdict(start), now=start - 1800)
    call = decision["call"]
    assert call is not None and call.window_start == start
    eng.publish(call, deliver=False)
    res = eng.score_due(store, now=start + 7200)
    assert res["scored_now"] == 1 and res["hits_now"] == 1


# ── desk orchestration ─────────────────────────────────────────────────

def test_desk_study_cycle_and_library(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMORALS_BETDESK_DIR", str(tmp_path))
    desk = BettingDesk(None, desk_dir=str(tmp_path))
    desk.store.append(_stream(6000, seed=5, boost_hours=(3, 4), boost=0.25))
    verdicts = desk.analyze_all()
    assert verdicts["synthetic/crash"]["verdict"] == "EDGE_CANDIDATE"
    lib = desk.library()
    assert lib["synthetic/crash"]["callable_structure"] == "time_of_day"
    st = desk.status()
    assert st["rounds_total"] == 6000
    assert st["games"][0]["verdict"] == "EDGE_CANDIDATE"
    # signals: a call may or may not be due depending on the wall clock,
    # but the report must always explain itself
    rep = desk.run_signals(deliver=False)
    assert rep["published"] or rep["no_call_reasons"]


def test_desk_import_inbox(tmp_path):
    desk = BettingDesk(None, desk_dir=str(tmp_path))
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "bustabit__crash.csv").write_text(
        "id,busted_at,time\n1,2.5,1700000000\n2,1.0,1700000060\n",
        encoding="utf-8")
    res = desk.import_inbox()
    assert res and res[0]["ok"] and res[0]["added"] == 2
    assert desk.store.count() == 2
    # imported file moved out of the inbox — not re-read next cycle
    assert not (inbox / "bustabit__crash.csv").exists()


# ── jobs + tool wiring ─────────────────────────────────────────────────

class _FakeScheduler:
    def __init__(self):
        self.jobs = []

    def list_jobs(self):
        return [{"name": j["name"]} for j in self.jobs]

    def add(self, name, spec, kind, payload, routine=False):
        self.jobs.append({"name": name, "spec": spec, "kind": kind,
                          "payload": payload, "routine": routine})
        return {"id": len(self.jobs)}


def test_ensure_betdesk_jobs_idempotent():
    sched = _FakeScheduler()
    out = ensure_betdesk_jobs(sched)
    assert len(out) == 2 and all(o.get("scheduled") for o in out)
    out2 = ensure_betdesk_jobs(sched)
    assert all(o.get("already_scheduled") for o in out2)
    tools = {j["payload"]["tool"] for j in sched.jobs}
    assert tools == {"betdesk"}
    assert all(j["routine"] for j in sched.jobs)


def test_tool_registered_in_agent_modules():
    from nomorals.tools.agents import AGENT_TOOL_MODULES
    assert "betdesk" in AGENT_TOOL_MODULES


def test_tool_binds_job_payloads(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMORALS_BETDESK_DIR", str(tmp_path))
    from nomorals.agents import betdesk as betdesk_pkg

    captured = {}

    class FakeRegistry:
        context = SimpleNamespace()

        def register(self, name, description="", capability=None,
                     parameters=None):
            def deco(fn):
                captured[name] = fn
                return fn
            return deco

    betdesk_pkg.register(FakeRegistry())
    fn = captured["betdesk"]
    import inspect
    inspect.signature(fn).bind(action="study")
    inspect.signature(fn).bind(action="signals", deliver=True)
    out = fn(action="status")
    assert out["rounds_total"] == 0
    assert fn(action="bogus")["error"].startswith("unknown action")
