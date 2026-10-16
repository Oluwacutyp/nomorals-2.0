"""Tests for the six new console modules: cockpit, termux, costmeter,
recording, discovery, alerts. No stubs: every test exercises real
behavior through the public API.
"""

import json

import pytest

from nomorals.console.alerts import AlertPolicy, SeverityRule
from nomorals.console.cockpit import Cockpit
from nomorals.console.costmeter import CostMeter
from nomorals.console.discovery import CommandRegistry
from nomorals.console.recording import SessionRecorder
from nomorals.console.termux import TermuxScreen, render as termux_render


# ── cockpit ──────────────────────────────────────────────────────────────

def _missions():
    return [
        {"id": "m1", "name": "morning pulse", "status": "running",
         "goal": "brief the owner before 6am WAT", "progress": 0.5},
        {"id": "m2", "name": "paper loop", "status": "paused",
         "reason": "watch XAUUSD for QM setups"},
    ]


def _organs():
    return [
        {"name": "brain", "enabled": True, "status": "ok"},
        {"name": "browser", "enabled": False, "status": "idle"},
    ]


def test_cockpit_pause_kill_call_callbacks():
    calls: list[tuple[str, str]] = []
    cockpit = Cockpit(
        missions=_missions,
        organs=_organs,
        on_pause_mission=lambda mid: calls.append(("pause", mid)),
        on_kill_mission=lambda mid: calls.append(("kill", mid)),
    )
    cockpit.pause_mission("m1")
    cockpit.kill_mission("m2")
    assert calls == [("pause", "m1"), ("kill", "m2")]


def test_cockpit_toggle_organ_calls_callback_and_flips():
    calls: list[tuple[str, bool]] = []
    cockpit = Cockpit(
        organs=_organs,
        on_toggle_organ=lambda name, enabled: calls.append((name, enabled)),
    )
    cockpit.toggle_organ("brain")  # flip: on -> off
    cockpit.toggle_organ("browser", True)
    assert calls == [("brain", False), ("browser", True)]


def test_cockpit_why_and_render():
    cockpit = Cockpit(
        missions=_missions, organs=_organs,
        browser_status={"running": True, "pages": 2, "last_action": "scrolled"},
        ledger=[{"ts": "10:00", "action": "posted", "detail": "morning brief"}],
        budget={"used": 30, "limit": 100, "unit": "NGN"},
    )
    assert "brief the owner" in cockpit.why("m1")
    assert "QM setups" in cockpit.why("m2")
    assert "unknown mission" in cockpit.why("nope")
    out = cockpit.render()
    assert "morning pulse" in out and "brief the owner" in out
    assert "browser daemon" in out and "running" in out
    assert "posted" in out and "morning brief" in out
    assert "30/100 NGN" in out


def test_cockpit_missing_callbacks_raise_helpfully():
    cockpit = Cockpit(missions=_missions)
    with pytest.raises(RuntimeError):
        cockpit.pause_mission("m1")


# ── termux ───────────────────────────────────────────────────────────────

def test_termux_render_has_no_alt_screen_escapes_and_max_80_cols():
    state = {
        "title": "DEVON",
        "missions": [{"name": "x" * 120, "status": "running"}],
        "organs": [{"name": "brain", "enabled": True}],
        "ledger": [{"ts": 1760000000.0, "action": "did", "detail": "y" * 100}],
        "budget": {"used": 5, "limit": 10},
        "history": [1, 2, 3, 2, 5],
    }
    out = termux_render(state)
    assert "\x1b[?1049" not in out  # no alt-screen
    assert "\x1b[" not in out  # no ANSI escapes at all
    assert "\x1b[2J" not in out
    assert all(len(line) <= 80 for line in out.split("\n"))


def test_termux_minimal_redraw_only_changed_lines():
    screen = TermuxScreen(clock=lambda: 0.0)
    state = {"missions": [{"name": "a", "status": "running"}],
             "budget": {"used": 1, "limit": 10}}
    screen.render(state)
    changed = screen.render_changed(state)
    assert changed == []  # nothing changed -> nothing to repaint
    state2 = {"missions": [{"name": "a", "status": "done"}],
              "budget": {"used": 1, "limit": 10}}
    changed2 = screen.render_changed(state2)
    assert changed2 and all("done" in line for _, line in changed2)
    assert len(changed2) < len(screen.render(state2).split("\n"))


def test_termux_battery_low_slows_refresh_and_drops_history():
    now = [0.0]
    screen = TermuxScreen(clock=lambda: now[0], battery_low=True)
    state = {"missions": [], "history": [1, 2, 3]}
    out = screen.render(state)
    assert "activity" not in out  # history dropped on low battery
    assert screen.min_interval == 5.0
    now[0] = 2.0
    assert screen.should_render() is False  # throttled: 2s < 5s
    now[0] = 6.0
    assert screen.should_render() is True


# ── costmeter ────────────────────────────────────────────────────────────

def test_cost_attributed_to_organ_sums_match():
    now = [1000.0]
    meter = CostMeter(clock=lambda: now[0])
    meter.record_cost("brain", 100, 10.0)
    meter.record_cost("brain", 50, 5.0)
    meter.record_cost("browser", 200, 4.0)
    per = meter.per_organ()
    assert per["brain"].tokens == 150 and per["brain"].ngn == 15.0
    assert per["browser"].tokens == 200 and per["browser"].ngn == 4.0
    assert sum(r.tokens for r in per.values()) == meter.total_tokens() == 350
    assert sum(r.ngn for r in per.values()) == meter.total_ngn() == pytest.approx(19.0)
    assert "brain" in meter.render() and "₦19.00" in meter.render()


def test_cost_total_per_hour_sliding_window():
    now = [0.0]
    meter = CostMeter(clock=lambda: now[0])
    meter.record_cost("brain", 10, 60.0)  # ₦60 in the first instant
    now[0] = 1800.0  # 30 min later, still inside the 60-min window
    assert meter.total_per_hour() == pytest.approx(60.0)  # 60 NGN / 1h window
    now[0] = 7200.0  # 2h later: entry aged out of the window
    assert meter.total_per_hour() == pytest.approx(0.0)
    assert meter.per_hour("brain") == pytest.approx(0.0)


def test_cost_rejects_negative():
    meter = CostMeter()
    with pytest.raises(ValueError):
        meter.record_cost("brain", -1, 1.0)


# ── recording ────────────────────────────────────────────────────────────

def test_recording_replay_order_and_save_load(tmp_path):
    rec = SessionRecorder(clock=lambda: 1.0)
    rec.record_event("command", {"line": "status"})
    rec.record_event("mission", {"id": "m1", "action": "paused"})
    rec.record_event("alert", {"msg": "hi"})
    replayed = list(rec.replay())
    assert [e["kind"] for e in replayed] == ["command", "mission", "alert"]
    assert [e["seq"] for e in replayed] == [0, 1, 2]
    path = tmp_path / "session.json"
    rec.save(path)
    assert json.loads(path.read_text())  # valid JSON on disk
    loaded = SessionRecorder.load(path)
    assert [e["kind"] for e in loaded.replay()] == ["command", "mission", "alert"]
    assert loaded.record_event("x")["seq"] == 3  # seq continues after load


# ── discovery ────────────────────────────────────────────────────────────

def test_discovery_fuzzy_search_and_registry_generated_help():
    reg = CommandRegistry()
    assert reg.help_all() == "no commands registered"  # starts empty: nothing hardcoded
    reg.register_command("dashboard", "dashboard [--watch] — full status")
    reg.register_command("status", "status — one-line compact status")
    reg.register_command("cockpit", "cockpit — what is she doing and why")
    reg.register_command("cost", "cost — live token/NGN meter")
    # Fuzzy: typo still finds the command.
    assert reg.search("dashbord")[:1] == ["dashboard"]
    assert "cockpit" in reg.search("cockpt")
    # Usage ranking: heavily-used near-match wins.
    reg.register_command("dash", "dash — alias", usage_count=50)
    reg.record_use("dashboard")
    assert reg.search("dash")[0] == "dash"
    # /why-style help is generated from the registry, not a hardcoded list.
    reg.register_command("whytest", "whytest — generated help proof")
    assert "generated help proof" in (reg.help_for("whytest") or "")
    assert reg.help_for("nope") is None
    assert "whytest — generated help proof" in reg.help_all()
    assert sum(1 for ln in reg.help_all().split("\n") if "whytest" in ln) == 1


def test_discovery_load_from_console_table():
    reg = CommandRegistry()
    count = reg.load_from_console()
    assert count > 0  # derived from the live console table, not hardcoded
    assert "dashboard" in reg
    assert reg.help_for("dashboard") is not None


# ── alerts ───────────────────────────────────────────────────────────────

def test_alert_throttle_dedupes():
    now = [1000.0]
    policy = AlertPolicy(
        {"critical": SeverityRule(channels=("telegram", "console"), throttle_s=300.0)},
        clock=lambda: now[0],
    )
    delivered: list[tuple[str, str, str]] = []

    def sender(channel, severity, msg):
        delivered.append((channel, severity, msg))
        return True

    assert policy.send_alert("critical", "budget exhausted", sender) is True
    assert len(delivered) == 2  # both channels on first send
    assert policy.send_alert("critical", "budget exhausted", sender) is False
    assert len(delivered) == 2  # deduped inside the throttle window
    assert policy.suppressed_count == 1
    now[0] += 301.0  # past the throttle window
    assert policy.send_alert("critical", "budget exhausted", sender) is True
    assert len(delivered) == 4
    # A different message is not deduped against the first.
    assert policy.send_alert("critical", "something else", sender) is True


def test_alert_one_policy_many_channels_default_routing():
    policy = AlertPolicy()
    seen: list[str] = []
    policy.send_alert("warning", "disk warm", lambda ch, s, m: seen.append(ch))
    assert seen == ["console"]  # warning -> console only
    policy.send_alert("critical", "down", lambda ch, s, m: seen.append(ch))
    assert seen[-2:] == ["telegram", "console"]
    assert policy.sent_count == 2
