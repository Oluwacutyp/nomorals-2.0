"""Tests for multi-slot briefings (morning/afternoon/evening)."""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.agents import morning_pulse as mp


class FakeSettings:
    def __init__(self, ws):
        self.workspace_dir = ws
        self.pulse = None


class FakeContext:
    def __init__(self, ws=None):
        self.settings = FakeSettings(ws or tempfile.mkdtemp())
        self.router = None
        self.db = None


def test_slots_defined():
    assert set(mp.BRIEFING_SLOTS) == {"morning", "afternoon", "evening"}
    for slot, spec in mp.BRIEFING_SLOTS.items():
        assert spec["job_name"]
        assert spec["default_time"]
        assert spec["title"]


def test_slot_prefs_defaults():
    ctx = FakeContext()
    assert mp.slot_time(ctx, "morning") == "23:00"
    assert mp.slot_time(ctx, "afternoon") == "13:00"
    assert mp.slot_time(ctx, "evening") == "19:00"
    assert mp.slot_enabled(ctx, "afternoon") is True
    assert mp.slot_voice(ctx, "morning") is True
    assert mp.slot_voice(ctx, "afternoon") is False


def test_slot_prefs_override():
    ctx = FakeContext()
    ws = ctx.settings.workspace_dir
    with open(os.path.join(ws, "pulse_prefs.json"), "w") as f:
        json.dump({"afternoon_time": "14:30", "evening_enabled": False,
                   "afternoon_voice": True}, f)
    assert mp.slot_time(ctx, "afternoon") == "14:30"
    assert mp.slot_enabled(ctx, "evening") is False
    assert mp.slot_voice(ctx, "afternoon") is True
    # morning untouched
    assert mp.slot_time(ctx, "morning") == "23:00"


def test_seen_store_no_repeats():
    ctx = FakeContext()
    items = [{"url": "http://a", "title": "A"},
             {"url": "http://b", "title": "B"}]
    # nothing seen yet
    fresh = mp._filter_unseen(ctx, items, lambda it: it["url"])
    assert len(fresh) == 2
    # mark one seen
    mp._mark_seen(ctx, ["http://a"], "morning")
    fresh = mp._filter_unseen(ctx, items, lambda it: it["url"])
    assert len(fresh) == 1
    assert fresh[0]["url"] == "http://b"


def test_unknown_slot_rejected():
    ctx = FakeContext()
    r = mp.run_slot(ctx, "midnight")
    assert r["ok"] is False
    assert "unknown slot" in r["error"]
    r = mp.ensure_slot_job(ctx, "midnight")
    assert "error" in r


def test_disabled_slot_removes_job():
    # ensure_slot_job with enabled=False should not raise even without a DB
    ctx = FakeContext()
    ws = ctx.settings.workspace_dir
    with open(os.path.join(ws, "pulse_prefs.json"), "w") as f:
        json.dump({"afternoon_enabled": False}, f)
    r = mp.ensure_slot_job(ctx, "afternoon")
    assert r.get("disabled") is True or "error" in r


def test_legacy_prefs_still_work():
    """Old flat time/enabled keys map to the morning slot."""
    ctx = FakeContext()
    assert mp.pulse_time(ctx) == "23:00"
    assert mp.pulse_enabled(ctx) is True
    assert mp.pulse_timezone(ctx) == "America/Denver"


def test_slot_marker_roundtrip():
    ctx = FakeContext()
    assert mp._read_slot_marker(ctx, "evening") == {}
    mp._write_slot_marker(ctx, "evening",
                          {"stages": ["compose", "deliver"],
                           "delivered_text": True, "elapsed_s": 1.2})
    m = mp._read_slot_marker(ctx, "evening")
    assert m["slot"] == "evening"
    assert m["delivered_text"] is True
