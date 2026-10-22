"""Tests for multi-slot briefings (morning/afternoon/evening/midnight)."""
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
    assert set(mp.BRIEFING_SLOTS) == {"morning", "afternoon", "evening",
                                       "midnight"}
    for slot, spec in mp.BRIEFING_SLOTS.items():
        assert spec["job_name"]
        assert spec["default_time"]
        assert spec["title"]


def test_slot_prefs_defaults():
    ctx = FakeContext()
    assert mp.slot_time(ctx, "morning") == "23:00"
    assert mp.slot_time(ctx, "afternoon") == "13:00"
    assert mp.slot_time(ctx, "evening") == "19:00"
    assert mp.slot_time(ctx, "midnight") == "00:00"
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


def test_ledger_no_repeats_across_slots():
    """F6: cross-slot dedupe is the news delivery ledger — a story a
    slot confirmed delivered leaves the undelivered set for every
    later slot.  (The retired per-day seen-store this test used to
    pin was folded into the ledger at cutover.)"""
    import time
    from types import SimpleNamespace
    from unittest.mock import patch

    from nomorals.agents.news import NewsAgent
    from nomorals.core.ids import new_id
    from nomorals.storage.db import Database

    ws = tempfile.mkdtemp()
    db = Database(os.path.join(ws, "t.db"))
    db.migrate()
    ctx = SimpleNamespace(
        db=db, settings=SimpleNamespace(workspace_dir=ws, pulse=None),
        extras={})
    agent = NewsAgent(ctx)  # ensures delivery columns
    for url, title in (("http://a", "A"), ("http://b", "B")):
        db.execute(
            "INSERT INTO news_items (id, source, title, url, summary, "
            "published, created_at, delivery_state) VALUES (?,?,?,?,?,?,?,?)",
            (new_id(), "src", title, url, "", 0, time.time(), "undelivered"))
    with patch.object(NewsAgent, "poll", lambda self: {"ok": True}):
        fresh = mp._fresh_news_items(ctx)
        assert [i["url"] for i in fresh] == ["http://a", "http://b"] or \
            {i["url"] for i in fresh} == {"http://a", "http://b"}
        agent.mark_delivered(["http://a"], channel="pulse-morning")
        fresh = mp._fresh_news_items(ctx)
    assert [i["url"] for i in fresh] == ["http://b"]


def test_unknown_slot_rejected():
    ctx = FakeContext()
    r = mp.run_slot(ctx, "dawn")
    assert r["ok"] is False
    assert "unknown slot" in r["error"]
    r = mp.ensure_slot_job(ctx, "dawn")
    assert "error" in r


def test_midnight_slot_accepted():
    # midnight is a real fourth slot now, not an unknown one
    assert "midnight" in mp.BRIEFING_SLOTS
    assert mp.slot_time(FakeContext(), "midnight") == "00:00"


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
