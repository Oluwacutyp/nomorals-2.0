"""Tail-80 failure pins — nomorals/partner/chat_profile.py.

Five sites are EXCLUSIONS: get_profile's fall-through to the
documented no-profile value {} (99), the topic-texts guard
retaining old topics (223), the vibe guard retaining the old
vibe (237), the activity guard retaining the old activity
values (247), and refresh_profile's own designed outer
fallback — return the stored profile, write nothing (259).

Site 193 is a CONVERT.  The participants aggregate query is
the profile's data source: its failure used to be swallowed
into participants={} / total=0, after which refresh rebuilt
and PERSISTED a profile byte-identical to a brand-new chat's
— participants wiped, message_count zeroed, the owner note
deleted — over the known-good stored profile the brain
renders into prompt context.  The module's own designed
failure path already exists (the 259 outer guard returns the
stored profile and writes nothing), so the inner swallow was
removed: an unreadable history now aborts the refresh into
that fallback instead of fabricating an empty one.  A
genuinely empty history does not raise and still yields
participants={} exactly as before.  The pin asserts the NEW
behavior; A/B proof in nm-tail80-verify/PROGRESS.md.

Probed live first (nm-tail80-verify/probes.py).
"""
import json
import time
from unittest import mock

from nomorals.partner import chat_profile as cp


class FakeProfileDB:
    """Minimal chat_profiles + messages stand-in with fault flags."""

    def __init__(self, messages=()):
        self.row = None
        self.messages = list(messages)
        self.fail_query_one = False
        self.fail_aggregate = False
        self.fail_texts = False
        self.fail_upsert = False
        self.upsert_attempts = 0
        self.upserts = []

    def execute(self, sql, params=()):
        if "INSERT INTO chat_profiles" in sql:
            self.upsert_attempts += 1
            if self.fail_upsert:
                raise RuntimeError("upsert boom")
            self.upserts.append(params)
            self.row = {"profile_json": params[1],
                        "message_count": params[2]}
        return None

    def query_one(self, sql, params=()):
        if self.fail_query_one:
            raise RuntimeError("query_one boom")
        return self.row

    def query(self, sql, params=()):
        if "GROUP BY name" in sql:
            if self.fail_aggregate:
                raise RuntimeError("aggregate boom")
            counts = {}
            for m in self.messages:
                counts[m["name"]] = counts.get(m["name"], 0) + 1
            return [{"name": n, "n": c} for n, c in counts.items()]
        if "FROM messages" in sql:
            if self.fail_texts:
                raise RuntimeError("texts boom")
            return [{"content": m["content"]} for m in self.messages]
        return []


def _seed(db, profile, count):
    db.row = {"profile_json": json.dumps(profile), "message_count": count}


_STORED = {"participants": {"amy": 5, "__owner__": 2},
           "topics": ["gold"],
           "owner_note": "the owner is active in this chat",
           "first_seen": 1_700_000_000.0}


# ── 99: get_profile failure returns the no-profile value ──
def test_99_get_profile_failure_returns_empty_dict():
    db = FakeProfileDB()
    db.fail_query_one = True
    assert cp.get_profile(db, "k") == {}
    # control: a healthy db with no row yields the same value
    assert cp.get_profile(FakeProfileDB(), "k") == {}


# ── 193 CONVERT: aggregate failure preserves the stored profile ──
def test_193_aggregate_failure_preserves_stored_profile():
    db = FakeProfileDB(messages=[{"name": "amy", "content": "gold gold"}])
    _seed(db, dict(_STORED), 7)
    before = dict(db.row)
    db.fail_aggregate = True
    out = cp.refresh_profile(db, "k")
    assert out["participants"] == {"amy": 5, "__owner__": 2}
    assert out["owner_note"] == "the owner is active in this chat"
    # nothing was written: the stored row is byte-identical
    assert db.upsert_attempts == 0
    assert db.row == before
    # control: a genuinely empty history still yields empty participants
    db2 = FakeProfileDB(messages=[])
    _seed(db2, dict(_STORED), 7)
    out2 = cp.refresh_profile(db2, "k")
    assert out2["participants"] == {}
    assert out2["message_count"] == 0


# ── 223: texts failure keeps the old topics ──
def test_223_texts_failure_keeps_old_topics():
    msgs = [{"name": "amy", "content": "silver silver rally"}] * 3
    db = FakeProfileDB(messages=msgs)
    _seed(db, {"topics": ["oldtopic"],
               "first_seen": time.time() - 86400}, 0)
    db.fail_texts = True
    out = cp.refresh_profile(db, "k", force=True)
    assert out["topics"] == ["oldtopic"]


# ── 237: vibe failure keeps the old vibe, topics still mined ──
def test_237_vibe_failure_keeps_old_vibe():
    class BoomScorer:
        def __init__(self):
            raise RuntimeError("scorer boom")

    msgs = [{"name": "amy", "content": "gold gold rally"}] * 3
    db = FakeProfileDB(messages=msgs)
    _seed(db, {"vibe": 0.5, "vibe_label": "warm",
               "first_seen": time.time() - 86400}, 0)
    with mock.patch("nomorals.partner.affect.AffectScorer", BoomScorer):
        out = cp.refresh_profile(db, "k", force=True)
    assert out["vibe_label"] == "warm"
    assert "gold" in out["topics"]


# ── 247: activity failure keeps the old activity values ──
def test_247_activity_failure_keeps_old_activity():
    msgs = [{"name": "amy", "content": "gold gold rally"}] * 3
    db = FakeProfileDB(messages=msgs)
    _seed(db, {"first_seen": "not-a-float", "msgs_per_day": 3.5,
               "activity_label": "steady"}, 0)
    out = cp.refresh_profile(db, "k")
    assert out["msgs_per_day"] == 3.5
    assert out["activity_label"] == "steady"


# ── 259: outer guard returns the stored profile, writes nothing ──
def test_259_upsert_failure_returns_stored_profile():
    msgs = [{"name": "amy", "content": "gold gold rally"}] * 3
    db = FakeProfileDB(messages=msgs)
    _seed(db, {"participants": {"amy": 5}, "topics": ["gold"],
               "first_seen": time.time() - 86400}, 5)
    before = dict(db.row)
    db.fail_upsert = True
    out = cp.refresh_profile(db, "k")
    assert out["participants"] == {"amy": 5}
    assert db.row == before
