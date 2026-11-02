"""Tail-92 pins: nomorals/missions/progress.py census sites.

Every site sits under an explicit never-raise contract printed
in the module itself: ``_historical_step_rate`` documents
"Never raises — no db, no history"; ``MissionWatchers`` declares
"every db touch is best-effort so a subscription failure never
breaks a mission run or a chat command"; ``MissionMilestones``
declares "Nothing here may raise: milestone reporting is
telemetry, and telemetry never breaks a run", and ``_fanout``
"Never raises: fan-out is telemetry". All five held as
exclusions:

226 — ETA history prior: a db query failure returns (None, 0),
the exact value "no history" returns. The consumer
(eta_breakdown, surfaced via MissionStore detail) falls back to
the mission's own ema / run-average methods — a less-informed
ETA, never a wrong action.
858 — ``MissionWatchers._read``: a KV read failure reads as
"no subscriptions" ([]). Fan-out and subscribe/unsubscribe all
treat [] as the empty state by the class's best-effort contract.
869 — ``MissionWatchers._write``: a KV write failure is
swallowed; subscribe() still returns True and the chat layer
replies "watching …". WEAKEST EXCLUSION of this batch — see the
pin below: the success reply is real, but the class contract is
explicit, the milestone itself still reaches the owner via the
primary Notifier publish (watchers are an extra fan-out), and
making the write raise would break the stated never-break-a-
chat-command contract.
989 — ``_fanout``: resolving the watcher list raises -> the
fan-out is skipped wholesale; the Notifier publish to owner
channels has already happened by design (fan-out rides on top).
1014 — ``_fanout`` per-chat send: one dead chat's send raises
-> continue, so the remaining subscribed chats still receive
the milestone. Loop isolation, the standard fan-out shape.

NOT-recording is pinned with the snapshot method: this module
calls no failures-ledger record function anywhere, so both
ledgers must be unchanged across each trigger. (The milestone
handlers' own ``_log.debug`` sites at 1038+ are outside this
census chunk and untouched.)
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.missions import progress as P


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


class _BoomDb:
    def query_one(self, *args, **kwargs):
        raise RuntimeError("db boom")


class _BoomGetKV:
    def __init__(self, db):
        self._db = db

    def get(self, key):
        raise RuntimeError("kv get boom")

    def set(self, key, value):
        raise AssertionError("set must not be reached in this pin")


class _BoomSetKV:
    def __init__(self, db):
        self._db = db

    def get(self, key):
        return None

    def set(self, key, value):
        raise RuntimeError("kv set boom")


# ── 226: history query failure == no history ────────────────────

def test_226_history_query_failure_returns_no_history():
    before = _snap()
    assert P._historical_step_rate(_BoomDb()) == (None, 0)
    assert P._historical_step_rate(None) == (None, 0)  # control
    assert _snap() == before


# ── 858: watcher read failure reads as no subscriptions ─────────

def test_858_watcher_read_failure_returns_empty():
    before = _snap()
    watchers = P.MissionWatchers(object())
    with mock.patch.object(P, "KVStore", _BoomGetKV):
        assert watchers._read("m1") == []
        assert watchers.watchers("m1") == []
    assert _snap() == before


# ── 869: watcher write failure swallowed (weakest exclusion) ────

def test_869_watcher_write_failure_is_best_effort():
    before = _snap()
    watchers = P.MissionWatchers(object())
    with mock.patch.object(P, "KVStore", _BoomSetKV):
        # the write itself: swallowed, returns None, never raises
        assert watchers._write("m1", ["telegram:1"]) is None
        # the pinned consequence, stated honestly: subscribe still
        # reports "newly added" although nothing persisted — held
        # only by the class's explicit best-effort contract and
        # because the owner still gets the milestone via Notifier.
        assert watchers.subscribe("m1", "telegram:1") is True
    assert _snap() == before


# ── 989/1014: fan-out never raises, one dead chat isolated ──────

def _milestones(watch_store, gateway):
    ms = P.MissionMilestones.__new__(P.MissionMilestones)
    ms.watch_store = watch_store
    ms._notifier = SimpleNamespace(gateway=gateway)
    ms.context = SimpleNamespace(
        settings=SimpleNamespace(
            partner=SimpleNamespace(owner_chats="")))
    return ms


class _BoomWatchers:
    def watchers(self, mission_id):
        raise RuntimeError("watch store boom")


class _StaticWatchers:
    def __init__(self, chats):
        self._chats = chats

    def watchers(self, mission_id):
        return list(self._chats)


class _Gateway:
    def __init__(self, dead_platform):
        self.sent = []
        self._dead = dead_platform

    def send(self, platform, ref, text):
        if platform == self._dead:
            raise RuntimeError("chat unreachable")
        self.sent.append((platform, ref.chat_id, text))


def test_989_fanout_skips_when_watcher_lookup_fails():
    before = _snap()
    gateway = _Gateway(dead_platform="__none__")
    ms = _milestones(_BoomWatchers(), gateway)
    mission = SimpleNamespace(id="m1", state={})
    assert ms._fanout(mission, "title", "body") is None
    assert gateway.sent == []
    assert _snap() == before


def test_1014_fanout_one_dead_chat_does_not_kill_the_rest():
    before = _snap()
    gateway = _Gateway(dead_platform="telegram")
    ms = _milestones(
        _StaticWatchers(["telegram:111", "console:222"]), gateway)
    mission = SimpleNamespace(id="m1", state={})
    assert ms._fanout(mission, "title", "body") is None
    # telegram:111 died inside its own try; console:222 still got it
    assert [(p, c) for p, c, _t in gateway.sent] == [("console", "222")]
    assert _snap() == before
