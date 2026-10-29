"""Tail-40 guard for convert 224 in nomorals/agents/ops_alerts.py.

Convert: the cooldown marker + sent.append moved into the try's else
branch, so they run ONLY when doc construction + notify succeed. On any
failure the finding is NEITHER reported sent NOR cooldown-suppressed, so
the next tick retries it instead of losing the page for 6h.

Stash-proven to fail on old code: with the production change stashed
(guard file untracked and kept), G1 and G2 FAIL (old code reports sent
+ sets the marker despite the raise); G3 and G4 pass on both (controls).
After pop, all four pass. The file itself contains no exception
handlers, so it adds zero census sites.
"""
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, "/home/hatch/workspace/nm-tail40-tree")

from nomorals.agents import ops_alerts as oa


class FakeKV:
    store = {}
    fail_get = False
    fail_set = False

    def __init__(self, db):
        pass

    def get_raw(self, key):
        if FakeKV.fail_get:
            raise RuntimeError("kv read down")
        return FakeKV.store.get(key)

    def set_raw(self, key, value, kind="text", ttl=None):
        if FakeKV.fail_set:
            raise RuntimeError("kv write down")
        FakeKV.store[key] = value


class FakeNotifier:
    def __init__(self, context, pending=None):
        self._pending = pending or []

    def pending(self):
        return self._pending


def _ctx():
    return SimpleNamespace(db=object())


def _finding():
    return {"kind": "stuck_mission", "key": "stuck:m123",
            "title": "Mission stuck: m123", "body": "b", "data": {}}


def _reset():
    FakeKV.store = {}
    FakeKV.fail_get = False
    FakeKV.fail_set = False


def test_224_notify_raise_is_not_sent_and_not_suppressed():
    """FAILS on old code (old code reports sent + sets the marker)."""
    _reset()

    def raising_notify(*a, **k):
        raise RuntimeError("notifier db down")

    with patch.object(oa, "KVStore", FakeKV), \
            patch.object(oa, "scan_ops_escalated",
                         return_value=([_finding()], [])), \
            patch("nomorals.agents.notifier.notify", raising_notify), \
            patch("nomorals.agents.notifier.Notifier",
                  lambda ctx: FakeNotifier(ctx)):
        res = oa.ops_alerts(_ctx())
    assert res["sent"] == []
    assert FakeKV.store.get("opsalert:stuck:m123") is None


def test_224_doc_raise_is_not_sent_and_not_suppressed():
    """FAILS on old code (same shape, doc-construction trigger)."""
    _reset()
    with patch.object(oa, "KVStore", FakeKV), \
            patch.object(oa, "scan_ops_escalated",
                         return_value=([_finding()], [])), \
            patch("nomorals.agents.alert_templates.ops_finding_doc",
                  side_effect=RuntimeError("doc boom")), \
            patch("nomorals.agents.notifier.Notifier",
                  lambda ctx: FakeNotifier(ctx)):
        res = oa.ops_alerts(_ctx())
    assert res["sent"] == []
    assert FakeKV.store.get("opsalert:stuck:m123") is None


def test_224_success_still_sent_and_marked_control():
    """Control: passes on old and new code."""
    _reset()
    calls = []

    def ok_notify(*a, **k):
        calls.append(a)
        return {"delivered": True}

    with patch.object(oa, "KVStore", FakeKV), \
            patch.object(oa, "scan_ops_escalated",
                         return_value=([_finding()], [])), \
            patch("nomorals.agents.notifier.notify", ok_notify), \
            patch("nomorals.agents.notifier.Notifier",
                  lambda ctx: FakeNotifier(ctx)):
        res = oa.ops_alerts(_ctx())
    assert calls, "notify was never attempted"
    assert res["sent"] == [_finding()]
    assert FakeKV.store.get("opsalert:stuck:m123") is not None


def test_224_force_success_still_sent_and_marked_control():
    """Control: force bypasses the check, not the success record."""
    _reset()

    def ok_notify(*a, **k):
        return {"delivered": True}

    with patch.object(oa, "KVStore", FakeKV), \
            patch.object(oa, "scan_ops_escalated",
                         return_value=([_finding()], [])), \
            patch("nomorals.agents.notifier.notify", ok_notify), \
            patch("nomorals.agents.notifier.Notifier",
                  lambda ctx: FakeNotifier(ctx)):
        res = oa.ops_alerts(_ctx(), force=True)
    assert res["sent"] == [_finding()]
    assert FakeKV.store.get("opsalert:stuck:m123") is not None
