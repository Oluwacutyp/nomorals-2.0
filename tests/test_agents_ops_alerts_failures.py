"""Tail-40 pins for nomorals/agents/ops_alerts.py — 7 exclusions, 1 convert.

Each test pins one exclusion site's fallback behavior plus a control half
proving the except path was taken (trigger == fallback AND control !=
fallback, except the documented default-equivalence case 41, where the
control proves failure == the documented never-alerted value). The file
itself contains no exception handlers, so it adds zero census sites.

ops_alerts.py calls no failures-ledger record function anywhere
(grep-proven: no `record_failure` import or call; the file imports only
time + KVStore), so NOT-recording is pinned with the tail-8 snapshot
method: agency recent_failures() before/after each trigger must be
unchanged.

Site 48's pin records OBSERVED behavior only, not endorsed: it is
FLAGGED for parent adjudication (see SURVEY.md) — the marker write
fails silently yet the heal-skip proceeds, so while KV writes fail the
mission is heal-attempted every tick and never paged. Do not read this
pin as blessing that behavior.

Verdict notes (full grounds in SURVEY.md):
- 41: KV read failure -> 0.0, the documented never-alerted value;
  fail-open toward paging, bounded by the notifier's 10-min dedupe.
- 48: FLAGGED (see above).
- 71: heal failure -> attempted False -> mission IN findings (paged).
  The honest pattern in this file.
- 111/160/138: one dead surface must not kill the scan; surviving
  surfaces' findings returned exactly. Observability gap noted (no
  surface-health field), not a convert.
- 230: pending-count failure -> 0; diagnostic field, console omits the
  line when 0 rather than claiming an empty queue.
"""
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, "/home/hatch/workspace/nm-tail40-tree")

from nomorals.agents import failures as ag_fail
from nomorals.agents import ops_alerts as oa


def _snap():
    return list(ag_fail.recent_failures())


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


class FakeRunner:
    def __init__(self, entries, heal_result=None, heal_exc=None):
        self._entries = entries
        self.heal_result = heal_result
        self.heal_exc = heal_exc
        self.heal_calls = []

    def health(self, limit=50):
        return {"active": self._entries}

    def self_heal(self, mission_id, background=True):
        self.heal_calls.append(mission_id)
        if self.heal_exc is not None:
            raise self.heal_exc
        return self.heal_result


STUCK = {"id": "m1234567890abcdef", "stuck": True, "goal": "do thing",
         "iterations": 3, "completed": ["a", "b"], "last_error": "boom"}


def _ctx():
    return SimpleNamespace(db=object())


def _reset():
    FakeKV.store = {}
    FakeKV.fail_get = False
    FakeKV.fail_set = False


def _canned_skill():
    s = SimpleNamespace(name="route:scout", success_count=1, failure_count=9,
                        success_rate=0.1, last_used=1000.0)
    s.to_dict = lambda: {"name": s.name}
    return s


class FakeLib:
    def __init__(self, db, items=None, exc=None):
        self._items = items
        self._exc = exc

    def list(self, kind="routing", limit=200):
        if self._exc is not None:
            raise self._exc
        return self._items or []


def _runs_fake(runs=None, exc=None):
    class FO:
        @staticmethod
        def runs(context, limit=40):
            if exc is not None:
                raise exc
            return runs or []
    return FO


class FakeNotifier:
    def __init__(self, context, pending=None, pending_exc=None):
        self._pending = pending
        self._pending_exc = pending_exc

    def pending(self):
        if self._pending_exc is not None:
            raise self._pending_exc
        return self._pending or []


def test_41_cooldown_read_failure_is_never_alerted():
    _reset()
    FakeKV.fail_get = True
    before = _snap()
    with patch.object(oa, "KVStore", FakeKV):
        assert oa._kv_get(_ctx().db, "opsalert:x") == 0.0
    FakeKV.fail_get = False
    FakeKV.store["opsalert:x"] = "1234.5"
    with patch.object(oa, "KVStore", FakeKV):
        assert oa._kv_get(_ctx().db, "opsalert:x") == 1234.5
    assert _snap() == before


def test_48_marker_write_failure_skip_proceeds_flagged():
    """OBSERVED ONLY, flagged, not endorsed (see module docstring)."""
    _reset()
    FakeKV.fail_set = True
    runner = FakeRunner([STUCK], heal_result={"attempted": True, "status": "queued"})
    before = _snap()
    with patch.object(oa, "KVStore", FakeKV), \
            patch("nomorals.missions.wired_runner", return_value=runner):
        findings = oa._stuck_findings(_ctx(), escalate=True,
                                      heal_background=True, healed_out=[])
    assert findings == []
    assert FakeKV.store.get("opsselfheal:stuck:m1234567890abcdef") is None
    assert runner.heal_calls == ["m1234567890abcdef"]
    _reset()
    runner2 = FakeRunner([STUCK], heal_result={"attempted": True, "status": "queued"})
    with patch.object(oa, "KVStore", FakeKV), \
            patch("nomorals.missions.wired_runner", return_value=runner2):
        oa._stuck_findings(_ctx(), escalate=True, heal_background=True,
                           healed_out=[])
    assert FakeKV.store.get("opsselfheal:stuck:m1234567890abcdef") is not None
    assert _snap() == before


def test_71_heal_failure_pages_mission():
    _reset()
    runner = FakeRunner([STUCK], heal_exc=RuntimeError("heal down"))
    before = _snap()
    healed = []
    with patch.object(oa, "KVStore", FakeKV), \
            patch("nomorals.missions.wired_runner", return_value=runner):
        findings = oa._stuck_findings(_ctx(), escalate=True,
                                      heal_background=True, healed_out=healed)
    assert len(findings) == 1 and findings[0]["kind"] == "stuck_mission"
    assert healed == []
    assert _snap() == before


def test_111_stuck_surface_dead_other_surfaces_survive():
    _reset()
    before = _snap()
    with patch.object(oa, "KVStore", FakeKV), \
            patch("nomorals.missions.wired_runner",
                  side_effect=RuntimeError("runner down")), \
            patch("nomorals.agents.skills.SkillLibrary",
                  lambda db: FakeLib(db, items=[_canned_skill()])), \
            patch("nomorals.agents.orchestrator.MasterOrchestrator",
                  _runs_fake(runs=[])):
        findings, _ = oa.scan_ops_escalated(_ctx(), escalate=False)
    assert [f["kind"] for f in findings] == ["chronic_role"]
    assert _snap() == before


def test_160_pivot_surface_dead_other_surfaces_survive():
    _reset()
    before = _snap()
    with patch.object(oa, "KVStore", FakeKV), \
            patch("nomorals.missions.wired_runner",
                  return_value=FakeRunner([STUCK])), \
            patch("nomorals.agents.skills.SkillLibrary",
                  lambda db: FakeLib(db, items=[_canned_skill()])), \
            patch("nomorals.agents.orchestrator.MasterOrchestrator",
                  _runs_fake(exc=RuntimeError("orch down"))):
        findings, _ = oa.scan_ops_escalated(_ctx(), escalate=False)
    assert sorted(f["kind"] for f in findings) == ["chronic_role",
                                                   "stuck_mission"]
    assert _snap() == before


def test_138_chronic_surface_dead_other_surfaces_survive():
    _reset()
    before = _snap()
    run = {"ts": 9999999999.0, "goal": "g", "pivots": [{"task": "t", "pivot": "p"}]}
    with patch.object(oa, "KVStore", FakeKV), \
            patch("nomorals.missions.wired_runner",
                  return_value=FakeRunner([STUCK])), \
            patch("nomorals.agents.skills.SkillLibrary",
                  lambda db: FakeLib(db, exc=RuntimeError("skills down"))), \
            patch("nomorals.agents.orchestrator.MasterOrchestrator",
                  _runs_fake(runs=[run])):
        findings, _ = oa.scan_ops_escalated(_ctx(), escalate=False)
    assert sorted(f["kind"] for f in findings) == ["stuck_mission",
                                                   "unrecovered_pivot"]
    assert _snap() == before


def test_230_pending_failure_is_zero_diagnostic():
    _reset()
    before = _snap()
    with patch.object(oa, "KVStore", FakeKV), \
            patch.object(oa, "scan_ops_escalated", return_value=([], [])), \
            patch("nomorals.agents.notifier.Notifier",
                  lambda ctx: FakeNotifier(ctx, pending_exc=RuntimeError("db"))):
        res = oa.ops_alerts(_ctx())
    assert res["pending"] == 0
    with patch.object(oa, "KVStore", FakeKV), \
            patch.object(oa, "scan_ops_escalated", return_value=([], [])), \
            patch("nomorals.agents.notifier.Notifier",
                  lambda ctx: FakeNotifier(ctx, pending=[{"a": 1}, {"b": 2}])):
        res2 = oa.ops_alerts(_ctx())
    assert res2["pending"] == 2
    assert _snap() == before
