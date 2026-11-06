"""AA-44 supersession pins for nomorals/agents/afternoon_pulse.py.

CONVERTED 2026-11-05 (final assembly-audit pass).  This file was the
tail-83 pin set for the legacy module's standalone pipeline (5
exclusions, 0 converts).  AA-44 consolidated the pulses: the legacy
afternoon/evening modules were parallel registrars whose job payload
({"tool": "pulse", "args": {"action": "run_afternoon"}}) named an
action the pulse tool cannot dispatch, while the live multi-slot
engine (morning_pulse.BRIEFING_SLOTS + ensure_pulse_jobs, registered
at partner boot) already ran both slots.  The legacy modules are now
documented delegation shims over morning_pulse.ensure_slot_job /
run_slot, so the old pipeline pins are replaced by pins of the NEW
truth:

- test_50 survives verbatim: the shim keeps the legacy flat-pref
  helpers (_prefs/pulse_time/pulse_timezone/pulse_enabled)
  byte-identical, including the documented-defaults fallback.
- 64/76 (scheduler edge cases) now pin the LIVE registrar's behavior
  reached through the shim: the payload is the pulse tool's real
  vocabulary (action=run, slot=afternoon), and list/remove failures
  are RECORDED in the failures ledger by the live engine before it
  proceeds — the inverse of the old pins' NOT-recording assertions,
  which described the dead registrar.  Fixture job dicts use the
  stored-spec shape the real Scheduler returns ("13:00", kind kept
  separately), not the legacy registrar's "daily 13:00" compare
  string.
- 119/163 (market stage / publish fallback) are retired with the
  pipeline they pinned.  The tail-39 BINDING caveat on site 163
  (fix the false deliver-ok FIRST if the module is ever wired) is
  DISCHARGED by the consolidation: the live engine's
  _deliver_slot_text returns the real delivery outcome and run_slot
  ledgers ok=delivered, so the false-ok shape does not exist on the
  live path.  Delegation itself is pinned below.
"""

from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import afternoon_pulse as ap
from nomorals.agents import failures as ag_fail
from nomorals.agents import morning_pulse as mp
from nomorals.agents import scheduler as sched_mod


def _snap():
    return list(ag_fail.recent_failures(limit=200))


def _fake_sched(store, list_raises=False, remove_raises=False):
    captured = {}

    class FakeSched:
        def __init__(self, context):
            self.store = list(store)
            self.add_calls = []
            self.remove_calls = []
            captured["inst"] = self

        def list_jobs(self):
            if list_raises:
                raise RuntimeError("db down")
            return list(self.store)

        def remove(self, ref):
            self.remove_calls.append(ref)
            if remove_raises:
                raise RuntimeError("db locked")
            self.store = [j for j in self.store if j["id"] != ref]
            return True

        def add(self, *a, **k):
            self.add_calls.append((a, k))
            return {"id": "new1"}

    return FakeSched, captured


def _ctx(tmp_path):
    # settings with a workspace_dir: morning_pulse._prefs completes its
    # sidecar branch cleanly (file absent) instead of recording a
    # sidecar failure to the agency ledger.
    return SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))


def test_50_prefs_failure_returns_empty_defaults():
    def boom(context):
        raise RuntimeError("prefs store down")

    before = _snap()
    with patch.object(mp, "_prefs", boom):
        assert ap._prefs(SimpleNamespace()) == {}
        assert ap.pulse_time(SimpleNamespace()) == "13:00"
        assert ap.pulse_timezone(SimpleNamespace()) == "America/Denver"
        assert ap.pulse_enabled(SimpleNamespace()) is True
    assert _snap() == before
    with patch.object(mp, "_prefs",
                      lambda c: {"afternoon_pulse_time": "15:30"}):
        assert ap.pulse_time(SimpleNamespace()) == "15:30"


def test_64_fresh_add_uses_live_payload_vocabulary(tmp_path):
    FakeSched, cap = _fake_sched([])
    with patch.object(sched_mod, "Scheduler", FakeSched):
        r = ap.ensure_afternoon_pulse_job(_ctx(tmp_path))
    assert r.get("scheduled") is True and r.get("job_id") == "new1"
    (args, _kwargs) = cap["inst"].add_calls[0]
    assert args[0] == "afternoon-pulse"
    assert args[1] == "daily 13:00"
    assert args[3] == {"tool": "pulse",
                       "args": {"action": "run", "slot": "afternoon"}}


def test_64_list_failure_recorded_and_proceeds_to_add(tmp_path):
    FakeSched, cap = _fake_sched(
        [{"id": "old1", "name": "afternoon-pulse", "spec": "13:00"}],
        list_raises=True)
    before = _snap()
    with patch.object(sched_mod, "Scheduler", FakeSched):
        r = ap.ensure_afternoon_pulse_job(_ctx(tmp_path))
    assert r.get("scheduled") is True and r.get("job_id") == "new1"
    assert len(cap["inst"].add_calls) == 1
    # the live engine records the failure (the dead one swallowed it)
    assert len(_snap()) > len(before)


def test_64_control_already_scheduled_skips_add(tmp_path):
    FakeSched, cap = _fake_sched(
        [{"id": "old1", "name": "afternoon-pulse", "spec": "13:00",
          "missed_fire_policy": "fire_now", "overlap_policy": "skip",
          "run_timeout_s": 1800}])
    with patch.object(sched_mod, "Scheduler", FakeSched):
        r = ap.ensure_afternoon_pulse_job(_ctx(tmp_path))
    assert r.get("already_scheduled") is True
    assert cap["inst"].add_calls == []


def test_76_remove_failure_recorded_leaves_stale_adds_new(tmp_path):
    FakeSched, cap = _fake_sched(
        [{"id": "old1", "name": "afternoon-pulse", "spec": "12:00"}],
        remove_raises=True)
    before = _snap()
    with patch.object(sched_mod, "Scheduler", FakeSched):
        r = ap.ensure_afternoon_pulse_job(_ctx(tmp_path))
    inst = cap["inst"]
    assert inst.remove_calls == ["old1"]
    assert len(inst.add_calls) == 1
    assert any(j["id"] == "old1" for j in inst.store)
    assert r.get("scheduled") is True
    assert len(_snap()) > len(before)


def test_76_control_remove_success_clears_stale(tmp_path):
    FakeSched, cap = _fake_sched(
        [{"id": "old1", "name": "afternoon-pulse", "spec": "12:00"}])
    with patch.object(sched_mod, "Scheduler", FakeSched):
        ap.ensure_afternoon_pulse_job(_ctx(tmp_path))
    assert not any(j["id"] == "old1" for j in cap["inst"].store)
    assert len(cap["inst"].add_calls) == 1


def test_run_delegates_to_live_slot_runner():
    sentinel = {"ok": True, "slot": "afternoon"}
    ctx = SimpleNamespace()
    with patch.object(mp, "run_slot", return_value=sentinel) as run:
        out = ap.run_afternoon_pulse(ctx)
    assert out is sentinel
    run.assert_called_once_with(ctx, "afternoon")
