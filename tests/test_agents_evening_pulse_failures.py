"""AA-44 supersession pins for nomorals/agents/evening_pulse.py.

CONVERTED 2026-11-05 (final assembly-audit pass).  This file was the
tail-39 pin set for the legacy module's standalone pipeline.  AA-44
consolidated the pulses: the legacy afternoon/evening modules were
parallel registrars whose job payload ({"tool": "pulse", "args":
{"action": "run_evening"}}) named an action the pulse tool cannot
dispatch, while the live multi-slot engine
(morning_pulse.BRIEFING_SLOTS + ensure_pulse_jobs, registered at
partner boot) already ran both slots.  The legacy modules are now
documented delegation shims over morning_pulse.ensure_slot_job /
run_slot, so the old pipeline pins are replaced by pins of the NEW
truth:

- test_49 survives: the shim keeps the legacy flat-pref helpers
  byte-identical, including the documented-defaults fallback.  (The
  helper's flat default stays "21:00" for import compatibility; the
  LIVE evening slot's settled default is 19:00 in BRIEFING_SLOTS —
  the shim's helpers are not the schedule.)
- 63/75 (scheduler edge cases) now pin the LIVE registrar's behavior
  reached through the shim: the payload is the pulse tool's real
  vocabulary (action=run, slot=evening), and list/remove failures
  are RECORDED in the failures ledger by the live engine before it
  proceeds — the inverse of the old pins' NOT-recording assertions,
  which described the dead registrar.  Fixture job dicts use the
  stored-spec shape the real Scheduler returns ("19:00"), not the
  legacy registrar's "daily 21:00" compare string.
- 123 (market stage), 144 (tomorrow weather), 205/220 (digest fold /
  mark-folded) and 223 (publish fallback) are retired with the
  pipeline they pinned.  Their behaviors live on, on the live
  engine, and are pinned in tests/test_final_audit_pass.py:
  weather → morning_pulse._tomorrow_preview; the F7 digest fold →
  morning_pulse._fold_evening_digest / _mark_evening_digest_folded
  inside run_slot (marked folded only on confirmed delivery — the
  old 205/220 discipline, preserved).  The tail-39 BINDING caveat on
  site 223 (fix the false deliver-ok FIRST if the module is ever
  wired) is DISCHARGED by the consolidation: the live engine's
  _deliver_slot_text returns the real delivery outcome and run_slot
  ledgers ok=delivered, so the false-ok shape does not exist on the
  live path.  Delegation itself is pinned below.
"""

from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import evening_pulse as ep
from nomorals.agents import failures as ag_fail
from nomorals.agents import morning_pulse as mp
from nomorals.agents import scheduler as sched_mod


def _snap():
    return list(ag_fail.recent_failures(limit=200))


def _ctx(tmp_path):
    return SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))


class _RaisingSettings:
    @property
    def settings(self):
        raise RuntimeError("settings broken")


class _PlainCtx:
    settings = SimpleNamespace(workspace_dir="/nonexistent-nm")


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


def test_49_prefs_failure_returns_empty_defaults():
    before = _snap()
    assert ep._prefs(_RaisingSettings()) == {}
    prefs_ok = ep._prefs(_PlainCtx())
    assert isinstance(prefs_ok, dict) and "time" in prefs_ok
    assert ep.pulse_time(_RaisingSettings()) \
        == ep.pulse_time(_PlainCtx()) == "21:00"
    assert _snap() == before


def test_63_fresh_add_uses_live_payload_vocabulary(tmp_path):
    FakeSched, cap = _fake_sched([])
    with patch.object(sched_mod, "Scheduler", FakeSched):
        r = ep.ensure_evening_pulse_job(_ctx(tmp_path))
    assert r.get("scheduled") is True and r.get("job_id") == "new1"
    (args, _kwargs) = cap["inst"].add_calls[0]
    assert args[0] == "evening-pulse"
    # the live slot's settled default time, not the legacy 21:00
    assert args[1] == "daily 19:00"
    assert args[3] == {"tool": "pulse",
                       "args": {"action": "run", "slot": "evening"}}


def test_63_list_failure_recorded_and_proceeds_to_add(tmp_path):
    FakeSched, cap = _fake_sched(
        [{"id": "old1", "name": "evening-pulse", "spec": "19:00"}],
        list_raises=True)
    before = _snap()
    with patch.object(sched_mod, "Scheduler", FakeSched):
        r = ep.ensure_evening_pulse_job(_ctx(tmp_path))
    assert r.get("scheduled") is True and r.get("job_id") == "new1"
    assert len(cap["inst"].add_calls) == 1
    assert len(_snap()) > len(before)


def test_63_control_already_scheduled_skips_add(tmp_path):
    FakeSched, cap = _fake_sched(
        [{"id": "old1", "name": "evening-pulse", "spec": "19:00",
          "missed_fire_policy": "fire_now", "overlap_policy": "skip",
          "run_timeout_s": 1800}])
    with patch.object(sched_mod, "Scheduler", FakeSched):
        r = ep.ensure_evening_pulse_job(_ctx(tmp_path))
    assert r.get("already_scheduled") is True
    assert cap["inst"].add_calls == []


def test_75_remove_failure_recorded_leaves_stale_adds_new(tmp_path):
    FakeSched, cap = _fake_sched(
        [{"id": "old1", "name": "evening-pulse", "spec": "20:00"}],
        remove_raises=True)
    before = _snap()
    with patch.object(sched_mod, "Scheduler", FakeSched):
        r = ep.ensure_evening_pulse_job(_ctx(tmp_path))
    inst = cap["inst"]
    assert inst.remove_calls == ["old1"]
    assert len(inst.add_calls) == 1
    assert any(j["id"] == "old1" for j in inst.store)
    assert r.get("scheduled") is True
    assert len(_snap()) > len(before)


def test_75_control_remove_success_clears_stale(tmp_path):
    FakeSched, cap = _fake_sched(
        [{"id": "old1", "name": "evening-pulse", "spec": "20:00"}])
    with patch.object(sched_mod, "Scheduler", FakeSched):
        ep.ensure_evening_pulse_job(_ctx(tmp_path))
    inst = cap["inst"]
    assert not any(j["id"] == "old1" for j in inst.store)
    assert len(inst.add_calls) == 1


def test_run_delegates_to_live_slot_runner():
    sentinel = {"ok": True, "slot": "evening"}
    ctx = SimpleNamespace()
    with patch.object(mp, "run_slot", return_value=sentinel) as run:
        out = ep.run_evening_pulse(ctx)
    assert out is sentinel
    run.assert_called_once_with(ctx, "evening")
