"""Tail-57 pins for nomorals/agents/modes.py —
0 converts / 7 exclusions (219, 244, 351, 841, 886, 1484, 1490).

7 tests for 7 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the operation's own call log / double.
The file contains no exception handlers, so it adds zero
census sites.

modes.py calls no failures-ledger record function anywhere
(grep-proven: the only `record_` calls are the learner's own
record_outcome, a different contract), so NOT-recording is
pinned against the agents package ledger
(nomorals/agents/failures.py): recent_failures() snapshot
before/after every trigger.

Verdict notes (full grounds in the tail-57 SURVEY.md):
- 219: _settings_home settings failure -> ~/.nomorals; the
  docstring's declared fallback ("~/.nomorals when settings
  cannot load"). Every store consumer resolves through the
  same function, so the fallback is consistent.
- 244: _default_store_path settings.resolve failure -> the
  same ~/.nomorals/data/agent-modes.json literal. Mirror of
  219 at the file level.
- 351: AdaptiveLearner store.learning() failure -> {}; a
  fresh learner with no evidence is the genuine value (equal
  weights, no preference), per the class docstring.
- 841: _step_spec_key json.dumps failure -> repr of the
  parameters; repr still fingerprints the step's work, and a
  fingerprint mismatch only ever causes a re-run (safe
  direction), never a wrongful skip.
- 886: _cancel_requested broken cancel signal -> True; the
  site's documented fail-safe ("reads as stop"). Stopping a
  run on a broken signal makes no false claim.
- 1484: stale tick-job remove() failure -> pass, then add()
  proceeds anyway; the registration is re-attempted in full,
  and any add() failure lands in the 1490 outer handler.
- 1490: ensure_agent_modes_tick_job any failure -> None;
  the docstring's best-effort contract ("a missing scheduler
  just means no scheduled ticks — /agent tick still works"),
  and the sole production caller (partner runtime boot)
  discards the return as a status receipt.
"""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nomorals.agents.modes as modes
from nomorals.agents.failures import recent_failures


def _snap():
    return list(recent_failures(200))


# ── 219: _settings_home settings failure -> ~/.nomorals ────────────────

def test_219_settings_home_failure_falls_back_to_dot_nomorals():
    before = _snap()

    class BadHome:
        def __init__(self):
            self.calls = 0

        @property
        def home_path(self):
            self.calls += 1
            raise RuntimeError("settings boom")

    bad = BadHome()
    assert modes._settings_home(bad) == Path.home() / ".nomorals"
    assert bad.calls == 1  # the settings read was actually attempted
    good = SimpleNamespace(home_path="/x/y")
    assert modes._settings_home(good) == Path("/x/y")  # control
    assert _snap() == before


# ── 244: _default_store_path resolve failure -> fallback path ──────────

def test_244_default_store_path_failure_falls_back(monkeypatch):
    monkeypatch.delenv("NM_AGENT_MODES_STORE", raising=False)
    before = _snap()

    class BadResolve:
        def __init__(self):
            self.calls = []

        def resolve(self, rel):
            self.calls.append(rel)
            raise RuntimeError("resolve boom")

    bad = BadResolve()
    expected = Path.home() / ".nomorals" / "data" / "agent-modes.json"
    assert modes._default_store_path(bad) == expected
    assert bad.calls == ["data/agent-modes.json"]  # resolve attempted
    good = SimpleNamespace(resolve=lambda rel: Path("/z") / rel)
    assert modes._default_store_path(good) == Path("/z/data/agent-modes.json")
    assert _snap() == before


# ── 351: AdaptiveLearner learning() failure -> fresh records ───────────

def test_351_learner_store_failure_starts_fresh():
    before = _snap()

    class BadStore:
        def __init__(self):
            self.calls = 0

        def learning(self):
            self.calls += 1
            raise RuntimeError("learning boom")

    bad = BadStore()
    learner = modes.AdaptiveLearner(store=bad)
    assert bad.calls == 1  # the store read was actually attempted
    assert learner._records == {}
    assert learner.learned_summary()["outcomes"] == 0
    # fresh learner = genuine no-evidence value: equal weights
    assert learner.weights(["a", "b"]) == {"a": 0.5, "b": 0.5}

    class GoodStore:
        def learning(self):
            return {"ctx": {"s1": {"ok": 3, "fail": 1}}}

    learner2 = modes.AdaptiveLearner(store=GoodStore())
    assert learner2._counts("ctx", "s1") == (3, 1)  # control
    assert _snap() == before


# ── 841: _step_spec_key dumps failure -> repr fingerprint ──────────────

def test_841_step_spec_key_falls_back_to_repr():
    before = _snap()
    real_dumps = json.dumps
    dump_calls = []

    def counting_dumps(obj, *a, **k):
        dump_calls.append(obj)
        return real_dumps(obj, *a, **k)

    params = {1: "x", "b": 2}  # mixed-type keys: sort_keys raises
    step = SimpleNamespace(action="act", name="nm", parameters=params)
    with mock.patch.object(modes.json, "dumps", counting_dumps):
        key = modes._step_spec_key(step)
    assert len(dump_calls) == 1  # the dumps was actually attempted
    assert key == ("act", "nm", repr(params))
    step2 = SimpleNamespace(action="act", name="nm", parameters={"b": 2})
    assert modes._step_spec_key(step2) == ("act", "nm", '{"b": 2}')
    assert _snap() == before


# ── 886: _cancel_requested broken signal -> True (stop) ────────────────

def test_886_broken_cancel_signal_reads_as_stop():
    before = _snap()
    calls = []

    def bad_cancel():
        calls.append(1)
        raise RuntimeError("cancel boom")

    assert modes.PlanningRunner._cancel_requested(bad_cancel) is True
    assert calls == [1]  # the signal was actually consulted
    ev = threading.Event()
    assert modes.PlanningRunner._cancel_requested(ev) is False
    ev.set()
    assert modes.PlanningRunner._cancel_requested(ev) is True
    assert modes.PlanningRunner._cancel_requested(lambda: False) is False
    assert modes.PlanningRunner._cancel_requested(None) is False
    assert _snap() == before


# ── 1484: stale tick-job remove failure -> add proceeds ────────────────

def test_1484_stale_job_remove_failure_still_registers():
    before = _snap()

    class Sched:
        def __init__(self):
            self.removed, self.added = [], []

        def list_jobs(self):
            return [{"name": modes.AGENT_MODES_TICK_JOB,
                     "spec": "every 99m", "run_timeout_s": 0.0,
                     "id": "old1"}]

        def remove(self, job_id):
            self.removed.append(job_id)
            raise RuntimeError("remove boom")

        def add(self, name, spec, kind, payload, **k):
            self.added.append(name)
            return {"id": "new1", "name": name}

    sched = Sched()
    out = modes.ensure_agent_modes_tick_job(sched)
    assert sched.removed == ["old1"]  # the remove was actually attempted
    assert sched.added == [modes.AGENT_MODES_TICK_JOB]
    assert out == {"id": "new1", "name": modes.AGENT_MODES_TICK_JOB}
    assert _snap() == before


# ── 1490: ensure_agent_modes_tick_job failure -> None ──────────────────

def test_1490_tick_job_failure_returns_none():
    before = _snap()

    class Sched:
        def __init__(self):
            self.calls, self.added = 0, []

        def list_jobs(self):
            self.calls += 1
            raise RuntimeError("list boom")

        def add(self, *a, **k):
            self.added.append(a)
            return {"id": "x"}

    sched = Sched()
    assert modes.ensure_agent_modes_tick_job(sched) is None
    assert sched.calls == 1 and sched.added == []  # list attempted, no add

    class SchedOK:
        def list_jobs(self):
            return []

        def add(self, name, spec, kind, payload, **k):
            return {"id": "j1", "name": name}

    assert modes.ensure_agent_modes_tick_job(SchedOK()) == {
        "id": "j1", "name": modes.AGENT_MODES_TICK_JOB}  # control
    assert modes.ensure_agent_modes_tick_job(None) is None
    assert _snap() == before
