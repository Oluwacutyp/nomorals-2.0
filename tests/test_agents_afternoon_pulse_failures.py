"""Tail-83 pins for nomorals/agents/afternoon_pulse.py — 5 exclusions, 0 converts.

Each test pins one site's fallback behavior plus a control half proving
the except path was taken (trigger == fallback AND control != fallback,
except the documented default-equivalence case 50, where the control
proves the fallback EQUALS the normal-path value for standard settings).
The file itself contains no exception handlers, so it adds zero census
sites.

afternoon_pulse.py calls no failures-ledger record function anywhere
(grep-proven: only _log, plus morning_pulse._ledger for pulse runs,
which writes the autonomy ledger, not the failures ledger), so
NOT-recording is pinned with the snapshot method: agency
recent_failures() before/after each trigger must be unchanged.

WIRING NOTE (applies to the whole file): afternoon_pulse.py is
currently unwired in production — no production caller for
run_afternoon_pulse or ensure_afternoon_pulse_job, and its
scheduled-job payload {"tool": "pulse", "args": {"action":
"run_afternoon"}} does not match the pulse tool's action vocabulary
(run|status|config|catchup with slot=afternoon, dispatched via
morning_pulse.run_slot). Every verdict below is therefore latent as
well as honest; the module must be re-examined if it is ever wired.

Verdict notes (full grounds in SURVEY.md):
- 50: _prefs failure -> {}, the documented-defaults fallback; for
  standard settings the normal path yields the same defaults (morning
  _prefs returns time/timezone/enabled, not afternoon_* keys), so
  failure == normal. Caveat: with sidecar afternoon_* keys AND a
  raising prefs source, the pulse would run at default time.
- 64: list_jobs failure -> proceeds to add; the return is honest
  (scheduled True, new job id). Caveat: Scheduler.add does NOT dedupe
  by name (straight INSERT), so IF the job were registered, a list
  failure would break the documented single-job promise with a
  duplicate. Dead today (nothing registers the job).
- 76: remove failure -> stale job stays, new job added, return honest.
  Same wiring caveat as 64.
- 119: per-symbol quote failure -> continue; the stage reports an
  honest symbol count (3 vs 4). Graceful degradation, documented.
- 163: publish failure -> log fallback + stage deliver ok=True DESPITE
  the failed publish. This is the identical twin of evening_pulse 223,
  adjudicated by the parent in tail-39: ACCEPTED as pins-only latent
  with a BINDING wiring caveat (fix 163 FIRST if the module is ever
  wired: return False on publish failure, keep True for the
  no-notifier log path; pass ok=delivery outcome to _ledger; fix the
  payload vocabulary). The pin records OBSERVED behavior only, not
  endorsed.
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
            self.add_calls.append(a)
            return {"id": "new1"}

    return FakeSched, captured


def _run_pipeline(tmp_path, quote_raises=(), publish_raises=False,
                  notifier=True):
    def fake_quote(sym, market=None):
        if sym in quote_raises:
            raise RuntimeError(f"quote {sym} down")
        return {"symbol": sym, "price": 100.0, "change_pct_24h": 1.0}

    class FakeBrain:
        def chat(self, msgs):
            return SimpleNamespace(text="Afternoon canned.")

    class FakeNotifier:
        def __init__(self):
            self.publish_calls = []

        def publish(self, kind, text):
            self.publish_calls.append((kind, text))
            if publish_raises:
                raise RuntimeError("gateway down")
            return {"delivered": True}

    fake_notifier = FakeNotifier()
    ctx = SimpleNamespace(
        notifier=fake_notifier if notifier else None,
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    ledger_calls = []
    with patch.object(mp, "_fetch_news",
                      return_value=[{"title": "Story A"}]), \
            patch("nomorals.integrations.market_data.quote",
                  side_effect=fake_quote), \
            patch("nomorals.llm.brain.brain_for",
                  return_value=FakeBrain()), \
            patch.object(mp, "_ledger",
                         side_effect=lambda *a, **k:
                         ledger_calls.append(a)):
        result = ap.run_afternoon_pulse(ctx)
    return result, fake_notifier, ledger_calls


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


def _ctx(tmp_path):
    # settings with a workspace_dir: morning_pulse._prefs completes its
    # sidecar branch cleanly (file absent) instead of recording a
    # sidecar failure to the agency ledger.
    return SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))


def test_64_list_failure_proceeds_to_add(tmp_path):
    FakeSched, cap = _fake_sched(
        [{"id": "old1", "name": "afternoon-pulse", "spec": "daily 13:00"}],
        list_raises=True)
    before = _snap()
    with patch.object(sched_mod, "Scheduler", FakeSched):
        r = ap.ensure_afternoon_pulse_job(_ctx(tmp_path))
    assert r.get("scheduled") is True and r.get("job_id") == "new1"
    assert len(cap["inst"].add_calls) == 1
    assert _snap() == before


def test_64_control_already_scheduled_skips_add(tmp_path):
    FakeSched, cap = _fake_sched(
        [{"id": "old1", "name": "afternoon-pulse", "spec": "daily 13:00",
          "missed_fire_policy": "fire_now", "overlap_policy": "skip",
          "run_timeout_s": 1800}])
    with patch.object(sched_mod, "Scheduler", FakeSched):
        r = ap.ensure_afternoon_pulse_job(_ctx(tmp_path))
    assert r.get("already_scheduled") is True
    assert cap["inst"].add_calls == []


def test_76_remove_failure_leaves_stale_adds_new(tmp_path):
    FakeSched, cap = _fake_sched(
        [{"id": "old1", "name": "afternoon-pulse", "spec": "daily 12:00"}],
        remove_raises=True)
    before = _snap()
    with patch.object(sched_mod, "Scheduler", FakeSched):
        r = ap.ensure_afternoon_pulse_job(_ctx(tmp_path))
    inst = cap["inst"]
    assert inst.remove_calls == ["old1"]
    assert len(inst.add_calls) == 1
    assert any(j["id"] == "old1" for j in inst.store)
    assert r.get("scheduled") is True
    assert _snap() == before


def test_76_control_remove_success_clears_stale(tmp_path):
    FakeSched, cap = _fake_sched(
        [{"id": "old1", "name": "afternoon-pulse", "spec": "daily 12:00"}])
    with patch.object(sched_mod, "Scheduler", FakeSched):
        ap.ensure_afternoon_pulse_job(_ctx(tmp_path))
    assert not any(j["id"] == "old1" for j in cap["inst"].store)
    assert len(cap["inst"].add_calls) == 1


def test_119_symbol_failure_continues_honest_count(tmp_path):
    before = _snap()
    res, _, _ = _run_pipeline(tmp_path, quote_raises=("XAUUSD",))
    assert res["stages"]["market"]["symbols"] == 3
    assert res["stages"]["market"]["ok"] is True
    assert _snap() == before
    res_c, _, _ = _run_pipeline(tmp_path)
    assert res_c["stages"]["market"]["symbols"] == 4


def test_163_publish_failure_still_reports_deliver_ok(tmp_path):
    # OBSERVED behavior only, under the tail-39 parent adjudication of
    # the identical evening site (see module docstring): a failed
    # publish still yields deliver ok=True, because the never-raises
    # pipeline falls through to the log fallback. The publish attempt
    # itself is proven via the notifier call log, and the ledger write
    # (which precedes delivery) via the ledger spy.
    before = _snap()
    res, notif, ledger = _run_pipeline(tmp_path, publish_raises=True)
    assert len(notif.publish_calls) == 1
    assert len(ledger) == 1
    assert res["stages"]["deliver"]["ok"] is True
    assert res["briefing"] == "Afternoon canned."
    assert _snap() == before
    res_c, notif_c, _ = _run_pipeline(tmp_path)
    assert len(notif_c.publish_calls) == 1
    assert res_c["stages"]["deliver"]["ok"] is True
    res_n, _, _ = _run_pipeline(tmp_path, notifier=False)
    assert res_n["stages"]["deliver"]["ok"] is True
