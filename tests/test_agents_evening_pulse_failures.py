"""Tail-39 pins for nomorals/agents/evening_pulse.py — 8 exclusions, 0 converts.

Each test pins one site's fallback behavior plus a control half proving the
except path was taken (trigger == fallback AND control != fallback, except
the documented default-equivalence case 49, where the control proves the
fallback EQUALS the normal-path value for standard settings). The file
itself contains no exception handlers, so it adds zero census sites.

evening_pulse.py calls no failures-ledger record function anywhere
(grep-proven: only _log, plus morning_pulse._ledger for pulse runs, which
writes the autonomy ledger, not the failures ledger), so NOT-recording is
pinned with the tail-8 snapshot method: agency recent_failures()
before/after each trigger must be unchanged.

WIRING NOTE (applies to the whole file): evening_pulse.py is currently
unwired in production — no production caller for run_evening_pulse or
ensure_evening_pulse_job, and its scheduled-job payload
{"tool": "pulse", "args": {"action": "run_evening"}} does not match the
pulse tool's action vocabulary (run|status|config|catchup with
slot=evening, dispatched via morning_pulse.run_slot). Every verdict below
is therefore latent as well as honest; the wiring caveats are recorded
per site and the module must be re-examined if it is ever wired.

Verdict notes (full grounds in SURVEY.md):
- 49: _prefs failure -> {}, the documented-defaults fallback; for
  standard settings the normal path yields the same defaults (morning
  _prefs returns time/timezone/enabled, not evening_* keys), so failure
  == normal. Trigger is exotic (morning _prefs essentially never
  raises — its own sidecar read is guarded). Caveat: with sidecar
  evening_* keys AND a property-raising settings object, the pulse
  would run at default time instead of configured time.
- 63: list_jobs failure -> proceeds to add; the return is honest
  (scheduled True, new job id). Caveat: Scheduler.add does NOT dedupe
  by name (straight INSERT), so IF the job were registered, a list
  failure would break the documented single-job promise with a
  duplicate. Dead today (nothing registers the job).
- 75: remove failure -> stale job stays, new job added, return honest.
  Same wiring caveat as 63.
- 123: per-symbol quote failure -> continue; the stage reports an
  honest symbol count (4 vs 5). Graceful degradation, documented.
- 144: weather failure -> preview without weather_tomorrow; the
  compose prompt simply omits the weather string. Honest omission.
- 205: digest fold failure -> digest_ids=[]; the digest is neither
  appended nor marked folded, so the next briefing retries it. By
  design ("the fold never sinks the pulse").
- 220: mark_digest_folded failure -> pass, "bookkeeping only"; delivery
  already confirmed, the digest stays unfolded and is honestly
  re-shown next time (duplicate digest content, not a false claim).
- 223: FLAGGED for parent adjudication (weakest). Publish failure ->
  log fallback + stage deliver ok=True DESPITE the failed publish.
  Under the natural "delivery succeeded" reading the stage bool is
  false in this branch; morning's own _deliver_text returns False on
  failure. Recommendation: ACCEPT as latent (the module is unwired,
  no consumer reads stages["deliver"]["ok"] today, the stage bool in
  this never-raises pipeline means "handled" and the briefing IS
  log-preserved) with a wiring caveat: if the module is ever wired,
  fix 223 FIRST (return False on publish failure, keep True for the
  no-notifier log path) and pass ok=delivery outcome to _ledger (which
  currently records default ok=True before delivery is attempted).
  The pin records OBSERVED behavior only, not endorsed.
"""

from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import failures as ag_fail
from nomorals.agents import evening_pulse as ep
from nomorals.agents import scheduler as sched_mod

WS = "/tmp/nm-tail39-noop"


def _snap():
    return list(ag_fail.recent_failures())


def _ctx_with_settings():
    return SimpleNamespace(settings=SimpleNamespace(workspace_dir=WS))


class _RaisingSettings:
    @property
    def settings(self):
        raise RuntimeError("settings broken")


class _PlainCtx:
    settings = SimpleNamespace(workspace_dir=WS)


class _FakeNotifier:
    def __init__(self, publish_raises=False, mark_raises=False):
        self.publish_calls = []
        self.mark_calls = []
        self.folded = []
        self.publish_raises = publish_raises
        self.mark_raises = mark_raises

    def publish(self, kind, text):
        self.publish_calls.append((kind, text))
        if self.publish_raises:
            raise RuntimeError("gateway down")
        return {"delivered": True}

    def mark_digest_folded(self, ids):
        self.mark_calls.append(list(ids))
        if self.mark_raises:
            raise RuntimeError("db locked")
        self.folded.extend(ids)


def _run_pipeline(quote_raises=(), weather_raises=False, digest_raises=False,
                  mark_raises=False, publish_raises=False):
    """Run the real pipeline with stubbed deps (call-time imports)."""
    quote_calls = []
    prompts = []

    def fake_quote(sym, market=None):
        quote_calls.append(sym)
        if sym in quote_raises:
            raise RuntimeError(f"quote {sym} down")
        return {"symbol": sym, "price": 100.0, "change_pct_24h": 1.0}

    class FakeBrain:
        def chat(self, msgs):
            prompts.append(msgs[0]["content"])
            return SimpleNamespace(text="Evening recap canned.")

    ctx_notifier = _FakeNotifier(publish_raises=publish_raises,
                                 mark_raises=mark_raises)
    ctx = SimpleNamespace(
        notifier=ctx_notifier,
        settings=SimpleNamespace(workspace_dir=WS))

    class WiredCls:
        def __init__(self, context):
            self.inst = ctx_notifier

        def digest_unfolded(self, limit=20):
            return [{"id": "d1"}, {"id": "d2"}]

    def fake_digest_lines(rows):
        if digest_raises:
            raise RuntimeError("digest render down")
        return ["digest line 1", "digest line 2"]

    def fake_geocode(place):
        if weather_raises:
            raise RuntimeError("geo down")
        return SimpleNamespace(latitude=6.5, longitude=3.4)

    def fake_forecast(lat, lon, days=2, units="metric"):
        if weather_raises:
            raise RuntimeError("forecast down")
        return {"daily": [{"temp_max": 30, "temp_min": 24},
                          {"temp_max": 31, "temp_min": 25}]}

    with patch("nomorals.agents.morning_pulse._fetch_news",
               return_value=[{"title": "Story A"}]), \
        patch("nomorals.integrations.market_data.quote",
              side_effect=fake_quote), \
        patch("nomorals.agents.weather.geocode", side_effect=fake_geocode), \
        patch("nomorals.agents.weather.fetch_forecast",
              side_effect=fake_forecast), \
        patch("nomorals.llm.brain.brain_for", return_value=FakeBrain()), \
        patch("nomorals.agents.morning_briefing.alert_digest_lines",
              side_effect=fake_digest_lines), \
        patch("nomorals.agents.notifier.Notifier", WiredCls), \
        patch("nomorals.agents.morning_pulse._ledger", return_value=None):
        result = ep.run_evening_pulse(ctx)
    return result, ctx_notifier, quote_calls, prompts


def test_49_prefs_failure_returns_empty_defaults():
    before = _snap()
    assert ep._prefs(_RaisingSettings()) == {}
    prefs_ok = ep._prefs(_PlainCtx())
    assert isinstance(prefs_ok, dict) and "time" in prefs_ok
    assert ep.pulse_time(_RaisingSettings()) \
        == ep.pulse_time(_PlainCtx()) == "21:00"
    assert _snap() == before


def test_63_list_failure_proceeds_to_add():
    class FakeSched:
        def __init__(self, context):
            self.store = [{"id": "old1", "name": "evening-pulse",
                           "spec": "daily 21:00"}]
            self.add_calls = []

        def list_jobs(self):
            raise RuntimeError("db down")

        def add(self, *a, **k):
            self.add_calls.append(a)
            return {"id": "new1"}

    before = _snap()
    with patch.object(sched_mod, "Scheduler", FakeSched):
        r = ep.ensure_evening_pulse_job(_ctx_with_settings())
    assert r.get("scheduled") is True and r.get("job_id") == "new1"
    assert _snap() == before


def test_63_control_already_scheduled_skips_add():
    class FakeSched:
        def __init__(self, context):
            self.add_calls = []

        def list_jobs(self):
            return [{"id": "old1", "name": "evening-pulse",
                     "spec": "daily 21:00",
                     "missed_fire_policy": "fire_now",
                     "overlap_policy": "skip", "run_timeout_s": 1800}]

        def add(self, *a, **k):
            self.add_calls.append(a)
            return {"id": "new1"}

    captured = {}

    class Cap(FakeSched):
        def __init__(self, context):
            super().__init__(context)
            captured["inst"] = self

    with patch.object(sched_mod, "Scheduler", Cap):
        r = ep.ensure_evening_pulse_job(_ctx_with_settings())
    assert r.get("already_scheduled") is True
    assert captured["inst"].add_calls == []


def test_75_remove_failure_leaves_stale_adds_new():
    class FakeSched:
        def __init__(self, context):
            self.store = [{"id": "old1", "name": "evening-pulse",
                           "spec": "daily 20:00"}]
            self.remove_calls = []
            self.add_calls = []

        def list_jobs(self):
            return list(self.store)

        def remove(self, ref):
            self.remove_calls.append(ref)
            raise RuntimeError("db locked")

        def add(self, *a, **k):
            self.add_calls.append(a)
            self.store.append({"id": "new1", "name": "evening-pulse"})
            return {"id": "new1"}

    captured = {}

    class Cap(FakeSched):
        def __init__(self, context):
            super().__init__(context)
            captured["inst"] = self

    before = _snap()
    with patch.object(sched_mod, "Scheduler", Cap):
        r = ep.ensure_evening_pulse_job(_ctx_with_settings())
    inst = captured["inst"]
    assert inst.remove_calls == ["old1"]
    assert len(inst.add_calls) == 1
    assert any(j["id"] == "old1" for j in inst.store)
    assert r.get("scheduled") is True
    assert _snap() == before


def test_75_control_remove_success_clears_stale():
    class FakeSched:
        def __init__(self, context):
            self.store = [{"id": "old1", "name": "evening-pulse",
                           "spec": "daily 20:00"}]
            self.add_calls = []

        def list_jobs(self):
            return list(self.store)

        def remove(self, ref):
            self.store = [j for j in self.store if j["id"] != ref]
            return True

        def add(self, *a, **k):
            self.add_calls.append(a)
            return {"id": "new1"}

    captured = {}

    class Cap(FakeSched):
        def __init__(self, context):
            super().__init__(context)
            captured["inst"] = self

    with patch.object(sched_mod, "Scheduler", Cap):
        ep.ensure_evening_pulse_job(_ctx_with_settings())
    inst = captured["inst"]
    assert not any(j["id"] == "old1" for j in inst.store)
    assert len(inst.add_calls) == 1


def test_123_symbol_failure_continues_honest_count():
    before = _snap()
    res, _, quote_calls, _ = _run_pipeline(quote_raises=("XAUUSD",))
    assert "XAUUSD" in quote_calls
    assert res["stages"]["market"]["symbols"] == 4
    assert res["stages"]["market"]["ok"] is True
    assert _snap() == before
    res_c, _, _, _ = _run_pipeline()
    assert res_c["stages"]["market"]["symbols"] == 5


def test_144_weather_failure_omits_weather_string():
    before = _snap()
    _, _, _, prompts_fail = _run_pipeline(weather_raises=True)
    assert "Tomorrow's weather:" not in prompts_fail[0]
    assert _snap() == before
    _, _, _, prompts_ok = _run_pipeline()
    assert "Tomorrow's weather:" in prompts_ok[0]


def test_205_digest_failure_leaves_digest_unfolded():
    before = _snap()
    res, _, _, _ = _run_pipeline(digest_raises=True)
    assert "Alert digest" not in res["briefing"]
    assert _snap() == before
    res_c, _, _, _ = _run_pipeline()
    assert "Alert digest" in res_c["briefing"]


def test_220_mark_folded_failure_is_bookkeeping_only():
    before = _snap()
    res, notif, _, _ = _run_pipeline(mark_raises=True)
    assert len(notif.mark_calls) == 1
    assert notif.folded == []
    assert res["stages"]["deliver"]["ok"] is True
    assert _snap() == before
    _, notif_c, _, _ = _run_pipeline()
    assert notif_c.folded == ["d1", "d2"]


def test_223_publish_failure_still_reports_deliver_ok():
    # OBSERVED behavior only, FLAGGED for parent adjudication (see module
    # docstring): a failed publish still yields deliver ok=True, because
    # the never-raises pipeline falls through to the log fallback. The
    # publish attempt itself is proven via the notifier call log, so
    # this is not the no-notifier path.
    before = _snap()
    res, notif, _, _ = _run_pipeline(publish_raises=True)
    assert len(notif.publish_calls) == 1
    assert res["stages"]["deliver"]["ok"] is True
    assert _snap() == before
    res_c, notif_c, _, _ = _run_pipeline()
    assert len(notif_c.publish_calls) == 1
    assert res_c["stages"]["deliver"]["ok"] is True
