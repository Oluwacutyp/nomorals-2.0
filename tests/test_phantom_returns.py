"""Pass 2 (return-discard / phantom family): the callee already returns
an honest outcome — the caller must branch on it instead of reporting
success regardless.

One section per row of ~/workspace/next-families-scope-2026-11-06.md
Pass 2. Tests marked TEETH fail on the pristine base for the right
reason (the phantom success itself); tests marked PIN pass on both
trees and lock the unchanged success path.
"""

import os
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import toolmaker as tm
from nomorals.agents.monitor import MonitorAgent
from nomorals.audio import overview as ov
from nomorals.core import self_heal as sh
from nomorals.games.film import FilmStore, analyze_vod
from nomorals.health import nutrition as N
from nomorals.legal import aid
from nomorals.legal import failures as lf
from nomorals.media.taste import TasteStore
from nomorals.social import whatsapp_cost as WC
from nomorals.travel import whitelabel
from nomorals.travel.whitelabel import TravelClientStore

# ── 2.1 complete_meal: save_meal's "" must reach the MealLog ─────────────


class _FakeSeer:
    def __init__(self, text):
        self.text = text

    def see(self, path, question=""):
        return self.text


class _FakeTimeline:
    def log(self, *a, **k):
        return SimpleNamespace(id="ev1")


def _draft():
    return N.log_meal("/tmp/meal.jpg",
                      seer=_FakeSeer("jollof rice | one plate"))


def test_meal_save_failure_is_carried_on_the_log(tmp_path):
    # TEETH: pristine discards save_meal's return — meal_saved doesn't
    # exist and the summary claims a clean log either way.
    store = N.NutritionStore(db_path=str(tmp_path / "n.db"))
    store._db = None  # save_meal's documented failure shape: return ""
    log = N.complete_meal(_draft(), {"portion": "medium"},
                          store=store, timeline=_FakeTimeline())
    assert log.meal_id == ""
    assert log.meal_saved is False
    assert "won't count in today's totals" in log.summary()


def test_meal_save_success_path_unchanged(tmp_path):
    # PIN: a landed save renders exactly as before and persists.
    store = N.NutritionStore(db_path=str(tmp_path / "n.db"))
    log = N.complete_meal(_draft(), {"portion": "medium"},
                          store=store, timeline=_FakeTimeline())
    assert "won't count in today's totals" not in log.summary()
    assert "logged:" in log.summary()
    rows = store.meals_on(time.strftime("%Y-%m-%d"))
    assert len(rows) == 1


# ── 2.2 record_feedback: save()'s False must not read "taste updated." ───


def _taste_store(tmp_path):
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    store = TasteStore(ctx)
    store.profile.last_production = {
        "title": "t", "bpm": 120.0, "key": "A", "mode": "minor",
        "energy_words": (), "genre": ""}
    return store


def test_record_feedback_save_failure_uses_hiccup_wording(tmp_path,
                                                         monkeypatch):
    # TEETH: pristine returns "taste updated." even when save() is False.
    store = _taste_store(tmp_path)
    monkeypatch.setattr(store, "save", lambda: False)
    out = store.record_feedback(True)
    assert out == "feedback noted (taste save hiccup)."
    assert "taste updated" not in out


def test_record_feedback_success_path_unchanged(tmp_path):
    # PIN
    store = _taste_store(tmp_path)
    out = store.record_feedback(True)
    assert "taste updated." in out
    assert store.path.is_file()


# ── 2.3 film: FilmStore.save's False must ride the breakdown ─────────────


def _vision(responses):
    calls = {"n": 0}

    def vision(image_path, question):
        calls["n"] += 1
        return responses[(calls["n"] - 1) % len(responses)]
    return vision


def _img(path):
    from PIL import Image
    Image.new("RGB", (64, 64), "green").save(path)
    return path


class _FalseStore:
    def save(self, bd):
        return False


_VISION_OK = _vision(["MISTAKE — overextended alone pushing mid\n"
                      "GOOD PLAY — clean headshot on the entry frag"])


def test_film_save_failure_marks_breakdown(tmp_path):
    # TEETH: pristine discards the False; the breakdown looks saved and
    # /film export <id> dead-ends later.
    img = _img(os.path.join(str(tmp_path), "frame.jpg"))
    bd = analyze_vod(img, "valorant", vision=_VISION_OK,
                     store=_FalseStore())
    assert bd.available
    assert bd.saved is False
    assert "not saved" in bd.format()


def test_film_workout_save_failure_marks_breakdown(tmp_path, monkeypatch):
    # TEETH (second caller, _analyze_workout_vod)
    fake_fa = SimpleNamespace(
        issues=[SimpleNamespace(observed="knees cave in",
                                fix="push knees out")],
        passes=["depth"], available=True, band="good", note="",
        frames_used=3)
    monkeypatch.setattr("nomorals.health.form.analyze_form",
                        lambda path, exercise: fake_fa)
    bd = analyze_vod("clip.mp4", "workout", vision=_VISION_OK,
                     store=_FalseStore(), exercise="squat")
    assert bd.available
    assert bd.saved is False


def test_film_save_success_path_unchanged(tmp_path, monkeypatch):
    # PIN
    monkeypatch.setenv("NOMORALS_HOME", str(tmp_path))
    store = FilmStore()
    img = _img(os.path.join(str(tmp_path), "frame.jpg"))
    bd = analyze_vod(img, "valorant", vision=_VISION_OK, store=store)
    assert store.get(bd.id) is not None
    assert "not saved" not in bd.format()


# ── 2.4 whitelabel: a failed INSERT must not mint a phantom client ───────


def _broken_client_store(tmp_path):
    store = TravelClientStore(str(tmp_path / "wl.db"))
    store._db.execute("DROP TABLE travel_clients")
    store._db.commit()
    return store


def test_whitelabel_create_failure_returns_none(tmp_path):
    # TEETH: pristine returns the client object despite the failed
    # INSERT — callers render "created", later lookups find nothing.
    store = _broken_client_store(tmp_path)
    assert store.create("Ghost Air") is None
    assert store.list(active_only=False) == []


def test_whitelabel_chat_create_failure_is_honest(tmp_path):
    # TEETH: the /travelclient surface must not print ✅ for a phantom.
    store = _broken_client_store(tmp_path)
    fake_self = SimpleNamespace(_travelclient_store=store)
    out = whitelabel._control_travelclient(fake_self, "create Ghost Air")
    assert "couldn't create" in out
    assert "✅" not in out


def test_whitelabel_create_success_path_unchanged(tmp_path):
    # PIN
    store = TravelClientStore(str(tmp_path / "wl.db"))
    c = store.create("Test Air")
    assert c is not None
    assert store.get(c.id).name == "Test Air"
    fake_self = SimpleNamespace(_travelclient_store=store)
    out = whitelabel._control_travelclient(fake_self, "create Second Air")
    assert "✅" in out and "created" in out


# ── 2.5 toolmaker: create() must carry install()'s outcome ──────────────


def _stubbed_maker(monkeypatch, install_result):
    maker = tm.ToolMaker(SimpleNamespace(router=None, db=None, tools=None))
    spec = mock.Mock()
    spec.name = "probe_tool"
    spec.description = "d"
    spec.to_dict.return_value = {"name": "probe_tool"}
    monkeypatch.setattr(maker, "design", lambda task: spec)
    monkeypatch.setattr(maker, "generate", lambda s: "code")
    monkeypatch.setattr(maker, "test", lambda code: {"ok": True})
    monkeypatch.setattr(
        maker, "install",
        lambda name, code, description="", tested=False: install_result)
    return maker


def test_toolmaker_create_carries_registration_outcome(monkeypatch):
    # TEETH: pristine drops registered/note — a registration failure
    # reads as a clean ok=True install.
    maker = _stubbed_maker(monkeypatch, {
        "ok": True, "path": "/tmp/probe_tool.py",
        "registered": False, "note": "failed to register"})
    res = maker.create("make a probe")
    assert res["ok"] is True  # install()'s own contract: file landed
    assert res["registered"] is False
    assert res["note"] == "failed to register"


def test_toolmaker_create_carries_deferred_outcome(monkeypatch):
    # TEETH: registered=None ("will register on next build") rides too.
    maker = _stubbed_maker(monkeypatch, {
        "ok": True, "path": "/tmp/probe_tool.py",
        "registered": None, "note": "will register on next build"})
    res = maker.create("make a probe")
    assert res["registered"] is None
    assert res["note"] == "will register on next build"


def test_toolmaker_create_install_failure_stays_failure(monkeypatch):
    # PIN
    maker = _stubbed_maker(monkeypatch, {"ok": False, "error": "boom"})
    res = maker.create("make a probe")
    assert res["ok"] is False


# ── 2.6 make_overview: OverviewStore.save's False must surface ───────────


def _overview_sources():
    return [{"title": "T", "text": "gold rose sharply [S1] today"}]


def _script_fn(p):
    return "A: gold rose sharply [S1] today\nB: indeed [S1]"


def test_make_overview_save_failure_is_named(tmp_path, monkeypatch):
    # TEETH: pristine returns ok=True + "overview ready" with the
    # save outcome discarded entirely.
    store = ov.OverviewStore(str(tmp_path / "ov.db"))
    monkeypatch.setattr(store, "save", lambda o: False)
    res = ov.make_overview(_overview_sources(), "deep-dive",
                           script_fn=_script_fn, render_audio=False,
                           store=store)
    assert res["ok"] is True  # the overview itself was built
    assert res["saved"] is False
    assert "not saved" in res["note"]


def test_make_overview_success_path_unchanged(tmp_path):
    # PIN
    store = ov.OverviewStore(str(tmp_path / "ov.db"))
    res = ov.make_overview(_overview_sources(), "deep-dive",
                           script_fn=_script_fn, render_audio=False,
                           store=store)
    assert res["ok"] is True
    assert "not saved" not in res["note"]
    assert store.get(res["overview"].overview_id) is not None


# ── 2.7 monitor: watch_down webhook outcome must ride the tick output ────


class _FakeDB:
    def __init__(self, rows=()):
        self.executes = []
        self._rows = list(rows)

    def execute(self, sql, params=()):
        self.executes.append((sql, params))

    def query_one(self, *a):
        return None

    def query(self, *a):
        return list(self._rows)


def _monitor_row(**over):
    row = {"id": "m1", "target": "https://example.com/x", "kind": "url",
           "watch": "content", "last_hash": "", "last_size": 0,
           "last_content": "", "last_change_ts": 0.0, "error_streak": 0,
           "webhook_url": "https://hooks.example.com/x",
           "webhook_secret": "", "min_alert_gap_s": 0.0,
           "last_alert_ts": 0.0, "auto_decode": 0, "volatile": ""}
    row.update(over)
    return row


_SENTINEL_WH = {"fired": True, "status": 500, "ok": False,
                "attempts": 3, "error": "HTTP 500"}


def _monitor_agent(monkeypatch, db, wh=_SENTINEL_WH):
    agent = MonitorAgent(SimpleNamespace(db=db),
                         notifier=SimpleNamespace(publish=lambda **kw: None))
    monkeypatch.setattr(agent, "_fire_webhook",
                        lambda row, kind, entry, now: wh)
    return agent


def test_note_error_returns_webhook_outcome(monkeypatch):
    # TEETH: pristine _note_error discards _fire_webhook's result,
    # against _fire_webhook's own "rides along in the tick output"
    # contract (which _note_check honors and _note_error didn't).
    agent = _monitor_agent(monkeypatch, _FakeDB())
    out = agent._note_error(_monitor_row(error_streak=2), "boom",
                            now=1000.0)
    assert out == _SENTINEL_WH


def test_tick_errors_carry_webhook_outcome(monkeypatch):
    # TEETH: the tick's errors list is where the outcome surfaces.
    row = _monitor_row(error_streak=2)
    agent = _monitor_agent(monkeypatch, _FakeDB(rows=[row]))

    def dead_fetch(r):
        raise RuntimeError("fetch dead")
    monkeypatch.setattr(agent, "_fetch", dead_fetch)
    out = agent.tick(now=1000.0)
    assert out["checked"] == 1
    assert out["errors"][0]["webhook"] == _SENTINEL_WH


def test_note_error_below_alert_streak_returns_none(monkeypatch):
    # PIN: no alert fired → no outcome, on both trees.
    agent = _monitor_agent(monkeypatch, _FakeDB())
    assert agent._note_error(_monitor_row(error_streak=0), "boom",
                             now=1000.0) is None


# ── 2.10 self_heal: a failed revert must not read "reverted" ─────────────


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src))
    return str(p)


def _raise_unbound(modpath):
    import importlib.util
    spec = importlib.util.spec_from_file_location("healmod_unbound2",
                                                  modpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod.handler("research")
    except UnboundLocalError as e:
        return e
    raise AssertionError("expected UnboundLocalError")


def _raise_attr(modpath):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("healmod_attr2", modpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["healmod_attr2"] = mod
    spec.loader.exec_module(mod)
    obj = mod.Widget()
    try:
        obj._control_podcast("")
    except AttributeError as e:
        return e
    raise AssertionError("expected AttributeError")


_UNBOUND_SRC = '''
def handler(kind):
    """Handle a command."""
    if kind == "miniapp":
        chat = "miniapp-chat"
        return chat.upper()
    if kind == "research":
        return f"research:{chat}"
'''

_DEAD_METHOD_SRC = '''
class Widget:
    def name(self):
        return "widget"


def _resolve_something(x):
    def _control_podcast(self, tail):
        """Build a podcast."""
        return f"podcast: {tail}"
    return x
'''


def _fail_second_write(monkeypatch):
    real_write = sh._write_source
    calls = {"n": 0}

    def flaky(filename, lines):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write(filename, lines)
        return False
    monkeypatch.setattr(sh, "_write_source", flaky)
    monkeypatch.setattr(sh, "_compiles", lambda filename: False)


def test_unbound_revert_failure_is_named(tmp_path, monkeypatch):
    # TEETH: pristine reports "reverted" while the non-compiling fix
    # is still on disk.
    p = _write(tmp_path, "m1.py", _UNBOUND_SRC)
    original = Path(p).read_text(encoding="utf-8")
    exc = _raise_unbound(p)
    _fail_second_write(monkeypatch)
    out = sh.fix_unbound_local(exc, dry_run=False)
    assert out["ok"] is False
    assert "revert write failed" in out["skip_reason"]
    # the broken fix really remains on disk
    assert Path(p).read_text(encoding="utf-8") != original


def test_dead_method_revert_failure_is_named(tmp_path, monkeypatch):
    # TEETH (second revert path)
    p = _write(tmp_path, "w1.py", _DEAD_METHOD_SRC)
    exc = _raise_attr(p)
    _fail_second_write(monkeypatch)
    out = sh.fix_dead_method(exc, dry_run=False)
    assert out["ok"] is False
    assert "revert write failed" in out["skip_reason"]


def test_unbound_revert_success_path_unchanged(tmp_path, monkeypatch):
    # PIN: a landed revert keeps the original wording + original bytes.
    p = _write(tmp_path, "m2.py", _UNBOUND_SRC)
    original = Path(p).read_text(encoding="utf-8")
    exc = _raise_unbound(p)
    monkeypatch.setattr(sh, "_compiles", lambda filename: False)
    out = sh.fix_unbound_local(exc, dry_run=False)
    assert out["skip_reason"] == "fixed source did not compile; reverted"
    assert Path(p).read_text(encoding="utf-8") == original


# ── 1.11 whatsapp_cost: a failed INSERT must not read as 0 ("no spend") ──


class _BoomDB:
    def execute(self, *a, **k):
        raise RuntimeError("db dead")

    def commit(self):
        raise RuntimeError("db dead")


def test_cost_track_write_failure_returns_none(monkeypatch):
    # TEETH: pristine swallows the INSERT failure and returns 0,
    # indistinguishable from "no spend".
    tracker = WC.CostTracker(":memory:")
    monkeypatch.setattr(tracker, "_db", _BoomDB())
    assert tracker.track("+1", "service") is None


def test_cost_track_success_path_unchanged():
    # PIN
    tracker = WC.CostTracker(":memory:")
    assert tracker.track("+1", "service") == WC.RATE_SERVICE_KOBO
    assert tracker.spent_since(0) == WC.RATE_SERVICE_KOBO


def test_bill_answer_track_failure_recorded_not_phantom(monkeypatch):
    # TEETH (downstream caller of track): bill_answer keeps its own
    # adjudicated contract — 0 returned AND the failure recorded —
    # now fed by track's honest None instead of the swallowed 0.
    monkeypatch.setattr(WC.CostTracker, "track",
                        lambda self, *a, **k: None)
    lf.clear_failures()
    try:
        assert aid.bill_answer("2348000000000",
                               db_path=":memory:") == 0
        assert any(e.site == "aid.bill_answer"
                   for e in lf.recent_failures())
    finally:
        lf.clear_failures()
