"""Tail-45 pins for nomorals/community/miniapps.py — 0 converts /
7 exclusions, 1 site flagged (1204).

8 tests for 8 sites (counts equal). Each test pins one site's
fallback behavior plus a control half proving the guarded path
was actually exercised (trigger == fallback AND control !=
fallback), asserting via the double's own call log that the
failing operation itself was attempted. The file contains no
exception handlers, so it adds zero census sites.

miniapps.py calls no failures-ledger record function anywhere
(grep-proven: no record_failure import or call), so
NOT-recording is pinned with the tail-8 snapshot method:
agents recent_failures() before/after each trigger must be
unchanged.

Verdict notes (full grounds in the tail-45 SURVEY.md):
- 1204 FLAGGED: create_panel catch-all -> a fabricated default
  Panel that was never persisted; its only production consumer
  (control_panel) renders "✅ Panel created" over it — the
  phantom-create shape. Fix = contract change (the docstring
  promises "Never raises — returns the panel"); pin is OBSERVED.
- 960: closes:<bad> token dropped at parse; the created poll's
  render is literally true (no deadline line when none exists).
- 1088: needs_refresh failure -> False, the documented
  never-raises safe-direction value; fields coerced on load.
- 1104: age_str failure -> "unknown", literally honest.
- 1221: due_refreshes failure -> [], a catch-all over two
  already fail-soft layers (load + needs_refresh).
- 1327: _flatten failure -> partial/{}; persisted paths always
  feed it a real dict.
- 1343: _fmt_value failure -> "?", an honest unknown glyph.
- 1457: render_panel failure -> name-only render; asserts
  nothing (the "no data yet" line is success-path only).
"""
import sys
import time

sys.path.insert(0, "/home/hatch/workspace/nm-tail45-tree")

from nomorals.agents import failures as ag_fail
from nomorals.community import miniapps as mp


def _snap():
    return list(ag_fail.recent_failures())


# ── chat harness (mirrors tests/test_community_miniapps.py) ──────

class _Chat:
    def __init__(self, kind="group", platform="telegram",
                 chat_id="123", thread_id=""):
        self.kind = kind
        self.platform = platform
        self.chat_id = chat_id
        self.thread_id = thread_id


class _Settings:
    def __init__(self, community_dir):
        self.community_dir = community_dir


class _Ctx:
    def __init__(self, community_dir):
        self.settings = _Settings(community_dir)


class RaisingPutStore:
    """PanelStore double whose put() raises; logs the attempt."""

    def __init__(self):
        self.put_calls = []
        self.saved = []

    def put(self, panel):
        self.put_calls.append(panel.id)
        raise RuntimeError("disk exploded")

    def load(self, user_key):
        return list(self.saved)


class RaisingLoadStore:
    """PanelStore double whose load() raises; logs the attempt."""

    def __init__(self):
        self.load_calls = 0

    def load(self, user_key):
        self.load_calls += 1
        raise RuntimeError("store unreadable")


class BadStr:
    def __init__(self):
        self.str_calls = 0

    def __str__(self):
        self.str_calls += 1
        raise RuntimeError("unprintable")


class ExplodingData:
    def __init__(self):
        self.get_calls = 0

    def __bool__(self):
        return True

    def get(self, *a, **k):
        self.get_calls += 1
        raise RuntimeError("corrupt data blob")


def test_960_malformed_closes_flag_is_dropped(tmp_path):
    before = _snap()
    ctx = _Ctx(str(tmp_path))
    chat = _Chat()
    reply = mp.control_miniapp('new poll "Q?" a b closes:abc',
                               context=ctx, chat=chat,
                               sender_id="u1", sender="Ada")
    # the reply is a normal Created reply — had the ValueError
    # escaped the parse guard, control_miniapp would have raised
    assert "Created" in reply
    apps = mp.MiniAppStore(data_dir=tmp_path).load("telegram:123")
    assert len(apps) == 1
    app = apps[0]
    assert app.state.get("closes_at") == 0.0
    assert app.state.get("options") == ["a", "b"]  # token not an option
    assert "closes in" not in reply  # no deadline claimed
    # control: a well-formed closes:<h> sets the deadline
    reply2 = mp.control_miniapp('new poll "Q2?" a b closes:2',
                                context=ctx, chat=chat,
                                sender_id="u1", sender="Ada")
    apps2 = [a for a in mp.MiniAppStore(data_dir=tmp_path).load("telegram:123")
             if a.title == "Q2?"]
    assert len(apps2) == 1
    closes_at = apps2[0].state.get("closes_at") or 0.0
    assert time.time() + 3600 < closes_at < time.time() + 3 * 3600
    assert "closes in" in reply2
    assert _snap() == before


def test_1088_needs_refresh_failure_returns_false():
    before = _snap()
    bad_cadence = mp.Panel(id="p1", name="X", user_key="owner",
                           data_source="manual",
                           refresh_cadence=["daily"], last_refresh=0.0)
    assert bad_cadence.needs_refresh() is False
    bad_ts = mp.Panel(id="p2", name="X", user_key="owner",
                      data_source="manual", refresh_cadence="daily",
                      last_refresh="soon")
    assert bad_ts.needs_refresh() is False
    # controls: the guarded computation really runs
    stale = mp.Panel(id="p3", name="X", user_key="owner",
                     data_source="manual", refresh_cadence="daily",
                     last_refresh=0.0)
    assert stale.needs_refresh() is True
    fresh = mp.Panel(id="p4", name="X", user_key="owner",
                     data_source="manual", refresh_cadence="daily",
                     last_refresh=time.time())
    assert fresh.needs_refresh() is False
    assert _snap() == before


def test_1104_age_str_failure_returns_unknown():
    before = _snap()
    bad_ts = mp.Panel(id="p2", name="X", user_key="owner",
                      data_source="manual", refresh_cadence="daily",
                      last_refresh="soon")
    assert bad_ts.age_str() == "unknown"
    fresh = mp.Panel(id="p4", name="X", user_key="owner",
                     data_source="manual", last_refresh=time.time())
    assert fresh.age_str() == "just now"
    never = mp.Panel(id="p5", name="X", user_key="owner",
                     data_source="manual")
    assert never.age_str() == "never refreshed"
    assert _snap() == before


def test_1204_create_panel_fallback_observed(tmp_path):
    """FLAGGED — observed behavior only (see module docstring)."""
    before = _snap()
    store = RaisingPutStore()
    panel = mp.create_panel("Marathon", "owner", "strava", store=store)
    assert len(store.put_calls) == 1  # persistence was attempted
    assert panel.name == "Untitled panel"
    assert panel.user_key == "owner"
    assert panel.data_source == "manual"
    assert store.saved == []  # nothing was persisted
    # the consumer renders a created claim over the fallback
    store2 = RaisingPutStore()
    reply = mp.control_panel('/panel new "Marathon" strava', store=store2)
    assert reply.startswith("✅ Panel created")
    assert "Untitled panel" in reply
    assert len(store2.put_calls) == 1
    # control: a working store persists the requested panel
    real = mp.PanelStore(data_dir=tmp_path / "panels")
    made = mp.create_panel("Marathon", "owner", "strava", store=real)
    assert made.name == "Marathon" and made.data_source == "strava"
    assert [p.id for p in real.load("owner")] == [made.id]
    assert _snap() == before


def test_1221_due_refreshes_failure_returns_empty(tmp_path):
    before = _snap()
    store = RaisingLoadStore()
    assert mp.due_refreshes("owner", store=store) == []
    assert store.load_calls == 1  # the load was attempted
    # control: a real store yields its stale panel
    real = mp.PanelStore(data_dir=tmp_path / "panels")
    made = mp.create_panel("Marathon", "owner", "strava", store=real)
    assert [p.id for p in mp.due_refreshes("owner", store=real)] == [made.id]
    assert _snap() == before


def test_1327_flatten_non_dict_returns_empty():
    before = _snap()
    assert mp._flatten(["not-a-dict"]) == {}
    flat = mp._flatten({"a": {"b": 1}, "c": [1, 2]})
    assert flat.get("a.b") == 1
    assert flat.get("c") == [1, 2]
    assert flat.get("c[0]") == 1 and flat.get("c[1]") == 2
    assert _snap() == before


def test_1343_fmt_value_failure_returns_question_mark():
    before = _snap()
    bad = BadStr()
    assert mp._fmt_value(bad) == "?"
    assert bad.str_calls == 1  # formatting was attempted
    assert mp._fmt_value(2.5) == "2.5"
    assert mp._fmt_value(True) == "yes"
    assert _snap() == before


def test_1457_render_panel_failure_returns_name_only():
    before = _snap()
    data = ExplodingData()
    corrupt = mp.Panel(id="p9", name="Runs", user_key="owner",
                       data_source="strava", data=data)
    assert mp.render_panel(corrupt) == "📊 Runs"
    assert data.get_calls >= 1  # the render body attempted the data
    healthy = mp.Panel(id="p10", name="Runs", user_key="owner",
                       data_source="strava", data={"weekly_km": 42})
    rendered = mp.render_panel(healthy)
    assert "weekly km: 42" in rendered
    assert "source: strava" in rendered
    assert _snap() == before
