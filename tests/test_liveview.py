"""Tests for the live agent window (build-map #33). All offline."""
import pytest

from nomorals.browser.liveview import LiveView, LIVE_VIEW_MAX_SHOTS
from nomorals.social.chat.base import ChatRef, MediaRef, SendResult


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class FakeAdapter:
    """Adapter with working edit_media."""

    name = "fake"

    def __init__(self):
        self.sends = []       # (kind, caption)
        self.edits = []       # (message_id, caption)
        self.raise_on = None  # method name to blow up

    def _maybe_raise(self, method):
        if self.raise_on == method:
            raise RuntimeError("boom")

    def send(self, chat, text, **kw):
        self._maybe_raise("send")
        self.sends.append(("text", text))
        return SendResult(ok=True, platform=self.name, message_id="11")

    def send_media(self, chat, media, *, caption=""):
        self._maybe_raise("send_media")
        self.sends.append(("media", caption, media.path))
        return SendResult(ok=True, platform=self.name, message_id="22")

    def edit_media(self, chat, message_id, media, *, caption=""):
        self._maybe_raise("edit_media")
        self.edits.append((message_id, caption, media.path))
        return SendResult(ok=True, platform=self.name, message_id=message_id)


class NoEditAdapter(FakeAdapter):
    """Adapter without edit_media — falls back to throttled new photos."""

    def edit_media(self, chat, message_id, media, *, caption=""):
        return SendResult(ok=False, platform=self.name,
                          error="edit_media not supported")


@pytest.fixture()
def png(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(b"\x89PNG fake")
    return p


@pytest.fixture()
def chat():
    return ChatRef(platform="fake", chat_id="1", kind="dm")


def make_view(png, clock=None, **kw):
    clock = clock or FakeClock()
    kw.setdefault("min_interval_s", 8.0)
    return LiveView(lambda: png, profile="workstation",
                    clock=clock, **kw), clock


def test_start_sends_initial_frame(png, chat):
    adapter = FakeAdapter()
    view, _ = make_view(png)
    view.start(chat, adapter, task_label="buying shoes")
    assert len(adapter.sends) == 1
    kind, caption, path = adapter.sends[0]
    assert kind == "media"
    assert "buying shoes" in caption
    assert path == str(png)
    assert view._message_id == "22"


def test_update_prefers_edit_media(png, chat):
    adapter = FakeAdapter()
    view, clock = make_view(png)
    view.start(chat, adapter, task_label="task")
    clock.advance(10)
    view.update("clicked 'Buy'")
    assert len(adapter.edits) == 1
    msg_id, caption, _ = adapter.edits[0]
    assert msg_id == "22"  # the start message, updated in place
    assert "clicked 'Buy'" in caption
    # no extra photo message sent
    assert len(adapter.sends) == 1


def test_update_throttles_rapid_actions(png, chat):
    adapter = FakeAdapter()
    view, clock = make_view(png)
    view.start(chat, adapter, task_label="task")
    view.update("clicked 'a'")   # too soon — coalesced
    view.update("clicked 'b'")
    assert adapter.edits == []
    clock.advance(9)
    view.update("clicked 'c'")
    assert len(adapter.edits) == 1
    # latest action wins
    assert "clicked 'c'" in adapter.edits[0][1]


def test_fallback_to_new_photo_without_edit_media(png, chat):
    adapter = NoEditAdapter()
    view, clock = make_view(png)
    view.start(chat, adapter, task_label="task")
    clock.advance(10)
    view.update("typed query")
    assert adapter.edits == []  # edit unsupported
    assert len(adapter.sends) == 2  # start + new photo
    assert adapter.sends[1][0] == "media"


def test_finish_sends_summary(png, chat):
    adapter = FakeAdapter()
    view, clock = make_view(png)
    view.start(chat, adapter, task_label="task")
    view.finish("bought 2 items for $40")
    assert len(adapter.edits) == 1
    assert "bought 2 items" in adapter.edits[0][1]
    assert view._active is False


def test_fail_honest_frame(png, chat):
    adapter = FakeAdapter()
    view, _ = make_view(png)
    view.start(chat, adapter, task_label="task")
    view.fail("login page changed")
    assert len(adapter.edits) == 1
    assert "login page changed" in adapter.edits[0][1]
    assert view._active is False


def test_never_raises_on_adapter_errors(png, chat):
    adapter = FakeAdapter()
    adapter.raise_on = "send_media"
    view, clock = make_view(png)
    view.start(chat, adapter, task_label="task")  # must not raise
    adapter.raise_on = "edit_media"
    clock.advance(10)
    view.update("clicked x")  # must not raise
    view.finish("done")       # must not raise
    view.fail("bad")          # must not raise (already detached)


def test_cap_enforced(png, chat):
    adapter = FakeAdapter()
    view, clock = make_view(png, max_shots=2, min_interval_s=0)
    view.start(chat, adapter, task_label="task")  # shot 1
    view.update("a")  # shot 2
    view.update("b")  # capped — no send
    assert len(adapter.edits) == 1
    # finish frame still goes through
    view.finish("done")
    assert len(adapter.edits) == 2


def test_shot_failure_sends_text_only(chat):
    adapter = FakeAdapter()
    view = LiveView(lambda: None, profile="workstation",
                    clock=FakeClock())
    view.start(chat, adapter, task_label="task")
    assert adapter.sends[0][0] == "text"
    assert "task" in adapter.sends[0][1]


def test_attach_opt_out_returns_none(chat):
    adapter = FakeAdapter()

    class FakeTab:
        on_action = None

    tab = FakeTab()
    view = LiveView.attach(tab, chat, adapter, task_label="t",
                           live_view=False)
    assert view is None
    assert tab.on_action is None


def test_attach_hooks_tab_actions(png, chat):
    adapter = FakeAdapter()
    clock = FakeClock()

    class FakeTab:
        on_action = None

        def screenshot(self):
            return {"path": str(png)}

    tab = FakeTab()
    view = LiveView.attach(tab, chat, adapter, task_label="t",
                           profile="workstation", min_interval_s=0,
                           clock=clock)
    assert view is not None
    # bound methods compare equal when same instance+function
    assert tab.on_action == view.update
    assert tab.on_action.__self__ is view
    # simulate a browser action firing the hook
    tab.on_action("opened https://example.com")
    assert len(adapter.edits) == 1
    assert "opened https://example.com" in adapter.edits[0][1]
    view.detach()
    assert tab.on_action is None


def test_termux_throttle_is_30s(png, chat):
    from nomorals.browser import liveview as lv
    adapter = FakeAdapter()
    view = LiveView(lambda: png, profile="termux", clock=FakeClock())
    assert view._interval == 30.0
    view2 = LiveView(lambda: png, profile="laptop", clock=FakeClock())
    assert view2._interval == 8.0
    assert lv.LIVE_VIEW_MAX_SHOTS == 25


def test_tab_hook_wired_in_service():
    """Tab.navigate fires on_action (the real service class, mocked session)."""
    from nomorals.browser.service import Tab

    class FakeSession:
        url = "https://example.com"
        title = "Example"

        def open(self, url):
            return {"ok": True}

    tab = Tab("t1", "s1", FakeSession())
    seen = []
    tab.on_action = seen.append
    tab.navigate("https://example.com")
    assert seen and "opened https://example.com" in seen[0]
    # default is None (opt-in)
    tab2 = Tab("t2", "s1", FakeSession())
    assert tab2.on_action is None
