"""Wiring tests: takeover escalation -> real Telegram sender; /browser takeover verb.

The takeover library (challenge ladder, grants, stream) exists, but until
this wiring the escalation message had NO caller sending it to the
owner, and /browser had no takeover verb. These tests pin the seam:

- a wall escalation sends a Telegram message to the OWNER chat with the
  attempt receipt + takeover link (webview + "open in Chrome" hint);
- a non-owner chat never receives it, and the verb refuses non-owners;
- /browser takeover shows status / opens a takeover session.
"""
from __future__ import annotations

import tempfile
from typing import Any

import pytest

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.browser.challenge import ChallengeSolver
from nomorals.browser.takeover import TakeoverConfig, TakeoverManager
from nomorals.core.config import load_settings
from nomorals.social.chat.base import ChatAdapter, ChatKind, ChatMessage, ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway


CF_HTML = (
    "<html><head><title>Just a moment...</title></head><body>"
    "Verify you are human — checking your browser</body></html>"
)
CLEAN_HTML = "<html><head><title>Home</title></head><body>hello</body></html>"


class FakeTab:
    """A tab with canned HTML, adapter-shaped (html() -> str)."""

    def __init__(self, html: str = "", url: str = "https://example.test/") -> None:
        self._html = html
        self._url = url
        self.clicks: list[tuple[float, float]] = []
        self.typed: list[str] = []
        self.keys: list[str] = []
        self.reloads = 0

    @property
    def url(self) -> str:
        return self._url

    @property
    def title(self) -> str:
        import re

        match = re.search(r"<title[^>]*>(.*?)</title>", self._html,
                          re.DOTALL | re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
        return "Example"

    def html(self) -> str:
        return self._html

    def reload(self) -> None:
        self.reloads += 1

    def screenshot_png(self) -> bytes:
        return b"\x89PNG\r\n\x1a\nfake-png"

    def click_norm(self, x: float, y: float) -> None:
        self.clicks.append((x, y))

    def type_text(self, text: str) -> None:
        self.typed.append(text)

    def press_key(self, key: str) -> None:
        self.keys.append(key)

    def scroll_norm(self, dx: float, dy: float) -> None:
        pass


class FakeRenderedTab:
    """A RenderedTab-shaped tab: html() -> dict, screenshot_bytes()."""

    def __init__(self, html: str = "") -> None:
        self._html = html
        self.url = "https://example.test/"
        self.title = "Example"
        self.clicks: list[tuple[float, float]] = []

    def html(self) -> dict[str, Any]:
        return {"url": self.url, "title": self.title,
                "chars": len(self._html), "html": self._html,
                "truncated": False}

    def reload(self) -> None:
        pass

    def screenshot_bytes(self) -> bytes:
        return b"\x89PNG\r\n\x1a\nfake-png"

    def click_norm(self, x: float, y: float) -> None:
        self.clicks.append((x, y))

    def type_text(self, text: str) -> None:
        pass

    def press_key(self, key: str) -> None:
        pass

    def scroll_norm(self, dx: float, dy: float) -> None:
        pass


class FakeBrowserService:
    def __init__(self, tab: Any) -> None:
        self._tab = tab
        self.opened: list[tuple[str, str]] = []

    def open_rendered_tab(self, session_name: str, url: str = "") -> Any:
        self.opened.append((session_name, url))
        return self._tab


class FakeTelegramAdapter(ChatAdapter):
    def __init__(self, name: str = "telegram") -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[tuple[str, str]] = []

    def run(self, handler: Any) -> None:
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, parse_mode: str = "") -> SendResult:
        self.sent.append((chat.chat_id, text))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


def _make_runtime() -> tuple[PartnerRuntime, FakeTelegramAdapter,
                             tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-takeover-wiring-")
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "telegram",
        "partner.owner_chats": "telegram:111",
        "partner.enabled": "false",
    })
    context = build_context(settings, with_executor=False, with_tools=False)
    adapter = FakeTelegramAdapter("telegram")
    gateway = ChatGateway({"telegram": adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    return runtime, adapter, tmp


def _owner_msg() -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform="telegram", chat_id="111", kind=ChatKind.DM),
        incoming=True, text="/browser takeover", sender="owner",
        sender_id="111", meta={"is_owner": True})


def _stranger_msg() -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform="telegram", chat_id="999", kind=ChatKind.DM),
        incoming=True, text="/browser takeover", sender="stranger",
        sender_id="999", meta={})


def _teardown(runtime: PartnerRuntime, tmp: tempfile.TemporaryDirectory) -> None:
    try:
        runtime.stop()
    except Exception:
        pass
    try:
        runtime.context.close()
    except Exception:
        pass
    tmp.cleanup()


# -- (a) escalation -> owner Telegram with attempts + link -------------------

def test_escalation_sends_telegram_to_owner_with_attempts_and_link():
    runtime, adapter, tmp = _make_runtime()
    try:
        notifier = runtime._takeover_notifier()
        mgr = TakeoverManager(TakeoverConfig(),
                              data_dir=f"{tmp.name}/takeover")
        solver = ChallengeSolver(wait_s=0, takeover_manager=mgr,
                                 notifier=notifier)
        outcome = solver.solve(tab=FakeTab(CF_HTML),
                               resume_state={"step": 1},
                               owner_chat="telegram:111",
                               task_label="login")
        assert outcome.via == "takeover_pending"
        assert outcome.takeover_url
        assert len(adapter.sent) == 1
        chat_id, text = adapter.sent[0]
        assert chat_id == "111"
        # Attempt receipt: what Devon already tried.
        assert "retry" in text
        # The takeover link itself.
        assert outcome.takeover_url in text
        # Webview + "open in Chrome" hint.
        assert "Chrome" in text
    finally:
        _teardown(runtime, tmp)


def test_escalation_needs_public_base_warns_honestly():
    """No NM_TAKEOVER_BASE_URL: the link is host-network-only — say so."""
    runtime, adapter, tmp = _make_runtime()
    try:
        notifier = runtime._takeover_notifier()
        mgr = TakeoverManager(TakeoverConfig(),
                              data_dir=f"{tmp.name}/takeover")
        solver = ChallengeSolver(wait_s=0, takeover_manager=mgr,
                                 notifier=notifier)
        outcome = solver.solve(tab=FakeTab(CF_HTML),
                               resume_state={"step": 1},
                               owner_chat="telegram:111")
        assert outcome.takeover_grant is not None
        assert outcome.takeover_grant.needs_public_base is True
        assert len(adapter.sent) == 1
        assert "NM_TAKEOVER_BASE_URL" in adapter.sent[0][1]
    finally:
        _teardown(runtime, tmp)


# -- (b) non-owner never receives it ------------------------------------------

def test_takeover_verb_refuses_non_owner_and_sends_nothing():
    runtime, adapter, tmp = _make_runtime()
    try:
        reply = runtime._control_browser(
            "takeover https://example.test/", chat_key="telegram:999",
            message=_stranger_msg())
        assert "owner-only" in reply.lower()
        assert adapter.sent == []
        assert runtime._takeover_manager().active_grants() == []
    finally:
        _teardown(runtime, tmp)


def test_escalation_sends_only_to_owner_targets():
    runtime, adapter, tmp = _make_runtime()
    try:
        notifier = runtime._takeover_notifier()
        mgr = TakeoverManager(TakeoverConfig(),
                              data_dir=f"{tmp.name}/takeover")
        solver = ChallengeSolver(wait_s=0, takeover_manager=mgr,
                                 notifier=notifier)
        solver.solve(tab=FakeTab(CF_HTML), resume_state={"step": 1},
                     owner_chat="telegram:111")
        assert adapter.sent
        assert {chat_id for chat_id, _ in adapter.sent} == {"111"}
    finally:
        _teardown(runtime, tmp)


# -- (c) /browser takeover verb -----------------------------------------------

def test_takeover_verb_status_empty_and_with_grant():
    runtime, adapter, tmp = _make_runtime()
    try:
        reply = runtime._control_browser(
            "takeover", chat_key="telegram:111", message=_owner_msg())
        assert "no active takeover" in reply.lower()
        # Issue a grant through the runtime's own manager, then status
        # shows it — without leaking the token-bearing link.
        mgr = runtime._takeover_manager()
        grant = mgr.request_takeover(
            session_id="s1", url="https://example.test/",
            task_label="login", signal_kind="cloudflare-challenge",
            attempts=[], resume_state={"step": 2}, tab=FakeTab(CF_HTML),
            owner_chat="telegram:111")
        reply = runtime._control_browser(
            "takeover", chat_key="telegram:111", message=_owner_msg())
        assert "login" in reply
        assert "s1" in reply
        assert grant.url not in reply
        assert grant.token not in reply
    finally:
        _teardown(runtime, tmp)


def test_takeover_verb_clean_page_reports_no_wall():
    runtime, adapter, tmp = _make_runtime()
    try:
        runtime._browser_service_for_takeover = (  # type: ignore[method-assign]
            lambda: FakeBrowserService(FakeTab(CLEAN_HTML)))
        reply = runtime._control_browser(
            "takeover https://example.test/", chat_key="telegram:111",
            message=_owner_msg())
        assert "no wall" in reply.lower()
        assert adapter.sent == []
    finally:
        _teardown(runtime, tmp)


def test_takeover_verb_walled_page_opens_grant_and_notifies():
    runtime, adapter, tmp = _make_runtime()
    try:
        runtime._browser_service_for_takeover = (  # type: ignore[method-assign]
            lambda: FakeBrowserService(FakeTab(CF_HTML)))
        reply = runtime._control_browser(
            "takeover https://example.test/", chat_key="telegram:111",
            message=_owner_msg())
        assert "takeover open" in reply.lower()
        assert "takeover/" in reply  # the link is in the reply
        assert len(adapter.sent) == 1  # and the escalation went to Telegram
        assert adapter.sent[0][0] == "111"
        assert "retry" in adapter.sent[0][1]
        assert runtime._takeover_manager().active_grants()
    finally:
        _teardown(runtime, tmp)


# -- adapter: RenderedTab -> TakeoverTabAdapter --------------------------------

def test_rendered_tab_adapter_bridges_dict_html_and_screenshot_bytes():
    from nomorals.browser.service import RenderedTabTakeoverAdapter

    tab = FakeRenderedTab(CF_HTML)
    adapter = RenderedTabTakeoverAdapter(tab)
    assert adapter.url == "https://example.test/"
    assert adapter.title == "Example"
    assert isinstance(adapter.html(), str)
    assert "checking your browser" in adapter.html()
    assert adapter.screenshot_png().startswith(b"\x89PNG")
    adapter.click_norm(0.5, 0.5)
    assert tab.clicks == [(0.5, 0.5)]
    adapter.reload()  # must not raise
