"""A4 — conversational reachability for the browser navigator.

The /browser run verb has driven sites since the navigator
landed; the conversational brain could only READ pages (the
browse intent). These pins hold the new door to the same
standard as the verb:

- classification: an act-on-a-site ask with a URL is a ``drive``
  intent (and the read/download readings of a URL are unchanged);
- one engine, two doors: the NL dispatch starts runs through
  the runtime's shared ``start_browser_run`` — the exact entry
  the verb uses — never a second navigator;
- report-back: the run's outcome (page, steps, stop reason,
  wall, consequential verdicts) is delivered into the ORIGIN
  chat when the run lands, on success and on failure alike;
- the live gate holds: non-operator refused, a degraded brain
  starts no drives (observe-only), a standing owner stop on the
  origin's personal platform halts the start;
- the autonomy knob is inherited exactly: default = she acts
  herself (broker untouched); knob on = the run parks and the
  verb's approve resumes it.
"""
from __future__ import annotations

import json
import re
import tempfile
import time
from contextlib import suppress
from types import SimpleNamespace as NS
from typing import Any

import pytest

from nomorals.agents.coremind import CoreMind, Intent, run_goal, understand
from nomorals.browser.jobs import BrowserJobManager

# ── classification (deterministic pass, no runtime) ──────────────────────────


class _FakeSettings:
    def resolve(self, key):  # noqa: ANN001, ANN202
        raise RuntimeError("no settings in tests")


class _FakeContext:
    def __init__(self) -> None:
        self.settings = _FakeSettings()
        self.extras: dict[str, Any] = {}
        self.memory = None
        self.router = None
        self.db = None


def _decide(text: str) -> Intent:
    return CoreMind(_FakeContext()).decide(text, allow_model=False)


def test_drive_intent_for_act_on_a_site_asks():
    intent = _decide("go to https://fakedetail.com and fill in the form")
    assert intent.kind == "drive"
    assert intent.action == "start"
    assert intent.target == "https://fakedetail.com"
    assert "fill" in intent.meta["goal"]

    intent = _decide("drive https://shop.test/checkout and buy the thing")
    assert intent.kind == "drive"
    assert intent.target == "https://shop.test/checkout"
    assert "buy the thing" in intent.meta["goal"]

    intent = _decide("log in to https://bank.test and pay the rent")
    assert intent.kind == "drive"
    assert intent.target == "https://bank.test"


def test_read_and_download_readings_of_a_url_are_unchanged():
    assert _decide("open https://example.com").kind == "browse"
    assert _decide("check out https://example.com").kind == "browse"
    assert _decide("https://example.com").kind == "browse"
    assert _decide("download https://example.com/file.zip").kind \
        == "download"


def test_drive_status_and_ask_forms():
    intent = _decide("how's the browser run going")
    assert intent.kind == "drive"
    assert intent.action == "status"

    intent = _decide("drive my store site and update the prices")
    assert intent.kind == "drive"
    assert intent.action == "ask"


def test_understand_offers_drive_beside_browse():
    kinds = {c.kind for c in understand(
        "navigate to https://x.test and submit the form")}
    assert "drive" in kinds


# ── the job record keeps the consequential evidence ──────────────────────────


def test_job_finish_retains_consequential_approvals():
    notes: list[tuple[str, str]] = []
    mgr = BrowserJobManager(
        notifier=lambda m, u: notes.append((m, u)))

    def runner(job):  # noqa: ANN001, ANN202
        return {"steps": 2, "stopped_reason": "goal reached",
                "final_url": "https://x.test/done",
                "consequential_approvals": [
                    {"kind": "submit", "url": "https://x.test/checkout",
                     "verdict": "acted", "answered_by": "devon"}]}

    job, duplicate = mgr.start("https://x.test", "do it", runner)
    assert not duplicate
    assert job.wait(10)
    assert job.status == "done"
    assert job.approvals[0]["kind"] == "submit"
    assert job.approvals[0]["verdict"] == "acted"
    status = job.status_text()
    assert "approvals:" in status and "submit" in status \
        and "acted" in status
    done_notes = [m for m, _u in notes if "done" in m]
    assert done_notes and "submit acted" in done_notes[0]


# ── the conversational gates (stub runtime) ──────────────────────────────────


class _StubRuntime:
    """Just the runtime surface the drive dispatch consults."""

    def __init__(self, *, operator: bool = True,
                 stopped: bool = False) -> None:
        self._operator = operator
        self._stopped = stopped
        self.started: list[tuple[str, str]] = []
        self.sent_long: list[tuple[str, str]] = []

    def _is_operator(self, message: Any) -> bool:
        return self._operator

    def _chatops_personal_account(self, platform: str):  # noqa: ANN202
        registry = NS(owner_stopped=lambda p: self._stopped)
        return {"registry": registry}, NS(platform=platform)

    def start_browser_run(self, url: str, goal: str, *,
                          chat_key: str = "", force: bool = False):  # noqa: ANN202
        self.started.append((url, goal))
        mgr = BrowserJobManager()

        def runner(job):  # noqa: ANN001, ANN202
            return {"steps": 3, "stopped_reason": "goal reached",
                    "final_url": "https://shop.test/done",
                    "consequential_approvals": [
                        {"kind": "submit",
                         "url": "https://shop.test/checkout",
                         "verdict": "acted", "answered_by": "devon"}]}

        job, _dup = mgr.start(url, goal, runner)
        return job, False

    def _browser_run_status(self, arg: str) -> str:
        return f"STATUS-TEXT[{arg}]"

    @staticmethod
    def _ref_from_key(key: str):  # noqa: ANN202
        platform = key.split(":", 1)[0] if ":" in key else "telegram"
        return NS(platform=platform)

    def _send_long(self, platform: str, ref: Any, text: str) -> int:
        self.sent_long.append((platform, text))
        return 1


def _real_context(tmp_home: str):  # noqa: ANN202
    from nomorals.agents.context import build_context
    from nomorals.core.config import load_settings

    settings = load_settings(overrides={"home": tmp_home})
    return build_context(settings, with_executor=False,
                         with_tools=False)


def _drive_intent(action: str = "start") -> Intent:
    return Intent("drive", 0.88, target="https://shop.test/checkout",
                  action=action, route="browser", why="test",
                  meta={"goal": "buy the thing"})


def _msg(platform: str = "whatsapp") -> Any:
    return NS(chat=NS(platform=platform, kind="dm"))


def test_non_operator_cannot_start_a_drive():
    with tempfile.TemporaryDirectory(prefix="nm-navreach-") as tmp:
        context = _real_context(tmp)
        try:
            runtime = _StubRuntime(operator=False)
            mind = CoreMind(context, runtime=runtime)
            reply = mind._dispatch(
                _drive_intent(), "telegram:999", _msg())
            assert "owner-only" in reply
            assert runtime.started == []
        finally:
            context.close()


def test_degraded_brain_starts_no_drive(monkeypatch):
    import nomorals.agents.notifier as notifier_mod

    monkeypatch.setattr(
        notifier_mod, "resolve_brain_gate",
        lambda ctx: NS(degraded_now=True))
    with tempfile.TemporaryDirectory(prefix="nm-navreach-") as tmp:
        context = _real_context(tmp)
        try:
            runtime = _StubRuntime(operator=True)
            mind = CoreMind(context, runtime=runtime)
            reply = mind._dispatch(
                _drive_intent(), "telegram:111", _msg())
            assert "degraded" in reply and "observe-only" in reply
            assert runtime.started == []
        finally:
            context.close()


def test_standing_owner_stop_halts_the_start():
    with tempfile.TemporaryDirectory(prefix="nm-navreach-") as tmp:
        context = _real_context(tmp)
        try:
            runtime = _StubRuntime(operator=True, stopped=True)
            mind = CoreMind(context, runtime=runtime)
            reply = mind._dispatch(
                _drive_intent(), "whatsapp:111", _msg("whatsapp"))
            assert "standing stop" in reply
            assert runtime.started == []
        finally:
            context.close()


def test_drive_status_routes_to_the_run_registry():
    with tempfile.TemporaryDirectory(prefix="nm-navreach-") as tmp:
        context = _real_context(tmp)
        try:
            runtime = _StubRuntime(operator=True)
            mind = CoreMind(context, runtime=runtime)
            intent = Intent("drive", 0.85, action="status",
                            route="browser", why="test")
            reply = mind._dispatch(intent, "telegram:111", _msg())
            assert reply == "STATUS-TEXT[]"
        finally:
            context.close()


def test_game_intent_never_fires_from_words_inside_a_url():
    # The phantom that hijacked drive asks into multis: the game
    # alias "shop" and the economy verb "buy" read out of a URL.
    intent = _decide("drive https://shop.test/checkout and buy the thing")
    assert intent.kind == "drive"
    # Prose game asks are untouched by the URL guard.
    assert _decide("let's play shop").kind == "game"


def test_report_back_lands_in_the_origin_chat_with_evidence():
    with tempfile.TemporaryDirectory(prefix="nm-navreach-") as tmp:
        context = _real_context(tmp)
        try:
            runtime = _StubRuntime(operator=True)
            mind = CoreMind(context, runtime=runtime)
            reply = mind._dispatch(
                _drive_intent(), "whatsapp:111", _msg("whatsapp"))
            assert "driving https://shop.test/checkout" in reply
            assert runtime.started == [
                ("https://shop.test/checkout", "buy the thing")]
            deadline = time.time() + 15
            report = ""
            while time.time() < deadline:
                for _platform, text in runtime.sent_long:
                    if "drive done" in text:
                        report = text
                        break
                if report:
                    break
                time.sleep(0.05)
            assert report, f"no report delivered: {runtime.sent_long!r}"
            assert "page: https://shop.test/done" in report
            assert "goal: buy the thing" in report
            assert "consequential: submit" in report
            assert "acted" in report
        finally:
            context.close()


def test_run_goal_console_answers_honestly():
    with tempfile.TemporaryDirectory(prefix="nm-navreach-") as tmp:
        context = _real_context(tmp)
        try:
            out = run_goal(
                context, "drive https://shop.test and fill the form")
            assert "/browser run" in out
        finally:
            context.close()


def test_intent_prompts_know_the_drive_kind():
    from nomorals.llm.prompts import render_prompt, system_prompt_for

    assert "drive" in system_prompt_for("intent")
    detail = render_prompt("intent_detail", games="hangman")
    assert "drive" in detail
    assert "operate a site" in detail


# ── end to end on the real runtime (the verb's own harness) ─────────────────


_SUBMIT_JSON = json.dumps({
    "action": {"kind": "submit", "target": "e2",
               "label": "Place order", "payload": {}},
    "reason": "the form is filled", "confidence": 0.9})


def _submit_brain(*replies: str) -> Any:
    queue = list(replies)

    class _Brain:
        def complete(self, prompt: str, **_kw: Any) -> Any:
            text = queue.pop(0) if queue else '{"action": null}'
            return NS(ok=True, text=text)

    return _Brain()


class _CheckoutTab:
    """RenderedTab-shaped checkout page with one Place-order button."""

    url = "https://shop.test/checkout"

    def __init__(self) -> None:
        self.clicks: list[str] = []

    def navigate(self, url: str) -> dict[str, Any]:
        return {"ok": True}

    def html(self) -> dict[str, Any]:
        return {"url": self.url, "title": "Checkout",
                "html": "<html><head><title>Checkout</title></head>"
                        "<body>your order is ready</body></html>"}

    def snapshot(self) -> dict[str, Any]:
        return {"snapshot": '- button "Place order" [ref=e2]'}

    def click(self, target: str) -> dict[str, Any]:
        self.clicks.append(target)
        return {"ok": True}

    def cookies(self) -> list[dict[str, Any]]:
        return []

    def storage_state(self) -> dict[str, Any]:
        return {"cookies": [], "origins": []}


def _runtime_harness(monkeypatch, *, require_approval: bool = False):  # noqa: ANN001, ANN202
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.base import (
        ChatAdapter,
        ChatKind,
        ChatMessage,
        ChatRef,
        SendResult,
    )
    from nomorals.social.chat.gateway import ChatGateway

    monkeypatch.setattr("nomorals.llm.brain.brain_for",
                        lambda ctx: _submit_brain(_SUBMIT_JSON))

    class _Adapter(ChatAdapter):
        def __init__(self) -> None:
            super().__init__(media_dir="/tmp/nm-test-media")
            self.name = "telegram"
            self.sent: list[tuple[str, str]] = []

        def run(self, handler: Any) -> None:
            pass

        def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
                 buttons: Any = None, parse_mode: str = "") -> SendResult:
            self.sent.append((chat.chat_id, text))
            return SendResult(ok=True, platform=self.name,
                              message_id=f"m{len(self.sent)}")

    class _Service:
        def __init__(self, tab: Any) -> None:
            self._tab = tab

        def open_rendered_tab(self, session_name: str,
                              url: str = "") -> Any:
            return self._tab

        def solved_session_store(self) -> Any:
            return None

    def _msg(text: str, owner: bool) -> Any:
        return ChatMessage(
            chat=ChatRef(platform="telegram",
                         chat_id="111" if owner else "999",
                         kind=ChatKind.DM),
            incoming=True, text=text,
            sender="owner" if owner else "stranger",
            sender_id="111" if owner else "999",
            meta={"is_owner": True} if owner else {})

    tmp = tempfile.TemporaryDirectory(prefix="nm-navreach-e2e-")
    overrides = {
        "home": tmp.name,
        "partner.platforms": "telegram",
        "partner.owner_chats": "telegram:111",
        "partner.enabled": "false",
    }
    if require_approval:
        overrides["partner.browser_require_approval"] = "true"
    settings = load_settings(overrides=overrides)
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    adapter = _Adapter()
    gateway = ChatGateway({"telegram": adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    tab = _CheckoutTab()
    runtime._browser_service_for_run_job = (  # type: ignore[method-assign]
        lambda job_id: _Service(tab))
    return tmp, context, runtime, adapter, tab, _msg


def _wait_for_send(adapter: Any, needle: str, timeout: float = 20.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for _chat, text in adapter.sent:
            if needle in text:
                return text
        time.sleep(0.05)
    raise AssertionError(
        f"no send containing {needle!r}; got: {adapter.sent!r}")


def test_nl_drive_acts_itself_and_reports_back(monkeypatch):
    tmp, context, runtime, adapter, tab, _msg = _runtime_harness(
        monkeypatch)
    try:
        calls: list[tuple[str, str]] = []
        real_start = runtime.start_browser_run

        def spy(url: str, goal: str, **kw: Any):  # noqa: ANN202
            calls.append((url, goal))
            return real_start(url, goal, **kw)

        runtime.start_browser_run = spy  # type: ignore[method-assign]
        mind = CoreMind(context, runtime=runtime)
        reply = mind.handle(
            "drive https://shop.test/checkout and buy the thing",
            message=_msg("drive it", True), chat_key="telegram:111")
        assert reply is not None and "driving" in reply
        match = re.search(r"br-[0-9a-f]{12}", reply)
        assert match, f"no job id in reply: {reply!r}"
        # The SAME starter the verb uses was the door's entry.
        assert calls == [("https://shop.test/checkout",
                          "buy the thing")]
        report = _wait_for_send(adapter, "drive done")
        assert "consequential: submit" in report
        assert "acted" in report
        assert tab.clicks == ["ref:e2"]
        # Default autonomy: the approval broker was never engaged.
        broker = runtime._browser_approval_broker()
        assert broker.pending() == []
        assert broker.history == []
        # And the verb door goes through the same starter too.
        verb_reply = runtime._control_browser(
            "run https://shop.test/checkout buy another one",
            chat_key="telegram:111",
            message=_msg("/browser run", True))
        verb_match = re.search(r"br-[0-9a-f]{12}", verb_reply)
        assert verb_match, f"no job id in verb reply: {verb_reply!r}"
        assert len(calls) == 2
        job = runtime._browser_run_jobs().get(verb_match.group(0))
        assert job is not None and job.wait(15)
    finally:
        with suppress(Exception):
            runtime.stop()
        with suppress(Exception):
            context.close()
        tmp.cleanup()


def test_nl_drive_with_the_knob_on_parks_and_the_verb_resumes(
        monkeypatch):
    tmp, context, runtime, adapter, tab, _msg = _runtime_harness(
        monkeypatch, require_approval=True)
    try:
        mind = CoreMind(context, runtime=runtime)
        reply = mind.handle(
            "drive https://shop.test/checkout and buy the thing",
            message=_msg("drive it", True), chat_key="telegram:111")
        assert reply is not None and "driving" in reply
        broker = runtime._browser_approval_broker()
        deadline = time.time() + 15
        while time.time() < deadline and not broker.pending():
            time.sleep(0.02)
        pending = broker.pending()
        assert pending, "the NL-started run never parked for approval"
        assert tab.clicks == []
        answered = runtime._control_browser(
            f"approve {pending[0]['request_id']}",
            chat_key="telegram:111",
            message=_msg("/browser approve", True))
        assert "approved" in answered
        report = _wait_for_send(adapter, "drive done")
        assert "approved" in report
        assert tab.clicks == ["ref:e2"]
    finally:
        with suppress(Exception):
            runtime.stop()
        with suppress(Exception):
            context.close()
        tmp.cleanup()


def test_nl_drive_refused_while_the_brain_is_degraded(monkeypatch):
    tmp, context, runtime, adapter, tab, _msg = _runtime_harness(
        monkeypatch)
    try:
        runtime._get_brain_gate().observe_outcome(True)
        mind = CoreMind(context, runtime=runtime)
        reply = mind.handle(
            "drive https://shop.test/checkout and buy the thing",
            message=_msg("drive it", True), chat_key="telegram:111")
        assert reply is not None
        assert "degraded" in reply and "observe-only" in reply
        assert runtime._browser_run_jobs().list_recent(5) == []
        assert tab.clicks == []
    finally:
        with suppress(Exception):
            runtime.stop()
        with suppress(Exception):
            context.close()
        tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
