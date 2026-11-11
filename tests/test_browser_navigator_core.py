"""Web navigator core: self-provisioning, vision wiring, approvals.

Pins for the three seams that completed the model-driven navigator
(webdrive census A1–A3):

* A1 — the browser engine self-provisions (Playwright + Chromium),
  mirroring yt-dlp's ``ensure_ytdlp``: detect → install → verify,
  one bounded attempt, honest journaled failure, no thrash on
  constrained profiles, and NEVER a real install under tests (the
  command runner and the presence probes are fakes throughout).
* A2 — the perception escalator is plugged into the loop's observe
  step with a live ``see`` (fake here): it fires on its own
  conditions, its mark answer becomes the step's action, and its
  verdict reaches the decider when no mark is named. The desktop
  decider likewise receives the real PNG via ``vision_fn``.
* A3 — a consequential proposal is surfaced to the loop and the
  approver PARKS the run for the owner's word: nothing executes
  before approval, approve resumes and performs it, deny/timeout
  stops the run with the verdict recorded.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from contextlib import suppress
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from nomorals.browser import provision
from nomorals.browser.agent import (
    Action,
    AgentLoop,
    Decision,
    Observation,
    Page,
)
from nomorals.browser.approvals import ApprovalBroker
from nomorals.browser.llm_decider import (
    ApprovalSurfacingDecider,
    LLMDecider,
)
from nomorals.browser.vision_marks import PerceptionEscalator

# ── A1: self-provisioning ---------------------------------------------------


def _no_pytest(monkeypatch: Any) -> None:
    monkeypatch.delitem(sys.modules, "pytest", raising=False)


def test_ensure_playwright_already_usable_installs_nothing(monkeypatch):
    monkeypatch.setattr(provision, "playwright_usable", lambda: True)
    monkeypatch.setattr(provision, "_PROVISION_ATTEMPTED", False)
    run = mock.Mock()
    assert provision.ensure_playwright(runner=run) is True
    run.assert_not_called()


def test_ensure_playwright_never_installs_under_tests(monkeypatch):
    monkeypatch.setattr(provision, "playwright_usable", lambda: False)
    monkeypatch.setattr(provision, "_PROVISION_ATTEMPTED", False)
    run = mock.Mock()
    # pytest is in sys.modules here: provisioning must not mutate
    # the test environment, and must report honestly.
    assert provision.ensure_playwright(runner=run) is False
    run.assert_not_called()


def test_ensure_playwright_installs_verifies_and_attempts_once(monkeypatch):
    _no_pytest(monkeypatch)
    state = {"ok": False}
    monkeypatch.setattr(provision, "playwright_usable",
                        lambda: state["ok"])
    monkeypatch.setattr(provision, "_PROVISION_ATTEMPTED", False)
    commands: list[list[str]] = []

    def _fake_run(cmd, _timeout):
        commands.append(list(cmd))
        if len(commands) == 2:
            state["ok"] = True
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    assert provision.ensure_playwright(runner=_fake_run) is True
    assert len(commands) == 2
    assert commands[0][1:] == ["-m", "pip", "install", "playwright"]
    assert commands[1][1:] == ["-m", "playwright", "install", "chromium"]
    # One bounded attempt per process: a later failure does not loop.
    state["ok"] = False
    assert provision.ensure_playwright(runner=_fake_run) is False
    assert len(commands) == 2


def test_ensure_playwright_failed_install_is_honest_and_journaled(monkeypatch):
    _no_pytest(monkeypatch)
    monkeypatch.setattr(provision, "playwright_usable", lambda: False)
    monkeypatch.setattr(provision, "_PROVISION_ATTEMPTED", False)
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "nomorals.browser.failures.record_failure",
        lambda site, exc: recorded.append((site, str(exc))))
    run = mock.Mock(return_value=SimpleNamespace(
        returncode=1, stderr="no network", stdout=""))
    assert provision.ensure_playwright(runner=run) is False
    assert run.call_count == 1  # pip failed: the browser step never runs
    assert recorded and recorded[0][0] == "browser.provision.playwright"


def test_ensure_playwright_constrained_profile_refuses_to_thrash(monkeypatch):
    _no_pytest(monkeypatch)
    monkeypatch.setattr(provision, "playwright_usable", lambda: False)
    monkeypatch.setattr(provision, "_PROVISION_ATTEMPTED", False)
    recorded: list[str] = []
    monkeypatch.setattr(
        "nomorals.browser.failures.record_failure",
        lambda site, exc: recorded.append(site))
    run = mock.Mock()
    assert provision.ensure_playwright(
        runner=run, profile_kind="termux") is False
    run.assert_not_called()
    assert recorded == ["browser.provision.profile"]


def test_chromium_present_detects_a_real_build_shape(tmp_path):
    root = tmp_path / "ms-playwright"
    exe = root / "chromium-1243" / "chrome-linux" / "chrome"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"#!/bin/sh\n")
    assert provision.chromium_present(root) is True
    assert provision.chromium_present(tmp_path / "nope") is False
    other = tmp_path / "other"
    (other / "firefox-1").mkdir(parents=True)
    assert provision.chromium_present(other) is False


def test_service_gate_provisions_before_the_hint(monkeypatch):
    from nomorals.browser import service

    calls: list[bool] = []
    monkeypatch.setattr(
        "nomorals.browser.provision.playwright_usable", lambda: False)
    monkeypatch.setattr(
        "nomorals.browser.provision.ensure_playwright",
        lambda: calls.append(True) or False)
    with pytest.raises(service.BrowserError) as err:
        service._require_playwright_sync()
    assert calls == [True]  # provisioning was attempted first
    assert "playwright" in str(err.value)


def test_cdp_driver_provisions_before_the_honest_error(monkeypatch):
    from nomorals.browser.agent import CdpDriver

    monkeypatch.setattr(
        "nomorals.browser.provision.ensure_playwright", lambda: False)
    with pytest.raises(RuntimeError) as err:
        CdpDriver()
    assert "self-provisioning" in str(err.value)


# ── A2: vision escalation in the loop ----------------------------------------

CANVAS_HTML = (
    "<html><head><title>Player</title></head><body>"
    "<canvas id='stage'></canvas>watch the show</body></html>")


class FakeDriver:
    """BrowserDriver seam over one fixed page; records every act."""

    def __init__(self, page: Page) -> None:
        self.page = page
        self.acted: list[Action] = []

    def fetch(self, url: str) -> Page:
        return self.page

    def act(self, action: Action, page: Page) -> Page:
        self.acted.append(action)
        return self.page


def _approve_all(_decision: Any, _obs: Any) -> bool:
    return True


def _canvas_page() -> Page:
    return Page(url="https://example.test/watch", title="Player",
                html=CANVAS_HTML,
                elements=[{"id": "ref:e9", "label": "Play"}])


def test_escalator_mark_answer_becomes_the_action():
    seen: list[tuple[str, str]] = []

    def _see(path: str, prompt: str) -> str:
        seen.append((path, prompt))
        return "click mark 1"

    escalator = PerceptionEscalator(
        see=_see, screenshot=lambda: "/tmp/shot.png")
    driver = FakeDriver(_canvas_page())
    progress: list[dict[str, Any]] = []
    loop = AgentLoop(driver, decide_llm=lambda obs: None,
                     approve=_approve_all, escalator=escalator,
                     max_steps=1, on_progress=progress.append)
    result = loop.run("https://example.test/watch", goal="watch")
    assert seen and "mark" in seen[0][1]
    assert [a.kind for a in driver.acted] == ["click"]
    assert driver.acted[0].target == "ref:e9"
    assert loop.vision_escalations == 1
    assert any("vision_escalation" in event for event in progress)
    assert result["steps"] == 1


def test_escalator_verdict_reaches_the_decider_without_a_mark():
    calls: list[Observation] = []

    def _decider(obs: Observation) -> Decision | None:
        calls.append(obs)
        if obs.vision_answer:
            return Decision(
                Action(kind="click", target="ref:e9", label="Play"),
                reason="the vision report names the play control")
        return None

    escalator = PerceptionEscalator(
        see=lambda path, prompt: "a big red play button, center screen",
        screenshot=lambda: "/tmp/shot.png")
    driver = FakeDriver(_canvas_page())
    loop = AgentLoop(driver, decide_llm=_decider, approve=_approve_all,
                     escalator=escalator, max_steps=1)
    loop.run("https://example.test/watch", goal="watch")
    assert len(calls) == 2  # plain decide, then the vision-fed decide
    assert "red play button" in calls[1].vision_answer
    assert calls[1].som is not None
    assert calls[1].som["marks"][0]["label"] == "Play"
    assert [a.kind for a in driver.acted] == ["click"]


def test_escalator_tier2_structural_elements_come_first():
    seen_elements: list[list[dict[str, Any]]] = []

    def _decider(obs: Observation) -> Decision | None:
        seen_elements.append(list(obs.page.elements))
        for el in obs.page.elements:
            if el.get("id") == "#deep":
                return Decision(
                    Action(kind="click", target="#deep", label="Deep"),
                    reason="found in the structural DOM")
        return None

    def _no_screenshot() -> str:
        raise AssertionError("tier 3 must not run when tier 2 lands")

    escalator = PerceptionEscalator(
        see=lambda path, prompt: "click mark 1",
        screenshot=_no_screenshot,
        extract_structural=lambda: [
            {"selector": "#deep", "label": "Deep", "kind": "button"}])
    page = Page(url="https://example.test/app", title="App",
                html="<html><body>" + "x" * 1500 + "</body></html>",
                elements=[])
    driver = FakeDriver(page)
    loop = AgentLoop(driver, decide_llm=_decider, approve=_approve_all,
                     escalator=escalator, max_steps=1)
    loop.run("https://example.test/app", goal="deep")
    assert driver.acted and driver.acted[0].target == "#deep"
    assert escalator.tier == 2


def test_escalation_failure_is_journaled_and_the_text_path_stands(monkeypatch):
    recorded: list[str] = []
    monkeypatch.setattr(
        "nomorals.browser.failures.record_failure",
        lambda site, exc: recorded.append(site))

    def _see(path: str, prompt: str) -> str:
        raise RuntimeError("vision model is down")

    escalator = PerceptionEscalator(
        see=_see, screenshot=lambda: "/tmp/shot.png")
    page = Page(url="https://example.test/", title="Home",
                html="<html><body><canvas></canvas>"
                     "<button>Claim reward</button></body></html>",
                elements=[{"id": "ref:e1", "label": "Claim reward"}])
    driver = FakeDriver(page)
    loop = AgentLoop(driver, approve=_approve_all, escalator=escalator,
                     max_steps=1)
    result = loop.run("https://example.test/", goal="claim")
    assert recorded == ["browser.agent.vision_escalation"]
    assert [a.kind for a in driver.acted] == ["click"]  # heuristic stood
    assert result["steps"] == 1


def test_no_escalator_means_the_plain_text_path_unchanged():
    obs = Observation(page=_canvas_page())
    keys = set(obs.to_dict())
    assert keys == {"url", "title", "tier", "step", "goal", "history",
                    "snapshot_hash", "elements", "last_result"}
    driver = FakeDriver(_canvas_page())
    loop = AgentLoop(driver, decide_llm=lambda o: None,
                     approve=_approve_all, max_steps=2)
    result = loop.run("https://example.test/watch", goal="watch")
    assert result["stopped_reason"] == "nothing safe to do"
    assert loop.vision_escalations == 0


def test_llm_decider_prompt_carries_vision_only_when_present():
    prompts: list[str] = []

    class _Brain:
        def complete(self, prompt: str, **_kw: Any) -> Any:
            prompts.append(prompt)
            return SimpleNamespace(ok=True, text="")

    decider = LLMDecider(_Brain(), goal="watch")
    plain = Observation(page=_canvas_page(), goal="watch")
    decider(plain)
    assert "vision model looked" not in prompts[-1]
    seen = Observation(page=_canvas_page(), goal="watch",
                       vision_answer="a red play button",
                       som={"marks": [{"mark": 1, "label": "Play"}]})
    decider(seen)
    assert "a red play button" in prompts[-1]
    assert "mark 1: Play" in prompts[-1]


def test_wired_loop_feeds_the_tab_screenshot_to_the_brain(tmp_path):
    from nomorals.browser.loop_wiring import build_wired_agent_loop

    shot = tmp_path / "frame.png"
    shot.write_bytes(b"\x89PNG-fake-frame")

    class _Tab:
        url = "https://example.test/watch"

        def __init__(self) -> None:
            self.clicks: list[str] = []

        def navigate(self, url: str) -> dict[str, Any]:
            return {"ok": True}

        def html(self) -> dict[str, Any]:
            return {"url": self.url, "title": "Player",
                    "html": CANVAS_HTML}

        def snapshot(self) -> dict[str, Any]:
            return {"snapshot": '- button "Play" [ref=e9]'}

        def click(self, target: str) -> dict[str, Any]:
            self.clicks.append(target)
            return {"ok": True}

        def screenshot(self) -> dict[str, Any]:
            return {"path": str(shot)}

    class _VisionBrain:
        def __init__(self) -> None:
            self.images: list[bytes] = []

        def describe_image(self, image: bytes, prompt: str) -> Any:
            self.images.append(image)
            return SimpleNamespace(ok=True, text="click mark 1")

    class _Manager:
        wait_timeout_s = 5

        def request_takeover(self, **_kw: Any) -> Any:
            raise AssertionError("no wall on this page")

    tab = _Tab()
    brain = _VisionBrain()
    notes: list[tuple[str, str]] = []
    loop = build_wired_agent_loop(
        tab=tab, takeover_manager=_Manager(),
        notifier=lambda m, u: notes.append((m, u)),
        owner_chat="telegram:111", task_label="watch",
        max_steps=1, approve=_approve_all,
        decide_llm=lambda obs: None, vision_brain=brain)
    loop.run("https://example.test/watch", goal="watch")
    assert brain.images == [b"\x89PNG-fake-frame"]
    assert tab.clicks == ["ref:e9"]


# ── A2: the desktop decider sees the frame ------------------------------------


def test_computer_decider_decides_from_the_screenshot():
    from nomorals.computer.loop import LLMDecider, Perception

    calls: list[tuple[bytes, str]] = []

    def _vision(png: bytes, prompt: str) -> str:
        calls.append((png, prompt))
        return json.dumps({"action": {"kind": "wait",
                                      "args": {"seconds": 0}},
                           "note": "saw the desktop"})

    def _complete(_prompt: str) -> str:
        raise AssertionError("text path must not run when vision answers")

    decider = LLMDecider(complete_fn=_complete, vision_fn=_vision)
    perception = Perception(
        session_id="s1", ts=0.0, width=800, height=600,
        png=b"\x89PNG-real-frame", frame_path="", frame_hash="h",
        windows=[], focused_id=0, pointer={"x": 1, "y": 2})
    action = decider.decide(perception, "look around", [])
    assert action is not None and action.kind == "wait"
    assert calls and calls[0][0] == b"\x89PNG-real-frame"
    assert "screenshot" in calls[0][1]
    assert decider.last_call_used_vision is True


def test_computer_decider_vision_failure_falls_back_to_text():
    from nomorals.computer.loop import LLMDecider, Perception

    def _vision(_png: bytes, _prompt: str) -> str:
        raise RuntimeError("no VL model on this profile")

    decider = LLMDecider(
        complete_fn=lambda prompt: json.dumps({"done": True}),
        vision_fn=_vision)
    perception = Perception(
        session_id="s1", ts=0.0, width=800, height=600,
        png=b"\x89PNG-real-frame", frame_path="", frame_hash="h",
        windows=[], focused_id=0, pointer={"x": 0, "y": 0})
    assert decider.decide(perception, "look", []) is None
    assert decider.last_call_used_vision is False


def test_computer_service_vision_fn_comes_from_the_router():
    from types import SimpleNamespace as NS

    from nomorals.computer.service import ComputerService

    router = NS(describe_image=lambda png, prompt: NS(
        ok=True, text="{}", error=""))
    svc = ComputerService(context=NS(router=router))
    fn = svc._vision_fn()
    assert fn is not None and fn(b"png", "look") == "{}"
    bad = NS(describe_image=lambda png, prompt: NS(
        ok=False, text="", error="no VL rung"))
    svc2 = ComputerService(context=NS(router=bad))
    fn2 = svc2._vision_fn()
    assert fn2 is not None
    with pytest.raises(RuntimeError):
        fn2(b"png", "look")
    assert ComputerService(context=None)._vision_fn() is None


# ── A3: approvals, not death ---------------------------------------------------


def _submit_obs() -> Observation:
    return Observation(
        page=Page(url="https://shop.test/checkout", title="Checkout",
                  html="<html><body>pay</body></html>",
                  elements=[{"id": "ref:e2", "label": "Place order"}]),
        goal="buy the thing")


def _submit_brain(*replies: str) -> Any:
    queue = list(replies)

    class _Brain:
        def complete(self, prompt: str, **_kw: Any) -> Any:
            text = queue.pop(0) if queue else '{"action": null}'
            return SimpleNamespace(ok=True, text=text)

    return _Brain()


_SUBMIT_JSON = json.dumps({
    "action": {"kind": "submit", "target": "e2",
               "label": "Place order", "payload": {}},
    "reason": "the form is filled", "confidence": 0.9})


def test_decider_stashes_an_actionable_consequential_proposal():
    decider = LLMDecider(_submit_brain(_SUBMIT_JSON), goal="buy")
    assert decider(_submit_obs()) is None  # never emitted directly
    assert "submit" in decider.last_denied
    denied = decider.last_denied_decision
    assert denied is not None
    assert denied.action.kind == "submit"
    assert denied.action.target == "ref:e2"  # resolved to the real ref


def test_decider_drops_an_unresolvable_consequential_proposal():
    bad = json.dumps({
        "action": {"kind": "submit", "target": "no-such-ref",
                   "payload": {}},
        "reason": "hallucinated", "confidence": 0.5})
    decider = LLMDecider(_submit_brain(bad), goal="buy")
    assert decider(_submit_obs()) is None
    assert decider.last_denied_decision is None


def test_surfacing_decider_offers_the_proposal_exactly_once():
    decider = LLMDecider(_submit_brain(_SUBMIT_JSON), goal="buy")
    wrapped = ApprovalSurfacingDecider(decider)
    offered = wrapped(_submit_obs())
    assert offered is not None and offered.action.kind == "submit"
    assert decider.last_denied_decision is None  # consumed
    assert wrapped(_submit_obs()) is None  # never offered twice


def test_broker_benign_acts_never_bother_the_owner():
    notes: list[tuple[str, str]] = []
    broker = ApprovalBroker(notifier=lambda m, u: notes.append((m, u)),
                            timeout_s=5)
    approver = broker.approver(task_label="browse")
    click = Decision(Action(kind="click", target="ref:e1"),
                     reason="benign")
    assert approver(click, _submit_obs()) is True
    assert notes == [] and approver.outcomes == []


# ── Autonomy default (owner order 2026-11-11): she acts herself ─────────────


def test_acting_approver_benign_acts_stay_silent():
    from nomorals.browser.approvals import ActingApprover

    approver = ActingApprover(task_label="browse")
    click = Decision(Action(kind="click", target="ref:e1"),
                     reason="benign")
    assert approver(click, _submit_obs()) is True
    assert approver.outcomes == []


def test_acting_approver_acts_and_records_consequential_steps():
    from nomorals.browser.approvals import ActingApprover

    approver = ActingApprover(task_label="browser run x")
    submit = Decision(Action(kind="submit", target="ref:e2",
                             label="Place order"), reason="checkout")
    assert approver(submit, _submit_obs()) is True
    assert len(approver.outcomes) == 1
    outcome = approver.outcomes[0]
    assert outcome["verdict"] == "acted"
    assert outcome["kind"] == "submit"
    assert outcome["answered_by"] == "devon"


def test_acting_approver_acts_on_irreversible_steps_too():
    from nomorals.browser.approvals import ActingApprover

    approver = ActingApprover(task_label="browser run x")
    click = Decision(Action(kind="click", target="ref:e9"),
                     reason="final", irreversible=True)
    assert approver(click, _submit_obs()) is True
    assert approver.outcomes[0]["verdict"] == "acted"


def _park_and_answer(broker: ApprovalBroker, approver: Any,
                     answer: bool | None) -> dict[str, Any]:
    """Run the approver on a thread; answer (or not) from here."""
    decision = Decision(
        Action(kind="submit", target="ref:e2", label="Place order"),
        reason="the form is filled")
    outcome: dict[str, Any] = {}

    def _run() -> None:
        outcome["verdict"] = approver(decision, _submit_obs())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    if answer is not None:
        deadline = time.time() + 5
        while time.time() < deadline and not broker.pending():
            time.sleep(0.01)
        pending = broker.pending()
        assert pending, "the run never parked for approval"
        request_id = pending[0]["request_id"]
        assert broker.resolve(request_id, answer) is True
    thread.join(timeout=10)
    assert not thread.is_alive()
    return outcome


def test_broker_approve_resumes_the_parked_run():
    notes: list[tuple[str, str]] = []
    broker = ApprovalBroker(notifier=lambda m, u: notes.append((m, u)),
                            timeout_s=30)
    approver = broker.approver(task_label="browser run x")
    outcome = _park_and_answer(broker, approver, True)
    assert outcome["verdict"] is True
    assert notes and "/browser approve appr-" in notes[0][0]
    assert "submit" in notes[0][0]
    assert approver.outcomes[0]["verdict"] == "approved"
    assert broker.pending() == []
    assert broker.resolve("appr-1", True) is False  # already answered


def test_broker_deny_stops_the_parked_run():
    broker = ApprovalBroker(notifier=lambda m, u: None, timeout_s=30)
    approver = broker.approver(task_label="browser run x")
    outcome = _park_and_answer(broker, approver, False)
    assert outcome["verdict"] is False
    assert approver.outcomes[0]["verdict"] == "denied"


def test_broker_silence_is_a_no():
    broker = ApprovalBroker(notifier=lambda m, u: None, timeout_s=0.2)
    approver = broker.approver(task_label="browser run x")
    outcome = _park_and_answer(broker, approver, None)
    assert outcome["verdict"] is False
    assert approver.outcomes[0]["verdict"] == "timeout"


def test_broker_without_a_channel_cannot_approve():
    broker = ApprovalBroker(notifier=None, timeout_s=5)
    approver = broker.approver(task_label="browser run x")
    decision = Decision(
        Action(kind="submit", target="ref:e2", label="Place order"),
        reason="consequential")
    assert approver(decision, _submit_obs()) is False
    assert approver.outcomes[0]["verdict"] == "no_channel"


def _parked_loop(broker: ApprovalBroker, answer: bool) -> dict[str, Any]:
    """A full loop whose decider proposes submit; the owner answers."""
    decider = ApprovalSurfacingDecider(
        LLMDecider(_submit_brain(_SUBMIT_JSON), goal="buy"))
    driver = FakeDriver(_submit_obs().page)
    approver = broker.approver(task_label="browser run shop")
    loop = AgentLoop(driver, decide_llm=decider, approve=approver,
                     autonomy="act_with_approval", max_steps=3)
    outcome: dict[str, Any] = {}

    def _run() -> None:
        outcome["result"] = loop.run("https://shop.test/checkout",
                                     goal="buy")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline and not broker.pending():
        time.sleep(0.01)
    pending = broker.pending()
    assert pending, "the loop never parked at the submit step"
    assert driver.acted == []  # NOTHING executed before the owner's word
    broker.resolve(pending[0]["request_id"], answer)
    thread.join(timeout=10)
    assert not thread.is_alive()
    outcome["driver"] = driver
    outcome["approver"] = approver
    return outcome


def test_loop_parks_at_submit_and_performs_it_only_on_approval():
    broker = ApprovalBroker(notifier=lambda m, u: None, timeout_s=30)
    outcome = _parked_loop(broker, True)
    driver: FakeDriver = outcome["driver"]
    assert [a.kind for a in driver.acted] == ["submit"]
    assert driver.acted[0].target == "ref:e2"
    assert outcome["approver"].outcomes[0]["verdict"] == "approved"


# ── A3 end to end: /browser run parks, /browser approve resumes ---------------


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


def test_browser_run_parks_for_approval_and_the_verb_resumes_it(
        monkeypatch):
    import re

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

    def _msg(text: str, owner: bool) -> ChatMessage:
        return ChatMessage(
            chat=ChatRef(platform="telegram",
                         chat_id="111" if owner else "999",
                         kind=ChatKind.DM),
            incoming=True, text=text,
            sender="owner" if owner else "stranger",
            sender_id="111" if owner else "999",
            meta={"is_owner": True} if owner else {})

    tmp = tempfile.TemporaryDirectory(prefix="nm-webnav-core-")
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "telegram",
        "partner.owner_chats": "telegram:111",
        "partner.enabled": "false",
        # Ask-first is OPT-IN since the owner's 2026-11-11 autonomy
        # order (she acts herself by default); this test pins the
        # parking channel, so it turns the knob on explicitly.
        "partner.browser_require_approval": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    adapter = _Adapter()
    gateway = ChatGateway({"telegram": adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    try:
        tab = _CheckoutTab()
        runtime._browser_service_for_run_job = (  # type: ignore[method-assign]
            lambda job_id: _Service(tab))
        reply = runtime._control_browser(
            "run https://shop.test/checkout buy the thing",
            chat_key="telegram:111",
            message=_msg("/browser run", True))
        match = re.search(r"br-[0-9a-f]{12}", reply)
        assert match, f"no job id in reply: {reply!r}"
        job_id = match.group(0)
        # The run parks at the submit step: nothing has executed,
        # and an approval is waiting on the owner.
        broker = runtime._browser_approval_broker()
        deadline = time.time() + 10
        while time.time() < deadline and not broker.pending():
            time.sleep(0.02)
        pending = broker.pending()
        assert pending, "the run never parked for approval"
        assert tab.clicks == []
        request_id = pending[0]["request_id"]
        # A stranger cannot answer it.
        refused = runtime._control_browser(
            f"approve {request_id}", chat_key="telegram:999",
            message=_msg("/browser approve", False))
        assert "owner-only" in refused
        assert tab.clicks == []
        # The owner approves through the verb; the run resumes and
        # performs the step, and only then.
        answered = runtime._control_browser(
            f"approve {request_id}", chat_key="telegram:111",
            message=_msg("/browser approve", True))
        assert "approved" in answered
        job = runtime._browser_run_jobs().get(job_id)
        assert job is not None and job.wait(15)
        assert tab.clicks == ["ref:e2"]
        assert job.steps == 1
        assert broker.history and \
            broker.history[0]["verdict"] == "approved"
        assert broker.history[0]["kind"] == "submit"
        # An already-answered id is honestly unknown afterwards.
        gone = runtime._control_browser(
            f"deny {request_id}", chat_key="telegram:111",
            message=_msg("/browser deny", True))
        assert "no waiting approval" in gone
    finally:
        with suppress(Exception):
            runtime.stop()
        with suppress(Exception):
            runtime.context.close()
        tmp.cleanup()


def test_browser_run_by_default_acts_on_the_submit_itself(monkeypatch):
    """Owner order 2026-11-11: no approval parking by default —
    the run performs the consequential step itself, and the
    approval broker is never engaged."""
    import re

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

    def _msg(text: str, owner: bool) -> ChatMessage:
        return ChatMessage(
            chat=ChatRef(platform="telegram",
                         chat_id="111" if owner else "999",
                         kind=ChatKind.DM),
            incoming=True, text=text,
            sender="owner" if owner else "stranger",
            sender_id="111" if owner else "999",
            meta={"is_owner": True} if owner else {})

    tmp = tempfile.TemporaryDirectory(prefix="nm-webnav-auto-")
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "telegram",
        "partner.owner_chats": "telegram:111",
        "partner.enabled": "false",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    adapter = _Adapter()
    gateway = ChatGateway({"telegram": adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    try:
        tab = _CheckoutTab()
        runtime._browser_service_for_run_job = (  # type: ignore[method-assign]
            lambda job_id: _Service(tab))
        reply = runtime._control_browser(
            "run https://shop.test/checkout buy the thing",
            chat_key="telegram:111",
            message=_msg("/browser run", True))
        match = re.search(r"br-[0-9a-f]{12}", reply)
        assert match, f"no job id in reply: {reply!r}"
        job = runtime._browser_run_jobs().get(match.group(0))
        assert job is not None and job.wait(15)
        # She did the step herself: the click landed, one step ran,
        # and the approval channel was never touched.
        assert tab.clicks == ["ref:e2"]
        assert job.steps == 1
        broker = runtime._browser_approval_broker()
        assert broker.pending() == []
        assert broker.history == []
    finally:
        with suppress(Exception):
            runtime.stop()
        with suppress(Exception):
            runtime.context.close()
        tmp.cleanup()


def test_loop_stops_at_submit_on_deny_and_acts_on_nothing():
    broker = ApprovalBroker(notifier=lambda m, u: None, timeout_s=30)
    outcome = _parked_loop(broker, False)
    assert outcome["driver"].acted == []
    result = outcome["result"]
    assert result["steps"] == 0
    assert result["stopped_reason"] == "approval denied / no approver"
    assert outcome["approver"].outcomes[0]["verdict"] == "denied"
