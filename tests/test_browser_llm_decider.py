"""LLM decider for the browser agent loop.

The loop used to decide with the heuristic only (free-gain clicks).
``build_llm_decider`` plugs the real brain in front of it: the decider
sees the goal, the current page (URL/title/element refs/page text) and
the history of steps so far, asks the LLM for the next action, and
falls back to the heuristic (``None``) whenever the LLM is unavailable
or returns nothing usable. Consequential acts are NEVER emitted by the
decider — the LLM cannot approve those itself.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from nomorals.browser.agent import (
    CONSEQUENTIAL_KINDS,
    Action,
    AgentLoop,
    Decision,
    Observation,
    Page,
)
from nomorals.browser.llm_decider import LLMDecider, build_llm_decider


class FakeBrain:
    """Duck-typed brain: ``complete(prompt)`` -> resp with .ok/.text."""

    def __init__(self, *, text: str = "", ok: bool = True,
                 explode: bool = False) -> None:
        self.text = text
        self.ok = ok
        self.explode = explode
        self.prompts: list[str] = []

    def complete(self, prompt: str, **_kw: Any) -> Any:
        self.prompts.append(prompt)
        if self.explode:
            raise RuntimeError("provider down")
        return SimpleNamespace(ok=self.ok, text=self.text)


def _page(*, url: str = "https://example.test/",
          elements: list[dict[str, Any]] | None = None,
          html: str = "<html><body>hello</body></html>") -> Page:
    return Page(url=url, title="Home", html=html,
                elements=elements if elements is not None else [])


def _obs(page: Page, *, goal: str = "browse",
         history: list[dict[str, Any]] | None = None) -> Observation:
    return Observation(page=page, step=len(history or []),
                       goal=goal, history=list(history or []))


def _click_decision(target: str, *, kind: str = "click") -> str:
    return json.dumps({
        "action": {"kind": kind, "target": target, "label": "chosen",
                   "payload": {}},
        "reason": "goal-relevant element", "confidence": 0.9,
        "irreversible": False})


# -- the decider sees goal + page + history -----------------------------------

def test_decider_prompt_carries_goal_page_and_history():
    brain = FakeBrain(text=json.dumps({"action": None,
                                       "reason": "nothing safe"}))
    decider = build_llm_decider(brain, goal="find the cats article")
    history = [{"step": 1,
                "action": {"kind": "click", "target": "ref:e9"},
                "result": {"ok": True, "verified": True}}]
    page = _page(elements=[{"id": "ref:e1", "label": "Cats article"}])
    assert decider(_obs(page, goal="find the cats article",
                        history=history)) is None
    prompt = brain.prompts[0]
    assert "find the cats article" in prompt
    assert "ref:e1" in prompt and "Cats article" in prompt
    assert "ref:e9" in prompt  # history is visible to the decider


def test_decider_picks_goal_relevant_action_over_heuristic():
    """On a page with BOTH a free-gain button and the goal element, the
    LLM's goal-relevant click wins — the heuristic would take the gain."""
    elements = [{"id": "ref:e1", "label": "Read the cats article"},
                {"id": "ref:e2", "label": "Claim free reward"}]
    brain = FakeBrain(text=_click_decision("ref:e1"))
    decider = build_llm_decider(brain, goal="read the cats article")
    got = decider(_obs(_page(elements=elements),
                       goal="read the cats article"))
    assert got is not None
    assert got.action.kind == "click"
    assert got.action.target == "ref:e1"
    assert "goal" in got.reason.lower() or got.reason


def test_decider_accepts_bare_ref_without_prefix():
    brain = FakeBrain(text=_click_decision("e1"))
    decider = build_llm_decider(brain)
    got = decider(_obs(_page(elements=[{"id": "ref:e1",
                                        "label": "Cats article"}])))
    assert got is not None
    assert got.action.target == "ref:e1"  # normalized to the real ref


def test_decider_rejects_target_not_on_page():
    brain = FakeBrain(text=_click_decision("ref:hallucinated"))
    decider = build_llm_decider(brain)
    got = decider(_obs(_page(elements=[{"id": "ref:e1",
                                        "label": "Cats article"}])))
    assert got is None  # hallucinated refs never act


def test_decider_fill_needs_a_value():
    brain = FakeBrain(text=json.dumps({
        "action": {"kind": "fill", "target": "ref:e1", "payload": {}},
        "reason": "fill it", "confidence": 0.8}))
    decider = build_llm_decider(brain)
    assert decider(_obs(_page(elements=[{"id": "ref:e1",
                                         "label": "Search box"}]))) is None
    brain2 = FakeBrain(text=json.dumps({
        "action": {"kind": "fill", "target": "ref:e1",
                   "payload": {"value": "cats"}},
        "reason": "search for cats", "confidence": 0.8}))
    decider2 = build_llm_decider(brain2)
    got = decider2(_obs(_page(elements=[{"id": "ref:e1",
                                         "label": "Search box"}])))
    assert got is not None
    assert got.action.payload["value"] == "cats"


# -- fallback: no LLM / unusable LLM -> None (heuristic runs) -----------------

def test_decider_returns_none_when_brain_errors():
    decider = build_llm_decider(FakeBrain(ok=False, text=""))
    assert decider(_obs(_page())) is None


def test_decider_returns_none_when_brain_raises():
    decider = build_llm_decider(FakeBrain(explode=True))
    assert decider(_obs(_page())) is None


def test_decider_returns_none_on_garbage_and_on_null_action():
    for text in ("not json at all", "{broken", "[]",
                 json.dumps({"action": None, "reason": "nothing"}),
                 json.dumps({"action": {"kind": "teleport",
                                        "target": "x"}})):
        decider = build_llm_decider(FakeBrain(text=text))
        assert decider(_obs(_page())) is None, text


def test_decider_returns_none_without_a_brain():
    decider = build_llm_decider(None)
    assert decider(_obs(_page())) is None


# -- consequential acts: the LLM can never approve them ------------------------

def test_decider_never_emits_consequential_action():
    for kind in sorted(CONSEQUENTIAL_KINDS):
        brain = FakeBrain(text=_click_decision("ref:e1", kind=kind))
        decider = build_llm_decider(brain)
        page = _page(elements=[{"id": "ref:e1", "label": f"{kind} now"}])
        got = decider(_obs(page))
        assert got is None, kind
        assert decider.last_denied  # the drop is recorded, not silent


def test_decider_never_emits_irreversible_action():
    brain = FakeBrain(text=json.dumps({
        "action": {"kind": "click", "target": "ref:e1", "payload": {}},
        "reason": "do it", "confidence": 0.9, "irreversible": True}))
    decider = build_llm_decider(brain)
    got = decider(_obs(_page(elements=[{"id": "ref:e1",
                                        "label": "Delete everything"}])))
    assert got is None
    assert "irreversible" in decider.last_denied.lower()


# -- loop level: LLM decides, heuristic falls back, LLM can't spend ----------

class FakeDriver:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.acted: list[Action] = []

    def fetch(self, url: str) -> Page:
        return self.page

    def act(self, action: Action, page: Page) -> Page:
        self.acted.append(action)
        return self.page


def test_loop_llm_decision_beats_heuristic():
    elements = [{"id": "ref:e1", "label": "Read the cats article"},
                {"id": "ref:e2", "label": "Claim free reward"}]
    page = _page(elements=elements)
    brain = FakeBrain(text=json.dumps({
        "action": {"kind": "click", "target": "ref:e1", "payload": {}},
        "reason": "goal", "confidence": 0.9}))
    loop = AgentLoop(FakeDriver(page),
                     decide_llm=build_llm_decider(brain, goal="cats"),
                     autonomy="propose",
                     approve=lambda d, o: True, max_steps=1)
    loop.run(page.url, goal="cats")
    assert [a.target for a in loop.driver.acted] == ["ref:e1"]


def test_loop_heuristic_fallback_when_no_llm():
    page = _page(elements=[{"id": "ref:e1",
                            "label": "Claim free reward"}])
    driver = FakeDriver(page)
    loop = AgentLoop(driver,
                     decide_llm=build_llm_decider(FakeBrain(ok=False)),
                     autonomy="propose",
                     approve=lambda d, o: True, max_steps=1)
    loop.run(page.url, goal="browse")
    assert [a.target for a in driver.acted] == ["ref:e1"]


def test_loop_llm_cannot_approve_consequential_act():
    page = _page(elements=[{"id": "ref:e1", "label": "Pay now"}])
    driver = FakeDriver(page)
    brain = FakeBrain(text=json.dumps({
        "action": {"kind": "pay", "target": "ref:e1", "payload": {}},
        "reason": "pay for it", "confidence": 1.0}))
    loop = AgentLoop(driver,
                     decide_llm=build_llm_decider(brain, goal="pay"),
                     autonomy="propose",
                     approve=lambda d, o: True, max_steps=3)
    result = loop.run(page.url, goal="pay")
    assert driver.acted == []  # the pay action never ran
    assert result["steps"] == 0
