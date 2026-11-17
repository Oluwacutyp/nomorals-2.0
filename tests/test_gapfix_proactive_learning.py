"""Pins for the proactive-learning wiring lane (scout 9 top pick #1).

The engine that suggests things never learned from anything the owner
did: ``record_action`` had zero production callers, ``accept`` /
``reject`` only logged, the bridge dropped the action name from the
payload, the engine instance was constructed inline and retained
nowhere, and patterns were memory-only. These tests pin the wired
behaviour end to end.
"""

import asyncio
from types import SimpleNamespace

import pytest

from nomorals.agents.modes import AutonomousEngine, proactive_source
from nomorals.agents.proactive import ProactiveEngine, Suggestion

CTX = {"day_of_week": "monday", "hour": 9}
ACTION = "order_coffee"


def _run(coro):
    return asyncio.run(coro)


def _pattern_suggestion(action=ACTION):
    return Suggestion(
        suggestion_id="sug-1",
        text="You usually order coffee on Mondays. Want me to do it now?",
        suggestion_type="pattern",
        action=action,
        parameters=dict(CTX),
        confidence=0.9,
        priority=2,
    )


async def _record_five(engine):
    for _ in range(5):
        await engine.record_action("owner", ACTION, context=dict(CTX))


async def _pattern_actions(engine):
    suggestions = await engine.get_suggestions(
        "owner", context=dict(CTX))
    return [s.action for s in suggestions]


# ── (a) the action name survives the payload end to end ──────────────


def test_payload_carries_action_name_and_user():
    async def _suggestions(user_id, limit=5, min_confidence=0.5,
                           context=None):
        return [_pattern_suggestion()]

    source = proactive_source(SimpleNamespace(get_suggestions=_suggestions),
                              "owner")
    specs = source({})
    assert len(specs) == 1
    payload = specs[0]["payload"]
    assert payload["action"] == ACTION
    assert payload["user_id"] == "owner"
    assert specs[0]["requires_approval"] is True


# ── the pattern generator itself (control: already true on base) ──────


def test_recorded_action_surfaces_as_suggestion(tmp_path):
    engine = ProactiveEngine(home=tmp_path)
    _run(_record_five(engine))
    assert ACTION in _run(_pattern_actions(engine))


# ── accept learns ─────────────────────────────────────────────────────


def test_accept_suggestion_records_the_action(tmp_path):
    engine = ProactiveEngine(home=tmp_path)
    for _ in range(4):
        _run(engine.record_action("owner", ACTION, context=dict(CTX)))
    # Four occurrences: confidence 0.4 — below the 0.5 floor, no
    # suggestion yet.
    assert ACTION not in _run(_pattern_actions(engine))
    _run(engine.accept_suggestion(_pattern_suggestion()))
    assert ACTION in _run(_pattern_actions(engine))
    [pattern] = engine.get_patterns("owner")
    assert pattern.occurrences == 5
    assert pattern.confidence == pytest.approx(0.5)


# ── deny suppresses ───────────────────────────────────────────────────


def test_reject_suggestion_suppresses_it(tmp_path):
    engine = ProactiveEngine(home=tmp_path)
    _run(_record_five(engine))
    assert ACTION in _run(_pattern_actions(engine))
    _run(engine.reject_suggestion(_pattern_suggestion()))
    assert ACTION not in _run(_pattern_actions(engine))


def test_doing_the_action_after_reject_unsuppresses(tmp_path):
    engine = ProactiveEngine(home=tmp_path)
    _run(_record_five(engine))
    _run(engine.reject_suggestion(_pattern_suggestion()))
    assert ACTION not in _run(_pattern_actions(engine))
    _run(engine.record_action("owner", ACTION, context=dict(CTX)))
    assert ACTION in _run(_pattern_actions(engine))


# ── persistence ───────────────────────────────────────────────────────


def test_patterns_survive_a_fresh_engine(tmp_path):
    home = tmp_path / "home"
    first = ProactiveEngine(home=home)
    _run(_record_five(first))
    store = home / "data" / "proactive_patterns.json"
    assert store.is_file()
    second = ProactiveEngine(home=home)
    [pattern] = second.get_patterns("owner")
    assert pattern.action == ACTION
    assert pattern.occurrences == 5
    assert ACTION in _run(_pattern_actions(second))


def test_rejection_survives_a_fresh_engine(tmp_path):
    home = tmp_path / "home"
    first = ProactiveEngine(home=home)
    _run(_record_five(first))
    _run(first.reject_suggestion(_pattern_suggestion()))
    second = ProactiveEngine(home=home)
    assert ACTION not in _run(_pattern_actions(second))


def test_corrupt_store_starts_empty(tmp_path):
    home = tmp_path / "home"
    store = home / "data" / "proactive_patterns.json"
    store.parent.mkdir(parents=True)
    store.write_text("{not json", encoding="utf-8")
    engine = ProactiveEngine(home=home)
    assert engine.get_patterns("owner") == []


def test_no_home_is_memory_only(tmp_path):
    engine = ProactiveEngine()
    assert engine._store_path is None
    _run(_record_five(engine))
    assert ACTION in _run(_pattern_actions(engine))


# ── the modes-engine loop: approve learns, deny suppresses ───────────


def _held_engine(tmp_path):
    engine = ProactiveEngine(home=tmp_path / "home")
    _run(_record_five(engine))
    filed: list[str] = []
    modes = AutonomousEngine(
        proactive_engine=engine,
        proposal_sink=lambda init, spec, ctx: filed.append(init.id)
        or init.id)
    modes.add_source("proactive", proactive_source(engine, "owner"))
    return engine, modes


def _pick(inits, needle):
    return next(i for i in inits if needle in i.label.lower())


def test_approve_via_modes_records_and_deny_suppresses(tmp_path):
    engine, modes = _held_engine(tmp_path)
    done_inits = modes.tick(dict(CTX))
    coffee = _pick(done_inits, "order coffee")
    morning = _pick(done_inits, "good morning")
    assert coffee.status == "proposed"
    assert morning.status == "proposed"
    assert coffee.label.startswith("proactive:")
    # Approve the coffee suggestion: the action runs and the pattern
    # gains an occurrence.
    done = modes.approve_proposal(coffee.proposal_id)
    assert done.status == "done"
    [pattern] = engine.get_patterns("owner")
    assert pattern.occurrences == 6
    # Deny the morning briefing: it stops being suggested.
    denied = modes.deny_proposal(morning.proposal_id)
    assert denied.status == "denied"
    assert "morning_briefing" not in _run(_pattern_actions(engine))
    assert ACTION in _run(_pattern_actions(engine))


def test_suggestions_still_require_approval(tmp_path):
    engine, modes = _held_engine(tmp_path)
    coffee = _pick(modes.tick(dict(CTX)), "order coffee")
    assert coffee.status == "proposed"
    assert coffee.result is None


def test_learning_failure_never_breaks_an_approval(tmp_path, monkeypatch):
    engine, modes = _held_engine(tmp_path)
    monkeypatch.setattr(engine, "_save_patterns",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    done_inits = modes.tick(dict(CTX))
    coffee = _pick(done_inits, "order coffee")
    morning = _pick(done_inits, "good morning")
    done = modes.approve_proposal(coffee.proposal_id)
    assert done.status == "done"
    denied = modes.deny_proposal(morning.proposal_id)
    assert denied.status == "denied"


def test_learning_failure_never_breaks_an_approval_broken_home(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    engine = ProactiveEngine(home=blocked)
    _run(_record_five(engine))
    filed: list[str] = []
    modes = AutonomousEngine(
        proactive_engine=engine,
        proposal_sink=lambda init, spec, ctx: filed.append(init.id)
        or init.id)
    modes.add_source("proactive", proactive_source(engine, "owner"))
    coffee = _pick(modes.tick(dict(CTX)), "order coffee")
    done = modes.approve_proposal(coffee.proposal_id)
    assert done.status == "done"
