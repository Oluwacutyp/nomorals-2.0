"""Offline tests for the NL routine builder (#62)."""

import asyncio

import pytest

from nomorals.integrations.routines import (
    RoutineBuildError,
    activate,
    build_routine,
    confirm,
    describe,
    load_aliases,
    resolve_entity,
    to_ha_automation,
    validate,
)


DEVICES = [
    {"entity_id": "light.kitchen", "friendly_name": "Kitchen Lights"},
    {"entity_id": "light.bedroom", "friendly_name": "Bedroom Light"},
    {"entity_id": "switch.coffee_maker", "friendly_name": "Coffee Maker"},
    {"entity_id": "cover.garage_door", "friendly_name": "Garage Door"},
    {"entity_id": "lock.front_door", "friendly_name": "Front Door"},
    {"entity_id": "climate.thermostat", "friendly_name": "Thermostat"},
]


# ── parsing ───────────────────────────────────────────────────────────────


def test_time_starter_parsing():
    d = build_routine("every morning at 7, turn on kitchen lights",
                      devices=DEVICES)
    assert d.starter is not None
    assert d.starter.kind == "time"
    assert d.starter.at == "07:00"
    assert validate(d) == []


def test_time_starter_pm():
    d = build_routine("daily at 6:30pm, turn off bedroom light",
                      devices=DEVICES)
    assert d.starter.at == "18:30"


def test_device_starter_parsing():
    d = build_routine("when the garage door opens, notify me",
                      devices=DEVICES)
    assert d.starter is not None
    assert d.starter.kind == "device"
    assert d.starter.entity_id == "cover.garage_door"
    assert d.starter.to_state == "open"
    assert validate(d) == []


def test_sun_starter_parsing():
    d = build_routine("at sunset, turn on kitchen lights", devices=DEVICES)
    assert d.starter.kind == "sun"
    assert d.starter.event == "sunset"
    assert validate(d) == []


def test_presence_starter():
    d = build_routine("when I leave home, lock the front door",
                      devices=DEVICES)
    assert d.starter.kind == "presence"
    assert d.starter.presence == "leave"
    assert validate(d) == []


def test_conditions_parsed():
    d = build_routine("at sunset, turn on kitchen lights only if dark",
                      devices=DEVICES)
    assert [c.kind for c in d.conditions] == ["dark"]
    assert "dark" in describe(d)


def test_unparseable_asks_for_clarification():
    d = build_routine("blargh flargh nothing", devices=DEVICES)
    assert d.needs_clarification
    assert "couldn't parse" in describe(d)
    errors = validate(d)
    assert errors and errors[0].code == "unparseable"


def test_build_never_raises():
    # garbage in, draft out — never an exception
    d = build_routine("", devices=DEVICES)
    assert d.needs_clarification
    d = build_routine(None, devices=DEVICES)  # type: ignore[arg-type]
    assert d.needs_clarification


# ── entity resolution ─────────────────────────────────────────────────────


def test_fuzzy_resolution():
    # "kitchen lights" (plural) → light.kitchen
    assert resolve_entity("kitchen lights", DEVICES) == "light.kitchen"
    # "coffee maker" exact
    assert resolve_entity("coffee maker", DEVICES) == "switch.coffee_maker"
    # fuzzy: "kitchin light" typo
    assert resolve_entity("kitchin light", DEVICES) == "light.kitchen"


def test_aliases_win():
    aliases = {"the big light": "light.kitchen"}
    assert resolve_entity("the big light", DEVICES,
                          aliases=aliases) == "light.kitchen"


def test_unknown_device_is_unresolved():
    assert resolve_entity("flux capacitor", DEVICES) is None
    d = build_routine("at 7am, turn on the flux capacitor", devices=DEVICES)
    errors = validate(d)
    assert any(e.code == "unknown_device" for e in errors)


def test_load_aliases_from_memory():
    class FakeRecord:
        text = "device alias: the big light -> light.kitchen"
    class FakeResult:
        def __iter__(self):
            return iter([FakeRecord()])
    class FakeMemory:
        def recall(self, query, **kwargs):
            assert kwargs.get("tags") == "device_alias"
            return FakeResult()
    assert load_aliases(FakeMemory()) == {"the big light": "light.kitchen"}


def test_load_aliases_never_raises():
    assert load_aliases(None) == {}
    class BrokenMemory:
        def recall(self, *a, **k):
            raise RuntimeError("boom")
    assert load_aliases(BrokenMemory()) == {}


# ── validation ────────────────────────────────────────────────────────────


def test_conflicting_actions_rejected():
    d = build_routine(
        "at 7am, turn on kitchen lights and turn off kitchen lights",
        devices=DEVICES)
    errors = validate(d)
    assert any(e.code == "conflict" for e in errors)


def test_no_starter_error():
    d = build_routine("turn on kitchen lights", devices=DEVICES)
    errors = validate(d)
    assert any(e.code == "no_starter" for e in errors)


def test_confirm_raises_on_errors():
    d = build_routine("turn on the flux capacitor", devices=DEVICES)
    with pytest.raises(RoutineBuildError):
        confirm(d)


def test_confirm_flow():
    d = build_routine("every morning at 7, kitchen lights + coffee maker",
                      devices=DEVICES)
    assert validate(d) == []
    routine = confirm(d)
    assert routine.starter.kind == "time"
    assert len(routine.actions) == 2
    # draft is consumed
    from nomorals.integrations.routines import pending_draft
    assert pending_draft(d.id) is None


# ── Home Assistant conversion ─────────────────────────────────────────────


def test_to_ha_time_trigger():
    d = build_routine("at 7am, turn on kitchen lights", devices=DEVICES)
    routine = confirm(d)
    trigger, actions, conditions = to_ha_automation(routine)
    assert trigger == {"platform": "time", "at": "07:00:00"}
    assert actions[0]["service"] == "light.turn_on"
    assert actions[0]["target"]["entity_id"] == "light.kitchen"


def test_to_ha_device_trigger():
    d = build_routine("when the garage door opens, turn on kitchen lights",
                      devices=DEVICES)
    routine = confirm(d)
    trigger, actions, _ = to_ha_automation(routine)
    assert trigger["platform"] == "state"
    assert trigger["entity_id"] == "cover.garage_door"
    assert trigger["to"] == "open"


def test_to_ha_conditions():
    d = build_routine("at sunset, turn on kitchen lights only if dark",
                      devices=DEVICES)
    routine = confirm(d)
    _, _, conditions = to_ha_automation(routine)
    assert conditions[0]["entity_id"] == "sun.sun"
    assert conditions[0]["state"] == "below_horizon"


def test_activate_calls_create_automation():
    d = build_routine("at 7am, turn on kitchen lights", devices=DEVICES)
    routine = confirm(d)
    calls = []
    class FakeIntegration:
        async def create_automation(self, name, trigger, actions,
                                    conditions=None):
            calls.append((name, trigger, actions, conditions))
            return "automation-123"
    result = asyncio.run(activate(routine, FakeIntegration()))
    assert result.ha_automation_id == "automation-123"
    assert calls[0][1]["platform"] == "time"
