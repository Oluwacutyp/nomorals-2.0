"""Natural-language routine builder: NL → validated smart-home automation.

"every morning at 7, kitchen lights + coffee" →
{starter: time(07:00), actions: [light.kitchen on, switch.coffee_maker on]}

Grammar:
    STARTERS    time ("every morning at 7", "daily at 18:30")
                sun ("at sunrise", "30 min before sunset")
                device ("when the garage opens", "when I leave home")
    CONDITIONS  "only if dark", "only on weekdays", "only if I'm home"
    ACTIONS     "turn on X", "lock Y", "set thermostat to 22", "notify me ..."

Entity resolution: HA registry + user aliases from memory (fuzzy).
Validation is mandatory — a routine can never be saved with errors.
The user always confirms the plain-language description before activation.
"""

from __future__ import annotations

import difflib
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

# ── data model ──────────────────────────────────────────────────────────────


@dataclass
class RoutineStarter:
    kind: str  # "time" | "sun" | "device" | "presence"
    # time
    at: str = ""            # "07:00"
    days: str = ""          # "" | "weekdays" | "weekends"
    # sun
    event: str = ""         # "sunrise" | "sunset"
    offset_min: int = 0     # e.g. -30 = 30 min before
    # device
    entity_id: str = ""
    to_state: str = ""      # "on" | "off" | "open" | "closed" | "home" | "not_home"
    # presence
    presence: str = ""      # "leave" | "arrive"


@dataclass
class RoutineCondition:
    kind: str  # "dark" | "light" | "weekdays" | "weekends" | "home" | "away"
    raw: str = ""


@dataclass
class RoutineAction:
    entity_id: str = ""     # "" for notify actions
    service: str = ""       # "turn_on" | "turn_off" | "lock" | "unlock" |
                            # "set_temperature" | "notify" | "activate_scene"
    params: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class RoutineError:
    code: str
    message: str


@dataclass
class RoutineDraft:
    id: str
    raw: str
    name: str = ""
    starter: RoutineStarter | None = None
    conditions: list[RoutineCondition] = field(default_factory=list)
    actions: list[RoutineAction] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification: str = ""


@dataclass
class Routine:
    id: str
    name: str
    starter: RoutineStarter
    conditions: list[RoutineCondition]
    actions: list[RoutineAction]
    ha_automation_id: str = ""


class RoutineBuildError(RuntimeError):
    """Raised by confirm() when the draft has validation errors."""


# Draft registry: pending confirmations, keyed by draft id.
_DRAFTS: dict[str, RoutineDraft] = {}


def pending_draft(draft_id: str) -> RoutineDraft | None:
    return _DRAFTS.get(draft_id)


# ── entity resolution ───────────────────────────────────────────────────────


def _norm(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"^(the|my)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _singular(text: str) -> str:
    # "kitchen lights" → "kitchen light" (also try the plural form)
    if text.endswith("s") and not text.endswith("ss"):
        return text[:-1]
    return text + "s"


def resolve_entity(
    phrase: str,
    devices: list[dict[str, Any]],
    aliases: dict[str, str] | None = None,
) -> str | None:
    """Resolve a device phrase to an entity_id.

    1. User aliases (exact, normalized).
    2. Exact match on friendly_name / entity_id.
    3. Fuzzy match (difflib, threshold 0.6) on friendly_name.
    Returns None when nothing matches.
    """
    want = _norm(phrase)
    if not want:
        return None
    aliases = aliases or {}
    for alias, entity_id in aliases.items():
        if _norm(alias) == want:
            return entity_id
    # exact
    for dev in devices:
        name = _norm(str(dev.get("friendly_name") or dev.get("name") or ""))
        eid = str(dev.get("entity_id") or dev.get("device_id") or "")
        if want == name or want == eid.lower() or want == _singular(name):
            return eid
    # fuzzy on friendly names
    best: tuple[float, str] | None = None
    for dev in devices:
        name = _norm(str(dev.get("friendly_name") or dev.get("name") or ""))
        eid = str(dev.get("entity_id") or dev.get("device_id") or "")
        if not name or not eid:
            continue
        for cand in (name, _singular(name)):
            score = difflib.SequenceMatcher(None, want, cand).ratio()
            if score >= 0.6 and (best is None or score > best[0]):
                best = (score, eid)
    return best[1] if best else None


def load_aliases(memory: Any) -> dict[str, str]:
    """Load user device-name aliases from memory (tag: device_alias).

    Expected record text format: "device alias: <alias> -> <entity_id>".
    Never raises — returns {} when memory is unavailable.
    """
    aliases: dict[str, str] = {}
    if memory is None:
        return aliases
    try:
        result = memory.recall("device alias", tags="device_alias", limit=50)
    except Exception:  # noqa: BLE001 — aliases are a nicety
        _log.debug("alias recall failed", exc_info=True)
        return aliases
    for record in result:
        text = getattr(record, "text", "") or ""
        m = re.search(r"device alias:\s*(.+?)\s*->\s*([a-z_]+\.[a-z0-9_]+)",
                      text, re.IGNORECASE)
        if m:
            aliases[m.group(1).strip()] = m.group(2).strip()
    return aliases


# ── parsing ─────────────────────────────────────────────────────────────────

_TIME_RE = re.compile(
    r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)
_SUN_RE = re.compile(
    r"\b(?:(\d+)\s*(?:min|minutes?)\s*(before|after)\s+)?(sunrise|sunset)\b",
    re.IGNORECASE)

_ON_RE = re.compile(
    r"\b(?:turn|switch)\s+(on|off)\s+(?:the\s+|my\s+)?(.+?)(?:\s+and\s+|\s*,|\s*$)",
    re.IGNORECASE)
_LOCK_RE = re.compile(
    r"\b(lock|unlock)\s+(?:the\s+|my\s+)?(.+?)(?:\s+and\s+|\s*,|\s*$)",
    re.IGNORECASE)
_TEMP_RE = re.compile(
    r"\bset\s+(?:the\s+|my\s+)?(.+?)\s+to\s+(\d+(?:\.\d+)?)\s*(?:degrees?)?\b",
    re.IGNORECASE)
_NOTIFY_RE = re.compile(
    r"\b(?:notify\s+me|send\s+me\s+(?:a\s+)?message)"
    r"(?:\s+that|\s*:\s*)?\s*(.*)",
    re.IGNORECASE)
_SCENE_RE = re.compile(
    r"\b(?:activate|enable|start)\s+(?:the\s+|my\s+)?(.+?)\s+(?:scene|mode)"
    r"|\b(.+?)\s+(?:scene|mode)\s+(?:on|activate)?\b",
    re.IGNORECASE)

_COND_PATTERNS = [
    (re.compile(r"\bonly\s+(?:if|when)\s+(?:it'?s\s+|it\s+is\s+)?dark\b",
                re.IGNORECASE), "dark"),
    (re.compile(r"\bonly\s+(?:if|when)\s+(?:it'?s\s+|it\s+is\s+)?(?:light|daytime)\b",
                re.IGNORECASE), "light"),
    (re.compile(r"\bonly\s+on\s+weekdays\b", re.IGNORECASE), "weekdays"),
    (re.compile(r"\bonly\s+on\s+weekends\b", re.IGNORECASE), "weekends"),
    (re.compile(r"\bonly\s+(?:if|when)\s+i'?m\s+home\b", re.IGNORECASE), "home"),
    (re.compile(r"\bonly\s+(?:if|when)\s+i'?m\s+(?:away|not\s+home)\b",
                re.IGNORECASE), "away"),
]

_PRESENCE_RE = re.compile(
    r"\bwhen\s+i\s+(leave(?:\s+home)?|arrive(?:\s+home)?|"
    r"get\s+home|come\s+home)\b",
    re.IGNORECASE)
_DEVICE_STATE_RE = re.compile(
    r"\bwhen\s+(?:the\s+|my\s+)?(.+?)\s+"
    r"(opens?|closes?|turns?\s+on|turns?\s+off|is\s+opened|is\s+closed)\b",
    re.IGNORECASE)


def _parse_time(text: str) -> tuple[str | None, str]:
    """Return (HH:MM, remaining_text)."""
    m = _TIME_RE.search(text)
    if not m:
        return None, text
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = (m.group(3) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None, text
    days = ""
    if re.search(r"\bweekdays?\b", text, re.IGNORECASE):
        days = "weekdays"
    elif re.search(r"\bweekends?\b", text, re.IGNORECASE):
        days = "weekends"
    rest = text[:m.start()] + text[m.end():]
    rest = re.sub(r"\bevery\s+(morning|day|evening|night|weekday|weekend)s?\b",
                  "", rest, flags=re.IGNORECASE)
    rest = re.sub(r"\bdaily\b", "", rest, flags=re.IGNORECASE)
    return f"{hour:02d}:{minute:02d}", rest.strip(" ,")


def _split_actions(text: str) -> list[str]:
    parts = re.split(r"\s*(?:,|\+|\band\b)\s*", text)
    return [p.strip(" ,") for p in parts if p.strip(" ,")]


def build_routine(
    nl_text: str,
    *,
    devices: list[dict[str, Any]] | None = None,
    aliases: dict[str, str] | None = None,
    memory: Any = None,
) -> RoutineDraft:
    """Parse natural language into a RoutineDraft. Never raises.

    When nothing parses, the draft has needs_clarification=True.
    """
    draft = RoutineDraft(id=uuid.uuid4().hex[:8], raw=nl_text or "")
    text = (nl_text or "").strip()
    if not text:
        draft.needs_clarification = True
        draft.clarification = "Tell me what the routine should do — e.g. " \
            "\"every morning at 7, turn on the kitchen lights\"."
        return draft

    devices = devices or []
    if aliases is None and memory is not None:
        aliases = load_aliases(memory)
    aliases = aliases or {}

    # ── conditions ("only if ...") ──
    for pattern, kind in _COND_PATTERNS:
        m = pattern.search(text)
        if m:
            draft.conditions.append(RoutineCondition(kind=kind, raw=m.group(0)))
            text = text[:m.start()] + text[m.end():]

    # ── starter ──
    starter_found = False
    sun_m = _SUN_RE.search(text)
    if sun_m:
        offset = int(sun_m.group(1) or 0)
        if (sun_m.group(2) or "").lower() == "before":
            offset = -offset
        draft.starter = RoutineStarter(
            kind="sun", event=sun_m.group(3).lower(), offset_min=offset)
        text = text[:sun_m.start()] + text[sun_m.end():]
        # "at sunset" → the "at" belongs to the starter
        text = re.sub(r"\bat\s*$", "", text.strip(),
                      flags=re.IGNORECASE).strip()
        text = re.sub(r"^\s*at\s+", "", text, flags=re.IGNORECASE)
        starter_found = True
    else:
        pres_m = _PRESENCE_RE.search(text)
        if pres_m:
            verb = pres_m.group(1).lower()
            draft.starter = RoutineStarter(
                kind="presence",
                presence="leave" if "leave" in verb else "arrive")
            text = text[:pres_m.start()] + text[pres_m.end():]
            starter_found = True
        else:
            dev_m = _DEVICE_STATE_RE.search(text)
            if dev_m:
                entity = resolve_entity(dev_m.group(1), devices, aliases)
                state_word = dev_m.group(2).lower()
                to_state = ("open" if "open" in state_word else
                            "closed" if "clos" in state_word else
                            "on" if "on" in state_word else "off")
                if entity:
                    draft.starter = RoutineStarter(
                        kind="device", entity_id=entity, to_state=to_state)
                    starter_found = True
                else:
                    draft.unresolved.append(dev_m.group(1).strip())
                text = text[:dev_m.start()] + text[dev_m.end():]
            else:
                at, rest = _parse_time(text)
                if at:
                    draft.starter = RoutineStarter(kind="time", at=at)
                    # re-check day words after time removal
                    if re.search(r"\bweekdays?\b", nl_text, re.IGNORECASE):
                        draft.starter.days = "weekdays"
                    elif re.search(r"\bweekends?\b", nl_text, re.IGNORECASE):
                        draft.starter.days = "weekends"
                    text = rest
                    starter_found = True

    text = text.strip(" ,")
    # ── actions ──
    for chunk in _split_actions(text):
        action = _parse_action(chunk, devices, aliases, draft)
        if action:
            draft.actions.append(action)

    if not starter_found and not draft.actions:
        draft.needs_clarification = True
        draft.clarification = (
            "I couldn't parse that. Try something like: "
            "\"every morning at 7, turn on the kitchen lights\" — "
            "tell me WHEN (time / sunrise / when X happens) and WHAT "
            "(turn on/off, lock, notify).")
    elif not starter_found:
        draft.unresolved.append("__starter__")
    elif not draft.actions and not draft.unresolved:
        draft.unresolved.append("__actions__")

    # default name
    draft.name = _default_name(draft)
    _DRAFTS[draft.id] = draft
    return draft


def _parse_action(
    chunk: str,
    devices: list[dict[str, Any]],
    aliases: dict[str, str],
    draft: RoutineDraft,
) -> RoutineAction | None:
    m = _ON_RE.match(chunk + " ")
    if m:
        entity = resolve_entity(m.group(2), devices, aliases)
        if entity:
            return RoutineAction(entity_id=entity,
                                 service=f"turn_{m.group(1).lower()}",
                                 raw=chunk)
        draft.unresolved.append(m.group(2).strip())
        return None
    m = _LOCK_RE.match(chunk + " ")
    if m:
        entity = resolve_entity(m.group(2), devices, aliases)
        if entity:
            return RoutineAction(entity_id=entity,
                                 service=m.group(1).lower(), raw=chunk)
        draft.unresolved.append(m.group(2).strip())
        return None
    m = _TEMP_RE.search(chunk)
    if m:
        entity = resolve_entity(m.group(1), devices, aliases)
        if entity:
            return RoutineAction(entity_id=entity, service="set_temperature",
                                 params={"temperature": float(m.group(2))},
                                 raw=chunk)
        draft.unresolved.append(m.group(1).strip())
        return None
    m = _NOTIFY_RE.search(chunk)
    if m:
        msg = m.group(1).strip()
        return RoutineAction(service="notify",
                             params={"message": msg or "routine fired"},
                             raw=chunk)
    m = _SCENE_RE.search(chunk)
    if m:
        scene = (m.group(1) or m.group(2) or "").strip()
        return RoutineAction(service="activate_scene",
                             params={"scene": scene}, raw=chunk)
    # bare device name → assume turn on ("kitchen lights + coffee")
    entity = resolve_entity(chunk, devices, aliases)
    if entity:
        return RoutineAction(entity_id=entity, service="turn_on", raw=chunk)
    draft.unresolved.append(chunk.strip())
    return None


def _default_name(draft: RoutineDraft) -> str:
    if draft.starter and draft.starter.kind == "time":
        return f"Routine at {draft.starter.at}"
    if draft.starter and draft.starter.kind == "sun":
        return f"Routine at {draft.starter.event}"
    if draft.starter and draft.starter.kind == "presence":
        return f"Routine when I {draft.starter.presence}"
    if draft.starter and draft.starter.kind == "device":
        return f"Routine on {draft.starter.entity_id}"
    return "Untitled routine"


# ── validation ──────────────────────────────────────────────────────────────


def validate(draft: RoutineDraft) -> list[RoutineError]:
    """Validate a draft. A routine can never be saved with errors."""
    errors: list[RoutineError] = []
    if draft.needs_clarification:
        errors.append(RoutineError(
            "unparseable", draft.clarification or
            "I couldn't understand that routine."))
        return errors
    if draft.starter is None or "__starter__" in draft.unresolved:
        errors.append(RoutineError(
            "no_starter",
            "I couldn't tell WHEN this should run — add a time "
            "(\"at 7am\"), \"at sunrise\", or \"when X happens\"."))
    for phrase in draft.unresolved:
        if phrase in ("__starter__", "__actions__"):
            continue
        errors.append(RoutineError(
            "unknown_device",
            f"I don't know which device \"{phrase}\" is. "
            "Check the name or teach me an alias."))
    if "__actions__" in draft.unresolved or not draft.actions:
        errors.append(RoutineError(
            "no_actions",
            "I couldn't tell WHAT to do — add an action like "
            "\"turn on the kitchen lights\"."))
    # conflicting actions on the same entity
    seen: dict[str, str] = {}
    for action in draft.actions:
        if not action.entity_id:
            continue
        want = "on" if action.service in ("turn_on", "unlock") else \
               "off" if action.service in ("turn_off", "lock") else None
        if want and action.entity_id in seen and seen[action.entity_id] != want:
            errors.append(RoutineError(
                "conflict",
                f"\"{action.entity_id}\" is told to turn both on and off — "
                "pick one."))
        elif want:
            seen[action.entity_id] = want
    return errors


# ── description / confirm / activate ────────────────────────────────────────


def _starter_text(starter: RoutineStarter) -> str:
    if starter.kind == "time":
        day = f" on {starter.days}" if starter.days else ""
        return f"every day at {starter.at}{day}"
    if starter.kind == "sun":
        off = ""
        if starter.offset_min:
            direction = "before" if starter.offset_min < 0 else "after"
            off = f" ({abs(starter.offset_min)} min {direction})"
        return f"at {starter.event}{off}"
    if starter.kind == "device":
        return f"when {starter.entity_id} turns {starter.to_state}"
    if starter.kind == "presence":
        return "when I leave home" if starter.presence == "leave" \
            else "when I arrive home"
    return "on trigger"


def _action_text(action: RoutineAction) -> str:
    if action.service == "notify":
        return f"notify me: \"{action.params.get('message', '')}\""
    if action.service == "activate_scene":
        return f"activate the {action.params.get('scene', '')} scene"
    if action.service == "set_temperature":
        return f"set {action.entity_id} to {action.params.get('temperature')}°"
    verb = {"turn_on": "turn on", "turn_off": "turn off",
            "lock": "lock", "unlock": "unlock"}.get(action.service,
                                                    action.service)
    return f"{verb} {action.entity_id}"


def _condition_text(cond: RoutineCondition) -> str:
    return {"dark": "only when it's dark",
            "light": "only in daylight",
            "weekdays": "only on weekdays",
            "weekends": "only on weekends",
            "home": "only when I'm home",
            "away": "only when I'm away"}.get(cond.kind, cond.kind)


def describe(draft: RoutineDraft) -> str:
    """Plain-language description for the user to confirm."""
    if draft.needs_clarification:
        return draft.clarification
    starter = _starter_text(draft.starter) if draft.starter else "???"
    actions = ", ".join(_action_text(a) for a in draft.actions) or "???"
    conds = "".join(f", {_condition_text(c)}" for c in draft.conditions)
    sentence = f"{starter}{conds}: {actions}."
    sentence = sentence[0].upper() + sentence[1:]
    return f"Here's what I'll do:\n{sentence}"


def confirm(draft: RoutineDraft) -> Routine:
    """Validate and freeze a draft into a Routine. Raises on errors."""
    errors = validate(draft)
    if errors:
        raise RoutineBuildError("; ".join(e.message for e in errors))
    assert draft.starter is not None
    _DRAFTS.pop(draft.id, None)
    return Routine(id=uuid.uuid4().hex[:8], name=draft.name,
                   starter=draft.starter, conditions=draft.conditions,
                   actions=draft.actions)


# ── Home Assistant conversion ───────────────────────────────────────────────


def _service_domain(action: RoutineAction) -> str:
    eid = action.entity_id or ""
    domain = eid.split(".")[0] if "." in eid else ""
    if action.service in ("turn_on", "turn_off"):
        return domain or "homeassistant"
    if action.service in ("lock", "unlock"):
        return "lock"
    if action.service == "set_temperature":
        return "climate"
    return "homeassistant"


def to_ha_automation(routine: Routine) -> tuple[dict, list[dict], list[dict]]:
    """Convert to (trigger, actions, conditions) for create_automation."""
    s = routine.starter
    if s.kind == "time":
        trigger: dict[str, Any] = {"platform": "time", "at": f"{s.at}:00"}
    elif s.kind == "sun":
        trigger = {"platform": "sun", "event": s.event}
        if s.offset_min:
            sign = "-" if s.offset_min < 0 else "+"
            trigger["offset"] = f"{sign}{abs(s.offset_min) // 60:02d}:" \
                f"{abs(s.offset_min) % 60:02d}:00"
    elif s.kind == "device":
        trigger = {"platform": "state", "entity_id": s.entity_id,
                   "to": s.to_state}
    elif s.kind == "presence":
        trigger = {"platform": "state", "entity_id": "person.owner",
                   "to": "not_home" if s.presence == "leave" else "home"}
    else:
        raise RoutineBuildError(f"unknown starter kind {s.kind!r}")

    actions = []
    for a in routine.actions:
        if a.service == "notify":
            actions.append({"service": "notify.notify",
                            "data": {"message": a.params.get("message", "")}})
        elif a.service == "activate_scene":
            scene = a.params.get("scene", "")
            actions.append({"service": "scene.turn_on",
                            "target": {"entity_id": f"scene.{scene}"}})
        elif a.service == "set_temperature":
            actions.append({"service": "climate.set_temperature",
                            "target": {"entity_id": a.entity_id},
                            "data": {"temperature": a.params["temperature"]}})
        else:
            actions.append({"service": f"{_service_domain(a)}.{a.service}",
                            "target": {"entity_id": a.entity_id}})

    conditions = []
    for c in routine.conditions:
        if c.kind == "dark":
            conditions.append({"condition": "state",
                               "entity_id": "sun.sun", "state": "below_horizon"})
        elif c.kind == "light":
            conditions.append({"condition": "state",
                               "entity_id": "sun.sun",
                               "state": "above_horizon"})
        elif c.kind == "weekdays":
            conditions.append({"condition": "time", "weekday": ["mon", "tue",
                               "wed", "thu", "fri"]})
        elif c.kind == "weekends":
            conditions.append({"condition": "time",
                               "weekday": ["sat", "sun"]})
        elif c.kind == "home":
            conditions.append({"condition": "state",
                               "entity_id": "person.owner", "state": "home"})
        elif c.kind == "away":
            conditions.append({"condition": "state",
                               "entity_id": "person.owner",
                               "state": "not_home"})
    return trigger, actions, conditions


async def activate(routine: Routine, integration: Any) -> Routine:
    """Activate via Home Assistant. Returns the routine with the HA id."""
    trigger, actions, conditions = to_ha_automation(routine)
    automation_id = await integration.create_automation(
        routine.name, trigger, actions, conditions=conditions or None)
    routine.ha_automation_id = automation_id
    return routine


def register_local(engine: Any, routine: Routine) -> Any | None:
    """Register a local mirror trigger for observability.

    The real automation lives in Home Assistant; this registers a
    matching trigger in Devon's own engine so the routine is visible
    locally (fires a notification, never controls devices — no
    double-execution).
    """
    if engine is None:
        return None
    try:
        from ...triggers.models import (SOURCE_ENTITY_STATE, SOURCE_SCHEDULE)
    except ImportError:  # pragma: no cover
        from nomorals.triggers.models import (SOURCE_ENTITY_STATE,
                                              SOURCE_SCHEDULE)
    s = routine.starter
    try:
        if s.kind == "time":
            trigger = engine.add(
                name=f"routine: {routine.name}", source=SOURCE_SCHEDULE,
                condition={"kind": "daily", "at": s.at},
                action="notify",
                action_params={"text": f"Routine fired: {routine.name}"},
                cooldown_s=300.0)
        elif s.kind in ("device", "presence"):
            eid = s.entity_id if s.kind == "device" else "person.owner"
            to = s.to_state if s.kind == "device" else \
                ("not_home" if s.presence == "leave" else "home")
            trigger = engine.add(
                name=f"routine: {routine.name}", source=SOURCE_ENTITY_STATE,
                condition={"entity_id": eid, "to": to},
                action="notify",
                action_params={"text": f"Routine fired: {routine.name}"},
                cooldown_s=300.0)
        else:
            return None
        return trigger
    except Exception:  # noqa: BLE001 — local mirror is best-effort
        _log.debug("local routine mirror failed", exc_info=True)
        return None
