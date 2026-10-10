"""Trigger model and definition validation.

Validation is fail-fast and happens at creation time (``nm trigger add``
or :meth:`TriggerEngine.add`): a bad cron, an unknown source/action, a
broken regex, or missing required fields raise :class:`TriggerError` with
a clear message — never a silent misfire later.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..agents.watchers import parse_condition
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..scheduler.scheduler import CronParser

_log = get_logger(__name__)

#: event sources a trigger can listen to
SOURCE_SCHEDULE = "schedule"
SOURCE_FILE = "file"
SOURCE_PRICE = "price"
SOURCE_MESSAGE = "message"
SOURCE_WEBHOOK = "webhook"
SOURCE_ENTITY_STATE = "entity_state"
SOURCE_BUS = "bus"
SOURCES = frozenset(
    {SOURCE_SCHEDULE, SOURCE_FILE, SOURCE_PRICE, SOURCE_MESSAGE,
     SOURCE_WEBHOOK, SOURCE_ENTITY_STATE, SOURCE_BUS}
)

#: actions a trigger can take
ACTION_NOTIFY = "notify"
ACTION_MESSAGE = "message"
ACTION_COMMAND = "command"
ACTION_MISSION = "mission"
ACTIONS = frozenset({ACTION_NOTIFY, ACTION_MESSAGE, ACTION_COMMAND, ACTION_MISSION})

#: history outcomes
OUTCOME_FIRED = "fired"
OUTCOME_NO_MATCH = "no_match"
OUTCOME_SKIPPED = "skipped"      # disabled / cooldown — evaluated, not fired
OUTCOME_ERROR = "error"


class TriggerError(Exception):
    """Invalid trigger definition or trigger operation failure."""


@dataclass
class Trigger:
    """One event-condition-action rule."""

    id: str
    name: str
    enabled: bool = True
    source: str = SOURCE_SCHEDULE
    condition: dict[str, Any] = field(default_factory=dict)
    action: str = ACTION_NOTIFY
    action_params: dict[str, Any] = field(default_factory=dict)
    cooldown_s: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_fired: float | None = None
    last_outcome: str | None = None
    fire_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "source": self.source,
            "condition": dict(self.condition),
            "action": self.action,
            "action_params": dict(self.action_params),
            "cooldown_s": self.cooldown_s,
            "created_at": self.created_at,
            "last_fired": self.last_fired,
            "last_outcome": self.last_outcome,
            "fire_count": self.fire_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trigger":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or data["id"]),
            enabled=bool(data.get("enabled", True)),
            source=str(data.get("source", SOURCE_SCHEDULE)),
            condition=dict(data.get("condition") or {}),
            action=str(data.get("action", ACTION_NOTIFY)),
            action_params=dict(data.get("action_params") or {}),
            cooldown_s=float(data.get("cooldown_s") or 0.0),
            created_at=float(data.get("created_at") or time.time()),
            last_fired=data.get("last_fired"),
            last_outcome=data.get("last_outcome"),
            fire_count=int(data.get("fire_count") or 0),
        )


def new_trigger_id() -> str:
    return new_id("trigger")


# ── schedule conditions ──────────────────────────────────────────────────

_INTERVAL_RE = re.compile(r"^\s*(\d+)\s*([smhd])\s*$", re.IGNORECASE)
_DAILY_RE = re.compile(r"^\s*(\d{2}):(\d{2})\s*$")
_WEEKLY_RE = re.compile(
    r"^\s*([A-Za-z]{3,9})\s+(\d{2}):(\d{2})\s*$")


def _interval_to_cron(text: str) -> str:
    """Map a human interval (``30m``, ``2h``, ``1d``) onto a cron expression.

    The scheduler's cron engine is minute-granularity, so only intervals
    expressible in 5-field cron are accepted; anything else fails fast
    with a clear reason instead of silently rounding.
    """
    m = _INTERVAL_RE.match(text or "")
    if not m:
        raise TriggerError(
            f"bad interval {text!r}: expected like '30s', '15m', '2h', '1d'")
    n, unit = int(m.group(1)), m.group(2).lower()
    if n <= 0:
        raise TriggerError(f"bad interval {text!r}: must be positive")
    if unit == "s":
        if n % 60:
            raise TriggerError(
                f"bad interval {text!r}: sub-minute intervals are not "
                "supported (cron is minute-granularity)")
        n, unit = n // 60, "m"
    if unit == "m":
        if n <= 59:
            cron = f"*/{n} * * * *"
        elif n == 60:
            cron = "0 * * * *"
        elif n < 1440 and n % 60 == 0 and n // 60 <= 23:
            cron = f"0 */{n // 60} * * *"
        elif n == 1440:
            cron = "0 0 * * *"
        else:
            raise TriggerError(
                f"bad interval {text!r}: {n}m is not expressible as cron "
                "(use minutes ≤ 59, whole hours, or 1d)")
    elif unit == "h":
        if n <= 23:
            cron = f"0 */{n} * * *"
        elif n == 24:
            cron = "0 0 * * *"
        else:
            raise TriggerError(
                f"bad interval {text!r}: use hours ≤ 24")
    else:  # d
        if n == 1:
            cron = "0 0 * * *"
        elif n == 7:
            cron = "0 0 * * SUN"
        else:
            raise TriggerError(
                f"bad interval {text!r}: day intervals must be 1d or 7d")
    # double-check the generated expression parses
    try:
        CronParser.parse(cron)
    except ValueError as exc:  # pragma: no cover - generator bug guard
        raise TriggerError(f"internal: generated bad cron {cron!r}: {exc}")
    return cron


def _daily_to_cron(text: str) -> str:
    m = _DAILY_RE.match(text or "")
    if not m:
        raise TriggerError(
            f"bad daily time {text!r}: expected 'HH:MM', e.g. '09:30'")
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        raise TriggerError(f"bad daily time {text!r}: out of range")
    return f"{minute} {hour} * * *"


def _weekly_to_cron(text: str) -> str:
    m = _WEEKLY_RE.match(text or "")
    if not m:
        raise TriggerError(
            f"bad weekly time {text!r}: expected 'DAY HH:MM', e.g. 'MON 09:30'")
    day = m.group(1).upper()[:3]
    if day not in CronParser.DAY_NAMES:
        raise TriggerError(
            f"bad weekly day {m.group(1)!r}: expected one of "
            f"{sorted(CronParser.DAY_NAMES)}")
    hour, minute = int(m.group(2)), int(m.group(3))
    if hour > 23 or minute > 59:
        raise TriggerError(f"bad weekly time {text!r}: out of range")
    return f"{minute} {hour} * * {day}"


def _once_to_ts(value: Any) -> float:
    if isinstance(value, (int, float)):
        ts = float(value)
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value.strip()).timestamp()
        except ValueError:
            raise TriggerError(
                f"bad once time {value!r}: expected epoch seconds or "
                "ISO datetime, e.g. '2026-10-03T09:00:00'")
    else:
        raise TriggerError(
            f"bad once time {value!r}: expected epoch seconds or ISO datetime")
    if ts <= time.time():
        raise TriggerError(
            f"bad once time {value!r}: it is in the past")
    return ts


def normalize_schedule_condition(condition: dict[str, Any]) -> dict[str, Any]:
    """Validate a schedule condition and normalize it to ``cron`` or ``once``.

    Accepted forms (exactly one): ``cron`` (validated by the scheduler's
    own ``CronParser``), ``interval`` (``30m``/``2h``/``1d`` …), ``daily``
    (``HH:MM``), ``weekly`` (``DAY HH:MM``), ``once`` (epoch/ISO future).
    """
    keys = [k for k in ("cron", "interval", "daily", "weekly", "once")
            if condition.get(k) not in (None, "")]
    if len(keys) != 1:
        raise TriggerError(
            "schedule condition needs exactly one of "
            "'cron' | 'interval' | 'daily' | 'weekly' | 'once'; "
            f"got {keys}")
    kind = keys[0]
    if kind == "once":
        return {"once": _once_to_ts(condition["once"])}
    if kind == "cron":
        expr = str(condition["cron"]).strip()
        try:
            CronParser.parse(expr)
        except ValueError as exc:
            raise TriggerError(f"bad cron {expr!r}: {exc}")
        return {"cron": expr}
    if kind == "interval":
        return {"cron": _interval_to_cron(str(condition["interval"]))}
    if kind == "daily":
        return {"cron": _daily_to_cron(str(condition["daily"]))}
    return {"cron": _weekly_to_cron(str(condition["weekly"]))}


# ── per-source validation ────────────────────────────────────────────────

def _validate_file(condition: dict[str, Any]) -> dict[str, Any]:
    path = str(condition.get("path") or "").strip()
    if not path:
        raise TriggerError("file condition needs 'path'")
    on = str(condition.get("on") or "change").strip().lower()
    if on not in ("change", "create", "delete"):
        raise TriggerError(
            f"bad file watch mode {on!r}: expected change|create|delete")
    return {"path": path, "on": on}


def _validate_price(condition: dict[str, Any]) -> dict[str, Any]:
    symbol = str(condition.get("symbol") or "").strip().upper()
    if not symbol:
        raise TriggerError("price condition needs 'symbol' (e.g. 'BTC')")
    market = str(condition.get("market") or "crypto").strip().lower()
    if market not in ("crypto", "stocks", "forex"):
        raise TriggerError(
            f"bad price market {market!r}: expected crypto|stocks|forex")
    op = str(condition.get("op") or "lt").strip().lower()
    try:
        parsed = parse_condition(
            {"op": op, "field": "value", "value": condition.get("value")})
    except ValueError as exc:
        raise TriggerError(f"bad price condition: {exc}")
    if parsed.op in ("lt", "lte", "gt", "gte", "eq", "ne", "changed_by_pct",
                    "contains", "matches") and condition.get("value") is None:
        raise TriggerError(
            f"price op {parsed.op!r} needs a numeric 'value'")
    return {"symbol": symbol, "market": market, "op": parsed.op,
            "value": condition.get("value")}


def _validate_message(condition: dict[str, Any]) -> dict[str, Any]:
    pattern = str(condition.get("pattern") or "")
    if not pattern:
        raise TriggerError("message condition needs 'pattern' (a regex)")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise TriggerError(f"bad message pattern {pattern!r}: {exc}")
    out = {"pattern": pattern}
    chat = str(condition.get("chat") or "").strip()
    if chat:
        out["chat"] = chat
    sender = str(condition.get("sender") or "").strip()
    if sender:
        out["sender"] = sender
    return out


def _validate_webhook(condition: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    secret = condition.get("secret")
    if secret not in (None, ""):
        out["secret"] = str(secret)
    return out


# ── per-action validation ────────────────────────────────────────────────

def _validate_action_params(action: str,
                            params: dict[str, Any]) -> dict[str, Any]:
    params = dict(params or {})
    if action == ACTION_MESSAGE:
        if not str(params.get("chat") or "").strip():
            raise TriggerError("message action needs 'chat' (chat key)")
        if not str(params.get("text") or ""):
            raise TriggerError("message action needs 'text'")
    elif action == ACTION_COMMAND:
        argv, command = params.get("argv"), params.get("command")
        if argv is not None and command:
            raise TriggerError(
                "command action takes 'argv' OR 'command', not both")
        if argv is not None:
            if (not isinstance(argv, list) or not argv
                    or not all(isinstance(a, str) for a in argv)):
                raise TriggerError(
                    "command action 'argv' must be a non-empty list of strings")
        elif not str(command or "").strip():
            raise TriggerError(
                "command action needs 'argv' (list) or 'command' (string)")
    elif action == ACTION_MISSION:
        if not str(params.get("goal") or "").strip():
            raise TriggerError("mission action needs 'goal'")
        mi = params.get("max_iterations")
        if mi is not None and (not isinstance(mi, int) or mi < 1):
            raise TriggerError(
                "mission action 'max_iterations' must be a positive int")
    elif action == ACTION_NOTIFY:
        pass  # title/body optional; title defaults to the trigger name
    else:  # pragma: no cover - guarded by the ACTIONS check above
        raise TriggerError(f"unknown action {action!r}")
    return params


def _validate_bus(condition: dict[str, Any]) -> dict[str, Any]:
    """Validate a bus-source condition.

    ``topic`` (required): a glob the event topic must match, e.g.
    ``"scheduler.job.finished"`` or ``"mission.*"``.  ``match`` (optional):
    a dict of ``event.data`` key/value pairs that must all be present and
    equal (subset match).  ``source`` (optional): the event's source
    module must equal it.
    """
    import fnmatch as _fnmatch

    topic = str(condition.get("topic") or "").strip()
    if not topic:
        raise TriggerError("bus condition needs 'topic' (a glob, e.g. "
                           "'scheduler.job.finished' or 'mission.*')")
    # fail fast on a glob that can never match anything sane
    try:
        _fnmatch.fnmatchcase("", topic)
    except Exception as exc:  # noqa: BLE001 - defensive; fnmatch rarely raises
        raise TriggerError(f"bad bus topic glob {topic!r}: {exc}")
    out: dict[str, Any] = {"topic": topic}
    match = condition.get("match")
    if match is not None:
        if not isinstance(match, dict):
            raise TriggerError("bus condition 'match' must be an object")
        out["match"] = {str(k): v for k, v in match.items()}
    source = str(condition.get("source") or "").strip()
    if source:
        out["source"] = source
    return out


def _validate_entity_state(condition: dict[str, Any]) -> dict[str, Any]:
    """Validate an entity_state (Home Assistant state_changed) condition.

    Optional filters (all ANDed): ``domain`` (e.g. "light"), ``entity_id``
    (e.g. "light.kitchen"), ``to`` (new state, e.g. "on"), ``from``
    (previous state).  An empty condition matches every state change.
    """
    out: dict[str, Any] = {}
    domain = str(condition.get("domain") or "").strip().lower()
    if domain:
        out["domain"] = domain
    entity_id = str(condition.get("entity_id") or "").strip().lower()
    if entity_id:
        out["entity_id"] = entity_id
    to_state = condition.get("to")
    if to_state is not None and str(to_state) != "":
        out["to"] = str(to_state)
    from_state = condition.get("from")
    if from_state is not None and str(from_state) != "":
        out["from"] = str(from_state)
    return out


def validate_definition(
    source: str,
    condition: dict[str, Any] | None,
    action: str,
    action_params: dict[str, Any] | None,
    *,
    cooldown_s: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Validate a trigger definition; return normalized (condition, params, cooldown).

    Raises :class:`TriggerError` on anything invalid — unknown source or
    action, malformed schedule/price/message conditions, bad action params.
    """
    if source not in SOURCES:
        raise TriggerError(
            f"unknown source {source!r}; expected one of {sorted(SOURCES)}")
    if action not in ACTIONS:
        raise TriggerError(
            f"unknown action {action!r}; expected one of {sorted(ACTIONS)}")
    condition = dict(condition or {})
    if source == SOURCE_SCHEDULE:
        condition = normalize_schedule_condition(condition)
    elif source == SOURCE_FILE:
        condition = _validate_file(condition)
    elif source == SOURCE_PRICE:
        condition = _validate_price(condition)
    elif source == SOURCE_MESSAGE:
        condition = _validate_message(condition)
    elif source == SOURCE_WEBHOOK:
        condition = _validate_webhook(condition)
    elif source == SOURCE_ENTITY_STATE:
        condition = _validate_entity_state(condition)
    elif source == SOURCE_BUS:
        condition = _validate_bus(condition)
    params = _validate_action_params(action, action_params)
    try:
        cooldown = float(cooldown_s or 0.0)
    except (TypeError, ValueError):
        raise TriggerError(
            f"bad cooldown {cooldown_s!r}: must be seconds as a number")
    if cooldown < 0:
        raise TriggerError("cooldown must be >= 0")
    return condition, params, cooldown
