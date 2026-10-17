"""Per-chat style overrides — the TUNE half of the style system (spec §3.4).

The mined style profile is computed per reply (``style_for_chat``); this
store holds the owner's explicit overrides as DELTAS on top of it, never
by editing the miner's output. Overrides are per-scope (a chat key, or
``"global"``), versioned, and reversible: ``/style reset`` clears them.

Storage follows the chat-modes pattern exactly: a JSON file with atomic
write (temp + rename), corrupt → defaults (never crash), env override
``NOMORALS_STYLE_OVERRIDES_PATH`` for tests.

Each override carries provenance (who set it, when) so ``/style`` can
show it — no hidden tuning the owner cannot see or reverse.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from typing import Any

_log = logging.getLogger("nomorals.partner.style_overrides")

STORE_VERSION = 1

#: Overrideable style fields -> (kind, min, max). Only these fields may be
#: overridden; anything else is rejected (the owner tunes numbers, never
#: tone labels — there are no tone labels to tune).
FIELDS: dict[str, tuple[str, Any, Any]] = {
    "emoji_cap": ("int", 0, 3),
    "target_chars": ("int", 40, 4000),
    "max_chars": ("int", 80, 4096),
    "allow_sections": ("bool", False, True),
    "allow_lists": ("bool", False, True),
    "allow_markup": ("bool", False, True),
}

GLOBAL_SCOPE = "global"


def _store_path(path: str = "") -> str:
    if path:
        return os.path.expanduser(path)
    override = os.environ.get("NOMORALS_STYLE_OVERRIDES_PATH", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".config",
                        "nomorals", "style_overrides.json")


def _load(path: str = "") -> dict[str, Any]:
    try:
        with open(_store_path(path), encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(state: dict[str, Any], path: str = "") -> None:
    try:
        target = _store_path(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, target)
    except OSError as exc:
        _log.warning("style overrides save failed: %s", exc)


def _coerce(field: str, value: Any) -> Any:
    kind, lo, hi = FIELDS[field]
    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{field} wants on/off, got {value!r}")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{field} wants an integer, got {value!r}") from None
    if number < lo or number > hi:
        raise ValueError(f"{field} wants {lo}..{hi}, got {number}")
    return number


def set_override(scope: str, field: str, value: Any, *,
                 set_by: str = "owner", path: str = "") -> dict[str, Any]:
    """Set one override delta. Returns the scope's override map.

    Raises :class:`ValueError` for an unknown field, an out-of-range
    value, or an empty scope.
    """
    field = (field or "").strip()
    if field not in FIELDS:
        raise ValueError(
            f"unknown style field {field!r} — want one of: {', '.join(sorted(FIELDS))}")
    scope = (scope or "").strip()
    if not scope:
        raise ValueError("scope is required (a chat key, or 'global')")
    coerced = _coerce(field, value)
    state = _load(path)
    scopes = state.get("scopes", {})
    if not isinstance(scopes, dict):
        scopes = {}
    entry = scopes.get(scope, {})
    if not isinstance(entry, dict):
        entry = {}
    entry[field] = {"value": coerced, "set_by": set_by, "set_at": time.time()}
    scopes[scope] = entry
    state["scopes"] = scopes
    state["version"] = STORE_VERSION
    _save(state, path)
    return {k: v["value"] for k, v in entry.items()}


def get_overrides(scope: str, path: str = "") -> dict[str, Any]:
    """A scope's override deltas (plain field -> value), merged global-first.

    A chat scope inherits the ``global`` overrides, then its own win.
    """
    scopes = _load(path).get("scopes", {})
    if not isinstance(scopes, dict):
        return {}
    merged: dict[str, Any] = {}
    for scope_name in (GLOBAL_SCOPE, (scope or "").strip()):
        entry = scopes.get(scope_name)
        if isinstance(entry, dict):
            for field, record in entry.items():
                if field in FIELDS and isinstance(record, dict) and "value" in record:
                    merged[field] = record["value"]
    return merged


def get_overrides_detailed(scope: str, path: str = "") -> dict[str, dict[str, Any]]:
    """A scope's overrides with provenance (value + set_by + set_at)."""
    scopes = _load(path).get("scopes", {})
    if not isinstance(scopes, dict):
        return {}
    merged: dict[str, dict[str, Any]] = {}
    for scope_name in (GLOBAL_SCOPE, (scope or "").strip()):
        entry = scopes.get(scope_name)
        if isinstance(entry, dict):
            for field, record in entry.items():
                if field in FIELDS and isinstance(record, dict) and "value" in record:
                    merged[field] = {"value": record["value"],
                                     "set_by": record.get("set_by", "?"),
                                     "set_at": record.get("set_at", 0),
                                     "scope": scope_name}
    return merged


def clear_overrides(scope: str, path: str = "") -> bool:
    """Clear one scope's overrides. Returns True when anything was cleared."""
    scope = (scope or "").strip()
    if not scope:
        raise ValueError("scope is required")
    state = _load(path)
    scopes = state.get("scopes", {})
    if not isinstance(scopes, dict) or scope not in scopes:
        return False
    del scopes[scope]
    state["scopes"] = scopes
    _save(state, path)
    return True


def list_scopes(path: str = "") -> list[str]:
    """Scopes with overrides set."""
    scopes = _load(path).get("scopes", {})
    return sorted(scopes) if isinstance(scopes, dict) else []


def apply_overrides(profile: Any, overrides: dict[str, Any] | None) -> Any:
    """Apply override deltas to a StyleProfile (returns a new profile).

    Field mapping: emoji_cap -> emoji_budget, target_chars, max_chars,
    allow_sections, allow_lists, allow_markup. Unknown fields are
    ignored (the store never writes them, but defence in depth).
    """
    if not overrides:
        return profile
    changes: dict[str, Any] = {}
    if "emoji_cap" in overrides:
        changes["emoji_budget"] = max(0, min(3, int(overrides["emoji_cap"])))
    if "target_chars" in overrides:
        changes["target_chars"] = max(40, min(int(overrides["target_chars"]),
                                               int(profile.max_chars)))
    if "max_chars" in overrides:
        changes["max_chars"] = max(80, min(4096, int(overrides["max_chars"])))
        changes["target_chars"] = min(changes.get("target_chars", int(profile.target_chars)),
                                      changes["max_chars"])
    for field in ("allow_sections", "allow_lists", "allow_markup"):
        if field in overrides:
            changes[field] = bool(overrides[field])
    if not changes:
        return profile
    return dataclasses.replace(profile, **changes)
