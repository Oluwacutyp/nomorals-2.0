"""Central profile-gated tuning values (wave: profile-gating).

Every hardcoded limit in the codebase should read from here instead of
being a magic number. Profiles:

- ``termux`` / ``mobile`` / ``embedded`` — lean: phone-safe defaults
- ``laptop`` / ``pc`` / ``vps`` — balanced defaults
- ``workstation`` — full capability

Detection reuses :mod:`nomorals.core.profile` (``detect_profile``), which
already handles Termux/Android heuristics. ``NM_PROFILE`` env var pins
a profile explicitly (overrides detection).

Usage::

    from nomorals.core.profiles import get_profile, profile_value

    prof = get_profile()               # dict of tuning values
    ctx = profile_value("ctx_size")    # single value, current profile
"""
from __future__ import annotations

import os
from typing import Any

from .profile import detect_profile

__all__ = ["PROFILES", "get_profile", "get_profile_kind", "profile_value",
           "KNOWN_PROFILE_NAMES"]

#: profile name → tuning values. ``threads: 0`` means auto-detect all cores.
PROFILES: dict[str, dict[str, Any]] = {
    "termux": {
        "ctx_size": 2048,
        "threads": 4,
        "step_budget": 10,
        "max_parallel": 4,
        "obs_chars": 2000,
        "max_tools_in_prompt": 40,
        "max_moves": 50,
        "daily_delivery_cap": 3,
        "research_workers": 2,
        "max_results": 6,
        "fetch_top": 2,
        "history_turns": 6,
        "idle_timeout": 60.0,
        "tick_seconds": 300.0,
        "max_tokens": 1024,
        "assist_max_inflight": 2,
        "summary_chars": 2000,
        "detail_chars": 2000,
        "plan_chars": 500,
        "lesson_chars": 300,
        "max_lessons": 8,
    },
    "laptop": {
        "ctx_size": 4096,
        "threads": 8,
        "step_budget": 15,
        "max_parallel": 8,
        "obs_chars": 4000,
        "max_tools_in_prompt": 60,
        "max_moves": 100,
        "daily_delivery_cap": 5,
        "research_workers": 3,
        "max_results": 10,
        "fetch_top": 3,
        "history_turns": 10,
        "idle_timeout": 120.0,
        "tick_seconds": 180.0,
        "max_tokens": 2048,
        "assist_max_inflight": 4,
        "summary_chars": 4000,
        "detail_chars": 4000,
        "plan_chars": 1000,
        "lesson_chars": 500,
        "max_lessons": 12,
    },
    "workstation": {
        "ctx_size": 8192,
        "threads": 0,  # 0 = auto-detect all cores
        "step_budget": 25,
        "max_parallel": 16,
        "obs_chars": 8000,
        "max_tools_in_prompt": 80,
        "max_moves": 200,
        "daily_delivery_cap": 10,
        "research_workers": 5,
        "max_results": 15,
        "fetch_top": 5,
        "history_turns": 15,
        "idle_timeout": 300.0,
        "tick_seconds": 120.0,
        "max_tokens": 4096,
        "assist_max_inflight": 8,
        "summary_chars": 8000,
        "detail_chars": 8000,
        "plan_chars": 2000,
        "lesson_chars": 800,
        "max_lessons": 16,
    },
}

#: detected kind → PROFILES key. ``mobile``/``embedded`` are lean like
#: termux; ``pc``/``vps`` are balanced like laptop.
_KIND_MAP: dict[str, str] = {
    "termux": "termux",
    "mobile": "termux",
    "embedded": "termux",
    "pc": "laptop",
    "laptop": "laptop",
    "vps": "laptop",
    "workstation": "workstation",
}

KNOWN_PROFILE_NAMES: list[str] = sorted(set(_KIND_MAP.values()))


def get_profile_kind() -> str:
    """Current profile key (``termux`` | ``laptop`` | ``workstation``).

    Precedence: ``NM_PROFILE`` env var → Termux ``PREFIX`` heuristic →
    :func:`nomorals.core.profile.detect_profile`.
    """
    pinned = (os.environ.get("NM_PROFILE") or "").strip().lower()
    if pinned in PROFILES:
        return pinned
    if pinned in _KIND_MAP:
        return _KIND_MAP[pinned]
    # Fast Termux check before the heavier detection.
    try:
        if os.environ.get("PREFIX", "").startswith("/data/data/com.termux"):
            return "termux"
    except Exception:  # noqa: BLE001 - detection is best-effort
        pass
    try:
        detected = detect_profile().kind
    except Exception:  # noqa: BLE001
        detected = "pc"
    return _KIND_MAP.get(detected, "laptop")


def get_profile(kind: str = "") -> dict[str, Any]:
    """Tuning values for ``kind`` (or the current profile when empty).

    Returns a copy — mutate freely, it won't affect other callers.
    """
    key = (kind or "").strip().lower()
    if not key:
        key = get_profile_kind()
    key = _KIND_MAP.get(key, key)
    return dict(PROFILES.get(key, PROFILES["laptop"]))


def profile_value(name: str, default: Any = None, kind: str = "") -> Any:
    """Single tuning value for the current profile."""
    return get_profile(kind).get(name, default)
