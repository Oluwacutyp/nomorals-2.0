"""Niche registry — add new niches without touching core."""

from __future__ import annotations

import importlib
import inspect
import re

from .base import NichePlugin

_REGISTRY: dict[str, NichePlugin] = {}
_BOOTSTRAPPED = False

_SHIPPED_MODULES = (
    "motivation",
    "finance_facts",
    "horror_stories",
    "reddit_stories",
    "did_you_know",
    "sports_edits",
    "anime_edits",
)


class NicheError(ValueError):
    """Raised when a plugin is malformed or a name is unknown."""


def validate(plugin: NichePlugin) -> list[str]:
    """Return a list of problems with ``plugin``; empty means valid."""
    issues: list[str] = []
    if not isinstance(plugin, NichePlugin):
        return [f"not a NichePlugin instance: {type(plugin).__name__}"]
    if not plugin.name or not str(plugin.name).strip():
        issues.append("name must be a non-empty slug")
    elif not re.match(r"^[a-z0-9_]+$", plugin.name):
        issues.append(f"name {plugin.name!r} must be lowercase slug (a-z0-9_)")
    if not (plugin.thesis or "").strip():
        issues.append("thesis must be non-empty")
    try:
        if float(plugin.cadence) <= 0:
            issues.append("cadence must be > 0 posts/day")
    except (TypeError, ValueError):
        issues.append("cadence must be a number > 0")
    for attr in ("title_template", "description_template"):
        if not (getattr(plugin, attr, "") or "").strip():
            issues.append(f"{attr} must be non-empty")
    if "{topic}" not in (plugin.title_template or ""):
        issues.append("title_template must contain a {topic} placeholder")
    tags = plugin.hashtags
    if isinstance(tags, dict):
        flat = [t for v in tags.values() for t in v]
    elif isinstance(tags, (list, tuple)):
        flat = list(tags)
    else:
        flat = []
        issues.append("hashtags must be a list or a platform→list dict")
    if not flat:
        issues.append("hashtags must contain at least one tag")
    for t in flat:
        if not str(t).startswith("#"):
            issues.append(f"hashtag {t!r} must start with '#'")
            break
    vs = getattr(plugin, "voice_spec", None)
    if vs is None or not hasattr(vs, "resolve"):
        issues.append("voice_spec must be a VoiceSpec (with .resolve)")
    if not (plugin.ypp_rationale or "").strip():
        issues.append("ypp_rationale must be non-empty")
    # script_prompt / visual_strategy must be real implementations
    for meth in ("script_prompt", "visual_strategy"):
        fn = getattr(plugin, meth, None)
        if not callable(fn):
            issues.append(f"{meth} must be callable")
            continue
        owner = getattr(fn, "__qualname__", "")
        if "NichePlugin" in owner and not hasattr(type(plugin), meth):
            issues.append(f"{meth} must be implemented (abstract on base)")
    try:
        prompt = plugin.script_prompt("sample topic") if callable(getattr(plugin, "script_prompt", None)) else ""
    except Exception as exc:  # noqa: BLE001 — validation, surface as issue
        issues.append(f"script_prompt raised on a sample topic: {exc}")
    else:
        if not prompt or len(prompt.strip()) < 200:
            issues.append("script_prompt must return a substantive prompt (≥200 chars)")
    return issues


def register(plugin: NichePlugin) -> NichePlugin:
    """Register a niche plugin. Raises :class:`NicheError` if malformed."""
    issues = validate(plugin)
    if issues:
        raise NicheError(
            f"invalid niche plugin {getattr(plugin, 'name', '?')!r}: "
            + "; ".join(issues))
    name = plugin.name
    if name in _REGISTRY and _REGISTRY[name] is not plugin:
        raise NicheError(f"niche {name!r} is already registered")
    _REGISTRY[name] = plugin
    return plugin


def _bootstrap() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    for mod in _SHIPPED_MODULES:
        try:
            module = importlib.import_module(f"{__package__}.{mod}")
        except Exception:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, NichePlugin) and obj is not NichePlugin
                    and obj.__module__ == module.__name__):
                try:
                    register(obj())
                except NicheError:
                    pass
    _BOOTSTRAPPED = True


def get(name: str) -> NichePlugin:
    """Return the registered niche. Raises :class:`NicheError` if unknown."""
    _bootstrap()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise NicheError(
            f"unknown niche {name!r}; available: {', '.join(sorted(_REGISTRY))}") from None


def list_niches() -> list[str]:
    """Sorted list of registered niche names."""
    _bootstrap()
    return sorted(_REGISTRY)


def all_plugins() -> list[NichePlugin]:
    _bootstrap()
    return [_REGISTRY[n] for n in sorted(_REGISTRY)]


def get_niche(name: str) -> NichePlugin:
    """Pipeline contract: return the plugin for ``name``.

    Falls back to the generic shim plugin for unknown names (never raises),
    so the short-form pipeline keeps working for ad-hoc niche names.
    """
    _bootstrap()
    try:
        return _REGISTRY[name]
    except KeyError:
        pass
    try:
        from .._shims import get_niche as _shim_get_niche
        return _shim_get_niche(name)
    except Exception:
        raise NicheError(
            f"unknown niche {name!r}; available: {', '.join(sorted(_REGISTRY))}") from None
