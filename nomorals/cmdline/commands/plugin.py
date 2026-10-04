"""``nm plugin`` — install, list, enable/disable, remove, run plugins."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _registry(context: Any) -> Any:
    from ...plugins import PluginRegistry
    settings = getattr(context, "settings", None)
    root = getattr(settings, "workspace_dir", None) if settings else None
    base = Path(root) if root else Path.cwd() / "workspace"
    return PluginRegistry(context.db, base / "plugins")


def _cmd_plugin(args: Any, context: Any) -> int:
    """Route ``nm plugin <verb>``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm plugin install <path|url|zip>\n"
              "       nm plugin list [--json]\n"
              "       nm plugin info <name> [--json]\n"
              "       nm plugin enable|disable <name>\n"
              "       nm plugin remove <name>\n"
              "       nm plugin run <name> [entry] [--json]",
              file=sys.stderr)
        return 2
    verb, rest = words[0], words[1:]
    if verb == "install":
        return _pl_install(args, context, rest)
    if verb == "list":
        return _pl_list(args, context)
    if verb == "info":
        return _pl_info(args, context, rest)
    if verb == "enable":
        return _pl_enable(args, context, rest)
    if verb == "disable":
        return _pl_disable(args, context, rest)
    if verb == "remove":
        return _pl_remove(args, context, rest)
    if verb == "run":
        return _pl_run(args, context, rest)
    print(f"unknown plugin verb: {verb}", file=sys.stderr)
    return 2


def _as_json(args: Any) -> bool:
    return bool(getattr(args, "json", False))


def _pl_install(args: Any, context: Any, rest: list[str]) -> int:
    from ...plugins import PluginError
    if not rest:
        print("usage: nm plugin install <path|url|zip>", file=sys.stderr)
        return 2
    try:
        plugin = _registry(context).install(rest[0])
    except PluginError as exc:
        print(f"install failed: {exc}", file=sys.stderr)
        return 1
    print(f"installed {plugin.name} {plugin.version}")
    if plugin.manifest.permissions:
        print("  permissions: " + ", ".join(plugin.manifest.permissions))
    print("  entry points: " + ", ".join(
        f"{k}={v}" for k, v in plugin.manifest.entry_points.items()))
    return 0


def _pl_list(args: Any, context: Any) -> int:
    plugins = _registry(context).list()
    if _as_json(args):
        print(json.dumps([p.to_dict() for p in plugins], indent=2))
        return 0
    if not plugins:
        print("no plugins installed. `nm plugin install <path|url|zip>`")
        return 0
    for p in plugins:
        mark = "on " if p.enabled else "off"
        print(f"  [{mark}] {p.name} {p.version}  "
              f"{p.manifest.description[:60]}")
    return 0


def _pl_info(args: Any, context: Any, rest: list[str]) -> int:
    from ...plugins import PluginError
    if not rest:
        print("usage: nm plugin info <name>", file=sys.stderr)
        return 2
    try:
        plugin = _registry(context).get(rest[0])
    except PluginError as exc:
        print(f"info failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(plugin.to_dict(), indent=2))
    return 0


def _pl_enable(args: Any, context: Any, rest: list[str]) -> int:
    from ...plugins import PluginError
    if not rest:
        print("usage: nm plugin enable <name>", file=sys.stderr)
        return 2
    try:
        plugin = _registry(context).enable(rest[0])
    except PluginError as exc:
        print(f"enable failed: {exc}", file=sys.stderr)
        return 1
    print(f"enabled {plugin.name} {plugin.version}")
    return 0


def _pl_disable(args: Any, context: Any, rest: list[str]) -> int:
    from ...plugins import PluginError
    if not rest:
        print("usage: nm plugin disable <name>", file=sys.stderr)
        return 2
    try:
        plugin = _registry(context).disable(rest[0])
    except PluginError as exc:
        print(f"disable failed: {exc}", file=sys.stderr)
        return 1
    print(f"disabled {plugin.name} {plugin.version}")
    return 0


def _pl_remove(args: Any, context: Any, rest: list[str]) -> int:
    from ...plugins import PluginError
    if not rest:
        print("usage: nm plugin remove <name>", file=sys.stderr)
        return 2
    try:
        _registry(context).remove(rest[0])
    except PluginError as exc:
        print(f"remove failed: {exc}", file=sys.stderr)
        return 1
    print(f"removed {rest[0]}")
    return 0


def _pl_run(args: Any, context: Any, rest: list[str]) -> int:
    """Run a plugin entry point with gated capabilities."""
    from ...plugins import (
        PluginError,
        load_plugin,
        unload_plugin,
        wire_capabilities,
    )
    if not rest:
        print("usage: nm plugin run <name> [entry]", file=sys.stderr)
        return 2
    name = rest[0]
    entry = rest[1] if len(rest) > 1 else "main"
    try:
        reg = _registry(context)
        plugin = reg.get(name)
        if not plugin.enabled:
            print(f"plugin {name!r} is disabled; enable it first.",
                  file=sys.stderr)
            return 1
        loaded = load_plugin(plugin.manifest, plugin.path)
        try:
            caps = wire_capabilities(context, plugin)
            result = loaded.entry(entry, caps)
        finally:
            unload_plugin(loaded)
    except PluginError as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 1
    if _as_json(args):
        print(json.dumps(result, indent=2, default=str))
    elif result is not None:
        print(result)
    return 0
