"""``nm skill`` — executable skill packages.

New-system verbs (this module): list | install | enable | disable | run |
benchmark.  The legacy knowledge-skill library verbs (create/show/delete/
prune/restore/stats) still live in ``nomorals/cmdline/commands/games.py``;
``nm skill library`` reaches the library's list view.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..emit import _emit


def _registry(context: Any):
    from ...skills.registry import SkillRegistry

    return SkillRegistry(context.db)


def _read_manifest_arg(raw: str) -> dict[str, Any]:
    """--manifest accepts a file path, @path, or inline JSON."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("install needs --manifest <file.json|@file.json|json>")
    if text.startswith("@"):
        text = Path(text[1:]).read_text(encoding="utf-8")
    elif Path(text).is_file():
        text = Path(text).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("manifest JSON must be an object")
    return data


def _cmd_skill_pkg(args: argparse.Namespace, context: Any) -> int:
    """`nm skill <list|install|enable|disable|run|benchmark|library>`."""
    from ...skills.bench import SkillBench
    from ...skills.manifest import ManifestError, SkillManifest
    from ...skills.repair import RepairTicketStore
    from ...skills.runner import SkillRunner

    action = getattr(args, "action", "list") or "list"
    name = (getattr(args, "name", "") or "").strip()
    db = getattr(context, "db", None)
    if db is None:
        print("skill: no database on context", file=sys.stderr)
        return 1
    registry = _registry(context)

    if action == "list":
        items = registry.list()
        payload = {"skills": items}
        if not items:
            _emit(args, payload, "no skill packages installed — "
                  "nm skill install --manifest <file.json>")
            return 0
        lines = []
        for item in items:
            flag = "enabled" if item["enabled"] else "DISABLED"
            lines.append(
                f"  {item['name']:<28} v{item['active_version']} "
                f"[{flag}] tools={len(item['tools'])} "
                f"versions={','.join(item['versions'])}")
            if item["description"]:
                lines.append(f"    {item['description'][:90]}")
        _emit(args, payload, "skill packages:\n" + "\n".join(lines))
        return 0

    if action == "install":
        try:
            data = _read_manifest_arg(getattr(args, "manifest", "") or "")
            manifest = SkillManifest.from_dict(data)
        except (ValueError, ManifestError, OSError) as exc:
            print(f"skill install: {exc}", file=sys.stderr)
            return 1
        try:
            installed = registry.install(manifest)
        except ManifestError as exc:
            print(f"skill install: invalid manifest: {exc}", file=sys.stderr)
            return 1
        _emit(args, installed.to_dict(),
              f"installed skill {installed.name} v{installed.version} "
              f"(active={installed.active}, enabled={installed.enabled})")
        return 0

    if action in ("enable", "disable"):
        if not name:
            print(f"skill {action} needs a name — nm skill {action} <name>",
                  file=sys.stderr)
            return 2
        found = (registry.enable(name) if action == "enable"
                 else registry.disable(name))
        if not found:
            print(f"skill {action}: no installed skill {name!r}",
                  file=sys.stderr)
            return 1
        _emit(args, {"name": name, "enabled": action == "enable"},
              f"skill {name!r} {action}d")
        return 0

    if action == "run":
        if not name:
            print("skill run needs a name — nm skill run <name> "
                  "--input '{...}'", file=sys.stderr)
            return 2
        raw_input = (getattr(args, "input", "") or "").strip()
        try:
            skill_input = json.loads(raw_input) if raw_input else {}
        except ValueError as exc:
            print(f"skill run: --input is not valid JSON: {exc}",
                  file=sys.stderr)
            return 1
        if not isinstance(skill_input, dict):
            print("skill run: --input must be a JSON object",
                  file=sys.stderr)
            return 1
        version = (getattr(args, "version", "") or "").strip() or None
        tools = context.tools.register_builtins()
        runner = SkillRunner(
            registry, tools,
            bench=SkillBench(db), tickets=RepairTicketStore(db),
            actor="cli")
        result = runner.run(name, skill_input, version=version)
        lines = [f"skill {result.name} v{result.version}: "
                 f"{'OK' if result.ok else 'FAILED'} "
                 f"({result.latency_ms:.1f} ms)"]
        for step in result.steps:
            mark = "ok" if step.ok else "FAIL"
            lines.append(f"  [{mark}] step {step.index} {step.tool} "
                         f"({step.latency_ms:.1f} ms)")
            if not step.ok:
                lines.append(f"        error: {step.error[:200]}")
        if result.ticket is not None:
            lines.append(f"  repair ticket {result.ticket.id}: "
                         f"{result.ticket.suggested_fix[:220]}")
        _emit(args, result.to_dict(), "\n".join(lines))
        return 0 if result.ok else 1

    if action == "benchmark":
        bench = SkillBench(db)
        if name:
            summary = bench.score(name)
            if not summary["runs"]:
                _emit(args, summary, f"no benchmark runs recorded for {name!r} — "
                      f"run it first: nm skill run {name}")
                return 0
            _emit(args, summary,
                  f"{name}: {summary['runs']} runs, "
                  f"success {summary['success_rate']:.0%}, "
                  f"avg {summary['avg_latency_ms']:.1f} ms")
            return 0
        overview = bench.overview()
        if not overview:
            _emit(args, {"skills": []}, "no benchmark runs recorded yet — "
                  "nm skill run <name> records one")
            return 0
        lines = ["  skill                        runs  success  avg ms"]
        for summary in overview:
            lines.append(f"  {summary['skill']:<28} {summary['runs']:<5} "
                         f"{summary['success_rate']:<8.0%} "
                         f"{summary['avg_latency_ms']:.1f}")
        _emit(args, {"skills": overview}, "skill benchmarks:\n" +
              "\n".join(lines))
        return 0

    if action == "library":
        # The pre-existing knowledge-skill library list view.
        from .games import _cmd_skill

        lib_args = argparse.Namespace(**{**vars(args), "action": "list"})
        return _cmd_skill(lib_args, context)

    print(f"skill: unknown action {action!r}", file=sys.stderr)
    return 2
