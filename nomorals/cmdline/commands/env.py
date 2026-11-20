"""``nm env`` — environment variables, live from the terminal.

A read-only inspection surface over :mod:`nomorals.core.easyenv`:
what is set, what parses, what the boot actually sees. Values for
secret-named variables are masked by default — ``--show`` reveals
one on this terminal only, and only because the terminal is the
owner's own. There is deliberately NO ``set`` subcommand here: a
fresh ``nm`` process cannot persist an environment change past its
own exit, so a setter would be a lie; in-process callers use
``easyenv.set_env`` / ``easyenv.env_override`` instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ...core import easyenv
from ...core.config import config_env_status

_LIST_CAP = 200


def _out(args: argparse.Namespace, obj: Any) -> int:
    if getattr(args, "json", False):
        print(json.dumps(obj, indent=2))
        return 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{key}: {value}")
    elif isinstance(obj, list):
        for item in obj:
            print(item)
    else:
        print(obj)
    return 0


def _shown(name: str, value: str, args: argparse.Namespace) -> str:
    if getattr(args, "show", False):
        return value
    return easyenv.display_value(name, value)


def _cmd_env(args: argparse.Namespace, context: Any) -> int:
    """Route `nm env` subcommands."""
    action = args.env_action
    try:
        if action == "get":
            name = (getattr(args, "name", "") or "").strip()
            if not name:
                print("get needs a variable name", file=sys.stderr)
                return 2
            info = easyenv.describe(name)
            if not info["present"]:
                print(f"{name}: (not set)")
                return 1
            raw = easyenv.get_env_string_optional(name) or ""
            value = raw if getattr(args, "show", False) else _shown(
                name, raw, args)
            if getattr(args, "json", False):
                return _out(args, {"name": name, "present": True,
                                   "secret": bool(info["secret"]),
                                   "value": value})
            print(f"{name}={value}")
            return 0
        if action == "list":
            prefix = (getattr(args, "prefix", "") or "").strip()
            if getattr(args, "all", False):
                prefix = ""
            elif not prefix:
                prefix = "NM_"
            names = sorted(k for k in os.environ if k.startswith(prefix))
            shown = {k: _shown(k, os.environ[k], args) for k in names}
            if getattr(args, "json", False):
                return _out(args, shown)
            if not names:
                print(f"no variables with prefix {prefix!r}")
                return 0
            if len(names) > _LIST_CAP:
                print(f"{len(names)} variables with prefix {prefix!r} "
                      f"(showing first {_LIST_CAP})")
                names = names[:_LIST_CAP]
            for key in names:
                print(f"{key}={shown[key]}")
            return 0
        if action == "check":
            names = [n.strip() for n in (getattr(args, "names", []) or [])
                     if n.strip()]
            if not names:
                print("check needs at least one variable name",
                      file=sys.stderr)
                return 2
            as_kind = (getattr(args, "as_kind", "string") or "string")
            failures: list[str] = []
            results: dict[str, Any] = {}
            for name in names:
                try:
                    if as_kind == "bool":
                        value = easyenv.get_env_bool_required(name)
                    elif as_kind == "int":
                        value = easyenv.get_env_num_required(name, int)
                    elif as_kind == "float":
                        value = easyenv.get_env_num_required(name, float)
                    elif as_kind == "seconds":
                        value = easyenv.get_env_duration_seconds_required(
                            name)
                    elif as_kind == "path":
                        value = easyenv.get_env_pathbuf_required(name)
                    else:
                        value = easyenv.get_env_string_required(name)
                except easyenv.EnvError as exc:
                    failures.append(f"{name}: {exc}")
                    results[name] = {"ok": False, "error": str(exc)}
                    continue
                shown = (str(value) if not easyenv.is_secret_name(name)
                         else easyenv.mask_secret(str(value)))
                results[name] = {"ok": True, "value": shown}
            if getattr(args, "json", False):
                _out(args, results)
            else:
                for name in names:
                    entry = results[name]
                    if entry["ok"]:
                        print(f"{name}: ok ({entry['value']})")
                    else:
                        print(f"{name}: FAIL — {entry['error']}")
            return 0 if not failures else 1
        if action == "status":
            status = config_env_status()
            summary = {key: len(value)
                       for key, value in status.items()}
            if getattr(args, "json", False):
                return _out(args, {"counts": summary, **status})
            print(f"mapped set: {summary['set_mapped']}, "
                  f"direct-read set: {summary['set_direct']}, "
                  f"mapped unset: {summary['unset_mapped']}, "
                  f"ignored NM_*: {summary['ignored']}")
            if status["ignored"]:
                print("ignored (set but nothing reads them):")
                for key in status["ignored"]:
                    print(f"  {key}")
            return 0
        if action == "dotenv":
            path = (getattr(args, "path", "") or "").strip()
            candidates = ([Path(path)] if path
                          else easyenv.env_file_candidates())
            target = next((c for c in candidates if c.is_file()), None)
            if target is None:
                print("no .env file found "
                      f"({' '.join(str(c) for c in candidates)})")
                return 1
            parsed = easyenv.parse_env_file(target)
            would_set = {k: v for k, v in parsed.items()
                         if k not in os.environ}
            already = {k: v for k, v in parsed.items()
                       if k in os.environ}
            if getattr(args, "json", False):
                return _out(args, {
                    "file": str(target),
                    "would_set": {k: easyenv.display_value(k, v)
                                  for k, v in would_set.items()},
                    "already_set": sorted(already)})
            print(f"{target}: {len(parsed)} keys, "
                  f"{len(would_set)} would load, "
                  f"{len(already)} already set in the environment")
            for key in sorted(would_set):
                print(f"  + {key}="
                      f"{easyenv.display_value(key, would_set[key])}")
            for key in sorted(already):
                print(f"  = {key} (already set)")
            return 0
    except Exception as exc:  # noqa: BLE001 - CLI prints, never traces
        print(f"env {action} failed: {exc}", file=sys.stderr)
        return 1
    print(f"unknown env action {action!r}", file=sys.stderr)
    return 2
