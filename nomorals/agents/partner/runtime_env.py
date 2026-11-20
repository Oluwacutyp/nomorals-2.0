"""RuntimeEnvMixin: PartnerRuntime command group (environment)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ...core import easyenv
from ...core.config import config_env_status
from ...core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)

_LIST_CAP = 60


class RuntimeEnvMixin:
    """RuntimeEnvMixin for :class:`PartnerRuntime`."""

    # ── environment: /env ─────────────────────────────────────────────
    def _control_env(self, tail: str, chat_key: str,
                     message: Any = None) -> str:
        """Environment variables — read-only inspection.

        /env get <NAME>              one variable (masked if secret)
        /env list [PREFIX]           variables by prefix (default NM_)
        /env check <NAME...>         required-var check, fails by name
        /env check --as bool <NAME>  ...and parse as bool/int/float/
                                     seconds/path
        /env status                  mapped / direct / ignored NM_* counts
        /env dotenv [PATH]           what a .env file would load

        Chat is a shared surface, so secret values are ALWAYS masked
        here — there is no --show in chat. The terminal CLI (`nm env
        get --show`) is the only place a secret is ever revealed.
        """
        parts = (tail or "").split()
        if not parts:
            return self._control_env_help()
        verb = parts[0]
        rest = parts[1:]
        try:
            if verb == "get":
                if not rest:
                    return "env get needs a variable name"
                name = rest[0]
                info = easyenv.describe(name)
                if not info["present"]:
                    return f"{name}: (not set)"
                return f"{name}={info['value']}"
            if verb == "list":
                prefix = (rest[0] if rest else "") or "NM_"
                names = sorted(k for k in os.environ
                               if k.startswith(prefix))
                if not names:
                    return f"no variables with prefix {prefix!r}"
                lines = [f"{len(names)} variables with prefix "
                         f"{prefix!r}:"]
                for key in names[:_LIST_CAP]:
                    lines.append(
                        f"  {key}="
                        f"{easyenv.display_value(key, os.environ[key])}")
                if len(names) > _LIST_CAP:
                    lines.append(f"  …and {len(names) - _LIST_CAP} more")
                return "\n".join(lines)
            if verb == "check":
                names: list[str] = []
                as_kind = "string"
                i = 0
                while i < len(rest):
                    if rest[i] == "--as" and i + 1 < len(rest):
                        as_kind = rest[i + 1]
                        i += 2
                        continue
                    names.append(rest[i])
                    i += 1
                if not names:
                    return "env check needs at least one variable name"
                if as_kind not in ("string", "bool", "int", "float",
                                   "seconds", "path"):
                    return (f"env check: unknown --as {as_kind!r} "
                            "(string|bool|int|float|seconds|path)")
                lines = []
                failed = False
                for name in names:
                    try:
                        if as_kind == "bool":
                            value = easyenv.get_env_bool_required(name)
                        elif as_kind == "int":
                            value = easyenv.get_env_num_required(name, int)
                        elif as_kind == "float":
                            value = easyenv.get_env_num_required(
                                name, float)
                        elif as_kind == "seconds":
                            value = (easyenv
                                     .get_env_duration_seconds_required(
                                         name))
                        elif as_kind == "path":
                            value = easyenv.get_env_pathbuf_required(name)
                        else:
                            value = easyenv.get_env_string_required(name)
                    except easyenv.EnvError as exc:
                        lines.append(f"{name}: FAIL — {exc}")
                        failed = True
                        continue
                    shown = (str(value)
                             if not easyenv.is_secret_name(name)
                             else easyenv.mask_secret(str(value)))
                    lines.append(f"{name}: ok ({shown})")
                lines.append("env check: "
                             + ("FAILED" if failed else "all ok"))
                return "\n".join(lines)
            if verb == "status":
                status = config_env_status()
                lines = [
                    f"mapped set: {len(status['set_mapped'])}, "
                    f"direct-read set: {len(status['set_direct'])}, "
                    f"mapped unset: {len(status['unset_mapped'])}, "
                    f"ignored NM_*: {len(status['ignored'])}",
                ]
                if status["ignored"]:
                    lines.append("ignored (set but nothing reads them):")
                    lines.extend(f"  {key}"
                                 for key in status["ignored"][:20])
                    if len(status["ignored"]) > 20:
                        lines.append(
                            f"  …and {len(status['ignored']) - 20} more")
                return "\n".join(lines)
            if verb == "dotenv":
                path = rest[0] if rest else ""
                candidates = ([Path(path)] if path
                              else easyenv.env_file_candidates())
                target = next((c for c in candidates if c.is_file()),
                              None)
                if target is None:
                    return "no .env file found"
                parsed = easyenv.parse_env_file(target)
                would_set = [k for k in parsed if k not in os.environ]
                already = [k for k in parsed if k in os.environ]
                lines = [f"{target}: {len(parsed)} keys, "
                         f"{len(would_set)} would load, "
                         f"{len(already)} already set"]
                for key in sorted(would_set):
                    lines.append(
                        f"  + {key}="
                        f"{easyenv.display_value(key, parsed[key])}")
                for key in sorted(already):
                    lines.append(f"  = {key} (already set)")
                return "\n".join(lines)
            return self._control_env_help()
        except Exception as exc:  # noqa: BLE001 - chat replies, never traces
            record_failure("runtime_env._control_env", exc)
            return f"env {verb} failed: {exc}"

    def _control_env_help(self) -> str:
        return (
            "env — environment variables, read-only:\n"
            "  /env get <NAME> — one variable (masked if secret)\n"
            "  /env list [PREFIX] — variables by prefix (default NM_)\n"
            "  /env check <NAME...> [--as bool|int|float|seconds|path]\n"
            "  /env status — mapped / direct / ignored NM_* counts\n"
            "  /env dotenv [PATH] — what a .env file would load"
        )
