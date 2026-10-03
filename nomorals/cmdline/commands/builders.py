"""``nm build`` — scaffold -> verify (-> deliver) for builder templates.

The ten-stack :class:`~nomorals.builders.AppBuilder` generator is served
by ``nm apps``; this command is the CLI surface for the scaffold-template
pipeline in :mod:`nomorals.builders` (kinds: webapp, bot, cli_tool,
rest_api, telegram_bot, dashboard):

* ``nm build kinds`` — list template kinds
* ``nm build verify <kind> <name> [--dest DIR]`` — scaffold + run the
  full verify lifecycle (install, tests, serve+smoke, export) and print
  the report; exit 1 when the build is BROKEN
* ``nm build deliver <kind> <name> --to platform:chat [--dest DIR]`` —
  the full pipeline: verify, zip, deliver the archive to a chat

Delivery needs a gateway: pass one via a context whose
``extras["gateway"]`` holds it (the Telegram bot wires this up).  When
no gateway is available the deliver step is recorded as failed in the
report and the local zip path is printed, so nothing is lost.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..emit import _emit


def _split_target(args: argparse.Namespace) -> tuple[str, str]:
    to = (getattr(args, "to", "") or "").strip()
    platform = (getattr(args, "platform", "") or "").strip()
    chat = to
    if to and ":" in to:
        platform, chat = to.split(":", 1)
        platform, chat = platform.strip(), chat.strip()
    return platform, chat


def _cmd_build(args: argparse.Namespace, context: Any) -> int:
    """``nm build`` — verify-first project pipeline."""
    from ...builders import KINDS, build_and_verify, build_zip_and_deliver
    from ...core.errors import ToolError

    action = getattr(args, "build_action", "") or "kinds"

    if action == "kinds":
        _emit(args, {"kinds": list(KINDS)},
              "template kinds:\n" + "\n".join(f"  {k}" for k in KINDS))
        return 0

    kind = (getattr(args, "kind", "") or "").strip()
    name = (getattr(args, "name", "") or "").strip()
    dest = (getattr(args, "dest", "") or ".").strip() or "."
    if action in ("verify", "deliver") and (not kind or not name):
        _emit(args, {"error": "kind and name required"},
              f"usage: nm build {action} <kind> <name> [--dest DIR] "
              "[--to platform:chat]")
        return 2

    if action == "verify":
        try:
            report = build_and_verify(
                kind, name, dest,
                startup_timeout=float(
                    getattr(args, "startup_timeout", 10.0) or 10.0),
                export_dir=(getattr(args, "export_dir", "") or None),
            )
        except ToolError as exc:
            _emit(args, {"error": str(exc)}, f"build verify: {exc}")
            return 1
        _emit(args, report.to_dict(), report.summary())
        return 0 if report.ok else 1

    if action == "deliver":
        platform, chat = _split_target(args)
        if not platform or not chat:
            _emit(args, {"error": "destination required"},
                  "usage: nm build deliver <kind> <name> "
                  "--to platform:chat (e.g. --to telegram:123456)")
            return 2
        caption = (getattr(args, "caption", "") or "").strip()
        report = build_zip_and_deliver(
            kind, name, dest, platform=platform, chat=chat,
            context=context, caption=caption,
            startup_timeout=float(
                getattr(args, "startup_timeout", 10.0) or 10.0),
            export_dir=(getattr(args, "export_dir", "") or None),
        )
        text = report.summary()
        if report.zip_path and not report.ok:
            text += f"\narchive kept at: {report.zip_path}"
        _emit(args, report.to_dict(), text)
        return 0 if report.ok else 1

    _emit(args, {"error": f"unknown build action {action!r}"},
          "usage: nm build kinds\n"
          "       nm build verify <kind> <name> [--dest DIR]\n"
          "       nm build deliver <kind> <name> --to platform:chat "
          "[--dest DIR]")
    return 2
