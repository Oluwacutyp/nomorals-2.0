"""``nm update`` — transactional self-update with automatic rollback."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from ...os.update import UpdateManager
from ..emit import _emit
from .snapshot import _manager


def _repo_dir(args: argparse.Namespace) -> Path:
    override = getattr(args, "repo", "") or os.environ.get("NM_REPO_DIR", "")
    if override:
        return Path(override).expanduser().resolve()
    # nomorals/cmdline/commands/update.py -> repo root is 3 levels up.
    return Path(__file__).resolve().parents[3]


def _cmd_update(args: argparse.Namespace, settings: Any) -> int:
    """Runs on settings only (no open DB) so the pre-update snapshot and
    restore see the live system honestly."""
    yes = bool(getattr(args, "yes", False))
    check_only = bool(getattr(args, "check_only", False))
    repo = _repo_dir(args)

    if not yes and not check_only:
        print(f"this will: snapshot state, git pull in {repo}, migrate, "
              "health-check, and roll back automatically on any failure.")
        try:
            answer = input("continue? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):  # noqa: E106 - deliberate: aborted prompt = "no"
            print()
            return 2
        if answer not in ("y", "yes"):
            print("aborted")
            return 2

    mgr = _manager(settings, getattr(args, "config", None))
    updater = UpdateManager(repo, mgr)

    if check_only:
        failures = []
        for check in updater.health_checks:
            name = getattr(check, "__name__", "health_check")
            try:
                ok, detail = check()
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"raised {type(exc).__name__}: {exc}"
            print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
            if not ok:
                failures.append(name)
        return 1 if failures else 0

    report = updater.run(pull=not bool(getattr(args, "no_pull", False)))
    _emit(args, report.to_dict(),
          _render_report(report))
    return 0 if report.ok else 1


def _render_report(report: Any) -> str:
    lines = []
    for step in report.steps:
        mark = "ok  " if step["ok"] else "FAIL"
        lines.append(f"{mark} {step['step']}: {step['detail']}")
    if report.ok:
        lines.append("update complete")
    elif report.rolled_back:
        lines.append(f"update FAILED and was rolled back "
                     f"(pre-update snapshot {report.pre_update_snapshot})")
        lines.append(f"error: {report.error}")
    else:
        lines.append(f"update FAILED: {report.error}")
    return "\n".join(lines)
