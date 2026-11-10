"""``nm repair`` — the doc-20 repair pipeline (rungs 7–10).

The console front door for :mod:`nomorals.agents.repair_pipeline`:
a candidate patch for a journaled incident is exercised in sandbox
worktrees (mandatory suite, then a single-profile canary) and only
then promoted — by policy for non-consequential changes, or parked
for the owner here.  On the console the operator IS the owner (the
``nm immune`` precedent), so ``approve`` is the owner gate arriving
through a shipped surface; no new prompt system exists.
"""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path
from typing import Any

from ..emit import _emit

_USAGE = (
    "usage:\n"
    "  nm repair run --incident <id> --patch <file.diff> "
    "[--probe \"<cmd>\"] [--producer <name>]\n"
    "  nm repair approve <run-id> | reject <run-id> [reason] | "
    "rollback <run-id> [reason]\n"
    "  nm repair runs [--incident <id>]"
)

_GOOD = {"PROMOTED", "AWAITING_APPROVAL", "ROLLED_BACK", "REJECTED"}


def _repo_dir(args: argparse.Namespace) -> Path:
    override = getattr(args, "repo", "") or os.environ.get(
        "NM_REPO_DIR", "")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _pipeline(settings: Any, args: argparse.Namespace):
    from ...agents.repair_pipeline import RepairPipeline
    from ...core.incidents import IncidentJournal
    from .snapshot import _manager

    home = Path(os.path.expanduser(
        getattr(settings, "home", "~/.nomorals"))).resolve()
    data_dir = home / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    journal = IncidentJournal(path=str(data_dir / "error_journal.db"))
    return RepairPipeline(
        journal=journal,
        repo_dir=_repo_dir(args),
        work_root=data_dir / "repair-runs",
        snapshots=_manager(settings, getattr(args, "config", None)),
        db_path=getattr(settings, "db_path", None),
    )


def _trail(state: Any) -> list:
    lines = []
    for rec in state.records:
        note = rec.get("verify_note") or ""
        lines.append("  " + str(rec.get("strategy", "?")) + " — " + note)
    return lines


def _report(args: argparse.Namespace, state: Any) -> int:
    payload = {
        "run": state.run_id, "incident": state.incident_id,
        "status": state.status,
        "records": [r.get("strategy") for r in state.records],
    }
    text = "\n".join(
        ["repair run " + state.run_id + " (incident "
         + str(state.incident_id) + "): " + state.status]
        + _trail(state))
    _emit(args, payload, text)
    return 0 if state.status in _GOOD else 1


def _cmd_repair(args: argparse.Namespace, settings: Any) -> int:
    from ...agents.repair_pipeline import PatchCandidate, RepairRefused

    task = list(getattr(args, "task", []) or [])
    if not task:
        print(_USAGE)
        return 2
    verb, rest = task[0], task[1:]
    try:
        pipe = _pipeline(settings, args)
        if verb == "runs":
            states = pipe.list_runs(
                incident_id=getattr(args, "incident", None))
            if not states:
                print("no repair runs recorded")
                return 0
            for state in states:
                print("repair run " + state.run_id + " (incident "
                      + str(state.incident_id) + "): " + state.status)
            return 0
        if verb == "run":
            incident = getattr(args, "incident", None)
            patch_file = getattr(args, "patch", "") or ""
            if incident is None or not patch_file:
                print(_USAGE)
                return 2
            diff_text = Path(patch_file).read_text(encoding="utf-8")
            probes = tuple(
                tuple(shlex.split(p)) for p in
                (getattr(args, "probe", None) or []))
            candidate = PatchCandidate(
                diff_text=diff_text,
                producer=getattr(args, "producer", "") or "console",
                rationale="console repair via nm repair",
                incident_id=int(incident), probes=probes)
            return _report(args, pipe.run(candidate))
        if verb in ("approve", "reject", "rollback"):
            if not rest:
                print(_USAGE)
                return 2
            if verb == "approve":
                return _report(args, pipe.approve(rest[0], by="owner"))
            if verb == "reject":
                return _report(args, pipe.reject(
                    rest[0], by="owner", reason=" ".join(rest[1:])))
            return _report(args, pipe.rollback(
                rest[0], by="owner", reason=" ".join(rest[1:])))
        print(_USAGE)
        return 2
    except RepairRefused as exc:
        print("repair refused: " + str(exc))
        return 2
