"""``python -m nomorals.backup`` — offline-safe coordinator CLI.

No network, no vault, no connector needed for plan / snapshot / drill
(the drill takes the key via ``NM_BACKUP_KEY_B64``). Shipping needs a
wired :class:`BackupCoordinator` (vault + GitHubConnector) — use the
library API for that path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from .plan import build_plan
    from .protect import key_from_base64
    from .snapshot import take_snapshot
    from .verify import drill

    parser = argparse.ArgumentParser(prog="nomorals.backup")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="show what a snapshot would contain")
    p_snap = sub.add_parser("snapshot", help="take a local snapshot")
    p_snap.add_argument("--work-dir", default="backups")
    p_drill = sub.add_parser("drill", help="restore drill for a generation")
    p_drill.add_argument("manifest")
    p_drill.add_argument("blobs")
    p_prune = sub.add_parser("prune", help="show retention keepers (dry run)")
    p_prune.add_argument("--work-dir", default="backups")
    args = parser.parse_args(argv)

    root = Path.cwd()
    if args.cmd == "plan":
        plan = build_plan(root)
        print(json.dumps(plan.to_dict(), indent=2, default=str))
        return 0
    if args.cmd == "snapshot":
        snap = take_snapshot(root, args.work_dir)
        print(json.dumps(snap.to_dict(), indent=2, default=str))
        return 0
    if args.cmd == "drill":
        key_b64 = os.environ.get("NM_BACKUP_KEY_B64", "")
        if not key_b64:
            print("NM_BACKUP_KEY_B64 is required (base64 backup key)",
                  file=sys.stderr)
            return 2
        report = drill(args.manifest, args.blobs, key_from_base64(key_b64))
        print(report.summary())
        for step in report.steps:
            print(f"  {'PASS' if step.ok else 'FAIL'} {step.name} {step.detail}")
        return 0 if report.ok else 1
    if args.cmd == "prune":
        from .policy import select_keepers

        work = Path(args.work_dir)
        entries = []
        if work.is_dir():
            for child in sorted(work.iterdir()):
                if child.is_dir() and child.name.startswith("backup-"):
                    entries.append({"backup_id": child.name,
                                    "created_at": child.stat().st_mtime})
        keepers = select_keepers(entries)
        print(json.dumps({"keep": sorted(keepers),
                          "prune": sorted(
                              e["backup_id"] for e in entries
                              if e["backup_id"] not in keepers)}, indent=2))
        return 0
    # Unreachable: argparse enforces a subcommand. Kept for clarity.
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
