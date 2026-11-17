"""``python -m nomorals.backup`` — offline-safe coordinator CLI.

No network, no vault, no connector needed for plan / snapshot / drill
(the drill takes the key via ``NM_BACKUP_KEY_B64``). ``ship`` runs one
full generation now through a wired :class:`BackupCoordinator` (vault +
GitHubConnector, opened from the Devon database under the current
directory) — the manual equivalent of the scheduled daily job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _build_cli_coordinator(root: Path, work_dir: str) -> Any:
    """Build a wired BackupCoordinator for the manual CLI path.

    The vault opens from the Devon database under ``root`` (the legacy
    ``NM_VAULT_PASSPHRASE`` env fallback applies when no other
    credential is at hand — see :func:`nomorals.accounts.vault.open_vault`).
    The GitHub connector attaches only when a GitHub credential is
    stored; without one the run queues in the outbox instead of failing.
    Raises a clear error — never a silent skip — when the database, the
    vault, or the key is missing.
    """
    from ..accounts.vault import open_vault
    from ..connectors.github import GitHubConnector
    from ..storage.db import Database
    from .coordinator import BackupCoordinator, open_s3_store
    from .errors import BackupError

    db_path = root / "data" / "nomorals.db"
    if not db_path.is_file():
        raise BackupError(
            f"no Devon database at {db_path} — run from the Devon home")
    vault = open_vault(Database(str(db_path)), context=None,
                       actor="backup-cli")
    connector = None
    github = GitHubConnector(vault)
    if github._load_credential() is not None:  # noqa: SLF001 - token via vault
        connector = github
    return BackupCoordinator(
        root=root, work_dir=work_dir, vault=vault,
        connector=connector, notifier=None,
        s3_store=open_s3_store(),
    )


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
    p_ship = sub.add_parser(
        "ship",
        help="run one full generation now (snapshot -> encrypt -> "
        "ship to GitHub Releases or queue in the outbox when offline)")
    p_ship.add_argument("--work-dir", default="backups")
    p_ship.add_argument("--label", default="manual")
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
    if args.cmd == "ship":
        # The manual equivalent of the scheduled daily job: one full
        # generation through the same coordinator path (snapshot ->
        # encrypt -> ship-or-queue -> prune). A separate `run` verb
        # would be a duplicate name for this operation.
        try:
            coordinator = _build_cli_coordinator(root, args.work_dir)
            result = coordinator.run_backup(label=args.label)
        except Exception as exc:  # noqa: BLE001 - manual CLI: clean error, exit 1
            print(f"backup ship failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 1
        print(json.dumps({
            "backup_id": result.get("backup_id"),
            "shipped": bool(result.get("shipped")),
            "queued": bool(result.get("queued")),
            "manifest_pushed": bool(result.get("manifest_pushed")),
            "total_encrypted_bytes": result.get("total_encrypted_bytes", 0),
            "reason": result.get("reason", ""),
        }, indent=2, default=str))
        return 0
    # Unreachable: argparse enforces a subcommand. Kept for clarity.
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
