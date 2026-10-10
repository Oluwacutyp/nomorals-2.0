"""Transactional self-update: pull, migrate, health-check, or roll back.

:class:`UpdateManager` owns the whole update as one transaction:

1. take a pre-update snapshot (via :class:`SnapshotManager`),
2. ``git pull --ff-only`` in the repo (skippable),
3. run database migrations,
4. run post-update health checks (importable package, layering subset,
   error scan, DB integrity).

If *any* step fails, the pre-update snapshot is restored automatically and
the report says so.  The system is never left half-updated: a failed pull
leaves the tree untouched (``--ff-only`` is all-or-nothing), and a failed
health check triggers a full state restore.

Crash recovery: every run appends to an update journal
(``<home>/update-journal.jsonl``).  :meth:`recover_interrupted` replays the
journal at boot — a run with no terminal event whose last step reached a
dangerous point (migrations applied) is rolled back automatically; a run
that died before that is marked aborted.  Failed versions are quarantined
so a bad update is never retried blindly.

Health checks are plain callables ``() -> (ok: bool, detail: str)`` so tests
can inject a broken one to prove the rollback path.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .snapshots import SnapshotManager

__all__ = [
    "HealthCheckFn",
    "UpdateReport",
    "UpdateManager",
    "default_health_checks",
]

_log = get_logger(__name__)

#: A health check returns (ok, detail).
HealthCheckFn = Callable[[], "tuple[bool, str]"]

JOURNAL_FILENAME = "update-journal.jsonl"
HISTORY_FILENAME = "update-history.jsonl"
QUARANTINE_FILENAME = "update-quarantine.json"

#: Steps at/after which a crash leaves the system in a state that must be
#: rolled back (migrations may have partially applied).
_DANGEROUS_STEPS = frozenset({"migrate", "health"})


@dataclass
class UpdateReport:
    """What the update did, step by step."""

    ok: bool = False
    rolled_back: bool = False
    pre_update_snapshot: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0
    #: Version/commit quarantined by this run ("" when none).
    quarantined: str = ""

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append({"step": name, "ok": ok, "detail": detail})

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rolled_back": self.rolled_back,
            "pre_update_snapshot": self.pre_update_snapshot,
            "steps": self.steps,
            "error": self.error,
            "seconds": round(self.seconds, 3),
            "quarantined": self.quarantined,
        }

    def render(self) -> str:
        """Plain-text update report."""
        lines = [f"update — {'✓ OK' if self.ok else '✗ FAILED'}"
                 f" ({self.seconds:.1f}s)"]
        for step in self.steps:
            mark = "✓" if step.get("ok") else "✗"
            lines.append(f"  {mark} {step.get('step')}")
            if step.get("detail"):
                lines.append(f"      {step['detail']}")
        if self.rolled_back:
            lines.append("  ↩ rolled back to pre-update snapshot"
                         f" {self.pre_update_snapshot}")
        if self.quarantined:
            lines.append(f"  ⚠ version quarantined: {self.quarantined}")
        if self.error:
            lines.append(f"  error: {self.error}")
        return "\n".join(lines)


def default_health_checks(
    repo_dir: str | os.PathLike[str] | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> list[HealthCheckFn]:
    """The standard post-update health checks.

    * the ``nomorals`` package imports and reports a version,
    * the layering subset passes (no upward imports in the changed tree),
    * the error scanner reports no findings on first-party code,
    * the live database passes ``PRAGMA integrity_check``.
    """
    checks: list[HealthCheckFn] = []

    def _import() -> tuple[bool, str]:
        try:
            import nomorals  # noqa: F401

            from ..version import __version__

            return True, f"nomorals {__version__} imports"
        except Exception as exc:  # noqa: BLE001 - the check IS the report
            return False, f"import failed: {exc}"

    checks.append(_import)

    if repo_dir is not None:

        def _layering() -> tuple[bool, str]:
            try:
                import sys

                tests_dir = str(Path(repo_dir) / "tests")
                if tests_dir not in sys.path:
                    sys.path.insert(0, tests_dir)
                import test_layering

                case = test_layering.TestLayering("test_no_upward_imports")
                case.test_no_upward_imports()
                return True, "layering subset green"
            except Exception as exc:  # noqa: BLE001
                return False, f"layering failed: {exc}"

        def _error_scan() -> tuple[bool, str]:
            try:
                from ..tools.error_scan import scan

                report = scan([str(Path(repo_dir) / "nomorals")])
                errors = [f for f in report.findings if f.severity == "error"]
                if errors:
                    return False, f"{len(errors)} error_scan errors"
                return True, "error_scan clean"
            except Exception as exc:  # noqa: BLE001
                return False, f"error_scan failed: {exc}"

        checks.extend([_layering, _error_scan])

    if db_path is not None:

        def _db() -> tuple[bool, str]:
            try:
                from ..storage.db import Database

                db = Database(str(db_path))
                try:
                    result = db.integrity_check()
                finally:
                    db.close()
                if str(result).lower() == "ok":
                    return True, "db integrity ok"
                return False, f"db integrity_check: {result}"
            except Exception as exc:  # noqa: BLE001
                return False, f"db check failed: {exc}"

        checks.append(_db)

    return checks


class UpdateManager:
    """Run a transactional update of the Devon installation."""

    def __init__(
        self,
        repo_dir: str | os.PathLike[str],
        snapshots: SnapshotManager,
        *,
        health_checks: list[HealthCheckFn] | None = None,
        git_bin: str = "git",
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.snapshots = snapshots
        self.git_bin = git_bin
        self.health_checks = (
            list(health_checks)
            if health_checks is not None
            else default_health_checks(repo_dir, snapshots.db_path)
        )
        home = Path(snapshots.home)
        self.journal_path = home / JOURNAL_FILENAME
        self.history_path = home / HISTORY_FILENAME
        self.quarantine_path = home / QUARANTINE_FILENAME

    # ── public ─────────────────────────────────────────────────────────────

    def run(self, *, pull: bool = True) -> UpdateReport:
        """Execute the update transaction. Never raises on update failure."""
        started = time.time()
        report = UpdateReport()
        run_id = uuid.uuid4().hex[:12]
        self._journal({"run_id": run_id, "event": "start", "ts": time.time()})

        # Step 0: snapshot of the current tree sha (for post-pull checks).
        previous_sha = self._git_sha() if pull else ""

        # Step 1: pre-update snapshot — the rollback anchor.
        try:
            snap = self.snapshots.create(label="pre-update")
            report.pre_update_snapshot = snap.id
            report.record("snapshot", True, f"pre-update snapshot {snap.id}")
            self._journal({"run_id": run_id, "event": "step", "step": "snapshot",
                           "ts": time.time(), "snapshot": snap.id})
        except Exception as exc:  # noqa: BLE001 - snapshot failure aborts
            report.record("snapshot", False, f"{type(exc).__name__}: {exc}")
            report.error = f"could not take pre-update snapshot: {exc}"
            report.seconds = time.time() - started
            self._journal({"run_id": run_id, "event": "aborted",
                           "ts": time.time(), "error": report.error})
            self._append_history(report)
            return report

        # Step 2: pull. --ff-only is all-or-nothing: a failed pull changes
        # nothing, so no rollback is needed for it.
        new_sha = ""
        if pull:
            ok, detail = self._git_pull()
            report.record("git_pull", ok, detail)
            self._journal({"run_id": run_id, "event": "step", "step": "git_pull",
                           "ts": time.time(), "ok": ok, "detail": detail})
            if not ok:
                report.error = f"git pull failed: {detail}"
                report.seconds = time.time() - started
                self._journal({"run_id": run_id, "event": "aborted",
                               "ts": time.time(), "error": report.error})
                self._append_history(report)
                return report
            new_sha = self._git_sha()
            if new_sha and self._is_quarantined(new_sha):
                report.error = (
                    f"pulled version {new_sha[:12]} is quarantined"
                    " (it failed a previous update) — refusing to apply it."
                    + (f" Undo the pull with: git -C {self.repo_dir} reset"
                       f" --hard {previous_sha[:12]}" if previous_sha else "")
                    + " Or clear the quarantine explicitly to retry.")
                report.seconds = time.time() - started
                self._journal({"run_id": run_id, "event": "aborted",
                               "ts": time.time(), "error": report.error})
                self._append_history(report)
                return report
        else:
            report.record("git_pull", True, "skipped (--no-pull)")
            self._journal({"run_id": run_id, "event": "step", "step": "git_pull",
                           "ts": time.time(), "ok": True,
                           "detail": "skipped (--no-pull)"})

        # Step 3: migrations.
        try:
            applied, version = self._migrate()
            report.record(
                "migrate", True,
                f"schema v{version}" + (f" (+{len(applied)} applied)" if applied else ""),
            )
            self._journal({"run_id": run_id, "event": "step", "step": "migrate",
                           "ts": time.time(), "ok": True,
                           "schema_version": version,
                           "applied": list(applied)})
        except Exception as exc:  # noqa: BLE001 - migrate failure rolls back
            report.record("migrate", False, f"{type(exc).__name__}: {exc}")
            self._journal({"run_id": run_id, "event": "step", "step": "migrate",
                           "ts": time.time(), "ok": False,
                           "error": f"{type(exc).__name__}: {exc}"})
            report = self._rollback(report, started, f"migration failed: {exc}",
                                    run_id=run_id, quarantine_sha=new_sha)
            self._append_history(report)
            return report

        # Step 4: health checks. ANY failure rolls back.
        failed: list[str] = []
        for check in self.health_checks:
            name = getattr(check, "__name__", "health_check")
            try:
                ok, detail = check()
            except Exception as exc:  # noqa: BLE001 - a raising check is a failure
                ok, detail = False, f"raised {type(exc).__name__}: {exc}"
            report.record(f"health:{name}", ok, detail)
            self._journal({"run_id": run_id, "event": "step",
                           "step": f"health:{name}", "ts": time.time(),
                           "ok": ok, "detail": detail})
            if not ok:
                failed.append(f"{name}: {detail}")
        if failed:
            report = self._rollback(
                report, started, "health check failed: " + "; ".join(failed),
                run_id=run_id, quarantine_sha=new_sha)
            self._append_history(report)
            return report

        report.ok = True
        report.seconds = time.time() - started
        self._journal({"run_id": run_id, "event": "complete",
                       "ts": time.time(), "sha": new_sha})
        self._append_history(report)
        _log.info("update completed cleanly")
        return report

    def dry_run(self) -> UpdateReport:
        """Run the health checks without pulling, migrating, or snapshotting.

        Answers "would this update survive the health gate?" safely.
        """
        started = time.time()
        report = UpdateReport()
        report.record("snapshot", True, "skipped (dry run)")
        report.record("git_pull", True, "skipped (dry run)")
        report.record("migrate", True, "skipped (dry run)")
        failed: list[str] = []
        for check in self.health_checks:
            name = getattr(check, "__name__", "health_check")
            try:
                ok, detail = check()
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"raised {type(exc).__name__}: {exc}"
            report.record(f"health:{name}", ok, detail)
            if not ok:
                failed.append(f"{name}: {detail}")
        report.ok = not failed
        if failed:
            report.error = "dry run health gate failed: " + "; ".join(failed)
        report.seconds = time.time() - started
        return report

    # ── crash recovery ─────────────────────────────────────────────────────
    def recover_interrupted(self) -> dict[str, Any]:
        """Replay the journal at boot and recover interrupted updates.

        A run with no terminal event (complete / rolled_back / aborted):
        * last step at/after a dangerous point (migrate, health) → roll
          back to its pre-update snapshot automatically;
        * died earlier → mark aborted (nothing user-visible changed).

        Returns a receipt dict describing what was done.
        """
        runs = self._journal_runs()
        if not runs:
            return {"recovered": False, "reason": "no journal entries"}
        last = runs[-1]
        events = last["events"]
        terminal = next((e for e in events
                         if e.get("event") in ("complete", "rolled_back",
                                               "aborted", "crash_recovery")),
                        None)
        if terminal is not None:
            return {"recovered": False, "reason": "last run terminated cleanly",
                    "run_id": last["run_id"]}
        steps = [e.get("step", "").split(":", 1)[-1] for e in events
                 if e.get("event") == "step"]
        snap_id = next((e.get("snapshot") for e in events
                        if e.get("event") == "step"
                        and e.get("step") == "snapshot"), "")
        receipt: dict[str, Any] = {
            "recovered": True, "run_id": last["run_id"],
            "last_steps": steps, "snapshot": snap_id,
        }
        if snap_id and any(s in _DANGEROUS_STEPS for s in steps):
            try:
                self.snapshots.restore(snap_id, force=True)
                receipt["action"] = "rolled_back"
                receipt["restored_snapshot"] = snap_id
                _log.warning("crash recovery: rolled back interrupted update"
                             " %s to %s", last["run_id"], snap_id)
            except Exception as exc:  # noqa: BLE001 — recovery must report
                receipt["action"] = "rollback_failed"
                receipt["error"] = f"{type(exc).__name__}: {exc}"
                _log.error("crash recovery rollback failed: %s", exc,
                           exc_info=True)
        else:
            receipt["action"] = "aborted"
            _log.warning("crash recovery: marked interrupted update %s"
                         " aborted (died before dangerous steps)",
                         last["run_id"])
        self._journal({"run_id": last["run_id"], "event": "crash_recovery",
                       "ts": time.time(), **{k: v for k, v in receipt.items()
                                             if k != "recovered"}})
        return receipt

    # ── quarantine ─────────────────────────────────────────────────────────
    def _quarantine(self) -> dict[str, Any]:
        try:
            return json.loads(self.quarantine_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"quarantined": []}

    def _is_quarantined(self, sha: str) -> bool:
        return any(q.get("sha") == sha
                   for q in self._quarantine().get("quarantined", []))

    def quarantine_list(self) -> list[dict[str, Any]]:
        """Versions that failed an update and will not be retried."""
        return list(self._quarantine().get("quarantined", []))

    def quarantine_clear(self, sha: str = "") -> int:
        """Clear the quarantine (one sha, or all when empty). Returns count
        removed."""
        data = self._quarantine()
        entries = data.get("quarantined", [])
        if sha:
            kept = [q for q in entries if q.get("sha") != sha]
        else:
            kept = []
        removed = len(entries) - len(kept)
        data["quarantined"] = kept
        self._write_quarantine(data)
        return removed

    def _write_quarantine(self, data: dict[str, Any]) -> None:
        self.quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        self.quarantine_path.write_text(json.dumps(data, indent=2),
                                        encoding="utf-8")

    def _quarantine_add(self, sha: str, reason: str) -> None:
        if not sha or self._is_quarantined(sha):
            return
        data = self._quarantine()
        data.setdefault("quarantined", []).append(
            {"sha": sha, "reason": reason, "ts": time.time()})
        self._write_quarantine(data)

    # ── history ────────────────────────────────────────────────────────────
    def _append_history(self, report: UpdateReport) -> None:
        entry = {"ts": time.time(), **report.to_dict()}
        try:
            with open(self.history_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError:  # noqa: BLE001 — history is best effort
            _log.debug("could not append update history", exc_info=True)

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Past update reports, newest first."""
        entries: list[dict[str, Any]] = []
        try:
            with open(self.history_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            return []
        return list(reversed(entries))[:max(0, int(limit))]

    # ── journal internals ──────────────────────────────────────────────────
    def _journal(self, entry: dict[str, Any]) -> None:
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.journal_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError:  # noqa: BLE001 — the journal must never break updates
            _log.debug("could not append update journal", exc_info=True)

    def _journal_runs(self) -> list[dict[str, Any]]:
        runs: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        try:
            with open(self.journal_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    run_id = str(entry.get("run_id", ""))
                    if run_id not in runs:
                        runs[run_id] = []
                        order.append(run_id)
                    runs[run_id].append(entry)
        except OSError:
            return []
        return [{"run_id": rid, "events": runs[rid]} for rid in order]

    # ── steps ──────────────────────────────────────────────────────────────

    def _git_pull(self) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                [self.git_bin, "-C", str(self.repo_dir), "pull", "--ff-only"],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except FileNotFoundError:
            return False, f"{self.git_bin} not found"
        except subprocess.TimeoutExpired:
            return False, "git pull timed out after 300s"
        out = (proc.stdout or "") + (proc.stderr or "")
        tail = "\n".join(out.strip().splitlines()[-3:])
        if proc.returncode != 0:
            return False, tail or f"exit {proc.returncode}"
        return True, tail or "already up to date"

    def _migrate(self) -> tuple[list[str], int]:
        from ..storage.db import Database

        db = Database(str(self.snapshots.db_path))
        try:
            summary = db.migrate()
            return list(summary.applied), int(summary.version)
        finally:
            db.close()

    def _rollback(
        self, report: UpdateReport, started: float, error: str, *,
        run_id: str = "", quarantine_sha: str = "",
    ) -> UpdateReport:
        """Restore the pre-update snapshot. The update is NOT ok."""
        report.error = error
        try:
            self.snapshots.restore(report.pre_update_snapshot, force=True)
            report.rolled_back = True
            report.record("rollback", True,
                          f"restored {report.pre_update_snapshot}")
            _log.warning("update rolled back: %s", error)
        except Exception as exc:  # noqa: BLE001 - rollback failure is critical
            report.rolled_back = False
            report.record("rollback", False, f"{type(exc).__name__}: {exc}")
            _log.error("ROLLBACK FAILED: %s", exc, exc_info=True)
        if quarantine_sha:
            self._quarantine_add(quarantine_sha, error)
            report.quarantined = quarantine_sha
        report.seconds = time.time() - started
        if run_id:
            self._journal({"run_id": run_id, "event": "rolled_back",
                           "ts": time.time(), "error": error,
                           "restored": report.pre_update_snapshot,
                           "quarantined": quarantine_sha})
        return report

    def _git_sha(self) -> str:
        """Current HEAD sha of the repo ("" when unknowable)."""
        try:
            proc = subprocess.run(
                [self.git_bin, "-C", str(self.repo_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""
        return proc.stdout.strip() if proc.returncode == 0 else ""
