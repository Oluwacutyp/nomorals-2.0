"""Checkpoints + rewind for the coding agent — recovery as a feature.

A checkpoint captures two things:

* **code state** — the working tree, via ``git stash create`` (a stash
  *commit* with no ref update: the user's own ``git stash`` list is never
  touched) plus the raw contents of untracked files;
* **conversation state** — the task / plan_id / scope / iteration count the
  :class:`~nomorals.agents.coding.CodingAgent` was working on.

``rewind(n)`` restores the *working-tree files* from a checkpoint (HEAD is
never moved, branches are never switched) and always saves a ``pre-rewind``
safety checkpoint first, so a rewind is itself reversible — nothing is ever
lost.

Storage: ``~/.nomorals/checkpoints/`` (honoring ``settings.home_path`` the
same way :func:`nomorals.finance.budgets.finance_paths` does) —
``<id>.json`` metadata, ``<id>/untracked/...`` raw file contents, and an
``index.json`` listing ids oldest-first.  Only the newest
:data:`CHECKPOINT_KEEP` checkpoints are kept.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: How many checkpoints survive pruning (oldest deleted first).
CHECKPOINT_KEEP = 10

#: Untracked files bigger than this are skipped at capture time (noted in
#: the checkpoint metadata) — checkpoints stay small and fast.
MAX_UNTRACKED_BYTES = 5 * 1024 * 1024

#: Schema version for generic agent-state snapshots (save_state/load_state).
#: Bump when the payload shape changes; load_state refuses older versions
#: loudly instead of mis-resuming.
AGENT_STATE_VERSION = 1

#: How many agent-state snapshots survive pruning per run id.
AGENT_STATE_KEEP = 5


def _json_safe(value: Any) -> Any:
    """Coerce ``value`` into JSON-serializable form (best-effort)."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _new_id(now: float | None = None) -> str:
    stamp = datetime.fromtimestamp(
        now if now is not None else time.time(), tz=timezone.utc
    ).strftime("%Y%m%d_%H%M%S")
    rand = os.urandom(2).hex()
    return f"ckpt_{stamp}_{rand}"


@dataclass
class Checkpoint:
    """One saved recovery point for a coding session."""

    id: str
    ts: float
    label: str
    head_hash: str
    stash_hash: str | None
    workdir: str
    # Tracked files that differed from HEAD when captured (the rewind scope).
    scope_files: list[str] = field(default_factory=list)
    # Untracked files whose contents were captured (relative paths).
    untracked_files: list[str] = field(default_factory=list)
    # Untracked files skipped at capture (too big / unreadable).
    skipped_files: list[str] = field(default_factory=list)
    # Conversation snapshot: task / plan_id / scope / iterations.
    convo: dict[str, Any] = field(default_factory=dict)
    # False when the workdir wasn't a git repo — convo-only checkpoint.
    code_captured: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(
            id=str(data.get("id", "")),
            ts=float(data.get("ts", 0) or 0),
            label=str(data.get("label", "") or ""),
            head_hash=str(data.get("head_hash", "") or ""),
            stash_hash=data.get("stash_hash"),
            workdir=str(data.get("workdir", "") or ""),
            scope_files=[str(s) for s in (data.get("scope_files") or [])],
            untracked_files=[str(s) for s in (data.get("untracked_files") or [])],
            skipped_files=[str(s) for s in (data.get("skipped_files") or [])],
            convo=dict(data.get("convo") or {}),
            code_captured=bool(data.get("code_captured", True)),
            note=str(data.get("note", "") or ""),
        )

    @property
    def short(self) -> str:
        """One-line human summary for /checkpoints listings."""
        when = datetime.fromtimestamp(self.ts).strftime("%m-%d %H:%M")
        label = f" {self.label}" if self.label else ""
        code = "code+convo" if self.code_captured else "convo-only"
        return f"{self.id} · {when}{label} · {code}"

    def verify(self, store: "CheckpointStore") -> dict[str, Any]:
        """Integrity check: is this checkpoint actually restorable?

        Checks the meta file, the git objects referenced (HEAD commit and
        stash still present in the workdir repo), and every captured
        untracked file still on disk. A saved checkpoint you can't restore
        is a lie — this is the difference between "saved" and
        "trustworthy" (durable-agent checklist).
        """
        problems: list[str] = []
        meta = store._meta_path(self.id)
        if not meta.exists():
            problems.append("meta file missing")
        if self.code_captured and self.workdir:
            workdir = Path(self.workdir)
            if not store._is_repo(workdir):
                problems.append("workdir is no longer a git repo")
            else:
                for ref, name in ((self.head_hash, "HEAD"),
                                  (self.stash_hash, "stash")):
                    if not ref:
                        continue
                    try:
                        proc = store._git(
                            ["cat-file", "-e", f"{ref}^{{commit}}"], workdir)
                        if proc.returncode != 0:
                            problems.append(f"{name} commit {ref[:12]} gone")
                    except Exception:  # noqa: BLE001
                        problems.append(f"could not check {name} commit")
        files_dir = store._files_dir(self.id)
        missing = [rel for rel in self.untracked_files
                   if not (files_dir / rel).exists()]
        if missing:
            problems.append(f"{len(missing)} untracked file(s) missing: "
                            + ", ".join(missing[:5]))
        return {"ok": not problems, "problems": problems}

    def describe(self) -> str:
        """Human-readable checkpoint card (god-tier /checkpoints listing)."""
        from .render import ICONS, banner, kv, truncate

        when = datetime.fromtimestamp(self.ts).strftime("%Y-%m-%d %H:%M")
        icon = ICONS["ok"] if self.code_captured else ICONS["warn"]
        head = {
            "id": self.id,
            "when": when,
            "label": self.label or "—",
            "head": self.head_hash[:12] if self.head_hash else "—",
            "scope": f"{len(self.scope_files)} tracked file(s) changed",
            "untracked": f"{len(self.untracked_files)} captured"
                         + (f" ({len(self.skipped_files)} skipped)"
                            if self.skipped_files else ""),
            "task": truncate(str(self.convo.get("task", "—")), 80),
        }
        lines = [banner(f"Checkpoint {self.id}", icon), kv(head.items())]
        if self.note:
            lines.append(f"note: {self.note}")
        return "\n".join(lines)

    def diff(self, other: "Checkpoint") -> dict[str, Any]:
        """What changed between two checkpoints?

        Returns added/removed/changed file lists across scope_files and
        untracked_files, plus whether the git HEAD moved. Never raises.
        """
        try:
            a_scope, b_scope = set(self.scope_files), set(other.scope_files)
            a_un, b_un = set(self.untracked_files), set(other.untracked_files)
            return {
                "from": self.id, "to": other.id,
                "head_moved": self.head_hash != other.head_hash,
                "scope_added": sorted(b_scope - a_scope),
                "scope_removed": sorted(a_scope - b_scope),
                "scope_common": sorted(a_scope & b_scope),
                "untracked_added": sorted(b_un - a_un),
                "untracked_removed": sorted(a_un - b_un),
            }
        except Exception:  # noqa: BLE001
            return {"from": self.id, "to": other.id, "error": "diff failed"}


class CheckpointStore:
    """Disk-backed checkpoint index + capture/restore machinery.

    All public methods are fail-closed: capture degrades to a convo-only
    checkpoint when git is unavailable, and rewind reports failures as
    strings instead of raising.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        settings: Any = None,
    ) -> None:
        if base_dir is not None:
            base = Path(base_dir).expanduser()
        else:
            home = getattr(settings, "home_path", None) if settings else None
            base = (Path(home) if home else Path.home()) / ".nomorals" / "checkpoints"
        self.base = base
        self._lock = threading.RLock()

    # ── paths ─────────────────────────────────────────────────────────
    def _meta_path(self, ckpt_id: str) -> Path:
        return self.base / f"{ckpt_id}.json"

    def _files_dir(self, ckpt_id: str) -> Path:
        return self.base / ckpt_id / "untracked"

    def _index_path(self) -> Path:
        return self.base / "index.json"

    def _state_dir(self) -> Path:
        return self.base / "agent_state"

    # ── git ───────────────────────────────────────────────────────────
    @staticmethod
    def _git(args: list[str], cwd: Path,
             timeout: int = 60) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=str(cwd),
            capture_output=True, text=True, timeout=timeout)

    @staticmethod
    def _is_repo(workdir: Path) -> bool:
        try:
            proc = CheckpointStore._git(
                ["rev-parse", "--is-inside-work-tree"], workdir)
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0 and proc.stdout.strip() == "true"

    # ── capture ───────────────────────────────────────────────────────
    def capture(self, workdir: Path | str, *, label: str = "",
                convo: dict[str, Any] | None = None) -> Checkpoint:
        """Snapshot ``workdir`` + conversation state. Never raises."""
        workdir = Path(workdir)
        now = time.time()
        ckpt = Checkpoint(
            id=_new_id(now), ts=now, label=label, head_hash="",
            stash_hash=None, workdir=str(workdir),
            convo=_json_safe(dict(convo or {})),
        )
        if not self._is_repo(workdir):
            ckpt.code_captured = False
            ckpt.note = f"not a git repo: {workdir}"
            _log.debug("checkpoint %s: %s", ckpt.id, ckpt.note)
            return ckpt
        try:
            head = self._git(["rev-parse", "HEAD"], workdir)
            if head.returncode != 0:
                raise RuntimeError(
                    f"rev-parse HEAD: {head.stderr.strip() or 'no commits?'}")
            ckpt.head_hash = head.stdout.strip()
            # Tracked files differing from HEAD — the rewind scope.
            diff = self._git(["diff", "--name-only", "HEAD"], workdir)
            if diff.returncode == 0:
                ckpt.scope_files = sorted(
                    line.strip().strip('"')
                    for line in diff.stdout.splitlines()
                    if line.strip())
            # Stash *commit* without touching refs — the user's own
            # `git stash` list is never modified (no push/pop/list).
            created = self._git(["stash", "create"], workdir)
            if created.returncode == 0 and created.stdout.strip():
                ckpt.stash_hash = created.stdout.strip()
            elif created.returncode != 0:
                _log.debug("checkpoint %s: stash create: %s", ckpt.id,
                           created.stderr.strip())
            ckpt.untracked_files, ckpt.skipped_files = \
                self._capture_untracked(workdir, ckpt.id)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            ckpt.code_captured = False
            ckpt.stash_hash = None
            ckpt.note = f"git capture failed ({exc}); convo-only checkpoint"
            _log.warning("checkpoint %s: %s", ckpt.id, ckpt.note)
        return ckpt

    def _capture_untracked(
            self, workdir: Path, ckpt_id: str) -> tuple[list[str], list[str]]:
        """Copy untracked file contents into the checkpoint dir."""
        try:
            status = self._git(["status", "--porcelain"], workdir)
        except (OSError, subprocess.SubprocessError) as exc:
            _log.debug("untracked capture: status failed: %s", exc)
            return [], []
        if status.returncode != 0:
            return [], []
        rels = sorted(
            line[3:].strip().strip('"')
            for line in status.stdout.splitlines()
            if line.startswith("??"))
        kept, skipped = [], []
        dest_root = self._files_dir(ckpt_id)
        for rel in rels:
            src = workdir / rel
            try:
                if src.is_symlink():
                    skipped.append(rel + " (symlink)")
                    continue
                if src.is_dir():
                    # Capture every file under an untracked directory.
                    for child in sorted(src.rglob("*")):
                        if child.is_symlink() or not child.is_file():
                            continue
                        child_rel = child.relative_to(workdir).as_posix()
                        if not self._store_one(child, child_rel, dest_root):
                            skipped.append(child_rel + " (too big/unreadable)")
                        else:
                            kept.append(child_rel)
                    continue
                if not src.is_file():
                    skipped.append(rel + " (not a file)")
                    continue
                if not self._store_one(src, rel, dest_root):
                    skipped.append(rel + " (too big/unreadable)")
                else:
                    kept.append(rel)
            except OSError as exc:
                _log.debug("untracked capture: %s: %s", rel, exc)
                skipped.append(rel + " (error)")
        return sorted(kept), sorted(skipped)

    @staticmethod
    def _store_one(src: Path, rel: str, dest_root: Path) -> bool:
        try:
            if src.stat().st_size > MAX_UNTRACKED_BYTES:
                return False
            data = src.read_bytes()
        except OSError:
            return False
        dest = dest_root / rel
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        except OSError:
            return False
        return True

    # ── persistence ───────────────────────────────────────────────────
    def save(self, ckpt: Checkpoint) -> Checkpoint:
        """Write metadata + update the index, then prune. Never raises."""
        with self._lock:
            try:
                self.base.mkdir(parents=True, exist_ok=True)
                self._meta_path(ckpt.id).write_text(
                    json.dumps(ckpt.to_dict(), indent=2), encoding="utf-8")
                ids = self._read_index()
                if ckpt.id not in ids:
                    ids.append(ckpt.id)
                self._write_index(ids)
                self.prune()
            except OSError as exc:
                _log.warning("checkpoint %s: save failed: %s", ckpt.id, exc)
        return ckpt

    def _read_index(self) -> list[str]:
        try:
            data = json.loads(self._index_path().read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(i) for i in data]
        except (OSError, ValueError):
            pass
        # Index missing/corrupt — rebuild from the metadata files on disk.
        try:
            metas = sorted(self.base.glob("ckpt_*.json"), key=lambda p: p.name)
            return [p.stem for p in metas]
        except OSError:
            return []

    def _write_index(self, ids: list[str]) -> None:
        self._index_path().write_text(json.dumps(ids), encoding="utf-8")

    def load(self, ckpt_id: str) -> Checkpoint | None:
        try:
            data = json.loads(
                self._meta_path(ckpt_id).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _log.debug("checkpoint load %s: %s", ckpt_id, exc)
            return None
        if not isinstance(data, dict):
            return None
        return Checkpoint.from_dict(data)

    def list(self) -> list[Checkpoint]:
        """All checkpoints, newest first. Never raises."""
        with self._lock:
            ids = self._read_index()
        out = []
        for ckpt_id in reversed(ids):
            ckpt = self.load(ckpt_id)
            if ckpt is not None:
                out.append(ckpt)
        return out

    def get_nth(self, n: int = 1) -> Checkpoint | None:
        """n=1 → the latest checkpoint. None when the store is empty."""
        if n < 1:
            return None
        items = self.list()
        return items[n - 1] if len(items) >= n else None

    def prune(self, keep: int = CHECKPOINT_KEEP) -> int:
        """Drop oldest checkpoints beyond ``keep``. Returns # removed."""
        removed = 0
        with self._lock:
            ids = self._read_index()
            while len(ids) > keep:
                old = ids.pop(0)
                for path in (self._meta_path(old), self.base / old):
                    try:
                        if path.is_dir():
                            import shutil
                            shutil.rmtree(path)
                        elif path.exists():
                            path.unlink()
                    except OSError as exc:
                        _log.debug("prune %s: %s", old, exc)
                removed += 1
            self._write_index(ids)
        if removed:
            _log.info("pruned %d old checkpoint(s), keeping %d",
                      removed, keep)
        return removed

    # ── generic agent-state snapshots ───────────────────────────────
    def save_state(self, run_id: str, payload: dict[str, Any], *,
                   label: str = "") -> dict[str, Any]:
        """Snapshot arbitrary agent-run state (task graph, swarm legs,
        orchestrator runs). ``payload`` must be JSON-serializable
        (best-effort coerced via :func:`_json_safe`).

        Snapshots are keyed by ``run_id`` and versioned
        (:data:`AGENT_STATE_VERSION`); only the newest
        :data:`AGENT_STATE_KEEP` per run id are kept. Best-effort:
        returns ``{"ok": False, "error": ...}`` instead of raising.
        """
        try:
            run_id = str(run_id or "").strip() or "adhoc"
            stamp = datetime.fromtimestamp(time.time(),
                                           tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            rand = os.urandom(2).hex()
            state_id = f"state_{run_id}_{stamp}_{rand}"
            record = {
                "version": AGENT_STATE_VERSION,
                "state_id": state_id,
                "run_id": run_id,
                "label": label,
                "saved_at": time.time(),
                "payload": _json_safe(payload),
            }
            dest = self._state_dir()
            dest.mkdir(parents=True, exist_ok=True)
            (dest / f"{state_id}.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8")
            self._prune_states(run_id)
            return {"ok": True, "state_id": state_id, "run_id": run_id}
        except Exception as exc:  # noqa: BLE001 - snapshots never break a run
            _log.warning("agent-state snapshot failed: %s", exc)
            return {"ok": False, "error": str(exc), "run_id": run_id}

    def _prune_states(self, run_id: str,
                      keep: int = AGENT_STATE_KEEP) -> int:
        removed = 0
        try:
            files = sorted(self._state_dir().glob(f"state_{run_id}_*.json"),
                           key=lambda p: p.name)
            while len(files) > keep:
                files.pop(0).unlink()
                removed += 1
        except OSError as exc:
            _log.debug("agent-state prune failed: %s", exc)
        return removed

    def load_state(self, state_id: str) -> dict[str, Any] | None:
        """Load a snapshot saved by :meth:`save_state`.

        Returns the full record (``version`` / ``run_id`` / ``payload``)
        or None when missing, corrupt, or from an older schema version —
        a stale schema resumes nothing.
        """
        try:
            raw = json.loads(
                (self._state_dir() / f"{state_id}.json").read_text(
                    encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _log.debug("agent-state load %s: %s", state_id, exc)
            return None
        if not isinstance(raw, dict):
            return None
        if raw.get("version") != AGENT_STATE_VERSION:
            _log.warning("agent-state %s: schema v%s != v%s — refusing",
                         state_id, raw.get("version"), AGENT_STATE_VERSION)
            return None
        return raw

    def list_states(self, run_id: str = "") -> list[dict[str, Any]]:
        """Snapshot metadata (no payloads), newest first. Best-effort.

        Ordered by ``saved_at`` — filenames carry a random suffix, so
        name order is unreliable when several snapshots land in the same
        second.
        """
        out: list[dict[str, Any]] = []
        try:
            pattern = (f"state_{run_id}_*.json" if run_id else "state_*.json")
            for path in self._state_dir().glob(pattern):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(raw, dict):
                    continue
                out.append({
                    "state_id": raw.get("state_id", path.stem),
                    "run_id": raw.get("run_id", ""),
                    "label": raw.get("label", ""),
                    "saved_at": raw.get("saved_at", 0),
                    "version": raw.get("version", 0),
                })
        except OSError as exc:
            _log.debug("agent-state list failed: %s", exc)
        out.sort(key=lambda s: (s["saved_at"], s["state_id"]), reverse=True)
        return out

    def delete_state(self, state_id: str) -> bool:
        """Remove one snapshot. Returns True when something was deleted."""
        try:
            path = self._state_dir() / f"{state_id}.json"
            if path.exists():
                path.unlink()
                return True
        except OSError as exc:
            _log.debug("agent-state delete %s: %s", state_id, exc)
        return False

    # ── rewind ────────────────────────────────────────────────────────
    def rewind_to(self, ckpt: Checkpoint) -> str:
        """Restore a checkpoint's working-tree files. Never raises.

        HEAD is never moved and branches are never switched — only files
        inside the checkpoint's scope are rewritten.
        """
        workdir = Path(ckpt.workdir)
        lines: list[str] = []
        if not ckpt.code_captured:
            note = ckpt.note or "no code snapshot"
            return (f"⏪ {ckpt.id}: convo-only checkpoint ({note}) — "
                    "no files to restore.")
        if not self._is_repo(workdir):
            return (f"⏪ {ckpt.id}: {workdir} is no longer a git repo — "
                    "nothing restored.")
        restored_tracked, restored_untracked = 0, []
        # Warn when HEAD moved on since the checkpoint — the restore still
        # runs (that's what rewind means) but the owner should know.
        try:
            cur = self._git(["rev-parse", "HEAD"], workdir)
            if (cur.returncode == 0 and ckpt.head_hash
                    and cur.stdout.strip() != ckpt.head_hash):
                lines.append(
                    "⚠️ HEAD moved since this checkpoint "
                    f"({ckpt.head_hash[:8]} → {cur.stdout.strip()[:8]}); "
                    "restoring file contents anyway.")
        except (OSError, subprocess.SubprocessError):
            pass
        if ckpt.stash_hash and ckpt.scope_files:
            try:
                proc = self._git(
                    ["checkout", ckpt.stash_hash, "--", *ckpt.scope_files],
                    workdir)
            except (OSError, subprocess.SubprocessError) as exc:
                proc = None
                _log.error("rewind checkout crashed: %s", exc)
            if proc is not None and proc.returncode == 0:
                restored_tracked = len(ckpt.scope_files)
            else:
                err = (proc.stderr.strip() if proc is not None else "crashed")
                lines.append(f"❌ tracked-file restore failed: {err}")
        elif ckpt.scope_files and not ckpt.stash_hash:
            lines.append("ℹ️ tree was clean at checkpoint — "
                         "no tracked files to restore.")
        # Untracked files: write the captured contents back (recreating
        # files deleted since the checkpoint).
        src_root = self._files_dir(ckpt.id)
        for rel in ckpt.untracked_files:
            src = src_root / rel
            dest = workdir / rel
            try:
                dest.resolve().relative_to(workdir.resolve())
            except ValueError:
                lines.append(f"⚠️ {rel} escapes the repo — skipped")
                continue
            try:
                data = src.read_bytes()
            except OSError:
                lines.append(f"⚠️ {rel}: captured copy missing — skipped")
                continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                restored_untracked.append(rel)
            except OSError as exc:
                lines.append(f"⚠️ {rel}: write failed ({exc}) — skipped")
        # Anything untracked NOW that wasn't captured then is out of scope —
        # left alone, but reported so the rewind isn't a mystery.
        try:
            status = self._git(["status", "--porcelain"], workdir)
            current_untracked = {
                line[3:].strip().strip('"')
                for line in status.stdout.splitlines()
                if line.startswith("??")} if status.returncode == 0 else set()
        except (OSError, subprocess.SubprocessError):
            current_untracked = set()
        new_untracked = sorted(
            current_untracked - set(ckpt.untracked_files))
        head = f"⏪ rewound to {ckpt.id}"
        if ckpt.label:
            head += f" ({ckpt.label})"
        lines.insert(0, head)
        lines.append(
            f"restored {restored_tracked} tracked file(s), "
            f"{len(restored_untracked)} untracked file(s)")
        if new_untracked:
            shown = ", ".join(new_untracked[:5])
            more = f" +{len(new_untracked) - 5} more" \
                if len(new_untracked) > 5 else ""
            lines.append(
                f"ℹ️ {len(new_untracked)} untracked file(s) created after "
                f"the checkpoint left in place: {shown}{more}")
        if ckpt.skipped_files:
            lines.append(
                f"ℹ️ {len(ckpt.skipped_files)} file(s) were skipped at "
                "capture (too big/unreadable) and could not be restored")
        return "\n".join(lines)
