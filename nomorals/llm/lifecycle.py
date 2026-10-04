"""Model lifecycle: registered → downloaded → verified → loaded → warm.

A model is not "the primary mind" because someone typed its name — it earns
the role by walking a real lifecycle, and every step is persisted so an
operator can see *where* a model is and roll back to the previous primary
when a promotion goes wrong.

Stages
------
``registered``
    Known to the system (HF repo id or a local GGUF path).  Nothing on disk
    is assumed.
``downloaded``
    The artifact bytes are present (downloaded, or the local file exists).
``verified``
    ``sha256`` of the artifact recorded; re-verifiable at any time.
``loaded``
    The provisioner has the model serving (e.g. ``llama-server`` spawned).
``warm``
    A health ping answered — the model is actually ready for traffic.

``unload()`` returns a model to ``verified`` (bytes stay on disk, the server
stops).  ``promote(id)`` pins a model as the primary mind and pushes the old
primary onto a persisted history stack; ``rollback()`` pops the stack.

The :class:`ModelProvisioner` protocol is duck-typed and injected, so tests
run the full transition matrix offline with a fake while production uses
:class:`LocalGGUFProvisioner` (``llama-server`` via
:class:`nomorals.llm.local_server.GGUFServerManager`).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..core.errors import NotFound, ValidationError
from ..core.logging_setup import get_logger
from ..storage.db import Database

__all__ = [
    "LifecycleError",
    "LocalGGUFProvisioner",
    "ManagedModel",
    "ModelLifecycle",
    "ModelProvisioner",
    "STAGES",
]

_log = get_logger(__name__)

STAGES = ("registered", "downloaded", "verified", "loaded", "warm")

#: Allowed transitions.  ``unload`` is loaded/warm → verified; a failed stage
#: lands in ``failed`` and can only be retried from ``registered`` so a retry
#: re-walks the whole pipeline instead of skipping the broken step.
_TRANSITIONS: dict[str, frozenset[str]] = {
    "registered": frozenset({"downloaded", "failed"}),
    "downloaded": frozenset({"verified", "failed"}),
    "verified": frozenset({"loaded", "failed"}),
    "loaded": frozenset({"warm", "verified", "failed"}),
    "warm": frozenset({"verified", "failed"}),
    "failed": frozenset({"registered"}),
}

#: A verified model may be promoted; anything earlier has not earned it.
_PROMOTABLE = frozenset({"verified", "loaded", "warm"})

_LIFECYCLE_DDL = """
CREATE TABLE IF NOT EXISTS model_lifecycle (
    id           TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    path         TEXT NOT NULL DEFAULT '',
    sha256       TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'registered',
    provider     TEXT NOT NULL DEFAULT '',
    capabilities TEXT NOT NULL DEFAULT '[]',
    context_len  INTEGER NOT NULL DEFAULT 0,
    quant        TEXT NOT NULL DEFAULT '',
    size_bytes   INTEGER NOT NULL DEFAULT 0,
    notes        TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS lifecycle_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    from_status TEXT NOT NULL,
    to_status   TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_events_model
    ON lifecycle_events(model_id);
CREATE TABLE IF NOT EXISTS lifecycle_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);
"""


class LifecycleError(Exception):
    """An illegal transition or a failed lifecycle stage."""


@dataclass
class ManagedModel:
    """One model under lifecycle management."""

    id: str
    source: str
    path: str = ""
    sha256: str = ""
    status: str = "registered"
    provider: str = ""
    capabilities: list[str] = field(default_factory=list)
    context_len: int = 0
    quant: str = ""
    size_bytes: int = 0
    notes: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def local(self) -> bool:
        return bool(self.path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "source": self.source, "path": self.path,
            "sha256": self.sha256[:16] + "…" if self.sha256 else "",
            "status": self.status, "provider": self.provider,
            "capabilities": list(self.capabilities),
            "context_len": self.context_len, "quant": self.quant,
            "size_bytes": self.size_bytes, "notes": self.notes,
        }


class ModelProvisioner(Protocol):
    """Duck-typed: anything with load/warm/unload can provision a model."""

    def load(self, model: ManagedModel) -> Any:
        """Make the model servable; return an opaque handle."""
        ...

    def warm(self, model: ManagedModel, handle: Any) -> bool:
        """Health ping.  True → the model answers traffic."""
        ...

    def unload(self, model: ManagedModel, handle: Any) -> None:
        """Stop serving.  Artifact bytes stay on disk."""
        ...


class LocalGGUFProvisioner:
    """Production provisioner: serve a local GGUF via ``llama-server``.

    One :class:`GGUFServerManager` per model id, keyed so two models never
    fight over one port — the second load gets the next free port.
    """

    def __init__(self, *, host: str = "127.0.0.1", base_port: int = 8080,
                 ctx_size: int = 4096) -> None:
        self.host = host
        self.base_port = base_port
        self.ctx_size = ctx_size
        self._managers: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _manager(self, model: ManagedModel) -> Any:
        from .local_server import GGUFServerManager  # lazy: heavy module
        with self._lock:
            manager = self._managers.get(model.id)
            if manager is None:
                port = self.base_port + (abs(hash(model.id)) % 16)
                manager = GGUFServerManager(
                    host=self.host, port=port, ctx_size=self.ctx_size or model.context_len or 4096,
                )
                self._managers[model.id] = manager
            return manager

    def load(self, model: ManagedModel) -> Any:
        if not model.path:
            raise LifecycleError(f"model {model.id!r} has no local file to load")
        manager = self._manager(model)
        diagnosis = manager.start(model.path)
        if not diagnosis.ok:
            raise LifecycleError(
                f"llama-server failed to load {model.id!r}: "
                + "; ".join(diagnosis.problems))
        return manager

    def warm(self, model: ManagedModel, handle: Any) -> bool:
        manager = handle if handle is not None else self._managers.get(model.id)
        if manager is None:
            return False
        return manager.health_status() in ("ok", "busy")

    def unload(self, model: ManagedModel, handle: Any) -> None:
        manager = handle if handle is not None else self._managers.pop(model.id, None)
        if manager is not None:
            manager.stop()
        with self._lock:
            self._managers.pop(model.id, None)


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class ModelLifecycle:
    """Persistent, transition-guarded lifecycle for servable models."""

    def __init__(
        self,
        db: Database | str | Path | None = None,
        *,
        provisioner: ModelProvisioner | None = None,
        models_dir: str | Path = "models",
        hf_token: str = "",
    ) -> None:
        if isinstance(db, Database):
            self.db = db
        else:
            self.db = Database(":memory:" if db is None else str(db))
        self.db.executescript(_LIFECYCLE_DDL)
        self.provisioner: ModelProvisioner = provisioner or LocalGGUFProvisioner()
        self.models_dir = Path(models_dir)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "") or os.environ.get("NM_HF_TOKEN", "")
        self._handles: dict[str, Any] = {}
        self._lock = threading.RLock()

    # ── registration ─────────────────────────────────────────────────────────
    def add(
        self,
        source: str,
        *,
        model_id: str = "",
        provider: str = "",
        capabilities: list[str] | tuple[str, ...] = ("chat",),
        context_len: int = 0,
        quant: str = "",
        notes: str = "",
    ) -> ManagedModel:
        """Register an HF repo id (``owner/name``) or a local ``.gguf`` path."""
        source = (source or "").strip()
        if not source:
            raise ValidationError("source is required (HF repo id or local .gguf path)")
        candidate = Path(source).expanduser()
        is_local_path = source.endswith(".gguf") or candidate.exists()
        path = str(candidate) if is_local_path else ""
        model_id = model_id or (candidate.stem if is_local_path else source.replace("/", "__"))
        if not provider:
            provider = "llama_cpp" if is_local_path else "hf_serverless"
        with self._lock:
            row = self.db.query(
                "SELECT id FROM model_lifecycle WHERE id = ?", (model_id,))
            now = time.time()
            if row:
                self.db.execute(
                    "UPDATE model_lifecycle SET source=?, path=?, provider=?, "
                    "capabilities=?, context_len=?, quant=?, notes=?, updated_at=? "
                    "WHERE id = ?",
                    (source, path, provider, json.dumps(list(capabilities)),
                     context_len, quant, notes, now, model_id),
                )
            else:
                self.db.execute(
                    "INSERT INTO model_lifecycle (id, source, path, sha256, status, "
                    "provider, capabilities, context_len, quant, size_bytes, notes, "
                    "created_at, updated_at) VALUES (?, ?, ?, '', 'registered', ?, ?, ?, ?, 0, ?, ?, ?)",
                    (model_id, source, path, provider, json.dumps(list(capabilities)),
                     context_len, quant, notes, now, now),
                )
            self._log_event(model_id, "", "registered", f"added source={source!r}")
        _log.info("lifecycle: registered %s (%s)", model_id, provider)
        return self.get(model_id)

    def add_gguf(
        self,
        path: str | Path,
        *,
        model_id: str = "",
        quant: str = "Q4_K_M",
        context_len: int = 0,
        notes: str = "",
    ) -> ManagedModel:
        """First-class local GGUF entry: the operator's own weights.

        This is how the phone's ``codebeast-3.8b Q4_K_M`` (~2.3GB) and the
        PC/VPS ``dolphin-8b-merged`` GGUF become selectable broker candidates —
        both map to the ``llama_cpp`` provider with ``{chat, code}``
        capabilities and are promotable with ``nm models use <id>``.
        """
        return self.add(
            str(path),
            model_id=model_id,
            provider="llama_cpp",
            capabilities=["chat", "code"],
            context_len=context_len,
            quant=quant,
            notes=notes or "operator GGUF",
        )

    # ── reads ────────────────────────────────────────────────────────────────
    def get(self, model_id: str) -> ManagedModel:
        rows = self.db.query("SELECT * FROM model_lifecycle WHERE id = ?", (model_id,))
        if not rows:
            raise NotFound(f"model {model_id!r} is not under lifecycle management")
        return self._row_to_model(rows[0])

    def list(self, *, status: str = "") -> list[ManagedModel]:
        if status:
            rows = self.db.query(
                "SELECT * FROM model_lifecycle WHERE status = ? ORDER BY updated_at DESC",
                (status,))
        else:
            rows = self.db.query(
                "SELECT * FROM model_lifecycle ORDER BY updated_at DESC")
        return [self._row_to_model(r) for r in rows]

    def history(self, model_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT from_status, to_status, detail, created_at FROM lifecycle_events "
            "WHERE model_id = ? ORDER BY id DESC LIMIT ?", (model_id, limit))

    @staticmethod
    def _row_to_model(row: dict[str, Any]) -> ManagedModel:
        try:
            caps = json.loads(row.get("capabilities") or "[]")
        except (json.JSONDecodeError, TypeError):
            caps = []
        return ManagedModel(
            id=row["id"], source=row["source"], path=row.get("path") or "",
            sha256=row.get("sha256") or "", status=row.get("status") or "registered",
            provider=row.get("provider") or "",
            capabilities=list(caps) if isinstance(caps, list) else [],
            context_len=int(row.get("context_len") or 0),
            quant=row.get("quant") or "", size_bytes=int(row.get("size_bytes") or 0),
            notes=row.get("notes") or "",
            created_at=float(row.get("created_at") or 0.0),
            updated_at=float(row.get("updated_at") or 0.0),
        )

    # ── transitions ──────────────────────────────────────────────────────────
    def _transition(self, model_id: str, to_status: str, detail: str = "") -> ManagedModel:
        with self._lock:
            model = self.get(model_id)
            allowed = _TRANSITIONS.get(model.status, frozenset())
            if to_status not in allowed:
                raise LifecycleError(
                    f"illegal transition {model.status!r} → {to_status!r} "
                    f"for {model_id!r} (allowed: {sorted(allowed)})")
            self.db.execute(
                "UPDATE model_lifecycle SET status = ?, updated_at = ? WHERE id = ?",
                (to_status, time.time(), model_id))
            self._log_event(model_id, model.status, to_status, detail)
        _log.info("lifecycle: %s %s → %s", model_id, model.status, to_status)
        return self.get(model_id)

    def _log_event(self, model_id: str, from_status: str, to_status: str,
                   detail: str = "") -> None:
        self.db.execute(
            "INSERT INTO lifecycle_events (model_id, from_status, to_status, detail, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (model_id, from_status, to_status, detail, time.time()))

    def download(self, model_id: str) -> ManagedModel:
        """Fetch the artifact.  For a local GGUF path this is a no-op that
        verifies the file exists; for an HF id it downloads via
        :class:`HuggingFaceDownloader`."""
        model = self.get(model_id)
        if model.path:
            path = Path(model.path)
            if not path.exists():
                self._transition(model_id, "failed", f"local file missing: {path}")
                raise LifecycleError(f"local GGUF not found: {path}")
            size = path.stat().st_size
            with self._lock:
                self.db.execute(
                    "UPDATE model_lifecycle SET size_bytes = ?, updated_at = ? WHERE id = ?",
                    (size, time.time(), model_id))
            return self._transition(model_id, "downloaded",
                                    f"local file present ({size} bytes)")
        from .download import HuggingFaceDownloader  # lazy: network-adjacent
        self.models_dir.mkdir(parents=True, exist_ok=True)
        downloader = HuggingFaceDownloader(
            token=self.hf_token, cache_dir=str(self.models_dir))
        try:
            result = downloader.download_repo(model.source, patterns=("*.gguf",))
        except Exception as exc:
            self._transition(model_id, "failed", f"download failed: {exc}")
            raise LifecycleError(f"download of {model.source!r} failed: {exc}") from exc
        files = getattr(result, "files", []) or []
        gguf = next((f for f in files if str(f).endswith(".gguf")), None)
        if gguf is None:
            self._transition(model_id, "failed", "no .gguf in repo")
            raise LifecycleError(f"no GGUF file found in {model.source!r}")
        with self._lock:
            self.db.execute(
                "UPDATE model_lifecycle SET path = ?, size_bytes = ?, updated_at = ? "
                "WHERE id = ?",
                (str(gguf), Path(str(gguf)).stat().st_size if Path(str(gguf)).exists() else 0,
                 time.time(), model_id))
        return self._transition(model_id, "downloaded", f"fetched {gguf}")

    def verify(self, model_id: str, *, expected_sha256: str = "") -> ManagedModel:
        """Hash the artifact with sha256 and record the digest."""
        model = self.get(model_id)
        if not model.path or not Path(model.path).exists():
            raise LifecycleError(
                f"nothing to verify for {model_id!r}: download it first")
        digest = _sha256_of(Path(model.path))
        if expected_sha256 and digest != expected_sha256.lower():
            self._transition(model_id, "failed",
                             f"sha256 mismatch: got {digest[:16]}…")
            raise LifecycleError(
                f"sha256 mismatch for {model_id!r}: expected "
                f"{expected_sha256[:16]}…, got {digest[:16]}…")
        if model.sha256 and model.sha256 != digest:
            self._transition(model_id, "failed",
                             f"artifact changed since last verify: {digest[:16]}…")
            raise LifecycleError(
                f"artifact for {model_id!r} changed on disk since it was verified")
        with self._lock:
            self.db.execute(
                "UPDATE model_lifecycle SET sha256 = ?, updated_at = ? WHERE id = ?",
                (digest, time.time(), model_id))
        return self._transition(model_id, "verified", f"sha256={digest[:16]}…")

    def load(self, model_id: str) -> ManagedModel:
        model = self.get(model_id)
        try:
            handle = self.provisioner.load(model)
        except Exception as exc:
            self._transition(model_id, "failed", f"load failed: {exc}")
            raise
        with self._lock:
            self._handles[model_id] = handle
        return self._transition(model_id, "loaded",
                                f"provisioned via {type(self.provisioner).__name__}")

    def warm(self, model_id: str) -> ManagedModel:
        model = self.get(model_id)
        handle = self._handles.get(model_id)
        try:
            alive = bool(self.provisioner.warm(model, handle))
        except Exception as exc:
            self._transition(model_id, "failed", f"warm ping failed: {exc}")
            raise LifecycleError(f"warm ping for {model_id!r} failed: {exc}") from exc
        if not alive:
            self._transition(model_id, "failed", "health ping did not answer")
            raise LifecycleError(f"model {model_id!r} loaded but not answering health pings")
        return self._transition(model_id, "warm", "health ping ok")

    def unload(self, model_id: str) -> ManagedModel:
        model = self.get(model_id)
        handle = self._handles.pop(model_id, None)
        try:
            self.provisioner.unload(model, handle)
        except Exception as exc:  # noqa: BLE001 — still mark it unloaded
            _log.warning("provisioner unload failed for %s: %s", model_id, exc)
        return self._transition(model_id, "verified", "unloaded; artifact kept")

    def retry(self, model_id: str) -> ManagedModel:
        """Reset a failed model to ``registered`` so the pipeline re-walks."""
        return self._transition(model_id, "registered", "operator retry")

    # ── primary: promote / rollback ──────────────────────────────────────────
    def _kv_get(self, key: str) -> str:
        rows = self.db.query("SELECT value FROM lifecycle_kv WHERE key = ?", (key,))
        return rows[0]["value"] if rows else ""

    def _kv_set(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO lifecycle_kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value))

    @property
    def primary(self) -> str:
        return self._kv_get("primary_model_id")

    def _history_stack(self) -> list[str]:
        raw = self._kv_get("primary_history")
        try:
            stack = json.loads(raw or "[]")
        except json.JSONDecodeError:
            stack = []
        return [s for s in stack if isinstance(s, str)]

    def promote(self, model_id: str) -> ManagedModel:
        """Pin ``model_id`` as the primary mind.

        The model must have earned it: only ``verified``/``loaded``/``warm``
        models are promotable.  The previous primary is pushed onto a
        persisted stack so ``rollback()`` survives restarts.
        """
        model = self.get(model_id)
        if model.status not in _PROMOTABLE:
            raise LifecycleError(
                f"cannot promote {model_id!r} from {model.status!r}: "
                f"walk it to 'verified' first (promotable: {sorted(_PROMOTABLE)})")
        with self._lock:
            previous = self.primary
            if previous and previous != model_id:
                stack = self._history_stack()
                stack.append(previous)
                self._kv_set("primary_history", json.dumps(stack[-32:]))
            self._kv_set("primary_model_id", model_id)
            self._log_event(model_id, model.status, model.status,
                            f"promoted to primary (previous={previous or 'none'})")
        _log.info("lifecycle: primary %s -> %s", previous or "(none)", model_id)
        return model

    def rollback(self) -> ManagedModel:
        """Restore the previous primary.  Raises when there is no history."""
        with self._lock:
            stack = self._history_stack()
            if not stack:
                raise LifecycleError("no previous primary to roll back to")
            restored_id = stack.pop()
            self._kv_set("primary_history", json.dumps(stack))
            self._kv_set("primary_model_id", restored_id)
            self._log_event(restored_id, "", "", "rolled back to primary")
        _log.info("lifecycle: rolled back to primary %s", restored_id)
        return self.get(restored_id)

    def remove(self, model_id: str) -> bool:
        """Forget a model.  Refuses while it is primary or serving."""
        model = self.get(model_id)
        if self.primary == model_id:
            raise LifecycleError(
                f"cannot remove {model_id!r}: it is the primary — "
                "rollback() or promote() another model first")
        if model.status in ("loaded", "warm"):
            self.unload(model_id)
        with self._lock:
            self.db.execute("DELETE FROM lifecycle_events WHERE model_id = ?", (model_id,))
            cur = self.db.execute("DELETE FROM model_lifecycle WHERE id = ?", (model_id,))
        return cur.rowcount > 0
