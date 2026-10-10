"""Chat-bound grounded-research sessions.

Each chat gets its own :class:`GroundedSession` with its own on-disk
directory: uploaded documents live there, and each session lazily opens its
own ``vectors.db`` sqlite for the vector backend (owner_type="grounded").
Sessions expire after ``ttl_seconds`` of inactivity; :meth:`sweep` reaps
them. Dropping a session deletes its whole directory, so no document or
vector outlives the session it was ingested into.
"""

from __future__ import annotations

import base64
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..memory.embeddings import Embedder
from ..memory.vector_backends import LegacyStoreBackend
from ..storage.db import Database
from .grounded import GroundedSession

_log = get_logger(__name__)

__all__ = ["GroundedSessionStore"]

_SESSION_JSON = "session.json"
_VECTORS_DB = "vectors.db"


def _urlsafe(chat_key: str) -> str:
    """Filesystem-safe directory name for a chat key."""
    return base64.urlsafe_b64encode(chat_key.encode("utf-8")).decode("ascii").rstrip("=")


def _sanitize_filename(name: str) -> str:
    base = Path(name or "upload").name
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base).strip("._")
    return safe or "upload"


@dataclass
class _Bound:
    chat_key: str
    dir: Path
    session: GroundedSession
    db: Database
    last_used: float
    docs: list[dict[str, str]] = field(default_factory=list)  # doc_id/title/filename
    qa: list[dict[str, Any]] = field(default_factory=list)  # ts/question/answer


class GroundedSessionStore:
    """Owns one :class:`GroundedSession` per chat key.

    ``embedder`` is optional; when omitted a hashing :class:`Embedder` is
    created lazily on first bind — deterministic, offline, zero-cost, and
    safe on every profile. Pass an explicit embedder (e.g. provider-backed)
    to upgrade retrieval quality.
    """

    def __init__(self, root_dir: str | Path, *, ttl_seconds: float = 86400,
                 embedder: Any = None) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = float(ttl_seconds)
        self.embedder = embedder
        self._hash_embedder: Embedder | None = None
        self._lock = threading.RLock()
        self._sessions: dict[str, _Bound] = {}

    # ── session lifecycle ────────────────────────────────────────────

    def bind(self, chat_key: str) -> GroundedSession:
        """Get-or-create the session for ``chat_key``; refreshes last-used."""
        key = str(chat_key)
        with self._lock:
            bound = self._sessions.get(key)
            if bound is not None:
                if self._expired(bound.last_used):
                    self._drop_locked(key)
                else:
                    bound.last_used = time.time()
                    self._persist_locked(bound)
                    return bound.session
            return self._create_locked(key)

    def get(self, chat_key: str) -> GroundedSession | None:
        """The session for ``chat_key``, or None when missing/expired.

        Any successful get counts as activity and refreshes last-used.
        """
        key = str(chat_key)
        with self._lock:
            bound = self._sessions.get(key)
            if bound is None:
                return None
            if self._expired(bound.last_used):
                self._drop_locked(key)
                return None
            bound.last_used = time.time()
            self._persist_locked(bound)
            return bound.session

    def drop(self, chat_key: str) -> None:
        """Remove the session and delete its whole directory."""
        with self._lock:
            self._drop_locked(str(chat_key))

    def sweep(self) -> int:
        """Drop every idle-expired session (in memory and orphaned on disk).

        Returns the number of sessions removed.
        """
        dropped = 0
        with self._lock:
            for key in [k for k, b in self._sessions.items()
                        if self._expired(b.last_used)]:
                self._drop_locked(key)
                dropped += 1
            live_dirs = {self._session_dir(k).name for k in self._sessions}
            for child in self.root.iterdir():
                if not child.is_dir() or child.name in live_dirs:
                    continue
                state = self._read_state(child)
                if state is None:
                    # no session.json: only reap when the dir itself is idle-old,
                    # so we never race a bind that is still being created
                    try:
                        idle = time.time() - child.stat().st_mtime
                    except OSError:
                        continue
                    if idle <= self.ttl:
                        continue
                elif not self._expired(float(state.get("last_used", 0))):
                    continue
                shutil.rmtree(child, ignore_errors=True)
                dropped += 1
                _log.debug("grounded store: swept orphan session dir %s", child.name)
        return dropped

    # ── documents ────────────────────────────────────────────────────

    def add_upload(self, chat_key: str, data: bytes, filename: str,
                   mime: str = "") -> str:
        """Store ``data`` in the session dir and ingest it. Returns doc id.

        The file on disk is the session's own copy, so dropping the session
        removes the document completely.
        """
        if not data:
            raise ValueError("empty upload")
        session = self.bind(chat_key)
        key = str(chat_key)
        with self._lock:
            bound = self._sessions[key]
            safe = _sanitize_filename(filename)
            target = bound.dir / safe
            n = 1
            while target.exists():
                n += 1
                stem, suffix = safe.rsplit(".", 1) if "." in safe else (safe, "")
                target = bound.dir / (f"{stem}-{n}.{suffix}" if suffix else f"{stem}-{n}")
            try:
                target.write_bytes(data)
            except OSError as exc:
                raise ValueError(f"could not store upload {safe!r}: {exc}") from exc
            try:
                doc_id = session.add_file(target)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            bound.docs.append({"doc_id": doc_id,
                               "title": filename or safe,
                               "filename": target.name})
            bound.last_used = time.time()
            self._persist_locked(bound)
            return doc_id

    def list_docs(self, chat_key: str) -> list[dict[str, str]]:
        """``[{doc_id, title}]`` for the chat's session; [] when none."""
        key = str(chat_key)
        with self._lock:
            bound = self._sessions.get(key)
            if bound is None or self._expired(bound.last_used):
                if bound is not None:
                    self._drop_locked(key)
                return []
            return [{"doc_id": d["doc_id"], "title": d["title"]} for d in bound.docs]

    # ── Q&A log & export ─────────────────────────────────────────────

    def log_qa(self, chat_key: str, question: str, answer: str) -> None:
        """Append a question/answer pair to the session's thread log.

        Powers :meth:`export_thread` — the session's grounded Q&A as
        portable markdown. Best-effort persistence; never raises.
        """
        key = str(chat_key)
        with self._lock:
            bound = self._sessions.get(key)
            if bound is None:
                return
            bound.qa.append({
                "ts": time.time(),
                "question": (question or "")[:500],
                "answer": (answer or "")[:8000],
            })
            # Keep the log bounded — a chat thread, not an archive.
            bound.qa = bound.qa[-100:]
            bound.last_used = time.time()
            self._persist_locked(bound)

    def export_thread(self, chat_key: str) -> str:
        """The session's Q&A thread as markdown (docs + questions + answers).

        Returns "" when the session has no logged Q&A.
        """
        key = str(chat_key)
        with self._lock:
            bound = self._sessions.get(key)
            if bound is None or self._expired(bound.last_used):
                if bound is not None:
                    self._drop_locked(key)
                return ""
            qa = list(bound.qa)
            docs = list(bound.docs)
        if not qa:
            return ""
        lines = ["# Grounded Q&A thread", ""]
        if docs:
            lines += ["## Documents",
                      "".join(f"- {d['title']}\n" for d in docs), ""]
        for i, entry in enumerate(qa, 1):
            lines += [f"## Q{i}: {entry['question']}", "",
                      str(entry["answer"]), ""]
        return "\n".join(lines).strip()

    def session_stats(self, chat_key: str) -> dict[str, Any]:
        """Observability for one chat's grounded session."""
        key = str(chat_key)
        with self._lock:
            bound = self._sessions.get(key)
            if bound is None or self._expired(bound.last_used):
                if bound is not None:
                    self._drop_locked(key)
                return {"active": False}
            try:
                size = sum(p.stat().st_size for p in bound.dir.rglob("*")
                           if p.is_file())
            except OSError:
                size = -1
            return {
                "active": True,
                "docs": len(bound.docs),
                "qa_logged": len(bound.qa),
                "chunks": len(getattr(bound.session, "doc_ids", [])),
                "idle_s": round(time.time() - bound.last_used, 1),
                "dir_bytes": size,
            }

    # ── internals ────────────────────────────────────────────────────

    def _session_dir(self, key: str) -> Path:
        return self.root / _urlsafe(key)

    def _expired(self, last_used: float) -> bool:
        return (time.time() - last_used) > self.ttl

    def _embedder_for_session(self) -> Embedder:
        if self.embedder is not None:
            return self.embedder
        if self._hash_embedder is None:
            self._hash_embedder = Embedder(provider="hashing")
            _log.debug("grounded store: using hashing embedder (offline, cheap)")
        return self._hash_embedder

    def _create_locked(self, key: str) -> GroundedSession:
        sdir = self._session_dir(key)
        state = self._read_state(sdir)
        if state is not None and self._expired(float(state.get("last_used", 0))):
            # a previous process left an expired session behind — start clean
            shutil.rmtree(sdir, ignore_errors=True)
            state = None
        sdir.mkdir(parents=True, exist_ok=True)
        # a stale vectors.db from a previous process must never be searched
        # by a session that has no metadata for its vectors
        vdb = sdir / _VECTORS_DB
        if vdb.exists():
            vdb.unlink()
        db = Database(vdb)
        db.migrate()  # creates the embeddings table the legacy backend needs
        try:
            backend = LegacyStoreBackend(db, owner_type="grounded")
            session = GroundedSession(embedder=self._embedder_for_session(),
                                      vector_db=backend)
        except Exception:
            db.close()
            raise
        bound = _Bound(chat_key=key, dir=sdir, session=session, db=db,
                       last_used=time.time())
        if state:
            self._rehydrate_locked(bound, state)
        self._sessions[key] = bound
        self._persist_locked(bound)
        return session

    def _rehydrate_locked(self, bound: _Bound, state: dict[str, Any]) -> None:
        """Re-ingest a previous process's uploads into a fresh session.

        The vector backend starts empty, so re-ingesting the same files
        reproduces identical chunk ids and vectors with no duplicates.
        """
        for doc in state.get("docs") or []:
            fpath = bound.dir / str(doc.get("filename", ""))
            if not fpath.is_file():
                continue
            try:
                doc_id = bound.session.add_file(fpath)
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the session
                _log.warning("grounded store: could not re-ingest %s (%s)",
                             fpath.name, exc)
                continue
            bound.docs.append({"doc_id": doc_id,
                               "title": str(doc.get("title", fpath.name)),
                               "filename": fpath.name})
        # Restore the Q&A thread log (bounded, like log_qa keeps it).
        bound.qa = [dict(e) for e in (state.get("qa") or [])
                    if isinstance(e, dict)][-100:]

    def _drop_locked(self, key: str) -> None:
        bound = self._sessions.pop(key, None)
        if bound is not None:
            try:
                bound.db.close()
            except Exception:  # noqa: BLE001 - teardown only
                pass
        shutil.rmtree(self._session_dir(key), ignore_errors=True)

    def _persist_locked(self, bound: _Bound) -> None:
        state = {"last_used": bound.last_used, "docs": bound.docs,
                 "qa": bound.qa}
        tmp = bound.dir / (_SESSION_JSON + ".tmp")
        try:
            tmp.write_text(json.dumps(state), encoding="utf-8")
            tmp.replace(bound.dir / _SESSION_JSON)
        except OSError as exc:  # noqa: BLE001 - persistence is best-effort
            _log.debug("grounded store: could not persist session state (%s)", exc)

    @staticmethod
    def _read_state(sdir: Path) -> dict[str, Any] | None:
        p = sdir / _SESSION_JSON
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None
