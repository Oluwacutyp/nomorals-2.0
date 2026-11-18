"""Expiring file shares — the File.io build (52-websites #26).

Devon moves files constantly (artifacts, exports, media) but her
storage is durable by default. This organ is the File.io shape over
her existing storage: a share record with a download budget and a
TTL, enforced by a reaper, deletion being the point rather than an
accident.

A share is minted by the owner only — she is the uploader,
recipients only download (narrower than File.io, deliberately so:
anonymous public upload is an abuse magnet the original mitigates
with a business, and Devon has no business to mitigate with). The
minted bytes live in her house blob store (content-addressed,
deduped, refcounted): minting a path stores it once, minting an
already-stored blob just bumps its refcount, and a share's death
releases its ref — the bytes die only when no one else holds them.

The link is a capability: ``/share/<token>`` where the token is 48
lowercase hex chars (192 bits of entropy, and lowercase so it
passes the API server's ``_ROUTE`` pattern). The token is never
logged anywhere in this path — journal entries carry the share id
only, never the token. Anyone holding the link can download until
the budget dies or the expiry passes; that is the design, not a
bug, and every reply that hands out a link says so.

The web surface is ``GET /share/<token>`` on her API server (a
special path beside the site pages — raw bytes cannot ride the
JSON dispatch). Unknown, expired and exhausted tokens all answer
404, identically: no oracle for guessing tokens.

The reaper is an hourly scheduler tick (``__share_tick__``): it
deletes expired and exhausted shares — file (blob ref released)
and record — and journals each death. The journal keeps receipts
only (share id + reason + time), never contents, never tokens.
"""

from __future__ import annotations

import contextlib
import json
import mimetypes
import os
import re
import secrets
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as _time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .blob import BlobStore
from .db import Database

__all__ = [
    "SHARE_JOB",
    "SHARE_LOG_MAX",
    "SHARE_SPEC",
    "SHARE_TOOL",
    "DEFAULT_TTL_S",
    "ShareRecord",
    "ShareStore",
    "ensure_share_job",
    "list_shares",
    "mint_share",
    "parse_share_expiry",
    "revoke_share",
    "run_share_tick_once",
    "serve_share",
    "sweep_shares",
]

_log = get_logger(__name__)

#: How long a share lives when no expiry is given (File.io's default).
DEFAULT_TTL_S = 14 * 24 * 3600.0
#: The share journal keeps the last N receipts (id + reason only).
SHARE_LOG_MAX = 200

#: Scheduler job (hourly sweep of expired/exhausted shares).
SHARE_JOB = "share sweep"
SHARE_SPEC = "0 * * * *"
SHARE_TOOL = "__share_tick__"

#: 24 random bytes -> 48 lowercase hex chars (192 bits). Lowercase
#: matters: the API server's _ROUTE pattern is ^/[a-z0-9_/-]+$,
#: so a token carrying uppercase would 400 before it ever reached
#: the share lookup.
_TOKEN_HEX_CHARS = 48
_TOKEN_RE = re.compile(r"^[0-9a-f]{48}$")
_EXPIRY_RE = re.compile(r"^(\d+(?:\.\d+)?)([smhdw])$")
_UNIT_S = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing/corrupt store starts empty
        return {"shares": {}, "journal": []}
    if not isinstance(data, dict):
        return {"shares": {}, "journal": []}
    data.setdefault("shares", {})
    data.setdefault("journal", [])
    if not isinstance(data["shares"], dict):
        data["shares"] = {}
    if not isinstance(data["journal"], list):
        data["journal"] = []
    return data


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(Exception):
            os.unlink(tmp)
        raise


def parse_share_expiry(value: str | None, *, now: float | None = None) -> float | None:
    """Parse a share expiry into an epoch timestamp (``None`` = never).

    Accepted forms — ``""``/``None`` means the default 14 days;
    ``never`` means no expiry; durations ``90s``/``30m``/``6h``/
    ``2d``/``1w`` count from now; a weekday name (``friday``) means
    the end of the next such day; an ISO date (``2026-12-25``) means
    the end of that day. A value that matches nothing raises
    ``ValueError`` naming the accepted forms — never a silent
    default.
    """
    moment = float(now if now is not None else time.time())
    text = (value or "").strip().lower()
    if not text:
        return moment + DEFAULT_TTL_S
    if text == "never":
        return None
    match = _EXPIRY_RE.match(text)
    if match:
        return moment + float(match.group(1)) * _UNIT_S[match.group(2)]
    if text in _WEEKDAYS:
        today = datetime.fromtimestamp(moment).date()
        delta = (_WEEKDAYS[text] - today.weekday()) % 7
        if delta == 0:
            delta = 7
        target = today + timedelta(days=delta)
        return datetime.combine(target, _time(23, 59, 59)).timestamp()
    match = _ISO_DATE_RE.match(text)
    if match:
        try:
            target = date(int(match.group(1)), int(match.group(2)),
                          int(match.group(3)))
        except ValueError as exc:
            raise ValueError(f"bad share expiry {value!r}: {exc}") from exc
        return datetime.combine(target, _time(23, 59, 59)).timestamp()
    raise ValueError(
        f"bad share expiry {value!r} — use a duration (90s|30m|6h|2d|1w), "
        "a weekday (friday), an ISO date (2026-12-25), or 'never'")


@dataclass
class ShareRecord:
    """One expiring share: a blob ref + a budget + a deadline."""

    share_id: str
    file_ref: str  # "blob:<sha256>" — the bytes live in the house blob store
    filename: str
    mime: str
    size: int
    token: str
    downloads_left: int
    downloads_used: int = 0
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None

    def __post_init__(self) -> None:
        if not self.share_id:
            raise ValueError("share_id is required")
        if not self.file_ref.startswith("blob:") or len(self.file_ref) != 69:
            raise ValueError(f"bad file_ref {self.file_ref!r} — 'blob:<sha256>'")
        if not self.filename:
            raise ValueError("filename is required")
        if not _TOKEN_RE.match(self.token or ""):
            raise ValueError("token must be 48 lowercase hex chars")
        if self.downloads_left < 0:
            raise ValueError("downloads_left cannot be negative")
        if self.size < 0:
            raise ValueError("size cannot be negative")

    @property
    def sha256(self) -> str:
        return self.file_ref[5:]

    @property
    def link_path(self) -> str:
        """The capability path — the token IS the link, never the file path."""
        return f"/share/{self.token}"

    def expired(self, *, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return float(now if now is not None else time.time()) >= self.expires_at

    def exhausted(self) -> bool:
        return self.downloads_left <= 0

    def live(self, *, now: float | None = None) -> bool:
        return not self.expired(now=now) and not self.exhausted()

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON shape. The token rides here because the owner
        needs the link — it is never written to a log."""
        return {
            "share_id": self.share_id,
            "file_ref": self.file_ref,
            "filename": self.filename,
            "mime": self.mime,
            "size": self.size,
            "token": self.token,
            "downloads_left": self.downloads_left,
            "downloads_used": self.downloads_used,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    def metadata(self, *, base_url: str = "") -> dict[str, Any]:
        """What list/status show: everything except the token's raw
        value is metadata; the link carries it, as it must."""
        return {
            "share_id": self.share_id,
            "filename": self.filename,
            "mime": self.mime,
            "size": self.size,
            "link": (base_url.rstrip("/") + self.link_path) if base_url else self.link_path,
            "downloads_left": self.downloads_left,
            "downloads_used": self.downloads_used,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> ShareRecord:
        return cls(
            share_id=str(row["share_id"]),
            file_ref=str(row["file_ref"]),
            filename=str(row["filename"]),
            mime=str(row.get("mime") or "application/octet-stream"),
            size=int(row.get("size") or 0),
            token=str(row["token"]),
            downloads_left=int(row.get("downloads_left") or 0),
            downloads_used=int(row.get("downloads_used") or 0),
            created_at=float(row.get("created_at") or 0),
            expires_at=(float(row["expires_at"])
                        if row.get("expires_at") is not None else None),
        )


class ShareStore:
    """Durable registry of expiring shares (<home>/data/shares.json).

    JSON-backed when a ``home`` is given (the share surface stores
    it at ``<home>/data/shares.json``); in-memory when it is not.
    The persistence shape mirrors the burn-note store: tolerant
    load (a missing file starts empty, a bad row is skipped), atomic
    tmp+replace save after every mutation. A corrupt store file is
    quarantined aside (``shares.json.corrupt-<epoch>``) instead of
    silently discarded, so the loss is visible and recoverable.
    """

    def __init__(self, home: str | None = None, *,
                 blob_store: BlobStore | None = None) -> None:
        self.home = str(home) if home else ""
        self.path = (str(Path(self.home).expanduser() / "data" / "shares.json")
                     if self.home else "")
        self._blob_store = blob_store
        self.shares: dict[str, ShareRecord] = {}
        self.journal: list[dict[str, Any]] = []
        if self.path:
            self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> None:
        path = Path(self.path)
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt store quarantined, not lost silently
            stamp = int(time.time())
            with contextlib.suppress(Exception):
                path.replace(path.with_name(f"shares.json.corrupt-{stamp}"))
            _log.warning("share store corrupt — quarantined, starting empty")
            return
        if not isinstance(raw, dict):
            return
        for row in (raw.get("shares") or {}).values():
            try:
                rec = ShareRecord.from_dict(row)
            except (TypeError, ValueError, KeyError):
                continue
            self.shares[rec.share_id] = rec
        journal = raw.get("journal") or []
        if isinstance(journal, list):
            self.journal = [j for j in journal if isinstance(j, dict)]

    def _save(self) -> None:
        if not self.path:
            return
        _atomic_json_write(Path(self.path), {
            "shares": {sid: rec.to_dict() for sid, rec in self.shares.items()},
            "journal": self.journal[-SHARE_LOG_MAX:],
        })

    # ── blob store ───────────────────────────────────────────────
    def blob_store(self) -> BlobStore:
        """The house blob store. The runtime migrates the home DB at
        boot, so the ``blobs`` table exists by the time a share is
        minted or served; a home whose DB was never migrated fails
        honestly here instead of inventing a second store."""
        if self._blob_store is None:
            if not self.home:
                raise ValueError("share store needs a home to open the blob store")
            root = Path(self.home).expanduser()
            db_path = root / "data" / "nomorals.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db = Database(db_path)
            if db.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='blobs'"
            ).fetchone() is None:
                # A fresh home (or a home whose boot migration has not
                # run yet) still mints and serves: bring the blob
                # tables up here instead of assuming boot did it.
                db.migrate()
            self._blob_store = BlobStore(db, root / "data" / "blobs")
        return self._blob_store

    # ── journal ──────────────────────────────────────────────────
    def _log(self, share_id: str, action: str) -> None:
        # The token never appears here: receipts carry the id only.
        self.journal.append({
            "share_id": share_id, "action": action,
            "at": time.time(),
        })
        if len(self.journal) > SHARE_LOG_MAX:
            del self.journal[: len(self.journal) - SHARE_LOG_MAX]

    # ── shares ───────────────────────────────────────────────────
    def get(self, share_id: str) -> ShareRecord | None:
        return self.shares.get(share_id)

    def by_token(self, token: str) -> ShareRecord | None:
        for rec in self.shares.values():
            if rec.token == token:
                return rec
        return None

    def _kill(self, rec: ShareRecord, reason: str) -> None:
        """Delete a share — record AND bytes. The blob ref is
        released, so the bytes die only when no one else holds
        them; a blob shared with an artifact or another share
        survives. The journal keeps the receipt."""
        try:
            self.blob_store().release(rec.sha256, delete_at_zero=True)
        except Exception as exc:  # noqa: BLE001 - a dead ref must not block the kill
            _log.warning("share kill: blob release failed for %s: %s",
                         rec.share_id, exc)
        self.shares.pop(rec.share_id, None)
        self._log(rec.share_id, reason)
        self._save()

    def journal_entries(self) -> list[dict[str, Any]]:
        return list(self.journal)


def mint_share(store: ShareStore, source: str, *,
               downloads: int = 1,
               expires_in_s: float | None = None,
               never: bool = False,
               filename: str = "",
               now: float | None = None) -> dict[str, Any]:
    """Mint an owner-only share over a file.

    ``source`` is a filesystem path (stored into the house blob
    store once — deduped, so an already-stored file costs nothing
    extra) or ``blob:<sha256>`` (an already-stored blob, whose
    refcount is bumped so the share owns a ref). ``downloads`` is
    the download budget (default 1 — share once); ``expires_in_s``
    is seconds from now (``None`` = the 14-day default); ``never``
    means no expiry at all. Returns the share metadata with its
    link. Never raises for a bad source — it refuses in-band with
    the reason named.
    """
    moment = float(now if now is not None else time.time())
    if not isinstance(downloads, int) or downloads < 1:
        return {"ok": False, "action": "mint",
                "error": "downloads must be a whole number >= 1"}
    if expires_in_s is not None and expires_in_s <= 0:
        return {"ok": False, "action": "mint",
                "error": "expiry must be in the future"}
    blobs = store.blob_store()
    ref = (source or "").strip()
    if ref.startswith("blob:"):
        sha = ref[5:]
        info = blobs.info(sha)
        if info is None:
            return {"ok": False, "action": "mint",
                    "error": f"no blob {sha[:12]}… in the store"}
        blobs._bump_refcount(sha, 1)
        name = filename or f"shared-{sha[:12]}"
        mime = info.mime
        size = info.size
    else:
        path = Path(ref).expanduser()
        if not path.is_file():
            return {"ok": False, "action": "mint",
                    "error": f"no such file: {ref}"}
        try:
            info = blobs.put_file(path)
        except Exception as exc:  # noqa: BLE001 - a failed store is a refusal, not a crash
            return {"ok": False, "action": "mint",
                    "error": f"could not store {ref}: {exc}"}
        sha = info.sha256
        name = filename or path.name
        mime = info.mime
        size = info.size
    if not name:
        name = f"shared-{sha[:12]}"
    if not mime:
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    if never:
        expires_at = None
    elif expires_in_s is not None:
        expires_at = moment + expires_in_s
    else:
        expires_at = moment + DEFAULT_TTL_S
    rec = ShareRecord(
        share_id=secrets.token_hex(8),
        file_ref=f"blob:{sha}",
        filename=name,
        mime=mime,
        size=size,
        token=secrets.token_hex(24),
        downloads_left=downloads,
        created_at=moment,
        expires_at=expires_at,
    )
    store.shares[rec.share_id] = rec
    store._log(rec.share_id, "minted")
    store._save()
    return {"ok": True, "action": "mint", **rec.metadata()}


def serve_share(store: ShareStore, token: str, *,
                now: float | None = None) -> tuple[str, dict[str, Any]]:
    """Serve one download for a token.

    Returns ``("ok", {bytes, filename, mime, size, downloads_left})``
    or ``("gone", {error})``. Unknown, expired and exhausted tokens
    all answer ``gone`` with the same words — no oracle for guessing
    tokens. A download that spends the last of the budget serves
    these bytes and then kills the share (record + blob ref). A
    record whose bytes are gone from the blob store is killed as
    ``lost`` and answers gone — the record never outlives its file.
    """
    moment = float(now if now is not None else time.time())
    token = (token or "").strip()
    rec = store.by_token(token) if _TOKEN_RE.match(token) else None
    if rec is None:
        return "gone", {"error": "no such share"}
    if rec.expired(now=moment):
        store._kill(rec, "expired")
        return "gone", {"error": "no such share"}
    if rec.exhausted():
        store._kill(rec, "exhausted")
        return "gone", {"error": "no such share"}
    try:
        data = store.blob_store().get_bytes(rec.sha256)
    except Exception:  # noqa: BLE001 - bytes gone means the share is dead, not an error page
        store._kill(rec, "lost")
        return "gone", {"error": "no such share"}
    rec.downloads_left -= 1
    rec.downloads_used += 1
    store._log(rec.share_id, "downloaded")
    if rec.exhausted():
        # These bytes are served; the share dies behind them.
        store._kill(rec, "exhausted")
        return "ok", {"bytes": data, "filename": rec.filename,
                      "mime": rec.mime, "size": len(data),
                      "downloads_left": 0}
    store._save()
    return "ok", {"bytes": data, "filename": rec.filename,
                  "mime": rec.mime, "size": len(data),
                  "downloads_left": rec.downloads_left}


def revoke_share(store: ShareStore, share_id: str) -> bool:
    """Revoke a share early (record + blob ref die now). False when
    the id is not a live share."""
    rec = store.get((share_id or "").strip())
    if rec is None:
        return False
    store._kill(rec, "revoked")
    return True


def list_shares(store: ShareStore, *, now: float | None = None,
                base_url: str = "") -> list[dict[str, Any]]:
    """Live shares only, as metadata with links. Expired or
    exhausted records are swept out of the listing (and killed) so
    a list never advertises a dead link."""
    moment = float(now if now is not None else time.time())
    out: list[dict[str, Any]] = []
    for rec in sorted(store.shares.values(), key=lambda r: r.created_at):
        if not rec.live(now=moment):
            store._kill(rec, "expired" if rec.expired(now=moment) else "exhausted")
            continue
        out.append(rec.metadata(base_url=base_url))
    return out


def sweep_shares(store: ShareStore, *, now: float | None = None) -> int:
    """Delete every expired or exhausted share (file + record),
    journalled. Returns how many died. A live share is never
    touched."""
    moment = float(now if now is not None else time.time())
    dead = [rec for rec in list(store.shares.values())
            if not rec.live(now=moment)]
    for rec in dead:
        store._kill(rec, "expired" if rec.expired(now=moment) else "exhausted")
    return len(dead)


def run_share_tick_once(context: Any, *, now: float | None = None) -> dict[str, Any]:
    """One sweep tick: delete expired/exhausted shares. Never raises.

    Routine and silent — expiry needs no announcement; the journal
    keeps the receipt. The tick never touches a live share.
    """
    home = None
    settings = getattr(context, "settings", None) if context else None
    if settings is not None:
        home = getattr(settings, "home", None)
    if not home:
        return {"ok": True, "swept": 0, "live": 0,
                "error": "no home — nothing to sweep"}
    try:
        store = ShareStore(home)
        swept = sweep_shares(store, now=now)
        return {"ok": True, "swept": swept,
                "live": len(list_shares(store, now=now)),
                "ran_at": float(now if now is not None else time.time()),
                "error": ""}
    except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
        return {"ok": False, "swept": 0, "live": 0,
                "error": f"share sweep failed: {exc}"}


def ensure_share_job(scheduler: Any, *,
                     spec: str = SHARE_SPEC) -> Any | None:
    """Register the hourly share sweep job; idempotent by name.

    A job with the same name but a stale spec is replaced; a job
    that already matches is returned untouched. Best-effort — a
    scheduler that cannot store the job returns None rather than
    raising.
    """
    if scheduler is None:
        return None
    try:
        for job in scheduler.list_jobs() or []:
            if isinstance(job, dict):
                name, job_spec, job_id = (job.get("name"),
                                          job.get("spec"),
                                          job.get("id"))
            else:
                name = getattr(job, "name", None)
                job_spec = getattr(job, "spec", None)
                job_id = getattr(job, "id", None)
            if name != SHARE_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            SHARE_JOB, spec, "tool",
            {"tool": SHARE_TOOL, "args": {}}, routine=True)
    except Exception:
        return None
