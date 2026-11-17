"""Burn-after-reading notes — the Privnote build-new (52-websites #25).

A note is sealed under a fresh random key, stored as ciphertext
only, and handed back as a link that carries the key. The key never
lives in the store, the plaintext never lives anywhere but the one
read, and the first read deletes the row — deletion is a storage
fact, provable from the burn log, not a flag.

The link form is ``burn:<note-id>:<base64url-key>``. The recipient
reads it with ``/burn read <code>`` (or ``nm burn read <code>``) on
any surface Devon runs on — the code travels across platforms the
way a link does, and the key travels inside it the way Privnote's
fragment does: the server copy (Devon's store) cannot read it.

Honest limits, stated plainly wherever the link is shown: the note
is protected from *her storage*, not from wherever the owner pastes
the link — anyone holding the link can read it once, and the
platform the link is pasted into can see it. A wrong code or a
wrong password refuses and leaves the note sealed; only a
successful read burns it. An optional extra password gates the
read on top of the link. An optional expiry (default 30 days) bounds
how long an unread note lives; an hourly sweep deletes the expired.

Encryption reuses the house idiom, :mod:`nomorals.core.cipher`
(pure-Python AES-256, HMAC-authenticated ``nmc1`` blobs) with a raw
per-note key — no new crypto, no passphrase derivation for the
note itself. The optional password is stored only as a
PBKDF2-HMAC-SHA256 hash with a random salt, verified in constant
time. The organ never logs a note's text or a link's key.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

from ..core.cipher import CipherError, aes_decrypt, aes_encrypt
from ..core.logging_setup import get_logger

__all__ = [
    "BURN_LOG_MAX",
    "BURN_NOTE_JOB",
    "BURN_NOTE_SPEC",
    "BURN_NOTE_TOOL",
    "DEFAULT_TTL_S",
    "MAX_NOTE_CHARS",
    "BurnNoteStore",
    "ensure_burn_note_job",
    "list_notes",
    "make_link",
    "parse_expiry",
    "parse_link",
    "purge_note",
    "read_note",
    "run_burn_note_tick_once",
    "seal_note",
    "sweep_expired",
]

_log = get_logger(__name__)

#: How long an unread note lives when no expiry is given.
DEFAULT_TTL_S = 30 * 24 * 3600.0
#: A note is a note, not a file.
MAX_NOTE_CHARS = 100_000
#: The burn log keeps the last N burn receipts (id + reason only).
BURN_LOG_MAX = 200

#: Scheduler job (hourly sweep of expired unread notes).
BURN_NOTE_JOB = "burn note sweep"
BURN_NOTE_SPEC = "0 * * * *"
BURN_NOTE_TOOL = "__burn_note_tick__"

_LINK_PREFIX = "burn:"
_PASSWORD_ITERATIONS = 200_000
_EXPIRY_RE = re.compile(r"^(\d+(?:\.\d+)?)([smhdw])$")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing/corrupt store starts empty
        return {"notes": {}, "burned": []}
    if not isinstance(data, dict):
        return {"notes": {}, "burned": []}
    data.setdefault("notes", {})
    data.setdefault("burned", [])
    if not isinstance(data["notes"], dict):
        data["notes"] = {}
    if not isinstance(data["burned"], list):
        data["burned"] = []
    return data


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(Exception):
            os.unlink(tmp)
        raise


def parse_expiry(spec: str | None) -> float | None:
    """Parse an expiry spec into seconds (``None`` = never expires).

    ``"90s"``, ``"1h"``, ``"2d"``, ``"1w"``; ``"never"`` or ``""`` means
    no expiry; ``None`` means the default (30 days). Anything else
    raises :class:`ValueError`.
    """
    if spec is None:
        return DEFAULT_TTL_S
    spec = str(spec).strip().lower()
    if spec in ("", "never", "none"):
        return None
    match = _EXPIRY_RE.match(spec)
    if not match:
        raise ValueError(
            f"bad expiry {spec!r} — use 90s, 1h, 2d, 1w, or never")
    value, unit = float(match.group(1)), match.group(2)
    if value <= 0:
        raise ValueError(f"bad expiry {spec!r} — must be positive")
    return value * {"s": 1.0, "m": 60.0, "h": 3600.0,
                    "d": 86400.0, "w": 7 * 86400.0}[unit]


def make_link(note_id: str, key: bytes) -> str:
    """Mint the link form: ``burn:<note-id>:<base64url-key>``."""
    return (f"{_LINK_PREFIX}{note_id}:"
            f"{base64.urlsafe_b64encode(key).decode('ascii')}")


def parse_link(code: str) -> tuple[str, bytes]:
    """Split a link into ``(note_id, key)``; malformed codes raise."""
    text = (code or "").strip()
    if not text.startswith(_LINK_PREFIX):
        raise ValueError("not a burn-note code — it starts with 'burn:'")
    rest = text[len(_LINK_PREFIX):]
    note_id, sep, key_b64 = rest.partition(":")
    if not sep or not note_id or not key_b64:
        raise ValueError("not a burn-note code — expected "
                         "burn:<note-id>:<key>")
    if ":" in key_b64:
        raise ValueError("not a burn-note code — too many parts")
    try:
        key = base64.urlsafe_b64decode(key_b64.encode("ascii"))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("not a burn-note code — the key is not "
                         "valid base64") from exc
    if len(key) != 32:
        raise ValueError("not a burn-note code — the key is the "
                         "wrong length")
    return note_id, key


def _password_hash(password: str) -> tuple[str, str]:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 salt, _PASSWORD_ITERATIONS)
    return salt.hex(), digest.hex()


def _password_matches(password: str, salt_hex: str,
                      hash_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 salt, _PASSWORD_ITERATIONS)
    return hmac.compare_digest(digest, expected)


class BurnNoteStore:
    """Durable burn-note store: ``<home>/data/burn_notes.json``.

    ``{"notes": {id: row}, "burned": [{note_id, burned_at, reason}]}``
    — the rows hold ciphertext only, never plaintext, never a key.
    The burn log is the proof: every deletion lands there with its
    reason (``read`` | ``expired`` | ``purged``). Writes are atomic
    (temp file + replace); a failed write never corrupts the store.
    """

    _NAME = "burn_notes.json"

    def __init__(self, home: Any) -> None:
        self._path = Path(home) / "data" / self._NAME
        self._data = _read_json(self._path)

    @property
    def path(self) -> str:
        return str(self._path)

    def _save(self) -> None:
        with contextlib.suppress(Exception):
            _atomic_json_write(self._path, self._data)

    def _burn(self, note_id: str, reason: str, *,
              now: float) -> None:
        self._data["notes"].pop(note_id, None)
        log = self._data.setdefault("burned", [])
        log.append({"note_id": note_id, "burned_at": now,
                    "reason": reason})
        del log[:max(0, len(log) - BURN_LOG_MAX)]
        self._save()

    def burned_log(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._data.get("burned", [])]


def seal_note(store: BurnNoteStore, text: str, *,
              expires_in_s: float | None = DEFAULT_TTL_S,
              password: str | None = None,
              now: float | None = None) -> dict[str, Any]:
    """Seal ``text`` under a fresh random key. Returns the link.

    The key is generated here, used once for the encryption, and
    returned inside the link — it is never written to the store.
    ``expires_in_s=None`` means the note lives until read.
    """
    clean = (text or "").strip()
    if not clean:
        raise ValueError("burn_note seal needs some text")
    if len(clean) > MAX_NOTE_CHARS:
        raise ValueError(
            f"burn note is {len(clean)} chars; the cap is "
            f"{MAX_NOTE_CHARS}")
    now_f = float(now if now is not None else time.time())
    key = secrets.token_bytes(32)
    note_id = secrets.token_hex(6)
    try:
        ciphertext = aes_encrypt(clean, key=key)
    except CipherError as exc:
        raise ValueError(f"could not seal the note: {exc}") from exc
    row: dict[str, Any] = {
        "ciphertext": ciphertext,
        "created_at": now_f,
        "expires_at": (now_f + float(expires_in_s)
                       if expires_in_s is not None else None),
        "reads_left": 1,
    }
    if password:
        salt_hex, hash_hex = _password_hash(password)
        row["password_salt"] = salt_hex
        row["password_hash"] = hash_hex
    store._data.setdefault("notes", {})[note_id] = row
    store._save()
    return {"ok": True, "note_id": note_id,
            "link": make_link(note_id, key),
            "expires_at": row["expires_at"],
            "has_password": bool(password)}


def read_note(store: BurnNoteStore, code: str, *,
              password: str | None = None,
              now: float | None = None) -> dict[str, Any]:
    """Read a note once. A successful read burns it.

    Never raises: every refusal carries the reason in words. A
    wrong code or a wrong password refuses and leaves the note
    sealed — only a successful decrypt burns it.
    """
    now_f = float(now if now is not None else time.time())
    try:
        note_id, key = parse_link(code)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    notes = store._data.get("notes") or {}
    row = notes.get(note_id)
    if row is None:
        return {"ok": False,
                "error": "that burn note is gone — already read, "
                         "expired, or never existed"}
    expires_at = row.get("expires_at")
    if expires_at is not None and now_f > float(expires_at):
        store._burn(note_id, "expired", now=now_f)
        return {"ok": False,
                "error": "that burn note expired and is gone"}
    if row.get("password_hash") and (
            not password or not _password_matches(
                password, str(row.get("password_salt") or ""),
                str(row["password_hash"]))):
        return {"ok": False,
                "error": "wrong password — the note stays sealed"}
    try:
        raw = aes_decrypt(str(row["ciphertext"]), key=key)
    except CipherError:
        return {"ok": False,
                "error": "wrong code — the note stays sealed"}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"ok": False,
                "error": "wrong code — the note stays sealed"}
    store._burn(note_id, "read", now=now_f)
    return {"ok": True, "note_id": note_id, "text": text}


def list_notes(store: BurnNoteStore, *,
               now: float | None = None) -> list[dict[str, Any]]:
    """Live notes as metadata only — never contents, never keys."""
    now_f = float(now if now is not None else time.time())
    out = []
    for note_id, row in (store._data.get("notes") or {}).items():
        expires_at = row.get("expires_at")
        out.append({
            "note_id": note_id,
            "created_at": float(row.get("created_at") or 0.0),
            "expires_at": (float(expires_at)
                           if expires_at is not None else None),
            "expired": bool(expires_at is not None
                            and now_f > float(expires_at)),
            "reads_left": int(row.get("reads_left") or 0),
            "has_password": bool(row.get("password_hash")),
        })
    out.sort(key=lambda item: item["created_at"])
    return out


def purge_note(store: BurnNoteStore, note_id: str, *,
               now: float | None = None) -> bool:
    """Delete an unread note early. Returns whether one existed."""
    notes = store._data.get("notes") or {}
    if note_id not in notes:
        return False
    store._burn(note_id, "purged",
                now=float(now if now is not None else time.time()))
    return True


def sweep_expired(store: BurnNoteStore, *,
                  now: float | None = None) -> int:
    """Delete every expired unread note. Returns how many burned."""
    now_f = float(now if now is not None else time.time())
    expired = [note_id for note_id, row in
               (store._data.get("notes") or {}).items()
               if row.get("expires_at") is not None
               and now_f > float(row["expires_at"])]
    for note_id in expired:
        store._burn(note_id, "expired", now=now_f)
    return len(expired)


def run_burn_note_tick_once(context: Any, *,
                            now: float | None = None) -> dict[str, Any]:
    """One sweep tick: delete expired unread notes. Never raises.

    Routine and silent — expiry needs no announcement; the burn log
    keeps the receipt. The tick never touches a live note.
    """
    home = None
    settings = getattr(context, "settings", None) if context else None
    if settings is not None:
        home = getattr(settings, "home", None)
    if not home:
        return {"ok": True, "swept": 0, "live": 0,
                "error": "no home — nothing to sweep"}
    try:
        store = BurnNoteStore(home)
        swept = sweep_expired(store, now=now)
        return {"ok": True, "swept": swept,
                "live": len(list_notes(store, now=now)),
                "ran_at": float(now if now is not None
                                else time.time()),
                "error": ""}
    except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
        return {"ok": False, "swept": 0, "live": 0,
                "error": f"burn note sweep failed: {exc}"}


def ensure_burn_note_job(scheduler: Any, *,
                         spec: str = BURN_NOTE_SPEC) -> Any | None:
    """Register the hourly burn-note sweep job; idempotent by name.

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
            if name != BURN_NOTE_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            BURN_NOTE_JOB, spec, "tool",
            {"tool": BURN_NOTE_TOOL, "args": {}}, routine=True)
    except Exception:
        return None
