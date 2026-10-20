"""Protect: gzip -> encrypt -> SHA-256(ciphertext) -> split.

Encryption is MANDATORY, not optional: the backup contains live session
cookies (``browser_login.py`` persists into SessionManager/DB), and
session cookies ARE credentials. Every blob is encrypted with
:func:`nomorals.core.cipher.aes_encrypt` (AES-256 + PBKDF2, pure Python)
before it leaves the machine. The manifest holds checksums of the
*ciphertext* — commit only the manifest, never the blob.

Key custody: the 32-byte key lives in the vault (service
``github-backup``, username ``backup-key``), never in the repo, never in
``.env``. See :data:`nomorals.backup.errors.KEY_LOSS_WARNING` — lose the
key, lose the backup; keep an offline copy (paper / password manager).

Splitting delegates to :func:`nomorals.builders.deliver._split_archive`
(the existing 100MB-block splitter) instead of re-implementing it.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.cipher import CipherError, aes_decrypt, aes_encrypt
from ..core.logging_setup import get_logger
from .errors import BackupKeyError, ChecksumMismatchError, DiskFullError

__all__ = [
    "VAULT_SERVICE",
    "VAULT_USERNAME",
    "ProtectedFile",
    "generate_backup_key",
    "store_backup_key",
    "load_backup_key",
    "key_from_base64",
    "sha256_file",
    "sha256_bytes",
    "encrypt_bytes",
    "decrypt_bytes",
    "encrypt_file",
    "decrypt_file",
    "split_encrypted",
    "join_parts",
    "protect_entry",
    "protect_snapshot",
]

_log = get_logger(__name__)

VAULT_SERVICE = "github-backup"
VAULT_USERNAME = "backup-key"

#: GitHub Release assets cap at 2GB each; 100MB blocks keep uploads
#: resumable-ish, Termux-friendly, and far under the cap.
SPLIT_BYTES = 100 * 1024 * 1024


@dataclass
class ProtectedFile:
    """One encrypted blob ready to ship (plus its split parts)."""

    rel: str
    encrypted_path: str
    sha256: str  # SHA-256 of the CIPHERTEXT, never the plaintext
    size: int
    parts: list[str]
    kind: str
    schema_version: int = 0
    tables: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rel": self.rel, "encrypted_path": self.encrypted_path,
            "sha256": self.sha256, "size": self.size, "parts": self.parts,
            "kind": self.kind, "schema_version": self.schema_version,
            "tables": self.tables,
        }


def generate_backup_key() -> bytes:
    """A fresh 32-byte backup key (AES-256)."""
    return secrets.token_bytes(32)


def key_from_base64(text: str) -> bytes:
    try:
        key = base64.b64decode(text.strip(), validate=True)
    except Exception as exc:
        raise BackupKeyError(f"stored backup key is not valid base64: {exc}") from exc
    if len(key) not in (16, 24, 32):
        raise BackupKeyError(
            f"stored backup key is {len(key)} bytes; must be 16/24/32"
        )
    return key


def store_backup_key(vault: Any, key: bytes) -> None:
    """Store the key in the vault. Callers must ALSO make the offline copy."""
    if len(key) not in (16, 24, 32):
        raise BackupKeyError("backup key must be 16/24/32 bytes")
    vault.store(
        VAULT_SERVICE, VAULT_USERNAME,
        base64.b64encode(key).decode("ascii"),
        credential_type="api_key",
        tags=["backup", "github-backup"],
        metadata={"purpose": "nomorals-backup blob encryption"},
    )
    _log.info("backup key stored in vault (%s/%s) — make the offline copy",
              VAULT_SERVICE, VAULT_USERNAME)


def load_backup_key(vault: Any) -> bytes:
    """Load the backup key from the vault. Raises BackupKeyError if absent."""
    try:
        cred = vault.get(VAULT_SERVICE, VAULT_USERNAME, mark_used=False)
    except Exception as exc:
        raise BackupKeyError(
            f"no backup key in vault ({VAULT_SERVICE}/{VAULT_USERNAME}) — "
            "generate one with generate_backup_key + store_backup_key, "
            f"and keep the offline copy: {exc}"
        ) from exc
    return key_from_base64(cred.password)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    """AES-256 encrypt via cipher.py. Returns the ``nmc1`` blob as bytes."""
    return aes_encrypt(data, key=key).encode("utf-8")


def decrypt_bytes(blob: bytes, key: bytes) -> bytes:
    """Decrypt an ``nmc1`` blob. CipherError = wrong key or tampered."""
    return aes_decrypt(blob.decode("utf-8"), key=key)


def encrypt_file(src: Path, dest: Path, key: bytes) -> ProtectedFile:
    """gzip -> encrypt -> write ciphertext; return its SHA-256 + size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        raw = src.read_bytes()
        blob = encrypt_bytes(gzip.compress(raw, compresslevel=6), key)
        tmp.write_bytes(blob)
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:
            raise DiskFullError(
                f"disk full while encrypting {src}: previous generation "
                "intact"
            ) from exc
        raise
    tmp.rename(dest)
    return ProtectedFile(
        rel=src.name, encrypted_path=str(dest),
        sha256=sha256_bytes(blob), size=len(blob), parts=[str(dest)],
        kind="file",
    )


def decrypt_file(src: Path, dest: Path, key: bytes) -> Path:
    """Decrypt + gunzip ``src`` -> ``dest``. Raises CipherError on bad key."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        blob = src.read_bytes()
        plain = gzip.decompress(decrypt_bytes(blob, key))
    except CipherError:
        raise
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise ChecksumMismatchError(
            f"cannot decode protected file {src}: {exc}"
        ) from exc
    dest.write_bytes(plain)
    return dest


def _split_archive(archive: Path, split_mb: float) -> list[Path]:
    """Split ``archive`` into ``split_mb``-sized ``.partNN`` pieces.

    Chat platforms cap file uploads (Telegram bots: 50MB) — multi-part
    archives are the honest answer, not silent truncation.

    This is THE one split format in the codebase: it used to live in
    ``builders.deliver`` and ``backup`` imported it from there (an
    upward import — backup is L5, builders is L7). It moved here
    (backup is its other user) and ``builders.deliver`` re-exports it,
    so ``builders.deliver._split_archive`` still exists.
    """
    chunk = int(split_mb * 1024 * 1024)
    data = archive.read_bytes()
    if len(data) <= chunk:
        return [archive]
    parts: list[Path] = []
    total = (len(data) + chunk - 1) // chunk
    for i in range(total):
        piece = archive.with_name(f"{archive.name}.part{i:02d}of{total:02d}")
        piece.write_bytes(data[i * chunk:(i + 1) * chunk])
        parts.append(piece)
    _log.info("split %s into %d parts of ~%.1fMB", archive.name, total,
              split_mb)
    return parts


def split_encrypted(path: Path, *, max_bytes: int = SPLIT_BYTES) -> list[Path]:
    """Split a protected file into <= ``max_bytes`` parts.

    Uses the single :func:`_split_archive` format. Returns ``[path]``
    unchanged when it already fits.
    """
    return _split_archive(path, max_bytes / (1024 * 1024))


_PART_NAME_RE = re.compile(r"\.part(\d+)of\d+")


def _part_order(path: Path) -> tuple[int, int, str]:
    """Sort key for split parts: by NUMERIC part index.

    Plain name order breaks past 99 parts — ``.part100of128`` sorts
    before ``.part99of128`` lexicographically — which reassembles the
    ciphertext in the wrong order. Non-part names keep name order.
    """
    match = _PART_NAME_RE.search(path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, 0, path.name)


def join_parts(parts: list[Path], dest: Path) -> Path:
    """Reassemble split parts (numeric part order) into ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        for part in sorted(parts, key=_part_order):
            with part.open("rb") as fh:
                out.write(fh.read())
    return dest


def protect_entry(entry: Any, out_dir: Path, key: bytes) -> ProtectedFile:
    """Protect one snapshot entry: gzip -> encrypt -> checksum -> split."""
    src = Path(entry.path)
    safe = entry.rel.replace("/", "__")
    dest = out_dir / (safe + ".gz.enc")
    protected = encrypt_file(src, dest, key)
    protected.rel = entry.rel
    protected.kind = entry.kind
    protected.schema_version = getattr(entry, "schema_version", 0)
    protected.tables = getattr(entry, "tables", 0)
    parts = split_encrypted(dest)
    protected.parts = [str(p) for p in parts]
    return protected


def protect_snapshot(snapshot: Any, out_dir: Path, key: bytes) -> list[ProtectedFile]:
    """Protect every entry of a snapshot. Returns the protected blobs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protected: list[ProtectedFile] = []
    for entry in snapshot.entries:
        protected.append(protect_entry(entry, out_dir, key))
    _log.info("protected %d snapshot entries -> %s",
              len(protected), out_dir)
    return protected
