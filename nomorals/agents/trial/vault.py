"""Encrypted credential vault for single-account trials.

One purpose: when the owner tries a platform with ONE personal account, the
login + password are stored encrypted under ``~/.nomorals`` (never in the
repo, never in chat logs) so they can be re-delivered on request.

Crypto (stdlib only, no new dependency):
* a 32-byte random master key lives in ``trial.key`` (mode 0600, next to
  the data — it protects against casual eyeballs, not a stolen device);
* each entry is sealed independently: ``scrypt`` KDF (per-entry salt) +
  HMAC-SHA256 counter keystream (AES-free CTR) + encrypt-then-MAC.
Tampering with the file fails the MAC and the entry is reported unreadable.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

__all__ = ["TrialVault"]

_SALT_LEN = 16
_NONCE_LEN = 16
_TAG_LEN = 32
_MAGIC = b"NMTRIAL1"


def _derive(master: bytes, salt: bytes) -> bytes:
    return hashlib.scrypt(master, salt=salt, n=1 << 14, r=8, p=1, dklen=32)


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        counter += 1
    return bytes(out[:length])


def _seal(master: bytes, plaintext: bytes) -> bytes:
    salt = secrets.token_bytes(_SALT_LEN)
    nonce = secrets.token_bytes(_NONCE_LEN)
    key = _derive(master, salt)
    ct = bytes(p ^ k for p, k in zip(plaintext, _keystream(key, nonce, len(plaintext))))
    tag = hmac.new(key, nonce + ct, hashlib.sha256).digest()[:_TAG_LEN]
    return _MAGIC + salt + nonce + tag + ct


def _open(master: bytes, blob: bytes) -> bytes:
    if not blob.startswith(_MAGIC) or len(blob) < len(_MAGIC) + _SALT_LEN + _NONCE_LEN + _TAG_LEN:
        raise ValueError("not a sealed trial credential")
    off = len(_MAGIC)
    salt = blob[off:off + _SALT_LEN]
    nonce = blob[off + _SALT_LEN:off + _SALT_LEN + _NONCE_LEN]
    tag = blob[off + _SALT_LEN + _NONCE_LEN:off + _SALT_LEN + _NONCE_LEN + _TAG_LEN]
    ct = blob[off + _SALT_LEN + _NONCE_LEN + _TAG_LEN:]
    off = off + _SALT_LEN + _NONCE_LEN + _TAG_LEN
    key = _derive(master, salt)
    expect = hmac.new(key, nonce + ct, hashlib.sha256).digest()[:_TAG_LEN]
    if not hmac.compare_digest(tag, expect):
        raise ValueError("credential tampered with or wrong key (MAC mismatch)")
    return bytes(c ^ k for c, k in zip(ct, _keystream(key, nonce, len(ct))))


class TrialVault:
    """Store / fetch / list single-account trial credentials."""

    def __init__(self, home: str | Path) -> None:
        self.home = Path(os.path.expanduser(str(home)))
        self.key_path = self.home / "trial.key"
        self.data_path = self.home / "trial_accounts.json"

    # ── key material ─────────────────────────────────────────────────────────
    def _master(self) -> bytes:
        if self.key_path.exists():
            data = self.key_path.read_bytes()
            if len(data) == 32:
                return data
        self.home.mkdir(parents=True, exist_ok=True)
        master = secrets.token_bytes(32)
        fd = os.open(str(self.key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(master)
        return master

    # ── persistence ──────────────────────────────────────────────────────────
    def _read_all(self) -> dict[str, dict]:
        if not self.data_path.exists():
            return {}
        try:
            raw = json.loads(self.data_path.read_text("utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:  # noqa: BLE001 - corrupt file → start fresh, keep old
            self.data_path.rename(self.data_path.with_suffix(".corrupt"))
            return {}

    def _write_all(self, all: dict[str, dict]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = self.data_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(all, indent=2), "utf-8")
        os.replace(tmp, self.data_path)

    # ── API ──────────────────────────────────────────────────────────────────
    def store(self, platform: str, login: str, secret: str, note: str = "") -> dict:
        platform = platform.strip().lower()
        if not platform or not login or not secret:
            raise ValueError("platform, login and secret are all required")
        entry = {
            "platform": platform,
            "login": login,
            "secret": _seal(self._master(), secret.encode("utf-8")).hex(),
            "note": note,
            "saved_at": time.time(),
        }
        all = self._read_all()
        all[platform] = entry
        self._write_all(all)
        return {"platform": platform, "login": login}

    def get(self, platform: str) -> dict | None:
        platform = platform.strip().lower()
        entry = self._read_all().get(platform)
        if entry is None:
            return None
        try:
            secret = _open(self._master(), bytes.fromhex(entry["secret"])).decode("utf-8")
        except Exception:  # noqa: BLE001 - report, don't crash
            return {**entry, "secret": None, "unreadable": True}
        return {**entry, "secret": secret}

    def list(self) -> list[dict]:
        out = []
        for platform, entry in sorted(self._read_all().items()):
            masked = entry.get("login", "?")
            out.append({
                "platform": platform,
                "login": masked,
                "note": entry.get("note", ""),
                "saved_at": entry.get("saved_at", 0.0),
            })
        return out

    def delete(self, platform: str) -> bool:
        platform = platform.strip().lower()
        all = self._read_all()
        if platform not in all:
            return False
        del all[platform]
        self._write_all(all)
        return True

    @staticmethod
    def mask(secret: str) -> str:
        """Display form for chat: first/last char only (or dots when short)."""
        if len(secret) <= 4:
            return "••••"
        return f"{secret[0]}…{secret[-1]} ({len(secret)} chars)"
