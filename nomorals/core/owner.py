"""Owner identity seal — who the owner is, proved in code, not env.

Two-part unlock, both checked here:

  1. **identity claim** — one of :data:`OWNER_IDENTITIES`, ingrained in this
     file.  The names are public handles, so a claim alone proves nothing.
  2. **passphrase proof** — verified against :data:`PASSPHRASE_SEAL`, a
     salted SHA-256 hash baked into this file by ``nm owner seal``.
     The passphrase itself never appears in code, logs, or audit records.

Why in code and not env: the owner asked for the unlock to live with the
program, not beside it.  A salted hash in a (public) repo reveals nothing —
an attacker learns the salt and the digest, which buys them nothing without
the passphrase.  The names stay as the identity half on purpose: unlocking
requires *being* the owner *and* proving it.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

__all__ = [
    "OWNER_IDENTITIES",
    "PASSPHRASE_SEAL",
    "MIN_PASSPHRASE_LEN",
    "is_owner_identity",
    "seal_configured",
    "verify_passphrase",
    "verify_owner",
    "make_seal",
    "bake_seal",
]

#: The owner's ingrained identities.  Public handles — the *claim* half of
#: the unlock, never the proof half.
OWNER_IDENTITIES: tuple[str, ...] = ("oluwacutyp", "peace", "peacethefirt1")

#: Salted passphrase seal, format ``v1$<salt_hex>$<hash_hex>``.
#: Written by ``nm owner seal`` — never hand-edit, never commit a plaintext
#: passphrase here.
PASSPHRASE_SEAL = ""

#: Minimum passphrase length enforced at seal time.
MIN_PASSPHRASE_LEN = 20


def is_owner_identity(name: str) -> bool:
    """True when ``name`` claims one of the ingrained owner identities."""
    claimed = (name or "").strip().lower()
    return bool(claimed) and claimed in OWNER_IDENTITIES


def seal_configured() -> bool:
    """True when a passphrase seal has been baked into this file."""
    return bool(PASSPHRASE_SEAL) and PASSPHRASE_SEAL.startswith("v1$")


def _check_format(seal: str) -> tuple[str, str] | None:
    try:
        version, salt_hex, hash_hex = (seal or "").split("$")
    except ValueError:
        return None
    if version != "v1" or not salt_hex or not hash_hex:
        return None
    return salt_hex, hash_hex


def verify_passphrase(secret: str) -> bool:
    """True when ``secret`` matches the baked seal.  Constant-time."""
    parsed = _check_format(PASSPHRASE_SEAL)
    if not parsed or not secret:
        return False
    salt_hex, hash_hex = parsed
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    digest = hashlib.sha256(salt + secret.encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, hash_hex)


def verify_owner(identity: str, secret: str) -> bool:
    """Full unlock check: known identity claim AND valid passphrase proof."""
    return is_owner_identity(identity) and verify_passphrase(secret)


def make_seal(secret: str) -> str:
    """Build a ``v1$`` seal string from a passphrase.

    Raises :class:`ValueError` when the passphrase is too weak to serve as
    the proof half.  The secret itself is never returned or stored.
    """
    if not secret or len(secret) < MIN_PASSPHRASE_LEN:
        raise ValueError(
            f"passphrase must be at least {MIN_PASSPHRASE_LEN} characters"
        )
    salt = secrets.token_bytes(16)
    digest = hashlib.sha256(salt + secret.encode("utf-8")).hexdigest()
    return f"v1${salt.hex()}${digest}"


_SEAL_RE = re.compile(r'^PASSPHRASE_SEAL = ""$', re.MULTILINE)
_SEAL_SET_RE = re.compile(r'^PASSPHRASE_SEAL = ".*"$', re.MULTILINE)


def bake_seal(seal: str, path: str | None = None) -> str:
    """Write ``seal`` into this module's ``PASSPHRASE_SEAL`` constant.

    Returns the file path written.  Used by ``nm owner seal``; not called
    at runtime.
    """
    import pathlib

    if not _check_format(seal):
        raise ValueError("not a valid v1 seal string")
    target = pathlib.Path(path) if path else pathlib.Path(__file__)
    src = target.read_text(encoding="utf-8")
    replacement = f'PASSPHRASE_SEAL = "{seal}"'
    if _SEAL_RE.search(src):
        src = _SEAL_RE.sub(replacement, src, count=1)
    elif _SEAL_SET_RE.search(src):
        src = _SEAL_SET_RE.sub(replacement, src, count=1)
    else:
        raise ValueError("PASSPHRASE_SEAL constant not found — file changed?")
    target.write_text(src, encoding="utf-8")
    return str(target)
