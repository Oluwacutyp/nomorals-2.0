"""Legacy / digital will — sealed, owner-only.

MORE_TO_ADD gap 8. NEW module; no existing organ covers wills/legacy.

Security model (read this before touching the code):

* OWNER-ONLY. Every public method requires ``is_owner=True``. Anything
  else raises :class:`PermissionError`. There is no backdoor, no admin
  override, no "break glass".
* SEALED, NEVER PLAINTEXT. Sealed content is stored ONLY as a blob
  produced by an owner-injected ``seal_fn`` (e.g. the owner's real
  encryption). This module performs NO cryptography itself and NEVER
  fakes decryption: there is no default cipher, no reversible "encoding",
  no base64-as-encryption. Sealing without an injected ``seal_fn``
  raises — the module would rather refuse than store a fake seal.
* REVEAL via injected ``unlock_fn`` only. :meth:`Will.reveal` calls the
  caller-supplied ``unlock_fn(blob)`` and verifies the SHA-256 hash of
  the result against the hash taken at seal time. A wrong key / wrong
  function fails loudly instead of returning garbage.
* Sealed content is NEVER returned by any other accessor. ``summary()``
  returns metadata only (executor name, counts, dates); the sealed body
  surfaces as the literal :data:`SEALED_MARKER`.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

__all__ = [
    "SEALED_MARKER",
    "SealedError",
    "DigitalAsset",
    "Beneficiary",
    "Executor",
    "Will",
]

#: What any non-reveal accessor returns in place of sealed content.
SEALED_MARKER = "🔒 SEALED — owner key required"


class SealedError(RuntimeError):
    """Raised when sealed content cannot be produced without the owner key."""


def _require_owner(is_owner: bool) -> None:
    if not is_owner:
        raise PermissionError("legacy/will is owner-only")


@dataclass(frozen=True)
class DigitalAsset:
    """A digital asset named in the will (account, wallet, vault, file)."""

    label: str
    kind: str  # e.g. "account" / "wallet" / "vault" / "file" / "domain"
    location_hint: str = ""  # where to find it — NEVER a secret itself
    instruction: str = ""  # e.g. "transfer to X", "delete", "memorialise"


@dataclass(frozen=True)
class Beneficiary:
    name: str
    relationship: str = ""
    share_note: str = ""


@dataclass(frozen=True)
class Executor:
    name: str
    contact: str = ""  # contact handle — never a secret
    backup_name: str = ""


@dataclass
class Will:
    """One will. The sealed body is opaque to this module by design."""

    owner_name: str
    executor: Executor
    beneficiaries: list[Beneficiary] = field(default_factory=list)
    digital_assets: list[DigitalAsset] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    witnesses: list[str] = field(default_factory=list)
    created: date = field(default_factory=date.today)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _sealed_blob: str | None = field(default=None, repr=False)
    _plaintext_hash: str | None = field(default=None, repr=False)

    # -- sealing ---------------------------------------------------------
    def seal_content(self, plaintext: str, seal_fn: Callable[[str], str] | None, *, is_owner: bool) -> None:
        """Seal the will body with an OWNER-INJECTED ``seal_fn``.

        ``seal_fn(plaintext) -> blob`` is the owner's real encryption
        (or vault seal). This module never sees the key and never
        invents a cipher: ``seal_fn=None`` raises :class:`ValueError`
        instead of storing plaintext or a fake seal. Only the blob and
        a SHA-256 hash of the plaintext are retained.
        """
        _require_owner(is_owner)
        if seal_fn is None:
            raise ValueError("seal_fn is required — will content is never stored as plaintext or fake-encrypted")
        if not plaintext:
            raise ValueError("cannot seal empty content")
        self._sealed_blob = seal_fn(plaintext)
        self._plaintext_hash = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()

    def reveal(self, unlock_fn: Callable[[str], str] | None, *, is_owner: bool) -> str:
        """Return the sealed body via an OWNER-INJECTED ``unlock_fn``.

        ``unlock_fn(blob) -> plaintext`` is the owner's real decryption.
        The result is hash-verified against the seal-time hash; a
        mismatch raises :class:`SealedError` (wrong key, wrong function,
        tampered blob). Without a sealed blob or without ``unlock_fn``,
        raises :class:`SealedError` — never returns plaintext, never
        guesses.
        """
        _require_owner(is_owner)
        if self._sealed_blob is None:
            raise SealedError("no sealed content stored on this will")
        if unlock_fn is None:
            raise SealedError("unlock_fn (owner key) is required to reveal sealed content")
        plaintext = unlock_fn(self._sealed_blob)
        digest = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
        if digest != self._plaintext_hash:
            raise SealedError("unlock produced content that does not match the seal-time hash — wrong key or tampered blob")
        return plaintext

    # -- metadata-only access ---------------------------------------------
    def sealed_preview(self, *, is_owner: bool) -> str:
        """What non-reveal accessors return for the sealed body: the marker."""
        _require_owner(is_owner)
        return SEALED_MARKER

    def summary(self, *, is_owner: bool) -> dict[str, Any]:
        """Metadata only — executor, counts, dates. Sealed body = marker."""
        _require_owner(is_owner)
        return {
            "id": self.id,
            "owner_name": self.owner_name,
            "executor": self.executor.name,
            "backup_executor": self.executor.backup_name,
            "beneficiaries": len(self.beneficiaries),
            "digital_assets": len(self.digital_assets),
            "instructions": len(self.instructions),
            "witnesses": len(self.witnesses),
            "created": self.created.isoformat(),
            "has_sealed_content": self._sealed_blob is not None,
            "sealed_content": SEALED_MARKER if self._sealed_blob is not None else None,
        }

    def digital_asset_list(self, *, is_owner: bool) -> list[DigitalAsset]:
        """Digital-asset roster (labels/hints/instructions — never secrets)."""
        _require_owner(is_owner)
        return list(self.digital_assets)
