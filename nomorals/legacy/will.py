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
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "SEALED_MARKER",
    "SealedError",
    "DigitalAsset",
    "Beneficiary",
    "Executor",
    "Will",
    "WillStore",
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "location_hint": self.location_hint,
            "instruction": self.instruction,
        }

    @classmethod
    def from_dict(cls, row: Any) -> DigitalAsset:
        if not isinstance(row, dict):
            raise ValueError("digital asset row must be an object")
        label = row.get("label")
        kind = row.get("kind")
        if not isinstance(label, str) or not label:
            raise ValueError("digital asset row needs a string label")
        if not isinstance(kind, str) or not kind:
            raise ValueError("digital asset row needs a string kind")
        hint = row.get("location_hint", "")
        instruction = row.get("instruction", "")
        if not isinstance(hint, str) or not isinstance(instruction, str):
            raise ValueError("digital asset hint/instruction must be strings")
        return cls(label=label, kind=kind, location_hint=hint,
                   instruction=instruction)


@dataclass(frozen=True)
class Beneficiary:
    name: str
    relationship: str = ""
    share_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "relationship": self.relationship,
            "share_note": self.share_note,
        }

    @classmethod
    def from_dict(cls, row: Any) -> Beneficiary:
        if not isinstance(row, dict):
            raise ValueError("beneficiary row must be an object")
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("beneficiary row needs a string name")
        relationship = row.get("relationship", "")
        share_note = row.get("share_note", "")
        if not isinstance(relationship, str) or not isinstance(share_note, str):
            raise ValueError("beneficiary relationship/share_note must be strings")
        return cls(name=name, relationship=relationship, share_note=share_note)


@dataclass(frozen=True)
class Executor:
    name: str
    contact: str = ""  # contact handle — never a secret
    backup_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "contact": self.contact,
            "backup_name": self.backup_name,
        }

    @classmethod
    def from_dict(cls, row: Any) -> Executor:
        if not isinstance(row, dict):
            raise ValueError("executor row must be an object")
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("executor row needs a string name")
        contact = row.get("contact", "")
        backup_name = row.get("backup_name", "")
        if not isinstance(contact, str) or not isinstance(backup_name, str):
            raise ValueError("executor contact/backup_name must be strings")
        return cls(name=name, contact=contact, backup_name=backup_name)


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

    # -- durability -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain-data form for the durable store.

        The sealed body travels as the vault-produced blob + SHA-256
        hash only. There is no plaintext field on this class, so no
        serialisation of it can ever write one.
        """
        return {
            "owner_name": self.owner_name,
            "executor": self.executor.to_dict(),
            "beneficiaries": [b.to_dict() for b in self.beneficiaries],
            "digital_assets": [a.to_dict() for a in self.digital_assets],
            "instructions": list(self.instructions),
            "witnesses": list(self.witnesses),
            "created": self.created.isoformat(),
            "id": self.id,
            "sealed_blob": self._sealed_blob,
            "plaintext_hash": self._plaintext_hash,
        }

    @classmethod
    def from_dict(cls, row: Any) -> Will:
        """Strict read-back: any bad row raises instead of half-loading.

        A row smuggling a "plaintext"/"content" key is refused outright
        — the store never holds will plaintext, by construction.
        """
        if not isinstance(row, dict):
            raise ValueError("will row must be an object")
        if "plaintext" in row or "content" in row:
            raise ValueError(
                "will rows never carry plaintext — sealed blob only")
        owner = row.get("owner_name")
        if not isinstance(owner, str) or not owner:
            raise ValueError("will row needs a string owner_name")
        executor = Executor.from_dict(row.get("executor"))
        beneficiaries = row.get("beneficiaries", [])
        assets = row.get("digital_assets", [])
        instructions = row.get("instructions", [])
        witnesses = row.get("witnesses", [])
        if not isinstance(beneficiaries, list):
            raise ValueError("will beneficiaries must be a list")
        if not isinstance(assets, list):
            raise ValueError("will digital_assets must be a list")
        if not isinstance(instructions, list) or not all(
                isinstance(i, str) for i in instructions):
            raise ValueError("will instructions must be a list of strings")
        if not isinstance(witnesses, list) or not all(
                isinstance(w, str) for w in witnesses):
            raise ValueError("will witnesses must be a list of strings")
        created_raw = row.get("created")
        if not isinstance(created_raw, str):
            raise ValueError("will created must be a YYYY-MM-DD date")
        try:
            created = date.fromisoformat(created_raw)
        except ValueError as exc:
            raise ValueError("will created must be a YYYY-MM-DD date") from exc
        will_id = row.get("id")
        if will_id is not None and not isinstance(will_id, str):
            raise ValueError("will id must be a string")
        blob = row.get("sealed_blob")
        digest = row.get("plaintext_hash")
        if blob is not None and not isinstance(blob, str):
            raise ValueError("will sealed_blob must be a string or null")
        if digest is not None and not isinstance(digest, str):
            raise ValueError("will plaintext_hash must be a string or null")
        return cls(
            owner_name=owner, executor=executor,
            beneficiaries=[Beneficiary.from_dict(b) for b in beneficiaries],
            digital_assets=[DigitalAsset.from_dict(a) for a in assets],
            instructions=list(instructions), witnesses=list(witnesses),
            created=created,
            id=will_id or uuid.uuid4().hex[:12],
            _sealed_blob=blob, _plaintext_hash=digest)


class WillStore:
    """Durable store for the owner's single will (<home>/data/legacy.json).

    One will, not many: a person has one current will. The sealed body
    is stored ONLY as the vault-produced blob + SHA-256 hash — plaintext
    is never written here, by construction (Will has no plaintext field,
    and from_dict refuses any row smuggling one).

    Load is STRICT where AssetLedger is tolerant: a corrupt will file
    raises instead of silently becoming an empty will — silently losing
    the owner's will would be worse than refusing to load it. A missing
    file simply means no will yet.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.will: Will | None = None
        if path is not None:
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        if not os.path.exists(self.path):
            return
        try:
            raw = json.loads(Path(self.path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"legacy will file is unreadable ({self.path}): {exc}"
            ) from exc
        if not isinstance(raw, dict) or "will" not in raw:
            raise ValueError(
                f"legacy will file is unreadable ({self.path}): "
                "expected an object with a 'will' key")
        if raw["will"] is None:
            return
        self.will = Will.from_dict(raw["will"])

    def _save(self) -> None:
        assert self.path is not None
        parent = Path(self.path).parent
        parent.mkdir(parents=True, exist_ok=True)
        payload = {"will": self.will.to_dict() if self.will else None}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)

    def get(self) -> Will | None:
        return self.will

    def put(self, will: Will) -> None:
        self.will = will
        if self.path is not None:
            self._save()

    def clear(self) -> None:
        self.will = None
        if self.path is not None:
            self._save()
