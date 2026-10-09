"""Disposable persona bank for *authorized* signups.

A persona is a consistent, clearly-disposable identity draft used for ONE
service's trial signup: the same persona supplies the name on the form,
the username, and the (temp/disposable) email address, so a signup never
mixes identities halfway through.

Boundaries (the owner's standing policy, enforced here and in the
signup driver):

* personas are **disposable drafts only** — generated names use an
  obviously-disposable surname pool (Trial/Demo/Test/…) and every
  persona carries ``disposable=True``. They are never the owner's real
  identity and never presented as such.
* a persona is **never used without explicit owner confirmation** —
  :class:`ConfirmationGate` is one-shot and defaults to "no". The chat
  flow shows the drafted persona with a warning and proceeds only on
  ``--yes`` or ``/trial confirm <token>``.
* personas exist for **one account per service** and for authorized
  trial signups only — never bulk creation, never fake-identity abuse.

Personas are stored vault-side (``CredentialVault``, service
``identity_bank``) so they survive restarts and stay encrypted at rest
next to the credentials they belong to. When no vault is available (unit
tests) a plain ``identity_bank_personas`` table in the app DB is used
instead.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.errors import NoMoralsError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database
from .creator import generate_password
from .vault import CredentialVault

__all__ = [
    "Persona",
    "IdentityBank",
    "ConfirmationGate",
    "ConfirmationRequired",
    "NIGERIAN_FIRST_NAMES",
    "INTERNATIONAL_FIRST_NAMES",
    "DISPOSABLE_SURNAMES",
    "render_persona_card",
]

_log = get_logger(__name__)

#: Nigerian first-name pool — the owner is Nigerian; trial personas draw
#: from here first, then the international pool.
NIGERIAN_FIRST_NAMES = (
    "Chinedu", "Adebayo", "Ngozi", "Adaeze", "Oluwaseun", "Funke",
    "Emeka", "Chiamaka", "Tunde", "Yetunde", "Ibrahim", "Fatima",
    "Nnamdi", "Blessing", "Kelechi", "Halima", "Segun", "Amara",
    "Olumide", "Zainab", "Ifeanyi", "Aisha", "Damilola", "Ngozika",
)

#: International first-name pool (fallback variety).
INTERNATIONAL_FIRST_NAMES = (
    "Alex", "Sam", "Jordan", "Casey", "Riley", "Morgan",
    "Priya", "Diego", "Mei", "Lucas", "Sofia", "Yuki",
)

#: Surnames that mark a persona as obviously disposable — never a real
#: person's name shape. This is deliberate: the standing policy forbids
#: fake-identity creation, so generated personas stay visibly synthetic.
DISPOSABLE_SURNAMES = (
    "Trial", "Demo", "Test", "Temp", "Guest",
)

#: vault service under which personas are stored.
VAULT_SERVICE = "identity_bank"

#: persona reuse window — a persona minted for a service is reused for
#: that service's retries inside this window instead of minting anew.
REUSE_WINDOW_SECONDS = 7 * 24 * 3600


class ConfirmationRequired(NoMoralsError):
    """Raised when a signup is attempted without explicit owner confirmation."""


@dataclass
class Persona:
    """One consistent disposable persona for a service's signup."""

    id: str
    service: str
    first_name: str
    last_name: str
    username_variants: list[str] = field(default_factory=list)
    dob: str = ""  # ISO date, always an adult (21–45y)
    email: str = ""  # filled at drive time from a disposable address
    email_provider: str = ""
    password: str = ""
    disposable: bool = True
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "service": self.service,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "username_variants": list(self.username_variants),
            "dob": self.dob,
            "email": self.email,
            "email_provider": self.email_provider,
            "password": self.password,
            "disposable": self.disposable,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        return cls(
            id=str(data.get("id", "")),
            service=str(data.get("service", "")),
            first_name=str(data.get("first_name", "")),
            last_name=str(data.get("last_name", "")),
            username_variants=list(data.get("username_variants") or []),
            dob=str(data.get("dob", "")),
            email=str(data.get("email", "")),
            email_provider=str(data.get("email_provider", "")),
            password=str(data.get("password", "")),
            disposable=bool(data.get("disposable", True)),
            created_at=float(data.get("created_at", 0) or 0),
            metadata=dict(data.get("metadata") or {}),
        )


def _username_variants(first: str, last: str, rng: random.Random) -> list[str]:
    """Deterministic-per-persona username candidates, most-likely first."""
    import re

    f = re.sub(r"[^a-z]", "", first.lower()) or "user"
    l = re.sub(r"[^a-z]", "", last.lower()) or "trial"
    suffix = f"{rng.randint(1000, 9999)}"
    cands = [
        f"{f}.{l}{suffix}",
        f"{f}_{l}{suffix}",
        f"{f}{l}{suffix}",
        f"{f[0]}{l}{suffix}",
        f"{f}.{l}",
    ]
    # de-dupe, keep order
    seen: set[str] = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _adult_dob(rng: random.Random) -> str:
    """Random adult date of birth (21–45 years ago), ISO format."""
    import datetime

    today = datetime.date.today()
    years = rng.randint(21, 45)
    try:
        dob = today.replace(year=today.year - years)
    except ValueError:  # Feb 29 edge
        dob = today.replace(year=today.year - years, day=28)
    return dob.isoformat()


class IdentityBank:
    """Mints, stores (vault-side), and reuses disposable signup personas."""

    def __init__(
        self,
        db: Database | None = None,
        vault: CredentialVault | None = None,
    ) -> None:
        self.db = db
        self.vault = vault
        if db is not None and vault is None:
            self._ensure_table()

    # ── persistence ───────────────────────────────────────────────

    def _ensure_table(self) -> None:
        assert self.db is not None
        with self.db.transaction():
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS identity_bank_personas (
                    id TEXT PRIMARY KEY,
                    service TEXT NOT NULL DEFAULT '',
                    data TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL DEFAULT 0
                )
                """
            )
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_identity_bank_service "
                "ON identity_bank_personas(service)"
            )

    def _save(self, persona: Persona) -> None:
        if self.vault is not None:
            self.vault.store(
                service=VAULT_SERVICE,
                username=persona.id,
                password="",
                credential_type="persona",
                tags=["persona", persona.service, "disposable"],
                metadata=persona.to_dict(),
            )
            return
        if self.db is None:
            return
        with self.db.transaction():
            self.db.execute(
                "INSERT OR REPLACE INTO identity_bank_personas "
                "(id, service, data, updated_at) VALUES (?, ?, ?, ?)",
                (persona.id, persona.service, json.dumps(persona.to_dict()),
                 time.time()),
            )

    def _load_all(self, service: str) -> list[Persona]:
        service = service.strip().lower()
        out: list[Persona] = []
        if self.vault is not None:
            for cred in self.vault.list_all(service=VAULT_SERVICE,
                                            active_only=True):
                meta = cred.metadata or {}
                if str(meta.get("service", "")).lower() != service:
                    continue
                try:
                    out.append(Persona.from_dict(meta))
                except Exception:  # noqa: BLE001 - skip corrupt rows
                    continue
            return sorted(out, key=lambda p: p.created_at)
        if self.db is None:
            return out
        rows = self.db.query(
            "SELECT data FROM identity_bank_personas WHERE service = ? "
            "ORDER BY updated_at",
            (service,),
        )
        for row in rows or []:
            try:
                out.append(Persona.from_dict(json.loads(row["data"] or "{}")))
            except Exception:  # noqa: BLE001 - skip corrupt rows
                continue
        return out

    # ── minting ───────────────────────────────────────────────────

    def mint(self, service: str, *,
             rng: random.Random | None = None) -> Persona:
        """Mint a fresh disposable persona for ``service`` and store it.

        The persona is a *draft* — it must pass the confirmation gate
        before any signup uses it.
        """
        rng = rng or random.Random()
        service = service.strip().lower()
        first_pool = NIGERIAN_FIRST_NAMES + INTERNATIONAL_FIRST_NAMES
        first = rng.choice(first_pool)
        last = rng.choice(DISPOSABLE_SURNAMES)
        persona = Persona(
            id=new_id("persona"),
            service=service,
            first_name=first,
            last_name=last,
            username_variants=_username_variants(first, last, rng),
            dob=_adult_dob(rng),
            password=generate_password(20),
            disposable=True,
            created_at=time.time(),
            metadata={"pool": "nigerian" if first in NIGERIAN_FIRST_NAMES
                      else "international"},
        )
        self._save(persona)
        _log.info("minted disposable persona %s for %s", persona.id, service)
        return persona

    def get_or_mint(self, service: str, *,
                    rng: random.Random | None = None) -> Persona:
        """Reuse the service's current persona, or mint one.

        The same persona is reused across a signup's retries (same name
        on the form, same username attempts, same temp email) inside the
        reuse window — a new persona is minted only when none exists or
        the stored one is stale.
        """
        service = service.strip().lower()
        now = time.time()
        for persona in reversed(self._load_all(service)):
            if now - persona.created_at < REUSE_WINDOW_SECONDS:
                return persona
        return self.mint(service, rng=rng)

    def get(self, persona_id: str) -> Persona | None:
        """Fetch a persona by id (vault or table backend)."""
        if self.vault is not None:
            for cred in self.vault.list_all(service=VAULT_SERVICE,
                                            active_only=True):
                if cred.username == persona_id:
                    try:
                        return Persona.from_dict(cred.metadata or {})
                    except Exception:  # noqa: BLE001
                        return None
            return None
        if self.db is None:
            return None
        row = self.db.query_one(
            "SELECT data FROM identity_bank_personas WHERE id = ?",
            (persona_id,),
        )
        if not row:
            return None
        try:
            return Persona.from_dict(json.loads(row["data"] or "{}"))
        except Exception:  # noqa: BLE001
            return None

    def list(self, service: str | None = None) -> list[Persona]:
        """All stored personas, optionally filtered by service."""
        if service:
            return self._load_all(service)
        # unfiltered: scan everything (small table by design)
        if self.vault is not None:
            out = []
            for cred in self.vault.list_all(service=VAULT_SERVICE,
                                            active_only=True):
                try:
                    out.append(Persona.from_dict(cred.metadata or {}))
                except Exception:  # noqa: BLE001
                    continue
            return sorted(out, key=lambda p: p.created_at)
        if self.db is None:
            return []
        rows = self.db.query(
            "SELECT data FROM identity_bank_personas ORDER BY updated_at")
        out = []
        for row in rows or []:
            try:
                out.append(Persona.from_dict(json.loads(row["data"] or "{}")))
            except Exception:  # noqa: BLE001
                continue
        return out


def render_persona_card(persona: Persona, service: str) -> str:
    """The draft-identity warning card shown before confirmation."""
    lines = [
        "⚠️  SIGNUP IDENTITY DRAFT — confirm before I use it",
        "",
        f"service:  {service}",
        f"name:     {persona.name}",
        f"username: {persona.username_variants[0] if persona.username_variants else '?'}",
        f"dob:      {persona.dob} (adult)",
        f"email:    disposable temp address (minted at signup time, "
        "never your real email)",
        "",
        "This is a DISPOSABLE draft identity for a trial signup — not your",
        "real details. I will not proceed without your explicit confirmation.",
        "One account per service. Credentials are stored in your vault.",
    ]
    return "\n".join(lines)


class ConfirmationGate:
    """One-shot explicit confirmation for persona use.

    ``request()`` registers a pending confirmation and returns a token.
    ``confirm(token, reply)`` returns True only for an explicit yes —
    anything else (including empty/ambiguous replies) is False and the
    token stays pending. ``consume(token)`` one-shot-redeems a token
    that was explicitly confirmed. The gate never defaults to yes.
    """

    #: replies that count as explicit confirmation (case-insensitive,
    #: stripped). Anything else is not confirmation.
    YES_REPLIES = frozenset({"yes", "y", "confirm", "confirmed",
                             "proceed", "go ahead", "do it", "ok"})

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._pending: dict[str, dict[str, Any]] = {}

    def request(self, *, subject: str, card: str,
                payload: dict[str, Any] | None = None) -> str:
        """Register a pending confirmation. Returns the token."""
        token = new_id("confirm")
        self._pending[token] = {
            "subject": subject,
            "card": card,
            "payload": dict(payload or {}),
            "expires_at": time.time() + self.ttl_seconds,
            "confirmed": False,
        }
        self._prune()
        return token

    def confirm(self, token: str, reply: str) -> bool:
        """Record an explicit yes. Returns True only on explicit yes."""
        entry = self._pending.get(token)
        if entry is None or time.time() > entry["expires_at"]:
            self._pending.pop(token, None)
            return False
        if (reply or "").strip().lower() in self.YES_REPLIES:
            entry["confirmed"] = True
            return True
        return False

    def consume(self, token: str) -> dict[str, Any] | None:
        """One-shot redeem: returns the payload iff explicitly confirmed,
        then invalidates the token. Never returns a payload twice."""
        entry = self._pending.pop(token, None)
        if entry is None or time.time() > entry["expires_at"]:
            return None
        if not entry.get("confirmed"):
            return None
        return dict(entry.get("payload") or {})

    def is_pending(self, token: str) -> bool:
        entry = self._pending.get(token)
        return bool(entry and time.time() <= entry["expires_at"]
                    and not entry.get("confirmed"))

    def _prune(self) -> None:
        now = time.time()
        for token in [t for t, e in self._pending.items()
                      if now > e["expires_at"]]:
            del self._pending[token]
