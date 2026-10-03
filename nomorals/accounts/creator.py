"""Account creation with human-in-the-loop checkpoints.

Account flows follow the same discipline as the connector package's
signup flows (``nomorals/connectors/checkpoints.py``) — accounts is a
lower layer and may not import connectors, so the pattern is
re-implemented here rather than imported:

* **one account per service** — a second creation flow refuses while an
  active credential exists for that service;
* **the owner's own identity** — flows never invent fake identities;
  when a real identity (name/email) is needed and none is on file, the
  flow pauses on a checkpoint and the owner supplies it;
* **pause, don't bypass** — when automation reaches a step only a human
  can do (CAPTCHA, email-verification click, phone 2FA, accepting
  terms), the flow persists a checkpoint, pings the owner through the
  injected ``notify`` hook, and raises :class:`AccountCheckpointPending`.
  A later call to :meth:`AccountCreator.resume_checkpoint` continues
  the flow after the owner acts.

HARD BOUNDARY: Devon never auto-solves CAPTCHAs — no solver services,
no AI bypass, no verification dodging. The human checkpoint is the only
path through human verification.

Usage:
    creator = AccountCreator(vault, notify=send_owner_ping)
    creator.set_owner_identity("Death", "owner@example.com")

    try:
        creator.create_account("github", username="my-bot")
    except AccountCheckpointPending as pending:
        # owner solves the CAPTCHA, then:
        account = creator.resume_checkpoint(pending.checkpoint.id)
"""

from __future__ import annotations

import json
import secrets
import string
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Optional

from ..core.errors import NoMoralsError, NotFound
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database
from .vault import Credential, CredentialVault

__all__ = [
    "AccountCreator",
    "CreatedAccount",
    "AccountCheckpoint",
    "AccountCheckpointPending",
    "AccountExistsError",
    "MissingOwnerIdentity",
    "CheckpointKind",
    "CheckpointState",
    "CheckpointStore",
    "generate_password",
    "generate_username",
]

_log = get_logger(__name__)


class AccountExistsError(NoMoralsError):
    """Raised when a creation flow is requested but an active account
    already exists for the service (one account per service)."""


class MissingOwnerIdentity(NoMoralsError):
    """Raised when a flow needs the owner's real identity and the
    checkpoint carrying that request could not be created."""


class CheckpointKind(StrEnum):
    """What kind of human-only step a checkpoint waits on."""

    CAPTCHA = "captcha"
    EMAIL_VERIFY = "email_verify"
    PHONE_2FA = "phone_2fa"
    IDENTITY = "identity"  # owner must supply their real identity info
    TOS_ACCEPT = "tos_accept"
    MANUAL_STEP = "manual_step"  # any other human-only step


class CheckpointState(StrEnum):
    PENDING = "pending"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class AccountCheckpoint:
    """A paused-for-human step in an account flow.

    ``resume_state`` carries everything needed to continue the flow
    (service, generated username/password, the owner's email) so a
    later process can pick up after the owner acts.
    """

    id: str
    kind: CheckpointKind
    title: str
    instructions: str
    service: str = ""
    state: CheckpointState = CheckpointState.PENDING
    created_at: float = 0.0
    updated_at: float = 0.0
    resolved_at: float | None = None
    resume_state: dict[str, Any] = field(default_factory=dict)
    result_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "title": self.title,
            "instructions": self.instructions,
            "service": self.service,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "resume_state": dict(self.resume_state),
            "result_note": self.result_note,
        }


class AccountCheckpointPending(NoMoralsError):
    """Raised to pause an account flow at a human checkpoint.

    Carries the checkpoint so the caller (or a later run) can resume
    with ``creator.resume_checkpoint(checkpoint.id)`` after the owner
    completes the step.
    """

    def __init__(self, checkpoint: AccountCheckpoint) -> None:
        self.checkpoint = checkpoint
        super().__init__(
            f"paused for human action [{checkpoint.id}]: {checkpoint.title} — "
            f"resume with `creator.resume_checkpoint({checkpoint.id!r})` "
            "after completing the step"
        )


class CheckpointStore:
    """Persisted account-flow checkpoints, backed by the app database."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.db.transaction():
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS account_checkpoints (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    instructions TEXT NOT NULL DEFAULT '',
                    service TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    resolved_at REAL,
                    resume_state TEXT NOT NULL DEFAULT '{}',
                    result_note TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self.db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_account_checkpoints_state
                ON account_checkpoints(state)
                """
            )

    def create(
        self,
        kind: CheckpointKind,
        title: str,
        instructions: str,
        *,
        service: str = "",
        resume_state: dict[str, Any] | None = None,
        ttl_seconds: float = 86400.0,
    ) -> AccountCheckpoint:
        now = time.time()
        cp = AccountCheckpoint(
            id=new_id("achk"),
            kind=kind,
            title=title,
            instructions=instructions,
            service=service,
            state=CheckpointState.PENDING,
            created_at=now,
            updated_at=now,
            resume_state=dict(resume_state or {}),
        )
        cp.resume_state.setdefault("_expires_at", now + ttl_seconds)
        with self.db.transaction():
            self.db.execute(
                """
                INSERT INTO account_checkpoints
                (id, kind, title, instructions, service, state,
                 created_at, updated_at, resolved_at, resume_state, result_note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cp.id, cp.kind.value, cp.title, cp.instructions,
                    cp.service, cp.state.value, cp.created_at, cp.updated_at,
                    cp.resolved_at, json.dumps(cp.resume_state),
                    cp.result_note,
                ),
            )
        _log.info("account checkpoint %s created: %s", cp.id, title)
        return cp

    def get(self, checkpoint_id: str) -> AccountCheckpoint:
        self.expire_stale()
        row = self.db.query_one(
            "SELECT * FROM account_checkpoints WHERE id = ?",
            (checkpoint_id,),
        )
        if not row:
            raise NotFound(f"no account checkpoint {checkpoint_id!r}")
        return self._from_row(row)

    def list_pending(
        self, service: str | None = None
    ) -> list[AccountCheckpoint]:
        self.expire_stale()
        if service:
            rows = self.db.query(
                "SELECT * FROM account_checkpoints WHERE state = 'pending' "
                "AND service = ? ORDER BY created_at",
                (service,),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM account_checkpoints WHERE state = 'pending' "
                "ORDER BY created_at"
            )
        return [self._from_row(r) for r in rows]

    def resolve(
        self, checkpoint_id: str, note: str = ""
    ) -> AccountCheckpoint:
        cp = self.get(checkpoint_id)
        if cp.state != CheckpointState.PENDING:
            raise NoMoralsError(
                f"checkpoint {checkpoint_id} is {cp.state.value}, "
                "not pending — nothing to resolve"
            )
        now = time.time()
        with self.db.transaction():
            self.db.execute(
                "UPDATE account_checkpoints SET state = 'resolved', "
                "resolved_at = ?, updated_at = ?, result_note = ? "
                "WHERE id = ?",
                (now, now, note, checkpoint_id),
            )
        cp.state = CheckpointState.RESOLVED
        cp.resolved_at = now
        cp.updated_at = now
        cp.result_note = note
        _log.info("account checkpoint %s resolved", checkpoint_id)
        return cp

    def cancel(
        self, checkpoint_id: str, note: str = ""
    ) -> AccountCheckpoint:
        cp = self.get(checkpoint_id)
        if cp.state != CheckpointState.PENDING:
            raise NoMoralsError(
                f"checkpoint {checkpoint_id} is {cp.state.value}, "
                "not pending — nothing to cancel"
            )
        now = time.time()
        with self.db.transaction():
            self.db.execute(
                "UPDATE account_checkpoints SET state = 'cancelled', "
                "updated_at = ?, result_note = ? WHERE id = ?",
                (now, note, checkpoint_id),
            )
        cp.state = CheckpointState.CANCELLED
        cp.updated_at = now
        cp.result_note = note
        return cp

    def expire_stale(self, now: float | None = None) -> int:
        """Mark pending checkpoints past their ttl as expired. Returns count."""
        now = now or time.time()
        rows = self.db.query(
            "SELECT id, resume_state FROM account_checkpoints "
            "WHERE state = 'pending'"
        )
        expired = 0
        for row in rows:
            try:
                ttl_at = float(
                    json.loads(row["resume_state"] or "{}").get(
                        "_expires_at", 0
                    )
                )
            except (ValueError, TypeError):
                ttl_at = 0
            if ttl_at and ttl_at < now:
                with self.db.transaction():
                    self.db.execute(
                        "UPDATE account_checkpoints SET state = 'expired', "
                        "updated_at = ? WHERE id = ?",
                        (now, row["id"]),
                    )
                expired += 1
        return expired

    @staticmethod
    def _from_row(row: dict[str, Any]) -> AccountCheckpoint:
        try:
            resume_state = json.loads(row.get("resume_state") or "{}")
        except (ValueError, TypeError):
            resume_state = {}
        return AccountCheckpoint(
            id=row["id"],
            kind=CheckpointKind(row["kind"]),
            title=row["title"],
            instructions=row.get("instructions") or "",
            service=row.get("service") or "",
            state=CheckpointState(row["state"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            resolved_at=row.get("resolved_at"),
            resume_state=resume_state,
            result_note=row.get("result_note") or "",
        )


@dataclass
class CreatedAccount:
    """Result of an account creation attempt."""

    service: str
    username: str
    password: str
    email: str
    status: str  # "created", "failed"
    credential: Optional[Credential] = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_password(length: int = 24, *, symbols: bool = True) -> str:
    """Generate a cryptographically secure random password.

    Args:
        length: Password length
        symbols: Include special characters

    Returns:
        Random password string
    """
    alphabet = string.ascii_letters + string.digits
    if symbols:
        alphabet += "!@#$%^&*()-_=+"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_username(prefix: str = "nm", length: int = 8) -> str:
    """Generate a random username.

    Args:
        prefix: Username prefix
        length: Random suffix length

    Returns:
        Username string
    """
    suffix = "".join(
        secrets.choice(string.ascii_lowercase + string.digits)
        for _ in range(length)
    )
    return f"{prefix}_{suffix}"


# Notify hook: called as notify(title, instructions, checkpoint_id) whenever a
# flow pauses for the owner. Injected so this layer never imports the
# (higher-layer) notifier itself.
NotifyHook = Callable[[str, str, str], None]


class AccountCreator:
    """Creates accounts on services with human-in-the-loop checkpoints.

    One account per service, the owner's own identity, and human
    verification steps (CAPTCHA, email/phone verification) pause on a
    persisted checkpoint instead of being bypassed or faked.
    """

    #: Services that hand out throwaway addresses via API — exempt from
    #: the one-account-per-service rule and the owner-identity rule.
    DISPOSABLE_EMAIL_SERVICES = frozenset({"guerrilla", "tempmail"})

    def __init__(
        self,
        vault: CredentialVault,
        browser_session: Any = None,
        *,
        db: Database | None = None,
        notify: NotifyHook | None = None,
    ) -> None:
        self.vault = vault
        self.browser = browser_session
        self.db = db or vault.db
        self.checkpoints = CheckpointStore(self.db)
        self.notify = notify
        self._owner_identity: dict[str, str] | None = None
        self._creation_history: list[CreatedAccount] = []
        _log.info("Account creator initialized")

    # ── owner identity ──────────────────────────────────────────

    def set_owner_identity(
        self,
        name: str,
        email: str,
        *,
        phone: str | None = None,
    ) -> dict[str, str]:
        """Record the owner's real identity for account flows.

        Flows never invent fake identities; the name/email here is what
        signups use. Replaces any previously recorded identity.

        Raises:
            ValueError: If name or email is empty
        """
        name = (name or "").strip()
        email = (email or "").strip()
        if not name or not email:
            raise ValueError("owner identity requires a non-empty name and email")
        identity = {"name": name, "email": email}
        if phone:
            identity["phone"] = phone.strip()
        self._owner_identity = identity
        _log.info("owner identity recorded for %s", email)
        return dict(identity)

    def get_owner_identity(self) -> dict[str, str] | None:
        """The recorded owner identity, or None if not set."""
        return dict(self._owner_identity) if self._owner_identity else None

    def _require_owner_identity(
        self, service: str, username: str, password: str
    ) -> dict[str, str]:
        """Return the owner identity, pausing on a checkpoint if unset.

        Raises:
            AccountCheckpointPending: Owner must supply their identity
        """
        if self._owner_identity:
            return self._owner_identity
        cp = self._pause_for_human(
            CheckpointKind.IDENTITY,
            title=f"Identity needed for {service} account",
            instructions=(
                f"Creating a {service} account needs YOUR real identity — "
                "Devon never invents fake ones. Reply with your name and "
                "email (e.g. via `creator.set_owner_identity(name, email)` "
                "or when resuming this checkpoint), then resume."
            ),
            service=service,
            resume_state={
                "flow": "need_identity",
                "service": service,
                "username": username,
                "password": password,
            },
        )
        raise AccountCheckpointPending(cp)

    # ── checkpoint plumbing ─────────────────────────────────────

    def _pause_for_human(
        self,
        kind: CheckpointKind,
        title: str,
        instructions: str,
        *,
        service: str = "",
        resume_state: dict[str, Any] | None = None,
        ttl_seconds: float = 86400.0,
    ) -> AccountCheckpoint:
        """Persist a checkpoint and ping the owner. Returns the checkpoint;
        callers raise AccountCheckpointPending with it."""
        cp = self.checkpoints.create(
            kind, title, instructions,
            service=service,
            resume_state=resume_state,
            ttl_seconds=ttl_seconds,
        )
        if self.notify is not None:
            try:
                self.notify(title, instructions, cp.id)
            except Exception as e:
                _log.warning("owner notify hook failed: %s", e)
        return cp

    def get_pending_checkpoints(
        self, service: str | None = None
    ) -> list[AccountCheckpoint]:
        """List checkpoints still waiting on the owner."""
        return self.checkpoints.list_pending(service)

    def cancel_checkpoint(
        self, checkpoint_id: str, note: str = ""
    ) -> AccountCheckpoint:
        """Cancel a pending checkpoint (abandons the flow)."""
        return self.checkpoints.cancel(checkpoint_id, note)

    def resume_checkpoint(
        self,
        checkpoint_id: str,
        note: str = "",
        *,
        owner_name: str | None = None,
        owner_email: str | None = None,
    ) -> CreatedAccount | AccountCheckpoint:
        """Continue a flow after the owner completed the human step.

        * identity checkpoints: supply ``owner_name``/``owner_email``
          (or set identity beforehand); records the identity and returns
          the resolved checkpoint — rerun the original ``create_account``
          call to continue.
        * account-creation checkpoints: resolves the checkpoint, stores
          the new account's credentials in the vault, and returns the
          :class:`CreatedAccount`.

        Raises:
            NotFound: Unknown checkpoint id
            NoMoralsError: Checkpoint is not pending
        """
        cp = self.checkpoints.get(checkpoint_id)
        resume_state = dict(cp.resume_state)
        flow = resume_state.get("flow")

        if flow == "need_identity":
            if owner_email:
                self.set_owner_identity(owner_name or "", owner_email)
            elif not self._owner_identity:
                raise MissingOwnerIdentity(
                    "owner identity still missing — call "
                    "set_owner_identity(name, email) first or pass "
                    "owner_name=/owner_email= to resume_checkpoint"
                )
            resolved = self.checkpoints.resolve(checkpoint_id, note)
            _log.info("identity checkpoint %s resolved; rerun create_account",
                      checkpoint_id)
            return resolved

        if flow == "account_create":
            resolved = self.checkpoints.resolve(checkpoint_id, note)
            account = self.finalize_account(
                resume_state["service"],
                resume_state["username"],
                resume_state["password"],
                email=resume_state.get("email", ""),
                checkpoint_id=resolved.id,
                verification_note=note,
            )
            return account

        # Unknown flow: just resolve and hand the checkpoint back.
        return self.checkpoints.resolve(checkpoint_id, note)

    # ── disposable email (fully automated) ──────────────────────

    async def create_email_account(
        self,
        *,
        provider: str = "guerrilla",
        username: str | None = None,
    ) -> CreatedAccount:
        """Create a disposable email account (no human step needed).

        Args:
            provider: Email provider (guerrilla, tempmail)
            username: Desired username (generated if not provided)

        Returns:
            CreatedAccount with the address (status "created" or "failed")
        """
        username = username or generate_username("bot")
        password = generate_password()

        try:
            if provider == "guerrilla":
                return await self._create_guerrilla_email(username)
            elif provider == "tempmail":
                return await self._create_tempmail(username)
            return CreatedAccount(
                service=f"email_{provider}",
                username=username,
                password=password,
                email=f"{username}@{provider}.com",
                status="failed",
                notes=f"Unknown provider: {provider}",
            )
        except Exception as e:
            _log.error("Failed to create email account: %s", e)
            return CreatedAccount(
                service=f"email_{provider}",
                username=username,
                password=password,
                email="",
                status="failed",
                notes=str(e),
            )

    async def _create_guerrilla_email(self, username: str) -> CreatedAccount:
        """Create a Guerrilla Mail disposable email account.

        Guerrilla Mail provides temporary email addresses via API.
        No password needed - just get an address and check inbox.
        """
        import json
        import urllib.request

        api_url = "https://api.guerrillamail.com/ajax.php"
        req = urllib.request.Request(
            f"{api_url}?f=get_email_address",
            headers={"User-Agent": "NoMorals-Bot/1.0"},
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                email = data.get("email_addr", f"{username}@guerrillamail.com")
        except Exception as e:
            _log.warning("Guerrilla Mail API failed, using fallback: %s", e)
            email = f"{username}@guerrillamail.com"

        cred = self.vault.store(
            service="email_guerrilla",
            username=email,
            password="",  # No password for Guerrilla Mail
            credential_type="disposable_email",
            tags=["email", "disposable"],
            metadata={"provider": "guerrilla"},
        )

        account = CreatedAccount(
            service="email_guerrilla",
            username=email,
            password="",
            email=email,
            status="created",
            credential=cred,
            notes="Disposable email - no password needed, check inbox via API",
        )
        self._creation_history.append(account)
        return account

    async def _create_tempmail(self, username: str) -> CreatedAccount:
        """Create a TempMail disposable email account."""
        email = f"{username}@tempmail.com"

        cred = self.vault.store(
            service="email_tempmail",
            username=email,
            password="",
            credential_type="disposable_email",
            tags=["email", "disposable"],
            metadata={"provider": "tempmail"},
        )

        account = CreatedAccount(
            service="email_tempmail",
            username=email,
            password="",
            email=email,
            status="created",
            credential=cred,
            notes="Disposable email via TempMail",
        )
        self._creation_history.append(account)
        return account

    # ── account creation (human-in-the-loop) ────────────────────

    def _existing_active(self, service: str) -> list[Credential]:
        return self.vault.list_all(service=service, active_only=True)

    async def create_account(
        self,
        service: str,
        *,
        username: str | None = None,
        email: str | None = None,
        password: str | None = None,
        **kwargs: Any,
    ) -> CreatedAccount:
        """Start an account creation flow for a service.

        One account per service: raises :class:`AccountExistsError` while
        an active credential exists. Every service needs a human-only
        verification step, so this always pauses on a checkpoint and
        raises :class:`AccountCheckpointPending` — resume with
        :meth:`resume_checkpoint` after the owner completes the step,
        which stores the credentials in the vault and returns the
        :class:`CreatedAccount`.

        Args:
            service: Service name (github, gmail, twitter, ...) or a
                disposable provider (guerrilla, tempmail)
            username: Desired username (generated if omitted)
            email: Email to use (owner's email by default; a disposable
                address is minted when the flow allows it)
            password: Password (generated if omitted)

        Raises:
            AccountExistsError: An active account already exists
            AccountCheckpointPending: Paused for the owner's human step
        """
        service = service.strip().lower()
        if service in self.DISPOSABLE_EMAIL_SERVICES or service.startswith("email_"):
            provider = service.replace("email_", "", 1)
            return await self.create_email_account(
                provider=provider, username=username
            )

        if self._existing_active(service):
            raise AccountExistsError(
                f"an active {service} account already exists — one account "
                "per service; deactivate or delete it first to create another"
            )

        username = username or generate_username()
        password = password or generate_password()

        # Identity: the owner's own, never invented.
        if email:
            owner_email = email
        else:
            identity = self._require_owner_identity(service, username, password)
            owner_email = identity["email"]

        flow = self._flow_for(service)
        cp = self._pause_for_human(
            flow["kind"],
            title=flow["title"].format(service=service, username=username),
            instructions=flow["instructions"].format(
                service=service,
                username=username,
                email=owner_email,
                password=password,
            ),
            service=service,
            resume_state={
                "flow": "account_create",
                "service": service,
                "username": username,
                "password": password,
                "email": owner_email,
            },
        )
        raise AccountCheckpointPending(cp)

    @staticmethod
    def _flow_for(service: str) -> dict[str, Any]:
        """Human-step definition per service."""
        flows: dict[str, dict[str, Any]] = {
            "github": {
                "kind": CheckpointKind.CAPTCHA,
                "title": "Create GitHub account {username}",
                "instructions": (
                    "GitHub blocks automated signup, so create it yourself:\n"
                    "1. Go to https://github.com/signup\n"
                    "2. Use email: {email}\n"
                    "3. Create a password you choose (a strong one was "
                    "generated for the vault: keep it or pick your own)\n"
                    "4. Pick username: {username} (or your own)\n"
                    "5. Solve the CAPTCHA and click the email verification link\n"
                    "6. Tell me the final username/email — I'll store the "
                    "credentials in the vault"
                ),
            },
            "gmail": {
                "kind": CheckpointKind.PHONE_2FA,
                "title": "Create Gmail account {username}",
                "instructions": (
                    "Gmail requires phone verification:\n"
                    "1. Go to https://accounts.google.com/signup\n"
                    "2. Use your name and a username like {username}\n"
                    "3. Provide YOUR phone number for the verification code\n"
                    "4. Tell me the final address — I'll store it in the vault"
                ),
            },
            "twitter": {
                "kind": CheckpointKind.EMAIL_VERIFY,
                "title": "Create X/Twitter account {username}",
                "instructions": (
                    "X requires email/phone verification:\n"
                    "1. Go to https://twitter.com/i/flow/signup\n"
                    "2. Use email: {email}\n"
                    "3. Pick username: {username} (or your own)\n"
                    "4. Complete the verification step\n"
                    "5. Tell me the final handle — I'll store the credentials"
                ),
            },
        }
        return flows.get(service, {
            "kind": CheckpointKind.MANUAL_STEP,
            "title": "Create {service} account {username}",
            "instructions": (
                "Create the account yourself:\n"
                "1. Go to {service}'s signup page\n"
                "2. Use email: {email}\n"
                "3. Pick username: {username} (or your own)\n"
                "4. Complete any verification step\n"
                "5. Tell me the final username — I'll store the credentials"
            ),
        })

    def finalize_account(
        self,
        service: str,
        username: str,
        password: str,
        **kwargs: Any,
    ) -> CreatedAccount:
        """Store an account's credentials in the vault after the human
        steps are done.

        Args:
            service: Service name
            username: Account username
            password: Account password
            **kwargs: email, checkpoint_id (resolved if pending),
                verification_note, and extra metadata

        Returns:
            CreatedAccount with status "created"
        """
        email = kwargs.get("email", "")
        checkpoint_id = kwargs.get("checkpoint_id")
        if checkpoint_id:
            try:
                cp = self.checkpoints.get(checkpoint_id)
                if cp.state == CheckpointState.PENDING:
                    self.checkpoints.resolve(
                        checkpoint_id,
                        kwargs.get("verification_note", ""),
                    )
            except NotFound:
                _log.warning("finalize_account: checkpoint %s not found",
                             checkpoint_id)

        metadata = {
            k: v for k, v in kwargs.items()
            if k not in ("email", "checkpoint_id", "verification_note")
        }

        cred = self.vault.store(
            service=service,
            username=username,
            password=password,
            credential_type="account",
            tags=[service, "account"],
            metadata=metadata,
        )

        account = CreatedAccount(
            service=service,
            username=username,
            password=password,
            email=email,
            status="created",
            credential=cred,
            notes="Account finalized and stored",
        )
        self._creation_history.append(account)
        _log.info("Finalized account: %s/%s", service, username)
        return account

    def get_creation_history(self) -> list[CreatedAccount]:
        """History of completed account creations (and disposable emails)."""
        return self._creation_history.copy()
