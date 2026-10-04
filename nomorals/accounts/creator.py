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
* **solver first, human fallback** — when automation hits a CAPTCHA, the
  CAPTCHA solver (``nomorals.tools.captcha``) is tried FIRST — it is ON
  by default (service backend, then owner-takeover backend as fallback).
  Only when the solver is disabled (``NM_CAPTCHA_SOLVER=0``) or the
  solve fails does the flow persist a checkpoint, ping the owner
  through the injected ``notify`` hook, and raise
  :class:`AccountCheckpointPending`. A later call to
  :meth:`AccountCreator.resume_checkpoint` continues the flow after the
  owner acts.

The solver itself lives in the higher ``tools`` layer, so it is
**injected** rather than imported here (layering: L2 may not import
L4) — see :func:`nomorals.tools.captcha.creator_solver_adapter`.

Usage:
    from nomorals.tools.captcha import creator_solver_adapter

    creator = AccountCreator(
        vault,
        notify=send_owner_ping,
        captcha_solver=creator_solver_adapter(),  # solver ON by default
    )
    creator.set_owner_identity("Death", "owner@example.com")

    try:
        creator.create_account("github", username="my-bot")
    except AccountCheckpointPending as pending:
        # solver failed or is off — owner solves the CAPTCHA, then:
        account = creator.resume_checkpoint(pending.checkpoint.id)
"""

from __future__ import annotations

import json
import os
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

# Injected CAPTCHA solver: takes a challenge dict
# {"kind", "sitekey", "page_url", "image_url", "image_bytes", "action",
#  "min_score"} and returns a result dict shaped like
# tools.captcha.SolveResult.to_dict():
# {"ok", "kind", "backend", "token", "text", "takeover", "elapsed_ms",
#  "detail"}.
# Injected (rather than imported) because accounts is L2 and the solver
# lives in tools (L4) — lower layers may not import higher ones. Build
# one with nomorals.tools.captcha.creator_solver_adapter().
CaptchaSolverFn = Callable[[dict[str, Any]], dict[str, Any]]


class AccountCreator:
    """Creates accounts on services with human-in-the-loop checkpoints.

    One account per service, the owner's own identity, and human
    verification steps (CAPTCHA, email/phone verification) pause on a
    persisted checkpoint instead of being bypassed or faked.

    CAPTCHA policy: the injected ``captcha_solver`` is tried FIRST
    whenever a challenge is hit — it is ON by default. The owner is
    pinged only when the solver is disabled (``solver_enabled=False`` or
    ``NM_CAPTCHA_SOLVER=0``) or the solve fails. Every solve attempt is
    audit-logged by the solver itself.
    """

    #: Services that hand out throwaway addresses via API — exempt from
    #: the one-account-per-service rule and the owner-identity rule.
    DISPOSABLE_EMAIL_SERVICES = frozenset({
        "guerrilla", "tempmail", "mailtm", "1secmail",
    })

    def __init__(
        self,
        vault: CredentialVault,
        browser_session: Any = None,
        *,
        db: Database | None = None,
        notify: NotifyHook | None = None,
        captcha_solver: CaptchaSolverFn | None = None,
        solver_enabled: bool | None = None,
    ) -> None:
        self.vault = vault
        self.browser = browser_session
        self.db = db or vault.db
        self.checkpoints = CheckpointStore(self.db)
        self.notify = notify
        # Solver wiring (injected: accounts/L2 may not import tools/L4).
        # solver_enabled=None → NM_CAPTCHA_SOLVER env, default ON.
        self.captcha_solver = captcha_solver
        self._solver_enabled = solver_enabled
        self._owner_identity: dict[str, str] | None = None
        self._creation_history: list[CreatedAccount] = []
        _log.info("Account creator initialized")

    # ── owner identity ──────────────────────────────────────────

    #: kv_store key for the persisted owner identity bank.
    IDENTITY_KV_KEY = "accounts.owner_identity"

    def set_owner_identity(
        self,
        name: str,
        email: str,
        *,
        phone: str | None = None,
    ) -> dict[str, str]:
        """Record the owner's real identity for account flows.

        Flows never invent fake identities; the name/email here is what
        signups use. Replaces any previously recorded identity. Persisted
        to the database so it survives restarts (the profile bank).

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
        self._persist_identity(identity)
        _log.info("owner identity recorded for %s", email)
        return dict(identity)

    def get_owner_identity(self) -> dict[str, str] | None:
        """The recorded owner identity, or None if not set.

        Loads from the database on first access if not in memory.
        """
        if self._owner_identity:
            return dict(self._owner_identity)
        loaded = self._load_identity()
        if loaded:
            self._owner_identity = loaded
            return dict(loaded)
        return None

    def _persist_identity(self, identity: dict[str, str]) -> None:
        """Save the identity bank to kv_store (best-effort)."""
        if self.db is None:
            return
        try:
            import json
            self.db.execute(
                "INSERT OR REPLACE INTO kv_store (key, value, kind, updated_at) VALUES (?, ?, 'json', ?)",
                (self.IDENTITY_KV_KEY, json.dumps(identity), __import__("time").time()),
            )
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            _log.warning("could not persist owner identity: %s", exc)

    def _load_identity(self) -> dict[str, str] | None:
        """Load the identity bank from kv_store, or None.

        Returns partial identities too — the signup flow validates
        completeness when actually needed.
        """
        if self.db is None:
            return None
        try:
            import json
            row = self.db.query_one(
                "SELECT value FROM kv_store WHERE key = ?", (self.IDENTITY_KV_KEY,)
            )
            if row and row.get("value"):
                data = json.loads(row["value"])
                if isinstance(data, dict) and (data.get("name") or data.get("email")):
                    out = {}
                    if data.get("name"):
                        out["name"] = str(data["name"])
                    if data.get("email"):
                        out["email"] = str(data["email"])
                    if data.get("phone"):
                        out["phone"] = str(data["phone"])
                    return out if out else None
        except Exception as exc:  # noqa: BLE001 - best-effort
            _log.debug("could not load owner identity: %s", exc)
        return None

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

    # ── CAPTCHA solver (tried first, human checkpoint is the fallback) ──

    def _solver_on(self) -> bool:
        """Is the CAPTCHA solver enabled? Explicit flag wins; otherwise
        the ``NM_CAPTCHA_SOLVER`` env var (default ON)."""
        if self._solver_enabled is not None:
            return self._solver_enabled
        return os.environ.get("NM_CAPTCHA_SOLVER", "1") != "0"

    def solve_captcha(self, challenge: dict[str, Any]) -> dict[str, Any]:
        """Run one challenge through the injected solver.

        ``challenge`` carries ``kind`` plus ``sitekey``/``page_url``/
        ``image_url``/``image_bytes``/``action``/``min_score`` as known.
        Returns the solver's result dict (``ok``, ``takeover``,
        ``token``/``text``, ``backend``, ``detail``). Every attempt is
        audit-logged by the solver itself.

        Raises:
            NoMoralsError: No solver injected, or the solver is disabled.
        """
        if self.captcha_solver is None:
            raise NoMoralsError(
                "no CAPTCHA solver injected — construct AccountCreator "
                "with captcha_solver=creator_solver_adapter()"
            )
        if not self._solver_on():
            raise NoMoralsError(
                "CAPTCHA solver is disabled "
                "(solver_enabled=False or NM_CAPTCHA_SOLVER=0)"
            )
        return self.captcha_solver(dict(challenge))

    def attempt_captcha_solve(
        self,
        *,
        kind: str,
        sitekey: str = "",
        page_url: str = "",
        image_url: str = "",
        image_bytes: bytes = b"",
        action: str = "",
        min_score: float = 0.3,
        service: str = "",
        resume_state: dict[str, Any] | None = None,
    ) -> str:
        """Try the solver first; fall back to a human checkpoint.

        This is the integration point for browser-driven ``_create_*``
        flows: when automation detects a CAPTCHA, call this instead of
        giving up. On success returns the solve token (or solved text
        for image captchas) so the flow can continue unattended.

        Only when the solver is unavailable/disabled or the solve fails
        does this persist a CAPTCHA checkpoint, ping the owner, and
        raise :class:`AccountCheckpointPending`.

        Raises:
            AccountCheckpointPending: Owner must solve it by hand.
        """
        result: dict[str, Any] | None = None
        if self.captcha_solver is not None and self._solver_on():
            try:
                result = self.solve_captcha({
                    "kind": kind,
                    "sitekey": sitekey,
                    "page_url": page_url,
                    "image_url": image_url,
                    "image_bytes": image_bytes,
                    "action": action,
                    "min_score": min_score,
                })
            except Exception as exc:  # noqa: BLE001 — fall back to human
                _log.warning("captcha solver raised, falling back to "
                             "human checkpoint: %s", exc)
                result = None

        if result and result.get("ok"):
            token = result.get("token") or result.get("text") or ""
            _log.info("captcha solved via %s backend",
                      result.get("backend", "unknown"))
            return token

        # Solver off, missing, or failed → the owner takes over.
        reason = "solver disabled" if not self._solver_on() else (
            "solver unavailable" if self.captcha_solver is None
            else f"solver failed: {(result or {}).get('detail', 'unknown')}"
        )
        state = dict(resume_state or {})
        state.update({
            "flow": "captcha_takeover",
            "service": service,
            "kind": kind,
            "sitekey": sitekey,
            "page_url": page_url,
        })
        cp = self._pause_for_human(
            CheckpointKind.CAPTCHA,
            title=f"CAPTCHA needs a human hand ({service or 'signup'})",
            instructions=(
                f"The automated solver could not clear this one ({reason}).\n"
                f"1. Open {page_url or 'the signup page'} in your browser\n"
                "2. Complete the CAPTCHA challenge yourself\n"
                "3. Resume — the flow continues from here"
            ),
            service=service,
            resume_state=state,
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
            provider: Email provider (guerrilla, tempmail, mailtm, 1secmail)
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
            elif provider == "mailtm":
                return await self._create_mailtm_email(username)
            elif provider == "1secmail":
                return await self._create_1secmail_email(username)
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

    # -- mail.tm (real REST API, no key needed) ---------------------

    _MAILTM_API = "https://api.mail.tm"

    def _mailtm_request(self, method: str, path: str,
                        payload: dict | None = None,
                        token: str | None = None,
                        timeout: float = 15) -> dict:
        """Synchronous mail.tm API call (called from async via executor)."""
        import json
        import urllib.request
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {
            "User-Agent": "Devon/1.0",
            "Accept": "application/json",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            f"{self._MAILTM_API}{path}", data=data,
            headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        return json.loads(body) if body.strip() else {}

    async def _create_mailtm_email(self, username: str) -> CreatedAccount:
        """Create a mail.tm disposable address via their public REST API.

        Full lifecycle: pick a domain -> create account -> fetch a JWT
        token.  The token is stored in the vault metadata so the inbox
        can be polled later without re-authenticating.
        """
        import asyncio
        import secrets

        loop = asyncio.get_running_loop()
        try:
            domains = await loop.run_in_executor(
                None, self._mailtm_request, "GET", "/domains")
            members = domains.get("hydra:member") or []
            domain = next((d.get("domain") for d in members
                           if d.get("isActive")), None)
            if not domain:
                raise RuntimeError("mail.tm returned no active domains")
            address = f"{username}@{domain}".replace(" ", "").lower()
            password = secrets.token_urlsafe(20)

            acc = await loop.run_in_executor(
                None, self._mailtm_request, "POST", "/accounts",
                {"address": address, "password": password})
            account_id = acc.get("id")
            if not account_id:
                raise RuntimeError(
                    f"mail.tm account creation failed: {str(acc)[:120]}")
            tok = await loop.run_in_executor(
                None, self._mailtm_request, "POST", "/token",
                {"address": address, "password": password})
            jwt = tok.get("token", "")
        except Exception as exc:  # noqa: BLE001 - provider down, report it
            _log.warning("mail.tm account creation failed: %s", exc)
            return CreatedAccount(
                service="email_mailtm",
                username=username,
                password="",
                email="",
                status="failed",
                notes=f"mail.tm error: {exc}",
            )

        cred = self.vault.store(
            service="email_mailtm",
            username=address,
            password=password,
            credential_type="disposable_email",
            tags=["email", "disposable"],
            metadata={"provider": "mailtm", "account_id": account_id,
                      "jwt": jwt},
        )
        account = CreatedAccount(
            service="email_mailtm",
            username=address,
            password=password,
            email=address,
            status="created",
            credential=cred,
            notes="Disposable email via mail.tm (API) - poll inbox with check_disposable_inbox",
        )
        self._creation_history.append(account)
        return account

    def mailtm_inbox(self, address: str, password: str,
                     limit: int = 10) -> list[dict]:
        """Poll a mail.tm inbox. Returns newest-first message dicts."""
        tok = self._mailtm_request(
            "POST", "/token", {"address": address, "password": password})
        jwt = tok.get("token", "")
        if not jwt:
            return []
        data = self._mailtm_request("GET", "/messages", token=jwt)
        out = []
        for m in (data.get("hydra:member") or [])[:limit]:
            out.append({
                "id": m.get("id"),
                "from": (m.get("from") or {}).get("address", ""),
                "subject": m.get("subject", ""),
                "date": m.get("createdAt", ""),
                "intro": m.get("intro", ""),
            })
        return out

    def mailtm_read(self, address: str, password: str,
                    message_id: str) -> dict:
        """Fetch one full mail.tm message (body included)."""
        tok = self._mailtm_request(
            "POST", "/token", {"address": address, "password": password})
        jwt = tok.get("token", "")
        if not jwt:
            return {}
        return self._mailtm_request("GET", f"/messages/{message_id}",
                                    token=jwt)

    # -- 1secmail (simple GET API, no key) --------------------------

    _ONEC_API = "https://www.1secmail.com/api/v1/"

    def _1secmail_get(self, params: dict,
                      timeout: float = 15) -> Any:
        """GET against the 1secmail API, returns parsed JSON."""
        import json
        import urllib.parse
        import urllib.request
        qs = urllib.parse.urlencode(params)
        req = urllib.request.Request(
            f"{self._ONEC_API}?{qs}",
            headers={"User-Agent": "Mozilla/5.0 (Devon/1.0)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    async def _create_1secmail_email(self, username: str) -> CreatedAccount:
        """Mint a 1secmail address.  No registration call needed — any
        login@domain on their domain list is instantly receivable; the
        address is reserved here and stored so the inbox can be polled.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        try:
            domains = await loop.run_in_executor(
                None, self._1secmail_get, {"action": "getDomainList"})
            if not isinstance(domains, list) or not domains:
                raise RuntimeError("1secmail returned no domains")
            domain = domains[0]
            login = username.replace(" ", "").lower() or "devon"
            address = f"{login}@{domain}"
        except Exception as exc:  # noqa: BLE001 - provider down
            _log.warning("1secmail setup failed: %s", exc)
            return CreatedAccount(
                service="email_1secmail",
                username=username,
                password="",
                email="",
                status="failed",
                notes=f"1secmail error: {exc}",
            )

        cred = self.vault.store(
            service="email_1secmail",
            username=address,
            password="",
            credential_type="disposable_email",
            tags=["email", "disposable"],
            metadata={"provider": "1secmail", "login": login,
                      "domain": domain},
        )
        account = CreatedAccount(
            service="email_1secmail",
            username=address,
            password="",
            email=address,
            status="created",
            credential=cred,
            notes="Disposable email via 1secmail (API) - poll inbox with check_disposable_inbox",
        )
        self._creation_history.append(account)
        return account

    def onec_inbox(self, login: str, domain: str,
                   limit: int = 10) -> list[dict]:
        """Poll a 1secmail inbox. Returns newest-first message dicts."""
        try:
            msgs = self._1secmail_get({
                "action": "getMessages", "login": login, "domain": domain})
        except Exception:  # noqa: BLE001
            return []
        out = []
        for m in (msgs or [])[:limit]:
            if isinstance(m, dict):
                out.append({
                    "id": m.get("id"),
                    "from": m.get("from", ""),
                    "subject": m.get("subject", ""),
                    "date": m.get("date", ""),
                })
        return out

    def onec_read(self, login: str, domain: str, message_id: int) -> dict:
        """Fetch one full 1secmail message (body included)."""
        try:
            return self._1secmail_get({
                "action": "readMessage", "login": login,
                "domain": domain, "id": message_id}) or {}
        except Exception:  # noqa: BLE001
            return {}

    # -- unified disposable inbox -----------------------------------

    def check_disposable_inbox(self, credential: Credential,
                               limit: int = 10) -> list[dict]:
        """Poll the inbox for a stored disposable-email credential.

        Dispatches on the credential metadata provider.  Returns a list
        of message dicts (id/from/subject/date/intro).  Sync — cheap
        enough to call from command handlers.
        """
        meta = credential.metadata or {}
        provider = str(meta.get("provider", ""))
        try:
            if provider == "mailtm":
                return self.mailtm_inbox(
                    credential.username, credential.password or "",
                    limit=limit)
            if provider == "1secmail":
                return self.onec_inbox(
                    str(meta.get("login", "")),
                    str(meta.get("domain", "")), limit=limit)
            if provider == "guerrilla":
                return self._guerrilla_inbox(
                    str(meta.get("sid_token", "")), limit=limit)
        except Exception as exc:  # noqa: BLE001 - inbox poll never crashes
            _log.warning("inbox poll failed for %s: %s", provider, exc)
        return []

    def _guerrilla_inbox(self, sid_token: str,
                         limit: int = 10) -> list[dict]:
        """Poll a Guerrilla Mail inbox via their API."""
        import json
        import urllib.request
        if not sid_token:
            return []
        url = (f"https://api.guerrillamail.com/ajax.php?f=get_email_list"
               f"&offset=0&sid_token={sid_token}")
        req = urllib.request.Request(
            url, headers={"User-Agent": "Devon/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        out = []
        for m in (data.get("list") or [])[:limit]:
            if isinstance(m, dict):
                out.append({
                    "id": m.get("mail_id"),
                    "from": m.get("mail_from", ""),
                    "subject": m.get("mail_subject", ""),
                    "date": m.get("mail_date", ""),
                    "intro": m.get("mail_excerpt", ""),
                })
        return out

    # -- temporary SMS numbers --------------------------------------

    def get_temp_number(self, country: str = "us",
                        provider: str = "simcodes") -> dict:
        """Grab a free temporary phone number for SMS verification.

        Returns a dict with number/masked/country/inbox_id/provider.
        Use :meth:`poll_sms_code` to wait for the verification code.
        """
        from .temp_sms import grab_number

        return grab_number(country=country, provider=provider)

    def poll_sms_code(self, number_info: dict, *,
                      sender_hint: str = "",
                      timeout: float = 180) -> str:
        """Wait for an SMS verification code on a temp number.

        ``number_info`` is the dict returned by :meth:`get_temp_number`.
        Returns the code or "" on timeout.
        """
        from .temp_sms import wait_code

        return wait_code(number_info, sender_hint=sender_hint,
                         timeout=timeout)

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
        an active credential exists. When automation hits a CAPTCHA, the
        injected solver is tried FIRST (ON by default) — only if it is
        disabled or fails does the flow pause on a checkpoint and raise
        :class:`AccountCheckpointPending`. Resume with
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
            ValueError: ``service`` is not a non-empty string
            TypeError: a provided ``username``/``email``/``password`` is not a string
        """
        if not isinstance(service, str) or not service.strip():
            raise ValueError(f"service must be a non-empty string, got {service!r}")
        for label, value in (("username", username), ("email", email),
                             ("password", password)):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{label} must be a string, got {type(value).__name__}")
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
                    "GitHub signup hit a CAPTCHA the automated solver "
                    "could not clear, so create it yourself:\n"
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
                    "   (or ask me: `/trial sms` grabs a free temp number and\n"
                    "   `/trial sms code` watches its inbox for the code)\n"
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
