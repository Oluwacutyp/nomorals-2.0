"""Browser-driven signup flow with a challenge-solving cascade.

``SignupDriver`` fills real signup forms through a duck-typed page
driver (the ``Tab`` interface from ``nomorals.browser`` — *not* imported
here; accounts is L2 and the browser is L4, so the page is injected),
mints a disposable email for verification, polls its inbox for the
verification link/code, and vault-stores the finished credentials.

Stage machine per attempt::

    started → form_filled → email_sent → verified → complete
       │           │            │            │
       └───────────┴────────────┴────────────┴──→ stopped_at_wall / failed

Challenge policy (the owner's standing rule): Devon does NOT stop at
verification walls — it solves them itself and pings the owner ONLY
when genuinely stuck:

* **CAPTCHA** → the injected solver first (default ON, per the standing
  CAPTCHA preference). Owner pinged only when the solver is disabled or
  can't crack it.
* **email verification** → disposable temp-mail inbox (already wired).
* **SMS/phone gate** → temp-number cascade (simcodes → 7sim → …),
  automatically. The owner's real phone number is NEVER used.
* **rate limit** → backoff retries, then owner ping.
* **manual review / real-ID demand / age gate / unknown signup URL** →
  genuinely stuck: hand to the owner with the exact status (what's done,
  what's needed).

Contact invariant: signups run on SELF-GENERATED identities — generated
names, temp emails, temp/virtual numbers. The owner's real email/phone
(``owner_contacts``) are checked against every contact detail the flow
touches; a match fails closed with ``NoMoralsError``.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Optional, Protocol, runtime_checkable, NoReturn

from ..core.errors import NoMoralsError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database
from .creator import (
    AccountCheckpoint,
    AccountCheckpointPending,
    AccountCreator,
    AccountExistsError,
    CheckpointKind,
    CreatedAccount,
)
from .identity_bank import ConfirmationRequired, Persona
from .vault import Credential, CredentialVault

__all__ = [
    "SignupStage",
    "WallKind",
    "SignupAttempt",
    "SignupAttemptStore",
    "SignupDriver",
    "PageDriver",
    "classify_wall",
    "KNOWN_SIGNUP_URLS",
    "ALLOWED_TRANSITIONS",
    "render_attempt_summary",
]

_log = get_logger(__name__)

#: notify hook: notify(title, body, attempt_id). Injected so this L2
#: module never imports a higher-layer notifier.
NotifyHook = Callable[[str, str, str], None]

#: page factory: returns an object satisfying the PageDriver protocol
#: (a ``nomorals.browser`` Tab). Injected from L5 (agents), never
#: imported here.
PageFactory = Callable[[], Any]


class SignupStage(StrEnum):
    """Lifecycle of one signup attempt."""

    STARTED = "started"
    FORM_FILLED = "form_filled"
    EMAIL_SENT = "email_sent"  # submitted, waiting on verification email
    VERIFIED = "verified"  # email verification completed
    COMPLETE = "complete"  # credentials vault-stored
    STOPPED_AT_WALL = "stopped_at_wall"
    FAILED = "failed"


class WallKind(StrEnum):
    """What stopped the flow. NONE = no wall detected."""

    NONE = "none"
    CAPTCHA = "captcha"
    SMS_PHONE = "sms_phone"  # phone-number/SMS-code gate
    MANUAL_REVIEW = "manual_review"  # human review / approval queue
    AGE_GATE = "age_gate"
    RATE_LIMIT = "rate_limit"
    REAL_ID = "real_id"  # government ID / document upload demand
    UNKNOWN = "unknown"


#: legal stage transitions. Terminal states have no outgoing edges —
#: a retry starts a *new* attempt linked via ``supersedes``.
ALLOWED_TRANSITIONS: dict[SignupStage, frozenset[SignupStage]] = {
    SignupStage.STARTED: frozenset({
        SignupStage.FORM_FILLED, SignupStage.STOPPED_AT_WALL,
        SignupStage.FAILED}),
    SignupStage.FORM_FILLED: frozenset({
        SignupStage.EMAIL_SENT, SignupStage.STOPPED_AT_WALL,
        SignupStage.FAILED}),
    SignupStage.EMAIL_SENT: frozenset({
        SignupStage.VERIFIED, SignupStage.STOPPED_AT_WALL,
        SignupStage.FAILED}),
    SignupStage.VERIFIED: frozenset({
        SignupStage.COMPLETE, SignupStage.STOPPED_AT_WALL,
        SignupStage.FAILED}),
    SignupStage.COMPLETE: frozenset(),
    SignupStage.STOPPED_AT_WALL: frozenset(),
    SignupStage.FAILED: frozenset(),
}


# ── wall classification ──────────────────────────────────────────────
#
# Signal lists are ordered: an explicit CAPTCHA detection beats page
# text, and a phone gate beats a generic "verify" mention. Email
# verification ("check your inbox") is deliberately NOT a wall — the
# driver handles it via the disposable inbox.

_CAPTCHA_HTML_MARKERS = (
    "g-recaptcha", "recaptcha/api", "hcaptcha", "h-captcha",
    "cf-turnstile", "turnstile", "data-sitekey",
    "arkose", "funcaptcha", "geetest",
)
_CAPTCHA_TEXT_PATTERNS = (
    r"verify you are (a )?human",
    r"i'?m not a robot",
    r"complete the (security )?challenge",
    r"select all (images|squares) with",
    r"captcha",
)

_WALL_PATTERNS: tuple[tuple[WallKind, tuple[str, ...]], ...] = (
    (WallKind.SMS_PHONE, (
        r"verify your (phone|mobile) number",
        r"enter your (phone|mobile) number",
        r"we(['’]ll| will) text (you|a code)",
        r"sms (verification )?code",
        r"enter the code (we|sent).*phone",
        r"phone verification",
    )),
    (WallKind.MANUAL_REVIEW, (
        r"under review",
        r"manually review",
        r"we(['’]ll| will) review your (account|application)",
        r"account.*(pending|awaiting).*(approval|review)",
        r"application.*(pending|awaiting).*(approval|review)",
    )),
    (WallKind.REAL_ID, (
        r"government[-\s]?issued id",
        r"upload.*(a )?(photo )?id\b",
        r"photo of your id",
        r"verify your identity.*document",
        r"id verification.*required",
    )),
    (WallKind.AGE_GATE, (
        r"you must be (at least )?(18|21) (years old )?to",
        r"not old enough",
        r"age requirement.*not met",
        r"come back when you['’]re older",
    )),
    (WallKind.RATE_LIMIT, (
        r"too many attempts",
        r"try again later",
        r"rate limit",
        r"temporarily blocked",
        r"unusual (traffic|activity).*try again",
    )),
)


def classify_wall(*, page_text: str = "", page_html: str = "",
                  captcha_detected: bool = False) -> tuple[WallKind, str]:
    """Classify the verification wall on a page.

    Returns ``(WallKind, evidence)`` — ``WallKind.NONE`` with empty
    evidence when no wall is found. Email verification ("check your
    inbox") is intentionally not a wall: the driver completes it via
    the disposable inbox.
    """
    text = (page_text or "").lower()
    html = (page_html or "").lower()

    if captcha_detected or any(m in html for m in _CAPTCHA_HTML_MARKERS):
        marker = next((m for m in _CAPTCHA_HTML_MARKERS if m in html),
                      "detected challenge")
        return WallKind.CAPTCHA, f"challenge marker on page: {marker}"
    for pattern in _CAPTCHA_TEXT_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return WallKind.CAPTCHA, f"page text: {m.group(0)!r}"

    for kind, patterns in _WALL_PATTERNS:
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return kind, f"page text: {m.group(0)!r}"
    return WallKind.NONE, ""


# ── attempt records ──────────────────────────────────────────────────

_SIGNUP_ATTEMPTS_DDL = """
CREATE TABLE IF NOT EXISTS signup_attempts (
    id TEXT PRIMARY KEY,
    service TEXT NOT NULL DEFAULT '',
    persona_id TEXT NOT NULL DEFAULT '',
    persona_name TEXT NOT NULL DEFAULT '',
    stage TEXT NOT NULL DEFAULT 'started',
    wall_kind TEXT NOT NULL DEFAULT 'none',
    wall_detail TEXT NOT NULL DEFAULT '',
    credential_ref TEXT NOT NULL DEFAULT '',
    signup_url TEXT NOT NULL DEFAULT '',
    supersedes TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_signup_attempts_service
    ON signup_attempts(service);
CREATE INDEX IF NOT EXISTS idx_signup_attempts_stage
    ON signup_attempts(stage);
"""


@dataclass
class SignupAttempt:
    """One tracked signup run: service, persona, stage, wall, credential."""

    id: str
    service: str
    persona_id: str
    persona_name: str = ""
    stage: SignupStage = SignupStage.STARTED
    wall_kind: WallKind = WallKind.NONE
    wall_detail: str = ""
    credential_ref: str = ""  # vault credential id once stored
    signup_url: str = ""
    supersedes: str = ""  # previous attempt id this one retries
    note: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "service": self.service,
            "persona_id": self.persona_id,
            "persona_name": self.persona_name,
            "stage": self.stage.value,
            "wall_kind": self.wall_kind.value,
            "wall_detail": self.wall_detail,
            "credential_ref": self.credential_ref,
            "signup_url": self.signup_url,
            "supersedes": self.supersedes,
            "note": self.note,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SignupAttemptStore:
    """Durable per-attempt signup records, backed by the app database."""

    def __init__(self, db: Database) -> None:
        self.db = db
        with self.db.transaction():
            self.db.executescript(_SIGNUP_ATTEMPTS_DDL)

    def create(self, *, service: str, persona: Persona, signup_url: str = "",
               supersedes: str = "", note: str = "") -> SignupAttempt:
        now = time.time()
        attempt = SignupAttempt(
            id=new_id("sua"),
            service=service.strip().lower(),
            persona_id=persona.id,
            persona_name=persona.name,
            stage=SignupStage.STARTED,
            signup_url=signup_url,
            supersedes=supersedes,
            note=note,
            created_at=now,
            updated_at=now,
        )
        with self.db.transaction():
            self.db.execute(
                """
                INSERT INTO signup_attempts
                (id, service, persona_id, persona_name, stage, wall_kind,
                 wall_detail, credential_ref, signup_url, supersedes, note,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (attempt.id, attempt.service, attempt.persona_id,
                 attempt.persona_name, attempt.stage.value,
                 attempt.wall_kind.value, attempt.wall_detail,
                 attempt.credential_ref, attempt.signup_url,
                 attempt.supersedes, attempt.note, attempt.created_at,
                 attempt.updated_at),
            )
        _log.info("signup attempt %s created for %s", attempt.id,
                  attempt.service)
        return attempt

    def get(self, attempt_id: str) -> SignupAttempt:
        from ..core.errors import NotFound

        row = self.db.query_one(
            "SELECT * FROM signup_attempts WHERE id = ?", (attempt_id,))
        if not row:
            raise NotFound(f"no signup attempt {attempt_id!r}")
        return self._from_row(row)

    def list(self, service: str | None = None,
             limit: int = 20) -> list[SignupAttempt]:
        if service:
            rows = self.db.query(
                "SELECT * FROM signup_attempts WHERE service = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (service.strip().lower(), limit),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM signup_attempts ORDER BY created_at DESC "
                "LIMIT ?", (limit,),
            )
        return [self._from_row(r) for r in rows or []]

    def transition(self, attempt_id: str, new_stage: SignupStage,
                   note: str = "") -> SignupAttempt:
        """Move an attempt to a new stage. Illegal transitions raise."""
        attempt = self.get(attempt_id)
        allowed = ALLOWED_TRANSITIONS[attempt.stage]
        if new_stage not in allowed:
            raise NoMoralsError(
                f"illegal signup transition {attempt.stage.value} → "
                f"{new_stage.value} for attempt {attempt_id}")
        now = time.time()
        with self.db.transaction():
            self.db.execute(
                "UPDATE signup_attempts SET stage = ?, note = ?, "
                "updated_at = ? WHERE id = ?",
                (new_stage.value, note or attempt.note, now, attempt_id),
            )
        attempt.stage = new_stage
        if note:
            attempt.note = note
        attempt.updated_at = now
        _log.info("signup attempt %s → %s", attempt_id, new_stage.value)
        return attempt

    def set_wall(self, attempt_id: str, wall: WallKind,
                 detail: str = "") -> SignupAttempt:
        attempt = self.get(attempt_id)
        now = time.time()
        with self.db.transaction():
            self.db.execute(
                "UPDATE signup_attempts SET wall_kind = ?, wall_detail = ?, "
                "updated_at = ? WHERE id = ?",
                (wall.value, detail, now, attempt_id),
            )
        attempt.wall_kind = wall
        attempt.wall_detail = detail
        attempt.updated_at = now
        return attempt

    def set_credential(self, attempt_id: str,
                       credential_ref: str) -> SignupAttempt:
        attempt = self.get(attempt_id)
        now = time.time()
        with self.db.transaction():
            self.db.execute(
                "UPDATE signup_attempts SET credential_ref = ?, "
                "updated_at = ? WHERE id = ?",
                (credential_ref, now, attempt_id),
            )
        attempt.credential_ref = credential_ref
        attempt.updated_at = now
        return attempt

    @staticmethod
    def _from_row(row: dict[str, Any]) -> SignupAttempt:
        return SignupAttempt(
            id=row["id"],
            service=row.get("service") or "",
            persona_id=row.get("persona_id") or "",
            persona_name=row.get("persona_name") or "",
            stage=SignupStage(row.get("stage") or "started"),
            wall_kind=WallKind(row.get("wall_kind") or "none"),
            wall_detail=row.get("wall_detail") or "",
            credential_ref=row.get("credential_ref") or "",
            signup_url=row.get("signup_url") or "",
            supersedes=row.get("supersedes") or "",
            note=row.get("note") or "",
            created_at=row.get("created_at") or 0,
            updated_at=row.get("updated_at") or 0,
        )


# ── page driver protocol ─────────────────────────────────────────────
#
# Structural typing only — the real Tab (nomorals.browser, L4) satisfies
# it without this L2 module importing it.

@runtime_checkable
class PageDriver(Protocol):
    """The tab operations a signup drive needs."""

    def navigate(self, url: str) -> Any: ...
    def fill(self, name: str, value: str) -> Any: ...
    def click(self, target: str) -> Any: ...
    def submit(self, target: str = "") -> Any: ...
    def text(self, max_chars: int = 40000) -> Any: ...
    def check_captcha(self, *, fetch_bytes: bool = False) -> Any: ...


def _page_text(page: Any, max_chars: int = 40000) -> str:
    """Best-effort page text from a dict- or str-returning text()."""
    try:
        raw = page.text(max_chars=max_chars)
    except Exception:  # noqa: BLE001
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for key in ("text", "content", "body", "markdown"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return str(raw)[:max_chars]
    return str(raw or "")[:max_chars]


def _page_html(page: Any) -> str:
    try:
        raw = page.html()
    except Exception:  # noqa: BLE001
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        val = raw.get("html")
        return val if isinstance(val, str) else ""
    return ""


def _captcha_detected(page: Any) -> bool:
    try:
        raw = page.check_captcha()
    except Exception:  # noqa: BLE001
        return False
    if isinstance(raw, dict):
        try:
            return int(raw.get("count", 0) or 0) > 0
        except (TypeError, ValueError):
            return bool(raw.get("challenges"))
    return False


#: Signup URLs grounded in the flows already referenced by
#: AccountCreator._flow_for (github/twitter/gmail instructions). Only
#: services with a known-good URL are driven; anything else stops at a
#: manual-step checkpoint instead of guessing a URL.
KNOWN_SIGNUP_URLS: dict[str, str] = {
    "github": "https://github.com/signup",
    "twitter": "https://twitter.com/i/flow/signup",
    "x": "https://twitter.com/i/flow/signup",
    "gmail": "https://accounts.google.com/signup",
    "google": "https://accounts.google.com/signup",
}

#: wall → human-checkpoint kind for the stop-and-hand-off.
_WALL_CHECKPOINT_KIND: dict[WallKind, CheckpointKind] = {
    WallKind.CAPTCHA: CheckpointKind.CAPTCHA,
    WallKind.SMS_PHONE: CheckpointKind.PHONE_2FA,
    WallKind.MANUAL_REVIEW: CheckpointKind.MANUAL_STEP,
    WallKind.AGE_GATE: CheckpointKind.MANUAL_STEP,
    WallKind.RATE_LIMIT: CheckpointKind.MANUAL_STEP,
    WallKind.REAL_ID: CheckpointKind.MANUAL_STEP,
    WallKind.UNKNOWN: CheckpointKind.MANUAL_STEP,
}

_VERIFICATION_LINK_RE = re.compile(r"https?://[^\s\"'<>]+", re.I)
_VERIFICATION_CODE_RE = re.compile(r"\b(\d{4,8})\b")


class SignupDriver:
    """Drives one signup attempt: form fill → verify → vault-store.

    Constructed with an :class:`AccountCreator` (temp email, vault,
    checkpoints, injected CAPTCHA solver), an attempt store, an optional
    notify hook, and an optional page factory (injected from L5 — the
    real browser Tab).
    """

    def __init__(
        self,
        creator: AccountCreator,
        attempts: SignupAttemptStore,
        *,
        notify: NotifyHook | None = None,
        page_factory: PageFactory | None = None,
        temp_email_provider: str = "tempmail",
        verify_timeout_s: float = 240.0,
        verify_poll_s: float = 12.0,
    ) -> None:
        self.creator = creator
        self.attempts = attempts
        self.notify = notify
        self.page_factory = page_factory
        self.temp_email_provider = temp_email_provider
        self.verify_timeout_s = verify_timeout_s
        self.verify_poll_s = verify_poll_s

    # ── public entry ──────────────────────────────────────────────

    async def adrive(
        self,
        *,
        service: str,
        persona: Persona,
        confirmed: bool,
        page: Any = None,
        signup_url: str | None = None,
        field_overrides: dict[str, str] | None = None,
        owner_contacts: dict[str, list[str]] | None = None,
        rate_limit_waits: tuple[float, ...] = (60.0, 300.0, 900.0),
        supersedes: str = "",
    ) -> SignupAttempt:
        """Run one signup attempt. Raises on genuine stuck-points (after
        the solve cascade is exhausted and the owner has been handed an
        exact status).

        ``confirmed`` is the code-level confirmation gate — False raises
        :class:`ConfirmationRequired` before anything happens. Chat/CLI
        callers set it from the ``--yes`` flag or a consumed
        ``ConfirmationGate`` token; there is no silent default.

        ``owner_contacts`` (``{"emails": [...], "phones": [...]}``) is the
        owner's real contact details — every email/phone the flow touches
        is checked against them and the drive fails closed on a match.
        The signup always runs on the persona's self-generated temp
        contacts, never the owner's.

        Raises:
            ConfirmationRequired: not explicitly confirmed
            NoMoralsError: a real owner contact detail would be used
            AccountExistsError: an active account already exists
            AccountCheckpointPending: genuinely stuck — the checkpoint
                carries the exact status and the owner's next step
        """
        if not confirmed:
            raise ConfirmationRequired(
                "signup needs explicit owner confirmation first — "
                "show the persona draft card and confirm it "
                "(--yes, or /trial confirm <token>)")
        if not persona.disposable:
            raise NoMoralsError(
                "refusing to drive a signup with a non-disposable persona")
        service = service.strip().lower()
        if not service:
            raise ValueError("service must be a non-empty string")
        owner_contacts = owner_contacts or {}
        self._assert_no_real_contact_email(persona, owner_contacts)

        existing = [
            c for c in self.creator.vault.list_all(service=service,
                                                   active_only=True)
            if (c.credential_type or "") == "account"
        ]
        if existing:
            raise AccountExistsError(
                f"an active {service} account already exists — one account "
                "per service")

        url = (signup_url or "").strip() or KNOWN_SIGNUP_URLS.get(service, "")
        attempt = self.attempts.create(
            service=service, persona=persona, signup_url=url,
            supersedes=supersedes)

        if not url:
            await self._hand_to_owner(
                attempt, WallKind.UNKNOWN,
                f"no known signup URL for {service!r} — refusing to guess",
                page_url="",
                needed="provide the signup page URL (or confirm the "
                       f"service name), then resume — nothing was filled yet.")

        tab = page if page is not None else self._open_page(attempt)
        if tab is None:
            await self._hand_to_owner(
                attempt, WallKind.UNKNOWN,
                "no browser page available for form driving", page_url=url,
                needed="no browser is attached to this run — start the "
                       "browser service and resume the attempt.")

        # pre-fill wall scan — handled walls (solved captcha, temp SMS,
        # waited-out rate limit) let the drive continue; genuine
        # stuck-points raise AccountCheckpointPending via _hand_to_owner.
        tab.navigate(url)
        wall, evidence = self._scan(tab)
        if wall is not WallKind.NONE:
            await self._handle_wall(
                attempt, tab, wall, evidence, url,
                owner_contacts=owner_contacts,
                rate_limit_waits=rate_limit_waits)

        email_account = await self._ensure_email(
            persona, owner_contacts=owner_contacts)
        username = await self._fill_and_submit(
            attempt, tab, persona, email_account.email,
            field_overrides or {})
        self.attempts.transition(attempt.id, SignupStage.FORM_FILLED)

        # post-submit wall scan
        wall, evidence = self._scan(tab)
        if wall is not WallKind.NONE:
            await self._handle_wall(
                attempt, tab, wall, evidence, url,
                owner_contacts=owner_contacts,
                rate_limit_waits=rate_limit_waits)

        text = _page_text(tab)
        if not self._looks_like_verification_pending(text):
            # submitted cleanly with no verification step detected —
            # treat as verified and finish.
            _log.info("signup %s: no verification step detected", service)
        self.attempts.transition(attempt.id, SignupStage.EMAIL_SENT,
                                 note="signup submitted; polling "
                                      "disposable inbox for verification")

        verified = await self._poll_verification(
            attempt, tab, email_account, service)
        if not verified:
            await self._hand_to_owner(
                attempt, WallKind.UNKNOWN,
                f"no verification email arrived within "
                f"{self.verify_timeout_s:.0f}s — the address may be "
                f"blocked",
                page_url=url,
                needed=f"check the disposable inbox for "
                       f"{email_account.email} manually; if the email is "
                       f"there, forward me the link/code and I'll finish "
                       f"the signup.")

        self.attempts.transition(attempt.id, SignupStage.VERIFIED)
        account = self.creator.finalize_account(
            service, username, persona.password,
            email=email_account.email,
            verification_note="email verified via disposable inbox",
            persona_id=persona.id,
            attempt_id=attempt.id,
        )
        if account.credential is not None:
            self.attempts.set_credential(attempt.id, account.credential.id)
        self.attempts.transition(attempt.id, SignupStage.COMPLETE,
                                 note=f"account {username} vault-stored")
        self._notify(f"signup complete — {service}",
                     f"✅ {service} account ready: {username} "
                     f"({email_account.email}). Credentials are in the vault.",
                     attempt.id)
        return self.attempts.get(attempt.id)

    # ── internals ─────────────────────────────────────────────────

    @staticmethod
    def _norm_contact(value: str) -> str:
        """Normalize an email/phone for comparison."""
        return re.sub(r"[\s\-().]", "", (value or "").lower())

    def _assert_no_real_contact_email(
        self, persona: Persona, owner_contacts: dict[str, list[str]]
    ) -> None:
        """Fail closed if the persona would use the owner's real email."""
        owner_emails = {self._norm_contact(e)
                        for e in (owner_contacts.get("emails") or [])}
        owner_emails.discard("")
        if persona.email and self._norm_contact(persona.email) in owner_emails:
            raise NoMoralsError(
                "refusing signup: the persona's email matches the owner's "
                "real email — trial signups never use the owner's real "
                "contact details")

    def _assert_no_real_contact_phone(
        self, number: str, owner_contacts: dict[str, list[str]]
    ) -> bool:
        """True if ``number`` is safe to use (not the owner's real phone).

        Returns False (skip this number) instead of raising, so the
        temp-number cascade can try the next source.
        """
        owner_phones = {self._norm_contact(p)
                        for p in (owner_contacts.get("phones") or [])}
        owner_phones.discard("")
        norm = self._norm_contact(number)
        if norm and norm in owner_phones:
            _log.warning("temp number matches owner's real phone — skipping")
            return False
        return True

    def _open_page(self, attempt: SignupAttempt) -> Any | None:
        if self.page_factory is None:
            return None
        try:
            return self.page_factory()
        except Exception as exc:  # noqa: BLE001
            _log.warning("page factory failed: %s", exc)
            return None

    def _scan(self, tab: Any) -> tuple[WallKind, str]:
        return classify_wall(
            page_text=_page_text(tab),
            page_html=_page_html(tab),
            captcha_detected=_captcha_detected(tab),
        )

    async def _ensure_email(
        self, persona: Persona,
        owner_contacts: dict[str, list[str]] | None = None,
    ) -> CreatedAccount:
        """Mint (or reuse) the persona's disposable verification address.

        Fails closed if the address would be the owner's real email.
        """
        self._assert_no_real_contact_email(persona, owner_contacts or {})
        if persona.email and persona.email_provider:
            return CreatedAccount(
                service=f"email_{persona.email_provider}",
                username=persona.email, password="",
                email=persona.email, status="created",
                notes="reused persona address")
        account = await self.creator.create_email_account(
            provider=self.temp_email_provider,
            username=(persona.username_variants[0]
                      if persona.username_variants else None) or None,
        )
        if account.status != "created" or not account.email:
            raise NoMoralsError(
                f"could not mint a disposable email "
                f"({self.temp_email_provider}): {account.notes}")
        persona.email = account.email
        persona.email_provider = self.temp_email_provider
        # invariant also holds for freshly minted addresses
        self._assert_no_real_contact_email(persona, owner_contacts or {})
        return account

    def _field_plan(self, persona: Persona, email: str,
                    overrides: dict[str, str]) -> list[tuple[str, str]]:
        """(field label, value) pairs, most-specific label first."""
        username = (persona.username_variants[0]
                    if persona.username_variants else "user")
        plan = [
            ("full name", persona.name),
            ("name", persona.name),
            ("email", email),
            ("email address", email),
            ("username", username),
            ("user name", username),
            ("password", persona.password),
            ("create password", persona.password),
            ("date of birth", persona.dob),
            ("birth date", persona.dob),
            ("birthday", persona.dob),
        ]
        plan.extend(overrides.items())
        return plan

    async def _fill_and_submit(
        self,
        attempt: SignupAttempt,
        tab: Any,
        persona: Persona,
        email: str,
        overrides: dict[str, str],
    ) -> str:
        """Fill the form; on username-taken, walk the variant list.

        Returns the username that was submitted.
        """
        variants = list(persona.username_variants) or ["user"]
        last_error = ""
        for i, username in enumerate(variants):
            plan = self._field_plan(persona, email, overrides)
            # swap the username value for this attempt's variant
            plan = [(label, username if "user name" in label or label == "username" else value)
                    for label, value in plan]
            filled = 0
            for label, value in plan:
                try:
                    tab.fill(label, value)
                    filled += 1
                except Exception as exc:  # noqa: BLE001 - field may not exist
                    _log.debug("fill %r skipped: %s", label, exc)
            _log.info("signup %s: filled %d fields (username %r, try %d)",
                      attempt.service, filled, username, i + 1)
            try:
                tab.submit()
            except Exception:
                # no form to submit — try the obvious buttons
                for target in ("sign up", "create account", "continue",
                               "register", "next"):
                    try:
                        tab.click(target)
                        break
                    except Exception:  # noqa: BLE001
                        continue
            await asyncio.sleep(2)
            text = _page_text(tab).lower()
            if re.search(r"username.*(taken|unavailable|already.*(use|taken))",
                         text):
                last_error = f"username {username!r} taken"
                _log.info("signup %s: %s — trying next variant",
                          attempt.service, last_error)
                continue
            wall, _ = self._scan(tab)
            if wall is not WallKind.NONE:
                # leave wall handling to the caller
                return username
            return username
        raise NoMoralsError(
            f"all {len(variants)} username variants exhausted for "
            f"{attempt.service} ({last_error})")

    @staticmethod
    def _looks_like_verification_pending(text: str) -> bool:
        low = (text or "").lower()
        return bool(re.search(
            r"check your (email|inbox)|verify your email|"
            r"verification (email|link|code).*sent|confirm your email",
            low))

    async def _poll_verification(
        self,
        attempt: SignupAttempt,
        tab: Any,
        email_account: CreatedAccount,
        service: str,
    ) -> bool:
        """Poll the disposable inbox; follow the link or enter the code."""
        cred = email_account.credential
        deadline = time.time() + self.verify_timeout_s
        provider = email_account.service.replace("email_", "", 1)
        while time.time() < deadline:
            messages = self._read_inbox(cred, provider)
            for msg in messages:
                body = f"{msg.get('subject', '')}\n{msg.get('text', '')}\n" \
                       f"{msg.get('html', '')}"
                if service not in body.lower() and not re.search(
                        r"verif|confirm|code", body, re.I):
                    continue
                link = self._extract_link(body)
                if link:
                    _log.info("signup %s: following verification link",
                              attempt.service)
                    tab.navigate(link)
                    await asyncio.sleep(2)
                    return True
                code = self._extract_code(msg)
                if code:
                    _log.info("signup %s: entering verification code",
                              attempt.service)
                    for label in ("verification code", "code",
                                  "enter code", "confirm"):
                        try:
                            tab.fill(label, code)
                            break
                        except Exception:  # noqa: BLE001
                            continue
                    try:
                        tab.submit()
                    except Exception:  # noqa: BLE001
                        try:
                            tab.click("verify")
                        except Exception:  # noqa: BLE001
                            pass
                    await asyncio.sleep(2)
                    return True
            await asyncio.sleep(self.verify_poll_s)
        return False

    def _read_inbox(self, cred: Credential | None,
                    provider: str) -> list[dict[str, Any]]:
        """Read full messages for link/code extraction."""
        if cred is None:
            return []
        try:
            if provider == "tempmail":
                msgs = self.creator.tempmail_inbox(cred.username, limit=5)
                out = []
                for m in msgs:
                    full = self.creator.tempmail_read(cred.username,
                                                      m.get("id") or "")
                    out.append({**m, **full})
                return out
            if provider == "mailtm":
                msgs = self.creator.mailtm_inbox(
                    cred.username, cred.password or "", limit=5)
                out = []
                for m in msgs:
                    full = self.creator.mailtm_read(
                        cred.username, cred.password or "",
                        str(m.get("id") or ""))
                    out.append({**m, **(full if isinstance(full, dict)
                                       else {})})
                return out
            # generic: summary list only
            return self.creator.check_disposable_inbox(cred, limit=5)
        except Exception as exc:  # noqa: BLE001 - poll never crashes
            _log.debug("verification inbox poll failed: %s", exc)
            return []

    @staticmethod
    def _extract_link(body: str) -> str:
        for m in _VERIFICATION_LINK_RE.finditer(body or ""):
            url = m.group(0).rstrip(".,);]")
            if re.search(r"verif|confirm|activate|token|code", url, re.I):
                return url
        return ""

    @staticmethod
    def _extract_code(msg: dict[str, Any]) -> str:
        for key in ("text", "intro", "subject"):
            val = str(msg.get(key) or "")
            m = _VERIFICATION_CODE_RE.search(val)
            # skip years / long numbers that are clearly not codes
            if m and not re.fullmatch(r"(19|20)\d{2}", m.group(1)):
                return m.group(1)
        return ""

    # ── challenge-solving cascade ───────────────────────────────
    #
    # Devon solves verification challenges itself and pings the owner
    # ONLY when genuinely stuck.  Each handler returns None when the
    # drive may continue, or raises AccountCheckpointPending (via
    # _hand_to_owner) with an exact status + the owner's next step.

    async def _handle_wall(
        self,
        attempt: SignupAttempt,
        tab: Any,
        wall: WallKind,
        evidence: str,
        page_url: str,
        *,
        owner_contacts: dict[str, list[str]] | None = None,
        rate_limit_waits: tuple[float, ...] = (),
    ) -> None:
        """Dispatch one detected wall to its solver."""
        if wall is WallKind.CAPTCHA:
            await self._handle_captcha_wall(attempt, tab, evidence, page_url)
            return
        if wall is WallKind.SMS_PHONE:
            await self._handle_sms_wall(
                attempt, tab, evidence, page_url,
                owner_contacts=owner_contacts or {})
            return
        if wall is WallKind.RATE_LIMIT:
            await self._handle_rate_limit_wall(
                attempt, tab, evidence, page_url,
                waits=rate_limit_waits)
            return
        await self._hand_to_owner(
            attempt, wall, evidence, page_url,
            needed=self._default_needed(wall, page_url))

    async def _handle_captcha_wall(
        self,
        attempt: SignupAttempt,
        tab: Any,
        evidence: str,
        page_url: str,
    ) -> None:
        """Try the injected solver first; owner takeover only if it fails.

        Raises AccountCheckpointPending when the solver is off or the
        solve fails (checkpoint + owner ping already done inside
        ``attempt_captcha_solve``). Returns None on a solved challenge so
        the drive continues.
        """
        challenge: dict[str, Any] = {"kind": "unknown", "page_url": page_url}
        try:
            raw = tab.check_captcha(fetch_bytes=True)
            if isinstance(raw, dict) and raw.get("challenges"):
                first = raw["challenges"][0]
                if isinstance(first, dict):
                    challenge.update({
                        "kind": first.get("kind", "unknown"),
                        "sitekey": first.get("sitekey", ""),
                        "image_bytes": first.get("image_bytes", b""),
                    })
        except Exception:  # noqa: BLE001
            pass
        try:
            token = self.creator.attempt_captcha_solve(
                kind=challenge.get("kind", "unknown"),
                sitekey=challenge.get("sitekey", ""),
                page_url=page_url,
                image_bytes=challenge.get("image_bytes", b"") or b"",
                service=attempt.service,
                resume_state={"attempt_id": attempt.id,
                              "flow": "signup_captcha"},
            )
        except AccountCheckpointPending as pending:
            # solver off/failed — a checkpoint was persisted inside
            # attempt_captcha_solve; mark the attempt, send the exact
            # status (the single owner ping for this stuck-point), re-raise.
            self.attempts.set_wall(attempt.id, WallKind.CAPTCHA, evidence)
            self.attempts.transition(
                attempt.id, SignupStage.STOPPED_AT_WALL,
                note="captcha unsolved — handed to owner")
            # NOTE: `attempt` still carries the pre-transition stage —
            # the status message must describe what was DONE, not the
            # terminal state.
            self._notify(
                f"⏸️ signup needs you — {attempt.service} (captcha)",
                self._owner_handoff_message(
                    attempt, WallKind.CAPTCHA, evidence,
                    f"solve the CAPTCHA at {page_url}, then resume "
                    f"checkpoint {pending.checkpoint.id} — I'll continue "
                    f"the signup from there."),
                attempt.id)
            raise
        _log.info("signup %s: captcha solved via solver, continuing",
                  attempt.service)
        self._notify(f"captcha auto-solved — {attempt.service}",
                     f"solver cleared the challenge ({evidence}); "
                     f"continuing the signup.", attempt.id)
        return

    async def _handle_sms_wall(
        self,
        attempt: SignupAttempt,
        tab: Any,
        evidence: str,
        page_url: str,
        *,
        owner_contacts: dict[str, list[str]],
        max_numbers: int = 3,
    ) -> None:
        """Solve an SMS/phone gate with the temp-number cascade.

        Grabs temp numbers across providers, fills each into the form,
        and polls its public inbox for the code.  The owner's real phone
        is never used — a grabbed number matching it is skipped.  Only
        when every source is exhausted (no number, or no code arrives)
        does the flow hand to the owner with the exact status.
        Returns None when the gate is passed so the drive continues.
        """
        import functools

        tried: list[str] = []
        for _ in range(max_numbers):
            info = self.creator.get_temp_number_cascade()
            if info.get("status") != "ok":
                tried.append(info.get("notes", "no number"))
                break
            number = info.get("number", "")
            masked = info.get("masked", "") or number
            if not self._assert_no_real_contact_phone(number,
                                                      owner_contacts):
                tried.append(f"{masked} (skipped: owner's number)")
                continue
            tried.append(f"{masked} via {info.get('provider')}")
            _log.info("signup %s: trying temp number %s",
                      attempt.service, masked)
            if not self._fill_first(tab, ("phone number", "mobile number",
                                          "phone", "number"), number):
                tried.append(f"{masked}: no phone field found")
                continue
            try:
                tab.submit()
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(2)
            loop = asyncio.get_event_loop()
            code = await loop.run_in_executor(
                None, functools.partial(
                    self.creator.poll_sms_code, info,
                    sender_hint=attempt.service, timeout=180))
            if not code:
                _log.info("signup %s: no SMS code on %s",
                          attempt.service, masked)
                continue
            self._fill_first(tab, ("sms code", "verification code",
                                   "code", "enter code"), code)
            try:
                tab.submit()
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(2)
            wall, _ = self._scan(tab)
            if wall is WallKind.SMS_PHONE:
                _log.info("signup %s: code rejected, trying next number",
                          attempt.service)
                continue
            self._notify(f"sms gate solved — {attempt.service}",
                         f"verification code received on temp number "
                         f"{masked}; continuing the signup.", attempt.id)
            return
        # genuinely stuck: every temp source exhausted
        await self._hand_to_owner(
            attempt, WallKind.SMS_PHONE,
            f"{evidence} — temp-number cascade exhausted "
            f"({'; '.join(tried) or 'no numbers'})",
            page_url,
            needed="the site demands an SMS code but no temp-number "
                   "source delivered one. Either wait and resume (I'll "
                   "retry the cascade), or give me a number to use and "
                   "I'll handle the code from there.")

    @staticmethod
    def _fill_first(tab: Any, labels: tuple[str, ...], value: str) -> bool:
        """Fill the first label that resolves. Returns False if none did."""
        for label in labels:
            try:
                tab.fill(label, value)
                return True
            except Exception:  # noqa: BLE001
                continue
        return False

    async def _handle_rate_limit_wall(
        self,
        attempt: SignupAttempt,
        tab: Any,
        evidence: str,
        page_url: str,
        *,
        waits: tuple[float, ...],
    ) -> None:
        """Wait out a rate limit with backoff, then re-scan.

        Returns None when the limit clears so the drive continues; hands
        to the owner only when all waits are exhausted.
        """
        for i, wait_s in enumerate(waits):
            _log.info("signup %s: rate-limited, waiting %.0fs (try %d/%d)",
                      attempt.service, wait_s, i + 1, len(waits))
            self._notify(f"rate limit — {attempt.service}",
                         f"the site is rate-limiting signups; waiting "
                         f"{wait_s:.0f}s before retrying "
                         f"(attempt {i + 1}/{len(waits)}).", attempt.id)
            await asyncio.sleep(wait_s)
            try:
                tab.navigate(page_url)
            except Exception:  # noqa: BLE001
                pass
            wall, _ = self._scan(tab)
            if wall is not WallKind.RATE_LIMIT:
                _log.info("signup %s: rate limit cleared", attempt.service)
                return
        await self._hand_to_owner(
            attempt, WallKind.RATE_LIMIT,
            f"{evidence} — still limited after {len(waits)} backoff waits",
            page_url,
            needed="the site is still rate-limiting signups. Wait a few "
                   "hours, then resume — I'll retry the attempt.")

    async def _hand_to_owner(
        self,
        attempt: SignupAttempt,
        wall: WallKind,
        detail: str,
        page_url: str,
        needed: str,
    ) -> "NoReturn":
        """Genuinely stuck: persist, checkpoint, notify with exact status,
        raise.  Reached ONLY after the solve cascade is exhausted."""
        self.attempts.set_wall(attempt.id, wall, detail)
        kind = _WALL_CHECKPOINT_KIND.get(wall, CheckpointKind.MANUAL_STEP)
        status = self._owner_handoff_message(attempt, wall, detail, needed)
        cp = self.creator.checkpoints.create(
            kind,
            title=f"Signup needs you — {wall.value} ({attempt.service})",
            instructions=status,
            service=attempt.service,
            resume_state={
                "flow": "signup_wall",
                "attempt_id": attempt.id,
                "wall": wall.value,
                "page_url": page_url,
            },
        )
        self.attempts.transition(
            attempt.id, SignupStage.STOPPED_AT_WALL,
            note=f"{wall.value}: {detail} (checkpoint {cp.id})")
        self._notify(
            f"⏸️ signup needs you — {attempt.service} ({wall.value})",
            f"{status}\n\nResume after acting: checkpoint {cp.id}",
            attempt.id)
        raise AccountCheckpointPending(cp)

    def _owner_handoff_message(self, attempt: SignupAttempt, wall: WallKind,
                               detail: str, needed: str) -> str:
        """Exact status for the owner: what's done, what's stuck, what's
        needed.  This is the ONLY owner ping in the flow."""
        done = {
            SignupStage.STARTED: "attempt recorded",
            SignupStage.FORM_FILLED: "signup form filled + submitted",
            SignupStage.EMAIL_SENT: "form submitted, verification email awaited",
            SignupStage.VERIFIED: "email verified",
        }.get(attempt.stage, attempt.stage.value)
        return (
            f"I drove the {attempt.service} signup as far as I could and "
            f"got genuinely stuck.\n\n"
            f"Completed: {done}\n"
            f"Persona: {attempt.persona_name} (disposable, "
            f"{attempt.persona_id})\n"
            f"Stuck at: {wall.value} — {detail}\n\n"
            f"What I need from you: {needed}")

    @staticmethod
    def _default_needed(wall: WallKind, page_url: str) -> str:
        if wall is WallKind.MANUAL_REVIEW:
            return ("the site put the signup in a manual review queue — "
                    "nothing to automate. Wait for their decision, then "
                    "tell me the outcome to finalize or abandon the attempt.")
        if wall is WallKind.REAL_ID:
            return ("the site demands a government ID / document upload. I "
                    "never fabricate identity documents — complete this "
                    "step yourself if you want the account, then resume.")
        if wall is WallKind.AGE_GATE:
            return ("the site's age gate blocked the disposable persona. "
                    "Confirm you want to continue with your own details, "
                    "or abandon this attempt.")
        return (f"open {page_url or 'the signup page'} and complete the "
                f"step yourself, then resume the checkpoint to continue.")

    def _notify(self, title: str, body: str, attempt_id: str) -> None:
        if self.notify is None:
            return
        try:
            self.notify(title, body, attempt_id)
        except Exception as exc:  # noqa: BLE001
            _log.warning("signup notify hook failed: %s", exc)


def render_attempt_summary(attempt: SignupAttempt) -> str:
    """One-block human-readable summary of an attempt."""
    lines = [
        f"signup attempt {attempt.id}",
        f"  service: {attempt.service}",
        f"  persona: {attempt.persona_name} ({attempt.persona_id})",
        f"  stage:   {attempt.stage.value}",
    ]
    if attempt.wall_kind is not WallKind.NONE:
        lines.append(f"  wall:    {attempt.wall_kind.value} — "
                     f"{attempt.wall_detail[:160]}")
    if attempt.credential_ref:
        lines.append(f"  vault:   credential {attempt.credential_ref}")
    if attempt.note:
        lines.append(f"  note:    {attempt.note[:200]}")
    return "\n".join(lines)
