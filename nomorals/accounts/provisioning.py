"""Self-provisioning credentials — missing key → offer → drive → vault.

The owner's doctrine: integrations must not be "dead commands waiting
for keys." When a feature fails because a credential was never
configured (a :class:`~nomorals.core.errors.MissingCredentialError`),
the system should offer to go and get the key itself — on the OWNER's
own account — instead of dead-ending.

This module is the service-general machinery:

* :data:`PROVISION_REGISTRY` — one declarative entry per service: the
  env/config keys the feature reads, where keys are created, the login
  URL, required scopes, license/consent steps, a key validator id and
  a page-flow id. Adding a service is a registry entry, not plumbing.
* :class:`ProvisionDriver` — the drive, built on the real accounts
  organs: :func:`browser_login.ensure_login` for the login (vault
  password, session restore, solver-first CAPTCHA), a per-service page
  flow for key creation, validation BEFORE the vault write counts as
  success, and the vault→env bridge that makes the key readable where
  the feature reads it (no restart where the seam reads env per call).
* Account-creation fallback: when no login exists and none is handed
  over, the driver falls into :class:`creator.AccountCreator` — the
  accounts-policy path (one account per service, human verification
  pauses on a persisted checkpoint). It never creates a duplicate
  when a login already exists. Identity resolution (owner's rule,
  2026-10-20 — "not limited to my provided identity"): a RECORDED
  owner identity is preferred when one exists; when none is
  recorded, the driver resolves the service's self-generated
  :class:`~nomorals.accounts.identity_bank.IdentityBank` persona
  (one per service, reused across retries) and mints its real
  disposable email through the same temp-mail path the trial signup
  driver uses — creation proceeds instead of dead-ending, and the
  result/audit state plainly that a generated identity was used.
* Audit: every step is logged (service, step, outcome) through the
  module logger and the driver's ``audit_trail``. Secret VALUES never
  appear in any log line, result, or audit record — only their
  presence, length class, and where they live.

Layering: accounts is L2. The browser tab factory, the CAPTCHA solver,
the HTTP getter and the creator factory are all INJECTED (the same
pattern as ``login_with_vault``'s ``open_tab``); production wiring
lives in the chat runtime (L5) and the account CLI.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Mapping

from ..core.errors import MissingCredentialError, NotFound, cause_chain
from ..core.logging_setup import get_logger
from .browser_login import (
    LoginCaptchaRequired,
    LoginConfig,
    LoginFailed,
    ensure_login,
)
from .manager import AccountManager
from .sessions import SessionManager
from .vault import CredentialVault

__all__ = [
    "PROVISION_REGISTRY",
    "LicenseStep",
    "ProvisionDriver",
    "ProvisionOutcome",
    "ProvisionResult",
    "ProvisionService",
    "ValidationResult",
    "bridge_vault_credentials",
    "classify_missing_credential",
    "get_provision_bundle",
    "get_provision_service",
    "provisioned_status",
    "register_provision_bundle",
    "register_provision_service",
    "resolve_provision_services",
    "validate_service_token",
]

_log = get_logger(__name__)

#: kv_store key prefix marking an account this driver asked the creator
#: to make, so the post-checkpoint drive can report "account CREATED"
#: (vs a key fetched on an account the owner already had).
_CREATED_MARKER_PREFIX = "provisioning.created_pending."

#: Strings that only ever appear in PRE-PURGE (2026-10-23) pending
#: output. A stored checkpoint carrying any of them in its title or
#: instructions is a stale echo of the purged generators, not a live
#: wall — it is invalidated on sight (see _invalidate_stale_pending)
#: instead of being re-reported. The kind alone never decides: a
#: genuine MANUAL_REVIEW / REAL_ID / AGE_GATE / RATE_LIMIT wall keeps
#: kind MANUAL_STEP with clean single-action text and must keep
#: reporting.
_PURGED_PENDING_MARKERS = (
    "create the account yourself",
    "identity: self-generated",
    "paused at a human step",
    "finish it, then run",
    "manual_step",
)


def _contains_purged_text(text: str) -> bool:
    """True when `text` carries pre-purge pending output."""
    lowered = (text or "").lower()
    if any(marker in lowered for marker in _PURGED_PENDING_MARKERS):
        return True
    # A numbered chore list (the pre-purge hand-back shape); genuine
    # post-purge checkpoints carry ONE plain action, never a list.
    return "\n1." in (text or "") and "\n2." in (text or "")

#: Temp-mail provider used to mint a generated persona's real address.
#: Matches the trial signup driver's default (``SignupDriver``'s
#: ``temp_email_provider``) — the mint itself is the creator's
#: ``create_email_account``, the exact call ``SignupDriver._ensure_email``
#: makes; nothing about the address is constructed locally.
_GENERATED_EMAIL_PROVIDER = "tempmail"


# ── registry ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LicenseStep:
    """A license/consent page a service requires before an API key is
    usable for a given model/product (e.g. gated-model agreements)."""

    url: str
    #: what the license gates (model id / product) — named in reports.
    subject: str
    #: button/link texts that count as "accept" on the page.
    agree_texts: tuple[str, ...] = ("Agree and access repository",
                                   "Agree and access", "Agree", "Accept")


@dataclass(frozen=True)
class ProvisionService:
    """One provisionable service. Declarative: a new service is a new
    entry here plus a page-flow function registered in ``_PAGE_FLOWS``
    and (optionally) a validator in ``_VALIDATORS``."""

    #: registry id — also the chat word (`provision <id>`).
    id: str
    display: str
    aliases: tuple[str, ...] = ()
    #: vault service name under which the login + the key live.
    vault_service: str = ""
    #: env/config keys the feature reads, in bridge order.
    env_keys: tuple[str, ...] = ()
    login_url: str = ""
    #: page where API keys/tokens are created.
    token_url: str = ""
    signup_url: str = ""
    #: scopes/role requested for the created key (least privilege).
    scopes: tuple[str, ...] = ()
    #: validator id (see ``_VALIDATORS``); "" = no remote validation.
    validator: str = ""
    #: page-flow id (see ``_PAGE_FLOWS``).
    flow: str = ""
    #: signup-flow id (see ``_SIGNUP_FLOWS``) — the automated
    #: account-creation drive for this service's signup page. "" =
    #: no NAMED flow: a service with a ``signup_url`` is still
    #: driven (the generic form flow); only a service with neither
    #: a flow nor a URL reaches the creator's checkpoint path.
    signup_flow: str = ""
    licenses: tuple[LicenseStep, ...] = ()
    #: env var that, when set, names an account the owner is KNOWN to
    #: already hold at this service (so creation is never attempted —
    #: the login handoff is the right path).
    known_account_env: str = ""

    def __post_init__(self) -> None:
        if not self.vault_service:
            object.__setattr__(self, "vault_service", self.id)


PROVISION_REGISTRY: dict[str, ProvisionService] = {}


def register_provision_service(service: ProvisionService) -> ProvisionService:
    """Register (or replace) a provisioning entry. Later registrations
    win, so tests and deployments can extend the registry without
    touching this module's plumbing."""
    PROVISION_REGISTRY[service.id] = service
    return service


def _normalize_service_key(raw: str) -> str:
    """Canonical service-key form: lowercase, with underscores,
    spaces and hyphens all equivalent (collapsed to single hyphens).

    ``odds_api``, ``Odds API``, ``odds-api`` and ``the_odds_api`` are
    the same word to the owner, so they are the same key here — an
    exact-string gate on the registry id was a defect (owner-flagged
    2026-10-23: ``/provision odds_api`` hard-failed while
    ``/provision odds-api`` worked)."""
    key = re.sub(r"[\s_]+", "-", (raw or "").strip().lower())
    return re.sub(r"-{2,}", "-", key).strip("-")


def get_provision_service(id_or_alias: str) -> ProvisionService | None:
    """Resolve a registry id or alias, forgivingly: case, underscores,
    hyphens and spaces are all equivalent (see
    :func:`_normalize_service_key`)."""
    key = _normalize_service_key(id_or_alias)
    if not key:
        return None
    for svc in PROVISION_REGISTRY.values():
        if key == _normalize_service_key(svc.id):
            return svc
        if key in (_normalize_service_key(a) for a in svc.aliases):
            return svc
    return None


#: Named bundles: one chat word → several services driven in turn
#: (e.g. ``desk`` = the betting desk's keyed APIs). Bundles are
#: registered by the layer that owns the grouping (the desk
#: registers ``desk``); this module only keeps the book.
PROVISION_BUNDLES: dict[str, tuple[str, ...]] = {}


def register_provision_bundle(name: str,
                              member_ids: tuple[str, ...]) -> None:
    """Register (or replace) a bundle: a word that provisions
    several services in turn."""
    PROVISION_BUNDLES[_normalize_service_key(name)] = tuple(member_ids)


def get_provision_bundle(id_or_alias: str) -> list[ProvisionService] | None:
    """Resolve a bundle word to its member services, or None."""
    members = PROVISION_BUNDLES.get(_normalize_service_key(id_or_alias))
    if not members:
        return None
    resolved = [svc for svc in
                (get_provision_service(m) for m in members)
                if svc is not None]
    return resolved or None


def resolve_provision_services(id_or_alias: str) -> list[ProvisionService]:
    """Everything a provision request names: the bundle's members
    when the word is a bundle, else the single service, else []."""
    bundle = get_provision_bundle(id_or_alias)
    if bundle:
        return bundle
    svc = get_provision_service(id_or_alias)
    return [svc] if svc is not None else []


register_provision_service(ProvisionService(
    id="huggingface",
    display="Hugging Face",
    aliases=("hf",),
    vault_service="huggingface",
    env_keys=("HF_TOKEN",),
    login_url="https://huggingface.co/login",
    token_url="https://huggingface.co/settings/tokens",
    signup_url="https://huggingface.co/join",
    scopes=("read",),
    validator="hf_whoami",
    flow="hf_token_v1",
    signup_flow="form_signup_v1",
    licenses=(LicenseStep(
        url="https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev",
        subject="black-forest-labs/FLUX.1-Kontext-dev",
    ),),
    known_account_env="NM_HF_ACCOUNT",
))


# ── outcomes ─────────────────────────────────────────────────────────


class ProvisionOutcome(StrEnum):
    PROVISIONED = "provisioned"
    ALREADY_PROVISIONED = "already_provisioned"
    NEEDS_OWNER_LOGIN = "needs_owner_login"
    #: retained for result compatibility — the driver no longer
    #: dead-ends on a missing owner identity (owner's rule 2026-10-20:
    #: creation falls back to a self-generated identity instead).
    NEEDS_OWNER_IDENTITY = "needs_owner_identity"
    IDENTITY_MINT_FAILED = "identity_mint_failed"
    CREATION_CHECKPOINT = "creation_checkpoint"
    LOGIN_FAILED = "login_failed"
    VERIFICATION_BLOCKED = "verification_blocked"
    PAGE_CHANGED = "page_changed"
    KEY_REJECTED = "key_rejected"
    VALIDATION_INCONCLUSIVE = "validation_inconclusive"
    SERVICE_UNKNOWN = "service_unknown"


@dataclass
class ProvisionResult:
    outcome: ProvisionOutcome
    service_id: str
    #: owner-safe summary. NEVER contains a secret value.
    message: str
    #: True when the account itself was created by this provisioning
    #: lineage (vs a key fetched on an existing account).
    account_created: bool = False
    username: str = ""
    #: env keys populated by the bridge during this drive.
    env_keys_applied: list[str] = field(default_factory=list)
    #: creation-checkpoint details when outcome is CREATION_CHECKPOINT.
    checkpoint: dict[str, Any] | None = None
    #: non-fatal notes (license step results, role selection, ...).
    warnings: list[str] = field(default_factory=list)
    #: the flow/login step that failed, for PAGE_CHANGED/LOGIN_FAILED.
    step: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome in (ProvisionOutcome.PROVISIONED,
                                ProvisionOutcome.ALREADY_PROVISIONED)


# ── classification ───────────────────────────────────────────────────


def classify_missing_credential(exc: BaseException | None,
                                ) -> MissingCredentialError | None:
    """Return the MissingCredentialError in ``exc``'s cause chain, or
    None. This is the honest classifier: features raise the typed error
    at the seam (see ``media_edit``'s HF backend); string matching is
    never used to decide a credential is missing."""
    if exc is None:
        return None
    for link in cause_chain(exc):
        if isinstance(link, MissingCredentialError):
            return link
    return None


# ── validation ───────────────────────────────────────────────────────

#: Injectable HTTP getter: (url, headers) -> (status, body_text).
HttpGet = Callable[[str, dict[str, str]], tuple[int, str]]


def _urllib_get(url: str, headers: dict[str, str]) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:  # noqa: BLE001 - status is data
        return exc.code, exc.read().decode("utf-8", "replace")


@dataclass
class ValidationResult:
    status: str  # "ok" | "rejected" | "inconclusive"
    detail: str = ""


def _validate_hf_whoami(token: str, http_get: HttpGet) -> ValidationResult:
    """HF whoami: 200 + account name for a live token; 401/403 for a
    dead one; anything else (network, 5xx) is inconclusive — never
    counted as success, never blamed on the key."""
    try:
        status, body = http_get(
            "https://huggingface.co/api/whoami-v2",
            {"Authorization": f"Bearer {token}"})
    except Exception as exc:  # noqa: BLE001 - transport is inconclusive
        return ValidationResult("inconclusive", f"validation request failed: {exc}")
    if status == 200:
        name = ""
        try:
            import json
            name = str(json.loads(body).get("name") or "")
        except Exception:  # noqa: BLE001 - name is cosmetic
            name = ""
        return ValidationResult("ok", f"token accepted for account {name}".strip())
    if status in (401, 403):
        return ValidationResult("rejected", f"whoami answered HTTP {status}")
    return ValidationResult("inconclusive", f"whoami answered HTTP {status}")


_VALIDATORS: dict[str, Callable[[str, HttpGet], ValidationResult]] = {
    "hf_whoami": _validate_hf_whoami,
}


def validate_service_token(service: ProvisionService, token: str,
                           http_get: HttpGet | None = None,
                           ) -> ValidationResult:
    """Validate a freshly created key with the service's validator."""
    fn = _VALIDATORS.get(service.validator or "")
    if fn is None:
        return ValidationResult(
            "inconclusive",
            f"no validator registered for service {service.id!r}")
    return fn(token, http_get or _urllib_get)


# ── vault → env bridge ───────────────────────────────────────────────


def bridge_vault_credentials(
    vault: CredentialVault,
    *,
    environ: Mapping[str, str] | Any = None,
    services: Mapping[str, ProvisionService] | None = None,
) -> list[str]:
    """Populate registry env keys from vaulted API keys.

    For every registered service whose vault holds an ``api_key``
    credential, set each of its env keys that is not already set —
    a real environment value always wins (the owner's explicit config
    outranks a provisioned fallback). Returns the keys populated.
    This is the bridge that lets a provisioned key take effect without
    a restart wherever the feature reads env at call time (the HF
    backend constructs per call), and at startup for everything else.
    """
    env = os.environ if environ is None else environ
    applied: list[str] = []
    for svc in (services or PROVISION_REGISTRY).values():
        creds = vault.list_all(service=svc.vault_service, active_only=True)
        key_cred = next(
            (c for c in creds if c.credential_type == "api_key"), None)
        if key_cred is None:
            continue
        secret = vault.get(svc.vault_service, key_cred.username,
                           mark_used=False).password
        if not secret:
            continue
        for env_key in svc.env_keys:
            if not env.get(env_key):
                env[env_key] = secret  # type: ignore[index]
                applied.append(env_key)
                _log.info("provisioning bridge: %s populated from vault "
                          "(service %s)", env_key, svc.id)
    return applied


def provisioned_status(
    vault: CredentialVault,
    *,
    environ: Mapping[str, str] | None = None,
    services: Mapping[str, ProvisionService] | None = None,
) -> list[dict[str, Any]]:
    """Per-service provisioning state for status boards. Metadata only:
    presence, types, usernames — never values."""
    env = os.environ if environ is None else environ
    out: list[dict[str, Any]] = []
    for svc in (services or PROVISION_REGISTRY).values():
        creds = vault.list_all(service=svc.vault_service, active_only=True)
        out.append({
            "service": svc.id,
            "display": svc.display,
            "env_keys": list(svc.env_keys),
            "env_set": [k for k in svc.env_keys if env.get(k)],
            "has_key": any(c.credential_type == "api_key" for c in creds),
            "has_login": any(c.credential_type == "password" for c in creds),
            "login_usernames": [c.username for c in creds
                                if c.credential_type == "password"],
        })
    return out


# ── page flows ───────────────────────────────────────────────────────


class FlowStepError(Exception):
    """A page-flow step did not find the page shape it depends on.

    Carries the exact step name so the failure is reported AT the step
    (a changed page fails loudly, never silently "succeeds")."""

    def __init__(self, step: str, detail: str = "") -> None:
        super().__init__(f"flow step {step!r} failed: {detail}".strip())
        self.step = step
        self.detail = detail


class FlowBlocked(Exception):
    """The flow is blocked on a human wall (CAPTCHA / verification)
    that the solver could not clear."""


def _tab_text(tab: Any, max_chars: int = 20000) -> str:
    try:
        return str(tab.text(max_chars=max_chars).get("text") or "")
    except Exception:  # noqa: BLE001 - text is best-effort
        return ""


def _click_text(tab: Any, *texts: str) -> str:
    """Click the first control whose label matches one of ``texts``.

    Tries the tab's own click (selector/label based), then a JS
    text-match click. Returns the label used; raises FlowStepError
    when nothing matched."""
    for text in texts:
        click = getattr(tab, "click", None)
        if callable(click):
            try:
                click(text)
                return text
            except Exception:  # noqa: BLE001 - try the next shape
                pass
        evaluate = getattr(tab, "evaluate", None)
        if callable(evaluate):
            js = (
                "(() => { const els = [...document.querySelectorAll("
                "'button, a, [role=button], input[type=submit]')];"
                " const el = els.find(e => (e.innerText || e.value || '')"
                ".trim().toLowerCase().includes(%r));"
                " if (el) { el.click(); return true; } return false; })()"
                % text.lower())
            try:
                if evaluate(js):
                    return text
            except Exception:  # noqa: BLE001 - try the next label
                pass
    raise FlowStepError("click", f"no clickable control matching {texts!r}")


def _detect_captchas(tab: Any) -> list[dict[str, Any]]:
    checker = getattr(tab, "check_captcha", None)
    if checker is None:
        return []
    try:
        return list(checker().get("challenges") or [])
    except Exception:  # noqa: BLE001 - detection best-effort
        return []


_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")


def _extract_hf_token(tab: Any) -> str:
    """Read the just-created token off the HF page. The token is shown
    once, in a readonly input (or a code element). Never logged."""
    evaluate = getattr(tab, "evaluate", None)
    if callable(evaluate):
        js = (
            "(() => { const vals = [...document.querySelectorAll"
            "('input, code, textarea')].map(e => e.value || e.textContent"
            " || ''); const hit = vals.find(v => /hf_[A-Za-z0-9]{20,}/"
            ".test(v || '')); return hit ? (hit.match"
            "(/hf_[A-Za-z0-9]{20,}/) || [null])[0] : null; })()")
        try:
            found = evaluate(js)
            if found:
                return str(found)
        except Exception:  # noqa: BLE001 - fall through to text scan
            pass
    match = _HF_TOKEN_RE.search(_tab_text(tab, max_chars=40000))
    return match.group(0) if match else ""


def hf_token_flow(tab: Any, service: ProvisionService, *,
                  token_name: str,
                  captcha_solver: Callable[[dict[str, Any]], dict[str, Any]]
                  | None = None) -> str:
    """Create a read token on huggingface.co/settings/tokens.

    Steps, each failing loudly by name: open the tokens page (a login
    wall here means the session restore failed), open the create
    dialog, name the token, pick the Read role when a role control is
    offered, create, read the once-shown token value off the page.
    """
    tab.navigate(service.token_url)
    text = _tab_text(tab)
    url = str(getattr(tab, "url", "") or "")
    if "/login" in url or ("Log In" in text and "password" in text.lower()):
        raise FlowStepError(
            "open_tokens_page",
            "the tokens page answered with a login wall — the browser "
            "session was not logged in")
    for ch in _detect_captchas(tab):
        if captcha_solver is None:
            raise FlowBlocked(
                f"CAPTCHA ({ch.get('kind')}) on the tokens page")
        result = captcha_solver(ch)
        if not result.get("ok"):
            raise FlowBlocked(
                f"CAPTCHA ({ch.get('kind')}) on the tokens page and the "
                f"solver could not clear it: {result.get('detail', '')}")
    try:
        _click_text(tab, "Create new token", "New token", "Create token")
    except FlowStepError as exc:
        raise FlowStepError(
            "open_create_dialog", "no 'create token' control found") from exc
    filled = ""
    for candidate in ("name", "token-name", "token_name", "new-token-name"):
        try:
            tab.fill(candidate, token_name)
            filled = candidate
            break
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
    if not filled:
        raise FlowStepError(
            "name_token", "no token-name field in the create dialog")
    # Role: HF's classic dialog defaults new tokens to the Read role;
    # when a role control is offered, choose the least-privilege scope
    # the registry asks for. Absence of the control is not fatal.
    role = (service.scopes[0] if service.scopes else "read")
    try:
        _click_text(tab, role.capitalize(), role)
    except FlowStepError:
        pass
    try:
        _click_text(tab, "Create token", "Create", "Generate token")
    except FlowStepError as exc:
        try:
            tab.submit("")
        except Exception as submit_exc:  # noqa: BLE001
            raise FlowStepError(
                "create_token", "no create control and submit failed"
            ) from submit_exc
    token = _extract_hf_token(tab)
    if not token:
        raise FlowStepError(
            "read_token_value",
            "the created token value was not found on the page — it is "
            "shown only once, so nothing was stored")
    return token


#: Page-flow implementations by registry flow id. A flow takes the tab,
#: the service entry, and keyword args (token_name, captcha_solver) and
#: returns the created secret VALUE (handled as a secret everywhere).
_PAGE_FLOWS: dict[str, Callable[..., str]] = {
    "hf_token_v1": hf_token_flow,
}


# ── signup flows (automated account creation) ────────────────────────


class SignupBlocked(Exception):
    """The automated signup hit a wall only the owner can clear.

    Carries the checkpoint shape the driver reports: a ``kind``, a
    one-line ``title``, and ``action`` — the ONE smallest step the
    owner takes (never a chore list). ``submitted`` says whether the
    form had already been submitted when the wall hit (the account
    may exist server-side, so the driver vaults the credentials it
    used and says the login is saved)."""

    def __init__(self, kind: str, title: str, action: str, *,
                 submitted: bool = False, detail: str = "") -> None:
        super().__init__(f"{title}: {action}")
        self.kind = kind
        self.title = title
        self.action = action
        self.submitted = submitted
        self.detail = detail


class SignupAccountExists(Exception):
    """The signup page says an account already exists for the owner's
    email — creation must not be retried (one account per service);
    the right path is the login handoff."""


@dataclass
class SignupFlowResult:
    """Outcome of an automated signup attempt.

    ``created`` — the account exists and the credentials are usable
    now. ``verification_pending`` — the form was submitted and the
    service sent its verification to the signup inbox (the owner's
    real inbox under a recorded identity — the one step Devon cannot
    do for him; the minted disposable inbox under a generated
    identity, which Devon can read). The credentials are vaulted and
    the drive continues on the next run."""

    status: str  # "created" | "verification_pending"
    detail: str = ""


def _fill_first(tab: Any, candidates: tuple[str, ...], value: str, *,
                step: str, required: bool) -> str:
    """Fill the first field on the page matching one of ``candidates``.

    Returns the field name used ("" when an optional field is simply
    absent — forms differ; absence of an optional field is not a
    failure). A required field with no match raises FlowStepError at
    the named step: that exact step is what the page does not offer."""
    for candidate in candidates:
        try:
            tab.fill(candidate, value)
            return candidate
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
    if required:
        raise FlowStepError(
            step, f"no field matching {candidates!r} on the signup page")
    return ""


#: post-submit page-text markers, checked in this order after walls.
_SIGNUP_EXISTS_MARKERS = (
    "already exists", "already registered", "already in use",
    "already taken", "an account with this email",
    "account already exists",
)
_SIGNUP_VERIFY_MARKERS = (
    "check your email", "check your inbox", "verification email",
    "confirm your email", "verify your email", "email sent",
    "sent you an email", "we've sent", "we have sent",
    "activate your account", "click the link in",
)
_SIGNUP_SUCCESS_MARKERS = (
    "welcome", "account created", "account is ready",
    "successfully registered", "registration successful",
    "registration complete", "signup complete", "sign-up complete",
    "your account has been created", "you are logged in",
    "you are now logged in",
)


def _solve_or_block(tab: Any, service: ProvisionService, *,
                    captcha_solver: Callable[[dict[str, Any]],
                                             dict[str, Any]] | None,
                    submitted: bool) -> None:
    """Clear detected CAPTCHAs through the solver; raise SignupBlocked
    (a single owner action) when the solver is absent or fails."""
    for ch in _detect_captchas(tab):
        kind = ch.get("kind", "captcha")
        if captcha_solver is None:
            raise SignupBlocked(
                "captcha",
                f"CAPTCHA ({kind}) on the {service.display} signup page",
                "one step: beat the CAPTCHA in the drive's live "
                "browser view (attached to this chat) — tell me when "
                "it's done and I'll refill and resubmit the form.",
                submitted=submitted)
        result = captcha_solver(ch)
        if not result.get("ok"):
            raise SignupBlocked(
                "captcha",
                f"CAPTCHA ({kind}) on the {service.display} signup page",
                "one step: beat the CAPTCHA in the drive's live "
                "browser view (attached to this chat) — tell me when "
                "it's done and I'll refill and resubmit the form.",
                submitted=submitted,
                detail=str(result.get("detail", "")))


def _wall_block(service: ProvisionService, kind: Any,
                evidence: str) -> SignupBlocked:
    """Map a classified post-submit wall to its single owner action."""
    from .signup_driver import WallKind

    page = service.signup_url or service.login_url
    if kind is WallKind.SMS_PHONE:
        return SignupBlocked(
            "phone_2fa", f"{service.display} insists on phone "
            "verification before creating the account",
            "one step: tell me a number that can receive the code "
            "(a virtual number is fine), or finish the phone step "
            "in the drive's live browser view — then tell me, and "
            "I'll take the signup from there.",
            detail=evidence)
    if kind is WallKind.MANUAL_REVIEW:
        return SignupBlocked(
            "manual_step", f"{service.display} took the signup into "
            "manual review",
            "one step: watch your inbox for their decision — once "
            "approved, tell me and I'll log in and fetch the key "
            "(your login is already saved).",
            submitted=True, detail=evidence)
    if kind is WallKind.REAL_ID:
        return SignupBlocked(
            "manual_step", f"{service.display} demands ID "
            "verification to open the account",
            f"one step: complete the ID check at {page} — tell me "
            "when it's done and I'll take it from there.",
            detail=evidence)
    if kind is WallKind.AGE_GATE:
        return SignupBlocked(
            "manual_step", f"{service.display} age-gates this signup",
            f"one step: confirm the account-holder details at "
            f"{page} — tell me when it's done and I'll take it "
            "from there.",
            detail=evidence)
    if kind is WallKind.RATE_LIMIT:
        return SignupBlocked(
            "manual_step", f"{service.display} rate-limited the "
            "signup attempt",
            "one step: wait a while — then tell me and I'll retry "
            "the signup; nothing else is needed from you.",
            detail=evidence)
    return SignupBlocked(
        "captcha", f"CAPTCHA on the {service.display} signup page",
        "one step: beat the CAPTCHA in the drive's live browser "
        "view (attached to this chat) — tell me when it's done "
        "and I'll refill and resubmit the form.",
        detail=evidence)


def form_signup_flow(tab: Any, service: ProvisionService, *,
                     name: str, email: str, username: str,
                     password: str,
                     captcha_solver: Callable[[dict[str, Any]],
                                              dict[str, Any]] | None = None,
                     ) -> SignupFlowResult:
    """Drive a standard web signup form on the resolved identity
    (the recorded owner identity, or the service's self-generated
    persona when none is recorded).

    Navigates to the service's signup page, clears CAPTCHAs through
    the solver, fills the form (email always; username, name and
    password where the form offers them — candidate field names, the
    ``hf_token_flow`` discipline), submits, and classifies the
    outcome page. Every mismatch fails BY STEP NAME (FlowStepError),
    every genuine wall raises SignupBlocked with the single owner
    action, and an "account already exists" answer raises
    SignupAccountExists — nothing is ever silently "successful"."""
    from .signup_driver import WallKind, classify_wall

    if not service.signup_url:
        raise FlowStepError(
            "open_signup_page",
            f"no signup URL registered for {service.display}")
    tab.navigate(service.signup_url)
    _solve_or_block(tab, service, captcha_solver=captcha_solver,
                    submitted=False)
    _fill_first(tab, ("email", "user_email", "signup_email",
                      "your-email", "mail", "user[email]"),
                email, step="fill_email", required=True)
    _fill_first(tab, ("username", "user_name", "handle", "login"),
                username, step="fill_username", required=False)
    if name:
        _fill_first(tab, ("name", "full_name", "fullname",
                          "display_name", "first_name"),
                    name, step="fill_name", required=False)
    _fill_first(tab, ("password", "user_password", "passwd", "pass"),
                password, step="fill_password", required=False)
    _fill_first(tab, ("password_confirmation", "password_confirm",
                      "confirm_password", "password2",
                      "repeat_password"),
                password, step="fill_password_confirm", required=False)
    try:
        _click_text(tab, "Sign Up", "Sign up", "Create account",
                    "Create Account", "Get Access", "Register", "Join",
                    "Continue", "Submit")
    except FlowStepError:
        try:
            tab.submit("")
        except Exception as exc:  # noqa: BLE001
            raise FlowStepError(
                "submit_signup",
                "no submit control matched and form submit failed"
            ) from exc
    _solve_or_block(tab, service, captcha_solver=captcha_solver,
                    submitted=True)
    text = _tab_text(tab, max_chars=40000).lower()
    wall, evidence = classify_wall(page_text=text)
    if wall is WallKind.CAPTCHA:
        _solve_or_block(tab, service, captcha_solver=captcha_solver,
                        submitted=True)
        text = _tab_text(tab, max_chars=40000).lower()
        wall, evidence = classify_wall(page_text=text)
    if wall is not WallKind.NONE and wall is not WallKind.CAPTCHA:
        raise _wall_block(service, wall, evidence)
    if any(m in text for m in _SIGNUP_EXISTS_MARKERS):
        raise SignupAccountExists(
            f"{service.display} says an account already exists for "
            f"{email}")
    if any(m in text for m in _SIGNUP_VERIFY_MARKERS):
        return SignupFlowResult(
            "verification_pending",
            "the service sent its verification to the signup inbox")
    if any(m in text for m in _SIGNUP_SUCCESS_MARKERS):
        return SignupFlowResult("created", "signup submitted cleanly")
    raise FlowStepError(
        "confirm_signup",
        "after submitting, the page matched no known outcome (no "
        "verification prompt, no success signal, no wall) — the "
        "post-submit step of this signup does not match the page, "
        "so it is the step that cannot be automated as-is")


#: Signup-flow implementations by registry signup-flow id. A flow
#: takes the tab, the service entry, and the resolved identity fields
#: (name/email/username/password + captcha_solver) and returns a
#: SignupFlowResult; the password is a secret everywhere.
_SIGNUP_FLOWS: dict[str, Callable[..., SignupFlowResult]] = {
    "form_signup_v1": form_signup_flow,
}


def _run_coro(coro: Any) -> Any:
    """Run a coroutine from sync code (the accounts creator is async;
    the chat runtime is not)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - ferried to caller
            box["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ── the driver ───────────────────────────────────────────────────────

#: Injected tab factory: service vault-name -> a driven browser tab in
#: that service's session (production: BrowserService rendered tabs —
#: see the CLI's `account login` wiring).
OpenTab = Callable[[str], Any]


class ProvisionDriver:
    """Drives one provisioning run for one service.

    All live surfaces are injected: ``open_tab`` (browser), ``http_get``
    (validation), ``captcha_solver`` (accounts solver shape), and
    ``creator_factory`` (defaults to the real AccountCreator).
    """

    def __init__(
        self,
        vault: CredentialVault,
        *,
        open_tab: OpenTab,
        captcha_solver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        http_get: HttpGet | None = None,
        environ: Mapping[str, str] | Any = None,
        creator_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.vault = vault
        self.manager = AccountManager(vault)
        self.sessions = SessionManager(vault)
        self.open_tab = open_tab
        self.captcha_solver = captcha_solver
        self.http_get = http_get or _urllib_get
        self.environ = os.environ if environ is None else environ
        self._creator_factory = creator_factory
        #: structured, value-free audit trail of this driver's actions.
        self.audit_trail: list[dict[str, Any]] = []

    # -- audit ---------------------------------------------------------

    def _audit(self, step: str, service_id: str, outcome: str,
               **fields: Any) -> None:
        record = {"step": step, "service": service_id,
                  "outcome": outcome, **fields}
        self.audit_trail.append(record)
        _log.info("provision audit: %s", record)

    # -- vault probes (metadata only) ----------------------------------

    def _key_cred(self, svc: ProvisionService) -> Any | None:
        creds = self.vault.list_all(service=svc.vault_service,
                                     active_only=True)
        return next((c for c in creds if c.credential_type == "api_key"),
                    None)

    def _login_cred(self, svc: ProvisionService) -> Any | None:
        """The vault credential that can drive a browser login.

        The creator stores finished accounts as credential_type
        "account"; the browser-login organ accepts only "password".
        A created account IS a password login, so the driver registers
        it as one (same username + secret, type tag only — audited,
        value never logged) instead of dead-ending on the seam."""
        creds = self.vault.list_all(service=svc.vault_service,
                                     active_only=True)
        pw = next((c for c in creds if c.credential_type == "password"),
                  None)
        if pw is not None:
            return pw
        acct = next((c for c in creds if c.credential_type == "account"),
                    None)
        if acct is None:
            return None
        full = self.vault.get(svc.vault_service, acct.username,
                              mark_used=False)
        metadata = dict(full.metadata or {})
        metadata.setdefault("login_url", svc.login_url)
        metadata["registered_from"] = "account"
        stored = self.vault.store(
            svc.vault_service, full.username, full.password,
            credential_type="password",
            tags=sorted(set(full.tags or []) | {"login"}),
            metadata=metadata)
        self._audit("login", svc.id, "registered_account_as_login",
                    username=full.username)
        return stored

    def _known_account(self, svc: ProvisionService) -> str:
        """A username the owner is KNOWN to already hold at this
        service (env hint or an account_ref vault marker) — the signal
        that creation must NOT be attempted."""
        if svc.known_account_env:
            val = str(self.environ.get(svc.known_account_env, "") or "").strip()
            if val:
                return val
        creds = self.vault.list_all(service=svc.vault_service,
                                     active_only=True)
        ref = next((c for c in creds
                    if c.credential_type == "account_ref"), None)
        return ref.username if ref is not None else ""

    # -- main entry ------------------------------------------------------

    def provision(self, service_id: str) -> ProvisionResult:
        svc = get_provision_service(service_id)
        if svc is None:
            return ProvisionResult(
                ProvisionOutcome.SERVICE_UNKNOWN, (service_id or "").strip(),
                f"no provisioning registry entry for {service_id!r} — "
                f"known services: {', '.join(sorted(PROVISION_REGISTRY))}")
        self._audit("start", svc.id, "begin")

        # 1. Already live? env first (owner config wins), then vault.
        if any(self.environ.get(k) for k in svc.env_keys):
            self._audit("probe", svc.id, "already_provisioned_env")
            return ProvisionResult(
                ProvisionOutcome.ALREADY_PROVISIONED, svc.id,
                f"{svc.display} is already configured "
                f"({', '.join(svc.env_keys)} set in the environment) — "
                "nothing to provision.")
        if self._key_cred(svc) is not None:
            applied = bridge_vault_credentials(
                self.vault, environ=self.environ,
                services={svc.id: svc})
            self._audit("probe", svc.id, "already_provisioned_vault",
                        applied=applied)
            return ProvisionResult(
                ProvisionOutcome.ALREADY_PROVISIONED, svc.id,
                f"{svc.display} already has a key in the vault — bridged "
                f"it into {', '.join(applied) or 'the environment'}.",
                env_keys_applied=applied)

        # 2. A login exists → straight to the drive.
        if self._login_cred(svc) is not None:
            return self._drive_with_login(svc, account_created=False)

        # 3. A creation is already stopped for this service (one
        #    owner-only step pending) → report the checkpoint,
        #    never re-create — UNLESS the stored checkpoint is a
        #    stale pre-purge echo: those are cancelled and the drive
        #    proceeds fresh, so a record the purge already deleted
        #    from the generators can never be re-reported forever.
        pending = self._pending_creation(svc)
        while pending is not None and _contains_purged_text(
                f"{pending.get('title', '')}\n"
                f"{pending.get('instructions', '')}"):
            self._invalidate_stale_pending(svc, pending)
            pending = self._pending_creation(svc)
        if pending is not None:
            self._audit("create", svc.id, "checkpoint_pending",
                        checkpoint=pending.get("id", ""))
            note = self._created_marker_note(svc)
            if _contains_purged_text(note):
                # The checkpoint itself is genuine, but the marker
                # note it carries is a pre-purge lecture — drop the
                # note, keep the checkpoint's own single action.
                self._audit("create", svc.id, "stale_marker_note_dropped",
                            checkpoint=pending.get("id", ""))
                note = ""
            return ProvisionResult(
                ProvisionOutcome.CREATION_CHECKPOINT, svc.id,
                f"creating your {svc.display} account stopped at "
                f"one step only you can do: "
                f"{pending.get('title', '')}. "
                f"{pending.get('instructions', '')}"
                + (f"\n{note}" if note else ""),
                checkpoint=pending)

        # 4. The owner is known to hold an account → handoff, never a
        #    silent duplicate.
        known = self._known_account(svc)
        if known:
            self._audit("probe", svc.id, "needs_owner_login",
                        known_account=known)
            return ProvisionResult(
                ProvisionOutcome.NEEDS_OWNER_LOGIN, svc.id,
                f"you already have a {svc.display} account ({known}) — "
                "I won't create a duplicate. Hand me that login and "
                "I'll fetch the key: /provision login "
                f"{svc.id} {known} (your next message is the password, "
                "kept out of history and logs), or on the host: "
                f"nm account store-login --service {svc.vault_service} "
                f"--username {known}",
                username=known)

        # 5. Nothing exists → account creation (accounts-policy path).
        return self._create_then_drive(svc)

    # -- creation fallback ------------------------------------------------

    def _make_creator(self) -> Any:
        if self._creator_factory is not None:
            return self._creator_factory(
                self.vault, db=self.vault.db,
                captcha_solver=self.captcha_solver)
        from .creator import AccountCreator

        return AccountCreator(self.vault, db=self.vault.db,
                              captcha_solver=self.captcha_solver)

    def _pending_creation(self, svc: ProvisionService) -> dict[str, Any] | None:
        """A pending account-creation checkpoint for this service."""
        try:
            from .creator import CheckpointStore

            store = CheckpointStore(self.vault.db)
            for cp in store.list_pending(service=svc.vault_service):
                state = cp.resume_state or {}
                if state.get("flow") == "account_create":
                    return {"id": cp.id, "kind": cp.kind.value,
                            "title": cp.title,
                            "instructions": cp.instructions}
        except Exception:  # noqa: BLE001 - probe is best-effort
            _log.debug("pending-creation probe failed", exc_info=True)
        return None

    def _invalidate_stale_pending(self, svc: ProvisionService,
                                  pending: dict[str, Any]) -> None:
        """Cancel a stale pre-purge pending echo (best-effort, audited).

        The cancel is recorded with a clean note and the service's
        created-marker is dropped with it, so the next step drives
        fresh with a plain-fact note of its own. A cancel failure never
        re-surfaces the stale text: the caller proceeds anyway and the
        next run invalidates again (idempotent).
        """
        checkpoint_id = pending.get("id", "")
        try:
            from .creator import CheckpointStore

            CheckpointStore(self.vault.db).cancel(
                checkpoint_id,
                "invalidated: pre-purge pending echo; drive proceeds fresh")
        except Exception:  # noqa: BLE001 - invalidation is best-effort
            _log.debug("stale-pending cancel failed", exc_info=True)
        try:
            from ..storage.kv import KVStore

            KVStore(self.vault.db).delete(
                _CREATED_MARKER_PREFIX + svc.id)
        except Exception:  # noqa: BLE001 - marker is bookkeeping
            _log.debug("stale-marker delete failed", exc_info=True)
        self._audit("create", svc.id, "pending_stale_invalidated",
                    checkpoint=checkpoint_id)

    # -- identity resolution for creation --------------------------------

    def _generated_identity(self, svc: ProvisionService,
                            creator: Any) -> tuple[Any, str, str]:
        """Resolve the service's SELF-GENERATED identity.

        Returns ``(persona, email, mint_failure_detail)``. The persona
        comes from the IdentityBank via ``get_or_mint`` — one persona
        per service, reused across retries, never re-minted per
        attempt. A bank persona's email starts as an EMPTY slot (it is
        not an address); a real disposable address is minted through
        the creator's ``create_email_account`` — the exact call the
        trial signup driver's ``_ensure_email`` makes — and persisted
        back onto the persona so retries reuse the same address. When
        minting fails, ``email`` is "" and the third element says why;
        no address is ever constructed locally.
        """
        from .identity_bank import IdentityBank

        bank = IdentityBank(db=self.vault.db, vault=self.vault)
        persona = bank.get_or_mint(svc.vault_service)
        if persona.email:
            return persona, persona.email, ""
        username = (persona.username_variants[0]
                    if persona.username_variants else None)
        account = _run_coro(creator.create_email_account(
            provider=_GENERATED_EMAIL_PROVIDER, username=username))
        if getattr(account, "status", "") != "created" \
                or not getattr(account, "email", ""):
            detail = (getattr(account, "notes", "") or "no address returned")
            self._audit("create", svc.id, "generated_email_failed",
                        persona_id=persona.id, detail=str(detail)[:200])
            return persona, "", str(detail)
        persona.email = account.email
        persona.email_provider = _GENERATED_EMAIL_PROVIDER
        bank._save(persona)  # persist the address onto the stored persona
        self._audit("create", svc.id, "generated_email_minted",
                    persona_id=persona.id, email=persona.email)
        return persona, persona.email, ""

    @staticmethod
    def _generated_note(svc: ProvisionService, name: str,
                        email: str) -> str:
        """The plain fact recorded with a creation that ran on a
        generated identity: which account it is. No provenance
        narration — the identity BEHAVIOUR (recorded owner identity
        preferred when set, a generated persona otherwise, its
        absence never blocking) is the owner's rule and needs no
        lecture in the result."""
        return f"Account: {name} <{email}>."

    # -- automated signup attempt ----------------------------------------

    @staticmethod
    def _owner_identity_note(name: str) -> str:
        """The plain fact appended to creation results that ran on
        the recorded owner identity: which account it is. Name
        only — never a credential value."""
        return f"Account: {name}." if name else ""

    @staticmethod
    def _username_from_email(email: str) -> str:
        """A username seed derived from the resolved identity's email
        local part, for signup forms that ask for a username the
        identity doesn't carry one for."""
        local = (email or "").split("@", 1)[0].strip().lower()
        local = re.sub(r"[^a-z0-9._-]", "", local)
        return local or "owner"

    def _store_pending_account(self, svc: ProvisionService, *,
                               username: str, password: str,
                               email: str, note: str) -> None:
        """Vault a just-submitted signup's credentials + the creation
        marker, so the next /provision run picks the drive up at the
        login step (and reports the account as created by Devon)."""
        from ..storage.kv import KVStore

        self.vault.store(
            svc.vault_service, username, password,
            credential_type="account",
            tags=["account", "signup-pending"],
            metadata={"email": email, "source": "provision_driver",
                      "signup_submitted_at": time.time()})
        try:
            KVStore(self.vault.db).set(
                _CREATED_MARKER_PREFIX + svc.id,
                {"checkpoint_id": "", "email": email,
                 "created_at": time.time(), "creation_note": note})
        except Exception:  # noqa: BLE001 - marker is bookkeeping
            _log.debug("created-marker write failed", exc_info=True)

    def _attempt_signup(self, svc: ProvisionService, creator: Any, *,
                        name: str, email: str, username: str,
                        password: str, note: str, identity_kind: str,
                        ) -> ProvisionResult | None:
        """Attempt the automated signup on the identity the caller
        resolved (recorded owner identity preferred; self-generated
        persona otherwise — the resolution itself is unchanged).

        The drive is the default, not the exception: a service with
        a named signup flow uses it, and a service carrying only a
        signup URL is driven with the generic form flow — Devon
        drives the signup herself either way. Returns None only when
        the service has neither a flow nor a URL (the caller then
        falls back to the creator's checkpoint path); otherwise the
        attempt's outcome — created, a verification-pending or wall
        checkpoint carrying ONE smallest owner action, an
        already-exists handoff, or a loud page-changed failure
        naming the exact step that does not match the page."""
        flow_id = svc.signup_flow or (
            "form_signup_v1" if svc.signup_url else "")
        flow = _SIGNUP_FLOWS.get(flow_id)
        if flow is None:
            return None
        tab = self.open_tab(svc.vault_service)
        try:
            result = flow(tab, svc, name=name, email=email,
                          username=username, password=password,
                          captcha_solver=self.captcha_solver)
        except SignupAccountExists:
            self._audit("create", svc.id, "signup_account_exists",
                        identity=identity_kind, username=username)
            return ProvisionResult(
                ProvisionOutcome.NEEDS_OWNER_LOGIN, svc.id,
                f"{svc.display} says an account already exists for "
                "the signup email — I won't create a duplicate. Hand "
                f"me that login and I'll fetch the key: /provision "
                f"login {svc.id} {username} (your next message is "
                "the password, kept out of history and logs)."
                + (f"\n{note}" if note else ""),
                username=username)
        except SignupBlocked as blocked:
            if blocked.submitted:
                self._store_pending_account(
                    svc, username=username, password=password,
                    email=email, note=note)
            self._audit("create", svc.id, "signup_blocked",
                        kind=blocked.kind, identity=identity_kind,
                        detail=blocked.detail[:200])
            drove = ("I filled and submitted the signup myself"
                     if blocked.submitted else
                     "I was driving the signup form myself")
            return ProvisionResult(
                ProvisionOutcome.CREATION_CHECKPOINT, svc.id,
                f"creating your {svc.display} account — {drove}, and "
                f"it stopped at exactly one step only you can do: "
                f"{blocked.title}. {blocked.action}"
                + (f"\n{note}" if note else ""),
                checkpoint={"id": "", "kind": blocked.kind,
                            "title": blocked.title,
                            "instructions": blocked.action})
        except FlowStepError as exc:
            self._audit("create", svc.id, "signup_page_changed",
                        flow_step=exc.step, detail=exc.detail[:200])
            return ProvisionResult(
                ProvisionOutcome.PAGE_CHANGED, svc.id,
                f"the {svc.display} signup page didn't match the "
                f"automated drive at step {exc.step!r}: {exc.detail} "
                "— nothing was created; that step is the one that "
                "cannot be automated against the page as it stands, "
                "and the signup flow needs updating for it.",
                step=f"signup:{exc.step}")
        finally:
            try:
                tab.close()
            except Exception:  # noqa: BLE001 - teardown best-effort
                pass
        if result.status == "created":
            creator.finalize_account(
                svc.vault_service, username, password, email=email,
                verification_note="automated signup driven by the "
                                  "provision driver")
            self._audit("create", svc.id, "created",
                        username=username, identity=identity_kind)
            return self._drive_with_login(svc, account_created=True,
                                          creation_note=note)
        # verification_pending: the service sent its verification to
        # the signup inbox. The credentials are already vaulted; the
        # next run picks the drive up at the login step.
        self._store_pending_account(
            svc, username=username, password=password, email=email,
            note=note)
        self._audit("create", svc.id, "signup_verification_pending",
                    username=username, identity=identity_kind)
        if identity_kind == "owner":
            action = (
                f"click the verification link {svc.display} just "
                "emailed to you — your login is already saved; tell "
                "me once you've clicked and I'll fetch the key. If "
                "that page had you set a password yourself, hand it "
                f"over with /provision login {svc.id} {username} "
                "instead.")
            lead = (f"I filled and submitted the signup myself as "
                    f"{name}; the one step only you can do: ")
        else:
            action = (
                "no step for you: the verification email went to "
                f"the inbox {email} — one I minted for this signup "
                "and can read. The login is saved; I'll finish the "
                "verification and fetch the key from there.")
            lead = ("I filled and submitted the signup myself; it "
                    "stopped at the service's email verification: ")
        return ProvisionResult(
            ProvisionOutcome.CREATION_CHECKPOINT, svc.id,
            f"creating your {svc.display} account — {lead}{action}"
            + (f"\n{note}" if note else ""),
            checkpoint={"id": "", "kind": "email_verify",
                        "title": f"Verify your {svc.display} email",
                        "instructions": action})

    def _create_then_drive(self, svc: ProvisionService) -> ProvisionResult:
        from .creator import (
            AccountCheckpointPending,
            AccountExistsError,
            generate_password,
        )
        from ..storage.kv import KVStore

        creator = self._make_creator()
        identity = creator.get_owner_identity()
        creation_note = ""
        create_kwargs: dict[str, Any] = {}
        if identity and identity.get("email"):
            # 1. A recorded owner identity stays PREFERRED — unchanged.
            email = identity["email"]
            create_kwargs["email"] = email
            signup_name = str(identity.get("name") or "").strip()
            signup_username = self._username_from_email(email)
            signup_password = generate_password()
            creation_note = self._owner_identity_note(signup_name)
            identity_kind = "owner"
            self._audit("create", svc.id, "begin", email=email,
                        identity="owner")
        else:
            # 2. No recorded identity → self-generated (owner's rule
            #    2026-10-20: "not limited to my provided identity").
            persona, email, mint_failure = self._generated_identity(
                svc, creator)
            if not email:
                return ProvisionResult(
                    ProvisionOutcome.IDENTITY_MINT_FAILED, svc.id,
                    f"creating your {svc.display} account stalled: "
                    f"minting its signup email address failed "
                    f"({mint_failure}) — nothing was created and no "
                    "address was invented. This is a temp-mail "
                    "problem; the same identity will be reused on "
                    "the next attempt.")
            create_kwargs["email"] = email
            if persona.username_variants:
                create_kwargs["username"] = persona.username_variants[0]
            if persona.password:
                create_kwargs["password"] = persona.password
            signup_name = persona.name
            signup_username = (create_kwargs.get("username")
                               or self._username_from_email(email))
            signup_password = create_kwargs.get("password") or ""
            if not signup_password:
                signup_password = generate_password()
            creation_note = self._generated_note(svc, persona.name, email)
            identity_kind = "generated"
            self._audit("create", svc.id, "begin", email=email,
                        identity="generated", persona_id=persona.id,
                        persona_name=persona.name)
        # Devon drives the signup herself wherever a signup flow is
        # registered — on whichever identity was resolved above.
        # Only services without one fall through to the creator's
        # human-checkpoint path below.
        attempted = self._attempt_signup(
            svc, creator, name=signup_name, email=email,
            username=signup_username, password=signup_password,
            note=creation_note, identity_kind=identity_kind)
        if attempted is not None:
            return attempted
        try:
            created = _run_coro(creator.create_account(
                svc.vault_service, **create_kwargs))
        except AccountExistsError:
            # A login landed meanwhile — the drive can proceed.
            self._audit("create", svc.id, "already_exists")
            return self._drive_with_login(svc, account_created=False,
                                          creation_note=creation_note)
        except AccountCheckpointPending as pending:
            cp = pending.checkpoint
            try:
                KVStore(self.vault.db).set(
                    _CREATED_MARKER_PREFIX + svc.id,
                    {"checkpoint_id": cp.id,
                     "email": email, "created_at": time.time(),
                     "creation_note": creation_note})
            except Exception:  # noqa: BLE001 - marker is bookkeeping
                _log.debug("created-marker write failed", exc_info=True)
            self._audit("create", svc.id, "checkpoint",
                        checkpoint=cp.id, kind=cp.kind.value)
            return ProvisionResult(
                ProvisionOutcome.CREATION_CHECKPOINT, svc.id,
                f"creating your {svc.display} account — it stopped "
                f"at one step only you can do: {cp.title}. "
                f"{cp.instructions}"
                + (f"\n{creation_note}" if creation_note else ""),
                checkpoint={"id": cp.id, "kind": cp.kind.value,
                            "title": cp.title,
                            "instructions": cp.instructions})
        # Synchronous completion (disposable-style flows): a login now
        # exists in the vault — continue into the drive.
        self._audit("create", svc.id, "created",
                    username=getattr(created, "username", ""))
        return self._drive_with_login(svc, account_created=True,
                                      creation_note=creation_note)

    def _created_marker(self, svc: ProvisionService, *,
                        consume: bool = False) -> dict[str, Any] | None:
        """The pending-creation marker for this service, if any."""
        try:
            from ..storage.kv import KVStore

            store = KVStore(self.vault.db)
            marker = store.get(_CREATED_MARKER_PREFIX + svc.id)
            if marker and consume:
                store.delete(_CREATED_MARKER_PREFIX + svc.id)
            return marker if isinstance(marker, dict) else None
        except Exception:  # noqa: BLE001 - bookkeeping only
            _log.debug("created-marker read failed", exc_info=True)
            return None

    def _created_marker_note(self, svc: ProvisionService) -> str:
        """The creation disclosure recorded with the pending marker
        ("" for owner-identity creations and when no marker exists)."""
        marker = self._created_marker(svc)
        return str((marker or {}).get("creation_note") or "")

    # -- the drive ---------------------------------------------------------

    def _drive_with_login(self, svc: ProvisionService, *,
                          account_created: bool,
                          creation_note: str = "") -> ProvisionResult:
        if not account_created:
            marker = self._created_marker(svc, consume=True)
            account_created = bool(marker)
            if marker:
                marker_note = str(marker.get("creation_note") or "")
                if _contains_purged_text(marker_note):
                    # A pre-purge lecture riding in a stored marker:
                    # dropped, never adopted into a driven result.
                    self._audit("create", svc.id,
                                "stale_marker_note_dropped")
                    marker_note = ""
                if marker_note:
                    # The creation happened in an earlier run — keep
                    # THIS run's audit trail honest about the identity
                    # the account was created with.
                    self._audit("create", svc.id,
                                "resumed_generated_creation",
                                note=marker_note[:300])
                    if not creation_note:
                        creation_note = marker_note
        login = self._login_cred(svc)
        username = login.username if login is not None else ""
        self._audit("login", svc.id, "begin", username=username)
        try:
            result = ensure_login(
                self.manager, self.sessions,
                lambda: self.open_tab(svc.vault_service),
                service=svc.vault_service,
                config=LoginConfig(login_url=svc.login_url),
                captcha_solver=self.captcha_solver,
            )
        except LoginCaptchaRequired as exc:
            self._audit("login", svc.id, "verification_blocked",
                        detail=str(exc)[:200])
            return ProvisionResult(
                ProvisionOutcome.VERIFICATION_BLOCKED, svc.id,
                f"the {svc.display} login hit a verification wall "
                f"the solver could not clear: {exc}. One step only "
                "you can do: clear that verification in the drive's "
                "live view — tell me when it's done and I'll take "
                "it from there.",
                username=username, step="login")
        except LoginFailed as exc:
            self._audit("login", svc.id, "failed", detail=str(exc)[:200])
            return ProvisionResult(
                ProvisionOutcome.LOGIN_FAILED, svc.id,
                f"the {svc.display} login in the vault didn't get in: "
                f"{exc} — if the password changed, hand me the current "
                f"one with /provision login {svc.id} <username>.",
                username=username, step="login")
        username = str(result.get("username") or username)
        self._audit("login", svc.id, "ok", username=username,
                    from_session=bool(result.get("from_session")))

        flow = _PAGE_FLOWS.get(svc.flow or "")
        if flow is None:
            return ProvisionResult(
                ProvisionOutcome.PAGE_CHANGED, svc.id,
                f"no page flow registered under id {svc.flow!r} for "
                f"{svc.display} — the registry entry is incomplete.",
                username=username, step="flow_lookup")
        tab = self.open_tab(svc.vault_service)
        try:
            token_name = f"devon-{time.strftime('%Y%m%d')}"
            try:
                token = flow(tab, svc, token_name=token_name,
                             captcha_solver=self.captcha_solver)
            except FlowBlocked as exc:
                self._audit("flow", svc.id, "verification_blocked",
                            detail=str(exc)[:200])
                return ProvisionResult(
                    ProvisionOutcome.VERIFICATION_BLOCKED, svc.id,
                    f"creating the {svc.display} key hit a wall: {exc}",
                    username=username, step="flow")
            except FlowStepError as exc:
                self._audit("flow", svc.id, "page_changed",
                            flow_step=exc.step, detail=exc.detail[:200])
                return ProvisionResult(
                    ProvisionOutcome.PAGE_CHANGED, svc.id,
                    f"the {svc.display} key-creation page didn't match "
                    f"the flow at step {exc.step!r}: {exc.detail} — "
                    "nothing was stored; the page has likely changed "
                    "and the flow needs updating.",
                    username=username, step=exc.step)
        finally:
            try:
                tab.close()
            except Exception:  # noqa: BLE001 - teardown best-effort
                pass
        self._audit("flow", svc.id, "token_created")

        validation = validate_service_token(svc, token,
                                            http_get=self.http_get)
        self._audit("validate", svc.id, validation.status,
                    detail=validation.detail[:200])
        if validation.status == "rejected":
            return ProvisionResult(
                ProvisionOutcome.KEY_REJECTED, svc.id,
                f"the {svc.display} key I just created failed validation "
                f"({validation.detail}) — I did not keep it. It may need "
                "a minute to activate, or the account may lack access; "
                "tell me to retry and I'll run it again.",
                username=username, step="validate")
        if validation.status != "ok":
            return ProvisionResult(
                ProvisionOutcome.VALIDATION_INCONCLUSIVE, svc.id,
                f"I created a {svc.display} key but could not validate "
                f"it ({validation.detail}) — I did not store an "
                "unproven key. Check connectivity, then tell me to "
                "retry and I'll run it again.",
                username=username, step="validate")

        warnings: list[str] = []
        license_notes = self._run_license_steps(svc)
        warnings.extend(license_notes)

        # The key lives under its OWN vault slot: the vault upserts by
        # (service, username), so storing it under the login's username
        # would overwrite the password credential it came from. The
        # owning account is recorded in metadata instead.
        self.vault.store(
            svc.vault_service, "api-key", token,
            credential_type="api_key",
            tags=["provisioned", "self-provisioned"],
            metadata={"source": "provision_driver",
                      "account": username,
                      "provisioned_at": time.time(),
                      "scopes": list(svc.scopes),
                      "account_created": account_created,
                      "validation": validation.detail[:200]},
        )
        self._audit("vault", svc.id, "stored", username=username,
                    account_created=account_created)
        applied: list[str] = []
        for env_key in svc.env_keys:
            self.environ[env_key] = token  # type: ignore[index]
            applied.append(env_key)
        self._audit("bridge", svc.id, "applied", env_keys=applied)

        verb = "created a new account and fetched" if account_created \
            else "fetched"
        message = (
            f"provisioned {svc.display}: {verb} an API key on "
            f"{username} and stored it in the credential vault "
            f"(service {svc.vault_service}); "
            f"{', '.join(applied)} now resolves from it — no restart "
            "needed for features that read it per call.")
        if validation.detail:
            message += f" Validation: {validation.detail}."
        if creation_note:
            message += f"\n{creation_note}"
        return ProvisionResult(
            ProvisionOutcome.PROVISIONED, svc.id, message,
            account_created=account_created, username=username,
            env_keys_applied=applied, warnings=warnings)

    def _run_license_steps(self, svc: ProvisionService) -> list[str]:
        """Best-effort license/consent acceptance for gated models.

        Each step opens the license page in the service session: if an
        agree control is present it is clicked; if the license is
        already accepted the page says so. A page-shape failure is a
        WARNING, not a lost key — the key is valid; the gated model
        just stays gated until the license is accepted, and the report
        says exactly that."""
        notes: list[str] = []
        for step in svc.licenses:
            tab = self.open_tab(svc.vault_service)
            try:
                tab.navigate(step.url)
                text = _tab_text(tab)
                if "you have already accepted" in text.lower() or \
                        "already accepted" in text.lower():
                    notes.append(
                        f"license for {step.subject}: already accepted.")
                    self._audit("license", svc.id, "already_accepted",
                                subject=step.subject)
                    continue
                try:
                    used = _click_text(tab, *step.agree_texts)
                    notes.append(
                        f"license for {step.subject}: accepted ({used}).")
                    self._audit("license", svc.id, "accepted",
                                subject=step.subject)
                except FlowStepError:
                    notes.append(
                        f"license for {step.subject}: no accept control "
                        "found on the page — the key works, but this "
                        "gated model may refuse calls until you accept "
                        f"the license at {step.url}.")
                    self._audit("license", svc.id, "no_control",
                                subject=step.subject)
            except Exception as exc:  # noqa: BLE001 - warning, not fatal
                notes.append(
                    f"license for {step.subject}: step failed ({exc}) — "
                    "accept it manually if the model refuses calls.")
                self._audit("license", svc.id, "failed",
                            subject=step.subject, detail=str(exc)[:160])
            finally:
                try:
                    tab.close()
                except Exception:  # noqa: BLE001
                    pass
        return notes

