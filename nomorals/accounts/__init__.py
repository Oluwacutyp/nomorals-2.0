"""Account management and credential vault.

This module provides secure credential storage, account lifecycle management,
and session handling for the bot's various service integrations.

The bot can:
- Create and manage its own accounts (email, social, etc.)
- Store credentials securely (AES-256 encryption)
- Handle OAuth tokens, API keys, session cookies
- Rotate credentials and manage expiry
- Support multiple identity profiles

Account creation follows the human-in-the-loop checkpoint discipline:
one account per service, the owner's own identity, and human
verification steps (CAPTCHA, email/phone checks) pause on a persisted
checkpoint instead of being bypassed.
"""

from .vault import CredentialVault, Credential, AccountProfile
from .manager import AccountManager
from .browser_login import (
    LoginConfig,
    LoginFailed,
    LoginCaptchaRequired,
    login_with_vault,
    ensure_login,
    PasswordChangeConfig,
    change_password_on_site,
    USERNAME_FIELD_CANDIDATES,
    PASSWORD_FIELD_CANDIDATES,
)
from .creator import (
    AccountCreator,
    CreatedAccount,
    AccountCheckpoint,
    AccountCheckpointPending,
    AccountExistsError,
    MissingOwnerIdentity,
    CheckpointKind,
    CheckpointState,
    CheckpointStore,
)
from .sessions import (
    SessionManager,
    Session,
    OAuthToken,
    SessionInvalid,
)
from .health import (
    AccountHealth,
    check_account_health,
    check_all_health,
    LOCKED_MARKERS,
    VERIFICATION_MARKERS,
    LOGGED_OUT_MARKERS,
)
from .temp_sms import (
    TempSmsProvider,
    SimcodesProvider,
    SevenSimProvider,
    TempNumber,
    SmsMessage,
    get_provider as get_sms_provider,
    grab_number as grab_temp_number,
    grab_number_cascade as grab_temp_number_cascade,
    wait_code as wait_temp_sms_code,
    PROVIDERS as SMS_PROVIDERS,
    CASCADE_PROVIDERS as SMS_CASCADE_PROVIDERS,
)
from .identity_bank import (
    Persona,
    IdentityBank,
    ConfirmationGate,
    ConfirmationRequired,
    NIGERIAN_FIRST_NAMES,
    INTERNATIONAL_FIRST_NAMES,
    DISPOSABLE_SURNAMES,
    render_persona_card,
)
from .signup_driver import (
    SignupStage,
    WallKind,
    SignupAttempt,
    SignupAttemptStore,
    SignupDriver,
    PageDriver,
    classify_wall,
    KNOWN_SIGNUP_URLS,
    ALLOWED_TRANSITIONS,
    render_attempt_summary,
)

__all__ = [
    "CredentialVault",
    "Credential",
    "AccountProfile",
    "AccountManager",
    "LoginConfig",
    "LoginFailed",
    "LoginCaptchaRequired",
    "login_with_vault",
    "ensure_login",
    "PasswordChangeConfig",
    "change_password_on_site",
    "USERNAME_FIELD_CANDIDATES",
    "PASSWORD_FIELD_CANDIDATES",
    "AccountCreator",
    "CreatedAccount",
    "AccountCheckpoint",
    "AccountCheckpointPending",
    "AccountExistsError",
    "MissingOwnerIdentity",
    "CheckpointKind",
    "CheckpointState",
    "CheckpointStore",
    "SessionManager",
    "Session",
    "OAuthToken",
    "SessionInvalid",
    "AccountHealth",
    "check_account_health",
    "check_all_health",
    "LOCKED_MARKERS",
    "VERIFICATION_MARKERS",
    "LOGGED_OUT_MARKERS",
    "TempSmsProvider",
    "SimcodesProvider",
    "SevenSimProvider",
    "TempNumber",
    "SmsMessage",
    "get_sms_provider",
    "grab_temp_number",
    "grab_temp_number_cascade",
    "wait_temp_sms_code",
    "SMS_PROVIDERS",
    "SMS_CASCADE_PROVIDERS",
    "Persona",
    "IdentityBank",
    "ConfirmationGate",
    "ConfirmationRequired",
    "NIGERIAN_FIRST_NAMES",
    "INTERNATIONAL_FIRST_NAMES",
    "DISPOSABLE_SURNAMES",
    "render_persona_card",
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
