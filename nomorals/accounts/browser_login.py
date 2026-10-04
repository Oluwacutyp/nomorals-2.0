"""Browser login for existing accounts.

"Login to existing accounts" as a real flow: take a password credential
from the vault, drive a browser tab through the service's login form,
handle CAPTCHAs through the injected solver (same callable shape as
``AccountCreator``'s ``captcha_solver``), and persist the logged-in
cookies into the :class:`SessionManager` so the session survives
restarts.

Layering: this module is L2 (accounts) and stays duck-typed — it never
imports the L4 browser or captcha modules. The caller supplies:

* ``open_tab`` — zero-arg factory returning a tab with this protocol::
      navigate(url) -> dict
      fill(name, value) -> dict            # raises when the field is missing
      submit(target="") -> dict
      text(max_chars=...) -> {"text": ...}
      cookies() -> [{"name":..., "value":...}, ...]
      check_captcha() -> {"challenges": [...]}   # optional
      evaluate(js) -> Any                         # optional, token injection
      wait_for_text(text, timeout=...) -> dict     # optional, explicit wait
      close() -> None

  Both ``nomorals.browser.service.Tab`` and ``RenderedTab`` satisfy this.

* ``captcha_solver`` — optional ``challenge_dict -> result_dict`` callable
  (see ``nomorals.tools.captcha.creator_solver_adapter``). When a login
  hits a CAPTCHA and no solver is supplied, the login fails fast with a
  takeover note instead of hanging.

Field discovery is automatic: unless the credential metadata pins
``username_field``/``password_field``, the usual field names are tried in
order and the first one the page accepts wins. Per-service overrides
live in the credential's metadata: ``login_url`` (required),
``username_field``, ``password_field``, ``username_value``,
``submit_target``, ``success_text``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.errors import NoMoralsError, NotFound
from ..core.logging_setup import get_logger

__all__ = [
    "LoginConfig",
    "LoginFailed",
    "LoginCaptchaRequired",
    "login_with_vault",
    "ensure_login",
    "PasswordChangeConfig",
    "change_password_on_site",
    "USERNAME_FIELD_CANDIDATES",
    "PASSWORD_FIELD_CANDIDATES",
]

_log = get_logger(__name__)

#: tried in order when the credential metadata doesn't pin a field name.
USERNAME_FIELD_CANDIDATES = (
    "email", "username", "login", "user", "account", "phone",
    "emailAddress", "userName",
)
PASSWORD_FIELD_CANDIDATES = (
    "password", "pass", "passwd", "pwd",
)
#: tried in order for solved image/audio captcha text.
CAPTCHA_TEXT_FIELDS = (
    "captcha", "captcha_code", "captchacode", "captchaCode",
    "code", "verification_code", "answer",
)
#: page text that means the login was rejected (checked case-insensitively).
FAILURE_MARKERS = (
    "incorrect password", "invalid password", "wrong password",
    "password is incorrect", "login failed", "log in failed",
    "could not log", "couldn't log", "invalid credentials",
    "invalid login", "authentication failed", "sign in failed",
    "account not found", "no account found",
)

#: token kinds solved to a response token injected into the page.
_TOKEN_KINDS = {
    "recaptcha_v2", "recaptcha_enterprise", "hcaptcha", "turnstile",
}
#: kinds solved to readable text typed into a field.
_TEXT_KINDS = {"image_captcha", "audio_captcha"}

#: JS snippets that drop a solved token into the page's response field.
_TOKEN_INJECT_JS = {
    "recaptcha_v2": (
        "var el=document.getElementById('g-recaptcha-response');"
        "if(el){el.innerHTML='{token}';}"),
    "recaptcha_enterprise": (
        "var el=document.getElementById('g-recaptcha-response');"
        "if(el){el.innerHTML='{token}';}"),
    "hcaptcha": (
        "var el=document.querySelector('[name=h-captcha-response]');"
        "if(el){el.innerHTML='{token}';}"),
    "turnstile": (
        "var el=document.querySelector('[name=cf-turnstile-response]');"
        "if(el){el.innerHTML='{token}';}"),
}


class LoginFailed(NoMoralsError):
    """The login flow could not complete (bad config, rejected creds)."""


class LoginCaptchaRequired(NoMoralsError):
    """A CAPTCHA blocked the login and no solver could clear it.

    Carries the detected challenges so the caller can page the owner
    with exactly what needs a human hand.
    """

    def __init__(self, message: str,
                 challenges: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.challenges = challenges or []


@dataclass
class LoginConfig:
    """Per-login overrides. Anything unset falls back to the credential's
    metadata, then to automatic field discovery."""

    login_url: str = ""
    username_field: str = ""
    password_field: str = ""
    username_value: str = ""   # default: the vault username
    submit_target: str = ""   # default: the page's first form
    success_text: str = ""    # text that must appear after login
    post_login_wait_ms: int = 5000


def _config_from_metadata(meta: dict[str, Any],
                          override: LoginConfig | None) -> LoginConfig:
    cfg = LoginConfig()
    for f in ("login_url", "username_field", "password_field",
              "username_value", "submit_target", "success_text",
              "post_login_wait_ms"):
        if f in meta:
            setattr(cfg, f, meta[f])
    if override is not None:
        for f in ("login_url", "username_field", "password_field",
                  "username_value", "submit_target", "success_text",
                  "post_login_wait_ms"):
            val = getattr(override, f)
            if val or f == "post_login_wait_ms":
                setattr(cfg, f, val)
    return cfg


def _fill_first(tab: Any, candidates: list[str], value: str,
                what: str) -> str:
    """Fill the first candidate field the page accepts. Returns the field
    name used; raises LoginFailed when none of them exists."""
    last_exc: Exception | None = None
    for name in candidates:
        try:
            tab.fill(name, value)
            return name
        except Exception as exc:  # noqa: BLE001 — try the next candidate
            last_exc = exc
    raise LoginFailed(
        f"no {what} field found (tried: {', '.join(candidates)}): "
        f"{last_exc}")


def _page_text(tab: Any) -> str:
    try:
        return str(tab.text(max_chars=20000).get("text") or "")
    except Exception:  # noqa: BLE001 — text is best-effort here
        return ""


def _wait_after_submit(tab: Any, success_text: str = "",
                     wait_ms: int = 5000) -> None:
    """Prefer an explicit wait when the tab supports it; sleep otherwise."""
    if success_text and hasattr(tab, "wait_for_text"):
        try:
            tab.wait_for_text(success_text, timeout=wait_ms)
            return
        except Exception as exc:  # noqa: BLE001 — fall through to sleep
            _log.debug("explicit wait_for_text missed: %s", exc)
    time.sleep(max(0.5, wait_ms / 1000.0))


def _detect_captchas(tab: Any) -> list[dict[str, Any]]:
    checker = getattr(tab, "check_captcha", None)
    if checker is None:
        return []
    try:
        return list(checker().get("challenges") or [])
    except Exception as exc:  # noqa: BLE001 — detection best-effort
        _log.debug("captcha detection failed: %s", exc)
        return []


def _solve_and_apply(tab: Any, challenge: dict[str, Any],
                     solver: Callable[[dict[str, Any]], dict[str, Any]],
                     service: str) -> str:
    """Run one challenge through the solver and apply the result to the
    page. Returns a human-readable note of what happened."""
    kind = challenge.get("kind", "")
    result = solver({
        "kind": kind,
        "sitekey": challenge.get("sitekey", "") or "",
        "page_url": challenge.get("page_url", "") or "",
        "image_url": challenge.get("image_url", "") or "",
        "image_bytes": b"",
        "action": challenge.get("action", "") or "",
        "min_score": 0.3,
    })
    if not result.get("ok"):
        if result.get("takeover"):
            raise LoginCaptchaRequired(
                f"CAPTCHA needs a human hand on {service}: "
                f"{result.get('detail', '')}",
                challenges=[challenge])
        raise LoginCaptchaRequired(
            f"CAPTCHA solver failed on {service}: "
            f"{result.get('detail', 'unknown')}",
            challenges=[challenge])
    if kind in _TOKEN_KINDS:
        token = result.get("token", "")
        if not token:
            raise LoginCaptchaRequired(
                f"solver returned no token for {kind} on {service}",
                challenges=[challenge])
        js_tpl = _TOKEN_INJECT_JS.get(kind, _TOKEN_INJECT_JS["recaptcha_v2"])
        escaped = token.replace("\\", "\\\\").replace("'", "\\'")
        if not hasattr(tab, "evaluate"):
            raise LoginCaptchaRequired(
                f"solved {kind} but this tab cannot inject the token "
                "(no evaluate) — submit it manually",
                challenges=[challenge])
        tab.evaluate(js_tpl.replace("{token}", escaped))
        return f"solved {kind} via {result.get('backend')} (token injected)"
    if kind in _TEXT_KINDS:
        text = result.get("text", "")
        if not text:
            raise LoginCaptchaRequired(
                f"solver returned no text for {kind} on {service}",
                challenges=[challenge])
        used = _fill_first(tab, list(CAPTCHA_TEXT_FIELDS), text,
                           "captcha answer")
        return (f"solved {kind} via {result.get('backend')} "
                f"(typed into {used!r})")
    raise LoginCaptchaRequired(
        f"solver returned an unusable result for kind {kind!r} on {service}",
        challenges=[challenge])


def _failure_reason(page_text: str) -> str:
    low = page_text.lower()
    for marker in FAILURE_MARKERS:
        if marker in low:
            return marker
    return ""


def login_with_vault(
    manager: Any,
    sessions: Any,
    open_tab: Callable[[], Any],
    *,
    service: str,
    username: str | None = None,
    config: LoginConfig | None = None,
    captcha_solver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Log in to ``service`` with the vault-stored password credential.

    ``username`` pins an account; otherwise the service default (see
    ``AccountManager.set_default``) or the single stored account wins.
    On success the tab's cookies are persisted into ``sessions`` (the
    :class:`SessionManager`), so the login survives restarts.

    Returns ``{"ok": True, "service", "username", "url",
    "cookies_saved", "captcha_solved": [...], "note"}``. Raises
    :class:`LoginFailed` on config/credential problems and
    :class:`LoginCaptchaRequired` when a CAPTCHA blocks the login and no
    solver cleared it.
    """
    service = (service or "").strip()
    if not service:
        raise LoginFailed("login_with_vault needs a service")
    cred = manager.resolve_account(service, username)
    if cred.credential_type != "password":
        raise LoginFailed(
            f"{service}/{cred.username} is a {cred.credential_type} "
            "credential, not a password — browser login needs a password")
    cfg = _config_from_metadata(cred.metadata or {}, config)
    if not cfg.login_url:
        raise LoginFailed(
            f"no login_url for {service} — store one in the credential "
            "metadata (key 'login_url') or pass LoginConfig(login_url=...)")
    login_name = cfg.username_value or cred.username
    secret = cred.password
    if not secret:
        raise LoginFailed(f"no password stored for {service}/{cred.username}")

    tab = open_tab()
    captcha_notes: list[str] = []
    solved: set[tuple[str, str, str]] = set()

    def _unsolved(challenges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Challenges not already cleared this run — solving services
        charge per solve, so never pay twice for the same challenge."""
        fresh = []
        for ch in challenges:
            key = (ch.get("kind", ""), ch.get("sitekey", "") or "",
                   ch.get("image_url", "") or "")
            if key not in solved:
                solved.add(key)
                fresh.append(ch)
        return fresh

    try:
        tab.navigate(cfg.login_url)

        user_candidates = ([cfg.username_field] if cfg.username_field
                           else []) + list(USERNAME_FIELD_CANDIDATES)
        pass_candidates = ([cfg.password_field] if cfg.password_field
                           else []) + list(PASSWORD_FIELD_CANDIDATES)
        user_field = _fill_first(tab, user_candidates, login_name,
                                 "username/email")
        _fill_first(tab, pass_candidates, secret, "password")
        _log.info("login %s/%s: filled %r + password field",
                  service, cred.username, user_field)

        # CAPTCHA already on the login page? clear it before submitting.
        for ch in _unsolved(_detect_captchas(tab)):
            if captcha_solver is None:
                raise LoginCaptchaRequired(
                    f"CAPTCHA ({ch.get('kind')}) on {service} login page "
                    "and no solver supplied",
                    challenges=[ch])
            captcha_notes.append(_solve_and_apply(tab, ch, captcha_solver,
                                                 service))

        tab.submit(cfg.submit_target)
        _wait_after_submit(tab, cfg.success_text, cfg.post_login_wait_ms)

        # CAPTCHA after submit (the common case) — solve, resubmit once.
        post = _unsolved(_detect_captchas(tab))
        if post:
            if captcha_solver is None:
                raise LoginCaptchaRequired(
                    f"CAPTCHA ({post[0].get('kind')}) blocked the "
                    f"{service} login and no solver was supplied",
                    challenges=post)
            for ch in post:
                captcha_notes.append(_solve_and_apply(tab, ch, captcha_solver,
                                                     service))
            tab.submit(cfg.submit_target)
            _wait_after_submit(tab, cfg.success_text, cfg.post_login_wait_ms)

        page_text = _page_text(tab)
        failure = _failure_reason(page_text)
        if failure:
            raise LoginFailed(
                f"{service} rejected the login ({failure}) — the vault "
                "password may be wrong or rotated")

        url = ""
        try:
            url = getattr(tab, "url", "") or ""
        except Exception:  # noqa: BLE001 — cosmetic
            url = ""
        if cfg.success_text and cfg.success_text not in page_text:
            raise LoginFailed(
                f"{service} login unverified: {cfg.success_text!r} not on "
                f"the page after submit (at {url or 'unknown url'})")

        cookies = tab.cookies() if hasattr(tab, "cookies") else []
        jar = {c.get("name", ""): c.get("value", "")
               for c in (cookies or []) if c.get("name")}
        if jar:
            sessions.set_cookies(service, cred.username, jar)

        note = f"logged in to {service} as {cred.username}"
        if captcha_notes:
            note += "; " + "; ".join(captcha_notes)
        _log.info("login ok: %s/%s (%d cookies saved)",
                  service, cred.username, len(jar))
        return {
            "ok": True,
            "service": service,
            "username": cred.username,
            "url": url,
            "cookies_saved": len(jar),
            "captcha_solved": captcha_notes,
            "note": note,
        }
    finally:
        try:
            tab.close()
        except Exception:  # noqa: BLE001 — teardown best-effort
            pass


# ── session-restore-aware login ──────────────────────────────────────────


def ensure_login(
    manager: Any,
    sessions: Any,
    open_tab: Callable[[], Any],
    *,
    service: str,
    username: str | None = None,
    config: LoginConfig | None = None,
    captcha_solver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Log in only when the stored session can't.

    Checks the :class:`SessionManager` first: when the account's session
    still holds usable cookies (or a valid OAuth token) and hasn't gone
    stale, no browser is opened and no login form is touched — the
    result reports ``from_session: True``. Otherwise falls through to
    :func:`login_with_vault` for a full browser login.

    Pass ``force=True`` to skip the restore check and re-login anyway.

    The result dict matches :func:`login_with_vault`'s shape plus
    ``from_session`` (bool) and ``note`` describing what happened.
    Raises :class:`LoginFailed` / :class:`LoginCaptchaRequired` only
    when a fresh login was actually attempted.
    """
    from .sessions import SessionInvalid

    service = (service or "").strip()
    if not service:
        raise LoginFailed("ensure_login needs a service")
    cred = manager.resolve_account(service, username)

    if not force:
        try:
            session = sessions.get_valid_session(service, cred.username)
        except SessionInvalid as exc:
            _log.info("stored session unusable for %s/%s: %s — re-logging in",
                      service, cred.username, exc)
            session = None
        if session is not None:
            has_cookies = bool(session.cookies)
            has_token = (session.oauth_token is not None
                         and not session.oauth_token.is_expired())
            if has_cookies or has_token:
                note = (f"session still valid for {service}/{cred.username} "
                        f"({len(session.cookies)} cookies) — skipped login")
                _log.info("login skipped: %s", note)
                return {
                    "ok": True,
                    "service": service,
                    "username": cred.username,
                    "from_session": True,
                    "cookies_saved": len(session.cookies),
                    "captcha_solved": [],
                    "note": note,
                }
            _log.info("stored session for %s/%s holds no credentials — "
                      "re-logging in", service, cred.username)

    result = login_with_vault(
        manager, sessions, open_tab,
        service=service,
        username=cred.username,
        config=config,
        captcha_solver=captcha_solver,
    )
    result["from_session"] = False
    return result


# ── on-site password rotation ────────────────────────────────────────────


@dataclass
class PasswordChangeConfig:
    """Per-service password-change flow. Anything unset falls back to the
    credential's metadata (same keys), then to automatic field discovery."""

    change_password_url: str = ""
    current_password_field: str = ""
    new_password_field: str = ""
    confirm_password_field: str = ""
    submit_target: str = ""
    success_text: str = ""
    post_login_wait_ms: int = 5000


#: tried in order when the credential metadata doesn't pin a field name.
CURRENT_PASSWORD_FIELDS = ("current_password", "currentPassword", "old_password")
NEW_PASSWORD_FIELDS = ("new_password", "newPassword", "password_new",
                       "change_password", "new_pass")
CONFIRM_PASSWORD_FIELDS = ("confirm_password", "confirmPassword",
                           "password_confirm", "new_password_confirm",
                           "verify_password")
#: page text that means the password change was rejected.
CHANGE_FAILURE_MARKERS = FAILURE_MARKERS + (
    "passwords do not match", "passwords don't match",
    "password too weak", "password is too weak",
    "password must be", "choose a stronger password",
    "password change failed", "could not change password",
    "couldn't change password", "incorrect current password",
    "wrong current password",
)


def _pw_config_from_metadata(meta: dict[str, Any],
                             override: PasswordChangeConfig | None
                             ) -> PasswordChangeConfig:
    cfg = PasswordChangeConfig()
    fields = ("change_password_url", "current_password_field",
              "new_password_field", "confirm_password_field",
              "submit_target", "success_text", "post_login_wait_ms")
    for f in fields:
        if f in meta:
            setattr(cfg, f, meta[f])
    if override is not None:
        for f in fields:
            val = getattr(override, f)
            if val or f == "post_login_wait_ms":
                setattr(cfg, f, val)
    return cfg


def change_password_on_site(
    manager: Any,
    sessions: Any,
    open_tab: Callable[[], Any],
    *,
    service: str,
    username: str | None = None,
    new_password: str = "",
    length: int = 32,
    config: PasswordChangeConfig | None = None,
    captcha_solver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rotate the password on the site itself, then in the vault.

    Flow: :func:`ensure_login` (authenticates when the session is gone),
    drive the site's password-change form with the vault password as the
    current one and a fresh secret as the new one, verify the change
    landed, persist the tab's cookies, and only THEN update the vault.

    The vault is never updated before the site confirms the change — a
    failed change with an already-rotated vault secret would lock the
    account out.

    ``new_password`` lets the caller pin the secret; otherwise a fresh
    cryptographically-secure one is generated. Returns
    ``{"ok": True, "service", "username", "rotated": True, "note"}``.
    Raises :class:`LoginFailed` / :class:`LoginCaptchaRequired` when the
    change cannot be completed.
    """
    from .creator import generate_password

    service = (service or "").strip()
    if not service:
        raise LoginFailed("change_password_on_site needs a service")
    cred = manager.resolve_account(service, username)
    if cred.credential_type != "password":
        raise LoginFailed(
            f"{service}/{cred.username} is a {cred.credential_type} "
            "credential, not a password — on-site rotation needs a password")
    cfg = _pw_config_from_metadata(cred.metadata or {}, config)
    if not cfg.change_password_url:
        raise LoginFailed(
            f"no change_password_url for {service} — store one in the "
            "credential metadata (key 'change_password_url') or pass "
            "PasswordChangeConfig(change_password_url=...)")
    current_secret = cred.password
    if not current_secret:
        raise LoginFailed(f"no password stored for {service}/{cred.username}")
    fresh = new_password or generate_password(length=length, symbols=True)

    # Authenticate first (restores the session when it is still valid).
    ensure_login(manager, sessions, open_tab,
                 service=service, username=cred.username,
                 captcha_solver=captcha_solver)

    tab = open_tab()
    try:
        tab.navigate(cfg.change_password_url)

        cur_candidates = ([cfg.current_password_field]
                          if cfg.current_password_field else []) + list(
                              CURRENT_PASSWORD_FIELDS)
        new_candidates = ([cfg.new_password_field]
                          if cfg.new_password_field else []) + list(
                              NEW_PASSWORD_FIELDS)
        confirm_candidates = ([cfg.confirm_password_field]
                              if cfg.confirm_password_field else []) + list(
                                  CONFIRM_PASSWORD_FIELDS)

        _fill_first(tab, cur_candidates, current_secret, "current password")
        _fill_first(tab, new_candidates, fresh, "new password")
        # Confirm field is optional — some forms only have one new field.
        try:
            _fill_first(tab, confirm_candidates, fresh, "confirm password")
        except LoginFailed:
            _log.debug("no confirm-password field on %s change form",
                       service)

        tab.submit(cfg.submit_target)
        _wait_after_submit(tab, cfg.success_text, cfg.post_login_wait_ms)

        page_text = _page_text(tab)
        low = page_text.lower()
        for marker in CHANGE_FAILURE_MARKERS:
            if marker in low:
                raise LoginFailed(
                    f"{service} rejected the password change ({marker}) — "
                    "the vault secret was NOT touched")
        if cfg.success_text and cfg.success_text not in page_text:
            raise LoginFailed(
                f"{service} password change unverified: "
                f"{cfg.success_text!r} not on the page — "
                "the vault secret was NOT touched")

        # The site confirmed the change: refresh cookies, then the vault.
        cookies = tab.cookies() if hasattr(tab, "cookies") else []
        jar = {c.get("name", ""): c.get("value", "")
               for c in (cookies or []) if c.get("name")}
        if jar:
            sessions.set_cookies(service, cred.username, jar)

        rotated = manager.refresh_credential(service, cred.username, fresh)
        # Record when the rotation happened for health checks / audits.
        meta = dict(rotated.metadata or {})
        meta["last_rotation_at"] = time.time()
        manager.vault.store(service, cred.username, fresh,
                            credential_type=rotated.credential_type,
                            tags=rotated.tags, metadata=meta,
                            expires_at=rotated.expires_at)

        note = (f"password rotated on {service} for {cred.username} "
                f"({len(jar)} cookies refreshed, vault updated)")
        _log.info("on-site rotation ok: %s/%s", service, cred.username)
        return {
            "ok": True,
            "service": service,
            "username": cred.username,
            "rotated": True,
            "note": note,
        }
    finally:
        try:
            tab.close()
        except Exception:  # noqa: BLE001 — teardown best-effort
            pass
