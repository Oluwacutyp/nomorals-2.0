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


def _wait_after_submit(tab: Any, cfg: LoginConfig) -> None:
    """Prefer an explicit wait when the tab supports it; sleep otherwise."""
    if cfg.success_text and hasattr(tab, "wait_for_text"):
        try:
            tab.wait_for_text(cfg.success_text,
                              timeout=cfg.post_login_wait_ms)
            return
        except Exception as exc:  # noqa: BLE001 — fall through to sleep
            _log.debug("explicit wait_for_text missed: %s", exc)
    time.sleep(max(0.5, cfg.post_login_wait_ms / 1000.0))


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
        _wait_after_submit(tab, cfg)

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
            _wait_after_submit(tab, cfg)

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
