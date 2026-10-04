"""Account health checks.

Periodic answer to: is the account still active? Locked? Does it need
verification? Has its session died server-side?

Each check combines three layers and never raises for a single account
(a sweep over fifty accounts must not die on account three):

1. **Vault** — does the credential exist, is it active, is it expired?
2. **Session store** — :meth:`SessionManager.peek_session` so the check
   itself never masks staleness, plus
   :meth:`Session.health_report` for the local view.
3. **Server probe** (optional) — fetch a logged-in-only URL with the
   session's cookies and classify the page: logged in, logged out,
   locked, or asking for verification.

Usage:
    from nomorals.accounts.health import check_account_health, check_all_health

    report = check_account_health(manager, sessions, "github",
                                  probe_url="https://github.com/settings/profile")
    report.status          # "healthy" | "locked" | "needs_verification" | ...
    report.to_dict()       # JSON-safe

Layering: L2, duck-typed on the manager/sessions objects — same style
as :mod:`nomorals.accounts.browser_login`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import NotFound
from ..core.logging_setup import get_logger

__all__ = [
    "AccountHealth",
    "check_account_health",
    "check_all_health",
    "LOCKED_MARKERS",
    "VERIFICATION_MARKERS",
    "LOGGED_OUT_MARKERS",
]

_log = get_logger(__name__)

#: page text suggesting the account itself is locked/suspended.
LOCKED_MARKERS = (
    "account locked",
    "account has been locked",
    "temporarily locked",
    "your account is locked",
    "account suspended",
    "account has been suspended",
    "this account has been suspended",
    "suspended for violating",
)

#: page text suggesting the account needs a verification step.
VERIFICATION_MARKERS = (
    "verify your account",
    "verify your identity",
    "verification required",
    "complete verification",
    "confirm it's you",
    "confirm it is you",
    "unusual sign-in",
    "unusual login",
    "we noticed a new sign-in",
    "enter the code we sent",
    "two-step verification",
    "2-step verification",
)

#: page text suggesting the session is simply logged out (not locked).
LOGGED_OUT_MARKERS = (
    "log in to continue",
    "login to continue",
    "sign in to continue",
    "please log in",
    "please sign in",
    "session expired",
    "you have been logged out",
    "you've been logged out",
)


@dataclass
class AccountHealth:
    """Structured health verdict for one account."""

    service: str
    username: str
    status: str = "unknown"
    credential_active: bool = False
    credential_expired: bool = False
    session_status: str = "none"
    probe_status: str | None = None
    issues: list[str] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (no secrets — never any)."""
        return {
            "service": self.service,
            "username": self.username,
            "status": self.status,
            "credential_active": self.credential_active,
            "credential_expired": self.credential_expired,
            "session_status": self.session_status,
            "probe_status": self.probe_status,
            "issues": list(self.issues),
            "checked_at": self.checked_at,
        }

    @property
    def ok(self) -> bool:
        """True when the account is fully usable."""
        return self.status == "healthy"


def check_account_health(
    manager: Any,
    sessions: Any,
    service: str,
    username: str | None = None,
    *,
    probe_url: str = "",
    ok_markers: tuple[str, ...] | list[str] = (),
    timeout: float = 15.0,
) -> AccountHealth:
    """Check one account's health end to end.

    Args:
        manager: AccountManager (duck-typed)
        sessions: SessionManager (duck-typed)
        service: Service name
        username: Pin an account; otherwise the service default / single
            stored account wins (same resolution as login)
        probe_url: Optional logged-in-only URL for the server-side check
        ok_markers: Text that must appear on the probe page when the
            session is accepted
        timeout: Probe HTTP timeout in seconds

    Returns:
        AccountHealth with status in {"healthy", "degraded", "expired",
        "locked", "needs_verification", "logged_out", "disabled",
        "missing", "unknown"}. Never raises for a bad account — a
        missing credential becomes status "missing".
    """
    service = (service or "").strip()
    report = AccountHealth(service=service, username=username or "")

    try:
        cred = manager.resolve_account(service, username)
    except NotFound:
        report.status = "missing"
        report.issues.append(f"no credential stored for {service!r}")
        return report
    except Exception as exc:  # noqa: BLE001 — fail soft per account
        report.status = "unknown"
        report.issues.append(f"credential lookup failed: {exc}")
        return report

    report.username = cred.username
    report.credential_active = bool(cred.is_active)
    report.credential_expired = bool(cred.is_expired())

    if not cred.is_active:
        report.status = "disabled"
        report.issues.append("credential deactivated in the vault")
        return report

    # Local session view — peek, never touch (a health check must not
    # mask staleness by refreshing last_used).
    session = sessions.peek_session(service, cred.username)
    if session is None:
        report.session_status = "none"
        report.issues.append("no session stored — login required")
    else:
        health = session.health_report()
        report.session_status = health["status"]
        for reason in health["reasons"]:
            report.issues.append(f"session {reason.replace('_', ' ')}")

    # Server-side probe when a URL is available.
    if probe_url and session is not None:
        probe = sessions.probe(
            service, cred.username, probe_url,
            ok_markers=ok_markers,
            bad_markers=list(LOGGED_OUT_MARKERS),
            marker_status={
                "locked": LOCKED_MARKERS,
                "needs_verification": VERIFICATION_MARKERS,
            },
            timeout=timeout,
        )
        report.probe_status = probe["status"]
        if probe["status"] == "logged_in":
            pass  # server agrees — nothing to add
        elif probe["status"] == "logged_out":
            report.issues.append(
                f"server rejected the session ({probe['reason']})")
        else:
            report.issues.append(
                f"probe inconclusive ({probe['reason']})")

    # Verdict, worst-first.
    if report.credential_expired:
        report.status = "expired"
        report.issues.append("vault credential past its expiry")
    elif report.probe_status == "locked":
        report.status = "locked"
    elif report.probe_status == "needs_verification":
        report.status = "needs_verification"
    elif report.probe_status == "logged_out":
        report.status = "logged_out"
    elif report.session_status in ("expired", "stale", "empty", "none"):
        report.status = "degraded"
    elif report.probe_status == "unknown":
        report.status = "degraded"
    else:
        report.status = "healthy"

    _log.info("health %s/%s: %s (%d issue(s))",
              service, cred.username, report.status, len(report.issues))
    return report


def _metadata_probe_hints(manager: Any, service: str,
                        username: str) -> tuple[str, list[str]]:
    """Probe URL/markers configured on the credential's metadata
    (``health_probe_url``, ``health_ok_markers``) — the configure-once
    hook periodic sweeps use when no explicit probe_urls are given."""
    try:
        vault = getattr(manager, "vault", None)
        if vault is None:
            return "", []
        cred = vault.get(service, username, mark_used=False)
        meta = cred.metadata or {}
    except Exception:  # noqa: BLE001 — hints are optional
        return "", []
    url = str(meta.get("health_probe_url", "") or "")
    markers = list(meta.get("health_ok_markers", []) or [])
    return url, markers


def check_all_health(
    manager: Any,
    sessions: Any,
    *,
    probe_urls: dict[str, str] | None = None,
    ok_markers: dict[str, list[str]] | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Sweep every stored account and roll up a summary.

    Args:
        manager: AccountManager (duck-typed)
        sessions: SessionManager (duck-typed)
        probe_urls: Optional mapping service -> logged-in-only probe URL
        ok_markers: Optional mapping service -> required page markers
        timeout: Probe HTTP timeout in seconds

    Returns:
        ``{"checked_at", "accounts": [AccountHealth.to_dict() ...],
        "summary": {status: count, ...}, "needs_attention": [...]}``
        where ``needs_attention`` lists the non-healthy accounts.
    """
    probe_urls = probe_urls or {}
    ok_markers = ok_markers or {}
    checked_at = time.time()

    accounts: list[dict[str, Any]] = []
    for info in manager.list_accounts(active_only=False):
        probe_url = probe_urls.get(info.service, "")
        markers = list(ok_markers.get(info.service, ()))
        if not probe_url:
            # Fall back to per-credential configured probe hints.
            hint_url, hint_markers = _metadata_probe_hints(
                manager, info.service, info.username)
            probe_url = probe_url or hint_url
            markers = markers or hint_markers
        try:
            report = check_account_health(
                manager, sessions, info.service, info.username,
                probe_url=probe_url,
                ok_markers=markers,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 — sweep never dies
            _log.warning("health sweep failed for %s/%s: %s",
                         info.service, info.username, exc)
            report = AccountHealth(
                service=info.service, username=info.username,
                status="unknown", issues=[f"check crashed: {exc}"])
        accounts.append(report.to_dict())

    summary: dict[str, int] = {}
    needs_attention: list[dict[str, Any]] = []
    for entry in accounts:
        summary[entry["status"]] = summary.get(entry["status"], 0) + 1
        if entry["status"] != "healthy":
            needs_attention.append(entry)

    _log.info("health sweep: %d accounts, %d need attention",
              len(accounts), len(needs_attention))
    return {
        "checked_at": checked_at,
        "accounts": accounts,
        "summary": summary,
        "needs_attention": needs_attention,
    }
