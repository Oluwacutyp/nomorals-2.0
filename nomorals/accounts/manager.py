"""High-level account management.

Provides a unified interface for working with accounts across different services.
Handles common patterns like OAuth token refresh, credential rotation, and account health checks.

Usage:
    manager = AccountManager(vault)
    
    # Get credentials for a service
    email_cred = manager.get_credential("gmail", "bot@example.com")
    
    # Check if credentials are expired
    if manager.is_expired("gmail", "bot@example.com"):
        manager.refresh_credential("gmail", "bot@example.com")
    
    # List all accounts
    for account in manager.list_accounts():
        print(f"{account.service}: {account.username}")
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..core.errors import NotFound
from ..core.logging_setup import get_logger
from .vault import AccountProfile, Credential, CredentialVault

__all__ = ["AccountManager", "AccountInfo", "estimate_secret_strength"]


def estimate_secret_strength(secret: str) -> dict[str, Any]:
    """Estimate a secret's strength (Bitwarden-report style).

    Shannon entropy per character × length gives an effective bit
    count; charset variety and length feed a human label. No network,
    no wordlists — a fast offline heuristic.

    Returns ``{"bits", "label", "length", "charset_size"}`` where label
    is one of ``weak`` / ``fair`` / ``strong`` / ``very_strong``.
    """
    if not secret:
        return {"bits": 0.0, "label": "weak", "length": 0,
                "charset_size": 0}
    charset = 0
    if any(c.islower() for c in secret):
        charset += 26
    if any(c.isupper() for c in secret):
        charset += 26
    if any(c.isdigit() for c in secret):
        charset += 10
    if any(not c.isalnum() for c in secret):
        charset += 32
    if charset == 0:
        charset = 1
    # Shannon entropy of the actual string (catches "aaaaaaa..." and
    # other low-entropy shapes a charset×length estimate would miss).
    from collections import Counter
    counts = Counter(secret)
    length = len(secret)
    shannon = -sum((c / length) * math.log2(c / length)
                   for c in counts.values())
    bits = min(shannon * length, math.log2(charset) * length)
    if bits < 40:
        label = "weak"
    elif bits < 60:
        label = "fair"
    elif bits < 80:
        label = "strong"
    else:
        label = "very_strong"
    return {"bits": round(bits, 1), "label": label, "length": length,
            "charset_size": charset}

_log = get_logger(__name__)

#: Service-name prefix the connector framework uses in the vault.
CONNECTOR_SERVICE_PREFIX = "connector:"


@dataclass
class AccountInfo:
    """Summary information about an account."""
    
    service: str
    username: str
    credential_type: str
    is_active: bool
    is_expired: bool
    last_used: Optional[float]
    use_count: int
    tags: list[str]


class AccountManager:
    """High-level account management interface.
    
    Wraps CredentialVault with business logic for account lifecycle management.
    """
    
    def __init__(self, vault: CredentialVault) -> None:
        self.vault = vault
        self._ensure_defaults_schema()
        _log.info("Account manager initialized")

    def _ensure_defaults_schema(self) -> None:
        """Create the per-service default-account table if missing."""
        with self.vault.db.transaction():
            self.vault.db.execute("""
                CREATE TABLE IF NOT EXISTS account_defaults (
                    service TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

    # ── default account per service (account switching) ──────────────────
    #
    # Several usernames can live in the vault for one service; the default
    # is the one automation reaches for when the caller doesn't pin one.

    def set_default(self, service: str, username: str) -> None:
        """Make ``username`` the default account for ``service``.

        Fails fast when no such credential exists — a default must always
        point at something real.
        """
        service = (service or "").strip()
        username = (username or "").strip()
        if not service or not username:
            raise ValueError("set_default needs a service and a username")
        # Fail fast on unknown credentials (no usage mark — this is admin).
        self.vault.get(service, username, mark_used=False)
        with self.vault.db.transaction():
            self.vault.db.execute(
                "INSERT OR REPLACE INTO account_defaults "
                "(service, username, updated_at) VALUES (?, ?, ?)",
                (service, username, time.time()),
            )
        _log.info("default account for %s -> %s", service, username)

    def get_default(self, service: str) -> str | None:
        """The default username for ``service``, or None when unset."""
        row = self.vault.db.query_one(
            "SELECT username FROM account_defaults WHERE service = ?",
            ((service or "").strip(),),
        )
        return row["username"] if row else None

    def clear_default(self, service: str) -> bool:
        """Remove the default for ``service``. Returns True when one existed."""
        with self.vault.db.transaction():
            cur = self.vault.db.execute(
                "DELETE FROM account_defaults WHERE service = ?",
                ((service or "").strip(),),
            )
        cleared = cur.rowcount > 0
        if cleared:
            _log.info("cleared default account for %s", service)
        return cleared

    def default_credential(self, service: str) -> Credential | None:
        """The default account's credential for ``service`` (decrypted).

        Returns None when no default is set; raises NotFound when the
        default points at a credential that no longer exists (stale
        pointer — clear it with :meth:`clear_default`).
        """
        username = self.get_default(service)
        if username is None:
            return None
        return self.vault.get(service, username)

    def resolve_account(self, service: str,
                        username: str | None = None) -> Credential:
        """Pin ``username``, else the service default, else the single
        stored account — the "which account?" decision in one call.

        Raises NotFound when nothing resolves.
        """
        service = (service or "").strip()
        if username:
            return self.vault.get(service, username)
        default = self.get_default(service)
        if default:
            return self.vault.get(service, default)
        creds = self.vault.list_all(service=service, active_only=True)
        if len(creds) == 1:
            return self.vault.get(service, creds[0].username)
        if not creds:
            raise NotFound(f"no account stored for service {service!r}")
        raise NotFound(
            f"{len(creds)} accounts stored for {service!r} and no default "
            "is set — set one with set_default() or pin a username")

    # ── credential rotation ──────────────────────────────────────────────

    def rotate_credential_auto(
        self,
        service: str,
        username: str,
        *,
        length: int = 32,
        symbols: bool = True,
    ) -> Credential:
        """Generate a fresh cryptographically-secure password and rotate the
        vault credential to it. Returns the updated Credential.

        This is the "expired password → generate new + update vault" step,
        automated. The new secret lives only in the vault — applying it on
        the service itself (the site's "change password" form) is the
        caller's job, because every service's rotation flow differs.
        Pair with :mod:`nomorals.accounts.browser_login` when the service
        exposes its password-change page over the web.
        """
        from .creator import generate_password

        new_password = generate_password(length=length, symbols=symbols)
        cred = self.vault.rotate(service, username, new_password)
        _log.info("auto-rotated credential: %s/%s", service, username)
        return cred
    
    def get_credential(self, service: str, username: str) -> Credential:
        """Get credentials for a specific account.
        
        Args:
            service: Service name (e.g., "gmail", "github")
            username: Username or identifier
            
        Returns:
            Credential object with decrypted password
        """
        return self.vault.get(service, username)
    
    def is_expired(self, service: str, username: str) -> bool:
        """Check if a credential is expired.
        
        Args:
            service: Service name
            username: Username or identifier
            
        Returns:
            True if credential has expired
        """
        try:
            cred = self.vault.get(service, username, mark_used=False)
            return cred.is_expired()
        except NotFound:
            # No credential stored — nothing to be expired; routine, not an error.
            _log.debug("credential expiry check: no credential for %s/%s", service, username)
            return False
        except Exception as e:
            # Vault/decryption failures here fail open (treated as "not expired"),
            # so they must be visible, not buried at debug.
            _log.warning("credential expiry check failed for %s/%s: %s", service, username, e)
            return False
    
    def refresh_credential(
        self,
        service: str,
        username: str,
        new_password: str,
        *,
        expires_at: float | None = None,
    ) -> Credential:
        """Refresh/update a credential with new password/key.
        
        Args:
            service: Service name
            username: Username or identifier
            new_password: New password/key/token
            expires_at: Optional new expiry timestamp
            
        Returns:
            Updated Credential object
        """
        cred = self.vault.rotate(service, username, new_password)
        if expires_at is not None:
            # Update expiry if provided
            with self.vault.db.transaction():
                self.vault.db.execute(
                    "UPDATE credentials SET expires_at = ? WHERE id = ?",
                    (expires_at, cred.id)
                )
            cred.expires_at = expires_at
        _log.info(f"Refreshed credential: {service}/{username}")
        return cred
    
    def list_accounts(
        self,
        *,
        service: str | None = None,
        tag: str | None = None,
        active_only: bool = True,
    ) -> list[AccountInfo]:
        """List all accounts with summary information.
        
        Args:
            service: Filter by service name
            tag: Filter by tag
            active_only: If True, only return active accounts
            
        Returns:
            List of AccountInfo objects
        """
        credentials = self.vault.list_all(
            service=service,
            tag=tag,
            active_only=active_only,
        )
        
        accounts = []
        for cred in credentials:
            accounts.append(AccountInfo(
                service=cred.service,
                username=cred.username,
                credential_type=cred.credential_type,
                is_active=cred.is_active,
                is_expired=cred.is_expired(),
                last_used=cred.last_used,
                use_count=cred.use_count,
                tags=cred.tags,
            ))
        
        return accounts
    
    def get_profile(self, profile_name: str) -> AccountProfile:
        """Get an account profile with all associated credentials.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            AccountProfile object
        """
        return self.vault.get_profile(profile_name)
    
    def deactivate_account(self, service: str, username: str) -> None:
        """Deactivate an account without deleting it.
        
        Args:
            service: Service name
            username: Username or identifier
        """
        self.vault.deactivate(service, username)
    
    def delete_account(self, service: str, username: str) -> None:
        """Permanently delete an account.
        
        Args:
            service: Service name
            username: Username or identifier
        """
        self.vault.delete(service, username)
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored credentials.
        
        Returns:
            Dict with counts and metadata
        """
        all_creds = self.vault.list_all(active_only=False)
        active_creds = [c for c in all_creds if c.is_active]
        expired_creds = [c for c in active_creds if c.is_expired()]
        
        # Count by service
        services: dict[str, int] = {}
        for cred in active_creds:
            services[cred.service] = services.get(cred.service, 0) + 1
        
        # Count by type
        types: dict[str, int] = {}
        for cred in active_creds:
            types[cred.credential_type] = types.get(cred.credential_type, 0) + 1
        
        return {
            "total_credentials": len(all_creds),
            "active_credentials": len(active_creds),
            "expired_credentials": len(expired_creds),
            "services": services,
            "credential_types": types,
        }
    
    def health_check(self) -> list[dict[str, Any]]:
        """Check health of all credentials and return issues.
        
        Returns:
            List of issues found (expired credentials, unused accounts, etc.)
        """
        issues = []
        all_creds = self.vault.list_all(active_only=True)
        now = time.time()
        
        for cred in all_creds:
            # Check for expired credentials
            if cred.is_expired():
                issues.append({
                    "type": "expired",
                    "service": cred.service,
                    "username": cred.username,
                    "message": f"Credential expired at {time.ctime(cred.expires_at)}",
                    "severity": "high",
                })
            
            # Check for unused credentials (not used in 30 days)
            if cred.last_used and (now - cred.last_used) > 30 * 24 * 3600:
                issues.append({
                    "type": "unused",
                    "service": cred.service,
                    "username": cred.username,
                    "message": f"Not used in {(now - cred.last_used) / 86400:.1f} days",
                    "severity": "low",
                })
            
            # Check for credentials expiring soon (within 7 days)
            if cred.expires_at and (cred.expires_at - now) < 7 * 24 * 3600:
                issues.append({
                    "type": "expiring_soon",
                    "service": cred.service,
                    "username": cred.username,
                    "message": f"Expires in {(cred.expires_at - now) / 86400:.1f} days",
                    "severity": "medium",
                })

        # Secret-quality issues (Bitwarden-report style): weak and
        # reused secrets. Decryption happens here, in memory only —
        # issue records never carry the secret itself.
        for cred in all_creds:
            try:
                full = self.vault.get(cred.service, cred.username,
                                      mark_used=False)
            except Exception:  # noqa: BLE001 — undecryptable: skip
                continue
            secret = full.password or ""
            strength = estimate_secret_strength(secret)
            if strength["label"] == "weak" and secret:
                issues.append({
                    "type": "weak_secret",
                    "service": cred.service,
                    "username": cred.username,
                    "message": (f"Weak secret (~{strength['bits']:.0f} bits, "
                                f"{strength['length']} chars)"),
                    "severity": "medium",
                })
        for group in self.reused_secrets():
            accts = ", ".join(
                f"{a['service']}/{a['username']}" for a in group["accounts"])
            for acct in group["accounts"]:
                issues.append({
                    "type": "reused_secret",
                    "service": acct["service"],
                    "username": acct["username"],
                    "message": (f"Secret reused across {group['count']} "
                                f"accounts ({accts})"),
                    "severity": "high",
                })

        return issues


    def reused_secrets(self) -> list[dict[str, Any]]:
        """Find secrets shared by more than one active credential.

        Returns groups of ``{"secret_sha256" (truncated), "count",
        "accounts": [{"service", "username"}]}`` — the secret itself
        never leaves this method.
        """
        buckets: dict[str, list[dict[str, str]]] = {}
        for summary in self.vault.list_all(active_only=True):
            try:
                full = self.vault.get(summary.service, summary.username,
                                      mark_used=False)
            except Exception:  # noqa: BLE001 — skip undecryptable
                continue
            secret = full.password or ""
            if not secret:
                continue
            digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
            buckets.setdefault(digest, []).append({
                "service": summary.service,
                "username": summary.username,
            })
        return [
            {"secret_sha256": digest[:16], "count": len(accts),
             "accounts": accts}
            for digest, accts in buckets.items()
            if len(accts) > 1
        ]

    # ── presentation ─────────────────────────────────────────────────
    #
    # Machine-readable dicts stay the automation surface; these
    # renderers are the human surface — styled for chat.

    def render_account_board(
        self,
        *,
        service: str | None = None,
        tag: str | None = None,
    ) -> str:
        """A styled vault dashboard — accounts grouped by service with
        status dots, usage, and expiry countdowns.

        Example::

            🔐 VAULT — 4 accounts across 3 services
            ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
            📦 github (2)
              🟢  devon-bot · used 12× · 3d ago · password
              🟡  old-ci · api_key
            📦 gmail (1)
              🔴  bot@example.com · EXPIRED
        """
        accounts = self.list_accounts(service=service, tag=tag,
                                      active_only=False)
        now = time.time()
        by_service: dict[str, list[AccountInfo]] = {}
        for a in accounts:
            by_service.setdefault(a.service, []).append(a)

        def _status(a: AccountInfo) -> tuple[str, str]:
            if not a.is_active:
                return "⛔", "disabled"
            if a.is_expired:
                return "🔴", "EXPIRED"
            return "🟢", "ok"

        lines = [f"🔐 VAULT — {len(accounts)} account(s) across "
                 f"{len(by_service)} service(s)",
                 "┈" * 46]
        for svc in sorted(by_service):
            group = sorted(by_service[svc],
                           key=lambda a: (not a.is_active, a.username))
            lines.append(f"📦 {svc} ({len(group)})")
            for a in group:
                dot, label = _status(a)
                if a.is_active and not a.is_expired:
                    bits = []
                    if a.use_count:
                        bits.append(f"used {a.use_count}×")
                    if a.last_used:
                        ago = (now - a.last_used) / 86400
                        bits.append(f"{ago:.0f}d ago" if ago >= 1
                                    else "today")
                    bits.append(a.credential_type)
                    detail = " · ".join(bits)
                else:
                    detail = label
                lines.append(f"  {dot}  {a.username} · {detail}")
        if not accounts:
            lines.append("(empty — no accounts stored)")
        return "\n".join(lines)

    # ── connector-vault integration ──────────────────────────────
    # Connectors store their credentials in this same vault under the
    # service name "connector:<id>" (see nomorals/connectors/base.py).
    # These helpers let account tooling manage connector credentials
    # without reaching into connector internals.

    @staticmethod
    def connector_service_name(connector_id: str) -> str:
        """Vault service name for a connector's credentials."""
        return f"{CONNECTOR_SERVICE_PREFIX}{connector_id}"

    def register_connector_credential(
        self,
        connector_id: str,
        username: str,
        secret: str,
        *,
        credential_type: str = "api_key",
        scopes: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: float | None = None,
    ) -> Credential:
        """Store a connector's credential in the vault.

        Args:
            connector_id: Connector id (e.g. "github", "mono")
            username: Credential identifier (key id, login, ...)
            secret: The secret itself (encrypted at rest)
            credential_type: Credential kind (api_key, oauth_token, ...)
            scopes: OAuth scopes, recorded in metadata when given
            metadata: Extra metadata
            expires_at: Optional expiry timestamp

        Returns:
            The stored Credential
        """
        meta = dict(metadata or {})
        if scopes:
            meta["scopes"] = list(scopes)
        cred = self.vault.store(
            service=self.connector_service_name(connector_id),
            username=username,
            password=secret,
            credential_type=credential_type,
            tags=["connector", connector_id],
            metadata=meta,
            expires_at=expires_at,
        )
        _log.info("registered connector credential: %s/%s",
                  connector_id, username)
        return cred

    def connector_credential(
        self,
        connector_id: str,
        *,
        username: str | None = None,
    ) -> Credential | None:
        """Fetch a connector's stored credential (decrypted).

        Args:
            connector_id: Connector id
            username: Pin to one username; otherwise the first active,
                non-expired credential wins

        Returns:
            Credential or None when the connector has nothing stored
        """
        service = self.connector_service_name(connector_id)
        creds = self.vault.list_all(service=service, active_only=True)
        if username is not None:
            creds = [c for c in creds if c.username == username]
        for summary in creds:
            try:
                cred = self.vault.get(service, summary.username)
            except Exception as e:  # pragma: no cover - defensive
                _log.warning("cannot decrypt %s/%s: %s",
                             service, summary.username, e)
                continue
            if not cred.is_expired():
                return cred
        return None

    def list_connector_credentials(
        self,
        *,
        connector_id: str | None = None,
        active_only: bool = True,
    ) -> list[AccountInfo]:
        """List credentials owned by connectors.

        Args:
            connector_id: Restrict to one connector
            active_only: Only active credentials

        Returns:
            List of AccountInfo for connector:* services
        """
        service = (self.connector_service_name(connector_id)
                   if connector_id else None)
        creds = self.vault.list_all(service=service, active_only=active_only)
        return [
            AccountInfo(
                service=c.service,
                username=c.username,
                credential_type=c.credential_type,
                is_active=c.is_active,
                is_expired=c.is_expired(),
                last_used=c.last_used,
                use_count=c.use_count,
                tags=c.tags,
            )
            for c in creds
            if c.service.startswith(CONNECTOR_SERVICE_PREFIX)
        ]

    def revoke_connector_credentials(self, connector_id: str) -> int:
        """Delete every stored credential for a connector.

        Returns:
            Number of credentials removed
        """
        service = self.connector_service_name(connector_id)
        creds = self.vault.list_all(service=service, active_only=False)
        for cred in creds:
            self.vault.delete(service, cred.username)
        _log.info("revoked %d credential(s) for connector %s",
                  len(creds), connector_id)
        return len(creds)

    def connector_summary(self) -> dict[str, dict[str, Any]]:
        """Per-connector credential status.

        Returns:
            Mapping connector_id -> {service, usernames, active, expired,
            credential_types}
        """
        summary: dict[str, dict[str, Any]] = {}
        for info in self.list_connector_credentials(active_only=False):
            cid = info.service[len(CONNECTOR_SERVICE_PREFIX):]
            entry = summary.setdefault(cid, {
                "service": info.service,
                "usernames": [],
                "active": 0,
                "expired": 0,
                "credential_types": set(),
            })
            entry["usernames"].append(info.username)
            if info.is_active:
                entry["active"] += 1
            if info.is_expired:
                entry["expired"] += 1
            entry["credential_types"].add(info.credential_type)
        for entry in summary.values():
            entry["credential_types"] = sorted(entry["credential_types"])
        return summary
