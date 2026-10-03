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

import time
from dataclasses import dataclass
from typing import Any, Optional

from ..core.logging_setup import get_logger
from .vault import AccountProfile, Credential, CredentialVault

__all__ = ["AccountManager", "AccountInfo"]

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
        _log.info("Account manager initialized")
    
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
        except Exception as e:
            _log.debug("credential expiry check failed: %s", e)
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
        
        return issues


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
