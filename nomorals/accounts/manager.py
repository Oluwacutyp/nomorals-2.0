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
        except Exception:
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
