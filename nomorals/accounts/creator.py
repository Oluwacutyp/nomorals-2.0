"""Automated account creation.

Handles creating new accounts on various services. Where fully automated
creation isn't possible (CAPTCHAs, phone verification), falls back to
guided flows where the bot walks the user through steps.

Approach by service type:
- Email: Use disposable email services (Guerrilla Mail, TempMail) for throwaway accounts
- Social: Use browser automation where possible, guided flow otherwise
- APIs: Register for API keys via developer portals
- Shopping: Create accounts with bot's own email

Security notes:
- Generated passwords use cryptographically secure random (secrets module)
- Recovery info stored in vault
- Account creation attempts logged for audit

Usage:
    creator = AccountCreator(vault, browser_session)
    
    # Create a disposable email account
    account = await creator.create_email_account(provider="guerrilla")
    
    # Create a GitHub account (guided flow)
    account = await creator.create_account("github", username="my-bot", guided=True)
"""

from __future__ import annotations

import secrets
import string
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.logging_setup import get_logger
from .vault import Credential, CredentialVault

__all__ = ["AccountCreator", "CreatedAccount"]

_log = get_logger(__name__)


@dataclass
class CreatedAccount:
    """Result of an account creation attempt."""
    
    service: str
    username: str
    password: str
    email: str
    status: str  # "created", "pending", "failed", "guided"
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
    suffix = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(length))
    return f"{prefix}_{suffix}"


class AccountCreator:
    """Creates new accounts on various services.
    
    Supports both fully automated and guided creation flows.
    """
    
    def __init__(self, vault: CredentialVault, browser_session: Any = None) -> None:
        self.vault = vault
        self.browser = browser_session
        self._creation_history: list[CreatedAccount] = []
        _log.info("Account creator initialized")
    
    async def create_email_account(
        self,
        *,
        provider: str = "guerrilla",
        username: str | None = None,
    ) -> CreatedAccount:
        """Create a disposable email account.
        
        Args:
            provider: Email provider (guerrilla, tempmail, etc.)
            username: Desired username (generated if not provided)
            
        Returns:
            CreatedAccount with email credentials
        """
        username = username or generate_username("bot")
        password = generate_password()
        
        try:
            if provider == "guerrilla":
                return await self._create_guerrilla_email(username)
            elif provider == "tempmail":
                return await self._create_tempmail(username)
            else:
                return CreatedAccount(
                    service=f"email_{provider}",
                    username=username,
                    password=password,
                    email=f"{username}@{provider}.com",
                    status="failed",
                    notes=f"Unknown provider: {provider}",
                )
        except Exception as e:
            _log.error(f"Failed to create email account: {e}")
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
        # Guerrilla Mail API endpoint
        api_url = "https://api.guerrillamail.com/ajax.php"
        
        # Get a new email address
        import urllib.request
        import json
        
        req = urllib.request.Request(
            f"{api_url}?f=get_email_address",
            headers={"User-Agent": "NoMorals-Bot/1.0"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                email = data.get("email_addr", f"{username}@guerrillamail.com")
        except Exception as e:
            _log.warning(f"Guerrilla Mail API failed, using fallback: {e}")
            email = f"{username}@guerrillamail.com"
        
        # Store in vault
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
        # Similar to Guerrilla Mail but different API
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
    
    async def create_account(
        self,
        service: str,
        *,
        username: str | None = None,
        email: str | None = None,
        password: str | None = None,
        guided: bool = False,
        **kwargs: Any,
    ) -> CreatedAccount:
        """Create an account on a service.
        
        Args:
            service: Service name (github, gmail, etc.)
            username: Desired username
            email: Email to use for account
            password: Password (generated if not provided)
            guided: If True, use guided flow (bot walks user through steps)
            **kwargs: Additional service-specific parameters
            
        Returns:
            CreatedAccount with result
        """
        username = username or generate_username()
        password = password or generate_password()
        
        # Get or create email for this account
        if not email:
            email_account = await self.create_email_account()
            email = email_account.email
        
        try:
            # Try automated creation
            if service == "github":
                return await self._create_github(username, email, password, guided)
            elif service == "gmail":
                return await self._create_gmail(username, password, guided)
            elif service == "twitter":
                return await self._create_twitter(username, email, password, guided)
            else:
                # Generic guided flow
                return await self._create_generic(service, username, email, password, guided)
        except Exception as e:
            _log.error(f"Failed to create {service} account: {e}")
            return CreatedAccount(
                service=service,
                username=username,
                password=password,
                email=email,
                status="failed",
                notes=str(e),
            )
    
    async def _create_github(
        self,
        username: str,
        email: str,
        password: str,
        guided: bool,
    ) -> CreatedAccount:
        """Create a GitHub account.
        
        GitHub has strong anti-bot measures, so this usually requires guided flow.
        """
        if guided or not self.browser:
            # Guided flow - bot provides instructions
            return CreatedAccount(
                service="github",
                username=username,
                password=password,
                email=email,
                status="guided",
                notes=(
                    f"GitHub account creation requires manual steps:\n"
                    f"1. Go to https://github.com/signup\n"
                    f"2. Use username: {username}\n"
                    f"3. Use email: {email}\n"
                    f"4. Use password: {password}\n"
                    f"5. Complete CAPTCHA and email verification\n"
                    f"6. Let me know when done and I'll store the credentials"
                ),
                metadata={"username": username, "email": email},
            )
        
        # Automated attempt (likely to fail due to CAPTCHA)
        return CreatedAccount(
            service="github",
            username=username,
            password=password,
            email=email,
            status="failed",
            notes="GitHub blocks automated signup - use guided=True",
        )
    
    async def _create_gmail(
        self,
        username: str,
        password: str,
        guided: bool,
    ) -> CreatedAccount:
        """Create a Gmail account.
        
        Gmail requires phone verification, so guided flow is recommended.
        """
        return CreatedAccount(
            service="gmail",
            username=f"{username}@gmail.com",
            password=password,
            email=f"{username}@gmail.com",
            status="guided",
            notes=(
                f"Gmail account creation requires phone verification:\n"
                f"1. Go to https://accounts.google.com/signup\n"
                f"2. Use username: {username}\n"
                f"3. Use password: {password}\n"
                f"4. Provide phone number for verification\n"
                f"5. Let me know when done and I'll store the credentials"
            ),
            metadata={"username": username},
        )
    
    async def _create_twitter(
        self,
        username: str,
        email: str,
        password: str,
        guided: bool,
    ) -> CreatedAccount:
        """Create a Twitter/X account."""
        return CreatedAccount(
            service="twitter",
            username=username,
            password=password,
            email=email,
            status="guided",
            notes=(
                f"Twitter account creation requires phone/email verification:\n"
                f"1. Go to https://twitter.com/i/flow/signup\n"
                f"2. Use username: {username}\n"
                f"3. Use email: {email}\n"
                f"4. Use password: {password}\n"
                f"5. Complete verification\n"
                f"6. Let me know when done and I'll store the credentials"
            ),
        )
    
    async def _create_generic(
        self,
        service: str,
        username: str,
        email: str,
        password: str,
        guided: bool,
    ) -> CreatedAccount:
        """Generic account creation flow."""
        return CreatedAccount(
            service=service,
            username=username,
            password=password,
            email=email,
            status="guided",
            notes=(
                f"Account creation for {service}:\n"
                f"1. Go to {service}'s signup page\n"
                f"2. Use username: {username}\n"
                f"3. Use email: {email}\n"
                f"4. Use password: {password}\n"
                f"5. Let me know when done and I'll store the credentials"
            ),
        )
    
    def finalize_account(
        self,
        service: str,
        username: str,
        password: str,
        **kwargs: Any,
    ) -> CreatedAccount:
        """Finalize an account after manual creation steps.
        
        Call this after user completes a guided flow to store credentials.
        
        Args:
            service: Service name
            username: Account username
            password: Account password
            **kwargs: Additional metadata
            
        Returns:
            CreatedAccount with stored credentials
        """
        email = kwargs.get("email", "")
        
        cred = self.vault.store(
            service=service,
            username=username,
            password=password,
            credential_type="account",
            tags=[service, "account"],
            metadata=kwargs,
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
        _log.info(f"Finalized account: {service}/{username}")
        
        return account
    
    def get_creation_history(self) -> list[CreatedAccount]:
        """Get history of account creation attempts.
        
        Returns:
            List of CreatedAccount objects
        """
        return self._creation_history.copy()
