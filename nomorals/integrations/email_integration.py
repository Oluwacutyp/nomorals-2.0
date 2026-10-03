"""Email integration with multiple backends.

Supports three methods for maximum flexibility:
1. Gmail API (OAuth) - fastest, most reliable for Gmail accounts
2. IMAP/SMTP - works with any email provider (Gmail, Outlook, Yahoo, etc.)
3. Browser automation - fallback for webmail interfaces

Usage:
    email = EmailIntegration(account_manager, session_manager)
    
    # Send an email
    await email.send(
        to="friend@example.com",
        subject="Hello",
        body="Hi there!",
        account="bot@example.com"
    )
    
    # Read inbox
    messages = await email.read_inbox(account="bot@example.com", limit=10)
    
    # Search emails
    results = await email.search(query="from:user@example.com", account="bot@example.com")
"""

from __future__ import annotations

import base64
import imaplib
import json
import smtplib
import time
from email.header import decode_header
from email.message import EmailMessage
from email.mime.text import MIMEText
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..accounts.sessions import SessionManager
from ..core.logging_setup import get_logger

__all__ = ["EmailIntegration", "EmailError", "EmailMessage"]

_log = get_logger(__name__)


class EmailError(Exception):
    """Raised when an email operation cannot be completed."""


class EmailMessage:
    """Represents an email message."""
    
    def __init__(
        self,
        message_id: str,
        from_addr: str,
        to_addrs: list[str],
        subject: str,
        body: str,
        date: float,
        cc_addrs: list[str] | None = None,
        bcc_addrs: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        is_read: bool = False,
        labels: list[str] | None = None,
    ):
        self.message_id = message_id
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.subject = subject
        self.body = body
        self.date = date
        self.cc_addrs = cc_addrs or []
        self.bcc_addrs = bcc_addrs or []
        self.attachments = attachments or []
        self.is_read = is_read
        self.labels = labels or []
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "message_id": self.message_id,
            "from": self.from_addr,
            "to": self.to_addrs,
            "cc": self.cc_addrs,
            "bcc": self.bcc_addrs,
            "subject": self.subject,
            "body": self.body,
            "date": self.date,
            "is_read": self.is_read,
            "labels": self.labels,
            "attachments": len(self.attachments),
        }


class EmailIntegration:
    """Multi-backend email integration.
    
    Automatically selects the best backend based on the email provider:
    - Gmail accounts: Gmail API (if OAuth configured) or IMAP
    - Other providers: IMAP/SMTP
    - Fallback: Browser automation
    """
    
    def __init__(
        self,
        account_manager: AccountManager,
        session_manager: SessionManager,
    ) -> None:
        self.account_manager = account_manager
        self.session_manager = session_manager
        _log.info("Email integration initialized")
    
    async def send(
        self,
        to: str | list[str],
        subject: str,
        body: str,
        *,
        account: str,
        cc: str | list[str] | None = None,
        bcc: str | list[str] | None = None,
        html: bool = False,
        attachments: list[dict[str, Any]] | None = None,
    ) -> str:
        """Send an email.
        
        Args:
            to: Recipient email address(es)
            subject: Email subject
            body: Email body (plain text or HTML)
            account: Sender account (email address)
            cc: CC recipients
            bcc: BCC recipients
            html: If True, body is HTML
            attachments: List of attachments (future)
            
        Returns:
            Message ID of sent email
        """
        to_addrs = [to] if isinstance(to, str) else to
        cc_addrs = ([cc] if isinstance(cc, str) else cc) if cc else []
        bcc_addrs = ([bcc] if isinstance(bcc, str) else bcc) if bcc else []
        
        # Detect backend
        backend = self._detect_backend(account)
        
        if backend == "gmail_api":
            return await self._send_gmail_api(
                to_addrs, subject, body, account, cc_addrs, bcc_addrs, html
            )
        elif backend == "imap":
            return await self._send_smtp(
                to_addrs, subject, body, account, cc_addrs, bcc_addrs, html
            )
        else:
            raise ValueError(f"No email backend available for {account}")
    
    async def read_inbox(
        self,
        account: str,
        *,
        limit: int = 10,
        folder: str = "INBOX",
        unread_only: bool = False,
    ) -> list[EmailMessage]:
        """Read emails from inbox.
        
        Args:
            account: Email account
            limit: Maximum messages to fetch
            folder: Folder to read (default: INBOX)
            unread_only: If True, only fetch unread messages
            
        Returns:
            List of EmailMessage objects
        """
        backend = self._detect_backend(account)
        
        if backend == "gmail_api":
            return await self._read_gmail_api(account, limit, folder, unread_only)
        elif backend == "imap":
            return await self._read_imap(account, limit, folder, unread_only)
        else:
            raise ValueError(f"No email backend available for {account}")
    
    async def search(
        self,
        query: str,
        account: str,
        *,
        limit: int = 10,
    ) -> list[EmailMessage]:
        """Search emails.
        
        Args:
            query: Search query (Gmail search syntax or IMAP search)
            account: Email account
            limit: Maximum results
            
        Returns:
            List of matching EmailMessage objects
        """
        backend = self._detect_backend(account)
        
        if backend == "gmail_api":
            return await self._search_gmail_api(query, account, limit)
        elif backend == "imap":
            return await self._search_imap(query, account, limit)
        else:
            raise ValueError(f"No email backend available for {account}")
    
    async def mark_read(self, message_id: str, account: str) -> None:
        """Mark an email as read.
        
        Args:
            message_id: Message ID
            account: Email account
        """
        backend = self._detect_backend(account)
        
        if backend == "gmail_api":
            await self._mark_read_gmail_api(message_id, account)
        elif backend == "imap":
            await self._mark_read_imap(message_id, account)
    
    async def delete(self, message_id: str, account: str) -> None:
        """Delete an email.
        
        Args:
            message_id: Message ID
            account: Email account
        """
        backend = self._detect_backend(account)
        
        if backend == "gmail_api":
            await self._delete_gmail_api(message_id, account)
        elif backend == "imap":
            await self._delete_imap(message_id, account)
    
    def _detect_backend(self, account: str) -> str:
        """Detect the best backend for an email account.
        
        Args:
            account: Email address
            
        Returns:
            Backend name: "gmail_api", "imap", or "browser"
        """
        # Check if this is a Gmail account with OAuth configured
        if account.endswith("@gmail.com"):
            try:
                # Check if we have OAuth credentials
                cred = self.account_manager.get_credential("gmail_oauth", account)
                if cred.credential_type == "oauth_token":
                    return "gmail_api"
            except Exception as e:
                _log.debug("gmail oauth probe failed, falling back to imap: %s", e)
        
        # Default to IMAP/SMTP
        return "imap"
    
    # ── Gmail API Backend ──────────────────────────────────────────────────
    
    async def _send_gmail_api(
        self,
        to_addrs: list[str],
        subject: str,
        body: str,
        account: str,
        cc_addrs: list[str],
        bcc_addrs: list[str],
        html: bool,
    ) -> str:
        """Send email via Gmail API."""
        import urllib.request
        
        # Get OAuth token
        cred = self.account_manager.get_credential("gmail_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        # Create message
        message = MIMEText(body, "html" if html else "plain")
        message["to"] = ", ".join(to_addrs)
        message["cc"] = ", ".join(cc_addrs) if cc_addrs else ""
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        # Send via API
        url = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
        data = json.dumps({"raw": raw}).encode()
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                message_id = result["id"]
                _log.info(f"Sent email via Gmail API: {subject}")
                return message_id
        except Exception as e:
            _log.error(f"Failed to send via Gmail API: {e}")
            raise
    
    async def _read_gmail_api(
        self,
        account: str,
        limit: int,
        folder: str,
        unread_only: bool,
    ) -> list[EmailMessage]:
        """Read inbox via Gmail API."""
        import urllib.request
        import urllib.parse
        
        # Get OAuth token
        cred = self.account_manager.get_credential("gmail_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        # Build query
        query = f"in:{folder.lower()}"
        if unread_only:
            query += " is:unread"
        
        # List messages
        params = urllib.parse.urlencode({
            "q": query,
            "maxResults": limit,
        })
        url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages?{params}"
        
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                message_ids = [m["id"] for m in result.get("messages", [])]
        except Exception as e:
            _log.error(f"Failed to list messages: {e}")
            return []
        
        # Fetch each message
        messages = []
        for msg_id in message_ids[:limit]:
            msg = await self._get_gmail_message(msg_id, account, access_token)
            if msg:
                messages.append(msg)
        
        return messages
    
    async def _get_gmail_message(
        self,
        message_id: str,
        account: str,
        access_token: str,
    ) -> Optional[EmailMessage]:
        """Fetch a single Gmail message."""
        import urllib.request
        
        url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}?format=full"
        
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                # Parse headers
                headers = {h["name"].lower(): h["value"] for h in data["payload"]["headers"]}
                
                # Extract body
                body = ""
                if "parts" in data["payload"]:
                    for part in data["payload"]["parts"]:
                        if part["mimeType"] == "text/plain":
                            body = base64.urlsafe_b64decode(part["body"]["data"]).decode()
                            break
                elif "body" in data["payload"] and "data" in data["payload"]["body"]:
                    body = base64.urlsafe_b64decode(data["payload"]["body"]["data"]).decode()
                
                return EmailMessage(
                    message_id=message_id,
                    from_addr=headers.get("from", ""),
                    to_addrs=[headers.get("to", "")],
                    subject=headers.get("subject", ""),
                    body=body,
                    date=int(data.get("internalDate", "0")) / 1000,
                    labels=data.get("labelIds", []),
                )
        except Exception as e:
            _log.error(f"Failed to fetch message {message_id}: {e}")
            return None
    
    async def _search_gmail_api(
        self,
        query: str,
        account: str,
        limit: int,
    ) -> list[EmailMessage]:
        """Search via Gmail API."""
        # Reuse read logic with custom query
        import urllib.request
        import urllib.parse
        
        cred = self.account_manager.get_credential("gmail_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        params = urllib.parse.urlencode({"q": query, "maxResults": limit})
        url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages?{params}"
        
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                message_ids = [m["id"] for m in result.get("messages", [])]
        except Exception as e:
            _log.error(f"Search failed: {e}")
            return []
        
        messages = []
        for msg_id in message_ids[:limit]:
            msg = await self._get_gmail_message(msg_id, account, access_token)
            if msg:
                messages.append(msg)
        
        return messages
    
    async def _mark_read_gmail_api(self, message_id: str, account: str) -> None:
        """Mark message as read via Gmail API."""
        import urllib.request
        
        cred = self.account_manager.get_credential("gmail_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}/modify"
        data = json.dumps({
            "removeLabelIds": ["UNREAD"]
        }).encode()
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10):
                _log.info(f"Marked message {message_id} as read")
        except Exception as e:
            _log.error(f"Failed to mark as read: {e}")
    
    async def _delete_gmail_api(self, message_id: str, account: str) -> None:
        """Delete message via Gmail API."""
        import urllib.request
        
        cred = self.account_manager.get_credential("gmail_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}/trash"
        
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10):
                _log.info(f"Deleted message {message_id}")
        except Exception as e:
            _log.error(f"Failed to delete: {e}")
    
    # ── IMAP/SMTP Backend ──────────────────────────────────────────────────
    
    async def _send_smtp(
        self,
        to_addrs: list[str],
        subject: str,
        body: str,
        account: str,
        cc_addrs: list[str],
        bcc_addrs: list[str],
        html: bool,
    ) -> str:
        """Send email via SMTP."""
        # Get credentials
        service = account.split("@")[1].split(".")[0]  # e.g., "gmail", "yahoo"
        cred = self.account_manager.get_credential(f"email_{service}", account)
        
        # Build message
        message = MIMEText(body, "html" if html else "plain")
        message["From"] = account
        message["To"] = ", ".join(to_addrs)
        message["Cc"] = ", ".join(cc_addrs) if cc_addrs else ""
        message["Subject"] = subject
        
        # Determine SMTP server
        smtp_servers = {
            "gmail": ("smtp.gmail.com", 587),
            "yahoo": ("smtp.mail.yahoo.com", 587),
            "outlook": ("smtp-mail.outlook.com", 587),
            "hotmail": ("smtp-mail.outlook.com", 587),
        }
        
        smtp_host, smtp_port = smtp_servers.get(service, (f"smtp.{service}.com", 587))
        
        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.starttls()
                server.login(account, cred.password)
                all_recipients = to_addrs + cc_addrs + bcc_addrs
                server.sendmail(account, all_recipients, message.as_string())
                
                _log.info(f"Sent email via SMTP: {subject}")
                return f"smtp-{int(time.time())}"
        except Exception as e:
            _log.error(f"Failed to send via SMTP: {e}")
            raise
    
    async def _read_imap(
        self,
        account: str,
        limit: int,
        folder: str,
        unread_only: bool,
    ) -> list[EmailMessage]:
        """Read inbox via IMAP."""
        # Get credentials
        service = account.split("@")[1].split(".")[0]
        cred = self.account_manager.get_credential(f"email_{service}", account)
        
        # Determine IMAP server
        imap_servers = {
            "gmail": "imap.gmail.com",
            "yahoo": "imap.mail.yahoo.com",
            "outlook": "outlook.office365.com",
            "hotmail": "outlook.office365.com",
        }
        
        imap_host = imap_servers.get(service, f"imap.{service}.com")
        
        try:
            with imaplib.IMAP4_SSL(imap_host, 993) as mail:
                mail.login(account, cred.password)
                mail.select(folder)
                
                # Search for messages
                search_criteria = "UNSEEN" if unread_only else "ALL"
                status, data = mail.search(None, search_criteria)
                
                if status != "OK":
                    return []
                
                message_ids = data[0].split()
                
                # Fetch recent messages
                messages = []
                for msg_id in message_ids[-limit:]:
                    status, msg_data = mail.fetch(msg_id, "(RFC822)")
                    if status == "OK":
                        msg = self._parse_imap_message(msg_id.decode(), msg_data[0][1])
                        if msg:
                            messages.append(msg)
                
                return messages
        except Exception as e:
            _log.error(f"Failed to read IMAP: {e}")
            return []
    
    def _parse_imap_message(self, message_id: str, raw_email: bytes) -> Optional[EmailMessage]:
        """Parse a raw IMAP message."""
        import email
        
        try:
            msg = email.message_from_bytes(raw_email)
            
            # Extract headers
            from_addr = msg.get("From", "")
            to_addr = msg.get("To", "")
            subject = msg.get("Subject", "")
            date_str = msg.get("Date", "")
            
            # Parse date
            date = time.time()  # Fallback
            try:
                from email.utils import parsedate_to_datetime
                date = parsedate_to_datetime(date_str).timestamp()
            except Exception as e:
                _log.debug("unparseable email date %r: %s", date_str, e)
            
            # Extract body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        charset = part.get_content_charset() or "utf-8"
                        body = part.get_payload(decode=True).decode(charset, errors="ignore")
                        break
            else:
                charset = msg.get_content_charset() or "utf-8"
                body = msg.get_payload(decode=True).decode(charset, errors="ignore")
            
            return EmailMessage(
                message_id=message_id,
                from_addr=from_addr,
                to_addrs=[to_addr],
                subject=subject,
                body=body,
                date=date,
            )
        except Exception as e:
            _log.error(f"Failed to parse message: {e}")
            return None
    
    async def _search_imap(
        self,
        query: str,
        account: str,
        limit: int,
    ) -> list[EmailMessage]:
        """Search via IMAP."""
        # IMAP search is limited, so we'll just read and filter
        messages = await self._read_imap(account, limit * 2, "INBOX", False)
        
        # Simple text search in subject/body
        query_lower = query.lower()
        results = []
        for msg in messages:
            if query_lower in msg.subject.lower() or query_lower in msg.body.lower():
                results.append(msg)
                if len(results) >= limit:
                    break
        
        return results
    
    async def _mark_read_imap(self, message_id: str, account: str) -> None:
        """Mark message as read via IMAP."""
        service = account.split("@")[1].split(".")[0]
        cred = self.account_manager.get_credential(f"email_{service}", account)
        imap_servers = {
            "gmail": "imap.gmail.com",
            "yahoo": "imap.mail.yahoo.com",
            "outlook": "outlook.office365.com",
            "hotmail": "outlook.office365.com",
        }
        imap_host = imap_servers.get(service, f"imap.{service}.com")
        try:
            with imaplib.IMAP4_SSL(imap_host, 993) as mail:
                mail.login(account, cred.password)
                mail.select("INBOX")
                # message_id may be a sequence number or UID; try UID first
                status, _ = mail.uid("STORE", message_id, "+FLAGS", "\\Seen")
                if status != "OK":
                    status, _ = mail.store(message_id, "+FLAGS", "\\Seen")
                if status != "OK":
                    raise EmailError(f"IMAP STORE failed for {message_id}: {status}")
                _log.info(f"Marked {message_id} as read")
        except Exception as e:
            _log.error(f"Failed to mark read via IMAP: {e}")
            raise EmailError(f"IMAP mark-read failed: {e}") from e
    
    async def _delete_imap(self, message_id: str, account: str) -> None:
        """Delete message via IMAP."""
        service = account.split("@")[1].split(".")[0]
        cred = self.account_manager.get_credential(f"email_{service}", account)
        imap_servers = {
            "gmail": "imap.gmail.com",
            "yahoo": "imap.mail.yahoo.com",
            "outlook": "outlook.office365.com",
            "hotmail": "outlook.office365.com",
        }
        imap_host = imap_servers.get(service, f"imap.{service}.com")
        try:
            with imaplib.IMAP4_SSL(imap_host, 993) as mail:
                mail.login(account, cred.password)
                mail.select("INBOX")
                status, _ = mail.uid("STORE", message_id, "+FLAGS", "\\Deleted")
                if status != "OK":
                    status, _ = mail.store(message_id, "+FLAGS", "\\Deleted")
                if status != "OK":
                    raise EmailError(f"IMAP DELETE failed for {message_id}: {status}")
                mail.expunge()
                _log.info(f"Deleted {message_id}")
        except Exception as e:
            _log.error(f"Failed to delete via IMAP: {e}")
            raise EmailError(f"IMAP delete failed: {e}") from e
