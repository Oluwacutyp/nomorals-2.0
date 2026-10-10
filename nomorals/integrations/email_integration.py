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

__all__ = ["EmailIntegration", "EmailError", "EmailMessage",
           "format_digest"]

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
        thread_id: str = "",
        snippet: str = "",
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
        self.thread_id = thread_id
        self.snippet = snippet

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
            "thread_id": self.thread_id,
            "snippet": self.snippet,
            "attachments": len(self.attachments),
        }

    def short_from(self) -> str:
        """'Alice <alice@x.com>' → 'Alice'."""
        name = self.from_addr.split("<")[0].strip().strip('"')
        return name or self.from_addr


def _html_to_text(html: str) -> str:
    """Readable text from HTML: strip tags, keep line breaks."""
    import re as _re
    import html as _html
    text = _re.sub(r"(?i)<(br|p|div|li|tr)[^>]*>", "\n", html)
    text = _re.sub(r"<[^>]+>", "", text)
    text = _html.unescape(text)
    text = _re.sub(r"\n{3,}", "\n\n", text)
    text = _re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _extract_body(payload: dict[str, Any]) -> tuple[str, list[dict]]:
    """Recursive MIME walk → (best text body, attachments).

    Prefers text/plain; falls back to text/html → _html_to_text.
    Attachment parts (with filename) are collected, not decoded.
    """
    best_plain, best_html = "", ""
    attachments: list[dict] = []

    def _walk(part: dict[str, Any]) -> None:
        nonlocal best_plain, best_html
        mime = part.get("mimeType", "")
        filename = part.get("filename", "")
        body = part.get("body", {}) or {}
        if filename and body.get("attachmentId"):
            attachments.append({
                "filename": filename, "mimeType": mime,
                "size": body.get("size", 0),
                "attachmentId": body.get("attachmentId"),
            })
            return
        data = body.get("data")
        if mime == "text/plain" and data and not best_plain:
            best_plain = base64.urlsafe_b64decode(data).decode(
                "utf-8", "replace")
        elif mime == "text/html" and data and not best_html:
            best_html = base64.urlsafe_b64decode(data).decode(
                "utf-8", "replace")
        for sub in part.get("parts", []) or []:
            _walk(sub)

    _walk(payload)
    text = best_plain or (_html_to_text(best_html) if best_html else "")
    return text, attachments


def format_digest(messages: list[EmailMessage], *,
                  title: str = "📬 inbox") -> str:
    """God-tier inbox rendering: unread-first, sender/subject/snippet,
    one-line action hints."""
    if not messages:
        return f"{title}\n_all clear — nothing here._"
    unread = [m for m in messages if not m.is_read]
    lines = [f"{title} — {len(unread)} unread / {len(messages)} shown"]
    ordered = sorted(messages,
                     key=lambda m: (m.is_read, -(m.date or 0)))
    for m in ordered:
        dot = "🔵" if not m.is_read else "⚪"
        when = time.strftime("%H:%M",
                            time.localtime(m.date)) if m.date else ""
        snippet = (m.snippet or m.body or "").strip().replace("\n", " ")
        if len(snippet) > 90:
            snippet = snippet[:87] + "…"
        lines.append(f"{dot} **{m.subject or '(no subject)'}**")
        lines.append(f"   {m.short_from()} · {when}")
        if snippet:
            lines.append(f"   _{snippet}_")
    return "\n".join(lines)


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

    # ── Gmail plumbing: token refresh, authed requests, cursors ──────

    def _gmail_token(self, account: str) -> str:
        """Access token, refreshing transparently via refresh_token."""
        import os
        import urllib.parse
        import urllib.request
        cred = self.account_manager.get_credential("gmail_oauth", account)
        try:
            token_data = json.loads(cred.password)
        except (TypeError, ValueError):
            return cred.password
        access = token_data.get("access_token", "")
        if access and self._gmail_token_ok(access):
            return access
        refresh = token_data.get("refresh_token", "")
        client_id = token_data.get("client_id", "") or os.environ.get(
            "GOOGLE_CLIENT_ID", "")
        client_secret = token_data.get("client_secret", "") or os.environ.get(
            "GOOGLE_CLIENT_SECRET", "")
        if not (refresh and client_id and client_secret):
            if access:
                return access  # no refresh material — use as-is
            raise EmailError(
                f"cannot refresh Gmail token for {account}: no "
                "refresh_token/client credentials stored")
        data = urllib.parse.urlencode({
            "grant_type": "refresh_token", "refresh_token": refresh,
            "client_id": client_id, "client_secret": client_secret,
        }).encode()
        req = urllib.request.Request(
            "https://oauth2.googleapis.com/token", data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            fresh = json.loads(resp.read().decode())
        token_data["access_token"] = fresh["access_token"]
        token_data["expires_at"] = time.time() + int(
            fresh.get("expires_in", 3600))
        try:
            self.account_manager.store_credential(
                "gmail_oauth", account, json.dumps(token_data),
                credential_type="oauth_token")
        except Exception:  # noqa: BLE001 - best effort
            _log.debug("could not persist refreshed gmail token",
                       exc_info=True)
        return token_data["access_token"]

    @staticmethod
    def _gmail_token_ok(access_token: str) -> bool:
        import urllib.request
        try:
            req = urllib.request.Request(
                "https://www.googleapis.com/oauth2/v3/tokeninfo",
                headers={"Authorization": f"Bearer {access_token}"})
            with urllib.request.urlopen(req, timeout=8):
                return True
        except Exception:  # noqa: BLE001
            return False

    def _gmail_request(self, account: str, method: str, url: str,
                       payload: dict | None = None,
                       params: dict | None = None) -> Any:
        """Authenticated Gmail API call with transparent token refresh."""
        import urllib.parse
        import urllib.request
        token = self._gmail_token(account)
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Authorization": f"Bearer {token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                # one retry after forced refresh
                cred = self.account_manager.get_credential(
                    "gmail_oauth", account)
                td = json.loads(cred.password)
                td.pop("access_token", None)
                token = self._gmail_token(account)
                headers["Authorization"] = f"Bearer {token}"
                req = urllib.request.Request(url, data=data,
                                             headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read().decode()
                    return json.loads(raw) if raw else None
            raise

    def _cursor_db(self):
        import os
        import sqlite3
        path = os.path.expanduser("~/.nomorals/email/cursors.db")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        db = sqlite3.connect(path, check_same_thread=False)
        db.execute("CREATE TABLE IF NOT EXISTS cursors ("
                   " account TEXT PRIMARY KEY, history_id TEXT NOT NULL,"
                   " updated REAL NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS watches ("
                   " account TEXT PRIMARY KEY, topic TEXT NOT NULL,"
                   " history_id TEXT NOT NULL, expires_at REAL NOT NULL,"
                   " updated REAL NOT NULL)")
        db.commit()
        return db

    # ── incremental sync (History API, not search) ────────────────────

    async def sync_new(self, account: str, *,
                       label_ids: list[str] | None = None) -> dict[str, Any]:
        """Fetch ONLY what's new since the last sync via the History API.

        Persists the historyId cursor in sqlite. First call seeds the
        cursor and returns []. This is the quota-friendly path — prefer
        it over polling read_inbox().
        """
        import urllib.error
        if self._detect_backend(account) != "gmail_api":
            raise EmailError("sync_new needs the Gmail API backend")
        db = self._cursor_db()
        row = db.execute("SELECT history_id FROM cursors WHERE account = ?",
                         (account,)).fetchone()
        profile = self._gmail_request(
            account, "GET",
            "https://gmail.googleapis.com/gmail/v1/users/me/profile")
        current_hid = str(profile.get("historyId", ""))
        if not row:
            db.execute("INSERT OR REPLACE INTO cursors VALUES (?, ?, ?)",
                       (account, current_hid, time.time()))
            db.commit(); db.close()
            return {"new": [], "cursor_seeded": True,
                    "history_id": current_hid}
        start_hid = row[0]
        params: dict[str, Any] = {"startHistoryId": start_hid,
                                  "historyTypes": ["messageAdded"]}
        if label_ids:
            params["labelId"] = label_ids[0]
        messages: list[EmailMessage] = []
        page_token = ""
        try:
            while True:
                if page_token:
                    params["pageToken"] = page_token
                result = self._gmail_request(
                    account, "GET",
                    "https://gmail.googleapis.com/gmail/v1/users/me/history",
                    params=params) or {}
                for h in result.get("history", []):
                    for added in h.get("messagesAdded", []):
                        mid = added["message"]["id"]
                        msg = await self._get_gmail_message(
                            mid, account, self._gmail_token(account))
                        if msg:
                            messages.append(msg)
                page_token = result.get("nextPageToken", "")
                if not page_token:
                    break
                new_hid = result.get("historyId", current_hid)
            db.execute("INSERT OR REPLACE INTO cursors VALUES (?, ?, ?)",
                       (account, current_hid, time.time()))
            db.commit()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # historyId too old — reseed
                db.execute(
                    "INSERT OR REPLACE INTO cursors VALUES (?, ?, ?)",
                    (account, current_hid, time.time()))
                db.commit()
                messages = []
            else:
                raise
        finally:
            db.close()
        return {"new": messages, "cursor_seeded": False,
                "history_id": current_hid}

    # ── push notifications (watch) ───────────────────────────────────

    async def start_watch(self, topic_name: str, account: str, *,
                          label_ids: list[str] | None = None) -> dict:
        """Register a Gmail push watch (Pub/Sub topic). Watch expires
        after ~7 days — :meth:`renew_watches` re-arms it."""
        if self._detect_backend(account) != "gmail_api":
            raise EmailError("watch needs the Gmail API backend")
        payload: dict[str, Any] = {"topicName": topic_name}
        if label_ids:
            payload["labelIds"] = label_ids
        result = self._gmail_request(
            account, "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/watch",
            payload) or {}
        db = self._cursor_db()
        db.execute("INSERT OR REPLACE INTO watches VALUES (?, ?, ?, ?, ?)",
                   (account, topic_name, str(result.get("historyId", "")),
                    int(result.get("expiration", 0)) / 1000, time.time()))
        db.commit(); db.close()
        _log.info("gmail watch started for %s (expires %s)", account,
                  time.strftime("%Y-%m-%d",
                                time.localtime(int(result.get("expiration", 0))
                                               / 1000)))
        return {"history_id": result.get("historyId"),
                "expiration": result.get("expiration")}

    async def stop_watch(self, account: str) -> None:
        if self._detect_backend(account) != "gmail_api":
            return
        try:
            self._gmail_request(
                account, "POST",
                "https://gmail.googleapis.com/gmail/v1/users/me/stop")
        finally:
            db = self._cursor_db()
            db.execute("DELETE FROM watches WHERE account = ?", (account,))
            db.commit(); db.close()

    def watch_status(self, account: str) -> dict | None:
        """Watch metadata: topic, expiry. None when no watch."""
        db = self._cursor_db()
        row = db.execute("SELECT topic, history_id, expires_at FROM watches"
                         " WHERE account = ?", (account,)).fetchone()
        db.close()
        if not row:
            return None
        return {"topic": row[0], "history_id": row[1],
                "expires_at": row[2],
                "expires_in_s": max(0.0, row[2] - time.time())}

    async def renew_watches(self, topic_name: str) -> list[str]:
        """Re-arm every watch expiring within 24h. Returns renewed accts."""
        db = self._cursor_db()
        rows = db.execute("SELECT account, expires_at FROM watches").fetchall()
        db.close()
        renewed = []
        for account, expires_at in rows:
            if expires_at - time.time() < 86400:
                try:
                    await self.start_watch(topic_name, account)
                    renewed.append(account)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("watch renew failed for %s: %s",
                                 account, exc)
        return renewed

    # ── threads / drafts / batch / attachments ───────────────────────

    async def get_thread(self, thread_id: str,
                         account: str) -> list[EmailMessage]:
        """All messages in a Gmail thread, oldest first."""
        if self._detect_backend(account) != "gmail_api":
            raise EmailError("threads need the Gmail API backend")
        token = self._gmail_token(account)
        result = self._gmail_request(
            account, "GET",
            f"https://gmail.googleapis.com/gmail/v1/users/me/threads/"
            f"{thread_id}",
            params={"format": "full"}) or {}
        out = []
        for m in result.get("messages", []):
            msg = self._message_from_full(m, token)
            if msg:
                out.append(msg)
        return sorted(out, key=lambda m: m.date or 0)

    def _message_from_full(self, data: dict, token: str) -> Optional[
            "EmailMessage"]:
        """Parse a full-format Gmail message dict (shared by get/thread)."""
        try:
            headers = {h["name"].lower(): h["value"]
                       for h in data["payload"]["headers"]}
            body, attachments = _extract_body(data["payload"])
            labels = data.get("labelIds", [])
            return EmailMessage(
                message_id=data["id"],
                from_addr=headers.get("from", ""),
                to_addrs=[headers.get("to", "")],
                subject=headers.get("subject", ""),
                body=body,
                date=int(data.get("internalDate", "0")) / 1000,
                labels=labels,
                is_read="UNREAD" not in labels,
                thread_id=data.get("threadId", ""),
                snippet=data.get("snippet", ""),
                attachments=attachments,
            )
        except Exception as e:
            _log.error(f"Failed to parse message {data.get('id')}: {e}")
            return None

    async def create_draft(self, to: str | list[str], subject: str,
                           body: str, *, account: str,
                           html: bool = False) -> str:
        """Create a Gmail draft (does NOT send). Returns draft id."""
        if self._detect_backend(account) != "gmail_api":
            raise EmailError("drafts need the Gmail API backend")
        to_addrs = [to] if isinstance(to, str) else to
        message = MIMEText(body, "html" if html else "plain")
        message["to"] = ", ".join(to_addrs)
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        result = self._gmail_request(
            account, "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/drafts",
            {"message": {"raw": raw}}) or {}
        return result.get("id", "")

    async def send_draft(self, draft_id: str, account: str) -> str:
        """Send a previously created draft. Returns message id."""
        if self._detect_backend(account) != "gmail_api":
            raise EmailError("drafts need the Gmail API backend")
        result = self._gmail_request(
            account, "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/drafts/send",
            {"id": draft_id}) or {}
        return result.get("id", "")

    async def batch_modify(self, message_ids: list[str], account: str, *,
                           add_labels: list[str] | None = None,
                           remove_labels: list[str] | None = None) -> int:
        """Add/remove labels on up to 1000 messages in ONE call."""
        if self._detect_backend(account) != "gmail_api":
            raise EmailError("batch_modify needs the Gmail API backend")
        ids = list(message_ids or [])[:1000]
        if not ids:
            return 0
        payload: dict[str, Any] = {"ids": ids}
        if add_labels:
            payload["addLabelIds"] = add_labels
        if remove_labels:
            payload["removeLabelIds"] = remove_labels
        self._gmail_request(
            account, "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/"
            "batchModify", payload)
        return len(ids)

    async def download_attachment(self, message_id: str, attachment_id: str,
                                  account: str, dest_path: str) -> str:
        """Download a Gmail attachment to ``dest_path``."""
        if self._detect_backend(account) != "gmail_api":
            raise EmailError("attachments need the Gmail API backend")
        result = self._gmail_request(
            account, "GET",
            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/"
            f"{message_id}/attachments/{attachment_id}") or {}
        data = result.get("data", "")
        raw = base64.urlsafe_b64decode(data)
        with open(dest_path, "wb") as f:
            f.write(raw)
        return dest_path
    
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
        """Fetch a single Gmail message (recursive MIME body extraction)."""
        import urllib.request

        url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}?format=full"

        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return self._message_from_full(data, access_token)
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
