"""Temporary email addresses for receiving verification emails.

Keyless temp-mail providers, no API keys needed. Used by the account
creator when a signup flow demands email verification: grab an address,
hand it to the site, then poll the inbox for the code/link.

Providers (all keyless):
- mail.tm — modern REST API with JWT accounts; the cascade primary
  (cleanest keyless REST API).
- 1secmail — real REST API (getMessages/readMessage/genRandomMailbox),
  pure GET+JSON, no auth.
- GuerrillaMail — AJAX API with session token, no account needed.
- owndomain — Devon-hosted disposable mailbox (10MinuteMail #50):
  an address on a domain the owner owns, received by a mail stack
  Devon runs (stdlib SMTP catch-all listener + SQLite store), same
  10-minute TTL semantics. First in the cascade when configured
  (NM_TEMPMAIL_DOMAIN); unconfigured it reports unavailable and the
  cascade falls through to the public providers.

The interface mirrors ``temp_sms`` so callers get the same shape:
grab an address, wait for a message, extract the verification code.
Use :func:`grab_address_cascade` to try providers in order.
"""

from __future__ import annotations

import json
import os
import random
import re
import secrets
import socketserver
import sqlite3
import string
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "MailMessage",
    "TempAddress",
    "TempMailProvider",
    "OneSecMailProvider",
    "GuerrillaMailProvider",
    "MailTmProvider",
    "OwnDomainProvider",
    "OwnDomainStore",
    "OwnDomainListener",
    "start_own_domain_listener",
    "stop_own_domain_listener",
    "own_domain_listener_alive",
    "own_domain_status",
    "own_domain_name",
    "default_store_path",
    "run_own_domain_mail_tick_once",
    "ensure_own_domain_mail_job",
    "OWN_DOMAIN_MAIL_JOB",
    "OWN_DOMAIN_MAIL_TOOL",
    "OWN_DOMAIN_MAIL_SPEC",
    "OWN_DOMAIN_TTL_S",
    "OWN_DOMAIN_EXTEND_S",
    "get_provider",
    "grab_address",
    "grab_address_cascade",
    "wait_message",
    "wait_code",
    "delete_message",
    "PROVIDERS",
    "CASCADE_PROVIDERS",
]


def _fetch_json(url: str, timeout: float = 20,
                headers: dict[str, str] | None = None) -> Any:
    req = urllib.request.Request(url, headers=headers or {
        "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _fetch_text(url: str, timeout: float = 20,
                data: bytes | None = None) -> str:
    req = urllib.request.Request(
        url, data=data,
        headers={"User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36",
                 "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


@dataclass
class MailMessage:
    """One received email."""
    id: str = ""
    sender: str = ""
    subject: str = ""
    date: str = ""
    body_text: str = ""
    body_html: str = ""
    code: str = ""  # extracted verification code, if any

    #: 4-8 digit runs are the usual OTP shape
    _CODE_RE = re.compile(r"(?<!\d)(\d{4,8})(?!\d)")

    def __post_init__(self) -> None:
        if not self.code:
            hay = f"{self.subject}\n{self.body_text}"
            m = self._CODE_RE.search(hay.replace(" ", "").replace("-", ""))
            if not m:
                m = self._CODE_RE.search(hay)
            if m:
                self.code = m.group(1)


@dataclass
class TempAddress:
    """A temporary receivable email address."""
    address: str = ""       # full address, e.g. user@1secmail.com
    login: str = ""         # local part
    domain: str = ""        # domain part
    provider: str = ""      # provider name
    token: str = ""         # provider session token (guerrillamail)
    expires_at: float = 0.0  # epoch seconds the address dies; 0 = unknown
    # (public providers don't publish a TTL — only the own-domain
    # rung sets this, from its 10-minute mint).


class TempMailProvider:
    """Interface every temp-mail provider implements."""

    name: str = ""

    def grab_address(self) -> TempAddress:
        raise NotImplementedError

    def get_messages(self, addr: TempAddress) -> list[MailMessage]:
        raise NotImplementedError

    def read_message(self, addr: TempAddress,
                     msg: MailMessage) -> MailMessage:
        """Fetch the full body for a message listing entry."""
        return msg

    def wait_for_message(self, addr: TempAddress, *,
                         timeout_s: float = 120,
                         poll_s: float = 8,
                         sender_hint: str = "",
                         subject_hint: str = "") -> MailMessage | None:
        deadline = time.time() + timeout_s
        seen: set[str] = set()
        while time.time() < deadline:
            try:
                msgs = self.get_messages(addr)
            except Exception as exc:  # noqa: BLE001
                _log.debug("%s poll failed: %s", self.name, exc)
                msgs = []
            for m in msgs:
                if m.id in seen:
                    continue
                seen.add(m.id)
                if sender_hint and sender_hint.lower() not in m.sender.lower():
                    continue
                if subject_hint and subject_hint.lower() not in m.subject.lower():
                    continue
                try:
                    m = self.read_message(addr, m)
                except Exception as exc:  # noqa: BLE001
                    _log.debug("%s read failed: %s", self.name, exc)
                return m
            time.sleep(poll_s)
        return None

    def delete_message(self, addr: TempAddress,
                       msg: MailMessage) -> bool:
        """Delete one message. Default: unsupported (returns False)."""
        return False

    def probe(self) -> bool:
        """True when the provider looks reachable. Default: try a cheap
        call. Never raises — a failed probe means "skip me"."""
        return True


class OneSecMailProvider(TempMailProvider):
    """1secmail.com — real REST API, keyless, pure GET+JSON."""

    name = "1secmail"
    BASE = "https://www.1secmail.com/api/v1/"

    def probe(self) -> bool:
        """True when the domain listing parses. Never raises — the
        cascade skips a down provider instead of burning a timeout in
        ``grab_address`` (1secmail has been observed to hang/drop
        connections for extended periods)."""
        try:
            domains = self._get(action="getDomainList")
            return bool(domains)
        except Exception:  # noqa: BLE001 - probe failure = unavailable
            _log.debug("1secmail probe failed")
            return False

    def _get(self, **params: str) -> Any:
        qs = urllib.parse.urlencode(params)
        return _fetch_json(self.BASE + "?" + qs)

    def grab_address(self) -> TempAddress:
        # try random mailbox first (fastest), fall back to manual
        try:
            addrs = self._get(action="genRandomMailbox", count="1")
            if addrs:
                address = addrs[0]
                login, domain = address.split("@", 1)
                return TempAddress(address=address, login=login,
                                   domain=domain, provider=self.name)
        except Exception as exc:  # noqa: BLE001
            _log.debug("1secmail genRandomMailbox failed: %s", exc)
        domains = self._get(action="getDomainList")
        login = "".join(random.choices(string.ascii_lowercase + string.digits,
                                       k=12))
        domain = domains[0] if domains else "1secmail.com"
        return TempAddress(address=f"{login}@{domain}", login=login,
                           domain=domain, provider=self.name)

    def get_messages(self, addr: TempAddress) -> list[MailMessage]:
        raw = self._get(action="getMessages", login=addr.login,
                        domain=addr.domain)
        out = []
        for m in raw or []:
            out.append(MailMessage(
                id=str(m.get("id", "")),
                sender=str(m.get("from", "")),
                subject=str(m.get("subject", "")),
                date=str(m.get("date", "")),
            ))
        return out

    def read_message(self, addr: TempAddress,
                     msg: MailMessage) -> MailMessage:
        raw = self._get(action="readMessage", login=addr.login,
                        domain=addr.domain, id=msg.id)
        msg.body_text = str(raw.get("textBody", "") or "")
        msg.body_html = str(raw.get("htmlBody", "") or "")
        msg.sender = str(raw.get("from", "") or msg.sender)
        msg.subject = str(raw.get("subject", "") or msg.subject)
        # re-run code extraction now that we have the body
        msg.code = ""
        msg.__post_init__()
        return msg


class GuerrillaMailProvider(TempMailProvider):
    """GuerrillaMail — AJAX API, session token, no account."""

    name = "guerrillamail"
    BASE = "https://api.guerrillamail.com/ajax.php"

    def _call(self, **params: str) -> Any:
        qs = urllib.parse.urlencode(params)
        return _fetch_json(self.BASE + "?" + qs)

    def probe(self) -> bool:
        """True when a session address comes back. Never raises."""
        try:
            raw = self._call(f="get_email_address", ip="127.0.0.1",
                             agent="Mozilla/5.0")
            return bool(raw.get("email_addr"))
        except Exception:  # noqa: BLE001 - probe failure = unavailable
            _log.debug("guerrillamail probe failed")
            return False

    def grab_address(self) -> TempAddress:
        raw = self._call(f="get_email_address", ip="127.0.0.1",
                         agent="Mozilla/5.0")
        address = str(raw.get("email_addr", ""))
        token = str(raw.get("sid_token", ""))
        login = address.split("@")[0] if "@" in address else address
        domain = address.split("@")[1] if "@" in address else ""
        return TempAddress(address=address, login=login, domain=domain,
                           provider=self.name, token=token)

    def get_messages(self, addr: TempAddress) -> list[MailMessage]:
        raw = self._call(f="check_email", sid_token=addr.token, seq="0")
        out = []
        for m in (raw.get("list") or []):
            out.append(MailMessage(
                id=str(m.get("mail_id", "")),
                sender=str(m.get("mail_from", "")),
                subject=str(m.get("mail_subject", "")),
                date=str(m.get("mail_date", "")),
                body_text=str(m.get("mail_excerpt", "")),
            ))
        return out

    def read_message(self, addr: TempAddress,
                     msg: MailMessage) -> MailMessage:
        raw = self._call(f="fetch_email", sid_token=addr.token,
                         email_id=msg.id)
        msg.body_text = str(raw.get("mail_body", "") or msg.body_text)
        # strip HTML tags for code extraction
        text = re.sub(r"<[^>]+>", " ", msg.body_text)
        msg.body_text = text
        msg.code = ""
        msg.__post_init__()
        return msg


class MailTmProvider(TempMailProvider):
    """mail.tm (and mail.gw — identical API) — the modern keyless temp-mail
    REST API and the primary of the cascade.

    Flow: GET /domains → POST /accounts {address, password} (201) →
    POST /token {address, password} → JWT → GET /messages (Bearer).
    Accounts are random-password JWT accounts; the JWT rides on
    ``TempAddress.token`` so ``wait_message`` works unchanged.
    """

    name = "mailtm"
    BASE = "https://api.mail.tm"
    _GW_BASE = "https://api.mail.gw"

    def __init__(self, base: str | None = None) -> None:
        self._base = base or self.BASE

    def _post_json(self, path: str, payload: dict[str, Any],
                   token: str = "") -> tuple[int, Any]:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "User-Agent": "Mozilla/5.0 (Linux; Android 10)"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(self._base + path, data=data,
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read().decode("utf-8", "replace")
                return resp.status, (json.loads(body) if body else {})
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                return exc.code, (json.loads(body) if body else {})
            except ValueError:
                return exc.code, {}

    def _get_auth(self, path: str, token: str) -> Any:
        return _fetch_json(
            self._base + path, timeout=20,
            headers={"Authorization": f"Bearer {token}",
                     "User-Agent": "Mozilla/5.0 (Linux; Android 10)"},
        )

    def probe(self) -> bool:
        """True when the domain listing parses. Never raises."""
        try:
            domains = self.list_domains()
            return bool(domains)
        except Exception:  # noqa: BLE001
            _log.debug("mail.tm probe failed")
            return False

    def list_domains(self) -> list[str]:
        raw = _fetch_json(self._base + "/domains", timeout=20)
        members = raw.get("hydra:member") or []
        return [str(m.get("domain", "")) for m in members
                if m.get("domain")]

    def grab_address(self) -> TempAddress:
        domains = self.list_domains()
        domain = domains[0] if domains else "mail.tm"
        login = "".join(random.choices(string.ascii_lowercase +
                                       string.digits, k=12))
        address = f"{login}@{domain}"
        password = "".join(secrets.choice(
            string.ascii_letters + string.digits) for _ in range(20))
        status, _ = self._post_json("/accounts",
                                    {"address": address, "password": password})
        if status not in (200, 201):
            raise RuntimeError(
                f"mail.tm account creation failed (HTTP {status})")
        status, tok = self._post_json(
            "/token", {"address": address, "password": password})
        jwt = str(tok.get("token", "")) if status == 200 else ""
        if not jwt:
            raise RuntimeError("mail.tm token fetch failed")
        return TempAddress(address=address, login=login, domain=domain,
                           provider=self.name, token=jwt)

    def _sender(self, m: dict[str, Any]) -> str:
        frm = m.get("from") or {}
        addr = frm.get("address", "") if isinstance(frm, dict) else str(frm)
        name = frm.get("name", "") if isinstance(frm, dict) else ""
        return f"{name} <{addr}>".strip() if name else str(addr)

    def get_messages(self, addr: TempAddress) -> list[MailMessage]:
        if not addr.token:
            return []
        raw = self._get_auth("/messages", addr.token)
        out = []
        for m in raw.get("hydra:member") or []:
            out.append(MailMessage(
                id=str(m.get("id", "")),
                sender=self._sender(m),
                subject=str(m.get("subject", "")),
                date=str(m.get("createdAt", "")),
                body_text=str(m.get("intro", "")),
            ))
        return out

    def read_message(self, addr: TempAddress,
                     msg: MailMessage) -> MailMessage:
        if not addr.token:
            return msg
        raw = self._get_auth(f"/messages/{msg.id}", addr.token)
        msg.body_text = str(raw.get("text", "") or msg.body_text)
        msg.body_html = str(raw.get("html", "") or "")
        msg.sender = self._sender(raw) or msg.sender
        msg.subject = str(raw.get("subject", "") or msg.subject)
        msg.code = ""
        msg.__post_init__()
        return msg

    def delete_message(self, addr: TempAddress,
                       msg: MailMessage) -> bool:
        """DELETE /messages/{id} — 204 on success."""
        if not addr.token:
            return False
        import urllib.request
        req = urllib.request.Request(
            f"{self._base}/messages/{msg.id}",
            headers={"Authorization": f"Bearer {addr.token}",
                     "User-Agent": "Mozilla/5.0 (Linux; Android 10)"},
            method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status in (200, 204)
        except Exception as exc:  # noqa: BLE001
            _log.debug("mail.tm delete failed: %s", exc)
            return False


# ── Own-domain rung (10MinuteMail #50) ─────────────────────────────────
# A disposable mailbox on infrastructure Devon controls: an address on
# a domain the owner owns, received by a mail stack Devon runs. Same
# TempMailProvider interface as the public providers, so the cascade,
# the account creator and trial flows gain it without changes.
# Receive-only — it never sends, never relays, never forwards.
# Storage is the house SQLite idiom (short-lived connections, one
# lock); expiry DELETES rows, never archives — the site's semantics.

#: the site's TTL: 10 minutes from mint (or from first receipt for a
#: catch-all arrival to an unminted local part)
OWN_DOMAIN_TTL_S = 600.0
#: "+10 minutes" — one extend moves expires_at by this much
OWN_DOMAIN_EXTEND_S = 600.0
#: largest single message the listener accepts (raw bytes, headers
#: included); larger DATA is refused with 552 and never stored
OWN_DOMAIN_MAX_MESSAGE_BYTES = 1_000_000
#: longest body text kept per message (the raw cap above still bounds
#: the whole message; this only trims pathological single parts)
OWN_DOMAIN_MAX_BODY_CHARS = 200_000

OWN_DOMAIN_MAIL_JOB = "own-domain mail sweep"
OWN_DOMAIN_MAIL_TOOL = "__own_domain_mail_tick__"
#: every 5 minutes — shorter than the 10-minute TTL, so an expired
#: mailbox is gone within one TTL of its death at most
OWN_DOMAIN_MAIL_SPEC = "*/5 * * * *"


def _default_home() -> str:
    return os.environ.get("NM_HOME", "") or str(Path.home() / ".nomorals")


def default_store_path(home: str | None = None) -> str:
    """SQLite path for the own-domain mailbox store.

    ``NM_TEMPMAIL_DB`` overrides (tests, operator moves); otherwise
    ``<home>/data/own_domain_mail.db`` under NM_HOME or ~/.nomorals.
    """
    override = os.environ.get("NM_TEMPMAIL_DB", "").strip()
    if override:
        return override
    base = Path(home) if home else Path(_default_home())
    return str(base / "data" / "own_domain_mail.db")


def own_domain_name() -> str:
    """The owner's temp-mail domain (NM_TEMPMAIL_DOMAIN), or ""."""
    return os.environ.get("NM_TEMPMAIL_DOMAIN", "").strip().lower()


def _own_domain_valid(domain: str) -> bool:
    return bool(domain) and "." in domain and "@" not in domain \
        and " " not in domain


def _own_ttl_s() -> float:
    try:
        return max(60.0, float(os.environ.get("NM_TEMPMAIL_TTL_S", "")
                               or OWN_DOMAIN_TTL_S))
    except ValueError:
        return OWN_DOMAIN_TTL_S


def _own_smtp_port() -> int:
    try:
        return int(os.environ.get("NM_TEMPMAIL_SMTP_PORT", "") or 25)
    except ValueError:
        return 25


def _own_bind_host() -> str:
    return os.environ.get("NM_TEMPMAIL_BIND", "").strip() or "0.0.0.0"


def _random_login(n: int = 12) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits,
                                  k=n))


class OwnDomainStore:
    """SQLite mailbox store for the own-domain rung.

    Two tables, both keyed by address: ``mailboxes`` (one row per
    minted or first-seen address, with ``expires_at``) and
    ``messages`` (one row per received mail, deleted with its
    mailbox). Short-lived connections per operation plus one lock,
    so the SMTP listener thread and the provider never share a
    connection across threads.
    """

    def __init__(self, path: str | None = None, *,
                 home: str | None = None) -> None:
        self.path = path or default_store_path(home)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS mailboxes ("
                " address TEXT PRIMARY KEY,"
                " created_at REAL NOT NULL,"
                " expires_at REAL NOT NULL)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                " id TEXT PRIMARY KEY,"
                " address TEXT NOT NULL"
                " REFERENCES mailboxes(address) ON DELETE CASCADE,"
                " sender TEXT NOT NULL DEFAULT '',"
                " subject TEXT NOT NULL DEFAULT '',"
                " date TEXT NOT NULL DEFAULT '',"
                " body_text TEXT NOT NULL DEFAULT '',"
                " body_html TEXT NOT NULL DEFAULT '',"
                " received_at REAL NOT NULL)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_address"
                " ON messages(address)")

    def mint(self, address: str, *, ttl_s: float,
             now: float | None = None) -> float:
        """Record a minted address; returns its expires_at."""
        now = time.time() if now is None else now
        expires = now + ttl_s
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO mailboxes"
                " (address, created_at, expires_at) VALUES (?, ?, ?)",
                (address, now, expires))
        return expires

    def ensure_mailbox(self, address: str, *, ttl_s: float,
                       now: float | None = None) -> float:
        """Catch-all receipt: a first-seen local part gets a mailbox
        with a fresh TTL; a live one keeps its expiry; an expired one
        is revived with a fresh TTL from receipt."""
        now = time.time() if now is None else now
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT expires_at FROM mailboxes WHERE address = ?",
                (address,)).fetchone()
            if row is not None and float(row[0]) > now:
                return float(row[0])
            expires = now + ttl_s
            conn.execute(
                "INSERT OR REPLACE INTO mailboxes"
                " (address, created_at, expires_at) VALUES (?, ?, ?)",
                (address, now, expires))
            return expires

    def mailbox(self, address: str, *,
                now: float | None = None) -> dict[str, Any] | None:
        """The mailbox row, or None when never seen or expired."""
        now = time.time() if now is None else now
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT address, created_at, expires_at FROM mailboxes"
                " WHERE address = ?", (address,)).fetchone()
        if row is None or float(row[2]) <= now:
            return None
        return {"address": str(row[0]), "created_at": float(row[1]),
                "expires_at": float(row[2])}

    def store_message(self, address: str, *, sender: str = "",
                      subject: str = "", date: str = "",
                      body_text: str = "", body_html: str = "",
                      now: float | None = None) -> str:
        """Store one received message (creating the mailbox catch-all
        style when first seen). Returns the message id."""
        now = time.time() if now is None else now
        msg_id = secrets.token_hex(8)
        with self._lock:
            self.ensure_mailbox(address, ttl_s=_own_ttl_s(), now=now)
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO messages (id, address, sender, subject,"
                    " date, body_text, body_html, received_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (msg_id, address, sender, subject, date,
                     body_text[:OWN_DOMAIN_MAX_BODY_CHARS],
                     body_html[:OWN_DOMAIN_MAX_BODY_CHARS], now))
        return msg_id

    def list_messages(self, address: str, *,
                      now: float | None = None) -> list[dict[str, Any]]:
        """Messages for a live mailbox, oldest first. An expired or
        never-seen address reads as empty — never an error."""
        if self.mailbox(address, now=now) is None:
            return []
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, sender, subject, date, body_text, body_html"
                " FROM messages WHERE address = ?"
                " ORDER BY received_at, id",
                (address,)).fetchall()
        return [{"id": str(r[0]), "sender": str(r[1]),
                 "subject": str(r[2]), "date": str(r[3]),
                 "body_text": str(r[4]), "body_html": str(r[5])}
                for r in rows]

    def delete_message(self, address: str, msg_id: str) -> bool:
        """Delete one message. True when a row was removed."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM messages WHERE address = ? AND id = ?",
                (address, msg_id))
            return cur.rowcount > 0

    def extend(self, address: str, *, extra_s: float,
               now: float | None = None) -> float:
        """Move a live mailbox's expires_at forward by extra_s.
        Returns the new expires_at, or 0.0 when the mailbox is gone.
        """
        now = time.time() if now is None else now
        box = self.mailbox(address, now=now)
        if box is None:
            return 0.0
        new_expires = box["expires_at"] + extra_s
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE mailboxes SET expires_at = ? WHERE address = ?",
                (new_expires, address))
        return new_expires

    def next_expiry(self, *, now: float | None = None) -> float:
        """Soonest live expires_at, or 0.0 when nothing is live."""
        now = time.time() if now is None else now
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(expires_at) FROM mailboxes"
                " WHERE expires_at > ?", (now,)).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def counts(self, *, now: float | None = None) -> dict[str, int]:
        """Live mailboxes and the messages inside them."""
        now = time.time() if now is None else now
        with self._lock, self._connect() as conn:
            boxes = conn.execute(
                "SELECT COUNT(*) FROM mailboxes WHERE expires_at > ?",
                (now,)).fetchone()[0]
            msgs = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE address IN"
                " (SELECT address FROM mailboxes WHERE expires_at > ?)",
                (now,)).fetchone()[0]
        return {"mailboxes": int(boxes), "messages": int(msgs)}

    def sweep(self, *, now: float | None = None) -> dict[str, int]:
        """Delete expired mailboxes (their messages cascade) plus any
        orphan messages. Returns what was deleted — never archives."""
        now = time.time() if now is None else now
        with self._lock, self._connect() as conn:
            doomed = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE address IN"
                " (SELECT address FROM mailboxes WHERE expires_at <= ?)",
                (now,)).fetchone()[0]
            cur = conn.execute(
                "DELETE FROM mailboxes WHERE expires_at <= ?", (now,))
            mailboxes = cur.rowcount
            cur = conn.execute(
                "DELETE FROM messages WHERE address NOT IN"
                " (SELECT address FROM mailboxes)")
            messages = int(doomed) + cur.rowcount
        return {"mailboxes": int(mailboxes), "messages": int(messages)}


def _message_parts(raw: bytes) -> tuple[str, str, str, str]:
    """Parse one received message: (sender, subject, date, body_text,
    body_html). Never raises — a malformed message still yields its
    raw text as the body."""
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        sender = str(msg.get("From", "") or "")
        subject = str(msg.get("Subject", "") or "")
        date = str(msg.get("Date", "") or "")
        text_parts: list[str] = []
        html_parts: list[str] = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.is_multipart():
                    continue
                try:
                    payload = part.get_payload(decode=True) or b""
                    charset = part.get_content_charset() or "utf-8"
                    decoded = payload.decode(charset, "replace")
                except Exception:  # noqa: BLE001 - one bad part skips
                    continue
                if part.get_content_type() == "text/html":
                    html_parts.append(decoded)
                else:
                    text_parts.append(decoded)
        else:
            try:
                payload = msg.get_payload(decode=True)
                body = (payload.decode(
                    msg.get_content_charset() or "utf-8", "replace")
                    if payload is not None else str(msg.get_payload() or ""))
            except Exception:  # noqa: BLE001 - fall back to raw text
                body = raw.decode("utf-8", "replace")
            if msg.get_content_type() == "text/html":
                html_parts.append(body)
            else:
                text_parts.append(body)
        return sender, subject, date, "\n".join(text_parts), \
            "\n".join(html_parts)
    except Exception:  # noqa: BLE001 - malformed mail still lands
        return "", "", "", raw.decode("utf-8", "replace"), ""


class _OwnDomainSMTPHandler(socketserver.StreamRequestHandler):
    """One SMTP session for the own-domain catch-all listener.

    Speaks just enough SMTP to receive: HELO/EHLO, MAIL FROM, RCPT
    TO, DATA, RSET, NOOP, VRFY, QUIT. Receive-only — there is no send
    path anywhere in this class. RCPT is accepted only for the
    configured domain (anything else: 550); DATA over the size cap
    is refused with 552 and never stored.
    """

    def handle(self) -> None:  # noqa: C901 - one small state machine
        try:
            self._send(b"220 Devon own-domain temp mail ready\r\n")
            sender = ""
            recipients: list[str] = []
            while True:
                line = self._readline()
                if line is None:
                    return
                verb = line.split(b" ", 1)[0].upper()
                rest = line[len(verb):].strip() if len(line) > len(verb) \
                    else b""
                if verb in (b"HELO", b"EHLO"):
                    self._send(
                        b"250-devon-temp-mail\r\n"
                        b"250 SIZE %d\r\n" % getattr(
                            self.server, "max_bytes",
                            OWN_DOMAIN_MAX_MESSAGE_BYTES))
                elif verb == b"MAIL":
                    sender = self._addr_of(rest) or ""
                    recipients = []
                    self._send(b"250 sender ok\r\n")
                elif verb == b"RCPT":
                    rcpt = self._addr_of(rest) or ""
                    if not rcpt:
                        self._send(b"501 bad recipient\r\n")
                    elif not rcpt.lower().endswith(
                            "@" + self.server.domain):
                        self._send(b"550 relaying denied\r\n")
                    else:
                        recipients.append(rcpt.lower())
                        self._send(b"250 recipient ok\r\n")
                elif verb == b"DATA":
                    if not recipients:
                        self._send(b"503 need a recipient first\r\n")
                        continue
                    self._send(b"354 end with <CRLF>.<CRLF>\r\n")
                    raw = self._read_data()
                    if raw is None:
                        self._send(b"552 message too large\r\n")
                        sender, recipients = "", []
                        continue
                    frm, subj, date, text, html = _message_parts(raw)
                    for rcpt in recipients:
                        try:
                            self.server.store.store_message(
                                rcpt, sender=frm or sender, subject=subj,
                                date=date, body_text=text, body_html=html)
                        except Exception as exc:  # noqa: BLE001
                            _log.warning("own-domain store failed: %s", exc)
                    sender, recipients = "", []
                    self._send(b"250 message stored\r\n")
                elif verb == b"RSET":
                    sender, recipients = "", []
                    self._send(b"250 ok\r\n")
                elif verb == b"NOOP":
                    self._send(b"250 ok\r\n")
                elif verb == b"VRFY":
                    self._send(b"252 cannot verify, will accept\r\n")
                elif verb == b"QUIT":
                    self._send(b"221 bye\r\n")
                    return
                else:
                    self._send(b"502 command not implemented\r\n")
        except Exception as exc:  # noqa: BLE001 - one bad session dies
            _log.debug("own-domain smtp session ended: %s", exc)

    def _send(self, data: bytes) -> None:
        self.wfile.write(data)
        self.wfile.flush()

    def _readline(self) -> bytes | None:
        try:
            line = self.rfile.readline(8192)
        except OSError:
            return None
        if not line:
            return None
        return line.rstrip(b"\r\n")

    def _read_data(self) -> bytes | None:
        """Read DATA until <CRLF>.<CRLF>, undoing dot-stuffing.
        None when the message exceeds the size cap."""
        cap = getattr(self.server, "max_bytes",
                      OWN_DOMAIN_MAX_MESSAGE_BYTES)
        out = bytearray()
        while True:
            try:
                line = self.rfile.readline(8192 + 2)
            except OSError:
                return b"".join(out) if out else None
            if not line:
                return bytes(out)
            if line in (b".\r\n", b".\n", b"."):
                return bytes(out)
            if line.startswith(b".."):
                line = line[1:]
            out += line
            if len(out) > cap:
                # drain to the terminator so the client stays in sync,
                # then refuse the message
                while True:
                    try:
                        drain = self.rfile.readline(8192 + 2)
                    except OSError:
                        break
                    if not drain or drain in (b".\r\n", b".\n", b"."):
                        break
                return None

    @staticmethod
    def _addr_of(rest: bytes) -> str:
        """Pull the <addr> out of MAIL FROM:<addr> / RCPT TO:<addr>."""
        text = rest.decode("utf-8", "replace").strip()
        if text.startswith(("FROM:", "TO:")):
            text = text.split(":", 1)[1].strip()
        text = text.split(" ", 1)[0].strip()
        return text.strip("<>") if text.startswith("<") else text


class OwnDomainListener(socketserver.ThreadingTCPServer):
    """Catch-all SMTP listener for the owner's domain.

    stdlib only (``smtpd`` is gone on Python 3.12, so the receiver is
    a small socketserver state machine instead). Binds the configured
    host/port and stores every accepted message in the store.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *, domain: str, store: OwnDomainStore,
                 host: str, port: int,
                 max_bytes: int = OWN_DOMAIN_MAX_MESSAGE_BYTES) -> None:
        self.domain = domain.lower()
        self.store = store
        self.max_bytes = max_bytes
        super().__init__((host, port), _OwnDomainSMTPHandler)


_LISTENERS: dict[str, OwnDomainListener] = {}
_LISTENER_THREADS: dict[str, threading.Thread] = {}


def start_own_domain_listener(*, domain: str | None = None,
                              store: OwnDomainStore | None = None,
                              host: str | None = None,
                              port: int | None = None,
                              max_bytes: int | None = None,
                              home: str | None = None
                              ) -> OwnDomainListener | None:
    """Start the catch-all listener in a daemon thread.

    Returns the server, or None when no domain is configured (honest
    absence — the cascade stays on the public providers) or the bind
    fails. Idempotent per store path: a second start returns the
    running server.
    """
    domain = (domain or own_domain_name() or "").strip().lower()
    if not _own_domain_valid(domain):
        _log.info("own-domain listener not started: NM_TEMPMAIL_DOMAIN"
                  " is not set")
        return None
    store = store or OwnDomainStore(home=home)
    key = store.path
    existing = _LISTENERS.get(key)
    if existing is not None:
        return existing
    host = host or _own_bind_host()
    port = _own_smtp_port() if port is None else port
    try:
        server = OwnDomainListener(domain=domain, store=store,
                                   host=host, port=port,
                                   max_bytes=(max_bytes or
                                              OWN_DOMAIN_MAX_MESSAGE_BYTES))
    except Exception as exc:  # noqa: BLE001 - bind failure = unavailable
        _log.warning("own-domain listener failed to bind %s:%s: %s",
                     host, port, exc)
        return None
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.5},
                              daemon=True, name="own-domain-smtp")
    thread.start()
    _LISTENERS[key] = server
    _LISTENER_THREADS[key] = thread
    _log.info("own-domain listener up on %s:%s for @%s",
              host, server.server_address[1], domain)
    return server


def stop_own_domain_listener(store: OwnDomainStore | None = None, *,
                             home: str | None = None) -> bool:
    """Stop the listener for a store path. True when one was running."""
    path = (store.path if store is not None
            else default_store_path(home))
    server = _LISTENERS.pop(path, None)
    thread = _LISTENER_THREADS.pop(path, None)
    if server is None:
        return False
    try:
        server.shutdown()
        server.server_close()
    except Exception:  # noqa: BLE001 - best-effort shutdown
        pass
    if thread is not None:
        thread.join(timeout=5)
    return True


def own_domain_listener_alive(store: OwnDomainStore | None = None, *,
                              home: str | None = None) -> bool:
    """True when a listener thread is up for this store path."""
    path = (store.path if store is not None
            else default_store_path(home))
    thread = _LISTENER_THREADS.get(path)
    return thread is not None and thread.is_alive()


class OwnDomainProvider(TempMailProvider):
    """Devon-hosted disposable mailbox (10MinuteMail #50, own-domain
    rung).

    An address on a domain the owner owns, received by a mail stack
    Devon runs. Same TempMailProvider interface as the public
    providers, so the cascade, the account creator and trial flows
    gain it without changes. Receive-only — it never sends.

    Unconfigured (no NM_TEMPMAIL_DOMAIN) it reports unavailable:
    probe() is False, so the cascade skips it, and grab_address()
    fails in words naming the missing setting. Nothing is faked.
    """

    name = "owndomain"

    def __init__(self, domain: str | None = None,
                 store: OwnDomainStore | None = None, *,
                 ttl_s: float | None = None) -> None:
        self._domain = (domain or own_domain_name() or "").strip().lower()
        self._store = store
        self._ttl_s = ttl_s if ttl_s is not None else _own_ttl_s()

    @property
    def domain(self) -> str:
        return self._domain

    def configured(self) -> bool:
        return _own_domain_valid(self._domain)

    def _store_for(self) -> OwnDomainStore:
        if self._store is None:
            self._store = OwnDomainStore()
        return self._store

    def probe(self) -> bool:
        """True only when a domain is configured and the store opens.
        Never raises — a failed probe means "skip me"."""
        if not self.configured():
            return False
        try:
            self._store_for()
            return True
        except Exception:  # noqa: BLE001 - probe failure = unavailable
            _log.debug("owndomain probe failed")
            return False

    def grab_address(self) -> TempAddress:
        if not self.configured():
            raise RuntimeError(
                "own-domain temp mail is not configured — set "
                "NM_TEMPMAIL_DOMAIN to a domain you own (with its MX "
                "pointed at this host)")
        login = _random_login()
        address = f"{login}@{self._domain}"
        expires = self._store_for().mint(address, ttl_s=self._ttl_s)
        return TempAddress(address=address, login=login,
                           domain=self._domain, provider=self.name,
                           expires_at=expires)

    def get_messages(self, addr: TempAddress) -> list[MailMessage]:
        try:
            rows = self._store_for().list_messages(addr.address)
        except Exception:  # noqa: BLE001 - store failure reads as empty
            _log.debug("owndomain list failed")
            return []
        return [MailMessage(id=str(r["id"]), sender=str(r["sender"]),
                            subject=str(r["subject"]), date=str(r["date"]),
                            body_text=str(r["body_text"]),
                            body_html=str(r["body_html"]))
                for r in rows]

    def read_message(self, addr: TempAddress,
                     msg: MailMessage) -> MailMessage:
        # Bodies are stored whole; the listing already carries them.
        return msg

    def delete_message(self, addr: TempAddress,
                       msg: MailMessage) -> bool:
        try:
            return self._store_for().delete_message(addr.address, msg.id)
        except Exception:  # noqa: BLE001
            return False

    def extend_address(self, addr: TempAddress, *,
                       extra_s: float = OWN_DOMAIN_EXTEND_S) -> float:
        """+10 minutes: move expires_at forward. Returns the new
        expires_at, or 0.0 when the mailbox is gone."""
        if not self.configured():
            return 0.0
        try:
            return self._store_for().extend(addr.address, extra_s=extra_s)
        except Exception:  # noqa: BLE001
            return 0.0


def _tick_home(context: Any = None, home: str | None = None) -> str | None:
    if home:
        return home
    settings = getattr(context, "settings", None) if context else None
    return getattr(settings, "home", None) if settings else None


def run_own_domain_mail_tick_once(*, context: Any = None,
                                  home: str | None = None,
                                  now: float | None = None
                                  ) -> dict[str, Any]:
    """One reaper pass: delete expired own-domain mailboxes (and
    their messages) from the durable store. Never raises — one bad
    tick reports its error and deletes nothing."""
    try:
        store = OwnDomainStore(home=_tick_home(context, home))
        swept = store.sweep(now=now)
        live = store.counts(now=now)
        return {"ok": True, "swept_mailboxes": swept["mailboxes"],
                "swept_messages": swept["messages"],
                "live_mailboxes": live["mailboxes"],
                "live_messages": live["messages"], "error": ""}
    except Exception as exc:  # noqa: BLE001 - one bad tick never kills
        return {"ok": False, "swept_mailboxes": 0, "swept_messages": 0,
                "live_mailboxes": 0, "live_messages": 0,
                "error": f"own-domain mail sweep failed: {exc}"}


def ensure_own_domain_mail_job(scheduler: Any, *,
                               spec: str = OWN_DOMAIN_MAIL_SPEC
                               ) -> Any | None:
    """Register the own-domain mail sweep job; idempotent by name.

    A job with the same name but a stale spec is replaced; a job
    that already matches is returned untouched. Best-effort — a
    scheduler that cannot store the job returns None rather than
    raising.
    """
    if scheduler is None:
        return None
    try:
        import contextlib

        for job in scheduler.list_jobs() or []:
            if isinstance(job, dict):
                name, job_spec, job_id = (job.get("name"),
                                          job.get("spec"),
                                          job.get("id"))
            else:
                name = getattr(job, "name", None)
                job_spec = getattr(job, "spec", None)
                job_id = getattr(job, "id", None)
            if name != OWN_DOMAIN_MAIL_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            OWN_DOMAIN_MAIL_JOB, spec, "tool",
            {"tool": OWN_DOMAIN_MAIL_TOOL, "args": {}}, routine=True)
    except Exception:
        return None


def own_domain_status(*, home: str | None = None) -> dict[str, Any]:
    """Honest status for the own-domain rung: configured?, domain,
    listener alive?, live counts, next expiry. Never raises."""
    domain = own_domain_name()
    if not _own_domain_valid(domain):
        return {"configured": False, "domain": "",
                "reason": "NM_TEMPMAIL_DOMAIN is not set — set it to a"
                          " domain you own (with its MX pointed at this"
                          " host); the cascade stays on the public"
                          " providers until then",
                "listener": False, "live_mailboxes": 0, "live_messages": 0,
                "next_expiry": 0.0, "ttl_s": _own_ttl_s(), "error": ""}
    try:
        store = OwnDomainStore(home=home)
        live = store.counts()
        return {"configured": True, "domain": domain,
                "reason": "",
                "listener": own_domain_listener_alive(store=store),
                "live_mailboxes": live["mailboxes"],
                "live_messages": live["messages"],
                "next_expiry": store.next_expiry(),
                "ttl_s": _own_ttl_s(), "error": ""}
    except Exception as exc:  # noqa: BLE001 - status never raises
        return {"configured": True, "domain": domain, "reason": "",
                "listener": False, "live_mailboxes": 0, "live_messages": 0,
                "next_expiry": 0.0, "ttl_s": _own_ttl_s(),
                "error": f"own-domain status failed: {exc}"}


PROVIDERS: dict[str, type[TempMailProvider]] = {
    "owndomain": OwnDomainProvider,
    "mailtm": MailTmProvider,
    "1secmail": OneSecMailProvider,
    "guerrillamail": GuerrillaMailProvider,
}

#: cascade order — the owner's own domain first when configured
#: (not a known disposable, so signup systems don't blocklist it;
#: its probe() is False when unconfigured, so the cascade skips it
#: silently), then mail.tm, 1secmail, guerrilla. Providers whose
#: probe() fails are skipped.
CASCADE_PROVIDERS = ["owndomain", "mailtm", "1secmail", "guerrillamail"]


def get_provider(name: str = "1secmail") -> TempMailProvider:
    cls = PROVIDERS.get(name.lower())
    if cls is None:
        raise ValueError(f"unknown temp-mail provider {name!r} "
                         f"(known: {sorted(PROVIDERS)})")
    return cls()


def grab_address(provider: str = "1secmail") -> TempAddress:
    return get_provider(provider).grab_address()


def grab_address_cascade(providers: list[str] | None = None) -> TempAddress:
    """Try providers in order until one yields a usable address.

    Providers whose ``probe()`` fails are skipped silently — a down
    provider is "skip me", never an exception to the caller.
    """
    last: Exception | None = None
    for name in providers or CASCADE_PROVIDERS:
        try:
            prov = get_provider(name)
        except Exception as exc:  # noqa: BLE001
            _log.debug("temp-mail provider %s unavailable: %s", name, exc)
            last = exc
            continue
        try:
            if not prov.probe():
                _log.debug("temp-mail provider %s probe failed — skipping",
                           name)
                continue
            addr = prov.grab_address()
            if addr.address:
                _log.info("temp-mail: got %s via %s", addr.address, name)
                return addr
        except Exception as exc:  # noqa: BLE001
            _log.debug("temp-mail provider %s failed: %s", name, exc)
            last = exc
    raise RuntimeError(f"all temp-mail providers failed (last: {last})")


def delete_message(addr: TempAddress, msg: MailMessage,
                   provider: str | None = None) -> bool:
    """Delete one message via its provider. Returns False when the
    provider doesn't support deletion."""
    prov = get_provider(provider or addr.provider or CASCADE_PROVIDERS[0])
    return prov.delete_message(addr, msg)


def wait_message(addr: TempAddress, *, timeout_s: float = 180,
                 poll_s: float = 8, sender_hint: str = "",
                 subject_hint: str = "",
                 provider: str | None = None) -> MailMessage | None:
    prov = get_provider(provider or addr.provider or "1secmail")
    return prov.wait_for_message(addr, timeout_s=timeout_s, poll_s=poll_s,
                                 sender_hint=sender_hint,
                                 subject_hint=subject_hint)


def wait_code(addr: TempAddress, *, timeout_s: float = 180,
              poll_s: float = 8, sender_hint: str = "",
              provider: str | None = None) -> str | None:
    """Wait for a verification email and return the extracted code."""
    msg = wait_message(addr, timeout_s=timeout_s, poll_s=poll_s,
                       sender_hint=sender_hint, provider=provider)
    return msg.code if msg else None
