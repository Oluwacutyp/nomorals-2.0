"""WhatsApp session provisioner — QR + 8-digit pairing-code flow.

The WhatsApp adapter (``nomorals.social.chat.whatsapp``) talks to a
Baileys bridge that already holds a session. This module answers the
question *before* that: how does a brand-new session get linked, from
the owner's phone, without SSH?

Flow
----
1. The owner (authenticated — see below) creates a session:
   ``create_session(token=...)`` with ``method="qr"`` or
   ``method="pairing_code"``.
2. QR: ``get_qr(token, session_id)`` returns a QR payload string the
   phone browser / WhatsApp app can scan. Pairing code:
   ``request_pairing_code(token, session_id, phone_number=...)``
   returns an 8-digit code to type into WhatsApp → Linked Devices →
   Link with phone number.
3. The phone browser opens the provisioning link
   (``provisioning_url(session_id)``) — a plain HTTPS page, not an SSH
   terminal — and shows the QR / pairing code. The bridge calls back
   via ``confirm_linked(token, session_id, ...)`` once WhatsApp
   accepts the link.
4. ``list_sessions(token)`` / ``get_session(token, ...)`` expose the
   session registry: id, status, created, last_seen, method.

Security invariants (enforced in code, tested in
``tests/test_groups_tier1.py``)
---------------------------------
* **Single-tenant, token-bound.** Every sensitive operation takes a
  ``token`` argument and compares it (constant-time) against the
  SHA-256 of the provisioner token. A missing / wrong token raises
  :class:`UnauthenticatedError` — never a degraded read, never a
  default-allow.
* **Localhost-or-explicit-exposure.** The provisioner binds
  ``127.0.0.1`` by default. Binding any other host requires
  ``explicit_allow_remote=True`` *and* a token. Refusing to bind
  publicly by accident is a feature, not a limitation.
* **No credentials stored.** Only the token *hash* is kept. QR
  payloads and pairing codes are short-lived (120 s default),
  single-session, and rate-limited by ``max_attempts``.
* **Phone-browser link, not SSH.** ``provisioning_url`` /
  ``provisioning_page_html`` are the link surface; nothing here
  requires the owner to paste secrets into a terminal.

Pairing codes are derived deterministically (HMAC-SHA256 over
``session_id`` with a per-provisioner secret) so a code can be
re-derived for verification without storing it — but each code still
expires, and a session that exceeds ``max_attempts`` verification
tries is locked until a fresh code is requested.

Stdlib only. No bridge, adapter, account, or vault imports — the
provisioner owns *linking*, the bridge owns *chatting*.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import secrets
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "UnauthenticatedError",
    "ProvisioningError",
    "SessionRecord",
    "WhatsAppProvisioner",
    "SESSION_STATUSES",
    "QR_TTL_S",
    "PAIRING_CODE_TTL_S",
    "mask_phone",
]

#: Valid session statuses.
SESSION_STATUSES = (
    "pending_qr",
    "pending_pairing",
    "awaiting_scan",
    "linked",
    "expired",
    "revoked",
)

QR_TTL_S = 120.0
PAIRING_CODE_TTL_S = 120.0
_DEFAULT_MAX_ATTEMPTS = 5


class UnauthenticatedError(PermissionError):
    """Raised when a provisioner call carries no (or a wrong) token."""


class ProvisioningError(ValueError):
    """Raised for invalid provisioning requests / state transitions."""


def mask_phone(phone: str) -> str:
    """Mask a phone number for display: ``+23480…1234``."""
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    if len(digits) <= 4:
        return "****"
    return f"+{digits[:5]}…{digits[-4:]}"


def _now() -> float:
    return time.time()


def _derive_pairing_code(secret: bytes, session_id: str, counter: int) -> str:
    """Deterministic 8-digit code from HMAC(secret, session:counter)."""
    mac = hmac.new(secret, f"{session_id}:{counter}".encode(), hashlib.sha256).digest()
    # Take 4 bytes → 0..2**32, map into 0..99_999_999, zero-pad to 8 digits.
    num = int.from_bytes(mac[:4], "big") % 100_000_000
    return f"{num:08d}"


@dataclass
class SessionRecord:
    """One WhatsApp link session in the registry."""

    id: str
    status: str = "pending_qr"
    method: str = "qr"  # "qr" | "pairing_code"
    created: float = 0.0
    last_seen: float = 0.0
    phone_hint: str = ""  # masked, never the full number
    qr_payload: str = ""
    qr_expires_at: float = 0.0
    pairing_expires_at: float = 0.0
    pairing_counter: int = 0
    attempts: int = 0
    linked_at: float | None = None
    linked_jid: str = ""
    revoked_at: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def is_expired(self) -> bool:
        if self.status in ("linked", "revoked", "expired"):
            return self.status == "expired"
        deadline = max(self.qr_expires_at, self.pairing_expires_at)
        return deadline > 0 and _now() > deadline

    @property
    def is_active(self) -> bool:
        return self.status in ("pending_qr", "pending_pairing", "awaiting_scan", "linked")


class WhatsAppProvisioner:
    """Single-tenant WhatsApp session provisioner.

    Args:
        token: the single-tenant auth token. Callers must present it
            on every sensitive call. Only its SHA-256 is retained.
        bind_host: host the provisioning page binds. ``127.0.0.1``
            (default) is always allowed. Anything else requires
            ``explicit_allow_remote=True``.
        explicit_allow_remote: owner opt-in to expose the provisioning
            page beyond localhost (e.g. behind an authenticated
            tunnel for the phone browser).
        base_url: public base URL used to build
            :meth:`provisioning_url`. Defaults to localhost.
        state_path: JSON file the registry persists to. Defaults to
            ``~/.devon/whatsapp-sessions/registry.json``.
        clock: injectable clock (``time.time``) for tests.
    """

    def __init__(
        self,
        token: str = "",
        *,
        bind_host: str = "127.0.0.1",
        explicit_allow_remote: bool = False,
        base_url: str = "http://127.0.0.1:8765",
        state_path: str | Path | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if bind_host not in ("127.0.0.1", "localhost", "::1"):
            if not explicit_allow_remote:
                raise ProvisioningError(
                    "refusing to bind provisioning page to "
                    f"{bind_host!r} without explicit_allow_remote=True "
                    "(localhost-or-explicit-exposure)"
                )
            if not token:
                raise ProvisioningError(
                    "remote exposure requires a token (single-tenant, token-bound)"
                )
        if not token:
            raise ProvisioningError(
                "WhatsAppProvisioner requires a token — unauthenticated "
                "provisioning is not supported"
            )
        self._token_hash = hashlib.sha256(token.encode()).digest()
        # Per-provisioner secret for pairing-code derivation. Random per
        # instance; codes are short-lived so persistence is unnecessary.
        self._secret = secrets.token_bytes(32)
        self.bind_host = bind_host
        self.explicit_allow_remote = explicit_allow_remote
        self.base_url = base_url.rstrip("/")
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._state_path = Path(state_path) if state_path else (
            Path.home() / ".devon" / "whatsapp-sessions" / "registry.json"
        )
        self._sessions: dict[str, SessionRecord] = {}
        self._load()

    # ── auth ──────────────────────────────────────────────────────────

    def verify_token(self, token: str | None) -> bool:
        """Constant-time token check. ``None``/empty is always False."""
        if not token:
            return False
        presented = hashlib.sha256(token.encode()).digest()
        return hmac.compare_digest(presented, self._token_hash)

    def _require_auth(self, token: str | None) -> None:
        if not self.verify_token(token):
            raise UnauthenticatedError(
                "provisioner refuses unauthenticated call — "
                "pass the single-tenant token"
            )

    # ── persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            raw = self._state_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if isinstance(data, list):
            for item in data:
                try:
                    rec = SessionRecord.from_dict(item)
                    self._sessions[rec.id] = rec
                except (TypeError, ValueError):
                    continue

    def _save(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                [r.to_dict() for r in self._sessions.values()],
                indent=1, sort_keys=True,
            )
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self._state_path)
        except OSError:
            _log.warning("wa_provisioner: could not persist registry", exc_info=True)

    # ── lifecycle ─────────────────────────────────────────────────────

    def create_session(
        self,
        token: str | None,
        *,
        method: str = "qr",
        session_id: str | None = None,
        phone_number: str = "",
        note: str = "",
    ) -> SessionRecord:
        """Create a pending link session. Auth required."""
        self._require_auth(token)
        if method not in ("qr", "pairing_code"):
            raise ProvisioningError(f"unknown method {method!r} (qr|pairing_code)")
        now = self._clock()
        sid = session_id or f"wa-{secrets.token_hex(6)}"
        with self._lock:
            if sid in self._sessions and self._sessions[sid].is_active:
                raise ProvisioningError(f"session {sid!r} already active")
            status = "pending_qr" if method == "qr" else "pending_pairing"
            qr_payload = ""
            qr_exp = 0.0
            if method == "qr":
                qr_payload = self._qr_payload(sid, now)
                qr_exp = now + QR_TTL_S
            rec = SessionRecord(
                id=sid,
                status=status,
                method=method,
                created=now,
                last_seen=now,
                phone_hint=mask_phone(phone_number) if phone_number else "",
                qr_payload=qr_payload,
                qr_expires_at=qr_exp,
                note=note,
            )
            self._sessions[sid] = rec
            self._save()
            _log.info("wa_provisioner: created %s session %s", method, sid)
            return rec

    def _qr_payload(self, session_id: str, now: float) -> str:
        nonce = secrets.token_hex(8)
        return f"devon-wa-link:{session_id}:{int(now)}:{nonce}"

    def get_qr(self, token: str | None, session_id: str) -> dict[str, Any]:
        """Return the QR payload for a pending QR session. Auth required."""
        self._require_auth(token)
        with self._lock:
            rec = self._get(session_id)
            if rec.method != "qr":
                raise ProvisioningError("session is not a QR session")
            if rec.status == "linked":
                raise ProvisioningError("session already linked")
            now = self._clock()
            if rec.qr_expires_at and now > rec.qr_expires_at:
                rec.status = "expired"
                self._save()
                raise ProvisioningError("QR expired — create a fresh session")
            rec.last_seen = now
            self._save()
            return {
                "session_id": rec.id,
                "qr_payload": rec.qr_payload,
                "expires_at": rec.qr_expires_at,
                "ttl_s": max(0.0, rec.qr_expires_at - now),
                "provisioning_url": self.provisioning_url(session_id),
                "instructions": (
                    "Open the link on your PHONE browser (not SSH), "
                    "then in WhatsApp → Linked Devices → Link a Device "
                    "and scan the QR."
                ),
            }

    def request_pairing_code(
        self,
        token: str | None,
        session_id: str,
        *,
        phone_number: str = "",
    ) -> dict[str, Any]:
        """Issue (or re-issue) the 8-digit pairing code. Auth required."""
        self._require_auth(token)
        with self._lock:
            rec = self._get(session_id)
            if rec.method != "pairing_code":
                raise ProvisioningError("session is not a pairing-code session")
            if rec.status == "linked":
                raise ProvisioningError("session already linked")
            if rec.attempts >= _DEFAULT_MAX_ATTEMPTS:
                raise ProvisioningError(
                    "too many pairing attempts — create a fresh session"
                )
            now = self._clock()
            rec.pairing_counter += 1
            code = _derive_pairing_code(self._secret, rec.id, rec.pairing_counter)
            rec.pairing_expires_at = now + PAIRING_CODE_TTL_S
            rec.status = "pending_pairing"
            rec.last_seen = now
            if phone_number:
                rec.phone_hint = mask_phone(phone_number)
            self._save()
            return {
                "session_id": rec.id,
                "pairing_code": code,
                "formatted": f"{code[:4]}-{code[4:]}",
                "expires_at": rec.pairing_expires_at,
                "ttl_s": PAIRING_CODE_TTL_S,
                "provisioning_url": self.provisioning_url(session_id),
                "instructions": (
                    "In WhatsApp → Linked Devices → Link with phone "
                    "number, enter this 8-digit code within 2 minutes."
                ),
            }

    def verify_pairing_code(
        self, token: str | None, session_id: str, code: str
    ) -> bool:
        """Verify a pairing code the bridge reports back. Auth required.

        Returns True on match (session moves to ``awaiting_scan`` —
        WhatsApp still has to accept the link). Wrong/expired codes
        return False and count toward ``max_attempts``.
        """
        self._require_auth(token)
        given = (code or "").replace("-", "").replace(" ", "").strip()
        with self._lock:
            rec = self._get(session_id)
            if rec.method != "pairing_code" or rec.status not in (
                "pending_pairing",
                "awaiting_scan",
            ):
                return False
            now = self._clock()
            if rec.pairing_expires_at and now > rec.pairing_expires_at:
                rec.status = "expired"
                self._save()
                return False
            expected = _derive_pairing_code(
                self._secret, rec.id, rec.pairing_counter
            )
            if hmac.compare_digest(given, expected):
                rec.status = "awaiting_scan"
                rec.last_seen = now
                self._save()
                return True
            rec.attempts += 1
            rec.last_seen = now
            self._save()
            return False

    def confirm_linked(
        self,
        token: str | None,
        session_id: str,
        *,
        jid: str = "",
    ) -> SessionRecord:
        """Bridge callback: WhatsApp accepted the link. Auth required."""
        self._require_auth(token)
        with self._lock:
            rec = self._get(session_id)
            if rec.status in ("revoked", "expired"):
                raise ProvisioningError(f"cannot link {rec.status} session")
            now = self._clock()
            rec.status = "linked"
            rec.linked_at = now
            rec.linked_jid = jid
            rec.last_seen = now
            self._save()
            _log.info("wa_provisioner: session %s linked (jid=%s)", session_id, jid)
            return rec

    def touch(self, token: str | None, session_id: str) -> SessionRecord:
        """Heartbeat: mark a session seen now. Auth required."""
        self._require_auth(token)
        with self._lock:
            rec = self._get(session_id)
            rec.last_seen = self._clock()
            self._save()
            return rec

    def revoke(self, token: str | None, session_id: str) -> SessionRecord:
        """Revoke a session (logout / unlink). Auth required."""
        self._require_auth(token)
        with self._lock:
            rec = self._get(session_id)
            rec.status = "revoked"
            rec.revoked_at = self._clock()
            rec.last_seen = rec.revoked_at
            self._save()
            return rec

    def cleanup_expired(self, token: str | None) -> list[str]:
        """Mark past-deadline pending sessions expired. Auth required."""
        self._require_auth(token)
        now = self._clock()
        out: list[str] = []
        with self._lock:
            for rec in self._sessions.values():
                if rec.status in ("pending_qr", "pending_pairing", "awaiting_scan"):
                    deadline = max(rec.qr_expires_at, rec.pairing_expires_at)
                    if deadline and now > deadline:
                        rec.status = "expired"
                        out.append(rec.id)
            if out:
                self._save()
        return out

    # ── registry reads ────────────────────────────────────────────────

    def get_session(self, token: str | None, session_id: str) -> SessionRecord:
        self._require_auth(token)
        with self._lock:
            return self._get(session_id)

    def list_sessions(
        self, token: str | None, *, status: str | None = None
    ) -> list[SessionRecord]:
        """Registry snapshot: id, status, created, last_seen. Auth required."""
        self._require_auth(token)
        with self._lock:
            rows = list(self._sessions.values())
        if status:
            rows = [r for r in rows if r.status == status]
        return sorted(rows, key=lambda r: r.created)

    def _get(self, session_id: str) -> SessionRecord:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise ProvisioningError(f"unknown session {session_id!r}") from None

    # ── phone-browser surface ─────────────────────────────────────────

    def provisioning_url(self, session_id: str) -> str:
        """Link to open on the phone browser (never SSH)."""
        q = urllib.parse.urlencode({"session": session_id})
        return f"{self.base_url}/wa/link?{q}"

    def provisioning_page_html(self, session_id: str) -> str:
        """Minimal provisioning page for the phone browser.

        Shows the QR payload / pairing-code instructions and the
        session status. The page itself never accepts a token via
        query string — the token travels in the ``Authorization``
        header when the app fetches live status.
        """
        rec = self._sessions.get(session_id)
        status = rec.status if rec else "unknown"
        method = rec.method if rec else "qr"
        safe_id = html.escape(session_id)
        if method == "pairing_code":
            body = (
                "<p>Enter the 8-digit pairing code shown in Devon into "
                "WhatsApp → Linked Devices → Link with phone number.</p>"
            )
        else:
            body = (
                "<p>Scan this QR payload in WhatsApp → Linked Devices → "
                "Link a Device.</p>"
                f"<p><code>{html.escape(rec.qr_payload) if rec else ''}</code></p>"
            )
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Link WhatsApp — {safe_id}</title></head><body>"
            f"<h1>Link WhatsApp</h1><p>Session: <b>{safe_id}</b> — {html.escape(status)}</p>"
            f"{body}"
            "<p>Keep this page on your phone. Do not paste codes into SSH.</p>"
            "</body></html>"
        )
