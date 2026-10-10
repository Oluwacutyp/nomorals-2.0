"""SMS fallback surface: text a number, get Devon. Works when data is off.

COST HONESTY — read before enabling: SMS runs on Twilio, which costs real
money (roughly $1/month for the phone number, plus a few cents per message
segment in *each* direction — you pay for her replies too). It is OFF by
default: :func:`sms_enabled` requires BOTH an explicit
``NM_CHAT_SMS_ENABLED=1`` opt-in AND a configured ``NM_CHAT_SMS_FROM_NUMBER``.
Never enabled silently.

Behavioral rules:

* SMS is 1:1 only — ``supported_kinds`` is DM; there are no SMS groups.
* Replies are split into numbered ≤160-char segments (153 of content when
  concatenated, leaving room for the ``(1/3)`` prefix and GSM headers).
  Long replies are never silently truncated.
* Only a restricted command set works over SMS (see :data:`SMS_COMMANDS`);
  games, media, exec and everything heavy get an honest "needs the app" reply.
* No proactive SMS, ever. The surface is purely reactive: the owner texts
  first. Urgent owner alerts may go through at most, DeliveryScorer-gated —
  never a broadcast channel.
* Twilio delivers inbound SMS as a form-encoded webhook POST. This adapter
  runs a tiny local HTTP server for that; put it behind your own
  reverse proxy / tunnel and set the Twilio webhook URL to it.
* Twilio signs webhooks (``X-Twilio-Signature``) and this adapter verifies
  them with the account's auth token (:func:`verify_twilio_signature`,
  pure stdlib HMAC-SHA1 — the same algorithm Twilio documents). With the
  token configured, an unsigned/forged POST is rejected 403. Without it,
  payloads are accepted with a warning log — do NOT expose the webhook
  without the token.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ...core.logging_setup import get_logger
from .base import ChatAdapter, ChatKind, ChatMessage, ChatRef, SendResult

__all__ = [
    "SMSAdapter",
    "SMS_COMMANDS",
    "SMS_BLOCKED_REPLY",
    "SMS_MEDIA_REPLY",
    "SMS_NO_TWILIO_REPLY",
    "sms_allows",
    "sms_enabled",
    "split_sms",
    "sms_encoding",
    "sms_segments",
    "normalize_gsm7",
    "verify_twilio_signature",
]

_log = get_logger(__name__)

#: One GSM segment. Multi-part messages use SMS_CONCAT_CHARS of content so the
#: "(i/n) " prefix and the GSM concatenation headers still fit in 160.
SMS_SEGMENT_CHARS = 160
SMS_CONCAT_CHARS = 147

#: Restricted command set for the SMS surface. Kinds are the
#: ``parse_control`` command kinds — anything not listed gets the
#: "needs the app" reply. Read-only and lightweight only: no games, no
#: media, no exec, nothing that burns money or attention.
SMS_COMMANDS: frozenset[str] = frozenset({
    "status",      # her live state
    "help",        # command catalog
    "platforms",   # which surfaces are up
    "list", "commands",  # command listing
    "remember",    # quick notes into long-term memory
    "recall",      # read back a memory
    "spending",    # spending summary vs budgets (read-only)
    "finance",     # price quotes / research (read-only finance brain)
})

SMS_BLOCKED_REPLY = (
    "That needs the app — over SMS I can do /status, /remember, "
    "/spending, and /finance quotes."
)
SMS_MEDIA_REPLY = (
    "I got your message but can't see pictures over SMS — "
    "describe it in words?"
)
SMS_NO_TWILIO_REPLY = (
    "SMS isn't wired up yet — no Twilio number configured. "
    "Set NM_CHAT_SMS_ENABLED=1 and NM_CHAT_SMS_FROM_NUMBER to enable it."
)


def sms_allows(command_kind: str) -> bool:
    """True when ``command_kind`` may run over the SMS surface."""
    return (command_kind or "").strip().lower() in SMS_COMMANDS


def sms_enabled(settings: Any) -> bool:
    """SMS is off by default: needs explicit opt-in AND a Twilio number."""
    chat = getattr(settings, "chat", settings)
    enabled = bool(getattr(chat, "sms_enabled", False))
    number = str(getattr(chat, "sms_from_number", "") or "").strip()
    return enabled and bool(number)


# ── real segment math ─────────────────────────────────────────────────────
# Segmentation is NOT a character count. One character outside the GSM-7
# alphabet forces the WHOLE message into UCS-2 and cuts the per-part
# budget: GSM-7 = 160 chars (153 concatenated), UCS-2 = 70 (67
# concatenated). A curly quote pasted from a word processor can triple
# the billed segments — this is where SMS money quietly leaks.

#: GSM-7 basic + extension characters (the billable-safe alphabet).
_GSM7 = frozenset(
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞ\x1bÆæßÉ !\"#¤%&'()*+,-./"
    "0123456789:;<=>?¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§"
    "¿abcdefghijklmnopqrstuvwxyzäöñüà"
    "^{}\\[~]|€"
)

#: Surgical substitutions that stay in GSM-7 — hand-written, never NFD.
#: NFD de-accenting would rewrite ñ→n (turning "año" into a vulgarity)
#: and strip diacritics that are already free (é, ñ, ü cost nothing).
#: Emoji have no ASCII equivalent and are left alone deliberately.
_GSM7_SUBSTITUTIONS: dict[str, str] = {
    "á": "a", "í": "i", "ó": "o", "ú": "u",
    "Á": "A", "Í": "I", "Ó": "O", "Ú": "U",
    "ç": "c", "Ç": "C",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "–": "-", "—": "-", "…": "...",
}


def sms_encoding(text: str) -> str:
    """``"gsm7"`` or ``"ucs2"`` for ``text``. Pure, testable."""
    try:
        for ch in str(text or ""):
            if ch not in _GSM7:
                return "ucs2"
        return "gsm7"
    except Exception:  # noqa: BLE001
        return "ucs2"


def sms_segments(text: str) -> dict[str, object]:
    """Real segment math for ``text``: encoding, per-part budget, count.

    Returns ``{"encoding", "per_part", "count", "chars"}`` — the numbers
    the carrier bills on, not a character count.
    """
    text = str(text or "")
    encoding = sms_encoding(text)
    single, concat = (160, 153) if encoding == "gsm7" else (70, 67)
    chars = len(text)
    if chars <= single:
        count = 1
    else:
        count = -(-chars // concat)  # ceil division
    return {"encoding": encoding, "per_part": concat, "count": count,
            "chars": chars}


def normalize_gsm7(text: str) -> tuple[str, list[str]]:
    """Replace non-GSM-7 chars that have safe ASCII equivalents.

    Returns ``(normalized_text, substituted_chars)`` so the caller can say
    what changed (and what it saved). Emoji and ñ/é/ü/è are never touched.
    """
    try:
        out: list[str] = []
        changed: list[str] = []
        for ch in str(text or ""):
            if ch in _GSM7:
                out.append(ch)
            elif ch in _GSM7_SUBSTITUTIONS:
                out.append(_GSM7_SUBSTITUTIONS[ch])
                if ch not in changed:
                    changed.append(ch)
            else:
                out.append(ch)  # no safe equivalent — keep, pay the segment
        return "".join(out), changed
    except Exception:  # noqa: BLE001
        return str(text or ""), []


def split_sms(text: str) -> list[str]:
    """Split ``text`` into numbered SMS segments on the REAL budget.

    A single segment goes out unnumbered. Multi-part messages are split on
    word boundaries at the encoding-correct content budget (153 for GSM-7,
    67 for UCS-2), then prefixed ``(i/n)``. Never silently truncates —
    callers send every segment.
    """
    text = text or ""
    info = sms_segments(text)
    single = 160 if info["encoding"] == "gsm7" else 70
    concat = int(info["per_part"])
    if len(text) <= single:
        return [text]
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        # A single pathological word longer than the budget: hard-split it.
        while len(word) > concat:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.append(word[:concat])
            word = word[concat:]
        extra = len(word) + (1 if current else 0)
        if current_len + extra > concat:
            chunks.append(" ".join(current))
            current, current_len = [word], len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        chunks.append(" ".join(current))
    total = len(chunks)
    return [f"({i + 1}/{total}) {chunk}" for i, chunk in enumerate(chunks)]


def verify_twilio_signature(
    auth_token: str,
    url: str,
    params: dict[str, Any],
    signature: str,
) -> bool:
    """Verify a Twilio webhook signature — Twilio's documented algorithm,
    pure stdlib (no twilio package needed).

    The signature is HMAC-SHA1 (key = auth token) over the full webhook URL
    followed by every POST parameter sorted by name (``url + k1 + v1 +
    k2 + v2 + ...``), base64-encoded. Comparison is constant-time. Any
    exception (bad inputs, wrong types) verifies False — never raises.
    """
    try:
        token = str(auth_token or "")
        sig = str(signature or "")
        if not token or not sig:
            return False
        body = str(url or "")
        for key in sorted(str(k) for k in params):
            value = params.get(key)
            if isinstance(value, (list, tuple)):
                value = value[0] if value else ""
            body += str(key) + str(value if value is not None else "")
        digest = hmac.new(token.encode("utf-8"), body.encode("utf-8"),
                          hashlib.sha1).digest()
        expected = base64.b64encode(digest).decode("ascii")
        return hmac.compare_digest(expected, sig)
    except Exception:  # noqa: BLE001 - verification fails closed
        return False


def _webhook_url(adapter: "SMSAdapter", path: str) -> str:
    """Best-effort public URL for signature verification.

    Twilio signs the exact URL it was configured with; behind a tunnel the
    adapter only sees the local path. ``sms_public_url`` (when set) takes
    precedence, else we fall back to the local host:port — which only
    verifies when Twilio posts to that exact URL.
    """
    base = str(getattr(adapter, "public_url", "") or "").rstrip("/")
    if base:
        return base + (path or "")
    return f"http://{adapter.host}:{adapter.port}{path or ''}"


class _TwilioHookHandler(BaseHTTPRequestHandler):
    """Form-encoded Twilio webhook POSTs → the adapter's inbound queue."""

    adapter: "SMSAdapter | None" = None

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet stdlib chatter
        pass

    def _respond(self, code: int = 200) -> None:
        body = b"<Response/>"
        self.send_response(code)
        self.send_header("Content-Type", "text/xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except (TypeError, ValueError):
            length = 0
        raw = self.rfile.read(max(0, length)) if length else b""
        form = urllib.parse.parse_qs(raw.decode("utf-8", "replace"))
        payload = {k: v[0] for k, v in form.items() if v}
        adapter = type(self).adapter
        # Twilio signs webhooks (X-Twilio-Signature). With the auth token
        # configured we verify; without it we accept with a warning (the
        # module docstring says don't expose the webhook that way).
        auth_token = getattr(adapter, "auth_token", "") if adapter else ""
        if auth_token:
            signature = self.headers.get("X-Twilio-Signature") or ""
            url = _webhook_url(adapter, self.path)
            if not verify_twilio_signature(auth_token, url, payload, signature):
                _log.warning("sms webhook: signature verification FAILED — "
                             "rejecting forged/unsigned POST")
                self._respond(code=403)
                return
        elif self.headers.get("X-Twilio-Signature"):
            _log.warning(
                "sms webhook: X-Twilio-Signature present but no auth token "
                "configured — accepting payload unsigned-verified"
            )
        if adapter is not None:
            try:
                message = adapter.handle_webhook(payload)
            except Exception as exc:  # noqa: BLE001 - never die on a bad POST
                _log.warning("sms webhook: handle_webhook failed: %s", exc)
                message = None
            if message is not None and adapter._handler is not None:
                try:
                    adapter._handler(message)
                except Exception as exc:  # noqa: BLE001 - handler errors stay local
                    _log.warning("sms webhook: inbound handler failed: %s", exc)
        self._respond()


class SMSAdapter(ChatAdapter):
    """Devon over SMS via Twilio. Reactive only — the owner texts first."""

    name = "sms"
    supported_kinds = (ChatKind.DM,)

    def __init__(
        self,
        twilio: Any = None,
        *,
        from_number: str = "",
        host: str = "127.0.0.1",
        port: int = 0,
        media_dir: str = "data/media/sms",
        auth_token: str = "",
        public_url: str = "",
    ) -> None:
        super().__init__(media_dir=media_dir)
        #: Duck-typed Twilio connector: needs
        #: ``send_sms(to, body, *, from_number, confirmed)``. None means
        #: "not configured" — sends fail closed, never raise.
        self._twilio = twilio
        self.from_number = (from_number or "").strip()
        self.host = host
        self.port = port
        #: Twilio auth token — enables X-Twilio-Signature verification on
        #: the webhook. Empty = accepted with a warning (don't expose the
        #: webhook that way).
        self.auth_token = (auth_token or "").strip()
        #: The public URL Twilio is configured to POST to (behind a tunnel
        #: the adapter only sees the local path — Twilio signs the exact
        #: configured URL, so set this for verification to pass).
        self.public_url = (public_url or "").strip()
        self._server: ThreadingHTTPServer | None = None
        self._handler: Any = None

    # ── inbound ──────────────────────────────────────────────────────────
    def handle_webhook(self, payload: dict[str, Any]) -> ChatMessage | None:
        """Parse a Twilio inbound-SMS form into a ChatMessage.

        Returns None on malformed payloads — never raises. Twilio sends
        ``NumMedia``/``MediaUrl0`` for MMS; SMS can't carry media into the
        brain, so the message is flagged (``meta["sms_has_media"]``) and the
        runtime answers honestly instead of hallucinating about a picture.
        """
        try:
            sender = str(payload.get("From", "") or "").strip()
            body = str(payload.get("Body", "") or "")
            sid = str(payload.get("MessageSid", "") or "").strip()
            try:
                num_media = int(str(payload.get("NumMedia", "0") or "0"))
            except (TypeError, ValueError):
                num_media = 0
        except Exception:  # noqa: BLE001 - malformed payload shape
            _log.warning("sms webhook: unparseable payload keys")
            return None
        if not sender:
            _log.warning("sms webhook: missing From — dropping")
            return None
        chat = ChatRef.parse(f"sms:{sender}", kind=ChatKind.DM)
        message = ChatMessage(
            chat=chat,
            incoming=True,
            text=body,
            sender=sender,
            sender_id=sender,
            message_id=sid or f"sms-{int(time.time() * 1000)}",
        )
        message.meta["sms_has_media"] = num_media > 0
        message.meta["twilio_sid"] = sid
        return message

    def run(self, handler: Any) -> None:
        self._handler = handler
        hook_handler = type("_BoundSmsHook", (_TwilioHookHandler,), {"adapter": self})
        self._server = ThreadingHTTPServer((self.host, self.port), hook_handler)
        self.port = self._server.server_address[1]
        _log.info("sms adapter listening for Twilio webhooks on %s:%d",
                  self.host, self.port)
        self._server.serve_forever(poll_interval=0.5)

    def stop(self) -> None:
        super().stop()
        server, self._server = self._server, None
        if server is None:
            return

        def _close() -> None:
            server.shutdown()
            server.server_close()

        threading.Thread(target=_close, daemon=True).start()

    def hook_url(self) -> str:
        """The URL to paste into the Twilio console as the inbound webhook."""
        return f"http://{self.host}:{self.port}/"

    # ── outbound ─────────────────────────────────────────────────────────
    def send(self, chat: ChatRef, text: str, *,
             reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             parse_mode: str = "") -> SendResult:
        started = time.perf_counter()
        # SMS is text-only: buttons and parse modes are dropped, not faked.
        if self._twilio is None:
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name,
                              error="twilio not configured — SMS is disabled",
                              seconds=time.perf_counter() - started)
        to = (chat.chat_id or "").strip()
        if not to:
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name,
                              error="no destination number on chat",
                              seconds=time.perf_counter() - started)
        segments = split_sms(text or "")
        last_id = ""
        try:
            for segment in segments:
                # Replies answer an owner-initiated conversation on an
                # explicitly opted-in surface: confirmed, not checkpointed.
                # (The connector still bills per segment — that's the
                # documented cost of this surface.)
                result = self._twilio.send_sms(
                    to, segment,
                    from_number=self.from_number,
                    confirmed=True,
                )
                last_id = str(result.get("sid", "") or last_id)
        except Exception as exc:  # noqa: BLE001 - ordinary failure, never raise
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name, error=str(exc),
                              seconds=time.perf_counter() - started)
        self.stats["sent"] += len(segments)
        return SendResult(ok=True, platform=self.name, message_id=last_id,
                          seconds=time.perf_counter() - started)

    def health(self) -> dict[str, Any]:
        info = super().health()
        info["listening"] = f"{self.host}:{self.port}" if self._server else ""
        info["twilio_configured"] = self._twilio is not None
        info["from_number"] = self.from_number
        info["signature_verified"] = bool(self.auth_token)
        return info
