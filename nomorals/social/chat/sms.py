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
* Twilio signs webhooks (``X-Twilio-Signature``). This codebase has no
  signature validator yet, so payloads are accepted with a warning log —
  do NOT treat inbound SMS as authenticated until one exists.
"""

from __future__ import annotations

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


def split_sms(text: str) -> list[str]:
    """Split ``text`` into numbered SMS segments, each ≤160 chars.

    A single segment goes out unnumbered. Multi-part messages are split on
    word boundaries at ≤147 chars of content, then prefixed ``(i/n)``.
    Never silently truncates — callers send every segment.
    """
    text = text or ""
    if len(text) <= SMS_SEGMENT_CHARS:
        return [text]
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        # A single pathological word longer than the budget: hard-split it.
        while len(word) > SMS_CONCAT_CHARS:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.append(word[:SMS_CONCAT_CHARS])
            word = word[SMS_CONCAT_CHARS:]
        extra = len(word) + (1 if current else 0)
        if current_len + extra > SMS_CONCAT_CHARS:
            chunks.append(" ".join(current))
            current, current_len = [word], len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        chunks.append(" ".join(current))
    total = len(chunks)
    return [f"({i + 1}/{total}) {chunk}" for i, chunk in enumerate(chunks)]


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
        # Twilio signs webhooks; we have no validator in this codebase yet.
        if self.headers.get("X-Twilio-Signature"):
            _log.warning(
                "sms webhook: X-Twilio-Signature present but no validator "
                "in this codebase — accepting payload unsigned-verified"
            )
        adapter = type(self).adapter
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
    ) -> None:
        super().__init__(media_dir=media_dir)
        #: Duck-typed Twilio connector: needs
        #: ``send_sms(to, body, *, from_number, confirmed)``. None means
        #: "not configured" — sends fail closed, never raise.
        self._twilio = twilio
        self.from_number = (from_number or "").strip()
        self.host = host
        self.port = port
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
        return info
