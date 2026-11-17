"""Web console adapter: the companion, in your browser.

Inbound never flows from a network peer — the console server below
pushes owner messages into the partner runtime directly, and her
replies land here as captured parts. That split is what makes the
console a *view* of the same partner who lives on the social
platforms: same brain, same mood, same memory, same commands — only
the wire is different.

The console server runs in the bot process, inside :meth:`WebAdapter.run`
(the API server is a separate process with no runtime handle, so it
cannot host the console — the same reason the webhook adapter serves
its own HTTP in-process). Endpoints, all JSON:

* ``POST /console/message {"text": ...}`` — one owner message into the
  runtime; replies are captured, not returned inline (the browser poll
  loop drains them — see :meth:`WebAdapter.take_parts`).
* ``GET /console/parts`` — drain captured reply parts for ``web:console``
  (oldest first; each part delivered once).
* ``GET /console/typing`` — whether a typing indicator should render.
* ``GET /health`` — ``{"ok": true, "platform": "web"}``.

Authentication mirrors the webhook adapter: ``?token=`` or the
``X-Console-Token`` header must match ``chat.web_console_token`` when
one is configured. No token configured means localhost-only binding
plus a warning at bind — the same stance as the API server's no-token
warning.
"""
from __future__ import annotations

import hmac
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ...core.logging_setup import get_logger
from .base import ChatAdapter, ChatKind, ChatMessage, ChatRef, IncomingHandler, MediaRef, SendResult

__all__ = ["WebAdapter"]

_log = get_logger(__name__)

#: Largest console request body accepted (1 MiB, the webhook adapter's bound).
_MAX_BODY_BYTES = 1_000_000


class _ConsoleHandler(BaseHTTPRequestHandler):
    adapter: WebAdapter  # set per server instance

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet stdlib chatter
        _log.debug("web console http: " + fmt, *args)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        adapter = self.adapter
        if not adapter.token:
            return True
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        given = (query.get("token", [""])[0]
                 or self.headers.get("X-Console-Token", ""))
        return hmac.compare_digest(given, adapter.token)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.rstrip("/") != "/console/message":
            self._send_json(404, {"ok": False,
                                  "error": "want POST /console/message"})
            return
        if not self._authorized():
            self._send_json(403, {"ok": False, "error": "bad token"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > _MAX_BODY_BYTES:
            self._send_json(400, {"ok": False, "error": "empty or oversized body"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, OSError):
            self._send_json(400, {"ok": False, "error": "invalid JSON"})
            return
        if not isinstance(payload, dict):
            # A JSON array / string / number has no .get — reject honestly
            # instead of raising AttributeError into the handler thread.
            self._send_json(400, {"ok": False, "error": "JSON body must be an object"})
            return
        text = str(payload.get("text", "") or "")
        if not text.strip():
            self._send_json(400, {"ok": False, "error": "text is required"})
            return
        adapter = self.adapter
        message = ChatMessage(
            chat=adapter.chat,
            incoming=True,
            text=text,
            sender="you",
            message_id=f"web-{int(time.time() * 1000)}",
        )
        adapter._deliver(adapter._handler, message)
        self._send_json(200, {"ok": True, "chat": message.chat.key})

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path == "/health":
            self._send_json(200, {"ok": True, "platform": "web"})
            return
        if not self._authorized():
            self._send_json(403, {"ok": False, "error": "bad token"})
            return
        adapter = self.adapter
        if path == "/console/parts":
            self._send_json(200, {"ok": True,
                                  "parts": adapter.take_parts(adapter.chat.key)})
        elif path == "/console/typing":
            self._send_json(200, {"ok": True,
                                  "typing": adapter.typing_active(adapter.chat.key)})
        else:
            self._send_json(404, {"ok": False})


class WebAdapter(ChatAdapter):
    name = "web"
    supported_kinds = (ChatKind.DM,)

    def __init__(
        self,
        *,
        chat_id: str = "console",
        title: str = "Web Console",
        peer: str = "you",
        host: str = "127.0.0.1",
        port: int = 8791,
        token: str = "",
    ) -> None:
        super().__init__()
        # chat_id "console" matters: PartnerRuntime._is_operator() treats
        # every "*:console" chat as the owner's, so slash commands work.
        self.chat = ChatRef(platform="web", chat_id=chat_id,
                            kind=ChatKind.DM, title=title, peer=peer)
        self.host = host
        self.port = port
        self.token = token
        self._server: ThreadingHTTPServer | None = None
        self._handler: IncomingHandler | None = None
        self._parts: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._last_send: dict[str, float] = {}
        #: Typing indicator state per chat key: unix timestamp until which
        #: "she is typing…" should render. The browser poll loop reads it
        #: via :meth:`typing_active`; ``typing()`` sets it for the full
        #: requested (length-scaled) window.
        self._typing_until: dict[str, float] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    def run(self, handler: IncomingHandler) -> None:
        # The console server, in-process: the API server is a separate
        # process with no runtime handle, so the bot process itself must
        # serve the console — the webhook adapter's proven shape.
        self._handler = handler
        console_handler = type("_BoundConsoleHandler", (_ConsoleHandler,),
                               {"adapter": self})
        self._server = ThreadingHTTPServer((self.host, self.port),
                                           console_handler)
        self.port = self._server.server_address[1]
        if not self.token:
            _log.warning(
                "web console listening on %s:%d with NO token configured; "
                "requests resolve to localhost only — set "
                "chat.web_console_token to require one", self.host, self.port)
        _log.info("web console adapter listening on %s:%d", self.host, self.port)
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

    # ── sending (reply capture) ──────────────────────────────────────────────

    def send(self, chat: ChatRef, text: str, *,
             reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             reply_keyboard: list[list[str]] | None = None,
             parse_mode: str = "",
             fresh: bool = False) -> SendResult:
        with self._lock:
            self._parts.setdefault(chat.key, []).append(text)
            self._last_send[chat.key] = time.time()
        self.stats["sent"] += 1
        return SendResult(ok=True, platform=self.name,
                          message_id=f"web-{int(time.time() * 1000)}")

    def take_parts(self, key: str) -> list[str]:
        """Drain captured reply parts for one chat (oldest first)."""
        with self._lock:
            return self._parts.pop(key, [])

    def pending_parts(self, key: str) -> list[str]:
        """Peek without draining."""
        with self._lock:
            return list(self._parts.get(key, []))

    def last_send_age(self, key: str) -> float | None:
        """Seconds since the last reply part for this chat (None = never)."""
        with self._lock:
            ts = self._last_send.get(key)
            return None if ts is None else time.time() - ts

    def typing(self, chat: ChatRef, seconds: float = 3.0,
               action: str = "typing") -> bool:
        # The web console has no push channel for a typing event — the
        # browser renders "typing…" from the poll loop, which reads
        # typing_active(key). Record the full length-scaled window so the
        # indicator, when the frontend consumes it, lasts as long as the
        # message takes to type.
        if seconds <= 0:
            return False
        with self._lock:
            self._typing_until[chat.key] = time.time() + float(seconds)
        return True

    def typing_active(self, key: str) -> bool:
        """True while a typing indicator should render for this chat.

        Consumed by the web console's poll loop. False when no typing
        window is open or it has expired.
        """
        with self._lock:
            until = self._typing_until.get(key, 0.0)
        return time.time() < until

    def send_media(self, chat: ChatRef, media: MediaRef, *, caption: str = "",
                   parse_mode: str = "") -> SendResult:
        text = f"[media: {media.path}]" + (f" — {caption}" if caption else "")
        return self.send(chat, text, reply_to="")

    def console_url(self) -> str:
        """The base URL the browser console talks to."""
        return f"http://{self.host}:{self.port}"

    def health(self) -> dict[str, Any]:
        info = {**super().health(),
                "note": "web console adapter: replies captured in-process"}
        info["listening"] = f"{self.host}:{self.port}" if self._server else ""
        return info
