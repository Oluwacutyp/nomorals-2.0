"""Webhook adapter: anything that can POST JSON can talk to her.

``POST /hook/<key>`` with a JSON body::

    {"text": "BTC just wicked below 60k", "sender": "sentinel", "chat_id": "alerts"}

The adapter turns it into a normal inbound ChatMessage on the ``webhook``
platform, so the brain, memory, gating and rate limits all apply exactly
like any other chat. Replies go to the configured ``reply_url`` when one
is set, otherwise they're captured for ``take_parts(key)`` polling.

Authentication is a shared secret: ``?token=`` query param or the
``X-Webhook-Token`` header must match ``chat.webhook_token`` when one is
configured. Binds to localhost by default; opening it wider is the
operator's call.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ...core.logging_setup import get_logger
from .base import ChatAdapter, ChatKind, ChatMessage, ChatRef, IncomingHandler, SendResult

__all__ = ["WebhookAdapter"]

_log = get_logger(__name__)


class _HookHandler(BaseHTTPRequestHandler):
    adapter: "WebhookAdapter"  # set per server instance

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet stdlib chatter
        _log.debug("webhook http: " + fmt, *args)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urllib.parse.urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "hook" or not parts[1]:
            self._send_json(404, {"ok": False, "error": "want POST /hook/<key>"})
            return
        key = parts[1]
        adapter = self.adapter
        if adapter.token:
            query = urllib.parse.parse_qs(parsed.query)
            given = (query.get("token", [""])[0]
                     or self.headers.get("X-Webhook-Token", ""))
            if given != adapter.token:
                self._send_json(403, {"ok": False, "error": "bad token"})
                return
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > 1_000_000:
            self._send_json(400, {"ok": False, "error": "empty or oversized body"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, OSError):
            self._send_json(400, {"ok": False, "error": "invalid JSON"})
            return
        text = str(payload.get("text", "") or "")
        if not text.strip():
            self._send_json(400, {"ok": False, "error": "text is required"})
            return
        sender = str(payload.get("sender", "") or key)
        chat_id = str(payload.get("chat_id", "") or key)
        kind = ChatKind.GROUP if payload.get("group") else ChatKind.DM
        message = ChatMessage(
            chat=ChatRef(platform=adapter.name, chat_id=chat_id, kind=kind,
                         title=str(payload.get("title", "") or key), peer=sender),
            incoming=True,
            text=text,
            sender=sender,
            message_id=f"wh-{int(time.time() * 1000)}",
            meta={"hook_key": key},
        )
        adapter._deliver(adapter._handler, message)
        self._send_json(200, {"ok": True, "chat": message.chat.key})

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.rstrip("/") == "/health":
            self._send_json(200, {"ok": True, "platform": "webhook"})
        else:
            self._send_json(404, {"ok": False})


class WebhookAdapter(ChatAdapter):
    """Inbound JSON webhooks in, replies out via reply_url or polling."""

    name = "webhook"
    supported_kinds = (ChatKind.DM, ChatKind.GROUP)

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        token: str = "",
        reply_url: str = "",
        media_dir: str = "data/media/webhook",
    ) -> None:
        super().__init__(media_dir=media_dir)
        self.host = host
        self.port = port
        self.token = token
        self.reply_url = reply_url
        self._server: ThreadingHTTPServer | None = None
        self._handler: IncomingHandler | None = None
        self._parts: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def run(self, handler: IncomingHandler) -> None:
        self._handler = handler
        hook_handler = type("_BoundHookHandler", (_HookHandler,), {"adapter": self})
        self._server = ThreadingHTTPServer((self.host, self.port), hook_handler)
        self.port = self._server.server_address[1]
        _log.info("webhook adapter listening on %s:%d", self.host, self.port)
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

    # ── outbound ──────────────────────────────────────────────────────────
    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        started = time.perf_counter()
        payload = {"chat": chat.key, "chat_id": chat.chat_id,
                   "text": text, "reply_to": reply_to}
        if self.reply_url:
            try:
                req = urllib.request.Request(
                    self.reply_url,
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    if resp.status >= 400:
                        raise OSError(f"reply_url returned {resp.status}")
            except Exception as exc:  # noqa: BLE001 - ordinary failure
                self.stats["send_errors"] += 1
                return SendResult(ok=False, platform=self.name, error=str(exc),
                                  seconds=time.perf_counter() - started)
        else:
            with self._lock:
                self._parts.setdefault(chat.key, []).append(text)
        self.stats["sent"] += 1
        return SendResult(ok=True, platform=self.name,
                          message_id=f"wh-{int(time.time() * 1000)}",
                          seconds=time.perf_counter() - started)

    def take_parts(self, key: str) -> list[str]:
        """Drain captured reply parts for one chat (oldest first)."""
        with self._lock:
            return self._parts.pop(key, [])

    def hook_url(self, key: str) -> str:
        """The URL an external service POSTs to for ``key``."""
        return f"http://{self.host}:{self.port}/hook/{key}"

    def health(self) -> dict[str, Any]:
        info = super().health()
        info["listening"] = f"{self.host}:{self.port}" if self._server else ""
        return info
