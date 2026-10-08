"""WhatsApp adapter — Python client for the Node/Baileys bridge.

WhatsApp has no friendly official API for a personal account (the Cloud API
is business-oriented and cannot do the userbot-style companionship), so this
system talks to the *bridge* in ``bridge/whatsapp-bridge.mjs`` — a small Node
process that holds the WhatsApp Web session (QR login, media, presence) and
speaks a JSON-lines protocol over localhost TCP.

Protocol (one JSON object per line):

  bridge -> python
    {"type":"status","state":"open|closed","user":"<jid>"}
    {"type":"qr","data":"<qr string>"}
    {"type":"message","chat":{"id","kind","title"},"from":{"id","name"},
     "text":"...","media":[{"path","mime","kind"}],"reply_to":"","ts":<ms>}
    {"id":"<req>","ok":true,"error":""}          # response to a command

  python -> bridge
    {"id":"<req>","cmd":"send","chat":"<jid>","text":"...","reply_to":""}
    {"id":"<req>","cmd":"send_media","chat":"<jid>","path":"...","caption":"...","ptt":false}
    {"id":"<req>","cmd":"typing","chat":"<jid>","seconds":3}
    {"id":"<req>","cmd":"history","chat":"<jid>","limit":20}
    {"id":"<req>","cmd":"read","chat":"<jid>"}            # best-effort receipts
    {"id":"<req>","cmd":"chats","limit":30}               # best-effort listing
    {"id":"<req>","cmd":"status"}

The bridge owns the credential state (``.creds/`` directory); this client
stores nothing. Commands the bridge doesn't implement answer ok:false and
the adapter degrades gracefully (``mark_read``/``chats`` return False/[]).
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from typing import Any

from ...core.logging_setup import get_logger
from .base import ChatAdapter, ChatKind, ChatMessage, ChatRef, IncomingHandler, MediaRef, SendResult

__all__ = ["WhatsAppAdapter"]

_log = get_logger(__name__)


def _split_text(text: str, *, limit: int = 4000) -> list[str]:
    """Split long text on paragraph/line boundaries so no chunk exceeds
    *limit* characters. Short text returns unchanged as a single chunk."""
    text = text or ""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in text.split("\n\n"):
        # a single huge paragraph: hard-split it
        while len(para) > limit:
            cut = para.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            chunks.append(para[:cut])
            para = para[cut:].lstrip()
        piece_len = len(para) + 2
        if current and current_len + piece_len > limit:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += piece_len
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if c] or [text[:limit]]


class WhatsAppAdapter(ChatAdapter):
    """Client for the local WhatsApp bridge (bridge/whatsapp-bridge.mjs)."""

    name = "whatsapp"
    supported_kinds = (ChatKind.DM, ChatKind.GROUP)

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8787,
        reconnect_delay: float = 3.0,
        media_dir: str = "data/media/whatsapp",
    ) -> None:
        super().__init__(media_dir=media_dir)
        self.host = host
        self.port = int(port)
        self.reconnect_delay = max(1.0, float(reconnect_delay))
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._responses: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._responses_lock = threading.Lock()
        self.connected = threading.Event()
        self.user_jid: str = ""
        # last QR login code, so the owner/app can surface it without
        # watching stdout (the bridge re-emits a QR whenever the session
        # needs re-linking)
        self.last_qr: str = ""
        self.last_qr_at: float = 0.0
        self.bridge_state: str = "unknown"

    # ── helpers ──────────────────────────────────────────────────────────────
    def _jid(self, chat: ChatRef) -> str:
        """Normalize a chat to a WhatsApp JID."""
        jid = (chat.chat_id or "").strip()
        if chat.kind == ChatKind.DM and jid and not jid.endswith(
                ("@s.whatsapp.net", "@whatsapp.net", "@c.us", "@g.us")):
            # plain phone number → personal chat JID
            digits = "".join(c for c in jid if c.isdigit() or c == "+")
            jid = f"{digits}@c.us" if digits else jid
        return jid

    # ── low-level protocol ───────────────────────────────────────────────────
    def _write_line(self, obj: dict[str, Any]) -> None:
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        with self._send_lock:
            if self._sock is None:
                raise ConnectionError("whatsapp bridge not connected")
            self._sock.sendall(data)

    def _send_cmd(self, cmd: dict[str, Any], *, timeout: float = 25.0) -> dict[str, Any]:
        req_id = uuid.uuid4().hex[:12]
        event = threading.Event()
        self._responses_lock.acquire()
        self._responses[req_id] = (event, {})
        self._responses_lock.release()
        try:
            self._write_line({**cmd, "id": req_id})
            if not event.wait(timeout):
                return {"id": req_id, "ok": False, "error": "bridge response timeout"}
            with self._responses_lock:
                return self._responses.pop(req_id, (event, {}))[1]
        except ConnectionError as exc:
            return {"id": req_id, "ok": False, "error": str(exc)}
        finally:
            # never leak the pending entry — a late response must not
            # wake a future command reusing nothing (ids are unique, but
            # the dict would grow without bound)
            with self._responses_lock:
                self._responses.pop(req_id, None)

    def _reader_loop(self, handler: IncomingHandler) -> None:
        sock = self._sock
        if sock is None:
            return
        buf = b""
        try:
            while not self.stopped:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line.decode("utf-8", "replace"))
                    except json.JSONDecodeError:
                        continue
                    self._dispatch(obj, handler)
        except OSError as e:
            _log.debug("whatsapp bridge read loop ended: %s", e)
        finally:
            self.connected.clear()

    def _dispatch(self, obj: dict[str, Any], handler: IncomingHandler) -> None:
        kind = obj.get("type")
        if kind in (None, "response") and "id" in obj:
            # response to one of our commands — the bridge sends these
            # both untyped and as {"type": "response", ...}; accept both
            req_id = str(obj.get("id") or "")
            if req_id:
                with self._responses_lock:
                    pending = self._responses.get(req_id)
                if pending is not None:
                    pending[1].update(obj)
                    pending[0].set()
            return
        if kind == "status":
            self.user_jid = str(obj.get("user") or "")
            state = str(obj.get("state") or "")
            self.bridge_state = state
            if state == "open":
                self.connected.set()
                _log.info("whatsapp bridge connected as %s", self.user_jid)
            else:
                self.connected.clear()
                if state:
                    _log.warning("whatsapp bridge state: %s", state)
        elif kind == "qr":
            self.last_qr = str(obj.get("data") or "")
            self.last_qr_at = time.time()
            self.connected.clear()
            _log.warning("whatsapp QR received — scan it with WhatsApp "
                         "(Linked Devices); also available via health()")
            print("WHATSAPP QR:\n" + self.last_qr)
        elif kind == "message":
            chat = obj.get("chat") or {}
            chat_id = str(chat.get("id") or "")
            if not chat_id:
                return
            ts = obj.get("ts") or 0
            try:
                ts = float(ts) / 1000.0
            except (TypeError, ValueError):
                ts = time.time()
            self._deliver(
                handler,
                ChatMessage(
                    chat=ChatRef(
                        platform=self.name,
                        chat_id=chat_id,
                        kind=str(chat.get("kind") or ChatKind.DM),
                        title=str(chat.get("title") or ""),
                        peer=str((obj.get("from") or {}).get("id") or chat_id),
                    ),
                    incoming=True,
                    text=str(obj.get("text") or ""),
                    sender=str((obj.get("from") or {}).get("name") or ""),
                    media=[
                        MediaRef(path=str(m.get("path") or ""), mime=str(m.get("mime") or ""),
                                 kind=str(m.get("kind") or "file"))
                        for m in (obj.get("media") or [])
                        if m.get("path")
                    ],
                    reply_to=str(obj.get("reply_to") or ""),
                    ts=ts,
                ),
            )

    # ── lifecycle ────────────────────────────────────────────────────────────
    def run(self, handler: IncomingHandler) -> None:
        """Connect, read forever, reconnect with backoff."""
        delay = self.reconnect_delay
        while not self.stopped:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=15)
                self._sock = sock
                sock.settimeout(1.0)
                reader = threading.Thread(target=self._reader_loop, args=(handler,),
                                          name=f"wa-{self.name}-reader", daemon=True)
                reader.start()
                delay = self.reconnect_delay
                # The reader clears `connected` on exit. If it dies the
                # socket dropped and we must reconnect — watch its
                # liveness, don't just sleep until stopped.
                while not self.stopped and reader.is_alive():
                    time.sleep(0.25)
                if not self.stopped:
                    _log.warning("whatsapp bridge connection lost; "
                                 "reconnecting in %.0fs", delay)
            except OSError as exc:
                self.connected.clear()
                self._sock = None
                if self.stopped:
                    return
                _log.warning("whatsapp bridge unreachable at %s:%s (%s); retrying in %.0fs",
                             self.host, self.port, exc, delay)
            finally:
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except OSError:  # noqa: E103 - socket already dead
                        pass
                    self._sock = None
            if self.stopped:
                return
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

    def wait_for_bridge(self, timeout: float = 60.0) -> bool:
        started = time.time()
        while time.time() - started < timeout:
            if self.connected.is_set():
                return True
            if self.stopped:
                return False
            time.sleep(0.2)
        return False

    def preflight(self) -> None:
        """One-time setup check: is the bridge reachable at all?

        Unlike Telethon's interactive login this never prompts — it just
        fails fast with a clear message when the Node bridge isn't running,
        instead of letting ``run()`` spin on reconnect backoff silently.
        """
        try:
            sock = socket.create_connection((self.host, self.port), timeout=5)
            sock.close()
        except OSError as exc:
            raise RuntimeError(
                f"whatsapp bridge not reachable at {self.host}:{self.port} "
                f"({exc}); start bridge/whatsapp-bridge.mjs first") from exc

    # ── outbound ─────────────────────────────────────────────────────────────
    def _send_with_retry(self, chat: ChatRef, text: str,
                         *, reply_to: str = "") -> SendResult:
        """One text send with a single retry on transient bridge failure."""
        started = time.perf_counter()
        jid = self._jid(chat)
        last_error = "bridge rejected send"
        for attempt in (1, 2):
            response = self._send_cmd(
                {"cmd": "send", "chat": jid, "text": text,
                 "reply_to": reply_to})
            if response.get("ok"):
                self.stats["sent"] += 1
                return SendResult(ok=True, platform=self.name,
                                  message_id=str(response.get("id") or ""),
                                  seconds=time.perf_counter() - started)
            last_error = str(response.get("error") or last_error)
            # only retry transient failures, not rejections
            if "timeout" not in last_error.lower() and \
               "not connected" not in last_error.lower():
                break
            if attempt == 1:
                time.sleep(1.0)
        self.stats["send_errors"] += 1
        return SendResult(ok=False, platform=self.name, error=last_error,
                          seconds=time.perf_counter() - started)

    def send(self, chat: ChatRef, text: str, *,
             reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             parse_mode: str = "") -> SendResult:
        started = time.perf_counter()
        if not self.connected.is_set():
            return SendResult(ok=False, platform=self.name, error="bridge not connected",
                              seconds=time.perf_counter() - started)
        # Convert Telegram HTML / canonical markdown to WhatsApp markdown
        # (game output and styled menus are authored Telegram-first).
        try:
            from .platforms import format_for_platform
            text = format_for_platform(text, "whatsapp")
        except Exception:  # noqa: BLE001
            pass
        # WhatsApp caps a text message well above this; split long sends so
        # one giant alert doesn't get silently truncated by the bridge.
        chunks = _split_text(text, limit=4000)
        result: SendResult | None = None
        for i, chunk in enumerate(chunks):
            result = self._send_with_retry(
                chat, chunk,
                reply_to=reply_to if i == 0 else "")
            if not result.ok:
                return result
        assert result is not None
        return result

    def send_media(self, chat: ChatRef, media: MediaRef, *,
                   caption: str = "") -> SendResult:
        """Send a photo/video/audio/document via the bridge's send_media."""
        started = time.perf_counter()
        if not self.connected.is_set():
            return SendResult(ok=False, platform=self.name,
                              error="bridge not connected",
                              seconds=time.perf_counter() - started)
        path = (media.path or "").strip()
        if not path:
            return SendResult(ok=False, platform=self.name,
                              error="media has no path",
                              seconds=time.perf_counter() - started)
        cmd: dict[str, Any] = {
            "cmd": "send_media",
            "chat": self._jid(chat),
            "path": path,
            "caption": caption,
        }
        kind = (media.kind or "").lower()
        if kind in ("voice", "ptt"):
            cmd["ptt"] = True  # voice-note bubble instead of an audio file
        response = self._send_cmd(cmd, timeout=60)
        if response.get("ok"):
            self.stats["sent"] += 1
            return SendResult(ok=True, platform=self.name,
                              message_id=str(response.get("id") or ""),
                              seconds=time.perf_counter() - started)
        self.stats["send_errors"] += 1
        return SendResult(ok=False, platform=self.name,
                          error=str(response.get("error") or "bridge rejected media"),
                          seconds=time.perf_counter() - started)

    def send_voice(self, chat: ChatRef, audio_path: str, *,
                   caption: str = "") -> SendResult:
        """Send a voice note. Convenience wrapper over :meth:`send_media`."""
        return self.send_media(
            chat, MediaRef(path=audio_path, kind="voice"), caption=caption)

    def mark_read(self, chat: ChatRef) -> bool:
        """Send read receipts for a chat. Best-effort: older bridges that
        don't implement the ``read`` command answer ok:false and we return
        False instead of raising."""
        if not self.connected.is_set():
            return False
        response = self._send_cmd(
            {"cmd": "read", "chat": self._jid(chat)}, timeout=10)
        return bool(response.get("ok"))

    def chats(self, limit: int = 30) -> list[dict[str, Any]]:
        """Recent chats (id, title, kind, last_ts). Best-effort: returns []
        when the bridge doesn't implement the ``chats`` command."""
        if not self.connected.is_set():
            return []
        response = self._send_cmd(
            {"cmd": "chats", "limit": max(1, int(limit))}, timeout=15)
        if not response.get("ok"):
            return []
        out = []
        for entry in response.get("chats") or []:
            out.append({
                "id": str(entry.get("id") or ""),
                "title": str(entry.get("title") or ""),
                "kind": str(entry.get("kind") or ChatKind.DM),
                "last_ts": float(entry.get("ts") or 0) / 1000.0,
            })
        return [c for c in out if c["id"]]

    def typing(self, chat: ChatRef, seconds: float = 3.0) -> bool:
        if not self.connected.is_set() or seconds <= 0:
            return False
        response = self._send_cmd(
            {"cmd": "typing", "chat": self._jid(chat),
             "seconds": max(1, int(seconds))}, timeout=5)
        return bool(response.get("ok"))

    def history(self, chat: ChatRef, limit: int = 20) -> list[ChatMessage]:
        if not self.connected.is_set():
            return []
        response = self._send_cmd(
            {"cmd": "history", "chat": self._jid(chat),
             "limit": int(limit)}, timeout=15)
        if not response.get("ok"):
            return []
        out: list[ChatMessage] = []
        for entry in response.get("messages") or []:
            try:
                ts = float(entry.get("ts") or time.time()) / 1000.0
            except (TypeError, ValueError):
                ts = time.time()
            out.append(
                ChatMessage(
                    chat=chat,
                    incoming=bool(entry.get("incoming", True)),
                    text=str(entry.get("text") or ""),
                    sender=str(entry.get("sender") or ""),
                    media=[
                        MediaRef(path=str(m.get("path") or ""),
                                 mime=str(m.get("mime") or ""),
                                 kind=str(m.get("kind") or "file"))
                        for m in (entry.get("media") or [])
                        if m.get("path")
                    ],
                    reply_to=str(entry.get("reply_to") or ""),
                    ts=ts,
                )
            )
        return out

    def health(self) -> dict[str, Any]:
        return {**super().health(), "connected": self.connected.is_set(),
                "user": self.user_jid, "bridge": f"{self.host}:{self.port}",
                "bridge_state": self.bridge_state,
                "qr_pending": bool(self.last_qr),
                "qr_age_s": round(time.time() - self.last_qr_at, 1)
                if self.last_qr else None}
