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
    {"id":"<req>","cmd":"typing","chat":"<jid>","seconds":3}
    {"id":"<req>","cmd":"history","chat":"<jid>","limit":20}
    {"id":"<req>","cmd":"status"}

The bridge owns the credential state (``.creds/`` directory); this client
stores nothing.
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
            with self._responses_lock:
                self._responses.pop(req_id, None)
            return {"id": req_id, "ok": False, "error": str(exc)}

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
        except OSError:
            pass
        finally:
            self.connected.clear()

    def _dispatch(self, obj: dict[str, Any], handler: IncomingHandler) -> None:
        kind = obj.get("type")
        if kind == "response" or "id" in obj and "cmd" not in obj and kind is None:
            req_id = obj.get("id", "")
            if req_id:
                with self._responses_lock:
                    pending = self._responses.get(req_id)
                if pending is not None:
                    pending[1].update(obj)
                    pending[0].set()
            return
        if kind == "status":
            self.user_jid = str(obj.get("user") or "")
            if obj.get("state") == "open":
                self.connected.set()
                _log.info("whatsapp bridge connected as %s", self.user_jid)
            else:
                self.connected.clear()
        elif kind == "qr":
            _log.info("whatsapp QR received — scan it with WhatsApp (Linked Devices)")
            print("WHATSAPP QR:\n" + str(obj.get("data", "")))
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
                # The reader loop sets/clears `connected` from status events;
                # just wait until we're told to stop.
                while not self.stopped:
                    time.sleep(0.25)
            except OSError as exc:
                self.connected.clear()
                self._sock = None
                if self.stopped:
                    return
                _log.warning("whatsapp bridge unreachable at %s:%s (%s); retrying in %.0fs",
                             self.host, self.port, exc, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            else:
                delay = self.reconnect_delay
            finally:
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                    self._sock = None

    def wait_for_bridge(self, timeout: float = 60.0) -> bool:
        started = time.time()
        while time.time() - started < timeout:
            if self.connected.is_set():
                return True
            if self.stopped:
                return False
            time.sleep(0.2)
        return False

    # ── outbound ─────────────────────────────────────────────────────────────
    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        started = time.perf_counter()
        if not self.connected.is_set():
            return SendResult(ok=False, platform=self.name, error="bridge not connected",
                              seconds=time.perf_counter() - started)
        jid = chat.chat_id
        if chat.kind == ChatKind.DM and not jid.endswith(("@s.whatsapp.net", "@whatsapp.net", "@c.us")):
            jid = f"{jid}@c.us"
        response = self._send_cmd({"cmd": "send", "chat": jid, "text": text, "reply_to": reply_to})
        if response.get("ok"):
            self.stats["sent"] += 1
            return SendResult(ok=True, platform=self.name,
                              message_id=str(response.get("id") or ""),
                              seconds=time.perf_counter() - started)
        self.stats["send_errors"] += 1
        return SendResult(ok=False, platform=self.name,
                          error=str(response.get("error") or "bridge rejected send"),
                          seconds=time.perf_counter() - started)

    def typing(self, chat: ChatRef, seconds: float = 3.0) -> bool:
        if not self.connected.is_set() or seconds <= 0:
            return False
        jid = chat.chat_id
        if chat.kind == ChatKind.DM and not jid.endswith(("@s.whatsapp.net", "@whatsapp.net", "@c.us")):
            jid = f"{jid}@c.us"
        response = self._send_cmd({"cmd": "typing", "chat": jid, "seconds": max(1, int(seconds))},
                                  timeout=5)
        return bool(response.get("ok"))

    def history(self, chat: ChatRef, limit: int = 20) -> list[ChatMessage]:
        if not self.connected.is_set():
            return []
        jid = chat.chat_id
        if chat.kind == ChatKind.DM and not jid.endswith(("@s.whatsapp.net", "@whatsapp.net", "@c.us")):
            jid = f"{jid}@c.us"
        response = self._send_cmd({"cmd": "history", "chat": jid, "limit": int(limit)}, timeout=15)
        if not response.get("ok"):
            return []
        out: list[ChatMessage] = []
        for entry in response.get("messages") or []:
            out.append(
                ChatMessage(
                    chat=chat,
                    incoming=True,
                    text=str(entry.get("text") or ""),
                    sender=str(entry.get("sender") or ""),
                    ts=float(entry.get("ts") or time.time()) / 1000.0,
                )
            )
        return out

    def send_voice(self, chat: ChatRef, audio_path: str, *, caption: str = "") -> SendResult:
        """Send a voice note via WhatsApp.
        
        Args:
            chat: Target chat
            audio_path: Path to audio file (OGG/MP3)
            caption: Optional caption
            
        Returns:
            SendResult
        """
        if not self.connected.is_set():
            return SendResult(ok=False, platform=self.name, error="not connected", seconds=0)
        
        jid = chat.chat_id
        if chat.kind == ChatKind.DM and not jid.endswith(("@s.whatsapp.net", "@whatsapp.net", "@c.us")):
            jid = f"{jid}@c.us"
        
        # Send as media with ptt=true (push-to-talk = voice note)
        cmd = {
            "cmd": "send_media",
            "chat": jid,
            "path": audio_path,
            "caption": caption,
            "ptt": True,  # Voice note flag
        }
        
        response = self._send_cmd(cmd, timeout=30)
        
        if response.get("ok"):
            return SendResult(ok=True, platform=self.name, seconds=0)
        
        return SendResult(ok=False, platform=self.name, error=str(response.get("error", "failed")), seconds=0)
    
    def health(self) -> dict[str, Any]:
        return {**super().health(), "connected": self.connected.is_set(),
                "user": self.user_jid, "bridge": f"{self.host}:{self.port}"}
