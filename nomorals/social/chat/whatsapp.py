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
     "text":"...","media":[{"path","mime","kind"}],"reply_to":"","mentioned":false,"ts":<ms>}
    {"type":"receipt","chat":"<jid>","ids":["<msg id>",...],"kind":"delivered|read"}
    {"id":"<req>","ok":true,"error":""}          # response to a command

  python -> bridge
    {"id":"<req>","cmd":"send","chat":"<jid>","text":"...","reply_to":""}
    {"id":"<req>","cmd":"send_media","chat":"<jid>","path":"...","caption":"...","ptt":false}
    {"id":"<req>","cmd":"typing","chat":"<jid>","seconds":3,"presence":"composing|recording"}
    {"id":"<req>","cmd":"history","chat":"<jid>","limit":20}
    {"id":"<req>","cmd":"read","chat":"<jid>"}            # best-effort receipts
    {"id":"<req>","cmd":"chats","limit":30}               # best-effort listing
    {"id":"<req>","cmd":"status"}
    # group/community depth (bridge/whatsapp-groups.mjs — read-only):
    {"id":"<req>","cmd":"group_list"}                   # all groups: id/subject/size/isCommunity
    {"id":"<req>","cmd":"group_info","chat":"<@g.us>"}  # metadata + admins + invite link (if permitted)
    {"id":"<req>","cmd":"group_participants","chat":"<@g.us>"}  # roster with admin roles
    {"id":"<req>","cmd":"community_list"}              # communities + their linked groups

Offline resilience: when the bridge is down, ``send``/``send_media`` queue
to a persistent outbox (``data/chat/outbox/whatsapp.jsonl``) and flush FIFO
on reconnect — a bridge restart no longer eats composed replies.

The bridge owns the credential state (``.creds/`` directory); this client
stores nothing. Commands the bridge doesn't implement answer ok:false and
the adapter degrades gracefully (``mark_read``/``chats`` return False/[]).
"""

from __future__ import annotations

import json
import os
import random as _random
import socket
import threading
import time
import uuid
from typing import Any

from ...core.logging_setup import get_logger
from .base import ChatAdapter, ChatKind, ChatMessage, ChatRef, IncomingHandler, MediaRef, SendResult

__all__ = ["WhatsAppAdapter", "Outbox", "GROUP_CAPABILITIES"]

_log = get_logger(__name__)

#: Group/community capabilities exposed over the bridge JSON-lines protocol
#: (bridge/whatsapp-groups.mjs — read-only). One source of truth: the
#: WhatsApp-native menu (control.whatsapp_menu) renders its
#: "groups & communities" section from this table, and the adapter methods
#: below implement it. ``(key, plain_words, what_it_does)``.
GROUP_CAPABILITIES: tuple[tuple[str, str, str], ...] = (
    ("groups", "list my groups",
     "every group: name, member count, community or not"),
    ("group info", "info on a group by name",
     "subject, description, admins, settings, invite link"),
    ("group members", "who is in a group by name",
     "the member roster with admin roles"),
    ("communities", "list my communities",
     "communities and their linked groups"),
)

#: How many unsent messages the outbox holds (oldest dropped past this).
OUTBOX_MAX_ENTRIES = 200
#: Entries older than this are dropped on flush — a day-old "typing…"
#: reply is worse than silence.
OUTBOX_TTL_S = 24 * 3600.0


def _jittered_backoff(base: float, attempt: int, *, cap: float = 60.0,
                      rng: Any = None) -> float:
    """Exponential backoff with full-ish jitter for reconnect loops.

    Deterministic when ``rng`` is an injected ``random.Random`` (tests);
    uses the module random otherwise.
    """
    rng = rng or _random
    delay = min(base * (2.0 ** max(0, attempt - 1)), cap)
    return max(0.5, delay * rng.uniform(0.7, 1.3))


class Outbox:
    """Persistent send queue for bridge outages.

    When the WhatsApp bridge is down, sends land here (JSON-lines, one
    file) instead of dying. The adapter flushes FIFO on reconnect; entries
    past ``OUTBOX_TTL_S`` are dropped, not delivered stale. Bounded at
    ``OUTBOX_MAX_ENTRIES`` — the queue is a resilience buffer, not an
    unbounded log.
    """

    def __init__(self, path: str, *, max_entries: int = OUTBOX_MAX_ENTRIES,
                 ttl_s: float = OUTBOX_TTL_S) -> None:
        self.path = path
        self.max_entries = max(1, int(max_entries))
        self.ttl_s = float(ttl_s)
        self._lock = threading.Lock()

    def _read_all(self) -> list[dict[str, Any]]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]
        except FileNotFoundError:
            return []
        except Exception as exc:  # noqa: BLE001 - corrupt queue: start over
            _log.warning("whatsapp outbox unreadable (%s) — starting fresh", exc)
            return []

    def _write_all(self, entries: list[dict[str, Any]]) -> None:
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                for entry in entries:
                    fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
            os.replace(tmp, self.path)
        except Exception as exc:  # noqa: BLE001 - outbox is best-effort
            _log.debug("whatsapp outbox write failed: %s", exc)

    def pending(self) -> list[dict[str, Any]]:
        """Entries not yet delivered (oldest first). Never raises."""
        with self._lock:
            return self._read_all()

    def pending_count(self) -> int:
        return len(self.pending())

    def enqueue(self, entry: dict[str, Any]) -> bool:
        """Append one entry. False when the entry was dropped (too old)."""
        with self._lock:
            entries = self._read_all()
            entries.append({**entry, "enqueued_at": time.time()})
            # Bound: drop oldest first.
            if len(entries) > self.max_entries:
                dropped = len(entries) - self.max_entries
                entries = entries[dropped:]
                _log.warning("whatsapp outbox full — dropped %d oldest", dropped)
            self._write_all(entries)
            return True

    def clear(self) -> None:
        with self._lock:
            self._write_all([])

    def flush(self, deliver: Any) -> dict[str, int]:
        """Deliver queued entries FIFO via ``deliver(entry) -> bool``.

        Expired entries are dropped; failed ones stay queued (in order).
        Returns ``{"sent": n, "dropped": n, "kept": n}``.
        """
        with self._lock:
            entries = self._read_all()
            now = time.time()
            sent, dropped, kept = 0, 0, []
            for entry in entries:
                if now - float(entry.get("enqueued_at", now)) > self.ttl_s:
                    dropped += 1
                    continue
                try:
                    ok = bool(deliver(entry))
                except Exception as exc:  # noqa: BLE001 - deliver must not break the queue
                    _log.debug("whatsapp outbox deliver failed: %s", exc)
                    ok = False
                if ok:
                    sent += 1
                else:
                    kept.append(entry)
            self._write_all(kept)
            return {"sent": sent, "dropped": dropped, "kept": len(kept)}


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
        outbox_dir: str = "data/chat/outbox",
        outbox_enabled: bool = True,
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
        # Outbound resilience: sends while the bridge is down are queued to
        # disk and flushed on reconnect, so a bridge restart doesn't eat
        # the reply the brain already composed.
        self.outbox_enabled = bool(outbox_enabled)
        self.outbox = Outbox(os.path.join(outbox_dir, "whatsapp.jsonl"))
        #: Delivery receipts from the bridge's messages.update feed:
        #: message_id -> {"status": "sent|delivered|read", "ts": float}.
        #: Bounded — only the freshest few hundred are kept.
        self._receipts: dict[str, dict[str, Any]] = {}
        self._receipts_lock = threading.Lock()

    # ── delivery receipts ────────────────────────────────────────────────
    def delivery_status(self, message_id: str) -> dict[str, Any] | None:
        """Last known receipt for a sent message, or None. Never raises."""
        try:
            with self._receipts_lock:
                row = self._receipts.get(str(message_id or ""))
                return dict(row) if row else None
        except Exception:  # noqa: BLE001
            return None

    def _note_receipt(self, message_id: str, status: str) -> None:
        message_id = str(message_id or "")
        if not message_id:
            return
        with self._receipts_lock:
            self._receipts[message_id] = {"status": status, "ts": time.time()}
            if len(self._receipts) > 500:
                # evict oldest
                oldest = sorted(self._receipts.items(),
                                key=lambda kv: kv[1].get("ts", 0.0))[:100]
                for key, _ in oldest:
                    self._receipts.pop(key, None)

    # ── outbox ───────────────────────────────────────────────────────────
    def _chat_from_entry(self, entry: dict[str, Any]) -> ChatRef:
        return ChatRef(
            platform=self.name,
            chat_id=str(entry.get("chat_id") or ""),
            kind=str(entry.get("chat_kind") or ChatKind.DM),
            thread_id=str(entry.get("thread_id") or ""),
        )

    def _enqueue(self, entry: dict[str, Any]) -> SendResult:
        """Queue one send for the reconnect flush. Returns a queued result."""
        if not self.outbox_enabled:
            return SendResult(ok=False, platform=self.name,
                              error="bridge not connected (outbox disabled)")
        self.outbox.enqueue(entry)
        _log.info("whatsapp: bridge down — queued %s send for %s",
                  entry.get("kind"), entry.get("chat_id"))
        return SendResult(
            ok=False, platform=self.name,
            message_id="queued-" + uuid.uuid4().hex[:8],
            error="bridge not connected — queued for delivery on reconnect",
        )

    def _flush_outbox_async(self) -> None:
        """Flush the outbox on a side thread — never on the reader thread."""
        def _job() -> None:
            try:
                report = self.outbox.flush(self._deliver_entry)
                if report["sent"] or report["dropped"]:
                    _log.info("whatsapp outbox flush: %s", report)
            except Exception as exc:  # noqa: BLE001
                _log.debug("whatsapp outbox flush failed: %s", exc)

        threading.Thread(target=_job, name="wa-outbox-flush", daemon=True).start()

    def _deliver_entry(self, entry: dict[str, Any]) -> bool:
        """Deliver one outbox entry. True when delivered."""
        if not self.connected.is_set():
            return False
        chat = self._chat_from_entry(entry)
        kind = str(entry.get("kind") or "text")
        try:
            if kind == "media":
                media = MediaRef(path=str(entry.get("path") or ""),
                                 kind=str(entry.get("media_kind") or "file"),
                                 name=str(entry.get("name") or ""))
                result = self.send_media(chat, media,
                                         caption=str(entry.get("caption") or ""))
            else:
                result = self._send_with_retry(
                    chat, str(entry.get("text") or ""),
                    reply_to=str(entry.get("reply_to") or ""))
            return bool(result.ok)
        except Exception as exc:  # noqa: BLE001
            _log.debug("whatsapp outbox entry failed: %s", exc)
            return False

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
                was_down = not self.connected.is_set()
                self.connected.set()
                _log.info("whatsapp bridge connected as %s", self.user_jid)
                if was_down:
                    # Bridge (re)connected — drain anything the brain sent
                    # while it was away.
                    self._flush_outbox_async()
            else:
                self.connected.clear()
                if state:
                    _log.warning("whatsapp bridge state: %s", state)
        elif kind == "receipt":
            # Delivery receipts from the bridge's messages.update feed:
            # {"type":"receipt","chat":"<jid>","ids":[...],"kind":"delivered|read"}
            status = str(obj.get("kind") or "").strip().lower()
            if status in ("delivered", "read"):
                for mid in obj.get("ids") or []:
                    self._note_receipt(str(mid or ""), status)
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
                    # @-mentions of this account in groups — the bridge
                    # computes it from contextInfo.mentionedJid, so group
                    # replies trigger exactly like Telegram's _mentions_me.
                    mentioned=bool(obj.get("mentioned")),
                    ts=ts,
                ),
            )

    # ── lifecycle ────────────────────────────────────────────────────────────
    def run(self, handler: IncomingHandler) -> None:
        """Connect, read forever, reconnect with jittered exponential backoff."""
        attempt = 0
        while not self.stopped:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=15)
                self._sock = sock
                sock.settimeout(1.0)
                reader = threading.Thread(target=self._reader_loop, args=(handler,),
                                          name=f"wa-{self.name}-reader", daemon=True)
                reader.start()
                attempt = 0  # a live connection resets the backoff ladder
                # The reader clears `connected` on exit. If it dies the
                # socket dropped and we must reconnect — watch its
                # liveness, don't just sleep until stopped.
                while not self.stopped and reader.is_alive():
                    time.sleep(0.25)
                if not self.stopped:
                    _log.warning("whatsapp bridge connection lost")
            except OSError as exc:
                self.connected.clear()
                self._sock = None
                if self.stopped:
                    return
                _log.warning("whatsapp bridge unreachable at %s:%s (%s)",
                             self.host, self.port, exc)
            finally:
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except OSError:  # noqa: E103 - socket already dead
                        pass
                    self._sock = None
            if self.stopped:
                return
            attempt += 1
            delay = _jittered_backoff(self.reconnect_delay, attempt, cap=60.0)
            _log.warning("whatsapp bridge: reconnecting in %.0fs (attempt %d)",
                         delay, attempt)
            time.sleep(delay)

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
            return self._enqueue({
                "kind": "text",
                "chat_id": chat.chat_id,
                "chat_kind": chat.kind,
                "thread_id": chat.thread_id,
                "text": text,
                "reply_to": reply_to,
            })
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
        """Send a photo/video/audio/document via the bridge's send_media.

        When the bridge is down the media is queued (path + caption) and
        flushed on reconnect — same as text."""
        started = time.perf_counter()
        path = (media.path or "").strip()
        if not path:
            return SendResult(ok=False, platform=self.name,
                              error="media has no path",
                              seconds=time.perf_counter() - started)
        if not self.connected.is_set():
            return self._enqueue({
                "kind": "media",
                "chat_id": chat.chat_id,
                "chat_kind": chat.kind,
                "thread_id": chat.thread_id,
                "path": path,
                "media_kind": (media.kind or "").lower(),
                "name": media.name or "",
                "caption": caption,
            })
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

    # ── groups & communities (bridge/whatsapp-groups.mjs — read-only) ────────
    def _group_jid(self, chat: ChatRef | str) -> str:
        """Normalize a chat ref or raw string to a group JID for group calls."""
        if isinstance(chat, ChatRef):
            jid = self._jid(chat)
        else:
            jid = str(chat or "").strip()
        return jid

    def groups(self, limit: int = 100) -> list[dict[str, Any]]:
        """Every group the account participates in.

        ``[{"id", "subject", "size", "is_community", "linked_parent"}]``,
        sorted by subject. Best-effort: [] when the bridge is down or an
        older bridge doesn't implement ``group_list``.
        """
        if not self.connected.is_set():
            return []
        response = self._send_cmd({"cmd": "group_list"}, timeout=30)
        if not response.get("ok"):
            return []
        out = []
        for entry in response.get("groups") or []:
            try:
                out.append({
                    "id": str(entry.get("id") or ""),
                    "subject": str(entry.get("subject") or ""),
                    "size": int(entry.get("size") or 0),
                    "is_community": bool(entry.get("isCommunity")),
                    "is_community_announce": bool(entry.get("isCommunityAnnounce")),
                    "linked_parent": str(entry.get("linkedParent") or ""),
                })
            except (TypeError, ValueError):  # noqa: BLE001 - skip malformed rows
                continue
        return [g for g in out if g["id"]][:max(1, int(limit))]

    def group_info(self, chat: ChatRef | str) -> dict[str, Any]:
        """Full metadata for one group: subject, description, size, admins,
        settings flags, and the invite link when the account may see it.
        {} when unknown, unreachable, or not a group.
        """
        if not self.connected.is_set():
            return {}
        response = self._send_cmd(
            {"cmd": "group_info", "chat": self._group_jid(chat)}, timeout=20)
        if not response.get("ok"):
            return {}
        group = response.get("group") or {}
        return {
            "id": str(group.get("id") or ""),
            "subject": str(group.get("subject") or ""),
            "desc": str(group.get("desc") or ""),
            "size": int(group.get("size") or 0),
            "creation": float(group.get("creation") or 0),
            "owner": str(group.get("owner") or ""),
            "is_community": bool(group.get("isCommunity")),
            "linked_parent": str(group.get("linkedParent") or ""),
            "announce": bool(group.get("announce")),
            "restrict": bool(group.get("restrict")),
            "member_add_mode": bool(group.get("memberAddMode")),
            "join_approval_mode": bool(group.get("joinApprovalMode")),
            "ephemeral_hours": int(group.get("ephemeralHours") or 0),
            "admins": [str(j) for j in (group.get("admins") or [])],
            "invite": str(group.get("invite") or ""),
        }

    def group_participants(self, chat: ChatRef | str) -> list[dict[str, Any]]:
        """The member roster: ``[{"jid", "name", "role"}]`` where role is
        ``""``/``"admin"``/``"superadmin"``. Names come from the local
        contact store and may be empty for strangers — never invented.
        [] on failure.
        """
        if not self.connected.is_set():
            return []
        response = self._send_cmd(
            {"cmd": "group_participants", "chat": self._group_jid(chat)},
            timeout=20)
        if not response.get("ok"):
            return []
        out = []
        for entry in response.get("participants") or []:
            try:
                out.append({
                    "jid": str(entry.get("jid") or ""),
                    "name": str(entry.get("name") or ""),
                    "role": str(entry.get("role") or ""),
                })
            except (TypeError, ValueError):  # noqa: BLE001 - skip malformed rows
                continue
        return [p for p in out if p["jid"]]

    def communities(self) -> list[dict[str, Any]]:
        """Communities the account participates in, each with its linked
        groups: ``[{"id", "subject", "desc", "size", "groups": [...]}]``.
        [] on failure.
        """
        if not self.connected.is_set():
            return []
        response = self._send_cmd({"cmd": "community_list"}, timeout=30)
        if not response.get("ok"):
            return []
        out = []
        for entry in response.get("communities") or []:
            try:
                out.append({
                    "id": str(entry.get("id") or ""),
                    "subject": str(entry.get("subject") or ""),
                    "desc": str(entry.get("desc") or ""),
                    "size": int(entry.get("size") or 0),
                    "groups": [
                        {"id": str(g.get("id") or ""),
                         "subject": str(g.get("subject") or ""),
                         "size": int(g.get("size") or 0)}
                        for g in (entry.get("groups") or [])
                        if g.get("id")
                    ],
                })
            except (TypeError, ValueError):  # noqa: BLE001 - skip malformed rows
                continue
        return [c for c in out if c["id"]]

    def typing(self, chat: ChatRef, seconds: float = 3.0,
               action: str = "typing") -> bool:
        """Typing indicator. ``action="recording"`` maps to WhatsApp's
        recording presence (voice-note replies); anything else is the
        regular "typing" indicator."""
        if not self.connected.is_set() or seconds <= 0:
            return False
        presence = "recording" if action == "recording" else "composing"
        response = self._send_cmd(
            {"cmd": "typing", "chat": self._jid(chat),
             "seconds": max(1, int(seconds)), "presence": presence},
            timeout=5)
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
                if self.last_qr else None,
                "outbox_pending": self.outbox.pending_count(),
                "receipts_tracked": len(self._receipts)}
