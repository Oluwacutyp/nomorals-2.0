"""WhatsApp adapter — Python client for the WhatsApp bridge.

WhatsApp has no friendly official API for a personal account (the Cloud API
is business-oriented and cannot do the userbot-style companionship), so this
system talks to the *bridge* — a per-account process that holds the
WhatsApp Web session (pairing-code/QR login, media, presence) and speaks a
JSON-lines protocol over localhost TCP. Two engines implement that one
protocol: the Go program (``bridge/bin/whatsapp-bridge-*``, whatsmeow —
the default, one self-contained file, nothing to install) and the
Node/Baileys bridge (``bridge/whatsapp-bridge.mjs``) as the second rung.

Protocol (one JSON object per line):

  bridge -> python
    {"type":"status","state":"open|closed","user":"<jid>"}
    {"type":"qr","data":"<qr string>"}
    {"type":"message","id":"<wa message id>","chat":{"id","kind","title"},
     "from":{"id","name"},"text":"...",
     "media":[{"path","mime","kind"}],"reply_to":"","mentioned":false,"ts":<ms>}
    {"type":"receipt","chat":"<jid>","ids":["<msg id>",...],"kind":"delivered|read"}
    {"id":"<req>","ok":true,"error":""}          # response to a command
    # send/send_media responses also carry "message_id": the real
    # WhatsApp message id (the receipt key space) — "id" alone is the
    # request id and identifies nothing on WhatsApp's side.

  python -> bridge
    {"id":"<req>","cmd":"send","chat":"<jid>","text":"...","reply_to":""}
    {"id":"<req>","cmd":"send_media","chat":"<jid>","path":"...","caption":"...","ptt":false}
    {"id":"<req>","cmd":"typing","chat":"<jid>","seconds":3,"presence":"composing|recording"}
    {"id":"<req>","cmd":"history","chat":"<jid>","limit":20}
    {"id":"<req>","cmd":"read","chat":"<jid>"}            # best-effort receipts
    {"id":"<req>","cmd":"chats","limit":30}               # best-effort listing
    {"id":"<req>","cmd":"status"}
    # group/community depth (bridge read-only set):
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
from .platform_limits import for_platform
from .failures import record_failure

__all__ = ["WhatsAppAdapter", "Outbox", "GROUP_CAPABILITIES"]

_log = get_logger(__name__)

#: Group/community capabilities exposed over the bridge JSON-lines protocol
#: (the bridge program's read-only group set). One source of truth: the
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
    # ── administration (owner-confirmed; denied to outsiders) ──
    ("group create", "create a group called <name>",
     "new WhatsApp group, optional initial members"),
    ("group members update", "add/remove/promote/demote <user> in <group>",
     "member management with per-user results"),
    ("group rename", "rename <group> to <name>",
     "change the group subject"),
    ("group description", "set the description of <group>",
     "change the group description"),
    ("group picture", "set <group>'s picture to <image>",
     "change the group picture"),
    ("group settings", "lock <group> / open <group> to admins only",
     "announce-only and admin-only-settings toggles"),
    ("group leave", "leave <group>",
     "exit a group"),
    ("group invite", "new invite link for <group>",
     "revoke and reissue the invite link"),
    ("community create", "create a community called <name>",
     "new WhatsApp community"),
    ("community broadcast", "announce <text> to <community>",
     "post to the community announcement group"),
    ("community link", "link <group> into <community>",
     "attach a subgroup to a community"),
    ("channel create", "create a channel called <name>",
     "new WhatsApp channel (newsletter)"),
    ("channel post", "post <text> to <channel>",
     "publish to a channel"),
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


#: Error text for a send whose command reached the bridge but whose
#: ack never came back (the flap ate it). Worded to fall in NEITHER
#: the transient nor the terminal substring table of
#: :mod:`nomorals.social.chat.send_retry`: upper layers must not file
#: it as a clean retry (that is how one reply became five copies in a
#: real group, 2026-11-04) nor as a clean failure — the truth is
#: "unknown", and the only safe handling is to not send it again.
UNCONFIRMED_SEND_ERROR = (
    "bridge ack unconfirmed after dispatch — held, not re-sent "
    "(a duplicate is worse than a held send)")


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

    def held_count(self) -> int:
        """Rows held after an unconfirmed dispatch (see :meth:`flush`)."""
        return sum(1 for e in self.pending() if e.get("held"))

    def flush(self, deliver: Any, reconcile: Any = None) -> dict[str, int]:
        """Deliver queued entries FIFO via ``deliver(entry) -> bool``.

        Send-once across reconnects: a fresh entry is stamped
        ``dispatched_at`` and the stamp is PERSISTED before
        ``deliver`` runs, so a flap or crash between the bridge send
        and its bookkeeping leaves a dispatched row — never a pristine
        row a later flush would blindly send a second time. Every
        fate change is written through immediately, not batched at
        the end.

        An entry found already stamped at flush time is NEVER
        re-delivered. ``reconcile(entry)`` gives it its one fate:
        ``"sent"`` (the bridge's own record shows the message — the
        ack was lost, not the send; row removed), ``"held"`` (no
        trace found: the row is marked held and stays in the file as
        the journal of that decision — ambiguity resolves to NOT
        re-sending, because a duplicate in a real group is worse
        than a held row), or anything else / no callback (cannot
        decide now; the row waits for a later flush). ``deliver``
        may set ``entry["_outcome_known"]`` before returning False
        to prove nothing was dispatched (no connection, write
        failed, bridge answered with a definitive refusal) — the
        stamp is then cleared and the row stays fresh.

        Expired entries are dropped; failed ones stay queued (in
        order). Returns ``{"sent", "dropped", "kept", "held"}``.
        """
        with self._lock:
            rows = self._read_all()
            now = time.time()
            sent = dropped = held = 0

            def persist() -> None:
                self._write_all(rows)

            for entry in list(rows):
                if now - float(entry.get("enqueued_at", now)) > self.ttl_s:
                    rows.remove(entry)
                    dropped += 1
                    persist()
                    continue
                if entry.get("held"):
                    held += 1
                    continue
                if entry.get("dispatched_at"):
                    fate = "unknown"
                    if reconcile is not None:
                        try:
                            fate = str(reconcile(entry))
                        except Exception as exc:  # noqa: BLE001 - reconcile must not break the queue
                            _log.debug("whatsapp outbox reconcile failed: %s", exc)
                            fate = "unknown"
                    if fate == "sent":
                        rows.remove(entry)
                        sent += 1
                        persist()
                    elif fate == "held":
                        entry["held"] = True
                        entry["held_reason"] = str(
                            entry.get("held_reason") or
                            "dispatched but never confirmed — held, not "
                            "re-sent (a duplicate is worse than a held send)")
                        held += 1
                        _log.warning(
                            "whatsapp outbox: holding %s entry for %s — %s",
                            entry.get("kind"), entry.get("chat_id"),
                            entry["held_reason"])
                        persist()
                    # "unknown": leave the row exactly as it stands; a
                    # later flush reconciles again. Never re-deliver.
                    continue
                entry["dispatched_at"] = now
                persist()
                try:
                    ok = bool(deliver(entry))
                except Exception as exc:  # noqa: BLE001 - deliver must not break the queue
                    _log.debug("whatsapp outbox deliver failed: %s", exc)
                    ok = False
                if ok:
                    rows.remove(entry)
                    sent += 1
                    persist()
                elif entry.pop("_outcome_known", False):
                    entry.pop("dispatched_at", None)
                    persist()
                else:
                    persist()
            return {"sent": sent, "dropped": dropped,
                    "kept": len(rows), "held": held}


def _split_text(text: str, *, limit: int | None = None) -> list[str]:
    """Split long text on paragraph/line boundaries so no chunk exceeds
    *limit* characters. Short text returns unchanged as a single chunk.
    Default limit is the PlatformLimits cap for WhatsApp."""
    if limit is None:
        limit = for_platform("whatsapp").max_text
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
    return [c for c in chunks if c] or [text]


def _wa_envelope(obj: dict[str, Any], text: str) -> dict[str, Any]:
    """The defense envelope (brief §1.3) from a bridge event —
    PASS-THROUGH ONLY: the bridge is an external process
    whose event schema this repo does not own, so every field is
    read defensively under its plausible names and stays absent
    when the bridge doesn't send it.  Nothing is invented."""
    mentions: list[str] = []
    raw = obj.get("mentions") or obj.get("mentioned_jids") \
        or obj.get("mentionedJids") or []
    if isinstance(raw, list):
        mentions = [str(m) for m in raw if m]
    ctx = obj.get("context") or obj.get("contextInfo") or {}
    if isinstance(ctx, dict):
        for jid in ctx.get("mentionedJid") or []:
            if str(jid) and str(jid) not in mentions:
                mentions.append(str(jid))
    entities = [{"type": "mention", "offset": 0, "length": 0,
                 "user_id": m} for m in mentions]
    media: dict[str, Any] | None = None
    sticker: dict[str, Any] | None = None
    media_list = obj.get("media") or []
    if isinstance(media_list, list) and media_list \
            and isinstance(media_list[0], dict):
        first = media_list[0]
        media = {"kind": str(first.get("kind") or ""),
                 "mime": str(first.get("mime") or ""),
                 "bytes": first.get("size") or first.get("bytes"),
                 "count": len(media_list)}
        fname = str(first.get("name") or first.get("path") or "")
        if "." in fname:
            media["ext"] = fname.rsplit(".", 1)[-1].lower()
        if media["kind"] == "sticker":
            sticker = {"animated": bool(first.get("animated")),
                       "bytes": media["bytes"],
                       "key": str(obj.get("id") or ""), "pack": ""}
    forward: dict[str, Any] | None = None
    fwd = obj.get("forward")
    if obj.get("is_forwarded") or obj.get("forwarding_score") \
            or isinstance(fwd, dict):
        fwd = fwd if isinstance(fwd, dict) else {}
        forward = {"is_forwarded": True,
                   "score": obj.get("forwarding_score")
                   or fwd.get("score"),
                   "origin_kind": str(fwd.get("origin_kind") or ""),
                   "origin_name": str(fwd.get("origin_name") or ""),
                   "sender_chat_id": str(
                       fwd.get("sender_chat_id") or ""),
                   "is_automatic": False}
    return {"entities": entities, "mentions": mentions,
            "media": media, "forward": forward,
            "quote_depth": 1 if obj.get("reply_to") else 0,
            "quoted_chars": len(str(obj.get("quoted_text") or "")),
            "sticker": sticker, "raw_len": len(text)}


def _jid_user_part(jid: str) -> str:
    """The user part of a JID: ``2348012345678:12@s.whatsapp.net``
    → ``2348012345678``. Device parts and domains are addressing
    detail; identity comparisons key off this."""
    local = (jid or "").split("@", 1)[0]
    return local.split(":", 1)[0].strip()


class WhatsAppAdapter(ChatAdapter):
    """Client for the local WhatsApp bridge (bridge/bin/ Go program
    by default, or the Node bridge/whatsapp-bridge.mjs)."""

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
        # last pairing code (bridge in WA_PAIRING_CODE mode), so the
        # /pair flow can surface it like the QR (§15)
        self.last_pairing_code: str = ""
        self.last_pairing_code_at: float = 0.0
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
        #: Message ids THIS adapter sent (bridge-assigned, from send
        #: responses): the bridge echoes Devon's own sends back as
        #: own_message events, and the self-chat carve-out below must
        #: tell her echoes apart from the owner typing on his phone.
        #: Bounded by count and age — an id is only meaningful while
        #: its echo could still plausibly arrive.
        self._sent_ids: dict[str, float] = {}
        self._sent_ids_lock = threading.Lock()
        #: Sends dispatched to the bridge whose ack was lost (a flap
        #: between the bridge send and its response): (chat jid, text)
        #: → dispatch time. A repeat of the same send inside the
        #: window is answered from the bridge's own history, or held
        #: unconfirmed — never dispatched a second time, no matter
        #: how many retry layers above ask again. Bounded like
        #: ``_sent_ids`` (same window: an ack older than that is not
        #: plausibly still in flight).
        self._unconfirmed: dict[tuple[str, str], float] = {}
        self._unconfirmed_lock = threading.Lock()

    # ── delivery receipts ────────────────────────────────────────────────
    def delivery_status(self, message_id: str) -> dict[str, Any] | None:
        """Last known receipt for a sent message, or None. Never raises."""
        try:
            with self._receipts_lock:
                row = self._receipts.get(str(message_id or ""))
                return dict(row) if row else None
        except Exception as exc:  # noqa: BLE001
            record_failure('whatsapp.WhatsAppAdapter.delivery_status', exc)
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

    # ── own-send echoes + the self-chat ─────────────────────────────
    def _note_sent_id(self, message_id: str) -> None:
        """Remember a bridge-assigned id this adapter just sent."""
        message_id = str(message_id or "")
        if not message_id:
            return
        now = time.time()
        with self._sent_ids_lock:
            self._sent_ids[message_id] = now
            if len(self._sent_ids) > 512:
                stale = [k for k, ts in self._sent_ids.items()
                         if now - ts > 600.0]
                for key in stale:
                    self._sent_ids.pop(key, None)
                while len(self._sent_ids) > 512:
                    oldest = min(self._sent_ids,
                                 key=lambda k: self._sent_ids[k])
                    self._sent_ids.pop(oldest, None)

    def _was_sent_by_me(self, message_id: str) -> bool:
        """True when *message_id* is one of this adapter's own sends
        echoing back through the bridge's own_message feed."""
        message_id = str(message_id or "")
        if not message_id:
            return False
        with self._sent_ids_lock:
            ts = self._sent_ids.get(message_id)
        return ts is not None and (time.time() - ts) <= 600.0

    # ── unconfirmed dispatches (send-once across reconnects) ───────
    #: How long a dispatch with a lost ack stays guarded. Matches the
    #: ``_sent_ids`` window — the bridge's per-chat history reaches
    #: back far beyond it in any chat quiet enough for a lost ack to
    #: matter.
    _UNCONFIRMED_WINDOW_S = 600.0

    def _note_unconfirmed(self, key: tuple[str, str]) -> None:
        now = time.time()
        with self._unconfirmed_lock:
            self._unconfirmed[key] = now
            if len(self._unconfirmed) > 512:
                stale = [k for k, ts in self._unconfirmed.items()
                         if now - ts > self._UNCONFIRMED_WINDOW_S]
                for k in stale:
                    self._unconfirmed.pop(k, None)
                while len(self._unconfirmed) > 512:
                    oldest = min(self._unconfirmed,
                                 key=lambda k: self._unconfirmed[k])
                    self._unconfirmed.pop(oldest, None)

    def _unconfirmed_at(self, key: tuple[str, str]) -> float | None:
        """Dispatch time of an unconfirmed send, or None when the key
        is not (or no longer) guarded."""
        with self._unconfirmed_lock:
            ts = self._unconfirmed.get(key)
        if ts is None or (time.time() - ts) > self._UNCONFIRMED_WINDOW_S:
            return None
        return ts

    def _clear_unconfirmed(self, key: tuple[str, str]) -> None:
        with self._unconfirmed_lock:
            self._unconfirmed.pop(key, None)

    def _history_find_own(self, chat: ChatRef, text: str, *,
                          since: float) -> bool:
        """Ask the bridge's own message store whether a text this
        account sent is already in *chat* — the reconciliation query
        for a dispatch whose ack was lost. The bridge remembers its
        sends independently of this adapter's connection state, so
        the answer survives a reconnect. Best-effort: any failure
        reads as "not found" (holding is the safe direction)."""
        if not self.connected.is_set():
            return False
        try:
            response = self._send_cmd(
                {"cmd": "history", "chat": self._jid(chat), "limit": 50},
                timeout=15)
        except Exception as exc:  # noqa: BLE001 - a failed query finds nothing
            _log.debug("whatsapp history reconcile query failed: %s", exc)
            return False
        if not response.get("ok"):
            return False
        floor_ms = (float(since) - 5.0) * 1000.0
        for entry in response.get("messages") or []:
            if entry.get("incoming"):
                continue
            if str(entry.get("text") or "") != text:
                continue
            try:
                ts_ms = float(entry.get("ts") or 0)
            except (TypeError, ValueError):
                ts_ms = 0.0
            if ts_ms and ts_ms < floor_ms:
                continue
            return True
        return False

    def _reconcile_entry(self, entry: dict[str, Any]) -> str:
        """The one fate for an outbox row dispatched by an earlier
        flush whose outcome was never recorded (see Outbox.flush):

        * ``"sent"`` — the bridge's history shows the message: the
          ack was lost, not the send.
        * ``"held"`` — no trace: the row is NOT re-sent. It may have
          landed; a duplicate in a real group is worse than a held
          row, so ambiguity resolves to holding.
        * ``"unknown"`` — cannot ask right now (bridge down); a
          later flush asks again.

        Media rows resolve to ``"held"`` directly: history cannot
        match them (the bridge's media paths are its own downloads,
        never our source paths), so no honest query exists.
        """
        if not self.connected.is_set():
            return "unknown"
        if str(entry.get("kind") or "text") == "media":
            return "held"
        chat = self._chat_from_entry(entry)
        text = str(entry.get("_dispatched_text") or entry.get("text") or "")
        if not text:
            return "held"
        since = float(entry.get("dispatched_at") or 0.0)
        return ("sent" if self._history_find_own(chat, text, since=since)
                else "held")

    def _is_self_chat(self, chat_id: str, kind: str) -> bool:
        """True when *chat_id* is this account's own "Message
        yourself" chat: a DM whose JID is the account's own JID
        (the ``user`` the bridge reports in its status events).
        Groups and broadcasts are never the self-chat, and while
        the account's own JID is unknown (no status event yet)
        nothing is — the persist-only default is the safe side."""
        if not chat_id or not self.user_jid:
            return False
        if (kind or "").strip().lower() == str(ChatKind.GROUP):
            return False
        if chat_id.endswith("@g.us") or chat_id == "status@broadcast":
            return False
        mine = _jid_user_part(self.user_jid)
        return bool(mine) and _jid_user_part(chat_id) == mine

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
                report = self.outbox.flush(
                    self._deliver_entry, reconcile=self._reconcile_entry)
                if report["sent"] or report["dropped"] or report["held"]:
                    _log.info("whatsapp outbox flush: %s", report)
            except Exception as exc:  # noqa: BLE001
                _log.debug("whatsapp outbox flush failed: %s", exc)

        threading.Thread(target=_job, name="wa-outbox-flush", daemon=True).start()

    def _deliver_entry(self, entry: dict[str, Any]) -> bool:
        """Deliver one outbox entry. True when delivered.

        The flush stamps the entry dispatched before this runs (see
        Outbox.flush). When the outcome is KNOWN — nothing was
        dispatched, or the bridge definitively answered — this sets
        ``entry["_outcome_known"]`` so the flush clears the stamp and
        the entry stays an ordinary queued row. When a dispatch's
        outcome is unknown (its ack was lost), the marker stays
        unset: the stamp survives and the next flush reconciles the
        entry instead of re-sending it.
        """
        if not self.connected.is_set():
            entry["_outcome_known"] = True  # nothing was dispatched
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
                if result.ok or result.error != UNCONFIRMED_SEND_ERROR:
                    entry["_outcome_known"] = True
                return bool(result.ok)
            # Text entries render EXACTLY like the live path in send():
            # WhatsApp-markdown conversion + platform split. The flush
            # used to hand the stored text to the bridge raw, so a reply
            # queued during an outage went out with Telegram HTML still
            # in it (and unsplit past the platform cap).
            text = str(entry.get("text") or "")
            try:
                from .platforms import format_for_platform
                text = format_for_platform(text, "whatsapp")
            except Exception as exc:  # noqa: BLE001 - same best-effort as send()
                record_failure('whatsapp.WhatsAppAdapter._deliver_entry', exc)
            reply_to = str(entry.get("reply_to") or "")
            entry_buttons = entry.get("buttons") or None
            chunks = _split_text(text)
            if chunks:
                # The first chunk is what a later reconciliation looks
                # for in the bridge's history (formatted, as sent).
                entry["_dispatched_text"] = chunks[0]
            for i, chunk in enumerate(chunks):
                result, outcome_known = self._send_text_outcome(
                    chat, chunk, reply_to=reply_to if i == 0 else "",
                    buttons=entry_buttons if i == 0 else None)
                if not result.ok:
                    if outcome_known:
                        entry["_outcome_known"] = True
                    return False
            return True
        except Exception as exc:  # noqa: BLE001
            _log.debug("whatsapp outbox entry failed: %s", exc)
            return False

    # ── helpers ──────────────────────────────────────────────────────────────
    def _jid(self, chat: ChatRef) -> str:
        """Normalize a chat to a WhatsApp JID."""
        jid = (chat.chat_id or "").strip()
        if chat.kind == ChatKind.DM and jid and "@" not in jid:
            # A bare phone number (no domain at all) → personal chat
            # JID. Anything already carrying a domain is a complete JID
            # and passes through untouched — the old suffix whitelist
            # rewrote unknown domains (``@lid`` privacy ids, future
            # Baileys domains) into ``digits@c.us``, a *different*
            # address that silently misroutes the send.
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
                # The command line reached the bridge; only its
                # response is missing. For a send that distinction is
                # life-or-death (the message may already be live), so
                # the timeout return is marked ambiguous — send paths
                # read the mark, everything else just sees a timeout.
                return {"id": req_id, "ok": False,
                        "error": "bridge response timeout",
                        "ambiguous": True}
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
                try:
                    chunk = sock.recv(65536)
                except TimeoutError:
                    # run() sets a 1.0s socket timeout so this loop can
                    # re-check `stopped` — a timeout (socket.timeout,
                    # the same class) means the bridge was simply quiet
                    # for a tick. It is NOT a disconnect: keep reading,
                    # and never let it reach the exit path below (which
                    # clears `connected` and makes run() reconnect —
                    # an idle bridge flapped once a second that way).
                    continue
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
        elif kind == "pairing_code":
            self.last_pairing_code = str(obj.get("data") or "")
            self.last_pairing_code_at = time.time()
            self.connected.clear()
            _log.warning(
                "whatsapp pairing code received — type it into "
                "WhatsApp (Linked Devices → Link with phone number); "
                "also available via health()")
        elif kind == "own_message":
            # The owner's own outgoing message (shared-account mode).
            # Persist-only: the brain records it (double-reply discipline,
            # per-chat profile) and stays silent — never replies to self.
            #
            # ONE carve-out: the account's own self-chat ("Message
            # yourself"). A message the owner types there is written
            # in a chat with nobody else in it — it can only be FOR
            # Devon — so it is delivered as an owner DM turn
            # (incoming, from_owner, self_chat) and commands and
            # conversation route exactly like the Telegram owner DM.
            # The bridge echoes Devon's own sends through this same
            # event, so ids she sent are checked first: her reply's
            # echo stays persist-only, or she would answer herself
            # forever. Own messages in every other chat — DMs with
            # other people, groups — keep the persist-only discipline
            # untouched: she never replies as the owner mid-chat.
            chat = obj.get("chat") or {}
            chat_id = str(chat.get("id") or "")
            if not chat_id:
                return
            ts = obj.get("ts") or 0
            try:
                ts = float(ts) / 1000.0
            except (TypeError, ValueError):
                ts = time.time()
            chat_kind = str(chat.get("kind") or ChatKind.DM)
            message_id = str(obj.get("id") or "")
            if not self._was_sent_by_me(message_id) and \
                    self._is_self_chat(chat_id, chat_kind):
                turn = ChatMessage(
                    chat=ChatRef(
                        platform=self.name,
                        chat_id=chat_id,
                        kind=chat_kind,
                        title=str(chat.get("title") or ""),
                        peer=chat_id,
                    ),
                    incoming=True,
                    text=str(obj.get("text") or ""),
                    sender=str(obj.get("name") or ""),
                    sender_id=self.user_jid,
                    message_id=message_id,
                    ts=ts,
                )
                turn.meta["from_owner"] = True
                turn.meta["self_chat"] = True
                self._deliver(handler, turn)
                return
            msg = ChatMessage(
                chat=ChatRef(
                    platform=self.name,
                    chat_id=chat_id,
                    kind=chat_kind,
                    title=str(chat.get("title") or ""),
                    peer=chat_id,
                ),
                incoming=False,
                text=str(obj.get("text") or ""),
                sender="",
                message_id=message_id,
                ts=ts,
            )
            msg.meta["own_message"] = True
            # The sender IS the owner — flag it for group recognition.
            msg.meta["from_owner"] = True
            self._deliver(handler, msg)
        elif kind == "status_post":
            # Personal surface feed (§15): a status post this
            # account can see, media already downloaded by the
            # bridge.  Delivered as a service message — the runtime's
            # personal-surface hook saves it (per-account toggle) and
            # consumes it; it is never conversation, never a reply
            # target.  An older bridge never sends these, and then
            # the status saver simply has no feed — nothing here
            # invents one.
            frm = obj.get("from") or {}
            poster_id = str(frm.get("id") or "")
            if not poster_id:
                return
            ts = obj.get("ts") or 0
            try:
                ts = float(ts) / 1000.0
            except (TypeError, ValueError):
                ts = time.time()
            msg = ChatMessage(
                chat=ChatRef(
                    platform=self.name,
                    chat_id="status@broadcast",
                    kind=ChatKind.DM,
                    title="status",
                    peer=poster_id,
                ),
                incoming=True,
                text=str(obj.get("text") or ""),
                sender=str(frm.get("name") or ""),
                sender_id=poster_id,
                message_id=str(obj.get("id") or ""),
                media=[
                    MediaRef(path=str(m.get("path") or ""),
                             mime=str(m.get("mime") or ""),
                             kind=str(m.get("kind") or "file"),
                             is_voice=str(m.get("kind") or "") == "voice")
                    for m in (obj.get("media") or [])
                    if m.get("path")
                ],
                ts=ts,
            )
            msg.meta["service"] = "status_post"
            if obj.get("view_once"):
                msg.meta["view_once"] = True
            self._deliver(handler, msg)
        elif kind in ("revoke", "edit"):
            # Evidentiary feeds (defense brief §2.1), pass-through:
            # recognized IF the bridge sends them (Baileys emits
            # protocol REVOKE / edited-message updates).  An older
            # bridge that never sends them simply never feeds the
            # sensors — nothing here invents an event.
            chat = obj.get("chat") or {}
            chat_id = str(chat.get("id") or obj.get("chat_id")
                          or "")
            mid = str(obj.get("id") or obj.get("message_id")
                      or "")
            if not chat_id or not mid:
                return
            ts = obj.get("ts") or 0
            try:
                ts = float(ts) / 1000.0
            except (TypeError, ValueError):
                ts = time.time()
            ref = ChatRef(
                platform=self.name, chat_id=chat_id,
                kind=str(chat.get("kind") or ChatKind.GROUP),
                title=str(chat.get("title") or ""), peer=chat_id)
            if kind == "revoke":
                self._deliver(handler, ChatMessage(
                    chat=ref, incoming=True, text="", sender="",
                    sender_id=str((obj.get("from") or {}).get("id")
                                  or ""),
                    message_id=mid, ts=ts,
                    meta={"service": "delete",
                          "deleted_ids": [mid],
                          "by_user_id": str(
                              (obj.get("from") or {}).get("id")
                              or "")}))
            else:
                msg = ChatMessage(
                    chat=ref, incoming=True,
                    text=str(obj.get("text") or ""),
                    sender=str((obj.get("from") or {}).get("name")
                             or ""),
                    sender_id=str((obj.get("from") or {}).get("id")
                                  or ""),
                    message_id=mid, ts=ts)
                msg.meta["edited"] = True
                msg.meta["envelope"] = _wa_envelope(
                    obj, msg.text)
                self._deliver(handler, msg)
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
            wa_msg = ChatMessage(
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
                    # The bridge sends the author's JID as from.id —
                    # keep it as the stable sender_id (group identity
                    # must not key off the mutable pushName alone), and
                    # the WhatsApp message id as message_id (the
                    # gateway/reply machinery keys off it).
                    sender_id=str((obj.get("from") or {}).get("id") or ""),
                    message_id=str(obj.get("id") or ""),
                    media=[
                        MediaRef(path=str(m.get("path") or ""), mime=str(m.get("mime") or ""),
                                 kind=str(m.get("kind") or "file"),
                                 is_voice=str(m.get("kind") or "") == "voice")
                        for m in (obj.get("media") or [])
                        if m.get("path")
                    ],
                    reply_to=str(obj.get("reply_to") or ""),
                    # @-mentions of this account in groups — the bridge
                    # computes it from contextInfo.mentionedJid, so group
                    # replies trigger exactly like Telegram's _mentions_me.
                    mentioned=bool(obj.get("mentioned")),
                    ts=ts,
                )
            wa_msg.meta["envelope"] = _wa_envelope(obj, wa_msg.text)
            if obj.get("view_once"):
                # The bridge unwrapped a view-once payload (§15):
                # the personal surface preserves the downloaded
                # media before WhatsApp vanishes it.
                wa_msg.meta["view_once"] = True
            self._deliver(handler, wa_msg)

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
        fails fast with a clear message when the bridge program isn't
        running, instead of letting ``run()`` spin on reconnect backoff
        silently.
        """
        try:
            sock = socket.create_connection((self.host, self.port), timeout=5)
            sock.close()
        except OSError as exc:
            raise RuntimeError(
                f"whatsapp bridge not reachable at {self.host}:{self.port} "
                f"({exc}); start a bridge first — Devon's pairing "
                "supervisor launches one per paired account (the Go "
                "program under bridge/bin/ by default, or the Node "
                "bridge/whatsapp-bridge.mjs)") from exc

    # ── outbound ─────────────────────────────────────────────────────────────
    def _send_text_outcome(self, chat: ChatRef, text: str,
                           *, reply_to: str = "",
                           buttons: Any = None) -> tuple[SendResult, bool]:
        """One text send; returns ``(result, outcome_known)``.

        ``outcome_known`` is False ONLY when a dispatch's ack was
        lost — the message may already be live and nobody may send
        it again. Every other outcome (delivered, definitively
        refused, never dispatched) is known.

        Send-once guard: a send identical to one whose ack was lost
        inside the window is never dispatched again. It is first
        reconciled against the bridge's own history — visible there
        means the original landed (report success; the bridge's
        history carries no message id, so the result's id is empty
        rather than invented). Not visible means it stays
        unconfirmed and held.
        """
        started = time.perf_counter()
        jid = self._jid(chat)
        key = (jid, text)
        prior = self._unconfirmed_at(key)
        if prior is not None:
            if self._history_find_own(chat, text, since=prior):
                self._clear_unconfirmed(key)
                self.stats["sent"] += 1
                _log.info("whatsapp: unconfirmed send to %s found in "
                          "bridge history — counting it sent, not "
                          "re-sending", jid)
                return SendResult(ok=True, platform=self.name,
                                  message_id="",
                                  seconds=time.perf_counter() - started), True
            return SendResult(ok=False, platform=self.name,
                              error=UNCONFIRMED_SEND_ERROR,
                              seconds=time.perf_counter() - started), False
        last_error = "bridge rejected send"
        request: dict[str, Any] = {"cmd": "send", "chat": jid,
                                   "text": text, "reply_to": reply_to}
        if buttons:
            # The bridge renders what WhatsApp supports natively
            # and degrades the rest to text action lines — the
            # keyboard is never silently dropped (presentation
            # contract, same as every other platform).
            request["buttons"] = [
                [[str(label), str(data)] for label, data in row]
                for row in buttons if row]
        for attempt in (1, 2):
            response = self._send_cmd(request)
            if response.get("ok"):
                self.stats["sent"] += 1
                # The bridge's reply id is OUR request id — the real
                # WhatsApp message id rides in ``message_id`` (it is
                # the key space delivery receipts use; the request id
                # matched nothing and made receipts unresolvable).
                self._note_sent_id(str(response.get("message_id") or ""))
                return SendResult(ok=True, platform=self.name,
                                  message_id=str(response.get("message_id") or ""),
                                  seconds=time.perf_counter() - started), True
            if response.get("ambiguous"):
                # The command reached the bridge; only its ack is
                # missing. The old code retried here on the "timeout"
                # substring — that retry is exactly how one reply
                # became five copies in a real group (2026-11-04).
                # An unconfirmed dispatch is never re-dispatched:
                # guard the key and report the outcome as unknown.
                self._note_unconfirmed(key)
                self.stats["send_errors"] += 1
                return SendResult(ok=False, platform=self.name,
                                  error=UNCONFIRMED_SEND_ERROR,
                                  seconds=time.perf_counter() - started), False
            last_error = str(response.get("error") or last_error)
            # only retry transient failures, not rejections
            if "timeout" not in last_error.lower() and \
               "not connected" not in last_error.lower():
                break
            if attempt == 1:
                time.sleep(1.0)
        self.stats["send_errors"] += 1
        return SendResult(ok=False, platform=self.name, error=last_error,
                          seconds=time.perf_counter() - started), True

    def _send_with_retry(self, chat: ChatRef, text: str,
                         *, reply_to: str = "",
                         buttons: Any = None) -> SendResult:
        """One text send with a single retry on transient bridge failure."""
        result, _outcome_known = self._send_text_outcome(
            chat, text, reply_to=reply_to, buttons=buttons)
        return result

    def send(self, chat: ChatRef, text: str, *,
             reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             reply_keyboard: list[list[str]] | None = None,
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
                "buttons": [[[str(label), str(data)]
                             for label, data in row]
                            for row in buttons if row]
                           if buttons else None,
            })
        # Convert Telegram HTML / canonical markdown to WhatsApp markdown
        # (game output and styled menus are authored Telegram-first).
        try:
            from .platforms import format_for_platform
            text = format_for_platform(text, "whatsapp")
        except Exception as exc:  # noqa: BLE001
            record_failure('whatsapp.WhatsAppAdapter.send', exc)
        # WhatsApp caps a text message well above this; split long sends so
        # one giant alert doesn't get silently truncated by the bridge.
        chunks = _split_text(text)
        result: SendResult | None = None
        for i, chunk in enumerate(chunks):
            result = self._send_with_retry(
                chat, chunk,
                reply_to=reply_to if i == 0 else "",
                buttons=buttons if i == 0 else None)
            if not result.ok:
                return result
        assert result is not None
        return result

    def send_media(self, chat: ChatRef, media: MediaRef, *,
                   caption: str = "", parse_mode: str = "") -> SendResult:
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
        # Same send-once guard as text (see _send_text_outcome): a
        # media dispatch whose ack was lost is never re-dispatched.
        # History cannot reconcile media, so a guarded repeat is
        # simply held unconfirmed.
        media_key = (self._jid(chat),
                     "\x00media\x00" + path + "\x00" + caption)
        if self._unconfirmed_at(media_key) is not None:
            return SendResult(ok=False, platform=self.name,
                              error=UNCONFIRMED_SEND_ERROR,
                              seconds=time.perf_counter() - started)
        cmd: dict[str, Any] = {
            "cmd": "send_media",
            "chat": self._jid(chat),
            "path": path,
            "caption": caption,
            # The kind rides along (the offline enqueue payload has
            # always carried it): a sticker WebP or an animation
            # should reach the bridge AS what it is (§21). Whether
            # the bridge renders kinds beyond voice is the bridge's
            # side of the contract — reported, not assumed.
            "media_kind": (media.kind or "").lower(),
        }
        kind = (media.kind or "").lower()
        if kind in ("voice", "ptt"):
            cmd["ptt"] = True  # voice-note bubble instead of an audio file
        response = self._send_cmd(cmd, timeout=60)
        if response.get("ok"):
            self._clear_unconfirmed(media_key)
            self.stats["sent"] += 1
            self._note_sent_id(str(response.get("message_id") or ""))
            return SendResult(ok=True, platform=self.name,
                              message_id=str(response.get("message_id") or ""),
                              seconds=time.perf_counter() - started)
        self.stats["send_errors"] += 1
        if response.get("ambiguous"):
            # Dispatched, ack lost — never re-dispatched (see
            # _send_text_outcome for the incident this prevents).
            self._note_unconfirmed(media_key)
            return SendResult(ok=False, platform=self.name,
                              error=UNCONFIRMED_SEND_ERROR,
                              seconds=time.perf_counter() - started)
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

    # ── groups & communities (bridge read-only set) ────────
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

    # ── group administration (bridge admin set) ──────────
    # Mutating ops on the owner's personal account. WhatsApp may flag
    # aggressive automation — every one of these is owner-confirmed on the
    # spine-tool side (capability-gated, denied to outsiders by the
    # public/private matrix). The bridge answers ok:false with the real
    # Baileys reason when WhatsApp refuses; nothing is faked.

    def _admin_cmd(self, cmd: str, payload: dict[str, Any],
                   *, timeout: float = 30.0) -> dict[str, Any]:
        """Send a mutating group/community/channel command to the bridge.

        Returns ``{"ok": True, ...}`` or ``{"ok": False, "error": ...}``
        with the bridge's honest reason. ``{}`` when disconnected.
        """
        if not self.connected.is_set():
            return {"ok": False, "error": "whatsapp bridge not connected"}
        payload = dict(payload)
        payload["cmd"] = cmd
        try:
            return self._send_cmd(payload, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - honest error, not a crash
            return {"ok": False, "error": str(exc)[:200]}

    def group_create(self, subject: str,
                     participants: list[str] | None = None) -> dict[str, Any]:
        """Create a group. ``participants`` are user JIDs (…@s.whatsapp.net)."""
        subject = (subject or "").strip()
        if not subject:
            return {"ok": False, "error": "missing subject for the new group"}
        return self._admin_cmd("group_create", {
            "subject": subject,
            "participants": [str(p) for p in (participants or []) if p],
        }, timeout=45)

    def group_members_update(self, chat: ChatRef | str, action: str,
                             participants: list[str]) -> dict[str, Any]:
        """Add/remove/promote/demote members. ``action`` ∈ add|remove|promote|demote."""
        action = (action or "").strip().lower()
        if action not in ("add", "remove", "promote", "demote"):
            return {"ok": False,
                    "error": f"bad action {action!r} (want add|remove|promote|demote)"}
        parts = [str(p) for p in (participants or []) if p]
        if not parts:
            return {"ok": False, "error": "no participants given"}
        return self._admin_cmd("group_members_update", {
            "chat": self._group_jid(chat), "action": action,
            "participants": parts,
        }, timeout=45)

    def group_set_subject(self, chat: ChatRef | str, subject: str) -> dict[str, Any]:
        """Rename a group."""
        return self._admin_cmd("group_set_subject", {
            "chat": self._group_jid(chat), "subject": (subject or "").strip(),
        })

    def group_set_description(self, chat: ChatRef | str,
                              description: str) -> dict[str, Any]:
        """Set a group's description."""
        return self._admin_cmd("group_set_description", {
            "chat": self._group_jid(chat), "description": description or "",
        })

    def group_set_picture(self, chat: ChatRef | str, path: str) -> dict[str, Any]:
        """Set a group's picture from a local image file."""
        return self._admin_cmd("group_set_picture", {
            "chat": self._group_jid(chat), "path": (path or "").strip(),
        }, timeout=45)

    def group_leave(self, chat: ChatRef | str) -> dict[str, Any]:
        """Leave a group."""
        return self._admin_cmd("group_leave", {"chat": self._group_jid(chat)})

    def group_set_settings(self, chat: ChatRef | str, *,
                           announce: bool | None = None,
                           restrict: bool | None = None) -> dict[str, Any]:
        """Group settings: ``announce`` (only admins send), ``restrict``
        (only admins change settings). Pass only what should change."""
        payload: dict[str, Any] = {"chat": self._group_jid(chat)}
        if announce is not None:
            payload["announce"] = bool(announce)
        if restrict is not None:
            payload["restrict"] = bool(restrict)
        if len(payload) == 1:
            return {"ok": False, "error": "nothing to change: pass announce= and/or restrict="}
        return self._admin_cmd("group_set_settings", payload)

    def group_revoke_invite(self, chat: ChatRef | str) -> dict[str, Any]:
        """Revoke the invite link and get a fresh one. Returns ``invite``."""
        return self._admin_cmd("group_revoke_invite",
                               {"chat": self._group_jid(chat)})

    def group_join_approval(self, chat: ChatRef | str, mode: str) -> dict[str, Any]:
        """Require admin approval for new members: ``mode`` ∈ on|off."""
        return self._admin_cmd("group_join_approval", {
            "chat": self._group_jid(chat), "mode": (mode or "").strip().lower(),
        })

    def group_member_add_mode(self, chat: ChatRef | str, mode: str) -> dict[str, Any]:
        """Who may add members: ``mode`` ∈ admin|all."""
        return self._admin_cmd("group_member_add_mode", {
            "chat": self._group_jid(chat), "mode": (mode or "").strip().lower(),
        })

    # ── communities ─────────────────────────────────────────────────────

    def community_create(self, subject: str,
                         description: str = "") -> dict[str, Any]:
        """Create a community."""
        subject = (subject or "").strip()
        if not subject:
            return {"ok": False, "error": "missing subject for the new community"}
        return self._admin_cmd("community_create", {
            "subject": subject, "description": description or "",
        }, timeout=45)

    def community_leave(self, chat: ChatRef | str) -> dict[str, Any]:
        """Leave a community."""
        return self._admin_cmd("community_leave",
                               {"chat": self._group_jid(chat)})

    def community_members_update(self, chat: ChatRef | str, action: str,
                                 participants: list[str]) -> dict[str, Any]:
        """Add/remove/promote/demote community members."""
        action = (action or "").strip().lower()
        if action not in ("add", "remove", "promote", "demote"):
            return {"ok": False,
                    "error": f"bad action {action!r} (want add|remove|promote|demote)"}
        parts = [str(p) for p in (participants or []) if p]
        if not parts:
            return {"ok": False, "error": "no participants given"}
        return self._admin_cmd("community_members_update", {
            "chat": self._group_jid(chat), "action": action,
            "participants": parts,
        }, timeout=45)

    def community_set_subject(self, chat: ChatRef | str,
                              subject: str) -> dict[str, Any]:
        """Rename a community."""
        return self._admin_cmd("community_set_subject", {
            "chat": self._group_jid(chat), "subject": (subject or "").strip(),
        })

    def community_set_description(self, chat: ChatRef | str,
                                  description: str) -> dict[str, Any]:
        """Set a community's description."""
        return self._admin_cmd("community_set_description", {
            "chat": self._group_jid(chat), "description": description or "",
        })

    def community_link_group(self, community: ChatRef | str,
                             group: ChatRef | str) -> dict[str, Any]:
        """Link a subgroup into a community."""
        return self._admin_cmd("community_link_group", {
            "community": self._group_jid(community),
            "group": self._group_jid(group),
        }, timeout=45)

    def community_unlink_group(self, community: ChatRef | str,
                               group: ChatRef | str) -> dict[str, Any]:
        """Unlink a subgroup from a community."""
        return self._admin_cmd("community_unlink_group", {
            "community": self._group_jid(community),
            "group": self._group_jid(group),
        }, timeout=45)

    def community_broadcast(self, community: ChatRef | str,
                            text: str) -> dict[str, Any]:
        """Post an announcement to a community's announcement group.

        Finds the community's announce group and sends there — that is how
        WhatsApp broadcasts to every member.
        """
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "empty announcement"}
        groups = self.groups(limit=500)
        want = self._group_jid(community)
        announce_jid = ""
        for g in groups:
            if g["id"] == want and g["is_community_announce"]:
                announce_jid = g["id"]
                break
        if not announce_jid:
            for g in groups:
                if g.get("linked_parent") == want and g["is_community_announce"]:
                    announce_jid = g["id"]
                    break
        if not announce_jid:
            return {"ok": False,
                    "error": "no announcement group found for that community"}
        chat = ChatRef(platform="whatsapp", chat_id=announce_jid,
                        kind=ChatKind.GROUP, title="")
        result = self.send(chat, text)
        return {"ok": result.ok, "message_id": result.message_id,
                "error": result.error or ""}

    # ── channels (WhatsApp newsletters) ─────────────────────────────────

    def channel_create(self, name: str,
                       description: str = "") -> dict[str, Any]:
        """Create a WhatsApp channel (newsletter)."""
        name = (name or "").strip()
        if not name:
            return {"ok": False, "error": "missing name for the new channel"}
        return self._admin_cmd("channel_create", {
            "name": name, "description": description or "",
        }, timeout=45)

    def channel_delete(self, chat: ChatRef | str) -> dict[str, Any]:
        """Delete a channel the account owns."""
        return self._admin_cmd("channel_delete",
                               {"chat": self._newsletter_jid(chat)})

    def channel_follow(self, chat: ChatRef | str) -> dict[str, Any]:
        """Follow a channel."""
        return self._admin_cmd("channel_follow",
                               {"chat": self._newsletter_jid(chat)})

    def channel_unfollow(self, chat: ChatRef | str) -> dict[str, Any]:
        """Unfollow a channel."""
        return self._admin_cmd("channel_unfollow",
                               {"chat": self._newsletter_jid(chat)})

    def channel_resolve_invite(self, code: str) -> dict[str, Any]:
        """Resolve a channel invite code (the tail of a
        whatsapp.com/channel/<code> link) to the channel's JID
        via the bridge's ``channel_resolve`` (Baileys
        newsletterMetadata 'invite').  Returns ``{"ok": True,
        "chat": "<jid>", "name": ...}`` or the bridge's honest
        error — a link that resolves nowhere stays unresolved."""
        code = (code or "").strip()
        if not code:
            return {"ok": False, "error": "missing channel invite code"}
        return self._admin_cmd("channel_resolve", {"invite": code})

    def channel_react(self, chat: ChatRef | str, server_id: str,
                      emoji: str) -> dict[str, Any]:
        """React to one channel post (§14 mass react): the
        bridge's ``channel_react`` → Baileys
        newsletterReactMessage(jid, serverId, emoji).  The
        server_id is the number at the end of the post link.
        A bridge whose Baileys build lacks the verb answers
        ok:false with that reason — never a fake success."""
        server_id = str(server_id or "").strip()
        if not server_id:
            return {"ok": False,
                    "error": "missing server_id (the number at "
                             "the end of the channel post link)"}
        return self._admin_cmd("channel_react", {
            "chat": self._newsletter_jid(chat),
            "server_id": server_id, "emoji": str(emoji or ""),
        })

    def channel_set_name(self, chat: ChatRef | str, name: str) -> dict[str, Any]:
        """Rename a channel the account owns."""
        return self._admin_cmd("channel_set_name", {
            "chat": self._newsletter_jid(chat), "name": (name or "").strip(),
        })

    def channel_set_description(self, chat: ChatRef | str,
                                description: str) -> dict[str, Any]:
        """Set a channel's description."""
        return self._admin_cmd("channel_set_description", {
            "chat": self._newsletter_jid(chat),
            "description": description or "",
        })

    def channel_post(self, chat: ChatRef | str, text: str) -> dict[str, Any]:
        """Post to a channel. Works on channels the account follows/owns —
        posting is a send to the newsletter JID."""
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "empty post"}
        chat_ref = ChatRef(platform="whatsapp",
                            chat_id=self._newsletter_jid(chat),
                            kind=ChatKind.CHANNEL, title="")
        result = self.send(chat_ref, text)
        return {"ok": result.ok, "message_id": result.message_id,
                "error": result.error or ""}

    def _newsletter_jid(self, chat: ChatRef | str) -> str:
        """Normalize a chat ref or raw string to a newsletter JID."""
        if isinstance(chat, ChatRef):
            jid = self._jid(chat)
        else:
            jid = str(chat or "").strip()
        return jid

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
                                 kind=str(m.get("kind") or "file"),
                                 is_voice=str(m.get("kind") or "") == "voice")
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
                "pairing_code_pending": bool(self.last_pairing_code),
                "pairing_code_age_s":
                round(time.time() - self.last_pairing_code_at, 1)
                if self.last_pairing_code else None,
                "outbox_pending": self.outbox.pending_count(),
                "outbox_held": self.outbox.held_count(),
                "receipts_tracked": len(self._receipts)}
