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
    {"type":"status","state":"open|closed","user":"<jid>","lid":"<lid>@lid"}
    # "lid" is the account's own LID on LID-migrated accounts (""
    # or absent when the account has none) — the self-chat may
    # live under it instead of the phone JID in "user".
    {"type":"qr","data":"<qr string>"}
    {"type":"message","id":"<wa message id>","chat":{"id","kind","title"},
     "from":{"id","name"},"text":"...",
     "media":[{"path","mime","kind"}],"reply_to":"","mentioned":false,"ts":<ms>}
    # when the message REPLY-quotes an earlier one, the bridges
    # also carry the quoted message's own text and media —
    # downloaded at inbound, the only moment the bytes are
    # reachable: "quoted_text":"..." and
    # "quoted_media":[{"path","mime","kind"}].  Without them a
    # reply-media command (/sticker on a replied photo) can see
    # THAT a message was quoted but never WHAT was quoted.
    # a chat whose id ends in "@newsletter" is a CHANNEL (kind
    # "channel"); "@g.us" is a group.  The adapter derives the
    # kind from the JID (see _kind_for_jid), so a bridge build
    # that still calls newsletter traffic "dm" cannot get a
    # channel post conversed with — channels stay silent and are
    # only noted into the channel registry (§16/P4).
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
    {"id":"<req>","cmd":"presence","state":"available|unavailable"}   # account presence (§15)
    {"id":"<req>","cmd":"status_view","message_id":"<msg id>","from":"<poster jid>"}  # view one status (§15)
    {"id":"<req>","cmd":"status_react","message_id":"<msg id>","from":"<poster jid>","emoji":"<emoji>"}  # react to one status (§15)
    {"id":"<req>","cmd":"reject_call","call_id":"<call id>","from":"<caller jid>"}  # reject one ringing call
    # "id" is ONLY the request correlation id — _send_cmd writes
    # it last, over whatever the caller put there.  A command's
    # TARGET therefore never travels in "id": a message target
    # rides "message_id", a call target rides "call_id".  (The
    # status family once sent its target in "id"; every view and
    # reaction was aimed at the random request id instead, and
    # both bridges answered ok — pinned on the real wire by
    # tests/test_truthbuild_wire.py.)
    {"id":"<req>","cmd":"chats","limit":30}               # best-effort listing
    {"id":"<req>","cmd":"status"}
    # group/community depth (bridge read-only set):
    {"id":"<req>","cmd":"group_list"}                   # all groups: id/subject/size/isCommunity
    {"id":"<req>","cmd":"group_info","chat":"<@g.us>"}  # metadata + admins + invite link (if permitted)
    {"id":"<req>","cmd":"group_participants","chat":"<@g.us>"}  # roster with admin roles
    {"id":"<req>","cmd":"community_list"}              # communities + their linked groups
    {"id":"<req>","cmd":"channel_info","chat":"<@newsletter>"}  # channel name/description + follower count (newsletter metadata)

Offline resilience: when the bridge is down, ``send``/``send_media`` queue
to a persistent outbox (``data/chat/outbox/whatsapp.jsonl``) and flush FIFO
on reconnect — a bridge restart no longer eats composed replies.

The bridge owns the credential state (``.creds/`` directory); this client
stores nothing. Commands the bridge doesn't implement answer ok:false and
the adapter degrades gracefully (``mark_read``/``chats`` return False/[]).
"""

from __future__ import annotations

import hashlib
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
#: Entries older than this are dropped on flush, never delivered. A
#: queued entry is a reply to a moment — a bridge flap or a restart —
#: and past a few minutes the moment is gone: replaying it at the
#: next attach presents a stale send as a fresh one (the 2026-11-04
#: boot flood: a full command catalog landing in the owner's
#: self-chat at startup). Five minutes covers reconnect backoff
#: (capped at 60s) and a bridge-supervisor restart with room to
#: spare; anything older is aged out and journaled, not sent.
OUTBOX_TTL_S = 300.0
#: Inbound messages older than this on arrival are replays, not live
#: turns: WhatsApp delivers everything composed while a linked
#: device was offline the moment the bridge (re)connects, and a
#: history sync can re-surface older messages still. The adapter
#: still delivers them (persistence, the antidelete cache and the
#: journals must see them) but stamps ``meta["stale_replay"]`` —
#: the runtime observes them and never answers or executes them, so
#: a /help typed during a restart cannot detonate as a boot flood.
STALE_INBOUND_S = 180.0


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
                age = now - float(entry.get("enqueued_at", now))
                if age > self.ttl_s:
                    rows.remove(entry)
                    dropped += 1
                    # Journaled, never silent: an aged-out entry is a
                    # reply that will never be sent, and the record
                    # (kind + chat + age — never the text) is how a
                    # "she never answered" report gets explained.
                    _log.warning(
                        "whatsapp outbox: aged out %s entry for %s "
                        "(%.0fs old, past the %.0fs reply window) — "
                        "dropped, not delivered stale",
                        entry.get("kind"), entry.get("chat_id"),
                        age, self.ttl_s)
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


def _wa_quoted_media(obj: dict[str, Any]) -> dict[str, Any] | None:
    """The replied-to message's media as the bridge downloaded
    it (``quoted_media`` rows of {path, mime, kind}) — the first
    row with a real path, shaped for the ``replied_media_local``
    meta the runtime's media resolver reads.  None when the
    event carries no quoted media (a text reply, no reply, or a
    bridge too old to send the field): the resolver then falls
    through exactly as it did before quoted media existed."""
    rows = obj.get("quoted_media") or []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("path"):
                return {"path": str(row["path"]),
                        "kind": str(row.get("kind") or "file"),
                        "mime": str(row.get("mime") or ""),
                        "name": str(row.get("name") or "")}
    return None


def _wa_stamp_quoted(msg: ChatMessage, obj: dict[str, Any]) -> None:
    """Carry a bridge event's quoted-message payload onto the
    delivered message: the quoted media's local bytes for the
    reply-media commands, and the quoted text for the reply
    context readers.  Absent fields stamp nothing."""
    quoted = _wa_quoted_media(obj)
    if quoted is not None:
        msg.meta["replied_media_local"] = quoted
    quoted_text = str(obj.get("quoted_text") or "")
    if quoted_text:
        msg.meta["quoted_text"] = quoted_text


def _whatsapp_poll_shape_error(question: str,
                               options: list[str]) -> str:
    """WhatsApp's own poll shape rules, checked BEFORE any
    bridge traffic (the bridges re-check them — a malformed
    poll must refuse in words at every layer, never as a raw
    protocol error): 2–12 options, a non-empty question of at
    most 255 characters, options of at most 100 characters."""
    if not question:
        return "a poll needs a question"
    if len(question) > 255:
        return "poll questions cap at 255 characters"
    if not 2 <= len(options) <= 12:
        return "a WhatsApp poll needs 2–12 options"
    for option in options:
        if not option:
            return "poll options cannot be empty"
        if len(option) > 100:
            return "poll options cap at 100 characters each"
    return ""


def _jid_user_part(jid: str) -> str:
    """The user part of a JID: ``2348012345678:12@s.whatsapp.net``
    → ``2348012345678``. Device parts and domains are addressing
    detail; identity comparisons key off this."""
    local = (jid or "").split("@", 1)[0]
    return local.split(":", 1)[0].strip()


def _kind_for_jid(chat_id: str, claimed: str = "",
                  *, default: str = ChatKind.DM) -> str:
    """The chat kind a JID *is*, whatever the bridge called it.

    The JID's server suffix is protocol truth on every WhatsApp
    engine: ``…@newsletter`` is a channel and ``…@g.us`` is a
    group.  Bridges older than the channel build label
    newsletter traffic "dm" (their kindOf knew two shapes), and
    a channel post filed as a DM would be conversed with — the
    one thing a channel must never get (§16 posture).  The
    adapter therefore derives the kind from the JID and only
    falls back to the bridge's claim for the shapes the JID
    itself doesn't decide (DMs, status)."""
    cid = (chat_id or "").strip()
    if cid.endswith("@newsletter"):
        return ChatKind.CHANNEL
    if cid.endswith("@g.us"):
        return ChatKind.GROUP
    kind = str(claimed or "").strip().lower()
    return kind if kind in ChatKind.ALL else default


def _is_command_text(text: str) -> bool:
    """True when *text* parses as a registered Devon command —
    the runtime's own verdict (``parse_control``), so a message
    promoted to a turn on this test would actually dispatch
    there.  Unknown slashes and plain text are False: they keep
    whatever fate they already had.  A parse failure resolves
    to False — the persist-only side is the safe side."""
    if not (text or "").strip().startswith("/"):
        return False
    try:
        from .control import parse_control

        return parse_control(text) is not None
    except Exception as exc:  # noqa: BLE001 - recognition
        record_failure(           # never breaks intake
            "whatsapp.WhatsAppAdapter._is_command_text", exc)
        return False


class WhatsAppAdapter(ChatAdapter):
    """Client for the local WhatsApp bridge (bridge/bin/ Go program
    by default, or the Node bridge/whatsapp-bridge.mjs)."""

    name = "whatsapp"
    supported_kinds = (ChatKind.DM, ChatKind.GROUP, ChatKind.CHANNEL)

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
        # The account's own LID ("<lid>@lid") on LID-migrated
        # accounts, from the bridge's status events; "" while the
        # bridge hasn't reported one (older bridge, or the account
        # has no LID). The self-chat may be addressed by it instead
        # of the phone JID — see _is_self_chat.
        self.user_lid: str = ""
        # The account's own WhatsApp profile (push) name, from the
        # bridge's status events ("user_name"). The bridge reports
        # NO sender name for the account's own messages, so this is
        # the name the owner's own turns are resolved with — without
        # it his identity degrades to a placeholder downstream
        # (register #29: /stats answered "unknown's attributes").
        self.user_name: str = ""
        # Progressive-display edit throttle (same seam as the
        # Telegram bot adapter's edit_progress): ~1 edit / 3s, on
        # change only.
        from .progress_throttle import ProgressThrottler
        self._progress = ProgressThrottler()
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
        #: Button derivation parity with the Telegram bot adapter
        #: (buttons completion, 2026-11-08): the freshest command
        #: per chat and whether the most recent inbound there was a
        #: command — a plain send with no declared buttons derives
        #: its keyboard from them (personal_buttons_for /
        #: adaptive_buttons_for), exactly like the bot adapter.
        self._last_command: dict[str, str] = {}
        self._last_inbound_was_command: dict[str, bool] = {}
        #: The interactive form this account's button sends use:
        #: "" = NativeFlow quick_reply (the default), "list" = a
        #: ListMessage. The value is EVIDENCE, never a guess: it
        #: flips to "list" when a send comes back
        #: buttons_form=text (the bridge had to degrade the
        #: quick_reply form) or when the owner pins it through
        #: /pair buttons; the render probe reads it back.
        self._buttons_form_pref: str = ""
        #: The buttons_form the bridge reported for the most recent
        #: button-carrying send ("" = none yet / not reported).
        self._last_buttons_form: str = ""

    # ── button derivation + form evidence ─────────────────────────
    def _deliver(self, handler: Any, message: Any) -> None:
        """Track the inbound turn for button derivation, then hand
        it to the base delivery — every dispatch site in this
        adapter funnels through here."""
        self._track_inbound(message)
        super()._deliver(handler, message)

    def _track_inbound(self, message: Any) -> None:
        """Remember the freshest command per chat (the derivation
        origin state the Telegram bot adapter keeps): a turn whose
        text is a command stamps the chat's origin; any other turn
        clears the was-a-command flag so a stale origin never
        lends buttons to an unrelated reply. Never raises."""
        try:
            chat_id = str(
                getattr(getattr(message, "chat", None), "chat_id", "")
                or "")
            if not chat_id:
                return
            text = str(getattr(message, "text", "") or "").strip()
            if text.startswith("/"):
                self._last_inbound_was_command[chat_id] = True
                self._last_command[chat_id] = text[1:].strip()
            else:
                self._last_inbound_was_command[chat_id] = False
            if len(self._last_command) > 256:
                self._last_command.pop(
                    next(iter(self._last_command)), None)
            if len(self._last_inbound_was_command) > 256:
                self._last_inbound_was_command.pop(
                    next(iter(self._last_inbound_was_command)), None)
        except Exception as exc:  # noqa: BLE001
            record_failure('whatsapp.WhatsAppAdapter._track_inbound', exc)

    def _derive_buttons(self, chat: ChatRef, text: str) -> Any:
        """The keyboard a plain send on this surface carries when
        the caller declared none — the Telegram bot adapter's
        derivation contract mirrored: this adapter IS the owner's
        personal account surface, so the personal set is tried
        first, then the adaptive set, both keyed on the chat's
        freshest command. Never raises; None when nothing fits."""
        try:
            from .adaptive_buttons import adaptive_buttons_for
            from .tgbot_buttons import personal_buttons_for

            fresh_origin = ""
            if self._last_inbound_was_command.get(chat.chat_id):
                fresh_origin = self._last_command.get(chat.chat_id, "")
            return (personal_buttons_for(text, fresh_origin)
                    or adaptive_buttons_for(
                        origin_command=fresh_origin, text=text))
        except Exception as exc:  # noqa: BLE001
            record_failure('whatsapp.WhatsAppAdapter._derive_buttons', exc)
            return None

    def _note_buttons_form(self, form: str, jid: str) -> None:
        """Record what form a button-carrying send actually went
        out in. A "text" report means the bridge had to degrade
        the declared buttons to text action lines: the actions
        still landed — as text — but a degraded form is a fact
        the ledger must carry (the failure kind /pair journal
        counts), and it is the EVIDENCE that flips this account's
        form preference to the list form for later sends."""
        form = str(form or "")
        if form:
            self._last_buttons_form = form
        if form != "text":
            return
        record_failure(
            "whatsapp.WhatsAppAdapter.send.buttons",
            RuntimeError(
                "buttons degraded to text action lines "
                f"for {jid} — the bridge reported "
                "buttons_form=text; the actions ride "
                "the message body as “label — /command” "
                "lines"))
        if self._buttons_form_pref != "list":
            self._buttons_form_pref = "list"
            _log.info("whatsapp: buttons_form=text for %s — this "
                      "account's sends now use the list form "
                      "(evidence, /pair buttons can pin it back)",
                      jid)

    def buttons_form_state(self) -> dict[str, Any]:
        """The account's button-form evidence, for /pair buttons:
        the remembered preference and the last reported form."""
        return {"preference": self._buttons_form_pref or "quick",
                "last_form": self._last_buttons_form}

    def set_buttons_form_preference(self, style: str) -> str:
        """Pin the account's button form ("list" or "quick") —
        the owner's explicit override of the evidence default,
        via /pair buttons. Returns the preference now in force."""
        style = str(style or "").strip().lower()
        self._buttons_form_pref = "list" if style == "list" else ""
        return self._buttons_form_pref or "quick"

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

    def _own_display_name(self, obj: dict[str, Any]) -> str:
        """The owner's display name for one of his own turns.

        Resolution chain (member-resolution law: every path, ends
        at the identity or an honest label — never an empty name
        that downstream code turns into a placeholder):
        1. the event's own name field (older wire shapes carried
           it flat; the Go bridge carries it as from.name),
        2. the account's WhatsApp profile name, learned from the
           bridge's status events (``user_name``),
        3. the account's own number — his id as the honest label.
        """
        name = str(obj.get("name") or "").strip()
        if not name:
            name = str((obj.get("from") or {}).get("name") or "").strip()
        if not name:
            name = (self.user_name or "").strip()
        if not name:
            name = _jid_user_part(self.user_jid)
        return name

    def _is_self_chat(self, chat_id: str, kind: str) -> bool:
        """True when *chat_id* is this account's own "Message
        yourself" chat: a DM whose JID is one of the account's own
        identities — the phone JID (the ``user`` the bridge reports
        in its status events) or, on LID-migrated accounts, the
        account's LID (``lid`` in the same events): WhatsApp
        addresses the self-chat by whichever identity the account
        lives under, and the LID's user part never equals the
        phone number, so both must be accepted. Groups and
        broadcasts are never the self-chat, and while neither of
        the account's own identities is known (no status event
        yet, or a bridge that reports no LID) an @lid chat matches
        nothing — the persist-only default is the safe side."""
        if not chat_id or not (self.user_jid or self.user_lid):
            return False
        if (kind or "").strip().lower() == str(ChatKind.GROUP):
            return False
        if chat_id.endswith("@g.us") or chat_id == "status@broadcast":
            return False
        chat_user = _jid_user_part(chat_id)
        if not chat_user:
            return False
        return any(
            mine and chat_user == mine
            for mine in (_jid_user_part(self.user_jid),
                         _jid_user_part(self.user_lid)))

    # ── outbox ───────────────────────────────────────────────────────────
    def _chat_from_entry(self, entry: dict[str, Any]) -> ChatRef:
        return ChatRef(
            platform=self.name,
            chat_id=str(entry.get("chat_id") or ""),
            kind=_kind_for_jid(str(entry.get("chat_id") or ""),
                               str(entry.get("chat_kind") or "")),
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
                result = self.send_media(
                    chat, media,
                    caption=str(entry.get("caption") or ""),
                    buttons=entry.get("buttons") or None)
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
            entry_mentions = entry.get("mentions") or None
            for i, chunk in enumerate(chunks):
                result, outcome_known = self._send_text_outcome(
                    chat, chunk, reply_to=reply_to if i == 0 else "",
                    buttons=entry_buttons if i == 0 else None,
                    mentions=entry_mentions if i == 0 else None)
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
            self.user_lid = str(obj.get("lid") or "")
            # An older bridge reports no user_name — never let its
            # empty value clobber a name already learned.
            self.user_name = str(obj.get("user_name") or "") or self.user_name
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
            # forever. Own CONVERSATION in every other chat — DMs
            # with other people, groups — keeps the persist-only
            # discipline untouched: she never replies as the owner
            # mid-chat.
            #
            # SECOND carve-out — the owner's commands typed inside
            # a GROUP or a 1:1 DM. On the Telegram userbot he is
            # the operator everywhere: his commands dispatch.
            # Here the bridge's own_message event is the same
            # proof of authorship the self-chat carve-out already
            # trusts (the event exists only for the paired
            # account's own messages), so a message of his in a
            # group — or in a DM with a third party — that parses
            # as a registered command is delivered as an owner
            # turn (incoming, from_owner, owner_command) and
            # reaches the same control dispatch. The DM half is
            # the 2026-11-08 truth fix: the groups-only promotion
            # left EVERY chatops command typed as-him inside a
            # contact's DM dead silent — he replied /viewonce to
            # the very photo he wanted revealed, in the chat
            # where the answer belongs, and the persist-only
            # fallback swallowed the command whole (the handler
            # was never broken; it was never reached). A typed
            # command is an explicit summons, not conversation:
            # promoting it does not open a conversation path,
            # because everything else he types in that DM —
            # plain text, unknown slashes — is NOT promoted: it
            # stays persist-only as-him conversation, never
            # answered, never a game move, exactly as before.
            # A stale command (typed while the surface was down)
            # gets the self-chat treatment too: stamped
            # stale_replay, observed, never detonated at
            # reconnect.
            chat = obj.get("chat") or {}
            chat_id = str(chat.get("id") or "")
            if not chat_id:
                return
            ts = obj.get("ts") or 0
            try:
                ts = float(ts) / 1000.0
            except (TypeError, ValueError):
                ts = time.time()
            chat_kind = _kind_for_jid(
                chat_id, str(chat.get("kind") or ""))
            message_id = str(obj.get("id") or "")
            sent_by_me = self._was_sent_by_me(message_id)
            if not sent_by_me and \
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
                    sender=self._own_display_name(obj),
                    sender_id=self.user_jid,
                    message_id=message_id,
                    ts=ts,
                )
                turn.meta["from_owner"] = True
                turn.meta["self_chat"] = True
                _wa_stamp_quoted(turn, obj)
                if ts and (time.time() - ts) > STALE_INBOUND_S:
                    # Offline-delivered at (re)connect: the owner
                    # typed this while the surface was down. It is
                    # observed, never answered — a stale self-chat
                    # command detonating at boot is the flood this
                    # stamp exists to prevent.
                    turn.meta["stale_replay"] = True
                self._deliver(handler, turn)
                return
            if not sent_by_me \
                    and (chat_kind or "").strip().lower() in (
                        str(ChatKind.GROUP), str(ChatKind.DM)) \
                    and _is_command_text(str(obj.get("text") or "")):
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
                    sender=self._own_display_name(obj),
                    sender_id=self.user_jid,
                    message_id=message_id,
                    reply_to=str(obj.get("reply_to") or ""),
                    ts=ts,
                )
                turn.meta["from_owner"] = True
                turn.meta["owner_command"] = True
                _wa_stamp_quoted(turn, obj)
                if ts and (time.time() - ts) > STALE_INBOUND_S:
                    # Offline-delivered at (re)connect, like the
                    # self-chat above: observed, never executed.
                    turn.meta["stale_replay"] = True
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
                reply_to=str(obj.get("reply_to") or ""),
                ts=ts,
            )
            msg.meta["own_message"] = True
            # Owner-replied suppression (register #28): the quoted
            # target rides along so the runtime's ledger can mark
            # that exact message answered-by-owner.
            if obj.get("reply_to"):
                msg.meta["reply_to"] = str(obj.get("reply_to"))
            # The sender IS the owner — flag it for group recognition.
            msg.meta["from_owner"] = True
            _wa_stamp_quoted(msg, obj)
            self._deliver(handler, msg)
        elif kind == "call_offer":
            # Personal surface feed (chat & privacy wave): a call
            # ringing this account, forwarded by the bridges
            # (whatsmeow CallOffer / Baileys 'call').  Delivered
            # as a service message — the runtime's personal hook
            # may reject it under the owner's anticall toggle and
            # consumes it; it is never conversation.  An older
            # bridge never sends these, and then anticall simply
            # has no feed — nothing here invents one.
            caller = str(obj.get("from") or "")
            call_id = str(obj.get("id") or "")
            if not caller or not call_id:
                return
            ts = obj.get("ts") or 0
            try:
                ts = float(ts) / 1000.0
            except (TypeError, ValueError):
                ts = time.time()
            msg = ChatMessage(
                chat=ChatRef(
                    platform=self.name,
                    chat_id=caller,
                    kind=ChatKind.DM,
                    title="",
                    peer=caller,
                ),
                incoming=True,
                text="",
                sender="",
                sender_id=caller,
                message_id="",
                ts=ts,
            )
            msg.meta["service"] = "call_offer"
            msg.meta["call_id"] = call_id
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
            # Media the bridge reported WITHOUT a local path
            # (its download failed) was filtered out of the
            # media list above — the same silent-drop shape the
            # 2026-11-06 photo fix killed for ordinary messages:
            # the status saver could not tell "a text status"
            # from "a photo status we dropped". Stamp the
            # arrival so the personal surface journals the miss
            # (meta["media_unavailable"] is the brain's copy of
            # this stamp; the saver's is status_media_missing).
            _status_missed = [
                m for m in (obj.get("media") or [])
                if isinstance(m, dict) and not m.get("path")
            ]
            if _status_missed:
                msg.meta["status_media_missing"] = True
                msg.meta["media_unavailable"] = [
                    {"kind": str(m.get("kind") or "file"),
                     "reason": "download_failed"}
                    for m in _status_missed
                ]
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
                kind=_kind_for_jid(
                    chat_id, str(chat.get("kind") or ""),
                    default=ChatKind.GROUP),
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
                        kind=_kind_for_jid(
                            chat_id, str(chat.get("kind") or "")),
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
            # Media the bridge reported WITHOUT a local path (its
            # download failed) was filtered out of wa_msg.media
            # above — silently, until 2026-11-06: the brain then
            # could not tell "no photo was sent" from "a photo was
            # sent and we dropped it", and in the owner's voice
            # claimed the photo never arrived. Stamp the arrival
            # so the brain records it honestly (the brain reads
            # meta["media_unavailable"]).
            _wa_missed = [
                {"kind": str(m.get("kind") or "file"),
                 "reason": "download_failed"}
                for m in (obj.get("media") or [])
                if isinstance(m, dict) and not m.get("path")
            ]
            if _wa_missed:
                wa_msg.meta["media_unavailable"] = _wa_missed
            _wa_stamp_quoted(wa_msg, obj)
            _alt = str((obj.get("from") or {}).get("alt") or "").strip()
            if _alt:
                # The sender's other addressing form (LID ↔ phone
                # JID), reported by the bridge alongside from.id:
                # identity resolution folds both forms of one
                # person into one identity instead of minting a
                # second one per form (register #29).
                wa_msg.meta["sender_alt_id"] = _alt
            if ts and (time.time() - ts) > STALE_INBOUND_S:
                # Offline-delivered / history-replayed on (re)connect:
                # observed by the runtime, never answered or executed
                # (see STALE_INBOUND_S).
                wa_msg.meta["stale_replay"] = True
            if obj.get("view_once"):
                # The bridge unwrapped a view-once payload (§15):
                # the personal surface preserves the downloaded
                # media before WhatsApp vanishes it.
                wa_msg.meta["view_once"] = True
                if not wa_msg.media:
                    # The bridge could not download it in time —
                    # the personal surface journals the miss
                    # instead of the message passing silently.
                    wa_msg.meta["view_once_media_missing"] = True
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
                           buttons: Any = None,
                           mentions: Any = None,
                           fresh: bool = False) -> tuple[SendResult, bool]:
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

        ``fresh`` marks the first dispatch attempt of a reply the
        runtime composed for a LIVE inbound turn (the gateway sets
        it on that attempt only — its retry layers re-enter with
        fresh=False). A fresh solicited reply is never swallowed by
        the guard: the owner asked again, so the guard entry is
        released and the reply dispatches. Without this, a menu
        whose chunks lost their acks in a boot flap made every
        later identical /help unanswerable for the whole window
        (2026-11-04: the flood, then silence).
        """
        started = time.perf_counter()
        jid = self._jid(chat)
        key = (jid, text)
        prior = self._unconfirmed_at(key)
        if prior is not None and fresh:
            self._clear_unconfirmed(key)
            _log.warning(
                "whatsapp: releasing a guarded identical send to %s "
                "for a fresh solicited reply (the earlier dispatch's "
                "ack was lost; a new live turn asked again)", jid)
            prior = None
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
        if mentions:
            # /tagall's payload: the bridges attach these JIDs as
            # the message's mention set (they notify whether or
            # not the text names them — hidetag rides this).
            request["mentions"] = [str(m) for m in mentions if m]
        if buttons:
            # The bridge renders what WhatsApp supports natively
            # and degrades the rest to text action lines — the
            # keyboard is never silently dropped (presentation
            # contract, same as every other platform). When this
            # account's evidence says the quick_reply form does
            # not land, the bridges are asked for the LIST form
            # instead (buttons_style) — same commands, other
            # interactive shape.
            request["buttons"] = [
                [[str(label), str(data)] for label, data in row]
                for row in buttons if row]
            if self._buttons_form_pref == "list":
                request["buttons_style"] = "list"
        for attempt in (1, 2):
            response = self._send_cmd(request)
            if response.get("ok"):
                self.stats["sent"] += 1
                if buttons:
                    # What form the keyboard actually went out in
                    # (interactive / list / text-degraded) is
                    # ledgered — and a text degradation is the
                    # evidence that flips later sends to the list
                    # form (see _note_buttons_form).
                    self._note_buttons_form(
                        str(response.get("buttons_form") or ""), jid)
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
                         buttons: Any = None,
                         mentions: Any = None,
                         fresh: bool = False) -> SendResult:
        """One text send with a single retry on transient bridge failure."""
        result, _outcome_known = self._send_text_outcome(
            chat, text, reply_to=reply_to, buttons=buttons,
            mentions=mentions, fresh=fresh)
        return result

    def send(self, chat: ChatRef, text: str, *,
             reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             reply_keyboard: list[list[str]] | None = None,
             parse_mode: str = "",
             mentions: list[str] | None = None,
             fresh: bool = False) -> SendResult:
        started = time.perf_counter()
        if buttons is None:
            # Derivation parity with the Telegram bot adapter: a
            # caller that declared no keyboard gets the surface's
            # derived one (explicit buttons — including an
            # explicit empty list — always win). Derived before
            # the connected check so a queued send carries the
            # same keyboard the live path would have sent.
            buttons = self._derive_buttons(chat, text)
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
                "mentions": [str(m) for m in mentions if m]
                            if mentions else None,
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
                buttons=buttons if i == 0 else None,
                mentions=mentions if i == 0 else None,
                fresh=fresh)
            if not result.ok:
                return result
        assert result is not None
        return result

    def edit_progress(self, chat: ChatRef, message_id: str,
                      new_text: str) -> dict[str, Any] | None:
        """Edit one of this account's OWN sent messages in place —
        the progressive-display seam (register #31), same contract
        as the Telegram bot adapter's ``edit_progress``:
        ``ProgressReporter`` flows bind this as their emit.

        WhatsApp edits are real but narrow: the Go bridge sends a
        protocol edit (whatsmeow BuildEdit) for a message this
        bridge itself sent, inside WhatsApp's edit window; the
        server refuses anything else (someone else's message, a
        window-expired message) and the refusal comes back by
        name. Returns the bridge's verdict dict on success, None
        when the edit was throttled (unchanged / too soon) or
        refused — refusals are logged, never hidden, and a None
        here never blocks the caller's real result. Never raises.
        """
        if not self.connected.is_set():
            return None
        if not message_id:
            return None
        if not self._progress.should_edit(new_text):
            return None
        try:
            from .platforms import format_for_platform
            wire = format_for_platform(new_text or "", "whatsapp")
        except Exception as exc:  # noqa: BLE001 - raw text beats no edit
            record_failure("whatsapp.WhatsAppAdapter.edit_progress", exc)
            wire = new_text or ""
        try:
            response = self._send_cmd(
                {"cmd": "edit", "chat": self._jid(chat),
                 "message_id": str(message_id),
                 "text": wire[:for_platform("whatsapp").max_text]},
                timeout=15)
        except Exception as exc:  # noqa: BLE001 - progress is best-effort
            record_failure("whatsapp.WhatsAppAdapter.edit_progress", exc)
            return None
        if not response.get("ok"):
            _log.debug("whatsapp: edit refused for %s in %s: %s",
                       message_id, chat.key,
                       response.get("error") or "bridge error")
            return None
        return response

    def send_media(self, chat: ChatRef, media: MediaRef, *,
                   caption: str = "", parse_mode: str = "",
                   buttons: list[list[tuple[str, str]]] | None = None
                   ) -> SendResult:
        """Send a photo/video/audio/document via the bridge's send_media.

        ``buttons`` (buttons completion, 2026-11-08): the media
        itself carries the keyboard — both bridges head an
        InteractiveMessage with the uploaded media (image / video
        / document; audio's actions ride as a follow-up text), the
        same 3-native + text-line degradation contract as text
        sends, and the reply's buttons_form is ledgered the same
        way. This parameter existing is also what unblocks the
        gateway's media path, which passes declared buttons only
        to adapters whose send_media declares them.

        When the bridge is down the media is queued (path + caption
        + buttons) and flushed on reconnect — same as text."""
        started = time.perf_counter()
        path = (media.path or "").strip()
        if not path:
            return SendResult(ok=False, platform=self.name,
                              error="media has no path",
                              seconds=time.perf_counter() - started)
        wire_buttons = (
            [[[str(label), str(data)] for label, data in row]
             for row in buttons if row] if buttons else None)
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
                "buttons": wire_buttons,
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
        if wire_buttons:
            # No buttons_style: a ListMessage cannot head media —
            # the media form is the interactive header on both
            # bridges regardless of the text-form preference.
            cmd["buttons"] = wire_buttons
        kind = (media.kind or "").lower()
        if kind in ("voice", "ptt"):
            cmd["ptt"] = True  # voice-note bubble instead of an audio file
        response = self._send_cmd(cmd, timeout=60)
        if response.get("ok"):
            self._clear_unconfirmed(media_key)
            self.stats["sent"] += 1
            if wire_buttons:
                self._note_buttons_form(
                    str(response.get("buttons_form") or ""),
                    self._jid(chat))
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

    def set_presence(self, online: bool) -> dict[str, Any]:
        """Assert the account's global presence (§15 always-online).

        The bridge's ``presence`` verb: whatsmeow SendPresence on
        the Go engine, Baileys sendPresenceUpdate with no chat on
        the Node engine.  Returns the bridge's verdict dict — a
        bridge that doesn't implement the verb answers ok:false
        with that reason, never a fake success."""
        return self._admin_cmd("presence", {
            "state": "available" if online else "unavailable"},
            timeout=10)

    def view_status(self, poster_id: str, message_id: str,
                    ts: float = 0.0) -> dict[str, Any]:
        """Mark one contact status as viewed by this account (§15
        status viewing): the bridge's ``status_view`` verb sends
        the read receipt for that status post (whatsmeow MarkRead
        on status@broadcast / Baileys readMessages with the
        status key).  The bridge's verdict dict comes back —
        ok:false carries the honest reason, and a landed view
        carries ``receipt_mode``: "read" when the poster is
        notified, "read-self" when the account's WhatsApp read
        receipts are OFF and the platform downgraded the receipt
        so the poster never sees it (the bridges read the
        account's privacy setting to say so; a rung that cannot
        read it reports "unknown").  *ts* (the post's timestamp)
        is accepted for call-shape compatibility and deliberately
        NOT sent: the bridges stamp the receipt at read time,
        like the phone (status mining, 2026-11-07)."""
        poster_id = str(poster_id or "").strip()
        message_id = str(message_id or "").strip()
        if not poster_id or not message_id:
            return {"ok": False,
                    "error": "status_view needs the poster's jid "
                             "and the status message id"}
        return self._admin_cmd("status_view", {
            "message_id": message_id, "from": poster_id},
            timeout=10)

    def send_status(self, text: str = "", *,
                    media_path: str = "",
                    caption: str = "",
                    audience: list[str] | None = None
                    ) -> dict[str, Any]:
        """Post a status as the account owner (§15 status
        posting — the send half of the status family, the
        owner's /poststatus): the bridges' ``status_send``
        verb. Text, or an image/video already on disk with an
        optional caption. The Go engine's whatsmeow resolves
        the audience from the account's server-side status
        recipients; the Baileys rung needs the explicit
        statusJidList — *audience* when given, else the
        bridge's own session contacts, else an honest refusal
        (a status encrypted for nobody is not a post). The
        bridge's verdict dict comes back — ok carries the new
        status message id, ok:false the honest reason."""
        text = str(text or "").strip()
        media_path = str(media_path or "").strip()
        if not text and not media_path:
            return {"ok": False,
                    "error": "status_send needs text or a "
                             "media path"}
        payload: dict[str, Any] = {
            "text": text, "path": media_path,
            "caption": str(caption or "")}
        if audience:
            payload["audience"] = [str(a) for a in audience if a]
        return self._admin_cmd("status_send", payload,
                               timeout=30)

    # ── chat & privacy ops (FEI gap wave F) ─────────────────────
    # Owner-invoked verbs on his own account, each riding a
    # bridge command both engines implement (Go/whatsmeow +
    # Baileys — see bridge/gobridge/messages.go and
    # bridge/whatsapp-bridge.mjs).  Every method returns the
    # bridge's verdict dict: ok:false carries the honest reason
    # (a rung that lacks the verb says so), never a fake success.

    def edit_message(self, chat: ChatRef, message_id: str,
                     text: str, *,
                     buttons: list[list[tuple[str, str]]] | None = None
                     ) -> dict[str, Any]:
        """Edit one of this account's OWN messages — the bridge's
        ``edit`` verb WITHOUT edit_progress's throttle (that one
        serves progressive display; this is the owner's /editmsg).
        Both bridges refuse a target that isn't the account's own
        or isn't in their memory, and WhatsApp's ~15-minute edit
        window is enforced server-side — refusals come back in
        the verdict, verbatim.

        ``buttons`` (buttons completion, 2026-11-08) is
        presence-significant, mirroring the session layer's
        contract: None leaves the call a text-only edit (the
        bridges' original behavior); a list REPLACES the message
        with that text + keyboard (both bridges rebuild the
        interactive/list replacement inside the edit wrap); an
        EMPTY list is a keyboard-CLEARING edit — the settled
        render. When a bridge refuses a clearing edit outright
        (window expired, target forgotten), the adapter falls
        back to revoke + a fresh send of the settled render, so a
        settled card never keeps live buttons: the verdict then
        carries ``via: "revoke_resend"`` and the NEW message_id
        (the session registry re-keys on the next render, its
        standing self-heal).

        The text is converted to the WhatsApp dialect first (the
        one-formatting-system law): ``send`` and ``edit_progress``
        both convert, and this verb skipping the conversion was a
        divergence — canonical ``**bold**`` edited in through here
        would have shown its raw markers in the chat.
        """
        message_id = str(message_id or "").strip()
        if not message_id:
            return {"ok": False, "error": "edit needs a message_id"}
        wire = str(text or "")
        try:
            from .platforms import format_for_platform
            wire = format_for_platform(wire, "whatsapp")
        except Exception as exc:  # noqa: BLE001 - raw beats no edit
            record_failure("whatsapp.WhatsAppAdapter.edit_message", exc)
        payload: dict[str, Any] = {
            "chat": self._jid(chat), "message_id": message_id,
            "text": wire}
        if buttons is not None:
            payload["buttons"] = [
                [[str(label), str(data)] for label, data in row]
                for row in buttons if row]
            if buttons and self._buttons_form_pref == "list":
                payload["buttons_style"] = "list"
        verdict = self._admin_cmd("edit", payload, timeout=15)
        if verdict.get("ok") or buttons is None or len(buttons) > 0:
            return verdict
        # A keyboard-CLEARING edit the bridge refused: revoke the
        # card and re-send the settled render without buttons, so
        # the settled state is what the chat actually shows.
        revoked = self._admin_cmd("message_delete", {
            "chat": self._group_jid(chat), "message_id": message_id})
        if not revoked.get("ok"):
            return verdict
        resent = self.send(chat, str(text or ""), buttons=[])
        if not resent.ok:
            return {"ok": False,
                    "error": "edit refused ("
                             + str(verdict.get("error") or "bridge error")
                             + "); the card was revoked but the settled "
                               "re-send failed: "
                             + str(resent.error or "send failed")}
        return {"ok": True, "message_id": resent.message_id,
                "via": "revoke_resend"}

    def chat_modify(self, chat: ChatRef, *,
                    archive: bool | None = None,
                    pin: bool | None = None,
                    mute_seconds: int | None = None,
                    read: bool | None = None) -> dict[str, Any]:
        """Chat-state changes on one chat — archive / pin / mute /
        mark read-or-unread — via the bridges' ``chat_modify``
        (whatsmeow app-state patches / Baileys chatModify).
        Tri-state: a field left None is not sent, so the bridge
        leaves that aspect alone.  mute_seconds: -1 = no end,
        0 = unmute, >0 = that many seconds."""
        payload: dict[str, Any] = {"chat": self._jid(chat)}
        if archive is not None:
            payload["archive"] = bool(archive)
        if pin is not None:
            payload["pin"] = bool(pin)
        if mute_seconds is not None:
            payload["mute_seconds"] = int(mute_seconds)
        if read is not None:
            payload["read"] = bool(read)
        if len(payload) == 1:
            return {"ok": False,
                    "error": "chat_modify needs at least one change"}
        return self._admin_cmd("chat_modify", payload, timeout=20)

    def chat_archive(self, chat: ChatRef,
                     archived: bool) -> dict[str, Any]:
        return self.chat_modify(chat, archive=archived)

    def chat_mute(self, chat: ChatRef, *,
                  seconds: int | None) -> dict[str, Any]:
        """Mute/unmute one CHAT (not a member): seconds -1 = no
        end, 0 = unmute, >0 = mute for that long."""
        return self.chat_modify(chat, mute_seconds=seconds)

    def chat_mark_read(self, chat: ChatRef,
                       read: bool) -> dict[str, Any]:
        """Mark a whole chat read (True) or unread (False) — the
        chat-list state, distinct from mark_read's receipts."""
        return self.chat_modify(chat, read=read)

    def star_message(self, chat: ChatRef, message_id: str,
                     star: bool = True) -> dict[str, Any]:
        """Star/unstar one message (bridges' ``message_star``):
        the key is built from bridge memory — a message neither
        bridge has seen is an honest refusal."""
        message_id = str(message_id or "").strip()
        if not message_id:
            return {"ok": False,
                    "error": "star needs a message_id"}
        return self._admin_cmd("message_star", {
            "chat": self._jid(chat), "message_id": message_id,
            "star": bool(star)}, timeout=15)

    def react_message(self, chat: ChatRef, message_id: str,
                      emoji: str = "") -> dict[str, Any]:
        """React to one CHAT message as the owner (bridges'
        ``react``; parity G1 — the message-level sibling of
        react_status).  The reaction key is built from bridge
        memory on both engines — a message neither bridge has
        seen is an honest refusal.  An empty emoji REMOVES the
        owner's reaction (the shape the phone itself sends)."""
        message_id = str(message_id or "").strip()
        if not message_id:
            return {"ok": False,
                    "error": "react needs a message_id"}
        return self._admin_cmd("react", {
            "chat": self._jid(chat), "message_id": message_id,
            "emoji": str(emoji or "")}, timeout=15)

    def forward_message(self, chat: ChatRef, message_id: str,
                        dest: ChatRef) -> dict[str, Any]:
        """Forward one message into another chat as the owner
        (bridges' ``forward``; parity G1).  Both engines re-send
        the ORIGINAL message out of bridge memory with the
        forwarded mark — never a retyped copy; a message the
        bridge never saw, and view-once wraps, refuse honestly.
        One call is one forward: this verb never retries, so a
        command can never double-send by accident."""
        message_id = str(message_id or "").strip()
        if not message_id:
            return {"ok": False,
                    "error": "forward needs a message_id"}
        return self._admin_cmd("forward", {
            "chat": self._jid(chat), "message_id": message_id,
            "dest": self._jid(dest)}, timeout=30)

    def react(self, chat: ChatRef, message_id: str,
              emoji: str) -> SendResult:
        """The §22 base reaction verb, carried on WhatsApp by
        the bridges' ``react`` (parity G1): before this, the
        adapter inherited the base class's honest "reactions
        not supported" — the verb simply didn't exist on either
        engine.  An empty emoji removes the reaction."""
        verdict = self.react_message(chat, message_id, emoji)
        if isinstance(verdict, dict) and verdict.get("ok"):
            return SendResult(ok=True, platform=self.name,
                              message_id=str(message_id or ""))
        error = ""
        if isinstance(verdict, dict):
            error = str(verdict.get("error") or "")
        return SendResult(ok=False, platform=self.name,
                          error=error or "the bridge refused")

    def react_status(self, poster_id: str, message_id: str,
                     emoji: str) -> dict[str, Any]:
        """React to one contact status post as the owner (the
        status_react toggle's verb): the bridges' ``status_react``
        — a reaction keyed on the post at status@broadcast."""
        poster_id = str(poster_id or "").strip()
        message_id = str(message_id or "").strip()
        emoji = str(emoji or "").strip()
        if not poster_id or not message_id or not emoji:
            return {"ok": False,
                    "error": "status_react needs the poster's jid, "
                             "the status message id and an emoji"}
        return self._admin_cmd("status_react", {
            "message_id": message_id, "from": poster_id,
            "emoji": emoji}, timeout=15)

    # -- own identity (§/parity G3, census bundle #5) --------
    # The account could be READ (a push name arrives on every
    # message) but nothing on it could be CHANGED — no verb
    # existed on either engine.  These ride the bridges'
    # profile/privacy verbs; a refusal (WhatsApp rate-limits
    # profile changes) surfaces verbatim, never retried here.

    def get_profile(self) -> dict[str, Any]:
        """The account's current identity as the bridges read
        it: {"ok", "name", "about", "has_photo"} — a bridge
        that could not read the about/photo says so in
        "info_error" instead of inventing values."""
        return self._admin_cmd("profile_get", {}, timeout=15)

    def set_profile_name(self, name: str) -> dict[str, Any]:
        """Set the account's profile (push) name (bridges'
        ``profile_set`` field=name: whatsmeow app-state
        push-name patch / Baileys updateProfileName)."""
        name = str(name or "").strip()
        if not name:
            return {"ok": False,
                    "error": "a profile name cannot be empty"}
        return self._admin_cmd("profile_set", {
            "field": "name", "value": name}, timeout=20)

    def set_about(self, text: str) -> dict[str, Any]:
        """Set the account's About text (bridges' ``profile_set``
        field=about: whatsmeow SetStatusMessage — the profile
        "about", not a status broadcast — / Baileys
        updateProfileStatus).  An empty text clears it."""
        return self._admin_cmd("profile_set", {
            "field": "about", "value": str(text or "")},
            timeout=20)

    def set_profile_photo(self, path: str = "", *,
                          remove: bool = False) -> dict[str, Any]:
        """Set (or remove) the account's profile photo from a
        local image file (bridges' ``profile_set`` field=photo).
        The Go rung needs JPEG — WhatsApp's own picture
        contract — and refuses other formats in words; the
        Baileys rung crops/resizes any image itself."""
        if remove:
            return self._admin_cmd("profile_set", {
                "field": "photo", "remove": True}, timeout=30)
        path = str(path or "").strip()
        if not path:
            return {"ok": False,
                    "error": "profile photo needs an image path "
                             "(or remove=True)"}
        return self._admin_cmd("profile_set", {
            "field": "photo", "path": path}, timeout=30)

    def get_privacy(self) -> dict[str, Any]:
        """The account's privacy settings as the bridge reads
        them: {"ok", "settings": {category: value}} — a
        category the protocol cannot read back is ABSENT from
        the map, never reported with a guessed value."""
        return self._admin_cmd("privacy_get", {}, timeout=15)

    def set_privacy(self, category: str,
                    value: str) -> dict[str, Any]:
        """Set one privacy category (bridges' ``privacy_set``).
        The vocabulary is the protocol's own, validated in the
        shared table (chatops.PRIVACY_CATEGORIES) and again in
        the bridges; a category one engine has no setter for
        answers unsupported on that rung — honestly, in the
        verdict."""
        category = str(category or "").strip().lower()
        value = str(value or "").strip().lower()
        if not category or not value:
            return {"ok": False,
                    "error": "privacy_set needs a category and "
                             "a value"}
        return self._admin_cmd("privacy_set", {
            "category": category, "value": value}, timeout=20)

    # -- polls (§/parity G3, census bundle #6) ----------------

    def send_poll(self, chat: ChatRef, question: str,
                  options: list[str], *, anonymous: bool = True,
                  multiple: bool = False, quiz: bool = False,
                  correct_option: int | None = None,
                  explanation: str = "") -> SendResult:
        """Send a native WhatsApp poll via the bridges'
        ``send_poll`` (whatsmeow BuildPollCreation / Baileys
        sendMessage {poll}).  Same signature as the Telegram
        adapters' verb, so the group runtime's cross-post call
        works unchanged.  Platform truth that shapes the
        mapping, stated rather than silently absorbed:

        * WhatsApp polls have NO quiz mode: a quiz request
          refuses in words — quietly posting it as an opinion
          poll would change its meaning.
        * WhatsApp polls are never anonymous — voters are
          visible to participants by design, so ``anonymous``
          is accepted for signature parity and documented
          here, not faked.
        * Shape rules (2–12 options, question ≤255 chars,
          options ≤100) refuse BEFORE traffic."""
        if quiz:
            return SendResult(ok=False, platform=self.name, error=(
                "WhatsApp polls have no quiz mode — there is "
                "no correct-answer or explanation field to "
                "carry it"))
        question = str(question or "").strip()
        options = [str(o).strip() for o in (options or [])]
        problem = _whatsapp_poll_shape_error(question, options)
        if problem:
            return SendResult(ok=False, platform=self.name,
                              error=problem)
        selectable = len(options) if multiple else 1
        result = self._admin_cmd("send_poll", {
            "chat": self._jid(chat), "question": question,
            "options": options, "selectable": selectable},
            timeout=20)
        if not result.get("ok"):
            return SendResult(ok=False, platform=self.name,
                              error=str(result.get("error") or
                                        "the bridge refused the poll"))
        message_id = str(result.get("message_id") or "")
        if not message_id:
            return SendResult(ok=False, platform=self.name,
                              error=("the bridge sent the poll but "
                                     "returned no message id — the "
                                     "send is unproven"))
        return SendResult(ok=True, platform=self.name,
                          message_id=message_id)

    def poll_results(self, chat: ChatRef,
                     message_id: str) -> dict[str, Any]:
        """Tallies for one poll (bridges' ``poll_results``):
        {question, options: [{text, voters}], total_voters} —
        the format_poll_results verdict shape.  Results live
        in bridge memory: the bridges tally the votes they
        decrypted on polls they have seen; a poll outside that
        memory answers ok=False in words, never numbers
        invented to fill the shape."""
        message_id = str(message_id or "").strip()
        if not message_id:
            return {"ok": False,
                    "error": "poll_results needs the poll's "
                             "message id"}
        return self._admin_cmd("poll_results", {
            "chat": self._jid(chat), "message_id": message_id},
            timeout=15)

    def reject_call(self, call_id: str,
                    from_jid: str) -> dict[str, Any]:
        """Reject one ringing call (the anticall toggle's verb):
        the bridges' ``reject_call`` — both engines can see call
        offers and reject them; neither can answer one."""
        call_id = str(call_id or "").strip()
        from_jid = str(from_jid or "").strip()
        if not call_id or not from_jid:
            return {"ok": False,
                    "error": "reject_call needs the call id and "
                             "the caller's jid"}
        return self._admin_cmd("reject_call", {
            "call_id": call_id, "from": from_jid}, timeout=15)

    def block_user(self, jid: str,
                   action: str) -> dict[str, Any]:
        """Block/unblock one user on this account (bridges'
        ``blocklist``: whatsmeow UpdateBlocklist / Baileys
        updateBlockStatus)."""
        jid = str(jid or "").strip()
        action = str(action or "").strip().lower()
        if not jid:
            return {"ok": False, "error": "blocklist needs a jid"}
        if action not in ("block", "unblock"):
            return {"ok": False,
                    "error": "blocklist action must be block "
                             "or unblock"}
        return self._admin_cmd("blocklist", {
            "jid": jid, "action": action}, timeout=15)

    def set_disappearing(self, chat: ChatRef,
                         seconds: int) -> dict[str, Any]:
        """Set one chat's disappearing-messages timer (bridges'
        ``chat_disappear``; 0 = off).  Both engines set it for
        DMs and groups: the Go engine via whatsmeow
        SetDisappearingTimer; the Baileys engine via
        groupToggleEphemeral for groups and the
        ephemeral-setting protocol message (sendMessage with
        disappearingMessagesInChat) for DMs."""
        return self._admin_cmd("chat_disappear", {
            "chat": self._jid(chat),
            "seconds": int(seconds)}, timeout=15)

    def check_number(self, phone: str) -> dict[str, Any]:
        """The one registration question WhatsApp publishes
        (bridges' ``check_number``): {"ok", "exists", "jid"}.
        There is deliberately no ban field — WhatsApp publishes
        nothing about other numbers' ban state."""
        phone = str(phone or "").strip()
        if not phone:
            return {"ok": False,
                    "error": "check_number needs a phone number"}
        return self._admin_cmd("check_number", {
            "phone": phone}, timeout=15)

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
                "kind": _kind_for_jid(
                    str(entry.get("id") or ""),
                    str(entry.get("kind") or "")),
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

    def group_join_link(self, link: str) -> dict[str, Any]:
        """Join a group through its invite link (parity G2).

        Accepts the full ``chat.whatsapp.com/<code>`` URL or
        the bare code — the normalization happens here so both
        bridge engines only ever see a code.  An empty argument
        refuses before any bridge traffic.  The verdict is the
        bridge's own state machine: ``state`` is one of
        ``joined`` (the accept landed and the membership
        re-read finds the account inside), ``already_member``
        (no act was needed or taken), or ``approval_pending``
        (the group files join requests for admin approval —
        the request is in, the account is NOT a member yet);
        a dead link is an ok:False whose error says revoked
        vs not-valid distinctly.  Nothing here re-labels one
        state as another."""
        raw = str(link or "").strip()
        code = raw
        if "chat.whatsapp.com/" in code:
            code = code.split("chat.whatsapp.com/")[-1]
        code = code.strip().strip("/").split("/")[0].split("?")[0].strip()
        if not code:
            return {"ok": False,
                    "error": "missing invite link: pass the "
                             "chat.whatsapp.com link or its code"}
        return self._admin_cmd("group_join_link", {"code": code},
                               timeout=45)

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

    # ── group moderation seam (the runtime's admin_* grammar) ──
    # The group runtime (runtime_group._mod_run and the defense
    # sensors) moderates through adapter verbs — admin_ban /
    # admin_restrict / admin_delete / admin_pin … — with the
    # Telegram contract: return {"ok": True, ...} on success,
    # {"ok": False, "error": ...} on refusal; the runtime audits
    # either way and never reports a refusal as success.  This
    # section is that grammar for WhatsApp, mapped onto what the
    # platform ACTUALLY affords — no verb fakes an affordance
    # WhatsApp does not have:
    #
    #   ban      → remove the member (group_members_update).
    #              WhatsApp keeps no ban list: removal IS the ban,
    #              and return is possible only via a fresh invite
    #              or re-add — the same end-state Telegram's ban
    #              produces for a member who never rejoins.
    #   unban    → vacuous success, deliberately NO bridge call:
    #              with no ban list, a removed member is already
    #              free to rejoin via the invite link.  (Kick is
    #              ban+unban in the runtime — if unban re-added
    #              the member, kick would undo itself.)
    #   promote / demote → group_members_update, real.
    #   delete   → bridge message_delete (protocol revoke; the
    #              bridges resolve the message author's key from
    #              their own message memory — an admin may revoke
    #              any member's message, the owner her own).
    #   pin      → bridge message_pin (BOTH engines: the
    #              protocol's pin-in-chat message — whatsmeow
    #              builds it directly; Baileys' sendMessage
    #              {pin} builds the same message from the
    #              bridge's cached key).
    #   restrict (mute) → UNSUPPORTED, honestly: WhatsApp has NO
    #              per-member mute.  The only silences are
    #              removing the member (a kick wearing a mute's
    #              clothes — it cannot auto-expire and unmute
    #              cannot reliably reverse it) and setting the
    #              WHOLE group to admins-only (punishing every
    #              member for one member's mute).  Neither honors
    #              the verb's contract, so the verb refuses with
    #              the reason and the runtime audits the refusal
    #              (admin_action_failed / moderation_unexecuted)
    #              instead of a fake success.  Timed mutes on
    #              WhatsApp stay a named platform gap.
    #   admin_member → the roster's word on one member's role;
    #              this is also the runtime's admin-gate source
    #              (_group_admin_ok) on this platform.
    #   join requests → bridge group_join_requests (the door
    #              list) + group_join_request_decide, real on
    #              BOTH engines (whatsmeow
    #              GetGroupRequestParticipants /
    #              UpdateGroupRequestParticipants; Baileys
    #              groupRequestParticipantsList /
    #              groupRequestParticipantsUpdate).  WhatsApp
    #              admits fully or not at all — there is no
    #              restricted-admit state — so §28's request
    #              decisions on this platform are approve /
    #              reject only, and the surfaces say so.

    def _member_jid(self, chat: ChatRef | str, user_id: str) -> str:
        """Resolve a runtime member id to the JID the bridge
        verbs need.  Runtime ids on this platform are usually
        full JIDs already (the bridges emit sender ids in JID
        form) and pass through untouched; a bare user part is
        matched against the group roster first — that is what
        keeps LID-addressed members (``…@lid``) reachable, since
        a bare LID user part is indistinguishable from a phone
        number — and only an unmatched bare number falls back to
        the phone server."""
        uid = str(user_id or "").strip()
        if not uid:
            return ""
        if "@" in uid:
            return uid
        for row in self.group_participants(chat):
            jid = str(row.get("jid") or "")
            if _jid_user_part(jid) == uid:
                return jid
        digits = "".join(c for c in uid if c.isdigit())
        return f"{digits}@s.whatsapp.net" if digits else ""

    def _members_update_checked(self, chat: ChatRef | str,
                                  action: str,
                                  user_id: str) -> dict[str, Any]:
        """group_members_update for ONE member, with the bridge's
        per-user status honored: the bridge answers ok=True for
        the batch even when WhatsApp refused this member (status
        401/403/404…), so the member's own row decides — a "200"
        (or the bridge's "sent" when the server returned no rows)
        is success, anything else is the refusal, named."""
        jid = self._member_jid(chat, user_id)
        if not jid:
            return {"ok": False,
                    "error": f"no WhatsApp identity for {user_id!r}"}
        out = self.group_members_update(chat, action, [jid])
        if not isinstance(out, dict) or out.get("ok") is False:
            return out if isinstance(out, dict) else \
                {"ok": False, "error": "bridge gave no answer"}
        return self._per_user_verdict(out, jid, action)

    @staticmethod
    def _per_user_verdict(out: dict[str, Any], jid: str,
                          action: str) -> dict[str, Any]:
        """One member's verdict inside a batch reply whose ok
        was already honored: the member's OWN row decides (see
        _members_update_checked) — a "200" (or the bridge's
        "sent" when the server returned no rows) is success,
        anything else is the refusal, named."""
        rows = out.get("results") or []
        mine = [r for r in rows
                if _jid_user_part(str(r.get("jid") or ""))
                == _jid_user_part(jid)]
        row = mine[0] if mine else (rows[0] if len(rows) == 1
                                    else None)
        if row is None:
            return out
        status = str(row.get("status") or "")
        if status in ("200", "sent"):
            return {"ok": True, "action": action, "jid": jid,
                    "status": status}
        return {"ok": False, "jid": jid,
                "error": f"WhatsApp refused the {action} for "
                         f"{jid} (status {status or 'unknown'})"}

    def admin_member(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        """One member's standing, from the group roster:
        ``{"status": "creator"|"administrator"|"member"}`` —
        the shape the runtime's admin gate reads.  A member the
        roster doesn't hold (or a bridge that can't answer) is
        an honest ok=False, never a guessed role."""
        uid = str(user_id or "").strip()
        if not uid:
            return {"ok": False, "error": "no member id given"}
        for row in self.group_participants(chat):
            jid = str(row.get("jid") or "")
            if jid == uid or _jid_user_part(jid) == \
                    _jid_user_part(uid):
                role = str(row.get("role") or "")
                status = {"superadmin": "creator",
                          "admin": "administrator"}.get(
                              role, "member")
                return {"ok": True, "status": status,
                        "role": role, "jid": jid,
                        "name": str(row.get("name") or "")}
        return {"ok": False,
                "error": "member not in the group roster "
                         "(or the bridge is unreachable)"}

    def admin_ban(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        """Ban = remove from the group (see the seam note above:
        WhatsApp has no ban list; removal is the whole verb)."""
        return self._members_update_checked(chat, "remove", user_id)

    def admin_unban(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        """Unban: nothing to lift — WhatsApp keeps no ban list,
        so a removed member is already free to rejoin through
        the group's invite link.  Vacuous success by design, and
        deliberately NOT a re-add: the runtime's kick is
        ban+unban, and a re-adding unban would undo every kick."""
        return {"ok": True, "action": "unban",
                "note": "WhatsApp keeps no ban list — a removed "
                        "member may rejoin via the invite link"}

    def admin_promote(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        """Promote a member to group admin."""
        return self._members_update_checked(chat, "promote", user_id)

    def admin_demote(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        """Demote a group admin back to member."""
        return self._members_update_checked(chat, "demote", user_id)

    def admin_restrict(self, chat: ChatRef, user_id: str,
                       minutes: int = 60,
                       until: int | None = None) -> dict[str, Any]:
        """Mute — UNSUPPORTED on WhatsApp, honestly (see the seam
        note above): there is no per-member mute to apply, and
        neither removal nor group-wide admins-only mode honors
        this verb's timed, per-member, reversible contract.  The
        refusal carries the reason; the runtime audits it as a
        non-execution, never as a mute that happened."""
        return {"ok": False,
                "error": "unsupported: WhatsApp has no per-member "
                         "mute — a member can only be removed, or "
                         "the whole group set to admins-only; "
                         "neither is a mute, so nothing was applied"}

    def admin_unrestrict(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        """Unmute — equally unsupported: no per-member mute
        exists on WhatsApp, so there is none to lift."""
        return {"ok": False,
                "error": "unsupported: WhatsApp has no per-member "
                         "mute, so there is no mute to lift"}

    def admin_delete(self, chat: ChatRef, message_id: str) -> dict[str, Any]:
        """Delete a message for everyone (protocol revoke).  The
        bridge resolves the author's message key from its own
        message memory; a message it never saw is an honest
        refusal, not a guessed key."""
        mid = str(message_id or "").strip()
        if not mid:
            return {"ok": False, "error": "no message id given"}
        return self._admin_cmd("message_delete", {
            "chat": self._group_jid(chat), "message_id": mid,
        })

    def admin_pin(self, chat: ChatRef, message_id: str,
                  silent: bool = True) -> dict[str, Any]:
        """Pin a message in the group.  Real on both bridge
        engines — the protocol's pin-in-chat message: whatsmeow
        builds it directly, and Baileys' sendMessage {pin}
        builds the same message from the bridge's own cached
        key (a message the bridge never saw refuses honestly
        on either engine, since its key cannot be built)."""
        mid = str(message_id or "").strip()
        if not mid:
            return {"ok": False, "error": "no message id given"}
        return self._admin_cmd("message_pin", {
            "chat": self._group_jid(chat), "message_id": mid,
        })

    def admin_unpin(self, chat: ChatRef,
                    message_id: str = "") -> dict[str, Any]:
        """Unpin.  WhatsApp unpins a SPECIFIC message (there is
        no 'unpin the pinned one' verb without its key), so the
        bare runtime call — which names no message — is an
        honest unsupported; with a message id the bridge
        message_unpin verb does the real thing (both engines)."""
        mid = str(message_id or "").strip()
        if not mid:
            return {"ok": False,
                    "error": "unsupported: WhatsApp unpins a "
                             "specific message — no message id "
                             "was given, so there is nothing to "
                             "unpin"}
        return self._admin_cmd("message_unpin", {
            "chat": self._group_jid(chat), "message_id": mid,
        })

    def admin_join_requests(self, chat: ChatRef) -> dict[str, Any]:
        """Pending join requests at an approval-mode group's
        door: ``{"ok": True, "requests": [{"jid", "name",
        "requested_at"}]}`` — §28's discovery read.  WhatsApp
        reports only who asked and when; the name rides along
        only when the bridge's contact store knows one (the Go
        engine), else "".  A group whose approval mode is off
        simply has nobody waiting — an empty list, not an
        error.  A refusal (bridge down, an older bridge without
        the verb, no admin standing) is an honest ok=False: the
        runtime reads it as "the door can't be seen", never as
        "nobody is waiting"."""
        if not self.connected.is_set():
            return {"ok": False,
                    "error": "whatsapp bridge not connected"}
        response = self._send_cmd(
            {"cmd": "group_join_requests",
             "chat": self._group_jid(chat)}, timeout=20)
        if not isinstance(response, dict) \
                or not response.get("ok"):
            return {"ok": False,
                    "error": str(
                        (response or {}).get("error")
                        or "bridge gave no answer")}
        out = []
        for entry in response.get("requests") or []:
            jid = str(entry.get("jid") or "")
            if not jid:
                continue
            try:
                ts = float(entry.get("requested_at") or 0)
            except (TypeError, ValueError):
                ts = 0.0
            out.append({"jid": jid,
                        "name": str(entry.get("name") or ""),
                        "requested_at": ts})
        return {"ok": True, "requests": out}

    def admin_join_request(self, chat: ChatRef, user_id: str,
                           approve: bool) -> dict[str, Any]:
        """Decide one pending join request (DESIGN §28) — the
        Telegram contract exactly: approve admits the requester;
        decline refuses them (they may request again — a decline
        is not a ban).  The requester's own per-user status
        decides, as with member updates: a batch ok cannot hide
        this requester's refusal."""
        jid = self._member_jid(chat, user_id)
        if not jid:
            return {"ok": False,
                    "error": f"no WhatsApp identity for {user_id!r}"}
        action = "approve" if approve else "reject"
        out = self._admin_cmd("group_join_request_decide", {
            "chat": self._group_jid(chat), "action": action,
            "participants": [jid],
        }, timeout=45)
        if not isinstance(out, dict) or out.get("ok") is False:
            return out if isinstance(out, dict) else \
                {"ok": False, "error": "bridge gave no answer"}
        return self._per_user_verdict(out, jid, action)

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

    #: The exactly-once window for channel posts: a post's
    #: deterministic stanza id derives from (channel, content,
    #: this window), so an accidental double-fire of the same
    #: post inside the window collapses onto ONE wire id (the
    #: bridges use a caller-supplied id verbatim), while a
    #: deliberate repost of the same words later gets a fresh
    #: id and posts again.
    _CHANNEL_POST_WINDOW = 600.0

    def _channel_post_id(self, jid: str, text: str,
                         media_path: str) -> str:
        """The deterministic stanza id for one channel post."""
        material = str(media_path or "")
        if media_path:
            try:
                st = os.stat(media_path)
                material = f"{media_path}:{st.st_size}:{st.st_mtime_ns}"
            except OSError:
                material = str(media_path)
        window = int(time.time() // self._CHANNEL_POST_WINDOW)
        digest = hashlib.sha256(
            f"{jid}\n{text}\n{material}\n{window}".encode()
        ).hexdigest()
        return digest[:24].upper()

    def channel_send(self, chat: ChatRef | str, text: str = "", *,
                     media_path: str = "",
                     caption: str = "") -> dict[str, Any]:
        """Post to a channel as the owner (channel stories
        build — the /chanstory send half): the bridges'
        ``channel_send`` verb.  On WhatsApp this posts a
        channel POST — it lands in the channel feed.
        WhatsApp's Channel Status (the Status-tray stories
        that expire in 24h) is a distinct feature neither
        bridge library exposes (status mining, 2026-11-07):
        a named wall, never something this verb dresses a
        post up as.  Text, or an image /
        video already on disk with an optional caption.  The
        verdict carries the stanza ``message_id`` and, on the
        Go engine, the post's ``server_id`` — its public
        identity (the number in its link); Baileys' send result
        exposes none, so the field is simply absent there,
        never invented.  Posting requires channel admin on the
        account; WhatsApp refuses a non-admin server-side and
        that refusal comes back in the verdict verbatim."""
        jid = self._newsletter_jid(chat)
        if not jid.endswith("@newsletter"):
            return {"ok": False,
                    "error": f"not a channel JID: {jid or '(empty)'} "
                             f"(channels end in @newsletter)"}
        text = str(text or "").strip()
        media_path = str(media_path or "").strip()
        if not text and not media_path:
            return {"ok": False,
                    "error": "channel_send needs text or a media path"}
        payload: dict[str, Any] = {
            "chat": jid, "text": text, "path": media_path,
            "caption": str(caption or ""),
            "message_id": self._channel_post_id(jid, text, media_path),
        }
        return self._admin_cmd("channel_send", payload, timeout=45)

    def channel_view_posts(self, chat: ChatRef | str,
                           server_ids: list[str]) -> dict[str, Any]:
        """Mark channel posts viewed as the owner (channel
        stories build — the view half): the bridges'
        ``channel_view`` verb.  On the Go engine that is
        whatsmeow's NewsletterMarkViewed, the view receipt
        that moves a post's view counter; the Baileys rung has
        no mark-viewed verb in its library and answers an
        honest ok:false naming the gap — the caller surfaces
        whichever truth came back, never a fake view."""
        jid = self._newsletter_jid(chat)
        ids = [str(s or "").strip() for s in (server_ids or [])]
        ids = [s for s in ids if s]
        if not ids:
            return {"ok": False,
                    "error": "channel_view needs at least one "
                             "server_id (the number at the end of "
                             "the channel post link)"}
        return self._admin_cmd("channel_view", {
            "chat": jid, "server_ids": ids}, timeout=20)

    def fetch_channel_posts(self, channels: Any = None, *,
                            count: int = 10) -> list[dict[str, Any]]:
        """Recent posts from channels this account follows —
        the feed the personal surface's channel cycle saves
        and views (channel stories build).  Row shape (both
        engines): ``{poster_id, poster_name, story_id, text,
        path, kind, media_missing, views, ts}`` where story_id
        is the post's server_id — the view/react key.

        Enumeration: the Go engine's verb enumerates the
        followed channels itself (GetSubscribedNewsletters);
        the Baileys library has no enumeration verb, so when
        its no-chat call refuses, the channels this surface
        already knows (its recent channel chats) are fetched
        one by one.  *channels* pins the list explicitly on
        either engine.  Any failure reads as an empty feed —
        absence, never invented posts."""
        count = max(1, min(50, int(count or 10)))
        jids: list[str] = []
        for c in (channels or []):
            jid = self._newsletter_jid(c)
            if jid.endswith("@newsletter") and jid not in jids:
                jids.append(jid)
        if not jids:
            res = self._admin_cmd("channel_fetch", {"count": count},
                                  timeout=45)
            if res.get("ok"):
                return self._channel_post_rows(res)
            # No enumeration on this engine: fall back to the
            # channel chats this surface already knows.
            for row in self.chats(limit=500):
                if str(row.get("kind") or "") == "channel":
                    jid = str(row.get("id") or "")
                    if jid.endswith("@newsletter") \
                            and jid not in jids:
                        jids.append(jid)
        out: list[dict[str, Any]] = []
        for jid in jids:
            res = self._admin_cmd("channel_fetch", {
                "chat": jid, "count": count}, timeout=45)
            if res.get("ok"):
                out.extend(self._channel_post_rows(res))
        return out

    @staticmethod
    def _channel_post_rows(res: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in (res.get("posts") or []):
            if not isinstance(row, dict):
                continue
            out.append({
                "poster_id": str(row.get("chat") or ""),
                "poster_name": str(row.get("name") or ""),
                "story_id": str(row.get("server_id") or ""),
                "text": str(row.get("text") or ""),
                "path": str(row.get("path") or ""),
                "kind": str(row.get("kind") or ""),
                "media_missing": bool(row.get("media_missing")),
                "views": int(row.get("views") or 0),
                "ts": float(row.get("ts") or 0) / 1000.0,
            })
        return out

    def channel_info(self, chat: ChatRef | str) -> dict[str, Any]:
        """Read a channel's newsletter metadata through the
        bridge's ``channel_info``: name, description, and the
        follower count the platform exposes.  ``subscribers``
        is an int when the metadata carries one and stays None
        when it doesn't — absence, never a fabricated 0.  A
        bridge without the verb (an older build) answers
        ok:false, and that refusal is the result."""
        res = self._admin_cmd("channel_info",
                              {"chat": self._newsletter_jid(chat)})
        if not res.get("ok"):
            return {"ok": False,
                    "error": str(res.get("error")
                                 or "channel_info failed")}
        subs = res.get("subscribers")
        try:
            subs = int(subs) if subs is not None else None
        except (TypeError, ValueError):
            subs = None
        return {"ok": True,
                "name": str(res.get("name") or ""),
                "description": str(res.get("description") or ""),
                "subscribers": subs}

    def admin_member_count(self, chat: ChatRef) -> dict[str, Any]:
        """The runtime's channel-sampling contract (the same
        ``{"ok", "count"}`` shape the Telegram bot adapter
        answers): the one count WhatsApp exposes to a surface
        is a channel's follower count, read from its newsletter
        metadata.  The follower LIST is never enumerable on
        WhatsApp (platform fact) — a count is all there is.
        Groups expose no count verb (their rosters are read
        member by member) and DMs have nothing to count, so
        both answer ok:False and callers render ABSENCE —
        sampled-never is not a 0."""
        if not self._jid(chat).endswith("@newsletter"):
            return {"ok": False,
                    "error": "whatsapp exposes a member count "
                             "only for channels (newsletter "
                             "metadata)"}
        info = self.channel_info(chat)
        if not info.get("ok"):
            return {"ok": False,
                    "error": str(info.get("error") or "")}
        if info.get("subscribers") is None:
            return {"ok": False,
                    "error": "the channel metadata carried no "
                             "follower count"}
        return {"ok": True, "count": int(info["subscribers"])}

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
                    message_id=str(entry.get("id") or ""),
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
