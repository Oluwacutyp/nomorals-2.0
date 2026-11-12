"""Cross-surface command claims (owner order 2026-11-05).

One Telegram message reaches Devon once per attached surface:
the bot gets it through the Bot API, the owner's userbot gets
the same message through MTProto (his own sends arrive as
own-echoes). Before this ledger, EVERY surface that parsed a
command handled it — in the owner's claimed group his
``/rmdeleted@Nomorals_bot`` was executed by the bot AND answered
by the as-him surface with the §33 gate text (which read a
different claim record and lied about the room), and one
``/play`` in the bot's DM minted two jobs, one per surface.

The law: a recognized command is claimed by exactly one
surface; every other surface that sees it stands down silently
(persist/observe only, never a reply). The runtime consults
this ledger at the command checkpoints of ``_process_inner``
— the first surface to claim a (platform-family, chat,
message) key owns the command; a claim repeated by the SAME
surface (the dispatch re-checks at several branches of one
turn) is still its own. Claims are keyed on the platform
MESSAGE identity, which is the same on every surface that
delivers the message (Telegram message ids are per-chat), with
the platform family collapsed (``telegram`` / ``telegram-bot``
/ ``telegram-personal`` are one family — one room, one chat).

The 2026-11-11 seam (owner screenshot, same morning): the
owner's ``/chatmode games-only`` in the personal bot's DM was
answered TWICE — once by the bot, once as-him — because the
two mechanisms above each stop at the runtime instance's edge:

* this ledger was in-memory, one per ``PartnerRuntime`` —
  surfaces that do not share an instance share no claims;
* the key embeds the SURFACE-PERSPECTIVE chat id: in a DM the
  bot's copy keys on the OWNER's user id while the userbot's
  echo keys on the BOT's account id, so the two copies could
  never collide on one key, in any ledger;
* the remaining DM guard (a bot's DM belongs to the bot) keys
  on adapter-carried companion ids — state that lives on
  adapter objects inside one runtime instance.

The fix keeps the law and moves the seam: claims live in the
home store (:class:`DurableCommandClaimLedger` — every surface
of Devon's, whatever process or instance hosts it, claims
against the same table), and the key is canonical
(:func:`command_claim_key`): a Telegram-family DM keys on the
sorted pair of BOTH participant ids, which each surface knows
from its own side (its own account id + the peer it sees), so
the bot's copy and the as-him echo name the same claim.
Groups/channels and other platforms keep the original key
shape — their chat ids are already perspective-independent.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

__all__ = [
    "CommandClaimLedger",
    "DurableCommandClaimLedger",
    "DurableGroupClaimStore",
    "WINDOW_S",
    "MAX_KEYS",
    "command_claim_key",
]

_log = logging.getLogger(__name__)

#: The duplicate deliveries of one message land seconds apart;
#: a slow bot poll is the long pole. Ten minutes matches the
#: runtime's bot-claim probe TTL — past it, a delivery is a new
#: event (adapter replays are held by the stale-replay gate).
WINDOW_S = 600.0
MAX_KEYS = 1024


class CommandClaimLedger:
    """Bounded, thread-safe record of which surface claimed
    which command message."""

    def __init__(self, *, window_s: float = WINDOW_S) -> None:
        self._window_s = float(window_s)
        self._lock = threading.Lock()
        # claim key -> (surface, ts)
        self._claims: OrderedDict[str, tuple[str, float]] = \
            OrderedDict()

    def claim(self, key: str, surface: str, *,
              now: float | None = None) -> bool:
        """Try to own ``key`` for ``surface``.

        True when this surface owns the command — a fresh claim,
        or its own still-live claim repeated. False when another
        surface holds a live claim: stand down silently. Never
        raises: claim bookkeeping must not eat a turn."""
        key = str(key or "")
        surface = str(surface or "")
        if not key or not surface:
            return True  # unkeyable: legacy single-surface path
        ts = time.time() if now is None else float(now)
        with self._lock:
            held = self._claims.get(key)
            if held is not None:
                holder, held_ts = held
                if ts - held_ts <= self._window_s:
                    if holder == surface:
                        self._claims.move_to_end(key)
                        return True
                    return False
            self._claims[key] = (surface, ts)
            self._claims.move_to_end(key)
            while len(self._claims) > MAX_KEYS:
                self._claims.popitem(last=False)
            return True

    def claimed_by(self, key: str, *,
                   now: float | None = None) -> str:
        """The surface holding a live claim on ``key``, or ""."""
        key = str(key or "")
        if not key:
            return ""
        ts = time.time() if now is None else float(now)
        with self._lock:
            held = self._claims.get(key)
            if held is None:
                return ""
            holder, held_ts = held
            if ts - held_ts > self._window_s:
                return ""
            return holder


def command_claim_key(message: Any, own_account_id: str = "") -> str:
    """The canonical claim key for one command message, or ""
    when the message is unkeyable (no platform message id — the
    legacy unclaimed path, never an invented identity).

    The original shape is ``family:chat_id:message_id``. For a
    Telegram-family DM that shape is perspective-split: the bot
    sees the chat as the OWNER's user id, the owner's userbot
    sees the same chat as the BOT's account id, and the two
    copies of one command could never collide on it. A DM has
    exactly two participants and every surface knows both ids
    — its own account id and the peer in front of it — so the
    DM key names the sorted pair:
    ``telegram:dm:<lo>:<hi>:<message_id>``, identical from both
    sides. When the surface cannot name its own account id the
    key falls back to the original shape (the bot-DM peer rule
    and the legacy behaviour still apply — never worse than
    before). Groups, channels, the self-chat (own id == chat
    id) and every other platform keep the original shape:
    their chat ids are perspective-independent.
    """
    message_id = str(getattr(message, "message_id", "") or "")
    if not message_id:
        return ""
    chat = getattr(message, "chat", None)
    platform = str(getattr(chat, "platform", "") or "")
    chat_id = str(getattr(chat, "chat_id", "") or "")
    if not platform or not chat_id:
        return ""
    family = "telegram" if platform.startswith("telegram") else platform
    own = str(own_account_id or "").strip()
    kind = str(getattr(chat, "kind", "") or "")
    if (family == "telegram" and kind == "dm" and own
            and own != chat_id):
        lo, hi = sorted((own, chat_id))
        return f"telegram:dm:{lo}:{hi}:{message_id}"
    return f"{family}:{chat_id}:{message_id}"


_CLAIMS_DDL = """
CREATE TABLE IF NOT EXISTS command_claims (
  claim_key TEXT PRIMARY KEY,
  surface TEXT NOT NULL DEFAULT '',
  claimed_at REAL NOT NULL DEFAULT 0
);
"""


class DurableCommandClaimLedger:
    """The command-claim law over the shared home store.

    Same contract as :class:`CommandClaimLedger` — ``claim``
    returns True when this surface owns the key (a fresh claim
    or its own still-live claim), False when another surface
    holds a live claim — but the record lives in the home
    database (runtime-created table, the Batch-B pattern: DDL
    in code, migrations untouched), so surfaces hosted by
    DIFFERENT runtime instances or processes claim against the
    same record. That is the seam the 2026-11-11 duplicate
    crossed: the in-memory ledger arbitrates only inside one
    instance.

    Claims expire after the window: a surface that claimed and
    then crashed mid-handling mutes its twins for at most
    ``WINDOW_S``, after which a late copy may claim and handle
    — never a permanent mute. The claim is atomic across
    processes (insert-or-nothing, then a compare-and-swap steal
    of an expired row). Never raises: a store fault degrades to
    the legacy unclaimed behaviour (proceed) and is logged —
    claim bookkeeping must not eat a turn.
    """

    def __init__(self, db: Any, *, window_s: float = WINDOW_S) -> None:
        self._db = db
        self._window_s = float(window_s)
        self._lock = threading.Lock()
        self._broken = False
        self._last_sweep = 0.0
        try:
            self._db.executescript(_CLAIMS_DDL)
        except Exception:  # noqa: BLE001 - degraded, never fatal
            _log.warning("command claims: DDL failed; claims "
                         "degrade to unclaimed", exc_info=True)
            self._broken = True

    # -- internals ----------------------------------------------------
    def _row(self, key: str) -> tuple[str, float] | None:
        row = self._db.query_one(
            "SELECT surface, claimed_at FROM command_claims"
            " WHERE claim_key = ?", (key,))
        if row is None:
            return None
        return str(row["surface"] or ""), float(row["claimed_at"] or 0)

    def _sweep(self, ts: float) -> None:
        """Drop expired rows, throttled — the table stays tiny
        and expired rows are stealable even unswept."""
        if ts - self._last_sweep < 60.0:
            return
        self._last_sweep = ts
        try:
            self._db.execute(
                "DELETE FROM command_claims WHERE claimed_at < ?",
                (ts - self._window_s,))
        except Exception:  # noqa: BLE001 - housekeeping only
            _log.debug("command claims: sweep failed", exc_info=True)

    def _steal_expired(self, key: str, surface: str, ts: float,
                       held_ts: float) -> bool:
        """Compare-and-swap an expired row to this surface."""
        cursor = self._db.execute(
            "UPDATE command_claims SET surface = ?, claimed_at = ?"
            " WHERE claim_key = ? AND claimed_at = ?",
            (surface, ts, key, held_ts))
        return bool(getattr(cursor, "rowcount", 0))

    # -- the claim contract -------------------------------------------
    def claim(self, key: str, surface: str, *,
              now: float | None = None) -> bool:
        """Try to own ``key`` for ``surface`` (see the class
        docstring). True = this surface owns the command."""
        key = str(key or "")
        surface = str(surface or "")
        if not key or not surface or self._broken:
            return True  # unkeyable / degraded: legacy path
        ts = time.time() if now is None else float(now)
        with self._lock:
            try:
                self._sweep(ts)
                row = self._row(key)
                if row is not None:
                    holder, held_ts = row
                    if ts - held_ts <= self._window_s:
                        return holder == surface
                    if self._steal_expired(key, surface, ts, held_ts):
                        return True
                    row = self._row(key)  # lost the steal race
                    if row is None:
                        return True
                    holder, held_ts = row
                    return (holder == surface
                            and ts - held_ts <= self._window_s)
                cursor = self._db.execute(
                    "INSERT INTO command_claims"
                    " (claim_key, surface, claimed_at)"
                    " VALUES (?, ?, ?)"
                    " ON CONFLICT(claim_key) DO NOTHING",
                    (key, surface, ts))
                if getattr(cursor, "rowcount", 0):
                    return True
                row = self._row(key)  # a twin inserted first
                if row is None:
                    return True
                holder, held_ts = row
                if ts - held_ts > self._window_s and self._steal_expired(
                        key, surface, ts, held_ts):
                    return True
                return holder == surface
            except Exception:  # noqa: BLE001 - never eats a turn
                _log.warning("command claims: claim failed for %s; "
                             "proceeding unclaimed", key,
                             exc_info=True)
                return True

    def claimed_by(self, key: str, *,
                   now: float | None = None) -> str:
        """The surface holding a live claim on ``key``, or ""."""
        key = str(key or "")
        if not key or self._broken:
            return ""
        ts = time.time() if now is None else float(now)
        try:
            row = self._row(key)
        except Exception:  # noqa: BLE001 - a probe, never fatal
            _log.debug("command claims: claimed_by failed",
                       exc_info=True)
            return ""
        if row is None:
            return ""
        holder, held_ts = row
        if ts - held_ts > self._window_s:
            return ""
        return holder


_GROUP_CLAIMS_DDL = """
CREATE TABLE IF NOT EXISTS group_claims (
  group_key TEXT PRIMARY KEY,
  bot_admin INTEGER,
  bot_jid TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL DEFAULT 0
);
"""


class DurableGroupClaimStore:
    """The room's §33 standing, durable in the home store.

    The group surface's JSON store is per-runtime: two runtimes
    on split homes hold SPLIT claim state for one group, and the
    bot's verified admin standing — stamped by whichever runtime
    probed it — never reaches the other surface. This table is
    the same Batch-B seam as the command claims: one row per
    group key, the tri-state bot_admin (1/0/NULL — verified,
    verified-demoted, unproven), the JID of the bot account the
    stamp belongs to, and the stamp time. The group surface
    merges it with its local records (same tri-state rule: True
    wins, then False, then None), so a probe that lands in a
    SHARED home store is one truth for every runtime that reads
    it — and a runtime that restarts keeps the standing instead
    of re-proving it.

    The honest limit, stated here and in the surface: on split
    homes (the live EC2 shape) this table is still per-home —
    it unifies the record's SHAPE and fixes the same-home case,
    but the live split-home arbitration is carried by the
    in-band defer-and-observe channel, not by this table. A
    store fault degrades to "unproven" (None) and is logged —
    claim bookkeeping must not eat a turn.
    """

    def __init__(self, db: Any) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._broken = False
        try:
            self._db.executescript(_GROUP_CLAIMS_DDL)
        except Exception:  # noqa: BLE001 - degraded, never fatal
            _log.warning("group claims: DDL failed; claims degrade "
                         "to unproven", exc_info=True)
            self._broken = True

    def _row(self, group_key: str) -> tuple[Any, str, float] | None:
        row = self._db.query_one(
            "SELECT bot_admin, bot_jid, updated_at FROM group_claims"
            " WHERE group_key = ?", (group_key,))
        if row is None:
            return None
        admin = row["bot_admin"]
        return (None if admin is None else bool(admin),
                str(row["bot_jid"] or ""),
                float(row["updated_at"] or 0))

    def note_bot_admin(self, group_key: str, is_admin: bool,
                       bot_jid: str | None = None) -> None:
        """Stamp the verified standing. A stamp that changes
        nothing (same standing, same JID) does not write. A
        demotion keeps the last known bot JID — the identity is
        still the sibling's, even when its standing lapsed."""
        group_key = str(group_key or "")
        if not group_key or self._broken:
            return
        ts = time.time()
        with self._lock:
            try:
                row = self._row(group_key)
                jid = str(bot_jid or "")
                if row is not None and jid == "":
                    jid = row[1]
                if row is not None and row[0] is bool(is_admin) \
                        and row[1] == jid:
                    return
                self._db.execute(
                    "INSERT INTO group_claims"
                    " (group_key, bot_admin, bot_jid, updated_at)"
                    " VALUES (?, ?, ?, ?)"
                    " ON CONFLICT(group_key) DO UPDATE SET"
                    " bot_admin = excluded.bot_admin,"
                    " bot_jid = excluded.bot_jid,"
                    " updated_at = excluded.updated_at",
                    (group_key, 1 if is_admin else 0, jid, ts))
            except Exception:  # noqa: BLE001 - degraded, never fatal
                _log.warning("group claims: note failed for %s",
                             group_key, exc_info=True)

    def bot_admin(self, group_key: str) -> bool | None:
        """The room's verified bot standing, or None when
        unproven (no record, or a store fault)."""
        group_key = str(group_key or "")
        if not group_key or self._broken:
            return None
        try:
            row = self._row(group_key)
        except Exception:  # noqa: BLE001 - a probe, never fatal
            _log.debug("group claims: read failed for %s",
                       group_key, exc_info=True)
            return None
        if row is None:
            return None
        return row[0]

    def record(self, group_key: str) -> dict[str, Any] | None:
        """The whole durable record for the room, or None when
        the room has no durable standing yet."""
        group_key = str(group_key or "")
        if not group_key or self._broken:
            return None
        try:
            row = self._row(group_key)
        except Exception:  # noqa: BLE001 - a probe, never fatal
            _log.debug("group claims: read failed for %s",
                       group_key, exc_info=True)
            return None
        if row is None:
            return None
        return {"bot_admin": row[0], "bot_jid": row[1],
                "updated_at": row[2]}
