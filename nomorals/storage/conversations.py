"""The conversation log: ``conversations`` + ``messages`` + ``chats``.

This is the canonical store for Devon's chat history — the highest-volume
append stream in the central database (census §3, "Core, KV,
conversations") and the first domain ported in Mongo program P3. The
writers and readers that used to open-code SQL against these tables —
the partner brain's turn persistence, the CLI partner command, the chat
gateway's registry, the session bridge's handoff re-key, Devon's message
tools, the chat profiler, and the training collector — all go through
here now, so the domain has exactly one persistence implementation per
rung:

* **Native SQL** (default): the statements this module's callers have
  always run, moved here verbatim. The SQLite default's behavior is
  byte-for-byte the pre-port behavior.
* **Document seam**: when boot attached a store to the Database
  (:func:`~nomorals.storage.docstore.attach_document_store`), the same
  operations run over :class:`~nomorals.storage.rowstore.RowTable` —
  MongoDB (or the shadow pair) holds the documents, in carry-v1 shape.

Atomicity, stated plainly: the native path wraps the conversation-upsert
+ message-insert pair in one transaction; the seam path performs two
single-document writes (conversation first, then the message) — each
atomic, the pair not transactional, which is the guarantee the seam
advertises (``multi_doc_transactions`` is False on Mongo). A crash
between the two leaves an empty conversation row, never a message
pointing at a conversation that was never written.

Ordering: message ids are ULIDs (time-ordered), so where the native
queries tie-break on ``rowid`` the seam orders by ``_id`` — the same
order for ULID-keyed rows.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from .rowstore import RowTable, attached_rows

__all__ = ["ConversationLog"]


class ConversationLog:
    """Conversations, their messages, and the chat registry, over one db."""

    def __init__(self, db: Any) -> None:
        self.db = db

    def transaction(self):
        """Compose several log writes the way the native path composes
        them: one ``db.transaction()`` on the SQLite rung (nested calls
        inside the individual operations become savepoints, as they do
        today), a no-op context on the seam — where each write is a
        single-document operation and the composition is sequential,
        per the seam's advertised guarantee."""
        if self._conversations() is None:
            return self.db.transaction()
        return nullcontext()

    # ── seam tables (None on the SQLite default) ─────────────────────────
    def _conversations(self) -> RowTable | None:
        return attached_rows(self.db, "conversations")

    def _messages(self) -> RowTable | None:
        return attached_rows(self.db, "messages")

    def _chats(self) -> RowTable | None:
        return attached_rows(self.db, "chats")

    # ── conversations ────────────────────────────────────────────────────
    def ensure_conversation(self, conv_id: str, *, title: str, agent: str,
                            channel: str, now: float,
                            created_at: float = 0,
                            bump_title: bool = True) -> None:
        """Upsert a conversation row: first write creates it (with the
        given ``created_at`` — the historical writers pass 0), later
        writes bump ``updated_at``. With ``bump_title`` (the brain's
        statement) a non-empty incoming title replaces the stored one;
        without it (the CLI's statement) the title never changes after
        creation. Both historical statements are preserved exactly."""
        table = self._conversations()
        if table is None:
            with self.db.transaction():
                if bump_title:
                    self.db.execute(
                        """INSERT INTO conversations (id, title, agent, channel, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at,
                                                        title = CASE WHEN excluded.title != '' THEN excluded.title ELSE conversations.title END""",
                        (conv_id, title, agent, channel, created_at, now),
                    )
                else:
                    self.db.execute(
                        "INSERT INTO conversations (id, title, agent, channel, "
                        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(id) DO UPDATE SET updated_at = "
                        "excluded.updated_at",
                        (conv_id, title, agent, channel, created_at, now),
                    )
            return
        existing = table.get(conv_id)
        if existing is None:
            table.put({
                "id": conv_id, "title": title, "agent": agent,
                "channel": channel, "created_at": created_at,
                "updated_at": now,
            })
        else:
            fields: dict[str, Any] = {"updated_at": now}
            if bump_title and title:
                fields["title"] = title
            table.update(conv_id, fields)

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        table = self._conversations()
        if table is None:
            return self.db.query_one(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,))
        return table.get(conv_id)

    def conversations(self) -> list[dict[str, Any]]:
        table = self._conversations()
        if table is None:
            return self.db.query("SELECT * FROM conversations")
        return table.all()

    # ── messages ─────────────────────────────────────────────────────────
    def append_message(self, *, row_id: str, conversation_id: str,
                       role: str, content: str, name: str = "",
                       model: str = "", created_at: float) -> None:
        table = self._messages()
        if table is None:
            self.db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, name, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (row_id, conversation_id, role, content, name, model,
                 created_at),
            )
            return
        table.put({
            "id": row_id, "conversation_id": conversation_id, "role": role,
            "content": content, "name": name, "model": model,
            "created_at": created_at,
        })

    def history(self, conversation_id: str, *, limit: int) -> list[dict[str, Any]]:
        """The newest ``limit`` messages of a conversation, newest first."""
        table = self._messages()
        if table is None:
            return self.db.query(
                "SELECT * FROM messages WHERE conversation_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit))
        rows = table.find({"conversation_id": conversation_id})
        rows.sort(key=lambda r: (r.get("created_at") or 0, r["id"]),
                  reverse=True)
        return rows[:limit]

    def messages_for(self, conversation_id: str) -> list[dict[str, Any]]:
        """Every message of a conversation, oldest first."""
        table = self._messages()
        if table is None:
            rows = self.db.query(
                "SELECT * FROM messages WHERE conversation_id = ? "
                "ORDER BY created_at, rowid",
                (conversation_id,))
            return rows
        rows = table.find({"conversation_id": conversation_id})
        rows.sort(key=lambda r: (r.get("created_at") or 0, r["id"]))
        return rows

    def messages_after(self, conversation_id: str, ts: float, *,
                       limit: int) -> list[dict[str, Any]]:
        """Messages strictly newer than ``ts``, newest first."""
        table = self._messages()
        if table is None:
            return self.db.query(
                "SELECT * FROM messages WHERE conversation_id = ? "
                "AND created_at > ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, ts, limit))
        rows = table.find({"conversation_id": conversation_id,
                           "created_at": {"$gt": ts}})
        rows.sort(key=lambda r: (r.get("created_at") or 0, r["id"]),
                  reverse=True)
        return rows[:limit]

    def messages_since(self, since: float) -> list[dict[str, Any]]:
        """Every message at or after ``since``, ordered by conversation,
        then time (the training collector's export order; the native
        ``rowid`` tiebreak becomes the ULID ``id`` on the seam)."""
        table = self._messages()
        if table is None:
            return self.db.query(
                "SELECT * FROM messages WHERE created_at >= ? "
                "ORDER BY conversation_id, created_at, rowid",
                (since,))
        rows = table.find({"created_at": {"$gte": since}})
        rows.sort(key=lambda r: (r.get("conversation_id") or "",
                                 r.get("created_at") or 0, r["id"]))
        return rows

    def recent_messages(self, *, limit: int) -> list[dict[str, Any]]:
        """The newest ``limit`` messages across all conversations."""
        table = self._messages()
        if table is None:
            return self.db.query(
                "SELECT * FROM messages ORDER BY created_at DESC LIMIT ?",
                (limit,))
        rows = table.all()
        rows.sort(key=lambda r: (r.get("created_at") or 0, r["id"]),
                  reverse=True)
        return rows[:limit]

    def participant_counts(self, conversation_id: str, *,
                           limit: int) -> list[dict[str, Any]]:
        """``(name, n)`` counts of user-role messages per sender name,
        most prolific first (the chat profiler's aggregate)."""
        table = self._messages()
        if table is None:
            return self.db.query(
                "SELECT name, COUNT(*) AS n FROM messages "
                "WHERE conversation_id = ? AND role = 'user' "
                "GROUP BY name ORDER BY n DESC LIMIT ?",
                (conversation_id, limit))
        counts: dict[str, int] = {}
        for row in table.find({"conversation_id": conversation_id,
                               "role": "user"}):
            name = row.get("name")
            counts[name] = counts.get(name, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
        return [{"name": name, "n": n} for name, n in ranked[:limit]]

    def recent_user_contents(self, conversation_id: str, *,
                             limit: int) -> list[str]:
        """Contents of the newest user-role messages, newest first."""
        table = self._messages()
        if table is None:
            rows = self.db.query(
                "SELECT content FROM messages "
                "WHERE conversation_id = ? AND role = 'user' "
                "ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit))
            return [r.get("content") or "" for r in rows]
        rows = table.find({"conversation_id": conversation_id,
                           "role": "user"})
        rows.sort(key=lambda r: (r.get("created_at") or 0, r["id"]),
                  reverse=True)
        return [r.get("content") or "" for r in rows[:limit]]

    def rekey(self, old_conv: str, new_conv: str) -> None:
        """Move a conversation's rows to a new id (session handoff).

        The target row is created first (an existing target wins), then
        the messages re-key, then the old row goes — the native order,
        kept on both rungs.
        """
        conv_table = self._conversations()
        msg_table = self._messages()
        if conv_table is None or msg_table is None:
            with self.db.transaction():
                self.db.execute(
                    "INSERT INTO conversations (id, title, agent, channel,"
                    " summary, tokens, created_at, updated_at, metadata)"
                    " SELECT ?, title, agent, channel, summary, tokens,"
                    " created_at, updated_at, metadata FROM conversations"
                    " WHERE id = ? ON CONFLICT(id) DO NOTHING",
                    (new_conv, old_conv))
                self.db.execute(
                    "UPDATE messages SET conversation_id = ?"
                    " WHERE conversation_id = ?",
                    (new_conv, old_conv))
                self.db.execute(
                    "DELETE FROM conversations WHERE id = ?", (old_conv,))
            return
        source = conv_table.get(old_conv)
        if source is not None and conv_table.get(new_conv) is None:
            moved = dict(source)
            moved["id"] = new_conv
            conv_table.put(moved)
        for row in msg_table.find({"conversation_id": old_conv}):
            msg_table.update(row["id"], {"conversation_id": new_conv})
        conv_table.delete(old_conv)

    # ── chats registry ───────────────────────────────────────────────────
    def upsert_chat(self, *, chat_id: str, platform: str, platform_chat_id: str,
                    kind: str, title: str, peer: str, is_owner: int,
                    in_us: int, now: float) -> dict[str, Any] | None:
        """The gateway's chat-registry upsert: kind always updates;
        title/peer only when the incoming value is non-empty;
        ``last_active`` always moves. Returns the stored row."""
        table = self._chats()
        if table is None:
            with self.db.transaction():
                self.db.execute(
                    """INSERT INTO chats (id, platform, chat_id, kind, title, peer, is_owner, in_us, last_active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                         kind=excluded.kind,
                         title=CASE WHEN excluded.title != '' THEN excluded.title ELSE chats.title END,
                         peer=CASE WHEN excluded.peer != '' THEN excluded.peer ELSE chats.peer END,
                         last_active=excluded.last_active""",
                    (chat_id, platform, platform_chat_id, kind, title, peer,
                     is_owner, in_us, now),
                )
            return self.db.query_one(
                "SELECT * FROM chats WHERE id = ?", (chat_id,))
        existing = table.get(chat_id)
        if existing is None:
            table.put({
                "id": chat_id, "platform": platform,
                "chat_id": platform_chat_id, "kind": kind, "title": title,
                "peer": peer, "is_owner": is_owner, "in_us": in_us,
                "last_active": now,
            })
        else:
            fields: dict[str, Any] = {"kind": kind, "last_active": now}
            if title:
                fields["title"] = title
            if peer:
                fields["peer"] = peer
            table.update(chat_id, fields)
        return table.get(chat_id)

    def touch_chat(self, chat_id: str, now: float) -> None:
        table = self._chats()
        if table is None:
            self.db.execute(
                "UPDATE chats SET last_active = ? WHERE id = ?",
                (now, chat_id))
            return
        table.update(chat_id, {"last_active": now})

    def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        table = self._chats()
        if table is None:
            return self.db.query_one(
                "SELECT * FROM chats WHERE id = ?", (chat_id,))
        return table.get(chat_id)

    def latest_chat(self) -> dict[str, Any] | None:
        """The most recently active chat row, if any."""
        table = self._chats()
        if table is None:
            return self.db.query_one(
                "SELECT * FROM chats ORDER BY last_active DESC LIMIT 1")
        rows = table.all()
        if not rows:
            return None
        rows.sort(key=lambda r: r.get("last_active") or 0, reverse=True)
        return rows[0]
