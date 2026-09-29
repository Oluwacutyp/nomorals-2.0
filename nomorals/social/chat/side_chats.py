"""Side chats - multiple persistent conversation threads.

Each side chat has its own memory, context, and topic. Users can spawn
unlimited side chats from the main conversation.

Usage:
    side_chats = SideChatManager(db, memory)
    
    # Create a new thread
    thread = await side_chats.create(
        user_id="user123",
        topic="Project Alpha",
        description="Discussion about Project Alpha"
    )
    
    # Send message in thread
    await side_chats.add_message(thread.thread_id, role="user", content="...")
    
    # Get thread history
    messages = await side_chats.get_history(thread.thread_id, limit=50)
    
    # List all threads for user
    threads = await side_chats.list_threads("user123")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ...core.ids import new_id
from ...core.logging_setup import get_logger
from ...storage.db import Database

__all__ = ["SideChatManager", "Thread", "ThreadMessage"]

_log = get_logger(__name__)


@dataclass
class ThreadMessage:
    """A message in a thread."""
    
    message_id: str
    thread_id: str
    role: str  # user, assistant, system
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class Thread:
    """A side chat thread."""
    
    thread_id: str
    user_id: str
    topic: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    message_count: int = 0
    is_archived: bool = False
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "topic": self.topic,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "is_archived": self.is_archived,
            "tags": self.tags,
        }


class SideChatManager:
    """Manages multiple persistent conversation threads."""
    
    def __init__(self, db: Database) -> None:
        self.db = db
        self._ensure_schema()
        _log.info("Side chat manager initialized")
    
    def _ensure_schema(self) -> None:
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    is_archived INTEGER NOT NULL DEFAULT 0,
                    tags TEXT NOT NULL DEFAULT '[]',
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS thread_messages (
                    message_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
                )
            """)
            
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_threads_user
                ON threads(user_id)
            """)
            
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_thread_messages_thread
                ON thread_messages(thread_id, timestamp)
            """)
    
    async def create(
        self,
        user_id: str,
        topic: str,
        *,
        description: str = "",
        tags: list[str] | None = None,
    ) -> Thread:
        """Create a new thread."""
        thread_id = new_id("thread")
        now = time.time()
        tags = tags or []
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO threads (thread_id, user_id, topic, description, created_at, updated_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (thread_id, user_id, topic, description, now, now, json.dumps(tags)))
        
        thread = Thread(
            thread_id=thread_id,
            user_id=user_id,
            topic=topic,
            description=description,
            tags=tags,
        )
        
        _log.info(f"Created thread: {topic} ({thread_id})")
        return thread
    
    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ThreadMessage:
        """Add a message to a thread."""
        message_id = new_id("msg")
        now = time.time()
        metadata = metadata or {}
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO thread_messages (message_id, thread_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (message_id, thread_id, role, content, now, json.dumps(metadata)))
            
            self.db.execute("""
                UPDATE threads SET updated_at = ?, message_count = message_count + 1 WHERE thread_id = ?
            """, (now, thread_id))
        
        return ThreadMessage(
            message_id=message_id,
            thread_id=thread_id,
            role=role,
            content=content,
            timestamp=now,
        )
    
    async def get_history(
        self,
        thread_id: str,
        *,
        limit: int = 50,
        before: float | None = None,
    ) -> list[ThreadMessage]:
        """Get thread message history."""
        query = "SELECT * FROM thread_messages WHERE thread_id = ?"
        params: list[Any] = [thread_id]
        
        if before:
            query += " AND timestamp < ?"
            params.append(before)
        
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        
        rows = self.db.query(query, params)
        
        return [
            ThreadMessage(
                message_id=row["message_id"],
                thread_id=row["thread_id"],
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]
    
    async def list_threads(
        self,
        user_id: str,
        *,
        include_archived: bool = False,
    ) -> list[Thread]:
        """List all threads for a user."""
        query = "SELECT * FROM threads WHERE user_id = ?"
        params: list[Any] = [user_id]
        
        if not include_archived:
            query += " AND is_archived = 0"
        
        query += " ORDER BY updated_at DESC"
        
        rows = self.db.query(query, params)
        
        return [
            Thread(
                thread_id=row["thread_id"],
                user_id=row["user_id"],
                topic=row["topic"],
                description=row["description"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
                is_archived=bool(row["is_archived"]),
                tags=json.loads(row["tags"]),
            )
            for row in rows
        ]
    
    async def archive(self, thread_id: str) -> bool:
        """Archive a thread."""
        with self.db.transaction():
            self.db.execute(
                "UPDATE threads SET is_archived = 1, updated_at = ? WHERE thread_id = ?",
                (time.time(), thread_id)
            )
        return True
    
    async def delete(self, thread_id: str) -> bool:
        """Delete a thread and all messages."""
        with self.db.transaction():
            self.db.execute("DELETE FROM thread_messages WHERE thread_id = ?", (thread_id,))
            self.db.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        return True
    
    async def search(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[ThreadMessage]:
        """Search across all threads for a user."""
        rows = self.db.query("""
            SELECT m.* FROM thread_messages m
            JOIN threads t ON m.thread_id = t.thread_id
            WHERE t.user_id = ? AND m.content LIKE ?
            ORDER BY m.timestamp DESC
            LIMIT ?
        """, (user_id, f"%{query}%", limit))
        
        return [
            ThreadMessage(
                message_id=row["message_id"],
                thread_id=row["thread_id"],
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]
