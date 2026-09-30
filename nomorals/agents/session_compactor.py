"""Session compaction - compress old conversation history to save context.

As conversations grow long, the context window fills up and performance
degrades. Session compaction summarizes old messages while preserving
key information, decisions, and context.

Features:
- Sliding window: keep recent N messages verbatim, summarize older ones
- Extract key decisions, facts, and action items
- Preserve file paths, code snippets, and important references
- Incremental compaction (only compact new messages)
- Multiple compaction strategies (LLM summary, extractive, hybrid)

Usage:
    compactor = SessionCompactor(db, llm)
    
    # Compact a session
    summary = await compactor.compact(session_id, keep_recent=20)
    
    # Get compacted context for LLM
    context = await compactor.get_context(session_id, max_tokens=4000)
    
    # Manual compaction of specific messages
    summary = await compactor.summarize_messages(message_ids)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..llm.base import LLMProvider
from ..storage.db import Database

__all__ = [
    "SessionCompactor",
    "CompactedSession",
    "CompactionStrategy",
]

_log = get_logger(__name__)


class CompactionStrategy:
    """Compaction strategies."""
    
    LLM_SUMMARY = "llm_summary"      # Use LLM to summarize
    EXTRACTIVE = "extractive"        # Extract key sentences
    HYBRID = "hybrid"                # Combine both
    SLIDING_WINDOW = "sliding"       # Keep recent, drop old


@dataclass
class CompactedSession:
    """A compacted session with summary and recent messages."""
    
    session_id: str
    summary: str
    key_facts: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    file_references: list[str] = field(default_factory=list)
    code_snippets: list[str] = field(default_factory=list)
    compacted_at: float = field(default_factory=time.time)
    messages_compacted: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "summary": self.summary,
            "key_facts": self.key_facts,
            "decisions": self.decisions,
            "action_items": self.action_items,
            "file_references": self.file_references,
            "messages_compacted": self.messages_compacted,
        }
    
    def to_context_string(self, max_tokens: int = 4000) -> str:
        """Convert to context string for LLM."""
        parts = []
        
        # Summary
        parts.append(f"SESSION SUMMARY:\n{self.summary}\n")
        
        # Key facts
        if self.key_facts:
            parts.append("KEY FACTS:")
            for fact in self.key_facts[:10]:  # Limit
                parts.append(f"- {fact}")
            parts.append("")
        
        # Decisions
        if self.decisions:
            parts.append("DECISIONS MADE:")
            for decision in self.decisions[:5]:
                parts.append(f"- {decision}")
            parts.append("")
        
        # Action items
        if self.action_items:
            parts.append("ACTION ITEMS:")
            for item in self.action_items[:5]:
                parts.append(f"- {item}")
            parts.append("")
        
        # File references
        if self.file_references:
            parts.append("FILES REFERENCED:")
            for f in self.file_references[:10]:
                parts.append(f"- {f}")
            parts.append("")
        
        # Recent messages (most recent first, then reverse for chronological)
        if self.recent_messages:
            parts.append("RECENT MESSAGES:")
            for msg in self.recent_messages[-20:]:  # Last 20
                role = msg.get("role", "user")
                content = msg.get("content", "")[:500]  # Limit per message
                parts.append(f"[{role}]: {content}")
        
        context = "\n".join(parts)
        
        # Rough token estimate (1 token ≈ 4 chars)
        estimated_tokens = len(context) // 4
        if estimated_tokens > max_tokens:
            # Truncate to fit
            max_chars = max_tokens * 4
            context = context[:max_chars] + "\n[... truncated ...]"
        
        return context


class SessionCompactor:
    """Compacts session history to save context space."""
    
    def __init__(
        self,
        db: Database,
        llm: LLMProvider,
        *,
        strategy: str = CompactionStrategy.HYBRID,
    ) -> None:
        self.db = db
        self.llm = llm
        self.strategy = strategy
        self._ensure_schema()
        _log.info(f"SessionCompactor initialized with {strategy} strategy")
    
    def _ensure_schema(self) -> None:
        """Create compaction tables."""
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS session_compactions (
                    compaction_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_facts TEXT NOT NULL DEFAULT '[]',
                    decisions TEXT NOT NULL DEFAULT '[]',
                    action_items TEXT NOT NULL DEFAULT '[]',
                    file_references TEXT NOT NULL DEFAULT '[]',
                    code_snippets TEXT NOT NULL DEFAULT '[]',
                    messages_compacted INTEGER NOT NULL DEFAULT 0,
                    last_message_id TEXT NOT NULL DEFAULT '',
                    compacted_at REAL NOT NULL
                )
            """)
            
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_compaction_session
                ON session_compactions(session_id)
            """)
    
    async def compact(
        self,
        session_id: str,
        *,
        keep_recent: int = 20,
        force: bool = False,
    ) -> CompactedSession:
        """Compact a session's message history.
        
        Args:
            session_id: Session to compact
            keep_recent: Number of recent messages to keep verbatim
            force: Force recompaction even if already compacted
            
        Returns:
            CompactedSession with summary and recent messages
        """
        # Get all messages
        messages = self.db.query("""
            SELECT id AS message_id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
        """, (session_id,))
        
        if not messages:
            return CompactedSession(session_id=session_id, summary="Empty session")
        
        # Check if already compacted
        if not force:
            existing = self.db.query_one("""
                SELECT * FROM session_compactions
                WHERE conversation_id = ?
                ORDER BY compacted_at DESC
                LIMIT 1
            """, (session_id,))
            
            if existing:
                # Check if there are new messages since last compaction
                last_compacted_id = existing["last_message_id"]
                last_idx = next(
                    (i for i, m in enumerate(messages) if m["message_id"] == last_compacted_id),
                    -1
                )
                
                new_messages = messages[last_idx + 1:] if last_idx >= 0 else messages
                
                if len(new_messages) < keep_recent:
                    # Not enough new messages to warrant recompaction
                    _log.info(f"Session {session_id} already compacted, skipping")
                    return self._load_compaction(existing, messages[-keep_recent:])
        
        # Split into to-compact and to-keep
        if len(messages) <= keep_recent:
            # Nothing to compact
            return CompactedSession(
                session_id=session_id,
                summary="Session too short to compact",
                recent_messages=[dict(m) for m in messages],
            )
        
        to_compact = messages[:-keep_recent]
        to_keep = messages[-keep_recent:]
        
        _log.info(f"Compacting {len(to_compact)} messages, keeping {len(to_keep)} recent")
        
        # Generate summary
        if self.strategy == CompactionStrategy.LLM_SUMMARY:
            summary_data = self._summarize_llm(to_compact)
        elif self.strategy == CompactionStrategy.EXTRACTIVE:
            summary_data = self._summarize_extractive(to_compact)
        else:  # HYBRID
            extractive = self._summarize_extractive(to_compact)
            llm_summary = self._summarize_llm(to_compact)
            summary_data = self._merge_summaries(extractive, llm_summary)
        
        # Create compaction record
        compaction_id = new_id("compact")
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO session_compactions
                (compaction_id, session_id, summary, key_facts, decisions,
                 action_items, file_references, code_snippets,
                 messages_compacted, last_message_id, compacted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                compaction_id,
                session_id,
                summary_data["summary"],
                json.dumps(summary_data["key_facts"]),
                json.dumps(summary_data["decisions"]),
                json.dumps(summary_data["action_items"]),
                json.dumps(summary_data["file_references"]),
                json.dumps(summary_data["code_snippets"]),
                len(to_compact),
                to_compact[-1]["message_id"],
                time.time(),
            ))
        
        return CompactedSession(
            session_id=session_id,
            summary=summary_data["summary"],
            key_facts=summary_data["key_facts"],
            decisions=summary_data["decisions"],
            action_items=summary_data["action_items"],
            recent_messages=[dict(m) for m in to_keep],
            file_references=summary_data["file_references"],
            code_snippets=summary_data["code_snippets"],
            messages_compacted=len(to_compact),
        )
    
    async def get_context(
        self,
        session_id: str,
        *,
        max_tokens: int = 4000,
        keep_recent: int = 20,
    ) -> str:
        """Get compacted context for LLM.
        
        Args:
            session_id: Session to get context for
            max_tokens: Maximum tokens for context
            keep_recent: Number of recent messages to keep
            
        Returns:
            Formatted context string
        """
        compacted = await self.compact(session_id, keep_recent=keep_recent)
        return compacted.to_context_string(max_tokens=max_tokens)
    
    async def summarize_messages(
        self,
        message_ids: list[str],
    ) -> dict[str, Any]:
        """Summarize specific messages.
        
        Args:
            message_ids: List of message IDs to summarize
            
        Returns:
            Summary data dict
        """
        placeholders = ",".join("?" * len(message_ids))
        messages = self.db.query(f"""
            SELECT id AS message_id, role, content, created_at
            FROM messages
            WHERE id IN ({placeholders})
            ORDER BY created_at ASC
        """, message_ids)
        
        if not messages:
            return {"summary": "No messages found", "key_facts": [], "decisions": [], "action_items": []}
        
        return self._summarize_llm(messages)
    
    def _summarize_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate summary using LLM."""
        # Prepare message text
        message_text = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"][:1000]  # Limit per message
            message_text.append(f"[{role}]: {content}")
        
        conversation = "\n".join(message_text)
        
        prompt = f"""Summarize this conversation, extracting:
1. Overall summary (2-3 sentences)
2. Key facts mentioned (user preferences, technical details, etc.)
3. Decisions made
4. Action items or TODOs
5. File paths and code references

CONVERSATION:
{conversation}

Return as JSON:
```json
{{
    "summary": "Brief summary",
    "key_facts": ["fact1", "fact2"],
    "decisions": ["decision1"],
    "action_items": ["action1"],
    "file_references": ["path/to/file.py"],
    "code_snippets": ["brief code description"]
}}
```
"""
        
        response = self.llm.chat([{"role": "user", "content": prompt}])
        
        # Parse JSON response
        import re
        json_match = re.search(r"```json\s*(.*?)\s*```", response.content, re.DOTALL)
        
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                _log.warning("Failed to parse LLM summary JSON")
        
        # Fallback: use raw response as summary
        return {
            "summary": response.content[:500],
            "key_facts": [],
            "decisions": [],
            "action_items": [],
            "file_references": [],
            "code_snippets": [],
        }
    
    def _summarize_extractive(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Extractive summary - pull key sentences and references."""
        import re
        
        all_text = " ".join(msg["content"] for msg in messages)
        
        # Extract file paths
        file_pattern = r'\b[\w/]+\.(?:py|js|ts|jsx|tsx|json|md|txt)\b'
        files = list(set(re.findall(file_pattern, all_text)))
        
        # Extract decisions (sentences with "decided", "will", "going to")
        decision_pattern = r'(?:decided|will|going to|should|let\'s)\s+[^.]+'
        decisions = re.findall(decision_pattern, all_text, re.IGNORECASE)
        
        # Extract action items (sentences with "TODO", "need to", "must")
        action_pattern = r'(?:TODO|need to|must|should)\s+[^.]+'
        actions = re.findall(action_pattern, all_text, re.IGNORECASE)
        
        # Key facts: first and last messages, plus any with numbers/dates
        key_facts = []
        if messages:
            key_facts.append(messages[0]["content"][:200])
            if len(messages) > 1:
                key_facts.append(messages[-1]["content"][:200])
        
        # Summary: concatenate key sentences
        summary_parts = []
        if files:
            summary_parts.append(f"Discussed files: {', '.join(files[:5])}")
        if decisions:
            summary_parts.append(f"Decisions: {'; '.join(decisions[:3])}")
        if actions:
            summary_parts.append(f"Actions: {'; '.join(actions[:3])}")
        
        summary = ". ".join(summary_parts) if summary_parts else "Conversation with no clear decisions or actions"
        
        return {
            "summary": summary,
            "key_facts": key_facts,
            "decisions": decisions[:5],
            "action_items": actions[:5],
            "file_references": files[:10],
            "code_snippets": [],
        }
    
    def _merge_summaries(
        self,
        extractive: dict[str, Any],
        llm: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge extractive and LLM summaries."""
        return {
            "summary": llm.get("summary", extractive["summary"]),
            "key_facts": list(set(
                extractive.get("key_facts", []) + llm.get("key_facts", [])
            ))[:10],
            "decisions": list(set(
                extractive.get("decisions", []) + llm.get("decisions", [])
            ))[:5],
            "action_items": list(set(
                extractive.get("action_items", []) + llm.get("action_items", [])
            ))[:5],
            "file_references": list(set(
                extractive.get("file_references", []) + llm.get("file_references", [])
            ))[:10],
            "code_snippets": llm.get("code_snippets", []),
        }
    
    def _load_compaction(
        self,
        row: dict[str, Any],
        recent_messages: list[dict[str, Any]],
    ) -> CompactedSession:
        """Load a compaction from database row."""
        return CompactedSession(
            session_id=row["session_id"],
            summary=row["summary"],
            key_facts=json.loads(row["key_facts"]),
            decisions=json.loads(row["decisions"]),
            action_items=json.loads(row["action_items"]),
            recent_messages=[dict(m) for m in recent_messages],
            file_references=json.loads(row["file_references"]),
            code_snippets=json.loads(row["code_snippets"]),
            compacted_at=row["compacted_at"],
            messages_compacted=row["messages_compacted"],
        )
