"""Character memory: full MemoryManager depth for character agents.

``Character.memory`` is a flat list with word-overlap recall — fine for a
sketch, wrong for a person. This module gives each character a scoped
view into the real MemoryManager: episodic/semantic tiers, FTS + vector
hybrid recall, contradiction detection, proactive surfacing, and delivery
scoring.

The flat list stays as the fallback when no database is available, so
characters work in every environment (tests, headless, minimal installs).

Usage:
    mem = CharacterMemory.for_character(char, db)  # db: Database | None
    mem.remember("Zara promised to call back tomorrow", importance=0.7)
    mem.recall("promises Zara made")  # -> list[str], ranked
    mem.proactive("incoming message text")  # -> list[str] worth surfacing
"""
from __future__ import annotations

import time
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["CharacterMemory"]


def _scope_for(char_id: str) -> str:
    return f"character:{char_id}"


class _ShimContext:
    """Minimal context MemoryManager needs: .db and .settings."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.settings = None
        self.suggest = None


class CharacterMemory:
    """Scoped MemoryManager view for one character.

    When ``db`` is None (or manager construction fails), every method
    degrades to the character's flat ``memory`` list — never raises.
    """

    def __init__(self, char: Any, db: Any | None = None) -> None:
        self.char = char
        self.scope = _scope_for(getattr(char, "id", "unknown"))
        self._manager: Any | None = None
        if db is not None:
            try:
                from ..memory.manager import MemoryManager
                self._manager = MemoryManager(_ShimContext(db))
            except Exception as exc:  # noqa: BLE001 — flat fallback
                _log.debug("character memory backend unavailable: %s", exc)
                self._manager = None

    @classmethod
    def for_character(cls, char: Any, db: Any | None = None) -> "CharacterMemory":
        return cls(char, db)

    @property
    def deep(self) -> bool:
        """True when the full MemoryManager backend is live."""
        return self._manager is not None

    # ── remember ─────────────────────────────────────────────────────
    def remember(self, text: str, importance: float = 0.5,
                 kind: str = "episode") -> None:
        text = (text or "").strip()
        if not text:
            return
        if self._manager is not None:
            try:
                self._manager.remember(
                    text, kind=kind, importance=importance,
                    scope=self.scope,
                )
                return
            except Exception as exc:  # noqa: BLE001
                _log.debug("character deep remember failed: %s", exc)
        # flat fallback — mirrors Character.remember
        mem = getattr(self.char, "memory", None)
        if isinstance(mem, list):
            mem.append({"ts": time.time(), "text": text[:500],
                        "salience": max(0.0, min(1.0, importance))})

    # ── recall ───────────────────────────────────────────────────────
    def recall(self, query: str, limit: int = 5) -> list[str]:
        if self._manager is not None:
            try:
                result = self._manager.recall(
                    query, limit=limit, scope=self.scope)
                records = getattr(result, "records", []) or []
                out = [str(getattr(r, "text", "")).strip() for r in records]
                return [t for t in out if t][:limit]
            except Exception as exc:  # noqa: BLE001
                _log.debug("character deep recall failed: %s", exc)
        # flat fallback — three-factor scoring (Stanford gold), mirrors
        # Character.recall so both paths rank the same way
        try:
            return self.char.recall(query, limit=limit)
        except Exception:
            return []

    # ── consolidate ──────────────────────────────────────────────────
    def consolidate(self) -> dict[str, int]:
        """Background tidy: dedupe near-duplicates, merge the trivial,
        keep the list honest. MIRIX/Mem0 gold — memory quality is
        retrieval quality, and retrieval drowns in duplicates.

        Returns {"merged": n, "dropped": n}. Never raises.
        """
        stats = {"merged": 0, "dropped": 0}
        try:
            mem = getattr(self.char, "memory", None)
            if not isinstance(mem, list) or len(mem) < 2:
                return stats
            if self._manager is not None:
                # deep backend owns its lifecycle; only trim the flat mirror
                return stats
            seen: dict[str, dict[str, Any]] = {}
            kept: list[dict[str, Any]] = []
            for m in mem:
                if not isinstance(m, dict):
                    continue
                text = str(m.get("text", "")).strip()
                if not text:
                    stats["dropped"] += 1
                    continue
                key = " ".join(text.lower().split())
                try:
                    sal = float(m.get("salience", 0.5))
                except (TypeError, ValueError):
                    sal = 0.5
                if key in seen:
                    # near-duplicate: keep the stronger salience, newer ts
                    old = seen[key]
                    try:
                        old_sal = float(old.get("salience", 0.5))
                    except (TypeError, ValueError):
                        old_sal = 0.5
                    if sal > old_sal:
                        old["salience"] = sal
                    try:
                        if float(m.get("ts", 0)) > float(old.get("ts", 0)):
                            old["ts"] = m.get("ts")
                    except (TypeError, ValueError):
                        pass
                    stats["merged"] += 1
                else:
                    seen[key] = m
                    kept.append(m)
            # drop the truly trivial when the list is crowded
            if len(kept) > 80:
                kept.sort(key=lambda m: float(m.get("salience", 0.5) or 0.5))
                before = len(kept)
                kept = kept[-80:]
                stats["dropped"] += before - len(kept)
            mem[:] = kept
        except Exception as exc:  # noqa: BLE001
            _log.debug("character consolidate failed: %s", exc)
        return stats

    # ── proactive ────────────────────────────────────────────────────
    def proactive(self, message_text: str, limit: int = 2) -> list[str]:
        """Memories worth surfacing unprompted for this character."""
        if self._manager is not None:
            try:
                from ..memory.proactive import surface
                return surface(message_text, self._manager,
                               limit=limit, origin=self.scope)
            except Exception as exc:  # noqa: BLE001
                _log.debug("character proactive failed: %s", exc)
        return []

    # ── contradictions ──────────────────────────────────────────────
    def check_contradiction(self, text: str) -> list[dict[str, Any]]:
        """Find existing memories that contradict a new statement.

        Remembers the text first (as a candidate), runs detection, and
        returns contradictions. The candidate stays — resolution is
        additive per the memory system's rules.
        """
        if self._manager is not None:
            try:
                from ..memory.contradictions import detect_for
                rec_id = self._manager.remember(
                    text, kind="episode", importance=0.5,
                    scope=self.scope,
                )
                # remember() returns the id or a record — normalize
                rid = getattr(rec_id, "id", rec_id)
                if isinstance(rid, str):
                    found = detect_for(self._manager, rid)
                    return [{"text": str(getattr(c, "text", "")),
                             "confidence": float(
                                 getattr(c, "confidence", 0.0))}
                            for c in (found or [])]
            except Exception as exc:  # noqa: BLE001
                _log.debug("character contradiction check failed: %s", exc)
        return []
