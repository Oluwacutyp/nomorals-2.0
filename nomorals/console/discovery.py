"""Command discovery: fuzzy search + usage-ranked help, generated from
the registry — never from a hardcoded list.

The registry starts EMPTY. Commands get in only by being registered
(``register_command``) or loaded from a live parser
(:meth:`CommandRegistry.load_from_console`, which reads the existing
console command table at runtime). ``help_all()`` and ``help_for()``
are generated from whatever is registered at call time, so help can
never drift from the real command set.
"""

from __future__ import annotations

import difflib
import threading
from dataclasses import dataclass, field


@dataclass
class CommandEntry:
    name: str
    help: str = ""
    usage_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "help": self.help,
                "usage_count": self.usage_count}


def _fuzzy_score(query: str, name: str) -> float:
    """0.0–1.0 match quality: substring/subsequence beats pure similarity."""
    q, n = query.lower(), name.lower()
    if not q:
        return 0.0
    if q == n:
        return 1.0
    if n.startswith(q):
        return 0.95
    if q in n:
        return 0.85
    # Subsequence (letters in order, e.g. "db" matches "dashboard").
    it = iter(n)
    if all(ch in it for ch in q):
        return 0.7
    return difflib.SequenceMatcher(None, q, n).ratio() * 0.6


class CommandRegistry:
    """The single source of truth for console command help + discovery."""

    def __init__(self) -> None:
        self._entries: dict[str, CommandEntry] = {}
        self._lock = threading.RLock()

    # ── registration ──

    def register_command(self, name: str, help: str = "",
                         usage_count: int = 0) -> CommandEntry:
        """Add (or update the help of) a command. Returns its entry."""
        name = str(name).strip().lower()
        if not name:
            raise ValueError("command name must not be empty")
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                entry = CommandEntry(name=name, help=str(help),
                                     usage_count=int(usage_count))
                self._entries[name] = entry
            else:
                if help:
                    entry.help = str(help)
                entry.usage_count = max(entry.usage_count, int(usage_count))
            return entry

    def record_use(self, name: str) -> None:
        """Bump a command's usage count (drives ranking in search)."""
        with self._lock:
            entry = self._entries.get(str(name).strip().lower())
            if entry is not None:
                entry.usage_count += 1

    def load_from_console(self) -> int:
        """Register every command from the live console command table.

        Reads ``nomorals.console.commands._COMMANDS`` at runtime — the
        table itself is the source, so nothing here is a hardcoded copy.
        Returns the number of commands registered. No-op (0) if the
        table cannot be imported.
        """
        try:
            from .commands import _COMMANDS
        except Exception:
            return 0
        count = 0
        for name, (usage, desc) in _COMMANDS.items():
            self.register_command(name, f"{usage} — {desc}")
            count += 1
        return count

    # ── discovery ──

    def search(self, query: str, *, limit: int = 8) -> list[str]:
        """Fuzzy-match command names; best match first.

        Ranking: fuzzy score first, then usage count (a heavily-used
        near-match beats a rarely-used one). Empty query returns the
        most-used commands.
        """
        q = (query or "").strip().lower()
        with self._lock:
            entries = list(self._entries.values())
        if not q:
            ranked = sorted(entries, key=lambda e: (-e.usage_count, e.name))
            return [e.name for e in ranked[:limit]]
        scored = []
        for e in entries:
            score = _fuzzy_score(q, e.name)
            if score > 0.25:  # noise floor: unrelated names stay out
                scored.append((score, e.usage_count, e.name))
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        return [name for _, _, name in scored[:limit]]

    def search_entries(self, query: str, *, limit: int = 8) -> list[CommandEntry]:
        return [self._entries[n] for n in self.search(query, limit=limit)
                if n in self._entries]

    def complete(self, prefix: str) -> list[str]:
        """Prefix completion candidates (for Tab completion)."""
        p = (prefix or "").lower()
        with self._lock:
            return sorted(n for n in self._entries if n.startswith(p))

    # ── help, generated from the registry ──

    def help_for(self, name: str) -> str | None:
        """``/why``-style help for one command, or None if unknown."""
        with self._lock:
            entry = self._entries.get(str(name).strip().lower())
        if entry is None:
            return None
        used = f" (used {entry.usage_count}x)" if entry.usage_count else ""
        return f"{entry.name}{used}\n  {entry.help or '(no help text)'}"

    def help_all(self) -> str:
        """Full help text, generated from the registry at call time."""
        with self._lock:
            entries = sorted(self._entries.values(), key=lambda e: e.name)
        if not entries:
            return "no commands registered"
        lines = ["console commands:"]
        for e in entries:
            used = f" [x{e.usage_count}]" if e.usage_count else ""
            lines.append(f"  {e.name}{used}  {e.help}".rstrip())
        return "\n".join(lines)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return str(name).strip().lower() in self._entries


__all__ = ["CommandEntry", "CommandRegistry"]
