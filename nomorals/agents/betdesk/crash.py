"""Crash-game round records + storage for the betting desk.

One normalized record for every crash-style game (Spribe Aviator,
Stake Originals Crash, bustabit, and the likes), whatever book or
provider served it: the desk studies ONE shape, providers are data.

Storage is an append-only JSONL ledger per desk directory (the
motion-ledger posture from pass #82: appends never rewrite history,
dedupe happens on the (provider, game, round_id) key at read time so
a re-collected round can never double-count in the lab).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

__all__ = [
    "CrashRound", "CrashStore", "betdesk_dir",
]


def betdesk_dir() -> str:
    """Desk data directory: ``NOMORALS_BETDESK_DIR`` or the config home
    next to the sports analyst's store."""
    override = os.environ.get("NOMORALS_BETDESK_DIR", "").strip()
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".config", "nomorals",
                        "betdesk")


@dataclass
class CrashRound:
    """One settled crash-game round.

    ``ts`` is the round's own timestamp (epoch seconds, UTC) — the
    time hypothesis lives or dies on real round times, so a round
    without a timestamp is stored but excluded from every time test.
    ``crash`` is the multiplier the round busted at (>= 1.00; 1.00 is
    the instant bust).  ``game_hash`` / ``seed_reveal`` carry whatever
    provably-fair material the provider publishes (hash of the round,
    the revealed seed/salt after the fact) — empty when it publishes
    none.
    """

    provider: str            # book / venue id, e.g. "stake", "bustabit"
    game: str                # game id, e.g. "aviator", "crash"
    round_id: str            # provider's own round identifier
    ts: float = 0.0          # epoch seconds; 0 = unknown
    crash: float = 0.0       # bust multiplier, >= 1.00
    game_hash: str = ""      # provably-fair hash for this round
    prev_hash: str = ""      # hash this round's hash chains from, if any
    seed_reveal: str = ""    # revealed seed/salt for this round, if any
    source: str = ""         # how the desk got it (collector id / import)

    @property
    def key(self) -> tuple:
        return (self.provider, self.game, self.round_id)

    @property
    def instant_bust(self) -> bool:
        return self.crash < 1.01

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CrashRound":
        fields = {k: v for k, v in (d or {}).items()
                  if k in cls.__dataclass_fields__}
        return cls(**fields)


class CrashStore:
    """Append-only round ledger (JSONL) with key-based dedupe on read."""

    def __init__(self, desk_dir: str = ""):
        self.dir = desk_dir or betdesk_dir()
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "crash_rounds.jsonl")

    # ── writes ────────────────────────────────────────────────────────

    def append(self, rounds: Iterable[CrashRound]) -> dict[str, int]:
        """Append rounds, skipping keys already in the ledger.

        Returns {"added": n, "duplicates": m}.  Existing keys are read
        once per call — collection batches stay cheap.
        """
        known = {r.key for r in self._iter_file()}
        added = dup = 0
        fresh: list[CrashRound] = []
        for r in rounds:
            if r.key in known:
                dup += 1
                continue
            known.add(r.key)
            fresh.append(r)
            added += 1
        if fresh:
            with open(self.path, "a", encoding="utf-8") as fh:
                for r in fresh:
                    fh.write(json.dumps(r.to_dict()) + "\n")
        return {"added": added, "duplicates": dup}

    # ── reads ─────────────────────────────────────────────────────────

    def _iter_file(self) -> Iterable[CrashRound]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield CrashRound.from_dict(json.loads(line))
                except (ValueError, TypeError):
                    continue  # a torn last line never kills the ledger

    def rounds(self, provider: str = "", game: str = "") -> list[CrashRound]:
        """All rounds (deduped, oldest first by ts where known)."""
        seen: set = set()
        out: list[CrashRound] = []
        for r in self._iter_file():
            if r.key in seen:
                continue
            seen.add(r.key)
            if provider and r.provider != provider:
                continue
            if game and r.game != game:
                continue
            out.append(r)
        out.sort(key=lambda r: (r.ts or 0.0, r.round_id))
        return out

    def games(self) -> list[tuple[str, str, int]]:
        """[(provider, game, round_count)] across the whole ledger."""
        counts: dict[tuple[str, str], int] = {}
        for r in self.rounds():
            counts[(r.provider, r.game)] = counts.get((r.provider, r.game), 0) + 1
        return sorted((p, g, n) for (p, g), n in counts.items())

    def count(self) -> int:
        return sum(n for _, _, n in self.games())
