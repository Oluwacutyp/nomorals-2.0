"""Market candle records + storage for the markets desk.

One normalized record for every price bar the desk studies, whatever
exchange or feed served it: the desk studies ONE shape, providers are
data.  A candle carries its own timestamp (epoch seconds, UTC) — the
time hypothesis lives or dies on real bar times, so a bar without a
timestamp is never stored.

Storage is an append-only JSONL ledger per desk directory (the
motion-ledger posture from pass #82: appends never rewrite history,
dedupe happens on the (symbol, timeframe, ts) key at read time so a
re-collected bar can never double-count in the lab).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "Candle", "CandleStore", "marketdesk_dir",
]


def marketdesk_dir() -> str:
    """Desk data directory: ``NOMORALS_MARKETDESK_DIR`` or the config
    home next to the betting desk's store."""
    override = os.environ.get("NOMORALS_MARKETDESK_DIR", "").strip()
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".config", "nomorals",
                        "marketdesk")


@dataclass
class Candle:
    """One price bar.

    ``ts`` is the bar's own open time (epoch seconds, UTC).  ``close``
    is the bar's settlement price; ``volume`` is 0.0 when the feed
    publishes none.  ``source`` names how the desk got it (feed id).
    """

    symbol: str            # e.g. "BTCUSD", "AAPL"
    timeframe: str         # "1d" | "1h"
    ts: float = 0.0        # epoch seconds; 0 = unknown (never stored)
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    source: str = ""       # how the desk got it (feed id)

    @property
    def key(self) -> tuple:
        return (self.symbol, self.timeframe, self.ts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Candle:
        fields = {k: v for k, v in (d or {}).items()
                  if k in cls.__dataclass_fields__}
        return cls(**fields)


class CandleStore:
    """Append-only candle ledger (JSONL) with key-based dedupe on read."""

    def __init__(self, desk_dir: str = ""):
        self.dir = desk_dir or marketdesk_dir()
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "candles.jsonl")

    # ── writes ────────────────────────────────────────────────────────

    def append(self, candles: Iterable[Candle]) -> dict[str, int]:
        """Append candles, skipping keys already in the ledger.

        Returns {"added": n, "duplicates": m}.  Existing keys are read
        once per call — collection batches stay cheap.  A candle with
        no timestamp or no usable close is refused, not stored: the
        lab cannot study a bar it cannot place in time or price.
        """
        known = {r.key for r in self._iter_file()}
        added = dup = refused = 0
        fresh: list[Candle] = []
        for r in candles:
            if not r.ts or not (r.close and math.isfinite(r.close)):
                refused += 1
                continue
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
        return {"added": added, "duplicates": dup, "refused": refused}

    # ── reads ─────────────────────────────────────────────────────────

    def _iter_file(self) -> Iterable[Candle]:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield Candle.from_dict(json.loads(line))
                    except (ValueError, TypeError):
                        continue  # a torn last line never kills the ledger

    def bars(self, symbol: str = "", timeframe: str = "") -> list[Candle]:
        """All bars (deduped, oldest first), optionally filtered."""
        seen: set = set()
        out: list[Candle] = []
        for r in self._iter_file():
            if r.key in seen:
                continue
            seen.add(r.key)
            if symbol and r.symbol != symbol:
                continue
            if timeframe and r.timeframe != timeframe:
                continue
            out.append(r)
        out.sort(key=lambda r: r.ts)
        return out

    def symbols(self) -> list[str]:
        """Symbols held, in first-seen order."""
        seen: list[str] = []
        for r in self.bars():
            if r.symbol not in seen:
                seen.append(r.symbol)
        return seen

    def count(self) -> int:
        return len(self.bars())
