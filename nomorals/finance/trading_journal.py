"""Append-only decision journal for Devon's own trading system.

One journal every trading-system pass writes to: Pass 1 records
``market_read`` snapshots here; later passes add setup, judgment and
forecast events beside them, so validation can grade decisions against
the reads they were taken from. This is the *decision* record — the
desk's own ``trading_journal.jsonl`` (orders/fills, see
:mod:`nomorals.finance.trading_desk`) stays the execution record.

Storage is a JSONL file beside the desk journal, resolved through
:func:`nomorals.finance.budgets.finance_paths` — the same pattern the
desk uses, so no storage migration is needed and the file rides the
runtime settings home. Records are plain dicts with an ``event`` kind
and a ``ts`` epoch; readers tolerate torn/corrupt lines (an append
interrupted by a crash loses at most its own record, never the file).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .budgets import finance_paths

_log = logging.getLogger(__name__)

JOURNAL_FILENAME = "trading_system_journal.jsonl"


def journal_path(settings: Any = None) -> Path:
    """The system journal path (beside the desk's order journal)."""
    _ledger_path, budgets_path = finance_paths(settings)
    return Path(budgets_path).parent / JOURNAL_FILENAME


def append_event(event: str, *, settings: Any = None,
                 path: str | Path | None = None,
                 **fields: Any) -> dict[str, Any]:
    """Append one event record; returns the record as written."""
    record: dict[str, Any] = {"event": str(event), "ts": time.time()}
    record.update(fields)
    target = Path(path) if path is not None else journal_path(settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return record


def read_events(*, limit: int = 50, event: str | None = None,
                settings: Any = None,
                path: str | Path | None = None) -> list[dict[str, Any]]:
    """Journal records, newest first; ``event`` filters by kind.

    Missing file reads as empty; unparseable lines are skipped (they
    are surfaced in the log, never silently treated as history).
    """
    target = Path(path) if path is not None else journal_path(settings)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            _log.warning("trading_journal: skipping unparseable line")
            continue
        if isinstance(rec, dict):
            records.append(rec)
    if event is not None:
        records = [r for r in records if r.get("event") == event]
    if limit > 0:
        records = records[-limit:]
    return records[::-1]


def latest_event(event: str, *, settings: Any = None,
                 path: str | Path | None = None) -> dict[str, Any] | None:
    """The newest record of one event kind, or None.

    Scans the journal from the end (newest first) and stops at the
    first match — it never parses the whole file. The journal is
    append-only and time-ordered, so the newest record of a kind is
    the first match from the end. Callers that only need the latest
    record (e.g. the pipeline tick's already-ran check) must use
    this, not :func:`read_events` with ``limit=0``: that parses every
    line of an unbounded file on every call, and the parse cost grows
    with the journal (the 2026-11-14 trading-pipeline timeout spent
    ~15s in ``json.loads`` on this path alone).
    """
    target = Path(path) if path is not None else journal_path(settings)
    if not target.exists():
        return None
    with target.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        # Walk backwards in chunks; a record is never larger than one
        # chunk in practice (market-read snapshots are tens of KB),
        # and the loop keeps going while no match is found, so a
        # larger record is still found — just more slowly.
        chunk = 65536
        pos = size
        tail = b""
        while pos > 0:
            step = min(chunk, pos)
            pos -= step
            fh.seek(pos)
            tail = fh.read(step) + tail
            lines = tail.split(b"\n")
            # The first piece is a partial line (it continues in the
            # earlier chunk) — hold it for the next round; the rest
            # are complete lines, newest first.
            tail = lines[0]
            for raw in reversed(lines[1:]):
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line.decode("utf-8"))
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get("event") == event:
                    return rec
        # The held first piece is the file's first line.
        line = tail.strip()
        if line:
            try:
                rec = json.loads(line.decode("utf-8"))
            except ValueError:
                rec = None
            if isinstance(rec, dict) and rec.get("event") == event:
                return rec
    return None
