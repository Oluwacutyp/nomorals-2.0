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
