"""``nm why`` — explain the last autonomous firing.

Every autonomous firing (a mission run, an organ tick, a presence
surface) should be explainable after the fact: *what* fired, on what
*evidence*, with what *model*, *what it did*, and *what happened*.
:class:`WhyStore` is that explanation log.

* :meth:`WhyStore.record` — append one firing explanation. Returns the
  firing id. When a file path is given, records are appended as JSONL
  (append-only, fsynced per record so a kill-9 loses at most the
  in-flight record).
* :meth:`WhyStore.why_last` — the most recent firing's explanation.
* :meth:`WhyStore.why_for` — the explanation for one firing id.
* :func:`why_command` — the ``nm why [id]`` handler text.

The honest-no-record contract: ``why_last()`` / ``why_for()`` /
``why_command()`` **always answer**. On an empty store or an unknown id
they return a dict / text that says so plainly
(``{"found": False, "reason": ...}``) — never an exception, never an
invented explanation for a firing that never happened.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["WhyStore", "why_command", "format_why"]


def _new_id() -> str:
    return f"fire-{uuid.uuid4().hex[:12]}"


class WhyStore:
    """Append-only log of autonomous firing explanations."""

    def __init__(self, path: str | Path | None = None, *,
                 clock: Callable[[], float] | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._clock = clock or time.time
        self._records: list[dict[str, Any]] = []
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn trailing line is skipped, not fatal
                if isinstance(record, dict) and record.get("id"):
                    self._records.append(record)
        except OSError as exc:
            _log.warning("why store unreadable (%s) — starting empty", exc)
            self._records = []

    def _append_file(self, record: dict[str, Any]) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def record(self, what_fired: str, *,
               evidence: Any = None,
               model: str = "",
               did: str = "",
               happened: str = "",
               firing_id: str | None = None,
               ts: float | None = None,
               **extra: Any) -> str:
        """Append one firing explanation. Returns its firing id."""
        record = {
            "id": str(firing_id or _new_id()),
            "ts": float(ts if ts is not None else self._clock()),
            "what_fired": str(what_fired or "unknown firing"),
            "evidence": evidence if evidence is not None else {},
            "model": str(model or ""),
            "did": str(did or ""),
            "happened": str(happened or ""),
        }
        for key, value in extra.items():
            record.setdefault(key, value)
        self._records.append(record)
        if self.path is not None:
            try:
                self._append_file(record)
            except OSError as exc:
                _log.warning("why store append failed: %s", exc)
        return record["id"]

    def why_last(self) -> dict[str, Any]:
        """The most recent firing — or an honest no-record answer."""
        if not self._records:
            return {"found": False,
                    "reason": "no autonomous firing has been recorded yet — "
                              "nothing has fired since this store started"}
        return {"found": True, **dict(self._records[-1])}

    def why_for(self, firing_id: str) -> dict[str, Any]:
        """The explanation for one firing id — or an honest no-record
        answer. Never raises for an unknown id."""
        for record in reversed(self._records):
            if record.get("id") == str(firing_id):
                return {"found": True, **dict(record)}
        return {"found": False,
                "id": str(firing_id),
                "reason": f"no firing recorded with id {str(firing_id)!r} — "
                          "it may predate this store, or never have fired"}

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return [dict(r) for r in self._records[-max(1, int(limit)):]]

    def __len__(self) -> int:
        return len(self._records)


def format_why(record: dict[str, Any]) -> str:
    """Render one why-record (or no-record answer) as ``nm why`` text."""
    if not record.get("found"):
        return ("❓ no record: " + str(record.get("reason") or "unknown")
                + "\n(nothing to explain — no such firing was recorded)")
    when = time.strftime("%Y-%m-%d %H:%M:%S",
                         time.localtime(float(record.get("ts") or 0)))
    lines = [
        f"🔍 why: {record.get('what_fired')}  (id {record.get('id')}, {when})",
        f"  evidence: {json.dumps(record.get('evidence'), ensure_ascii=False, default=str)[:600] or '(none)'}",
        f"  model: {record.get('model') or '(no model recorded — rule/heuristic firing)'}",
        f"  did: {record.get('did') or '(not recorded)'}",
        f"  happened: {record.get('happened') or '(not recorded)'}",
    ]
    return "\n".join(lines)


def why_command(store: WhyStore, firing_id: str | None = None) -> str:
    """The ``nm why [id]`` handler: explain the last firing, or the one
    named by ``firing_id``. Always answers, even for unknown ids."""
    if firing_id:
        return format_why(store.why_for(firing_id))
    return format_why(store.why_last())
