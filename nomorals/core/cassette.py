"""Generic cassette store — ONE record/replay mechanism for everything.

LLM calls (:mod:`nomorals.llm.cassette`) and L2 network I/O
(:mod:`nomorals.core.egress`) both record to and replay from this store.
Same file format (append-only JSONL), same first-write-wins semantics,
same canonical-key primitive — so a replay run is byte-identical
regardless of which plane produced the recording.

Record shape is caller-defined; the store only guarantees the record
carries its ``"key"``. LLM records look like
``{"key", "provider", "model", "response"}``; network records look like
``{"key", "kind": "net", "method", "url", "response": {...}}``. Both are
opaque dicts to this store.

Semantics (deliberate):

* **First-write-wins.** A duplicate key never overwrites the original
  recording — replays are stable across runs.
* **Lazy index.** No I/O until the first lookup/record.
* **Never raises.** A cassette failure degrades to a miss / dropped
  record; it must never break the live path it is observing.
* **No secrets.** Callers strip credentials before recording (egress
  drops ``Authorization``/``Cookie``; LLM records carry no secrets).
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "GenericCassetteStore",
    "canonical_json",
    "canonical_key",
]


def canonical_json(payload: Any) -> str:
    """Stable JSON for hashing: sorted keys, tight separators, no ASCII escape."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    )


def canonical_key(kind: str, payload: Any) -> str:
    """``sha256`` of ``kind`` + canonical JSON of ``payload``.

    The kind prefix keeps LLM keys, network keys, and future kinds in
    disjoint namespaces even when their payloads coincide.
    """
    material = f"{kind}\0{canonical_json(payload)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class GenericCassetteStore:
    """Append-only JSONL cassette store with a lazy in-memory index."""

    def __init__(self, directory: str | Path, *, filename: str = "cassettes.jsonl") -> None:
        self.directory = Path(directory)
        self.path = self.directory / filename
        self._lock = threading.RLock()
        self._index: dict[str, dict[str, Any]] | None = None

    # ── index ────────────────────────────────────────────────────────────
    def _load_index(self) -> dict[str, dict[str, Any]]:
        if self._index is not None:
            return self._index
        index: dict[str, dict[str, Any]] = {}
        try:
            if self.path.exists():
                with self.path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except ValueError:
                            continue
                        # First-write-wins: a duplicate key never
                        # overwrites the original recording.
                        index.setdefault(record.get("key", ""), record)
        except OSError:
            pass
        self._index = index
        return index

    def lookup(self, key: str) -> dict[str, Any] | None:
        """The stored record for ``key``, or None on a miss."""
        with self._lock:
            return self._load_index().get(key)

    # ── recording ────────────────────────────────────────────────────────
    def record(self, key: str, record: dict[str, Any]) -> bool:
        """Persist ``record`` under ``key``. False when already stored."""
        with self._lock:
            index = self._load_index()
            if key in index:
                return False
            stored = dict(record)
            stored["key"] = key
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(stored, ensure_ascii=False) + "\n")
            except OSError:
                return False
            index[key] = stored
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._load_index())

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Snapshot of all records (oldest first)."""
        with self._lock:
            return iter(list(self._load_index().values()))
