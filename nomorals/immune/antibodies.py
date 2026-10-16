"""Antibody store — immune memory.

In-memory store with optional JSON persistence (atomic tmp + os.replace,
like ``commitments/ledger.py``). Second exposure to a known pathogen is
caught at innate speed via a compiled-regex cache.

Rules enforced here, not by convention:

* **Reason required.** ``add`` without a reason raises :class:`ValueError`.
* **Versioning.** Re-adding an antibody with an existing id bumps its
  version and keeps the superseded copy in history.
* **Memory transfer.** :meth:`AntibodyStore.export_bundle` /
  :meth:`AntibodyStore.import_bundle` move immune memory across tissues
  (phone immunizes workstation): version + reason are preserved, and
  on conflict the HIGHER version wins.

The store is duck-typed: it accepts antibody dataclasses (e.g.
:class:`nomorals.immune.thymus.Antibody`), dicts, or any object with
the antibody fields as attributes. It imports nothing from its
siblings, so the package stays stdlib-only.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["AntibodyStore"]

_ANTIBODY_FIELDS = (
    "id",
    "pattern",
    "pathogen_class",
    "reason",
    "created_ts",
    "version",
    "status",
    "retire_reason",
)


@lru_cache(maxsize=512)
def _compile(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern)


def _to_dict(antibody: Any) -> Dict[str, Any]:
    if isinstance(antibody, dict):
        return copy.deepcopy(antibody)
    if is_dataclass(antibody):
        return copy.deepcopy(asdict(antibody))
    data: Dict[str, Any] = {}
    for field_name in _ANTIBODY_FIELDS:
        if hasattr(antibody, field_name):
            data[field_name] = copy.deepcopy(getattr(antibody, field_name))
    if not data:
        raise ValueError(f"cannot interpret {antibody!r} as an antibody")
    return data


def _require_reason(data: Dict[str, Any]) -> None:
    reason = data.get("reason")
    if not reason or not str(reason).strip():
        raise ValueError("antibody requires a reason — add without reason raises")


class AntibodyStore:
    """In-memory antibody store with optional atomic JSON persistence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        self._store: Dict[str, Dict[str, Any]] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        if self.path is not None and self.path.exists():
            self.load()

    # ── persistence ──────────────────────────────────────────────

    def save(self) -> None:
        if self.path is None:
            return
        payload = {
            "format": "immune.antibodies.v1",
            "saved_ts": time.time(),
            "antibodies": list(self._store.values()),
            "history": self._history,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            payload = json.load(fh)
        self._store = {
            str(ab["id"]): dict(ab) for ab in payload.get("antibodies", [])
        }
        self._history = {
            str(k): [dict(entry) for entry in v]
            for k, v in payload.get("history", {}).items()
        }

    # ── CRUD (no delete: immune memory is append-only) ───────────

    def add(self, antibody: Any) -> Dict[str, Any]:
        """Add an antibody (normally via thymus accept). Re-adding an
        existing id bumps the version and keeps history."""
        data = _to_dict(antibody)
        _require_reason(data)
        ab_id = str(data.get("id") or uuid.uuid4().hex)
        data["id"] = ab_id
        existing = self._store.get(ab_id)
        if existing is not None:
            self._history.setdefault(ab_id, []).append(copy.deepcopy(existing))
            data["version"] = int(existing.get("version", 1)) + 1
        else:
            data["version"] = int(data.get("version") or 1)
        data.setdefault("status", "active")
        data.setdefault("created_ts", time.time())
        data.setdefault("retire_reason", None)
        self._store[ab_id] = copy.deepcopy(data)
        self.save()
        return copy.deepcopy(data)

    def update(self, antibody: Any) -> Dict[str, Any]:
        """Update an antibody: same as add — version bumps, history kept."""
        return self.add(antibody)

    def get(self, antibody_id: str) -> Optional[Dict[str, Any]]:
        found = self._store.get(str(antibody_id))
        return copy.deepcopy(found) if found is not None else None

    def all(self) -> List[Dict[str, Any]]:
        return [copy.deepcopy(ab) for ab in self._store.values()]

    def list_active(self) -> List[Dict[str, Any]]:
        return [ab for ab in self.all() if ab.get("status") == "active"]

    def history(self, antibody_id: str) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._history.get(str(antibody_id), []))

    # ── second exposure: innate-speed matching ───────────────────

    def match(self, text: str) -> List[Dict[str, Any]]:
        """Return the active antibodies whose pattern matches ``text``.

        Second exposure to a known pathogen is caught at innate speed:
        compiled regexes, no labelling round-trip.
        """
        hits: List[Dict[str, Any]] = []
        haystack = text or ""
        for antibody in self.list_active():
            pattern = str(antibody.get("pattern") or "")
            if not pattern:
                continue
            try:
                compiled = _compile(pattern)
            except re.error:
                continue
            if compiled.search(haystack):
                hits.append(antibody)
        return hits

    # ── immune memory transfer ───────────────────────────────────

    def export_bundle(self) -> Dict[str, Any]:
        """Export immune memory for another tissue. Version + reason
        travel with every entry."""
        return {
            "format": "immune.antibodies.v1",
            "exported_ts": time.time(),
            "antibodies": self.all(),
            "history": copy.deepcopy(self._history),
        }

    def import_bundle(
        self, bundle: Dict[str, Any], *, overwrite: bool = False
    ) -> Dict[str, int]:
        """Import an immune-memory bundle. Conflicts keep the HIGHER
        version; reason and version are preserved, never reset."""
        imported = 0
        kept_existing = 0
        skipped = 0
        for incoming in bundle.get("antibodies", []):
            data = _to_dict(incoming)
            try:
                _require_reason(data)
            except ValueError:
                skipped += 1
                continue
            ab_id = str(data.get("id") or uuid.uuid4().hex)
            data["id"] = ab_id
            existing = self._store.get(ab_id)
            if existing is not None and not overwrite:
                if int(existing.get("version", 1)) >= int(data.get("version", 1)):
                    kept_existing += 1
                    continue
                self._history.setdefault(ab_id, []).append(copy.deepcopy(existing))
            elif existing is not None:
                self._history.setdefault(ab_id, []).append(copy.deepcopy(existing))
            self._store[ab_id] = copy.deepcopy(data)
            imported += 1
        for ab_id, entries in bundle.get("history", {}).items():
            merged = self._history.setdefault(str(ab_id), [])
            seen = {json.dumps(e, sort_keys=True, default=str) for e in merged}
            for entry in entries:
                key = json.dumps(entry, sort_keys=True, default=str)
                if key not in seen:
                    merged.append(copy.deepcopy(entry))
                    seen.add(key)
        self.save()
        return {"imported": imported, "kept_existing": kept_existing, "skipped": skipped}
