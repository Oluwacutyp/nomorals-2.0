"""Preference storage — descriptors as data, weights learned from signals.

Design rules:

* **Descriptors are data, not taste.** ``DOMAIN_DESCRIPTORS`` names the
  *axes* per domain (e.g. music: energy, tempo, language, era) — it
  never says which value on an axis the owner likes.  All taste is
  learned from explicit approve / disapprove signals, starting at
  zero.  Adding a domain never touches the learning logic.
* **No hardcoded user taste.** A fresh store scores every candidate
  0 and preserves the caller's order; ranking only moves after real
  signals.
* **Universal.** Axes are plain strings; values are whatever the
  caller's items carry on those axes (any language, any vocabulary).
  The store never parses or judges item content beyond reading the
  descriptor axes.
* **Two backends, one API.** SQLite (``:memory:`` by default, or a
  ``.db`` / ``.sqlite`` path) or a JSON file (``.json`` path).  Both
  persist weights + the signal log.

Only the standard library is imported.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any

__all__ = [
    "DOMAIN_DESCRIPTORS",
    "PreferenceStore",
    "extract_axis_values",
]

#: Domain -> descriptor.  Each descriptor names the axes that describe
#: items in that domain.  Axes are dimension names only — no values,
#: no defaults, no taste is baked in.
DOMAIN_DESCRIPTORS: dict[str, dict[str, Any]] = {
    "media": {
        "axes": ["genre", "format", "language", "era", "mood"],
        "description": "Media items (film / video / audio / general).",
    },
    "music": {
        "axes": ["energy", "tempo", "language", "era", "genre", "mood"],
        "description": "Music items.",
    },
    "food": {
        "axes": ["spice", "cuisine", "diet", "flavour", "meal"],
        "description": "Food / recipe / restaurant items.",
    },
    "travel": {
        "axes": ["climate", "pace", "budget", "setting", "activity"],
        "description": "Travel destinations / trips.",
    },
    "games": {
        "axes": ["genre", "pace", "players", "setting", "difficulty"],
        "description": "Games.",
    },
    "books": {
        "axes": ["genre", "pace", "length", "language", "era", "mood"],
        "description": "Books / stories.",
    },
    "news": {
        "axes": ["topic", "region", "depth", "tone", "source_type"],
        "description": "News / articles.",
    },
}


def _normalise_value(value: Any) -> str:
    """One axis value → stable weight key.  Empty / None → ''."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    return text


def extract_axis_values(item: Any, domain: str) -> dict[str, list[str]]:
    """Read an item's descriptor-axis values.

    Accepts, in order of preference:
    * a dict with axis keys at the top level
      (``{"energy": "high", "tempo": "fast"}``);
    * a dict nested under the domain key
      (``{"music": {"energy": "high"}}``);
    * a dict nested under ``attributes`` / ``axes`` / ``features`` /
      ``descriptor`` / ``tags`` (tags may be a list of ``"axis:value"``
      strings or plain values — plain values are skipped: without an
      axis they cannot be attributed honestly);
    * anything else → ``{}`` (no axes, no invented values).

    Values may be a scalar or a list of scalars.  Returns
    ``{axis: [value, ...]}`` with only the domain's axes, de-duplicated,
    empty values dropped.
    """
    descriptor = DOMAIN_DESCRIPTORS.get(domain)
    if descriptor is None or not isinstance(item, dict):
        return {}
    axes = descriptor["axes"]
    axes_set = set(axes)
    source: dict[str, Any] = {}
    # Top-level axis keys win.
    for key, value in item.items():
        if key in axes_set:
            source[key] = value
    if not source:
        # Nested under the domain name.
        nested = item.get(domain)
        if isinstance(nested, dict):
            for key, value in nested.items():
                if key in axes_set:
                    source[key] = value
    if not source:
        for container in ("attributes", "axes", "features", "descriptor"):
            nested = item.get(container)
            if isinstance(nested, dict):
                for key, value in nested.items():
                    if key in axes_set:
                        source[key] = value
                if source:
                    break
    if not source:
        tags = item.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and ":" in tag:
                    axis, _, value = tag.partition(":")
                    axis = axis.strip().lower()
                    if axis in axes_set:
                        source.setdefault(axis, [])
                        if isinstance(source[axis], list):
                            source[axis].append(value.strip())
                        else:
                            source[axis] = [source[axis], value.strip()]
    out: dict[str, list[str]] = {}
    for axis in axes:
        if axis not in source:
            continue
        raw = source[axis]
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        seen: list[str] = []
        for value in values:
            norm = _normalise_value(value)
            if norm and norm not in seen:
                seen.append(norm)
        if seen:
            out[axis] = seen
    return out


_PREFS_DDL = """
CREATE TABLE IF NOT EXISTS preferences (
    domain TEXT NOT NULL,
    axis   TEXT NOT NULL,
    value  TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (domain, axis, value)
);
CREATE TABLE IF NOT EXISTS preference_signals (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    domain   TEXT NOT NULL,
    item_json TEXT NOT NULL DEFAULT '{}',
    approve  INTEGER NOT NULL,
    delta    REAL NOT NULL,
    source   TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
"""


class PreferenceStore:
    """Learned preference weights, keyed by (domain, axis, value).

    ``path``: ``None`` → SQLite ``:memory:``; a ``.json`` path → JSON
    file backend; anything else → SQLite file at that path.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._is_json = bool(path and str(path).lower().endswith(".json"))
        if self._is_json:
            self._weights: dict[str, dict[str, dict[str, float]]] = {}
            self._signals: list[dict[str, Any]] = []
            self._load_json()
        else:
            db_path = ":memory:" if path is None else str(path)
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            with self._lock:
                self._conn.executescript(_PREFS_DDL)
                self._conn.commit()

    # ── JSON backend ─────────────────────────────────────────────
    def _load_json(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        weights = raw.get("weights") or {}
        if isinstance(weights, dict):
            for domain, axes in weights.items():
                if not isinstance(axes, dict):
                    continue
                for axis, values in axes.items():
                    if not isinstance(values, dict):
                        continue
                    for value, weight in values.items():
                        if isinstance(weight, (int, float)):
                            self._weights.setdefault(str(domain), {}).setdefault(
                                str(axis), {})[str(value)] = float(weight)
        signals = raw.get("signals") or []
        if isinstance(signals, list):
            self._signals = [s for s in signals if isinstance(s, dict)]

    def _save_json(self) -> None:
        if not self._is_json or not self.path:
            return
        try:
            parent = os.path.dirname(str(self.path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = str(self.path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"weights": self._weights,
                           "signals": self._signals}, fh,
                          ensure_ascii=False)
            os.replace(tmp, str(self.path))
        except OSError:
            pass

    # ── descriptors ──────────────────────────────────────────────
    def describe(self, domain: str) -> dict[str, Any] | None:
        """The descriptor for a domain, or ``None`` when unknown."""
        descriptor = DOMAIN_DESCRIPTORS.get(domain)
        return dict(descriptor) if descriptor is not None else None

    def domains(self) -> list[str]:
        return sorted(DOMAIN_DESCRIPTORS)

    # ── weights ──────────────────────────────────────────────────
    def get_weights(self, domain: str) -> dict[str, dict[str, float]]:
        """{axis: {value: weight}} for a domain.  Empty when unknown / no signals."""
        if domain not in DOMAIN_DESCRIPTORS:
            return {}
        if self._is_json:
            with self._lock:
                return {axis: dict(values)
                        for axis, values in
                        self._weights.get(domain, {}).items()}
        with self._lock:
            rows = self._conn.execute(
                "SELECT axis, value, weight FROM preferences WHERE domain = ?",
                (domain,)).fetchall()
        out: dict[str, dict[str, float]] = {}
        for row in rows:
            out.setdefault(row["axis"], {})[row["value"]] = float(row["weight"])
        return out

    def weights(self, domain: str) -> dict[str, dict[str, float]]:
        return self.get_weights(domain)

    def get_weight(self, domain: str, axis: str, value: str) -> float:
        """Weight for one (domain, axis, value).  0.0 when unseen."""
        norm = _normalise_value(value)
        if self._is_json:
            with self._lock:
                return float(self._weights.get(domain, {}).get(axis, {}).get(norm, 0.0))
        with self._lock:
            row = self._conn.execute(
                "SELECT weight FROM preferences WHERE domain = ? AND axis = ? AND value = ?",
                (domain, axis, norm)).fetchone()
        return float(row["weight"]) if row else 0.0

    def update_weight(self, domain: str, axis: str, value: str,
                      delta: float) -> float:
        """Add ``delta`` to one weight; returns the new weight.

        Unknown domains / axes outside the descriptor are refused
        (returns the current weight, 0.0) — the store never learns a
        dimension the descriptor does not name.
        """
        descriptor = DOMAIN_DESCRIPTORS.get(domain)
        if descriptor is None or axis not in descriptor["axes"]:
            return 0.0
        norm = _normalise_value(value)
        if not norm:
            return 0.0
        with self._lock:
            if self._is_json:
                bucket = self._weights.setdefault(domain, {}).setdefault(axis, {})
                bucket[norm] = float(bucket.get(norm, 0.0)) + float(delta)
                self._save_json()
                return bucket[norm]
            current = self.get_weight(domain, axis, norm)
            new = current + float(delta)
            self._conn.execute(
                "INSERT INTO preferences (domain, axis, value, weight)"
                " VALUES (?, ?, ?, ?) ON CONFLICT(domain, axis, value)"
                " DO UPDATE SET weight = excluded.weight",
                (domain, axis, norm, new))
            self._conn.commit()
            return new

    # ── signals log ──────────────────────────────────────────────
    def log_signal(self, domain: str, item: Any, approve: bool,
                   delta: float, source: str = "") -> None:
        try:
            item_json = json.dumps(item, default=str, ensure_ascii=False)[:2000]
        except (TypeError, ValueError):
            item_json = "{}"
        with self._lock:
            if self._is_json:
                self._signals.append({
                    "domain": domain, "item": item,
                    "approve": bool(approve), "delta": float(delta),
                    "source": source, "created_at": time.time()})
                self._save_json()
                return
            self._conn.execute(
                "INSERT INTO preference_signals (domain, item_json, approve, delta, source, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (domain, item_json, 1 if approve else 0, float(delta),
                 source, time.time()))
            self._conn.commit()

    def signals(self, domain: str | None = None) -> list[dict[str, Any]]:
        if self._is_json:
            with self._lock:
                rows = list(self._signals)
            if domain is not None:
                rows = [r for r in rows if r.get("domain") == domain]
            return rows
        with self._lock:
            if domain is None:
                rows = self._conn.execute(
                    "SELECT * FROM preference_signals ORDER BY id").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM preference_signals WHERE domain = ? ORDER BY id",
                    (domain,)).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                item = json.loads(row["item_json"] or "{}")
            except ValueError:
                item = {}
            out.append({"domain": row["domain"], "item": item,
                        "approve": bool(row["approve"]),
                        "delta": float(row["delta"]),
                        "source": row["source"],
                        "created_at": float(row["created_at"])})
        return out

    # ── serialisation ────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": {domain: self.get_weights(domain)
                        for domain in self.domains()},
            "signals": self.signals(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PreferenceStore":
        store = cls()
        weights = raw.get("weights") or {}
        if isinstance(weights, dict):
            for domain, axes in weights.items():
                if not isinstance(axes, dict):
                    continue
                for axis, values in axes.items():
                    if not isinstance(values, dict):
                        continue
                    for value, weight in values.items():
                        if isinstance(weight, (int, float)):
                            store.update_weight(domain, axis, value, float(weight))
        return store

    def close(self) -> None:
        if not self._is_json:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
