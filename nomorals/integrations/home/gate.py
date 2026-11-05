"""The owner's per-device control gate (warn-first, standing rule).

Devon may *discover* and *read* the owner's home devices freely —
knowing is harmless.  **Control** (anything that actuates a device) is
different: it stays off, per device, until the owner explicitly enables
it.  This module is that enable registry:

* default: **off** — a device with no record is not controllable;
* the only way a record appears is an explicit owner action through
  :meth:`ControlGate.set_enabled` (surfaced as
  ``SmartHomeIntegration.enable_device_control`` /
  ``disable_device_control``);
* enforcement lives in the backends' control seam
  (:meth:`HomeBackend.control`), which calls :meth:`require_enabled`
  before any transport call — a refused control never reaches the
  device, by construction, not by convention;
* records persist as JSON (atomic tmp+replace, the consent-ledger
  idiom); a corrupt gate file raises rather than reading as "all
  enabled" or "all disabled" — the gate never guesses its own state.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ...core.errors import StorageError
from .errors import ControlNotEnabledError

__all__ = ["ControlGate"]


class ControlGate:
    """Persistent per-device control-enable registry."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._records: dict[str, dict[str, Any]] = {}
        if self._path is not None and self._path.exists():
            self._records = self._load(self._path)

    # ── persistence ──────────────────────────────────────────────────
    @staticmethod
    def _load(path: Path) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StorageError(
                f"home control gate file is unreadable: {path}: {exc}",
                details={"path": str(path)}) from exc
        if not isinstance(data, dict) or not isinstance(
                data.get("records", {}), dict):
            raise StorageError(
                f"home control gate file has an unexpected shape: {path}",
                details={"path": str(path)})
        records = data.get("records", {})
        return {str(k): dict(v) for k, v in records.items()
                if isinstance(v, dict)}

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"records": self._records}, indent=2,
                       sort_keys=True),
            encoding="utf-8")
        os.replace(tmp, self._path)

    # ── keys ─────────────────────────────────────────────────────────
    @staticmethod
    def _key(backend: str, device_id: str) -> str:
        return f"{backend}::{device_id}"

    # ── reads (always allowed) ───────────────────────────────────────
    def is_enabled(self, backend: str, device_id: str) -> bool:
        """True only when an explicit owner record enables this device.
        Absence of a record is **off**."""
        record = self._records.get(self._key(backend, device_id))
        return bool(record and record.get("enabled"))

    def require_enabled(self, backend: str, device_id: str, *,
                        action: str = "") -> None:
        """Raise :class:`ControlNotEnabledError` unless enabled.  The
        control seam calls this before any transport use."""
        if not self.is_enabled(backend, device_id):
            raise ControlNotEnabledError(
                backend=backend, device_id=device_id, action=action)

    def record_for(self, backend: str, device_id: str
                   ) -> dict[str, Any] | None:
        record = self._records.get(self._key(backend, device_id))
        return dict(record) if record is not None else None

    # ── the owner action surface ─────────────────────────────────────
    def set_enabled(self, backend: str, device_id: str, enabled: bool,
                    *, by: str = "owner", note: str = ""
                    ) -> dict[str, Any]:
        """Enable or disable control for one device.  This is the only
        writer of enable records — an explicit owner action, recorded
        with who and when."""
        record = {
            "backend": backend,
            "device_id": device_id,
            "enabled": bool(enabled),
            "updated_by": by,
            "updated_at": time.time(),
            "note": note,
        }
        self._records[self._key(backend, device_id)] = record
        self._save()
        return dict(record)

    # ── status as data ────────────────────────────────────────────────
    def status(self) -> list[dict[str, Any]]:
        """Every recorded enable/disable decision, as data."""
        return [dict(r) for r in sorted(
            self._records.values(),
            key=lambda r: (r.get("backend", ""), r.get("device_id", "")))]

    def status_for(self, backend: str, device_ids: list[str]
                   ) -> list[dict[str, Any]]:
        """Gate state for a set of devices (unrecorded = disabled)."""
        out = []
        for device_id in device_ids:
            record = self.record_for(backend, device_id)
            out.append({
                "backend": backend,
                "device_id": device_id,
                "control_enabled": bool(record and record.get("enabled")),
                "record": record,
            })
        return out
