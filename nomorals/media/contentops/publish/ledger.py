"""Posting ledger — what was posted where, when, and at what cost.

JSON-persisted (atomic write: tmp file + rename) so a crash mid-publish
can't corrupt the record. Every entry carries the platform video id,
status, and quota spent — the receipts Devon's scheduler and the owner
need to answer "did this go out?" without guessing.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from ....core.logging_setup import get_logger

__all__ = ["PublishLedger", "PublishLedgerError"]

_log = get_logger(__name__)

_STATUSES = ("uploaded", "scheduled", "processing", "failed")


class PublishLedgerError(RuntimeError):
    """The ledger file is unreadable or unwritable."""


class PublishLedger:
    """Append-mostly JSON record of every publish attempt.

    ``path`` defaults to the workspace state dir; pass an explicit path
    in tests.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path.home() / "workspace" / "state" / "publish_ledger.json"
        self.path = Path(path)
        self._entries: list[dict[str, Any]] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            self._entries = []
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise PublishLedgerError(
                f"publish ledger at {self.path} is unreadable "
                f"({exc}) — refusing to silently start over; back it up "
                "and repair or delete it deliberately"
            ) from exc
        if not isinstance(data, list):
            raise PublishLedgerError(
                f"publish ledger at {self.path} is not a JSON list — "
                "refusing to silently start over"
            )
        self._entries = [dict(e) for e in data if isinstance(e, dict)]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(
                json.dumps(self._entries, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
        except OSError as exc:
            raise PublishLedgerError(
                f"could not write publish ledger to {self.path}: {exc}"
            ) from exc

    # ── records ──────────────────────────────────────────────────

    def record(
        self,
        *,
        platform: str,
        file: str,
        title: str,
        platform_video_id: str = "",
        status: str = "uploaded",
        quota_spent: int = 0,
        privacy: str = "",
        publish_at: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        """Append a publish record. Returns the stored entry."""
        if status not in _STATUSES:
            raise PublishLedgerError(
                f"invalid status {status!r}: use one of {', '.join(_STATUSES)}"
            )
        entry = {
            "id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "platform": platform,
            "file": file,
            "title": title,
            "platform_video_id": platform_video_id,
            "status": status,
            "quota_spent": int(quota_spent),
            "privacy": privacy,
            "publish_at": publish_at,
            "notes": notes,
        }
        self._entries.append(entry)
        self._save()
        _log.info(
            "ledger: %s %s -> %s (%s)",
            platform, title, platform_video_id or "?", status,
        )
        return dict(entry)

    def update(self, entry_id: str, **fields: Any) -> dict[str, Any]:
        """Patch an entry (e.g. processing -> uploaded with the video id)."""
        for entry in self._entries:
            if entry.get("id") == entry_id:
                if "status" in fields and fields["status"] not in _STATUSES:
                    raise PublishLedgerError(
                        f"invalid status {fields['status']!r}"
                    )
                entry.update(fields)
                self._save()
                return dict(entry)
        raise PublishLedgerError(f"no ledger entry {entry_id!r}")

    def get(self, entry_id: str) -> dict[str, Any] | None:
        for entry in self._entries:
            if entry.get("id") == entry_id:
                return dict(entry)
        return None

    def list(
        self,
        *,
        platform: str = "",
        status: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Newest-first entries, optionally filtered."""
        out = list(reversed(self._entries))
        if platform:
            out = [e for e in out if e.get("platform") == platform]
        if status:
            out = [e for e in out if e.get("status") == status]
        return [dict(e) for e in out[: max(1, limit)]]

    def summary(self) -> dict[str, Any]:
        """Counts per platform/status + total quota spent."""
        by_platform: dict[str, int] = {}
        by_status: dict[str, int] = {}
        quota = 0
        for entry in self._entries:
            plat = str(entry.get("platform", "?"))
            by_platform[plat] = by_platform.get(plat, 0) + 1
            st = str(entry.get("status", "?"))
            by_status[st] = by_status.get(st, 0) + 1
            quota += int(entry.get("quota_spent", 0) or 0)
        return {
            "total": len(self._entries),
            "by_platform": by_platform,
            "by_status": by_status,
            "quota_spent_total": quota,
        }
