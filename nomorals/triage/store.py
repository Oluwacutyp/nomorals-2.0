"""Durable triage store (autocraft port).

One JSON file under ``<home>/triage/triage.json`` — the same shape
as the ticket relay's store: atomic save (tmp + replace), one row
per mirrored item keyed ``owner/repo#number``, tracked repos,
duplicate groups, decisions and a write audit. The write audit is
the exactly-once half of autocraft's Act: every GitHub write is
recorded right after it happens, so a crash or retry never repeats
a comment or a close.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["triage_path", "item_key", "text_hash", "TriageStore"]


def triage_path(settings: Any = None) -> Path:
    """The triage store path, honoring the runtime settings home."""
    if settings is not None:
        home = getattr(settings, "home_path", None)
        if home is not None:
            return Path(home) / "triage" / "triage.json"
    return Path.home() / ".nomorals" / "triage" / "triage.json"


def item_key(repo: str, number: int) -> str:
    return f"{repo}#{int(number)}"


def text_hash(title: str, body: str) -> str:
    return hashlib.sha256(
        f"{title or ''}\n{body or ''}".encode()).hexdigest()


def _blank() -> dict[str, Any]:
    return {
        "repos": [],
        "items": {},
        "groups": [],
        "candidates": [],
        "decisions": [],
        "writes": [],
        "ruled_out": [],
        "next_decision": 1,
    }


class TriageStore:
    """A JSON-backed mirror + triage state for tracked repos."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = _blank()
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
            except (ValueError, OSError):
                loaded = {}
            if isinstance(loaded, dict):
                merged = _blank()
                merged.update(loaded)
                self.data = merged

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True))
        tmp.replace(self.path)

    # ── repos ──────────────────────────────────────────────────────
    def track(self, repo: str) -> bool:
        repo = (repo or "").strip()
        if not repo or repo in self.data["repos"]:
            return False
        self.data["repos"].append(repo)
        self.save()
        return True

    def untrack(self, repo: str) -> bool:
        if repo not in self.data["repos"]:
            return False
        self.data["repos"].remove(repo)
        self.save()
        return True

    def tracked(self) -> list[str]:
        return list(self.data["repos"])

    # ── items ──────────────────────────────────────────────────────
    def upsert_item(self, item: dict[str, Any]) -> str:
        """Insert or refresh a mirrored item.

        Returns ``new`` / ``updated`` / ``unchanged`` — a changed
        title/body (``text_hash``) is what re-opens classification.
        """
        key = item_key(str(item.get("repo", "")),
                       int(item.get("number", 0)))
        row = dict(item)
        row["key"] = key
        row["text_hash"] = text_hash(str(item.get("title", "")),
                                     str(item.get("body", "")))
        old = self.data["items"].get(key)
        if old is None:
            self.data["items"][key] = row
            self.save()
            return "new"
        if old.get("text_hash") != row["text_hash"]:
            merged = dict(old)
            merged.update(row)
            self.data["items"][key] = merged
            self.save()
            return "updated"
        merged = dict(old)
        for field in ("state", "labels", "comments", "updated_at"):
            if field in row:
                merged[field] = row[field]
        self.data["items"][key] = merged
        self.save()
        return "unchanged"

    def get_item(self, key: str) -> dict[str, Any] | None:
        row = self.data["items"].get(key)
        return dict(row) if row is not None else None

    def items_for(self, repo: str, *,
                  state: str | None = None) -> list[dict[str, Any]]:
        out = [dict(row) for row in self.data["items"].values()
               if row.get("repo") == repo
               and (state is None or row.get("state") == state)]
        out.sort(key=lambda row: int(row.get("number", 0)))
        return out

    def set_classification(self, key: str,
                           classification: dict[str, Any]) -> None:
        row = self.data["items"].get(key)
        if row is None:
            return
        row.update({
            "type": classification.get("type"),
            "type_source": classification.get("source"),
            "type_confidence": classification.get("confidence"),
            "security": classification.get("security"),
            "needs_decision": classification.get("needs_decision"),
            "priority": classification.get("priority"),
            "reach": classification.get("reach"),
            "work_rank": classification.get("work_rank"),
        })
        self.save()

    # ── duplicate candidates + groups ──────────────────────────────
    def record_candidates(
            self, pairs: list[dict[str, Any]]) -> None:
        for pair in pairs:
            self.data["candidates"].append(dict(pair))
        self.save()

    def candidates(self, *, status: str | None = None
                   ) -> list[dict[str, Any]]:
        out = [dict(c) for c in self.data["candidates"]]
        if status is not None:
            out = [c for c in out if c.get("status") == status]
        return out

    def add_group(self, group: dict[str, Any]) -> dict[str, Any]:
        row = dict(group)
        self.data["groups"].append(row)
        self.save()
        return row

    def groups(self) -> list[dict[str, Any]]:
        return [dict(g) for g in self.data["groups"]]

    def ruled_out_pairs(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            tuple(p) for p in self.data.get("ruled_out", []))

    def rule_out(self, key_a: str, key_b: str) -> None:
        pair = [key_a, key_b]
        if pair not in self.data["ruled_out"]:
            self.data["ruled_out"].append(pair)
            self.save()

    # ── decisions ──────────────────────────────────────────────────
    def add_decision(self, decision: dict[str, Any]) -> dict[str, Any]:
        row = dict(decision)
        number = int(self.data.get("next_decision", 1))
        row.setdefault("id", f"dec-{number:04d}")
        self.data["next_decision"] = number + 1
        row.setdefault("status", "open")
        self.data["decisions"].append(row)
        self.save()
        return dict(row)

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        for row in self.data["decisions"]:
            if row.get("id") == decision_id:
                return dict(row)
        return None

    def decisions(self, *, status: str | None = None,
                  repo: str | None = None) -> list[dict[str, Any]]:
        out = [dict(d) for d in self.data["decisions"]]
        if status is not None:
            out = [d for d in out if d.get("status") == status]
        if repo is not None:
            out = [d for d in out
                   if d.get("repo") == repo
                   or any(str(m).startswith(repo + "#")
                          for m in d.get("members", []))]
        return out

    def update_decision(self, decision_id: str,
                        **fields: Any) -> dict[str, Any] | None:
        for row in self.data["decisions"]:
            if row.get("id") == decision_id:
                row.update(fields)
                self.save()
                return dict(row)
        return None

    # ── write audit ────────────────────────────────────────────────
    def record_write(self, entry: dict[str, Any]) -> None:
        self.data["writes"].append(dict(entry))
        self.save()

    def already_written(self, key: str) -> bool:
        return any(w.get("key") == key for w in self.data["writes"])
