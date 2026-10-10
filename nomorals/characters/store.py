"""CharacterStore: JSON persistence for character agents."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .character import Character


def default_character_dir() -> Path:
    base = os.environ.get("NOMORALS_DATA_DIR") or str(
        Path.home() / ".nomorals")
    return Path(base) / "characters"


class CharacterStore:
    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.dir = Path(data_dir or default_character_dir())
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, char_id: str) -> Path:
        safe = "".join(c for c in char_id if c.isalnum() or c in "-_")[:32]
        return self.dir / f"{safe}.json"

    def save(self, char: Character) -> None:
        with self._lock:
            tmp = self._path(char.id).with_suffix(".tmp")
            tmp.write_text(json.dumps(char.to_dict(), indent=1),
                           encoding="utf-8")
            tmp.replace(self._path(char.id))

    def get(self, char_id: str) -> Character | None:
        p = self._path(char_id)
        if not p.exists():
            return None
        try:
            return Character.from_dict(json.loads(p.read_text("utf-8")))
        except Exception:
            return None

    def get_by_name(self, name: str) -> Character | None:
        low = (name or "").strip().lower()
        for c in self.all():
            if c.name.lower() == low:
                return c
        return None

    def all(self) -> list[Character]:
        out: list[Character] = []
        with self._lock:
            for p in sorted(self.dir.glob("*.json")):
                if p.name.startswith("_"):
                    continue  # sidecar files, not characters
                try:
                    out.append(Character.from_dict(
                        json.loads(p.read_text("utf-8"))))
                except Exception:
                    continue
        return out

    def delete(self, char_id: str) -> bool:
        p = self._path(char_id)
        with self._lock:
            if p.exists():
                p.unlink()
                return True
        return False

    def find(self, *, role: str = "", trait: str = "",
             skill: str = "", min_value: float = 0.5,
             name_contains: str = "") -> list[Character]:
        """Query the bank without loading everything into the caller:
        by role, by trait/skill strength, or by name fragment."""
        out = []
        for c in self.all():
            if role and role not in (c.roles or []):
                continue
            if trait and float(c.persona.get(trait, 0.0)) < min_value:
                continue
            if skill and float(c.skills.get(skill, 0.0)) < min_value:
                continue
            if name_contains and name_contains.lower() not in c.name.lower():
                continue
            out.append(c)
        return out

    def export_all(self) -> list[dict[str, Any]]:
        """Whole bank as plain dicts — for backups, migration, inspection."""
        return [c.to_dict() for c in self.all()]

    def import_all(self, dicts: list[dict[str, Any]],
                   overwrite: bool = False) -> dict[str, int]:
        """Restore from export_all(). Skips existing names unless
        ``overwrite``. Returns counts."""
        stats = {"created": 0, "skipped": 0, "overwritten": 0}
        for d in dicts or []:
            try:
                c = Character.from_dict(d)
            except Exception:
                continue
            existing = self.get_by_name(c.name)
            if existing and not overwrite:
                stats["skipped"] += 1
                continue
            if existing and overwrite:
                try:
                    self.delete(existing.id)
                except Exception:
                    pass
                stats["overwritten"] += 1
            else:
                stats["created"] += 1
            try:
                self.save(c)
            except Exception:
                pass
        return stats
