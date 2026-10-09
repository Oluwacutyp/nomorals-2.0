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
