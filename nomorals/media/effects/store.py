"""Durable composition store — ``<home>/effects/compositions/``.

Same idiom as the triage store: one JSON file per composition,
atomic write (temp file + rename), content hash for identity.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .composition import Composition

__all__ = ["CompositionStore", "effects_path"]


def effects_path(settings: Any | None = None) -> Path:
    """``<home>/effects/compositions`` — ``settings.home_path`` when
    available, else ``~/.nomorals``."""
    home = getattr(settings, "home_path", None) if settings else None
    base = Path(home) if home else Path.home() / ".nomorals"
    return base / "effects" / "compositions"


class CompositionStore:
    """One JSON file per composition, atomically written."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _file(self, name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_"
                       for c in name)
        if not safe or safe in (".", ".."):
            raise ValueError(f"bad composition name {name!r}")
        return self.root / f"{safe}.json"

    def save(self, name: str, comp: Composition) -> Path:
        data = comp.to_dict()
        blob = json.dumps(data, sort_keys=True, indent=2).encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()
        payload = {"name": name, "sha256": digest, "composition": data}
        fd, tmp = tempfile.mkstemp(prefix=".comp-", dir=str(self.root))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(json.dumps(payload, sort_keys=True,
                                   indent=2).encode("utf-8"))
            os.replace(tmp, self._file(name))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return self._file(name)

    def load(self, name: str) -> Composition:
        path = self._file(name)
        if not path.exists():
            raise KeyError(f"no composition {name!r}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        blob = json.dumps(payload["composition"], sort_keys=True,
                          indent=2).encode("utf-8")
        if hashlib.sha256(blob).hexdigest() != payload.get("sha256"):
            raise ValueError(f"composition {name!r} is corrupted "
                             f"(hash mismatch)")
        return Composition.from_dict(payload["composition"])

    def list_names(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def delete(self, name: str) -> bool:
        path = self._file(name)
        if path.exists():
            path.unlink()
            return True
        return False
