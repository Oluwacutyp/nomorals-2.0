"""Durable project store — ``<home>/film/projects/``.

Same idiom as the triage store: one JSON file per project, atomic
write (temp file + rename), content hash for identity.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .project import FilmProject

__all__ = ["FilmStore", "film_path"]


def film_path(settings: Any | None = None) -> Path:
    """``<home>/film/projects`` — ``settings.home_path`` when
    available, else ``~/.nomorals``."""
    home = getattr(settings, "home_path", None) if settings else None
    base = Path(home) if home else Path.home() / ".nomorals"
    return base / "film" / "projects"


class FilmStore:
    """One JSON file per project, atomically written."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _file(self, name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_"
                       for c in name)
        if not safe or safe in (".", ".."):
            raise ValueError(f"bad project name {name!r}")
        return self.root / f"{safe}.json"

    def save(self, project: FilmProject) -> Path:
        data = project.to_dict()
        blob = json.dumps(data, sort_keys=True, indent=2).encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()
        payload = {"name": project.name, "sha256": digest,
                   "project": data}
        fd, tmp = tempfile.mkstemp(prefix=".film-", dir=str(self.root))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(json.dumps(payload, sort_keys=True,
                                   indent=2).encode("utf-8"))
            os.replace(tmp, self._file(project.name))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return self._file(project.name)

    def load(self, name: str) -> FilmProject:
        path = self._file(name)
        if not path.exists():
            raise KeyError(f"no film project {name!r}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        blob = json.dumps(payload["project"], sort_keys=True,
                          indent=2).encode("utf-8")
        if hashlib.sha256(blob).hexdigest() != payload.get("sha256"):
            raise ValueError(f"film project {name!r} is corrupted "
                             f"(hash mismatch)")
        return FilmProject.from_dict(payload["project"])

    def list_names(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def delete(self, name: str) -> bool:
        path = self._file(name)
        if path.exists():
            path.unlink()
            return True
        return False
