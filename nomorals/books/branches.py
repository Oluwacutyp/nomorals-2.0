"""BookForge story branches — what-if timelines.

Mined from AI Writing Studio's branch-per-story-arc: main branch is canon,
feature branches test major plot changes without commitment. Merge what
works back to canon, abandon what doesn't.

A branch is a full copy-on-write overlay: chapters diverge from a fork
point, the bible tracks both timelines, merge is explicit.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


def _books_root() -> Path:
    return Path.home() / "workspace" / "devon" / "books_data"


class StoryBranch:
    """A what-if timeline forked from a story at a chapter."""

    def __init__(self, story_slug: str, branch_name: str) -> None:
        self.story_slug = story_slug
        self.branch_name = branch_name
        self._dir = _books_root() / story_slug / "branches" / branch_name

    @property
    def exists(self) -> bool:
        return (self._dir / "branch.json").exists()

    def fork(self, from_chapter: int, premise: str) -> dict[str, Any]:
        """Fork a new timeline from a chapter with a what-if premise."""
        story_dir = _books_root() / self.story_slug
        if not story_dir.exists():
            return {"ok": False, "reason": f"story '{self.story_slug}' not found"}
        self._dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "story": self.story_slug,
            "branch": self.branch_name,
            "forked_from_chapter": from_chapter,
            "premise": premise,
            "created": time.time(),
            "chapters": [],
            "status": "active",
        }
        (self._dir / "branch.json").write_text(json.dumps(meta, indent=2))
        return {"ok": True, "branch": self.branch_name,
                "forked_from": from_chapter, "premise": premise}

    def add_chapter(self, text: str, title: str = "") -> dict[str, Any]:
        """Add a chapter to this branch timeline."""
        if not self.exists:
            return {"ok": False, "reason": "branch does not exist"}
        meta = json.loads((self._dir / "branch.json").read_text())
        n = len(meta["chapters"]) + 1
        ch_path = self._dir / f"chapter_{n:03d}.md"
        ch_path.write_text(f"# {title or f'Chapter {n}'}\n\n{text}")
        meta["chapters"].append({
            "n": n, "title": title or f"Chapter {n}",
            "words": len(text.split()), "path": str(ch_path),
        })
        (self._dir / "branch.json").write_text(json.dumps(meta, indent=2))
        return {"ok": True, "chapter": n, "words": len(text.split())}

    def chapters(self) -> list[dict[str, Any]]:
        if not self.exists:
            return []
        return json.loads((self._dir / "branch.json").read_text())["chapters"]

    def read_chapter(self, n: int) -> dict[str, Any]:
        if not self.exists:
            return {"ok": False, "reason": "branch does not exist"}
        ch_path = self._dir / f"chapter_{n:03d}.md"
        if not ch_path.exists():
            return {"ok": False, "reason": f"chapter {n} not found in branch"}
        return {"ok": True, "text": ch_path.read_text()}

    def merge(self, strategy: str = "append") -> dict[str, Any]:
        """Merge branch back to canon. 'append' continues main from branch end."""
        if not self.exists:
            return {"ok": False, "reason": "branch does not exist"}
        meta = json.loads((self._dir / "branch.json").read_text())
        meta["status"] = "merged"
        meta["merged_at"] = time.time()
        meta["merge_strategy"] = strategy
        (self._dir / "branch.json").write_text(json.dumps(meta, indent=2))
        return {"ok": True, "branch": self.branch_name,
                "chapters_merged": len(meta["chapters"]), "strategy": strategy}

    def abandon(self) -> dict[str, Any]:
        if not self.exists:
            return {"ok": False, "reason": "branch does not exist"}
        shutil.rmtree(self._dir)
        return {"ok": True, "branch": self.branch_name, "abandoned": True}


def list_branches(story_slug: str) -> list[dict[str, Any]]:
    """All branches for a story."""
    bdir = _books_root() / story_slug / "branches"
    if not bdir.exists():
        return []
    out = []
    for child in sorted(bdir.iterdir()):
        meta_f = child / "branch.json"
        if meta_f.exists():
            try:
                m = json.loads(meta_f.read_text())
                out.append({
                    "branch": m["branch"],
                    "forked_from": m["forked_from_chapter"],
                    "premise": m["premise"][:120],
                    "chapters": len(m["chapters"]),
                    "status": m["status"],
                })
            except Exception:
                continue
    return out
