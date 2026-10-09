"""BookForge serialized publishing — drip-publish chapters like a webnovel.

Mined from Royal Road / Wattpad / Tapas patterns:
- Scheduled chapter releases (daily/weekly cadence)
- Chapters drop to chat as they release
- Reader reactions/comments feed back into story direction
- Follow/favorite mechanics for serials

A publication is a schedule attached to a story. The scheduler (or a
manual tick) releases the next unpublished chapter on cadence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _books_root() -> Path:
    return Path.home() / "workspace" / "devon" / "books_data"


class SerialPublication:
    """A drip-publish schedule for a story."""

    CADENCES = ("daily", "weekly", "manual")

    def __init__(self, story_slug: str) -> None:
        self.story_slug = story_slug
        self._dir = _books_root() / story_slug
        self._pub_file = self._dir / "publication.json"

    @property
    def exists(self) -> bool:
        return self._pub_file.exists()

    def start(self, cadence: str = "daily", title: str = "",
              announce_chat: str = "") -> dict[str, Any]:
        """Start serializing a story on a release cadence."""
        if cadence not in self.CADENCES:
            return {"ok": False,
                    "reason": f"cadence must be one of {self.CADENCES}"}
        if not self._dir.exists():
            return {"ok": False, "reason": f"story '{self.story_slug}' not found"}
        pub = {
            "story": self.story_slug,
            "title": title,
            "cadence": cadence,
            "started": time.time(),
            "released": [],       # chapter numbers released
            "next_due": time.time() if cadence == "manual" else time.time() + 86400,
            "announce_chat": announce_chat,
            "followers": [],
            "feedback": [],       # reader reactions fed back to the author
            "status": "active",
        }
        self._pub_file.write_text(json.dumps(pub, indent=2))
        return {"ok": True, "cadence": cadence, "status": "active"}

    def _load(self) -> dict[str, Any] | None:
        if not self.exists:
            return None
        try:
            return json.loads(self._pub_file.read_text())
        except Exception:
            return None

    def _save(self, pub: dict[str, Any]) -> None:
        self._pub_file.write_text(json.dumps(pub, indent=2))

    def _story_chapters(self) -> list[int]:
        """Chapter numbers available in the story."""
        out = []
        for child in sorted(self._dir.glob("chapter_*.md")):
            try:
                out.append(int(child.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return out

    def due(self) -> dict[str, Any]:
        """Check if a chapter release is due. Returns release info or not-due."""
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        if pub["status"] != "active":
            return {"ok": False, "reason": f"publication {pub['status']}"}
        if pub["cadence"] != "manual" and time.time() < pub["next_due"]:
            return {"ok": False, "due": False,
                    "next_due_in_s": int(pub["next_due"] - time.time())}
        chapters = self._story_chapters()
        released = set(pub["released"])
        pending = [c for c in chapters if c not in released]
        if not pending:
            return {"ok": False, "due": False, "reason": "no unpublished chapters"}
        return {"ok": True, "due": True, "chapter": pending[0]}

    def release(self) -> dict[str, Any]:
        """Release the next chapter. Returns chapter text for delivery."""
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        d = self.due()
        if not d.get("due"):
            return {"ok": False, "reason": d.get("reason", "not due")}
        ch_n = d["chapter"]
        ch_path = self._dir / f"chapter_{ch_n:03d}.md"
        text = ch_path.read_text() if ch_path.exists() else ""
        pub["released"].append(ch_n)
        interval = {"daily": 86400, "weekly": 604800, "manual": 0}[pub["cadence"]]
        pub["next_due"] = time.time() + interval if interval else time.time()
        self._save(pub)
        return {
            "ok": True, "chapter": ch_n,
            "title": pub["title"] or self.story_slug,
            "text": text,
            "words": len(text.split()),
            "released_count": len(pub["released"]),
            "announce_chat": pub["announce_chat"],
            "followers": pub["followers"],
        }

    def follow(self, reader_id: str) -> dict[str, Any]:
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        if reader_id not in pub["followers"]:
            pub["followers"].append(reader_id)
            self._save(pub)
        return {"ok": True, "followers": len(pub["followers"])}

    def unfollow(self, reader_id: str) -> dict[str, Any]:
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        if reader_id in pub["followers"]:
            pub["followers"].remove(reader_id)
            self._save(pub)
        return {"ok": True, "followers": len(pub["followers"])}

    def feedback(self, reader_id: str, chapter: int, reaction: str,
                 comment: str = "") -> dict[str, Any]:
        """A reader reacts to a chapter. Feeds back into story direction."""
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        pub["feedback"].append({
            "reader": reader_id, "chapter": chapter,
            "reaction": reaction, "comment": comment[:500],
            "ts": time.time(),
        })
        # Keep the log bounded
        pub["feedback"] = pub["feedback"][-500:]
        self._save(pub)
        return {"ok": True, "feedback_count": len(pub["feedback"])}

    def feedback_summary(self) -> dict[str, Any]:
        """Aggregate reader sentiment per chapter for the author."""
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        by_chapter: dict[int, dict[str, Any]] = {}
        for f in pub["feedback"]:
            ch = f["chapter"]
            slot = by_chapter.setdefault(ch, {"reactions": {}, "comments": []})
            r = f["reaction"]
            slot["reactions"][r] = slot["reactions"].get(r, 0) + 1
            if f["comment"]:
                slot["comments"].append(f["comment"])
        return {"ok": True, "by_chapter": by_chapter,
                "total": len(pub["feedback"])}

    def pause(self) -> dict[str, Any]:
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        pub["status"] = "paused"
        self._save(pub)
        return {"ok": True, "status": "paused"}

    def resume(self) -> dict[str, Any]:
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        pub["status"] = "active"
        self._save(pub)
        return {"ok": True, "status": "active"}

    def status(self) -> dict[str, Any]:
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        chapters = self._story_chapters()
        return {
            "ok": True, "story": self.story_slug,
            "cadence": pub["cadence"], "status": pub["status"],
            "released": len(pub["released"]),
            "total_chapters": len(chapters),
            "followers": len(pub["followers"]),
            "feedback_count": len(pub["feedback"]),
        }
