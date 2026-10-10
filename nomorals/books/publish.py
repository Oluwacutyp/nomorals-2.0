"""BookForge serialized publishing — drip-publish chapters like a webnovel.

Mined from Royal Road author practice:
- Consistency beats volume: readers learn "new chapter Tuesday" faster
  than "four a week" — weekday schedules supported;
- The BACKLOG is the real schedule: write ahead, load drafts, the site
  posts at a fixed minute.  buffer_status() watches it;
- Launch with ~10 chapters on day one; 2-4k word chapters; author notes
  for schedule/stub alerts; track followers (views are noisy).

A publication is a schedule attached to a story. The scheduler (or a
manual tick) releases the next unpublished chapter on cadence.
"""

from __future__ import annotations

import calendar
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _books_root() -> Path:
    return Path.home() / "workspace" / "devon" / "books_data"


_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3,
             "fri": 4, "sat": 5, "sun": 6}


def parse_cadence(cadence: str) -> dict[str, Any]:
    """Parse 'daily' | 'weekly' | 'manual' | 'weekly:mon,wed,fri'.

    Returns {kind, weekdays} — raises ValueError on garbage.
    """
    c = (cadence or "").strip().lower()
    if c in ("daily", "weekly", "manual"):
        return {"kind": c, "weekdays": []}
    if c.startswith("weekly:"):
        days = []
        for part in c.split(":", 1)[1].split(","):
            key = part.strip()[:3]
            if key not in _WEEKDAYS:
                raise ValueError(
                    f"bad weekday {part!r} — use mon,tue,wed,thu,fri,sat,sun")
            days.append(_WEEKDAYS[key])
        if not days:
            raise ValueError("weekly: needs at least one weekday")
        return {"kind": "weekday", "weekdays": sorted(set(days))}
    raise ValueError(
        f"cadence must be daily | weekly | manual | weekly:mon,wed,fri — "
        f"got {cadence!r}")


def _next_due_ts(cadence: dict[str, Any], from_ts: float | None = None,
                 hour: int = 12, minute: int = 5) -> float:
    """Next release timestamp.  Default 12:05 — Royal Road wisdom says
    never release on the hour (front-page competition)."""
    now = datetime.now(timezone.utc)
    base = (datetime.fromtimestamp(from_ts, timezone.utc) if from_ts
            else now)
    kind = cadence["kind"]
    if kind == "manual":
        return time.time()
    if kind == "daily":
        nxt = base.replace(hour=hour, minute=minute, second=0,
                           microsecond=0)
        if nxt <= base:
            nxt += timedelta(days=1)
        return nxt.timestamp()
    # weekly / weekday: next matching weekday at hour:minute
    days = cadence["weekdays"] or [base.weekday()]
    best = None
    for wd in days:
        delta = (wd - base.weekday()) % 7
        cand = (base + timedelta(days=delta)).replace(
            hour=hour, minute=minute, second=0, microsecond=0)
        if cand <= base:
            cand += timedelta(days=7)
        if best is None or cand < best:
            best = cand
    return best.timestamp()  # type: ignore[union-attr]


class SerialPublication:
    """A drip-publish schedule for a story."""

    def __init__(self, story_slug: str) -> None:
        self.story_slug = story_slug
        self._dir = _books_root() / story_slug
        self._pub_file = self._dir / "publication.json"

    @property
    def exists(self) -> bool:
        return self._pub_file.exists()

    def start(self, cadence: str = "daily", title: str = "",
              announce_chat: str = "") -> dict[str, Any]:
        """Start serializing a story on a release cadence.

        ``cadence``: daily | weekly | manual | weekly:mon,wed,fri
        ("new chapter Tuesday" beats "four a week").
        """
        try:
            parsed = parse_cadence(cadence)
        except ValueError as exc:
            return {"ok": False, "reason": str(exc)}
        if not self._dir.exists():
            return {"ok": False, "reason": f"story '{self.story_slug}' not found"}
        pub = {
            "story": self.story_slug,
            "title": title,
            "cadence": cadence,
            "cadence_parsed": parsed,
            "started": time.time(),
            "released": [],       # chapter numbers released
            "next_due": _next_due_ts(parsed),
            "announce_chat": announce_chat,
            "followers": [],
            "feedback": [],       # reader reactions fed back to the author
            "author_notes": {},   # chapter_no -> note
            "status": "active",
        }
        self._pub_file.write_text(json.dumps(pub, indent=2))
        return {"ok": True, "cadence": cadence, "status": "active",
                "next_due": pub["next_due"]}

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
        parsed = pub.get("cadence_parsed") or {"kind": pub.get("cadence", "manual"),
                                               "weekdays": []}
        if parsed["kind"] != "manual" and time.time() < pub["next_due"]:
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
        parsed = pub.get("cadence_parsed") or {"kind": "manual", "weekdays": []}
        pub["next_due"] = _next_due_ts(parsed, from_ts=time.time())
        self._save(pub)
        return {
            "ok": True, "chapter": ch_n,
            "title": pub["title"] or self.story_slug,
            "text": text,
            "words": len(text.split()),
            "author_note": pub.get("author_notes", {}).get(str(ch_n), ""),
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
            "buffer": self.buffer_status().get("buffer", 0),
            "next_due": pub.get("next_due"),
        }

    # ── backlog buffer (Royal Road: the backlog IS the schedule) ──────────
    def buffer_status(self) -> dict[str, Any]:
        """Chapters written ahead of the release pointer.

        A buffer < 2 means one bad week kills the streak — the author
        should write before the next release, not after.
        """
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        chapters = self._story_chapters()
        released = set(pub["released"])
        buffer = [c for c in chapters if c not in released]
        healthy = len(buffer) >= 2
        return {"ok": True, "buffer": len(buffer),
                "buffer_chapters": buffer[:10],
                "released": len(released), "total": len(chapters),
                "healthy": healthy,
                "warning": "" if healthy else
                "buffer under 2 chapters — write ahead before the streak breaks"}

    # ── author notes ──────────────────────────────────────────────────────
    def set_author_note(self, chapter: int, note: str) -> dict[str, Any]:
        """Attach an author's note to a chapter (schedule updates, thanks,
        stub alerts — Royal Road's own convention)."""
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        notes = pub.setdefault("author_notes", {})
        notes[str(chapter)] = (note or "").strip()[:1000]
        self._save(pub)
        return {"ok": True, "chapter": chapter}

    # ── launch plan (day-one bulk release) ────────────────────────────────
    def launch_plan(self, day_one: int = 10) -> dict[str, Any]:
        """Royal Road launch: drop the first N chapters at once, then hold
        cadence.  Returns the plan; call release() N times to execute."""
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        chapters = self._story_chapters()
        released = set(pub["released"])
        pending = [c for c in chapters if c not in released]
        bulk = pending[:max(1, day_one)]
        rest = pending[len(bulk):]
        upcoming = self.next_releases(min(len(rest), 6))
        return {"ok": True, "story": self.story_slug,
                "day_one_chapters": bulk,
                "then_cadence": pub["cadence"],
                "upcoming": upcoming,
                "advice": (f"release chapters {bulk} together on day one, "
                           f"then hold '{pub['cadence']}' — the backlog "
                           f"({len(rest)} chapters) is the real schedule")}

    def next_releases(self, n: int = 6) -> list[dict[str, Any]]:
        """Preview the next N release datetimes on this cadence."""
        pub = self._load()
        if pub is None:
            return []
        parsed = pub.get("cadence_parsed") or {"kind": "manual",
                                               "weekdays": []}
        if parsed["kind"] == "manual":
            return [{"n": i + 1, "at": None, "note": "manual — release anytime"}
                    for i in range(n)]
        out = []
        ts = pub.get("next_due", time.time())
        for i in range(n):
            if i:
                ts = _next_due_ts(parsed, from_ts=ts)
            dt = datetime.fromtimestamp(ts, timezone.utc)
            out.append({"n": i + 1, "at": ts,
                        "at_iso": dt.isoformat(timespec="minutes"),
                        "weekday": calendar.day_name[dt.weekday()]})
        return out

    # ── retention curve (which chapter lost readers) ──────────────────────
    def retention_curve(self) -> dict[str, Any]:
        """Per-chapter reader retention from feedback volume.

        Chapters are points where readers bothered to react; a sharp drop
        in reactions between consecutive released chapters flags where
        the story lost people.
        """
        pub = self._load()
        if pub is None:
            return {"ok": False, "reason": "no publication"}
        released = pub["released"]
        if not released:
            return {"ok": True, "curve": [], "note": "nothing released yet"}
        counts: dict[int, int] = {}
        for f in pub["feedback"]:
            ch = f["chapter"]
            counts[ch] = counts.get(ch, 0) + 1
        peak = max([counts.get(c, 0) for c in released] + [1])
        curve = []
        prev_rate = 100.0
        for c in released:
            rate = round(100.0 * counts.get(c, 0) / peak, 1)
            drop = prev_rate - rate >= 25 and prev_rate > 0
            curve.append({"chapter": c, "reactions": counts.get(c, 0),
                          "retention_pct": rate, "drop": drop})
            prev_rate = rate
        drops = [r["chapter"] for r in curve if r["drop"]]
        return {"ok": True, "curve": curve,
                "drop_chapters": drops,
                "note": (f"sharp drop after ch.{drops}" if drops else
                         "no sharp drops — retention holding")}
