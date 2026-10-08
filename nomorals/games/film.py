"""VOD → coaching breakdown (mistakes + highlights). Build-map #88.

Automated film study: upload a match/gameplay video → Devon finds the
key moments, reads each one with Seer, and returns 3 mistakes + 3 good
plays with timestamps, plus coaching notes. Longitudinal player
fingerprints ("you always overextend at minute 8") come from the
FingerprintStore across multiple VODs.

Games: valorant, lol (League of Legends), bgmi, sport (generic
traditional sports), workout (delegates to #86's form coach).

Seer-based, injectable vision seam. Never raises. Honest when vision,
ffmpeg, or frames are unavailable.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger("nomorals.games.film")

# ── constants ──────────────────────────────────────────────────────────

GAMES = ("valorant", "lol", "bgmi", "sport", "workout")

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Max key moments analyzed per VOD (keeps Seer calls bounded).
_MAX_MOMENTS = 8

# Scene-change sensitivity for the ffmpeg scene filter.
_SCENE_THRESHOLD = 0.4

_GAME_LABELS = {
    "valorant": "Valorant",
    "lol": "League of Legends",
    "bgmi": "BGMI (PUBG Mobile)",
    "sport": "sports",
    "workout": "workout",
}

# Game-specific common mistake patterns — used for coaching notes and
# for normalizing fingerprint observations.
_GAME_PATTERNS: dict[str, dict[str, str]] = {
    "valorant": {
        "overextend": "overextending / pushing without team support",
        "crosshair": "crosshair placement too low",
        "economy": "poor economy management (forcing)",
        "utility": "wasted or mistimed utility",
        "rotation": "slow rotation",
    },
    "lol": {
        "overextend": "overextending without vision",
        "cs": "missing CS under pressure",
        "map": "poor map awareness",
        "objective": "bad objective setup / coinflip fights",
        "positioning": "poor teamfight positioning",
    },
    "bgmi": {
        "overextend": "overextending in the open",
        "positioning": "poor positioning in final circles",
        "rotation": "late rotation into zone",
        "cover": "fighting without cover",
        "loot": "looting too long in hot zones",
    },
    "sport": {
        "positioning": "poor positioning",
        "awareness": "low situational awareness",
        "technique": "technique breakdown under pressure",
        "transition": "slow transition (attack ↔ defense)",
    },
}

_DISCLAIMER = ("coaching notes, not professional advice — these reads come "
               "from video frames, not a measured data feed.")


# ── data ───────────────────────────────────────────────────────────────

@dataclass
class Moment:
    """One key moment in the VOD."""
    timestamp: float  # seconds into the video
    frame_path: str = ""

    @property
    def label(self) -> str:
        m = int(self.timestamp // 60)
        s = int(self.timestamp % 60)
        return f"{m:02d}:{s:02d}"


@dataclass
class PlayNote:
    """One mistake or highlight with its timestamp."""
    kind: str          # "mistake" | "highlight"
    text: str
    timestamp: float = 0.0
    fix: str = ""      # coaching fix for mistakes

    @property
    def label(self) -> str:
        m = int(self.timestamp // 60)
        s = int(self.timestamp % 60)
        return f"{m:02d}:{s:02d}"


@dataclass
class FilmBreakdown:
    """The full VOD analysis."""
    id: str = ""
    game: str = ""
    video: str = ""
    available: bool = True
    mistakes: list[PlayNote] = field(default_factory=list)
    highlights: list[PlayNote] = field(default_factory=list)
    coaching: str = ""
    moments_used: int = 0
    note: str = ""
    created_at: float = 0.0

    def format(self) -> str:
        lines = [f"🎬 VOD breakdown — {_GAME_LABELS.get(self.game, self.game)} "
                 f"[{self.id}]"]
        if not self.available:
            lines.append(f"couldn't analyze: {self.note}")
            return "\n".join(lines)
        if self.mistakes:
            lines.append("\n❌ 3 mistakes to fix:")
            for n in self.mistakes:
                lines.append(f"  [{n.label}] {n.text}")
                if n.fix:
                    lines.append(f"    → {n.fix}")
        if self.highlights:
            lines.append("\n✅ 3 good plays:")
            for n in self.highlights:
                lines.append(f"  [{n.label}] {n.text}")
        if self.coaching:
            lines.append(f"\n🧠 coaching: {self.coaching}")
        lines.append(f"\n_{_DISCLAIMER}_")
        return "\n".join(lines)


VisionFn = Callable[[str, str], str]


# ── vision plumbing ────────────────────────────────────────────────────

def _default_vision() -> VisionFn:
    """Lazy Seer — heavy vision deps never import at module load."""
    from ..vision.seer import get_seer
    seer = get_seer()
    return seer.see


def _scenes_for(video_path: str, max_moments: int = _MAX_MOMENTS
                ) -> tuple[list[Moment] | None, str]:
    """Video → key moments via scene detection. Never raises.

    Returns (moments, "") or (None, honest-reason).
    """
    try:
        p = Path(video_path or "")
        if not p.is_file():
            return None, f"no video found at '{video_path}'."
        if p.suffix.lower() in _IMAGE_SUFFIXES:
            return [Moment(timestamp=0.0, frame_path=str(p))], ""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return (None, "can't pull scenes from video — ffmpeg isn't "
                          "available here. Send a frame/image instead.")
        tmpdir = tempfile.mkdtemp(prefix="film_scenes_")
        pattern = os.path.join(tmpdir, "scene_%04d.jpg")
        try:
            proc = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "info", "-i", str(p),
                 "-vf", f"select='gt(scene,{_SCENE_THRESHOLD})',showinfo",
                 "-vsync", "0", pattern],
                timeout=120, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True)
        except Exception:  # noqa: BLE001
            return None, "scene extraction failed for this video."
        stamps = re.findall(r"pts_time:([\d.]+)",
                            proc.stderr or "")
        frames = sorted(glob.glob(os.path.join(tmpdir, "scene_*.jpg")))
        pairs = [(float(t), f) for t, f in zip(stamps, frames)]
        if not pairs:
            # No scene changes detected → uniform fallback, one per 30s.
            return _uniform_moments(video_path, tmpdir, max_moments)
        # Cap to evenly-spread moments.
        if len(pairs) > max_moments:
            step = len(pairs) / max_moments
            pairs = [pairs[int(i * step)] for i in range(max_moments)]
        return [Moment(timestamp=t, frame_path=f) for t, f in pairs], ""
    except Exception as exc:  # noqa: BLE001
        _log.warning("scene extraction failed: %s", exc)
        return None, "scene extraction failed for this video."


def _uniform_moments(video_path: str, tmpdir: str, max_moments: int
                     ) -> tuple[list[Moment] | None, str]:
    """Fallback: one frame every 30s when no scene changes fire."""
    try:
        ffmpeg = shutil.which("ffmpeg") or ""
        pattern = os.path.join(tmpdir, "uni_%04d.jpg")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", video_path,
             "-vf", "fps=1/30", "-frames:v", str(max_moments), pattern],
            timeout=120, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        frames = sorted(glob.glob(os.path.join(tmpdir, "uni_*.jpg")))
        if not frames:
            return None, "couldn't extract any frames from this video."
        return [Moment(timestamp=i * 30.0, frame_path=f)
                for i, f in enumerate(frames)], ""
    except Exception:  # noqa: BLE001
        return None, "couldn't extract any frames from this video."


# ── per-moment analysis ────────────────────────────────────────────────

def _moment_prompt(game: str, moment: Moment) -> str:
    label = _GAME_LABELS.get(game, game)
    return "\n".join([
        f"You are watching a {label} VOD frame at timestamp {moment.label}.",
        "Be a precise visual sensor: describe what is factually visible.",
        "Then reply with exactly two lines:",
        "MISTAKE — <one specific mistake you can see>  (or MISTAKE — none)",
        "GOOD PLAY — <one specific good play you can see>  (or GOOD PLAY — none)",
        "One line each, nothing else."])


def _parse_moment_lines(text: str) -> tuple[list[str], list[str]]:
    """Seer text → (mistakes, highlights).

    Accepts ``MISTAKE: ...`` or ``MISTAKE — ...`` separators.
    """
    mistakes: list[str] = []
    highlights: list[str] = []
    try:
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if ":" in line:
                name, _, rest = line.partition(":")
            elif "—" in line:
                name, _, rest = line.partition("—")
            elif "-" in line:
                name, _, rest = line.partition("-")
            else:
                continue
            name = name.strip().upper()
            body = rest.strip().lstrip("—- ").strip()
            if not body or body.lower() in ("none", "none visible", "n/a"):
                continue
            if name == "MISTAKE":
                mistakes.append(body)
            elif name == "GOOD PLAY":
                highlights.append(body)
        return mistakes, highlights
    except Exception:  # noqa: BLE001
        return [], []


def _normalize(text: str, game: str) -> str:
    """Map an observation to a fingerprint pattern key, or ''."""
    low = text.lower()
    for key, desc in _GAME_PATTERNS.get(game, {}).items():
        first = desc.split("/")[0].strip().split()
        if len(first) >= 2 and " ".join(first[:2]) in low:
            return key
        if key in low:
            return key
    return ""


def _fix_for(pattern_key: str, game: str) -> str:
    desc = _GAME_PATTERNS.get(game, {}).get(pattern_key, "")
    fixes = {
        "overextend": "play off your team's position — if you're alone, you're wrong.",
        "crosshair": "keep crosshair at head height on common angles.",
        "economy": "save when the round is lost; force only with a plan.",
        "utility": "call your utility before you throw it.",
        "rotation": "rotate the moment info is confirmed, not after.",
        "cs": "prioritize last-hits over harass when the wave is under tower.",
        "map": "glance at the minimap every 3–5 seconds.",
        "objective": "set up vision 60s before the objective spawns.",
        "positioning": "stay at your effective range — don't face-check.",
        "cover": "never fight in the open; reposition to cover first.",
        "loot": "cap looting at 30s in hot drops.",
        "awareness": "scan the field before receiving — know your options.",
        "technique": "slow it down in practice until the technique holds.",
        "transition": "sprint back on turnovers — first 3 seconds decide it.",
    }
    if pattern_key in fixes:
        return fixes[pattern_key]
    if desc:
        return f"work on: {desc}."
    return "review this moment and drill the correction."


# ── public API ─────────────────────────────────────────────────────────

def analyze_vod(video_path: str, game: str, *,
                vision: VisionFn | None = None,
                store: "FilmStore | None" = None,
                exercise: str = "") -> FilmBreakdown:
    """Analyze a VOD → mistakes + highlights with timestamps. Never raises.

    ``vision(image_path, question) -> str`` is the injectable Seer seam.
    ``game``: valorant | lol | bgmi | sport | workout (workout delegates
    to #86's form coach and needs ``exercise``).
    """
    bid = "film_" + uuid.uuid4().hex[:8]
    try:
        game = (game or "").lower().strip()
        if game not in GAMES:
            return FilmBreakdown(
                id=bid, game=game or "?", available=False, note=(
                    "unknown game — I break down: "
                    f"{', '.join(g for g in GAMES)}."))
        if game == "workout":
            return _analyze_workout_vod(video_path, exercise, bid, store)

        vision = vision or _default_vision()
        moments, reason = _scenes_for(video_path)
        if moments is None:
            return FilmBreakdown(id=bid, game=game, video=video_path,
                                 available=False, note=reason,
                                 created_at=time.time())

        mistakes: list[PlayNote] = []
        highlights: list[PlayNote] = []
        seen_m = set()
        seen_h = set()
        for moment in moments:
            try:
                text = vision(moment.frame_path,
                              _moment_prompt(game, moment))
            except Exception as exc:  # noqa: BLE001
                _log.warning("vision failed on moment %s: %s",
                             moment.label, exc)
                continue
            ms, hs = _parse_moment_lines(text)
            for m in ms:
                key = m.lower()[:80]
                if key not in seen_m:
                    seen_m.add(key)
                    pat = _normalize(m, game)
                    mistakes.append(PlayNote(
                        kind="mistake", text=m,
                        timestamp=moment.timestamp,
                        fix=_fix_for(pat, game) if pat else ""))
            for h in hs:
                key = h.lower()[:80]
                if key not in seen_h:
                    seen_h.add(key)
                    highlights.append(PlayNote(
                        kind="highlight", text=h,
                        timestamp=moment.timestamp))

        if not mistakes and not highlights:
            return FilmBreakdown(
                id=bid, game=game, video=video_path, available=False,
                moments_used=len(moments),
                note=("the vision model didn't return usable moment reads "
                      "— try a clearer, higher-resolution video."),
                created_at=time.time())

        # Top 3 each: earliest occurrence wins (recency-neutral, simple).
        top_m = mistakes[:3]
        top_h = highlights[:3]
        coaching = _coaching_summary(game, top_m, top_h)
        breakdown = FilmBreakdown(
            id=bid, game=game, video=video_path, mistakes=top_m,
            highlights=top_h, coaching=coaching,
            moments_used=len(moments), created_at=time.time())
        if store is not None:
            try:
                store.save(breakdown)
            except Exception:  # noqa: BLE001
                _log.warning("film store save failed", exc_info=True)
        return breakdown
    except Exception as exc:  # noqa: BLE001
        _log.warning("analyze_vod failed: %s", exc)
        return FilmBreakdown(id=bid, game=game or "?", video=video_path,
                             available=False,
                             note="analysis failed unexpectedly.",
                             created_at=time.time())


def _analyze_workout_vod(video_path: str, exercise: str, bid: str,
                         store: "FilmStore | None") -> FilmBreakdown:
    """Workout VODs delegate to #86's form coach."""
    try:
        from ..health.form import analyze_form
    except Exception:  # noqa: BLE001
        return FilmBreakdown(id=bid, game="workout", video=video_path,
                             available=False,
                             note="workout analysis needs the form coach "
                                  "(nomorals.health.form), unavailable.",
                             created_at=time.time())
    if not exercise:
        return FilmBreakdown(
            id=bid, game="workout", video=video_path, available=False,
            note="workout VODs need an exercise — e.g. "
                 "'/film analyze <video> workout squat'.",
            created_at=time.time())
    fa = analyze_form(video_path, exercise)
    mistakes = [PlayNote(kind="mistake", text=i.observed, fix=i.fix)
                for i in fa.issues][:3]
    highlights = [PlayNote(kind="highlight",
                           text=f"{p} — solid") for p in fa.passes][:3]
    breakdown = FilmBreakdown(
        id=bid, game="workout", video=video_path, mistakes=mistakes,
        highlights=highlights, available=fa.available,
        coaching=(f"form coach read on {exercise}: {fa.band}."
                  if fa.available else ""),
        note="" if fa.available else fa.note,
        moments_used=fa.frames_used, created_at=time.time())
    if store is not None and fa.available:
        try:
            store.save(breakdown)
        except Exception:  # noqa: BLE001
            _log.warning("film store save failed", exc_info=True)
    return breakdown


def _coaching_summary(game: str, mistakes: list[PlayNote],
                      highlights: list[PlayNote]) -> str:
    bits: list[str] = []
    if mistakes:
        top = mistakes[0]
        bits.append(f"priority fix: {top.text.lower()}")
    if highlights:
        bits.append(f"keep doing: {highlights[0].text.lower()}")
    if len(mistakes) >= 2:
        bits.append("one thing at a time — drill the top mistake first.")
    return " ".join(bits) or "nothing conclusive from these frames."


def export_breakdown(bd: FilmBreakdown) -> str:
    """Shareable timestamped text summary."""
    lines = [f"🎬 {_GAME_LABELS.get(bd.game, bd.game)} VOD breakdown"]
    for n in bd.mistakes:
        lines.append(f"❌ [{n.label}] {n.text}")
        if n.fix:
            lines.append(f"   fix: {n.fix}")
    for n in bd.highlights:
        lines.append(f"✅ [{n.label}] {n.text}")
    if bd.coaching:
        lines.append(f"🧠 {bd.coaching}")
    return "\n".join(lines)


# ── fingerprints (longitudinal) ───────────────────────────────────────

def _film_db_path(db_path: str = "") -> str:
    if db_path:
        return db_path
    base = os.environ.get("NOMORALS_HOME", os.path.expanduser("~/.nomorals"))
    return os.path.join(base, "games", "film.db")


class FilmStore:
    """SQLite store for VOD breakdowns. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            p = _film_db_path(db_path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            self._db = sqlite3.connect(p)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS film_breakdowns (
                       id TEXT PRIMARY KEY, game TEXT, video TEXT,
                       mistakes TEXT, highlights TEXT, coaching TEXT,
                       created_at REAL)""")
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.warning("film: db unavailable, running empty",
                         exc_info=True)
            self._db = None

    def save(self, bd: FilmBreakdown) -> bool:
        if self._db is None:
            return False
        try:
            import json
            self._db.execute(
                "INSERT OR REPLACE INTO film_breakdowns VALUES "
                "(?, ?, ?, ?, ?, ?, ?)",
                (bd.id, bd.game, bd.video,
                 json.dumps([{"text": n.text, "timestamp": n.timestamp,
                              "fix": n.fix} for n in bd.mistakes]),
                 json.dumps([{"text": n.text, "timestamp": n.timestamp}
                             for n in bd.highlights]),
                 bd.coaching, bd.created_at or time.time()))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def list(self, game: str = "") -> list[dict[str, Any]]:
        if self._db is None:
            return []
        try:
            import json
            q = ("SELECT * FROM film_breakdowns ORDER BY created_at DESC"
                 if not game else
                 "SELECT * FROM film_breakdowns WHERE game = ? "
                 "ORDER BY created_at DESC")
            rows = (self._db.execute(q).fetchall() if not game
                    else self._db.execute(q, (game,)).fetchall())
            out = []
            for r in rows:
                out.append({
                    "id": r["id"], "game": r["game"],
                    "mistakes": json.loads(r["mistakes"] or "[]"),
                    "created_at": r["created_at"]})
            return out
        except Exception:  # noqa: BLE001
            return []

    def get(self, breakdown_id: str) -> FilmBreakdown | None:
        if self._db is None:
            return None
        try:
            import json
            r = self._db.execute(
                "SELECT * FROM film_breakdowns WHERE id = ?",
                (breakdown_id,)).fetchone()
            if r is None:
                return None
            return FilmBreakdown(
                id=r["id"], game=r["game"], video=r["video"],
                mistakes=[PlayNote(kind="mistake", text=m["text"],
                                   timestamp=m.get("timestamp", 0.0),
                                   fix=m.get("fix", ""))
                          for m in json.loads(r["mistakes"] or "[]")],
                highlights=[PlayNote(kind="highlight", text=h["text"],
                                     timestamp=h.get("timestamp", 0.0))
                            for h in json.loads(r["highlights"] or "[]")],
                coaching=r["coaching"] or "",
                created_at=r["created_at"])
        except Exception:  # noqa: BLE001
            return None


class FingerprintStore:
    """Per-user playstyle fingerprints from VOD history. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            p = _film_db_path(db_path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            self._db = sqlite3.connect(p)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS fingerprints (
                       user TEXT, game TEXT, pattern TEXT,
                       minute INTEGER, count INTEGER DEFAULT 0,
                       PRIMARY KEY (user, game, pattern, minute))""")
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.warning("film fingerprints: db unavailable",
                         exc_info=True)
            self._db = None

    def ingest(self, user: str, bd: FilmBreakdown) -> int:
        """Fold one breakdown's mistakes into the fingerprint. Returns new."""
        if self._db is None or not bd.available:
            return 0
        try:
            added = 0
            for n in bd.mistakes:
                pat = _normalize(n.text, bd.game)
                if not pat:
                    continue
                minute = int(n.timestamp // 60)
                self._db.execute(
                    "INSERT INTO fingerprints (user, game, pattern, minute, "
                    "count) VALUES (?, ?, ?, ?, 1) "
                    "ON CONFLICT(user, game, pattern, minute) "
                    "DO UPDATE SET count = count + 1",
                    (user, bd.game, pat, minute))
                added += 1
            self._db.commit()
            return added
        except Exception:  # noqa: BLE001
            return 0

    def fingerprint(self, user: str, game: str = "",
                    min_count: int = 2) -> list[str]:
        """Recurring patterns: 'you always overextend at minute 8'."""
        if self._db is None:
            return []
        try:
            q = ("SELECT game, pattern, minute, SUM(count) AS c "
                 "FROM fingerprints WHERE user = ? GROUP BY game, pattern, "
                 "minute HAVING c >= ? ORDER BY c DESC")
            rows = self._db.execute(q, (user, min_count)).fetchall()
            out = []
            for r in rows:
                if game and r["game"] != game:
                    continue
                desc = _GAME_PATTERNS.get(r["game"], {}).get(
                    r["pattern"], r["pattern"])
                out.append(
                    f"you tend to {desc} around minute {r['minute']} — "
                    f"seen in {r['c']} VOD reads.")
            return out
        except Exception:  # noqa: BLE001
            return []


# ── chat ───────────────────────────────────────────────────────────────

_USAGE = ("usage:\n"
          "  /film analyze <video> <game> [exercise] — VOD breakdown\n"
          "  /film games — supported games\n"
          "  /film fingerprint [game] — your recurring patterns\n"
          "  /film list [game] — past breakdowns\n"
          "  /film export <id> — shareable summary\n"
          "games: valorant, lol, bgmi, sport, workout")


def control_film(tail: str, *,
                 vision: VisionFn | None = None,
                 store: FilmStore | None = None,
                 fingerprints: FingerprintStore | None = None,
                 user: str = "owner") -> str:
    """Chat entry: /film. Owner-only at the dispatch layer. Never raises."""
    try:
        raw = (tail or "").strip()
        if not raw or raw.split()[0] in ("help", "?"):
            return _USAGE
        verb, _, rest = raw.partition(" ")
        verb = verb.lower()
        store = store or FilmStore()
        fps = fingerprints or FingerprintStore()

        if verb == "games":
            return "supported: " + ", ".join(GAMES)

        if verb == "analyze":
            parts = rest.split()
            if len(parts) < 2:
                return ("usage: /film analyze <video> <game> [exercise]\n"
                        f"games: {', '.join(GAMES)}")
            path, game = parts[0], parts[1]
            exercise = " ".join(parts[2:]) if game == "workout" else ""
            bd = analyze_vod(path, game, vision=vision, store=store,
                             exercise=exercise)
            if bd.available and game != "workout":
                fps.ingest(user, bd)
            head = f"analysis id: {bd.id}\n"
            return head + bd.format()

        if verb == "fingerprint":
            pats = fps.fingerprint(user, game=rest.strip())
            if not pats:
                return ("no recurring patterns yet — analyze 2+ VODs and "
                        "I'll spot your habits.")
            return "🎯 your playstyle fingerprint:\n" + "\n".join(
                f"• {p}" for p in pats)

        if verb == "list":
            rows = store.list(game=rest.strip())
            if not rows:
                return "no VOD breakdowns yet."
            return "\n".join(
                f"• {r['id']}: {r['game']} — "
                f"{len(r['mistakes'])} mistake(s)" for r in rows[:10])

        if verb == "export":
            bd = store.get(rest.strip())
            if bd is None:
                return f"no breakdown '{rest.strip()}' saved."
            return export_breakdown(bd)

        return _USAGE
    except Exception as exc:  # noqa: BLE001
        _log.warning("control_film failed: %s", exc)
        return "film study hit a snag — try again."
