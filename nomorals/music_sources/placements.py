"""Music placements: which songs play in which TV episode, film or game.

The TuneFind core, Devon-native: a placement relation
(work -> episode/film -> scene -> song) with a verified-vs-community
trust model, over a SQLite store next to the fingerprint DB. Every
record carries its provenance and confidence, and the reply always
shows them — an unverified entry is labelled, never presented as
confirmed.

Provenance levels:

* ``supervisor`` — submitted by the work's music supervisor; the
  verified rung (confidence 1.0).
* ``owner`` — the owner's own watching, added by him or on his word
  (confidence 0.95).
* ``curated`` — taken from public soundtrack listings and episode
  guides (confidence 0.8). Labelled, never supervisor-verified.
* ``community`` — a community submission (confidence 0.5, moving
  +0.1 per confirm and -0.1 per deny, clamped 0.1..0.9). It becomes
  verified when confirms outnumber denies by 3 or more.

The seed set is curated from public soundtrack listings whose
work/track/artist pairings were verified against live sources at
build time (Stranger Things S4, the Sopranos finale, Pulp Fiction,
Guardians of the Galaxy, Wayne's World, Reservoir Dogs, Titanic,
Rocky III, Ghost, Saturday Night Fever, The Bodyguard).

The killer Devon-native move TuneFind cannot do: the owner plays a
clip, :func:`nomorals.audio.fingerprint.identify` names the track
from her own local recognition memory, and
:func:`placements_for_clip` answers where that track was used. The
identifier is injected by the caller (the tool, the chat handler,
the CLI) — this module never imports the fingerprint organ itself,
so the layering stays clean: stdlib only, top to bottom.

Every lookup is honest: an unknown work, an unrecorded episode, or
an identified track with no placements says so and names what would
help (adding the placement, or naming the work more precisely).
Nothing here invents a soundtrack entry.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Placement",
    "PlacementStore",
    "SEED_PLACEMENTS",
    "COMMUNITY_VERIFY_MARGIN",
    "default_store_path",
    "open_store",
    "parse_soundtrack_query",
    "lookup",
    "placements_for_clip",
    "format_placement_reply",
    "format_reverse_reply",
    "play_handoff",
]

#: A community placement counts as verified once confirms lead denies
#: by this many votes.
COMMUNITY_VERIFY_MARGIN = 3

_DDL = (
    "CREATE TABLE IF NOT EXISTS placements ("
    "id INTEGER PRIMARY KEY,"
    "work_id TEXT NOT NULL,"
    "work_kind TEXT NOT NULL,"
    "work_title TEXT NOT NULL,"
    "season INTEGER NOT NULL DEFAULT -1,"
    "episode INTEGER NOT NULL DEFAULT -1,"
    "episode_title TEXT NOT NULL DEFAULT '',"
    "scene_note TEXT NOT NULL DEFAULT '',"
    "timestamp_hint TEXT NOT NULL DEFAULT '',"
    "track_title TEXT NOT NULL,"
    "track_artist TEXT NOT NULL,"
    "track_ref TEXT NOT NULL DEFAULT '',"
    "provenance TEXT NOT NULL DEFAULT 'curated',"
    "confidence REAL NOT NULL DEFAULT 0.8,"
    "confirms INTEGER NOT NULL DEFAULT 0,"
    "denies INTEGER NOT NULL DEFAULT 0,"
    "UNIQUE(work_id, season, episode, track_title, track_artist));"
    "CREATE INDEX IF NOT EXISTS idx_placements_track "
    "ON placements(track_title, track_artist);"
    "CREATE INDEX IF NOT EXISTS idx_placements_work "
    "ON placements(work_id, season, episode)"
)

_PREFIXES = (
    "what song plays in ",
    "what songs play in ",
    "what song is in ",
    "songs in ",
    "songs from ",
    "soundtrack of ",
    "soundtrack for ",
    "music from ",
    "music in ",
)

_SXE_RE = re.compile(r"\bs(\d+)e(\d+)\s*$", re.IGNORECASE)
_SEASON_EPISODE_RE = re.compile(
    r"\bseason\s+(\d+)\s+episode\s+(\d+)\s*$", re.IGNORECASE)
_EPISODE_RE = re.compile(r"\bepisode\s+(\d+)\s*$", re.IGNORECASE)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower())
    return slug.strip("-") or "unknown"


def _norm(text: str) -> str:
    """Normalise a title for matching: lower, no leading 'the', punctuation gone."""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.startswith("the "):
        cleaned = cleaned[4:]
    return cleaned


@dataclass
class Placement:
    """One song placed in one work at one spot."""

    work_id: str = ""
    work_kind: str = "film"  # tv | film | game
    work_title: str = ""
    season: int = -1  # -1 = not applicable (film / whole work)
    episode: int = -1
    episode_title: str = ""
    scene_note: str = ""
    timestamp_hint: str = ""
    track_title: str = ""
    track_artist: str = ""
    track_ref: str = ""  # MusicBrainz / AcoustID id, when known
    provenance: str = "curated"  # supervisor | owner | community | curated
    confidence: float = 0.8
    confirms: int = 0
    denies: int = 0

    def __post_init__(self) -> None:
        if not self.work_id and self.work_title:
            self.work_id = _slug(self.work_title)
        if self.work_kind not in ("tv", "film", "game"):
            self.work_kind = "film"
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def is_verified(self) -> bool:
        """True only for supervisor, owner, or community past the vote margin."""
        if self.provenance in ("supervisor", "owner"):
            return True
        if self.provenance == "community":
            return (self.confirms - self.denies) >= COMMUNITY_VERIFY_MARGIN
        return False

    def trust_label(self) -> str:
        """The honest label the reply shows beside this placement."""
        if self.provenance == "supervisor":
            return "verified by the music supervisor"
        if self.provenance == "owner":
            return "from your own watching"
        if self.provenance == "curated":
            return "curated listing — not supervisor-verified"
        if self.is_verified:
            return f"community — confirmed by {self.confirms} votes"
        return (f"community — not yet confirmed "
                f"({self.confirms}✓ {self.denies}✗)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "work_kind": self.work_kind,
            "work_title": self.work_title,
            "season": self.season,
            "episode": self.episode,
            "episode_title": self.episode_title,
            "scene_note": self.scene_note,
            "timestamp_hint": self.timestamp_hint,
            "track_title": self.track_title,
            "track_artist": self.track_artist,
            "track_ref": self.track_ref,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "confirms": self.confirms,
            "denies": self.denies,
            "verified": self.is_verified,
            "trust_label": self.trust_label(),
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Placement:
        # noqa: SIM118 — sqlite3.Row iterates VALUES, not keys;
        # .keys() is the only way to iterate the column names.
        return cls(**{key: row[key] for key in row.keys()  # noqa: SIM118
                       if key != "id"})


def _seed_row(work_kind: str, work_title: str, season: int, episode: int,
              episode_title: str, scene_note: str, track_title: str,
              track_artist: str) -> Placement:
    return Placement(
        work_id=_slug(work_title), work_kind=work_kind, work_title=work_title,
        season=season, episode=episode, episode_title=episode_title,
        scene_note=scene_note, track_title=track_title,
        track_artist=track_artist, provenance="curated", confidence=0.8)


#: Curated seed set. Work/track/artist pairings verified against live
#: soundtrack listings at build time; scene notes state only what
#: those listings state.
SEED_PLACEMENTS: list[Placement] = [
    _seed_row("tv", "Stranger Things", 4, 1, "Chapter One: The Hellfire Club",
              "Max walks through school listening on her headphones",
              "Running Up That Hill (A Deal with God)", "Kate Bush"),
    _seed_row("tv", "Stranger Things", 4, 4, "Dear Billy",
              "Vecna confronts Max — the song pulls her back",
              "Running Up That Hill (A Deal with God)", "Kate Bush"),
    _seed_row("tv", "Stranger Things", 4, 9, "The Piggyback",
              "Eddie's guitar solo in the Upside Down draws off the demobats",
              "Master of Puppets", "Metallica"),
    _seed_row("tv", "Stranger Things", 4, 9, "The Piggyback",
              "Max listens before facing Vecna",
              "Running Up That Hill (A Deal with God)", "Kate Bush"),
    _seed_row("tv", "The Sopranos", 6, 21, "Made in America",
              "Tony picks it on the diner jukebox as the family arrives; cut to black",
              "Don't Stop Believin'", "Journey"),
    _seed_row("film", "Pulp Fiction", -1, -1, "",
              "opening credits", "Misirlou", "Dick Dale"),
    _seed_row("film", "Pulp Fiction", -1, -1, "",
              "plays over Mia Wallace's introduction",
              "Son of a Preacher Man", "Dusty Springfield"),
    _seed_row("film", "Guardians of the Galaxy", -1, -1, "",
              "opening — Peter Quill dancing on Morag",
              "Come and Get Your Love", "Redbone"),
    _seed_row("film", "Wayne's World", -1, -1, "",
              "Wayne, Garth and friends headbang in the hatchback",
              "Bohemian Rhapsody", "Queen"),
    _seed_row("film", "Reservoir Dogs", -1, -1, "",
              "the torture scene",
              "Stuck in the Middle with You", "Stealers Wheel"),
    _seed_row("film", "Titanic", -1, -1, "",
              "end credits", "My Heart Will Go On", "Céline Dion"),
    _seed_row("film", "Rocky III", -1, -1, "",
              "the film's theme song, written for Rocky III",
              "Eye of the Tiger", "Survivor"),
    _seed_row("film", "Ghost", -1, -1, "",
              "the film's signature love theme",
              "Unchained Melody", "The Righteous Brothers"),
    _seed_row("film", "Saturday Night Fever", -1, -1, "",
              "the film's defining disco anthem",
              "Stayin' Alive", "Bee Gees"),
    _seed_row("film", "The Bodyguard", -1, -1, "",
              "the film's theme song",
              "I Will Always Love You", "Whitney Houston"),
]


def default_store_path(home: str | Path | None = None) -> Path:
    """The durable store: <home>/data/soundtrack_placements.sqlite."""
    base = Path(home) if home else Path.home() / ".nomorals"
    return base / "data" / "soundtrack_placements.sqlite"


class PlacementStore:
    """SQLite placement catalogue. Never raises on a lookup."""

    def __init__(self, path: str | Path = "") -> None:
        self.path = str(path) if path else str(default_store_path())
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connect()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def seed(self) -> int:
        """Load the curated seed set. Idempotent — returns new rows added."""
        added = 0
        for placement in SEED_PLACEMENTS:
            if self.add(placement):
                added += 1
        return added

    def add(self, placement: Placement) -> bool:
        """Insert one placement. False when it is already recorded."""
        with self._lock:
            try:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO placements "
                    "(work_id, work_kind, work_title, season, episode, "
                    "episode_title, scene_note, timestamp_hint, track_title, "
                    "track_artist, track_ref, provenance, confidence, "
                    "confirms, denies) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (placement.work_id, placement.work_kind,
                     placement.work_title, placement.season,
                     placement.episode, placement.episode_title,
                     placement.scene_note, placement.timestamp_hint,
                     placement.track_title, placement.track_artist,
                     placement.track_ref, placement.provenance,
                     placement.confidence, placement.confirms,
                     placement.denies))
                self._conn.commit()
                return cur.rowcount == 1
            except Exception:  # noqa: BLE001 — a failed add is data, not a crash
                return False

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM placements").fetchone()
            return int(row["n"])

    def all_works(self) -> list[dict[str, str]]:
        """Distinct works, as work_id / work_title / work_kind dicts."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT work_id, work_title, work_kind "
                "FROM placements ORDER BY work_title").fetchall()
            return [dict(row) for row in rows]

    def resolve_work(self, title: str) -> list[dict[str, str]]:
        """Works matching ``title`` — exact normalised match first, then substring."""
        want = _norm(title)
        if not want:
            return []
        works = self.all_works()
        exact = [w for w in works if _norm(w["work_title"]) == want]
        if exact:
            return exact
        return [w for w in works if want in _norm(w["work_title"])]

    def search_works(self, query: str) -> list[str]:
        """Work titles containing the query (for did-you-mean)."""
        want = _norm(query)
        if not want:
            return []
        return [w["work_title"] for w in self.all_works()
                if want in _norm(w["work_title"])]

    def songs_for_episode(self, work_title: str, season: int,
                          episode: int) -> list[Placement]:
        """Songs in one TV episode, in catalogue order."""
        works = self.resolve_work(work_title)
        if not works:
            return []
        work_id = works[0]["work_id"]
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM placements WHERE work_id = ? "
                "AND season = ? AND episode = ? ORDER BY id",
                (work_id, season, episode)).fetchall()
            return [Placement.from_row(r) for r in rows]

    def songs_for_work(self, work_title: str,
                       kind: str | None = None) -> list[Placement]:
        """Every song recorded for a work, episode order first."""
        works = self.resolve_work(work_title)
        if kind:
            works = [w for w in works if w["work_kind"] == kind]
        if not works:
            return []
        work_id = works[0]["work_id"]
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM placements WHERE work_id = ? "
                "ORDER BY season, episode, id",
                (work_id,)).fetchall()
            return [Placement.from_row(r) for r in rows]

    def works_for_track(self, track_title: str,
                        artist: str | None = None) -> list[dict[str, Any]]:
        """Reverse index: every work a song plays in."""
        want_title = _norm(track_title)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM placements ORDER BY work_title, season, "
                "episode").fetchall()
        out = []
        for row in rows:
            p = Placement.from_row(row)
            if _norm(p.track_title) != want_title:
                continue
            if artist and _norm(p.track_artist) != _norm(artist):
                continue
            out.append({**p.to_dict(), "work": p.work_title})
        return out

    def vote(self, work_id: str, track_title: str,
             confirm: bool = True) -> Placement | None:
        """Confirm/deny a community placement; returns the updated row or None."""
        want = _norm(track_title)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM placements WHERE work_id = ?",
                (work_id,)).fetchall()
            target = None
            for row in rows:
                p = Placement.from_row(row)
                if _norm(p.track_title) == want:
                    target = p
                    break
            if target is None or target.provenance != "community":
                return None
            target.confirms += 1 if confirm else 0
            target.denies += 0 if confirm else 1
            target.confidence = max(
                0.1, min(0.9, 0.5 + 0.1 * target.confirms
                         - 0.1 * target.denies))
            self._conn.execute(
                "UPDATE placements SET confirms = ?, denies = ?, "
                "confidence = ? WHERE work_id = ? AND track_title = ?",
                (target.confirms, target.denies, target.confidence,
                 work_id, target.track_title))
            self._conn.commit()
            return target


def open_store(home: str | Path | None = None) -> PlacementStore:
    """Open (creating and seeding if needed) the durable store."""
    store = PlacementStore(default_store_path(home))
    store.seed()
    return store


def parse_soundtrack_query(text: str) -> dict[str, Any]:
    """Split a free-form query into work + optional season/episode."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    lowered = cleaned.lower()
    for prefix in _PREFIXES:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            lowered = cleaned.lower()
            break
    season: int | None = None
    episode: int | None = None
    m = _SXE_RE.search(cleaned)
    if m:
        season, episode = int(m.group(1)), int(m.group(2))
        cleaned = _SXE_RE.sub("", cleaned).strip()
    else:
        m = _SEASON_EPISODE_RE.search(cleaned)
        if m:
            season, episode = int(m.group(1)), int(m.group(2))
            cleaned = _SEASON_EPISODE_RE.sub("", cleaned).strip()
        else:
            m = _EPISODE_RE.search(cleaned)
            if m:
                episode = int(m.group(1))
                cleaned = _EPISODE_RE.sub("", cleaned).strip()
    return {"work": cleaned.strip(" -–—"), "season": season,
            "episode": episode}


def _episode_label(season: int | None, episode: int | None) -> str:
    if season is not None and episode is not None:
        return f"S{season}E{episode}"
    if episode is not None:
        return f"E{episode}"
    return ""


def lookup(store: PlacementStore, query: str) -> dict[str, Any]:
    """Answer 'what songs play in <work> [S#E#]?'. Never raises."""
    try:
        parsed = parse_soundtrack_query(query)
        work = parsed["work"]
        season, episode = parsed["season"], parsed["episode"]
        if not work:
            return {"ok": False,
                    "error": "give me a show or a film — "
                             "/soundtrack <show|film> [S1E2]"}
        works = store.resolve_work(work)
        if not works:
            suggestions = store.search_works(work)
            hint = ""
            if suggestions:
                hint = " Did you mean: " + ", ".join(suggestions[:3]) + "?"
            return {"ok": False,
                    "error": (f"“{work}” is not in the catalogue yet — add its "
                              f"soundtrack with /soundtrack add.{hint}"),
                    "suggestions": suggestions}
        matched = works[0]
        if season is not None and episode is not None:
            if matched["work_kind"] != "tv":
                return {"ok": False,
                        "error": (f"“{matched['work_title']}” is a "
                                  f"{matched['work_kind']} — ask for its songs "
                                  f"without an episode."),
                        "suggestions": []}
            placements = store.songs_for_episode(
                matched["work_title"], season, episode)
            if not placements:
                recorded = store.songs_for_work(matched["work_title"])
                eps = sorted({(p.season, p.episode) for p in recorded
                              if p.season >= 0 and p.episode >= 0})
                label = _episode_label(season, episode)
                detail = ""
                if eps:
                    detail = (" Recorded episodes: " + ", ".join(
                        f"S{s}E{e}" for s, e in eps) + ".")
                return {"ok": False,
                        "error": (f"no songs recorded for "
                                  f"{matched['work_title']} {label} yet.{detail}"),
                        "suggestions": [f"S{s}E{e}" for s, e in eps]}
            return {"ok": True, "work": matched["work_title"],
                    "kind": matched["work_kind"], "season": season,
                    "episode": episode,
                    "episode_title": placements[0].episode_title,
                    "placements": [p.to_dict() for p in placements]}
        placements = store.songs_for_work(matched["work_title"])
        if not placements:
            return {"ok": False,
                    "error": (f"no songs recorded for "
                              f"“{matched['work_title']}” yet."),
                    "suggestions": []}
        return {"ok": True, "work": matched["work_title"],
                "kind": matched["work_kind"], "season": None,
                "episode": None, "episode_title": "",
                "placements": [p.to_dict() for p in placements]}
    except Exception as exc:  # noqa: BLE001 — a lookup never raises
        return {"ok": False, "error": f"soundtrack lookup failed: {exc}"}


def placements_for_clip(
        store: PlacementStore, path: str,
        *,
        identify_fn: Callable[[str], dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Identify a clip, then answer where that track was used.

    ``identify_fn`` is injected by the caller — the production leg is
    :func:`nomorals.audio.fingerprint.identify`, wired in by the tool,
    the chat handler and the CLI. With no identifier wired the clip
    leg refuses honestly instead of guessing. Never raises.
    """
    try:
        if identify_fn is None:
            return {"ok": False,
                    "error": ("clip identification needs an identifier — "
                              "wire audio.fingerprint.identify via "
                              "identify_fn")}
        try:
            found = identify_fn(path)
        except Exception as exc:  # noqa: BLE001 — identification failure is data
            return {"ok": False,
                    "error": f"could not identify that clip: {exc}"}
        if not isinstance(found, dict) or not found.get("ok"):
            reason = ""
            if isinstance(found, dict):
                reason = str(found.get("reason") or found.get("error") or "")
            return {"ok": False,
                    "error": ("could not identify that clip"
                              + (f" — {reason}" if reason else "")
                              + "; the track has to be in the fingerprint "
                                "library first")}
        title = str(found.get("title") or "")
        artist = str(found.get("artist") or "")
        if not title:
            return {"ok": False,
                    "error": "that clip identified with no title attached"}
        works = store.works_for_track(title, artist or None)
        if not works:
            return {"ok": False,
                    "error": (f"identified “{title}”"
                              + (f" — {artist}" if artist else "")
                              + ", but no placements recorded for it yet — "
                                "add one with /soundtrack add."),
                    "track": title, "artist": artist}
        return {"ok": True, "track": title, "artist": artist,
                "score": found.get("score"), "placements": works}
    except Exception as exc:  # noqa: BLE001 — a clip lookup never raises
        return {"ok": False, "error": f"soundtrack clip lookup failed: {exc}"}


def play_handoff(track_title: str, track_artist: str) -> str:
    """The way to hear it: straight into Devon's own /play chain."""
    if track_artist:
        return f"/play {track_title} — {track_artist}"
    return f"/play {track_title}"


def format_reverse_reply(works: list[dict[str, Any]]) -> str:
    """Render a song -> works answer: each placement names its work."""
    lines = ["🎵 Where that song plays:"]
    for p in works:
        where = p.get("work") or p.get("work_title") or ""
        season, episode = p.get("season"), p.get("episode")
        if season is not None and season >= 0 and episode is not None \
                and episode >= 0:
            where += f" — S{season}E{episode}"
        scene = p.get("scene_note") or ""
        label = p.get("trust_label") or ""
        line = f"• {p.get('track_title')} — {p.get('track_artist')} in {where}"
        if scene:
            line += f" — {scene}"
        if label:
            line += f" [{label}]"
        lines.append(line)
        lines.append(f"  Play it: {play_handoff(p.get('track_title') or '', p.get('track_artist') or '')}")
    return "\n".join(lines)


def format_placement_reply(answer: dict[str, Any]) -> str:
    """One rendering for chat and CLI. Provenance is always shown."""
    if not answer.get("ok"):
        error = str(answer.get("error") or "soundtrack lookup failed")
        suggestions = answer.get("suggestions") or []
        if suggestions and "Did you mean" not in error:
            error += " Did you mean: " + ", ".join(suggestions[:3]) + "?"
        return error
    placements = answer.get("placements") or []
    work = answer.get("work") or ""
    kind = answer.get("kind") or ""
    season, episode = answer.get("season"), answer.get("episode")
    episode_title = answer.get("episode_title") or ""
    if not work and answer.get("track"):
        work = str(answer.get("track"))
        if answer.get("artist"):
            work += f" — {answer.get('artist')}"
    header = f"🎵 {work}"
    if kind == "tv" and season is not None and episode is not None:
        header += f" — S{season}E{episode}"
        if episode_title:
            header += f" “{episode_title}”"
    elif kind:
        header += f" ({kind})"
    lines = [header]
    for p in placements:
        if isinstance(p, Placement):
            p = p.to_dict()
        scene = p.get("scene_note") or ""
        label = p.get("trust_label") or ""
        line = f"• {p.get('track_title')} — {p.get('track_artist')}"
        if scene:
            line += f" — {scene}"
        if label:
            line += f" [{label}]"
        lines.append(line)
        lines.append(f"  Play it: {play_handoff(p.get('track_title') or '', p.get('track_artist') or '')}")
    return "\n".join(lines)
