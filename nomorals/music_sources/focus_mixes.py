"""Focus mixes — the MusicForProgramming editorial layer (52-websites #31).

MusicForProgramming is deliberately minimal: a numbered series of
~1-hour ambient/electronic/instrumental mixes, curated by Datassette
to sustain concentration while programming. One page, a list of
mixes, each with a tracklist; free streaming and MP3 download; an
RSS feed so the series behaves like a podcast. No accounts, no
customisation, no algorithm — the curation *is* the product.

Devon's equivalent is the *editorial* layer over pieces that already
exist: the ``music_sources/`` registry + play spine, the DJ and
long-form mix machinery (``media/dj_engine.py`` /
``media/dj_long.py``), and the focus-session ledger
(``focus/sessions.py``). This module is the organ that ties them
together:

* :class:`FocusMix` — ``{id, title, curator(owner|devon), tracks[],
  duration, tags(tempo/energy), sessions_played}``; mixes assembled
  from the legal registry and the owner's library, sequenced by the
  DJ engine.
* :class:`FocusMixStore` — durable JSON at
  ``<home>/data/focus_mixes.json``; seeds when missing, quarantines
  a corrupt store aside and re-seeds, atomic writes.
* The Music For Programming series itself sits in the registry as a
  seed (``musicforprogramming``) and its RSS feed is the real seed
  source: :func:`refresh_from_series` pulls the live episode list —
  79 episodes at last check, each with a direct MP3 enclosure —
  and records each episode as a one-track mix. No episode is ever
  invented; an episode the feed does not list does not exist here.
* :func:`assemble_mix` builds a mix from the owner's own listening
  history and library (data she legitimately has) — the taste model
  is the owner's: liked/kept tracks rank up, skipped rank down.
  Sequencing goes through an injected ``planner``; the default is a
  deterministic energy arc (warm-up → peak → cool-down) and the
  DJ-engine planner (``plan_long_mix`` with stem-aware transitions
  and drop awareness) is injected by the tool / chat layers, which
  are the ones allowed to reach into ``media/``.
* :func:`start_focus_session_with_mix` starts a focus session with
  the mix attached and the first track playing — the session ledger
  records what played, so the ledger entry links session ↔ mix.

Honest limits, carried from the report: the site's value is one
human's taste over 15 years; Devon can't import Datassette's ear.
Her equivalent curates from the owner's own listening history and
library, and can *include* sources like this series in the registry
as seeds — the registry grows, the taste model is the owner's.

Only the standard library is imported here: ``music_sources/`` is a
layer-0 package and must not reach up into ``media/`` (layer 4).
The DJ engine is injected, never imported.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "FocusTrack",
    "FocusMix",
    "FocusMixStore",
    "MUSIC_FOR_PROGRAMMING_SITE",
    "MUSIC_FOR_PROGRAMMING_FEED",
    "MUSIC_FOR_PROGRAMMING_SOURCE_ID",
    "MOOD_TAGS",
    "parse_focus_mix_request",
    "parse_rss",
    "refresh_from_series",
    "assemble_mix",
    "start_focus_session_with_mix",
    "find_mix",
    "render_mix",
    "format_mix_reply",
    "format_duration",
    "default_planner",
    "taste_scores",
]

#: The series itself.
MUSIC_FOR_PROGRAMMING_SITE = "https://musicforprogramming.net/"
MUSIC_FOR_PROGRAMMING_FEED = "https://musicforprogramming.net/rss.xml"
MUSIC_FOR_PROGRAMMING_SOURCE_ID = "musicforprogramming"

#: What a mood word asks the mix to lean towards. The mix records the
#: *requested* mood in its tags; whether the pool could honour it is
#: recorded in the mix notes, never silently pretended.
MOOD_TAGS: dict[str, dict[str, str]] = {
    "darker": {"mood": "dark", "energy": "low", "tempo": "slow"},
    "lighter": {"mood": "light", "energy": "low"},
    "deep": {"mood": "deep", "energy": "low"},
    "driving": {"mood": "driving", "energy": "high", "tempo": "fast"},
    "calm": {"mood": "calm", "energy": "low", "tempo": "slow"},
    "ambient": {"mood": "ambient", "energy": "low"},
}

_MOOD_WORDS = ("darker", "lighter", "deeper", "driving", "calm",
               "ambient", "dark", "light", "deep")

_MOOD_ALIASES = {"dark": "darker", "light": "lighter", "deeper": "deep"}


def _norm_mood(word: str | None) -> str | None:
    if not word:
        return None
    word = word.strip().lower()
    return _MOOD_ALIASES.get(word, word) if word in _MOOD_WORDS else None


def format_duration(seconds: float) -> str:
    """``7380`` → ``2h 3m``; ``900`` → ``15m``; ``45`` → ``45s``."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m"
    return f"{secs}s"


def _slug(prefix: str, *parts: str) -> str:
    raw = "|".join(parts)
    return prefix + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _norm_key(text: str) -> str:
    """Normalise a title/artist for best-effort taste joins."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


# ── data shapes ──────────────────────────────────────────────────────

@dataclass
class FocusTrack:
    """One track inside a focus mix. Descriptor data only."""

    id: str
    title: str
    artist: str = ""
    duration_s: float = 180.0
    bpm: float | None = None
    energy: float = 0.5
    key: str = ""
    tags: list[str] = field(default_factory=list)
    source_id: str = ""
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "artist": self.artist,
                "duration_s": self.duration_s, "bpm": self.bpm,
                "energy": self.energy, "key": self.key,
                "tags": list(self.tags), "source_id": self.source_id,
                "url": self.url}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FocusTrack:
        return cls(
            id=str(d.get("id", "")),
            title=str(d.get("title", d.get("id", ""))),
            artist=str(d.get("artist", "")),
            duration_s=float(d.get("duration_s") or 180.0),
            bpm=d.get("bpm"),
            energy=float(d.get("energy", 0.5)),
            key=str(d.get("key", "")),
            tags=[str(t) for t in (d.get("tags") or [])],
            source_id=str(d.get("source_id", "")),
            url=str(d.get("url", "")))


@dataclass
class FocusMix:
    """A standing focus mix: an editorial series entry tied to work.

    ``curator`` is who assembled it in Devon's library: ``"devon"``
    for series-seeded and auto-assembled mixes, ``"owner"`` for mixes
    the owner built or edited. ``source`` names where the tracks came
    from (``"musicforprogramming"``, ``"library"``, ``"history"``,
    ``"custom"``). ``plan`` holds the DJ-engine plan dict when the
    mix was sequenced by ``plan_long_mix`` (transitions, drop
    positions); the default arc planner leaves it empty.
    """

    id: str
    title: str
    curator: str = "devon"
    tracks: list[FocusTrack] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    sessions_played: int = 0
    created_at: float = field(default_factory=time.time)
    source: str = "library"
    plan: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return sum(t.duration_s for t in self.tracks)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title,
                "curator": self.curator,
                "tracks": [t.to_dict() for t in self.tracks],
                "tags": dict(self.tags),
                "sessions_played": self.sessions_played,
                "created_at": self.created_at, "source": self.source,
                "plan": self.plan, "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FocusMix:
        return cls(
            id=str(d.get("id", "")),
            title=str(d.get("title", d.get("id", ""))),
            curator=str(d.get("curator", "devon")),
            tracks=[FocusTrack.from_dict(t)
                    for t in (d.get("tracks") or [])],
            tags={str(k): str(v) for k, v in (d.get("tags") or {}).items()},
            sessions_played=int(d.get("sessions_played") or 0),
            created_at=float(d.get("created_at") or 0.0),
            source=str(d.get("source", "library")),
            plan=dict(d.get("plan") or {}),
            notes=[str(n) for n in (d.get("notes") or [])])


# ── store ────────────────────────────────────────────────────────────

class FocusMixStore:
    """Durable JSON store for focus mixes.

    Seeds when missing (empty — mixes are assembled from the owner's
    library or refreshed from the series, never invented). A corrupt
    store is quarantined aside and re-seeded empty, never fatal.
    """

    def __init__(self, home: str | Path | None = None,
                 path: str | Path | None = None) -> None:
        if path is not None:
            self.path = Path(path)
        elif home is not None:
            self.path = Path(home) / "data" / "focus_mixes.json"
        else:
            self.path = (Path.home() / ".nomorals" / "data"
                         / "focus_mixes.json")
        self.mixes: dict[str, FocusMix] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            try:
                stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
                aside = self.path.with_name(
                    self.path.name + f".corrupt-{stamp}")
                os.replace(self.path, aside)
            except OSError:
                pass
            return
        for row in (raw.get("mixes") or []):
            try:
                mix = FocusMix.from_dict(row)
            except (TypeError, ValueError):
                continue
            if mix.id:
                self.mixes[mix.id] = mix

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"mixes": [m.to_dict()
                                     for m in self.mixes.values()]},
                          fh, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def add(self, mix: FocusMix) -> FocusMix:
        self.mixes[mix.id] = mix
        self._save()
        return mix

    def get(self, mix_id: str) -> FocusMix | None:
        return self.mixes.get(mix_id)

    def list(self) -> list[FocusMix]:
        return sorted(self.mixes.values(),
                      key=lambda m: (m.created_at, m.id))

    def remove(self, mix_id: str) -> bool:
        if mix_id not in self.mixes:
            return False
        del self.mixes[mix_id]
        self._save()
        return True

    def record_played(self, mix_id: str) -> FocusMix | None:
        mix = self.mixes.get(mix_id)
        if mix is None:
            return None
        mix.sessions_played += 1
        self._save()
        return mix


def find_mix(store: FocusMixStore, ref: str) -> FocusMix | None:
    """Find a mix by id, id prefix, title substring, or ``episode N``."""
    ref = (ref or "").strip()
    if not ref:
        return None
    mix = store.get(ref)
    if mix is not None:
        return mix
    ep = re.match(r"(?i)^episode\s+(\d+)$", ref)
    if ep:
        for candidate in store.list():
            if candidate.id == f"mfp-{int(ep.group(1))}":
                return candidate
    lowered = ref.lower()
    for candidate in store.list():
        if candidate.id.startswith(ref):
            return candidate
    for candidate in store.list():
        if lowered in candidate.title.lower():
            return candidate
    return None


# ── NL intent ────────────────────────────────────────────────────────

_MIX_RE = re.compile(
    r"(?:put on|play|start|give me)(?: a| an| the)?"
    r" (coding|focus|programming|deep[ -]work|concentration) mix"
    r"(?:,? (darker|lighter|deeper|driving|calm|ambient|dark|light|deep))?",
    re.IGNORECASE)
_MUSIC_RE = re.compile(
    r"focus music(?:,? (darker|lighter|deeper|driving|calm|ambient|dark|light|deep))?",
    re.IGNORECASE)
_WITH_MUSIC_RE = re.compile(
    r"start(?: a)? focus(?: session)? with (?:music|a mix|a coding mix|a focus mix)",
    re.IGNORECASE)
_EPISODE_RE = re.compile(r"(?:put on|play) episode (\d+)", re.IGNORECASE)
_BARE_MIX_RE = re.compile(
    r"(coding|focus|programming|deep[ -]work|concentration) mix"
    r"(?:,? (darker|lighter|deeper|driving|calm|ambient|dark|light|deep))?",
    re.IGNORECASE)


def parse_focus_mix_request(text: str) -> dict[str, Any] | None:
    """Parse a focus-mix request from free text.

    ``"put on a coding mix"`` / ``"focus music, darker"`` /
    ``"start a focus session with music"`` / ``"put on episode 79"``
    → ``{"kind": "mix"|"episode", "mood": ..., "episode": ...}``.
    Anything else → None. Deliberately narrow: ``"play despacito"``
    is a song request, not a mix request, and ``"start a focus
    session"`` without music is the plain ``/focus start`` path.
    """
    text = (text or "").strip()
    if not text or text.startswith("/"):
        return None
    ep = _EPISODE_RE.search(text)
    if ep:
        return {"kind": "episode", "mood": None,
                "episode": int(ep.group(1))}
    hit = _MIX_RE.search(text)
    if hit:
        return {"kind": "mix", "mood": _norm_mood(hit.group(2)),
                "episode": None}
    hit = _BARE_MIX_RE.search(text)
    if hit:
        return {"kind": "mix", "mood": _norm_mood(hit.group(2)),
                "episode": None}
    hit = _MUSIC_RE.search(text)
    if hit:
        return {"kind": "mix", "mood": _norm_mood(hit.group(1)),
                "episode": None}
    if _WITH_MUSIC_RE.search(text):
        return {"kind": "mix", "mood": None, "episode": None}
    return None


# ── the series: RSS ──────────────────────────────────────────────────

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _parse_duration(raw: str | None) -> float:
    """``"4:00:00"`` / ``"60:00"`` / ``"3600"`` → seconds."""
    if not raw:
        return 0.0
    parts = str(raw).strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return 0.0
    total = 0.0
    for num in nums:
        total = total * 60.0 + num
    return total


def parse_rss(xml_text: str) -> list[dict[str, Any]]:
    """Parse the Music For Programming RSS feed into episode dicts.

    Each episode: ``{number, title, artist, url, duration_s, tags,
    pub_date}``. Never raises — bad XML is an empty episode list.
    """
    try:
        root = ET.fromstring(xml_text or "")
    except ET.ParseError:
        return []
    episodes: list[dict[str, Any]] = []
    for item in root.iter():
        if _local(item.tag) != "item":
            continue
        fields: dict[str, str] = {}
        for child in item:
            name = _local(child.tag)
            if name in ("title", "link", "guid", "pubDate",
                        "duration", "keywords"):
                fields[name] = (child.text or "").strip()
            if name == "enclosure":
                fields["enclosure_url"] = (
                    child.get("url") or "").strip()
        title = fields.get("title", "")
        hit = re.match(r"(?i)episode\s+(\d+)\s*:\s*(.+)", title)
        number = int(hit.group(1)) if hit else None
        name = hit.group(2).strip() if hit else title
        url = fields.get("enclosure_url") or fields.get("guid") or ""
        if not title or not url:
            continue
        keywords = [k.strip().lower()
                    for k in fields.get("keywords", "").split(",")
                    if k.strip()]
        episodes.append({
            "number": number,
            "title": title,
            "name": name,
            "artist": "Music For Programming",
            "url": url,
            "link": fields.get("link", ""),
            "duration_s": _parse_duration(fields.get("duration")),
            "tags": keywords,
            "pub_date": fields.get("pubDate", ""),
        })
    return episodes


def _default_fetcher(url: str, timeout_s: float = 20.0) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Devon focus mixes)"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", "replace")


def refresh_from_series(
        store: FocusMixStore, *,
        fetcher: Callable[[str], str] | None = None,
        limit: int | None = None) -> dict[str, Any]:
    """Pull the live series feed and record new episodes as mixes.

    One mix per episode: a single track — the episode's MP3 — with
    the feed's duration and keywords as tags, ``curator="devon"``,
    ``source="musicforprogramming"``. Episodes already in the store
    are skipped. Never raises: a dead feed is ``{"added": 0, ...,
    "error": ...}``.
    """
    fetch = fetcher or _default_fetcher
    try:
        xml_text = fetch(MUSIC_FOR_PROGRAMMING_FEED)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {"added": 0, "total": 0,
                "error": f"couldn't reach the series feed: {exc}"}
    episodes = parse_rss(xml_text)
    if not episodes:
        return {"added": 0, "total": 0,
                "error": "the series feed came back empty or unparseable"}
    added = 0
    for episode in episodes:
        number = episode.get("number")
        if number is None:
            continue
        mix_id = f"mfp-{number}"
        if store.get(mix_id) is not None:
            continue
        duration = float(episode.get("duration_s") or 0.0)
        track = FocusTrack(
            id=f"{mix_id}-audio",
            title=episode["title"],
            artist=episode.get("artist", "Music For Programming"),
            duration_s=duration if duration > 0 else 3600.0,
            energy=0.4,
            tags=list(episode.get("tags") or ["ambient"]),
            source_id=MUSIC_FOR_PROGRAMMING_SOURCE_ID,
            url=episode["url"])
        mix = FocusMix(
            id=mix_id,
            title=episode["title"],
            curator="devon",
            tracks=[track],
            tags={"mood": "ambient", "energy": "low", "tempo": "slow"},
            source=MUSIC_FOR_PROGRAMMING_SOURCE_ID,
            notes=[f"episode {number} of Music For Programming, "
                   "from the series RSS feed"])
        store.add(mix)
        added += 1
        if limit is not None and added >= limit:
            break
    return {"added": added, "total": len(episodes), "error": ""}


# ── taste: the owner's own ───────────────────────────────────────────

def taste_scores(feedback_rows: list[dict[str, Any]] | None
                 ) -> dict[str, float]:
    """Score keys from DJ feedback rows (``dj_engine.log_feedback``).

    ``liked`` / ``kept`` count for a track, ``played`` counts a
    little, ``skipped`` counts against. Keys are the normalised
    title and the raw track id, so a library row joins on either.
    Best-effort by design: feedback that names nothing in the
    library simply scores nothing.
    """
    scores: dict[str, float] = {}
    for row in (feedback_rows or []):
        if not isinstance(row, dict):
            continue
        event = str(row.get("event", "")).strip().lower()
        weight = {"liked": 1.0, "kept": 1.0, "played": 0.25,
                  "skipped": -1.0}.get(event)
        if weight is None:
            continue
        track_id = str(row.get("track_id", "") or "")
        title = str(row.get("title", "") or "")
        for key in {track_id, _norm_key(title)}:
            if key:
                scores[key] = scores.get(key, 0.0) + weight
    return scores


def _track_score(track: FocusTrack, scores: dict[str, float]) -> float:
    best = 0.0
    for key in (track.id, _norm_key(track.title),
                _norm_key(f"{track.artist} {track.title}")):
        if key and key in scores:
            best = max(best, scores[key])
    return best


def default_planner(tracks: list[FocusTrack], target_minutes: float,
                    scores: dict[str, float] | None = None
                    ) -> tuple[list[FocusTrack], dict[str, Any]]:
    """Deterministic energy arc, stdlib only: warm-up → peak → cool-down.

    The fallback rung: the same arc shape ``plan_long_mix`` uses,
    without stem-aware transitions or drop awareness. Taste scores
    order tracks inside each third; a seeded shuffle (seeded from
    the track ids, so the same pool always plans the same mix) keeps
    the order from being just chart order.
    """
    import random as _random

    scores = scores or {}
    if not tracks:
        return [], {}
    ordered = sorted(
        tracks,
        key=lambda t: (t.energy, -_track_score(t, scores)))
    n = len(ordered)
    third = max(1, n // 3)
    thirds = [ordered[:third], ordered[third:max(1, 2 * third)],
              ordered[max(1, 2 * third):]]
    thirds[1] = sorted(thirds[1],
                       key=lambda t: (-t.energy, -_track_score(t, scores)))
    seed = int(hashlib.sha1(
        "|".join(sorted(t.id for t in tracks)).encode("utf-8")
    ).hexdigest()[:8], 16)
    rng = _random.Random(seed)
    arc: list[FocusTrack] = []
    for part in thirds:
        chunk = list(part)
        rng.shuffle(chunk)
        arc.extend(chunk)
    target_s = float(target_minutes) * 60.0
    total = sum(t.duration_s for t in arc)
    chosen = list(arc)
    while len(chosen) > 1 and total > target_s * 1.15:
        dropped = chosen.pop()
        total -= dropped.duration_s
    guard = 0
    while total < target_s * 0.8 and guard < 4 and arc:
        chosen.extend(list(arc))
        total = sum(t.duration_s for t in chosen)
        guard += 1
    return chosen, {}


# ── assembly ─────────────────────────────────────────────────────────

def _library_candidates(library: Any, limit: int = 200
                        ) -> list[FocusTrack]:
    """Pull candidate tracks from the owner's library (duck-typed).

    ``MusicLibrary`` (``media/library.py``): ``favorites`` /
    ``top_played`` / ``history`` — rows carry ``path`` / ``title`` /
    ``artist`` / ``album`` / ``duration`` / ``plays``. A library
    that raises, or returns nothing usable, contributes nothing —
    the caller then answers honestly instead of inventing tracks.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for method in ("favorites", "top_played", "history"):
        getter = getattr(library, method, None)
        if not callable(getter):
            continue
        try:
            found = getter(limit)
        except Exception:  # noqa: BLE001 - a broken library rung is data
            continue
        if not isinstance(found, list):
            continue
        for row in found:
            if not isinstance(row, dict):
                continue
            key = str(row.get("path") or row.get("title") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(row)
        if len(rows) >= limit:
            break
    tracks: list[FocusTrack] = []
    for row in rows:
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        path = str(row.get("path") or "")
        duration = row.get("duration")
        try:
            duration_s = float(duration) if duration else 180.0
        except (TypeError, ValueError):
            duration_s = 180.0
        tracks.append(FocusTrack(
            id=_slug("ft-", path or title,
                     str(row.get("artist") or "")),
            title=title,
            artist=str(row.get("artist") or ""),
            duration_s=duration_s if duration_s > 0 else 180.0,
            energy=0.5,
            tags=[str(row.get("album") or "")] if row.get("album") else [],
            source_id=str(row.get("source") or "library"),
            url=path))
    return tracks


def _mood_boost(track: FocusTrack, mood: str | None) -> float:
    """Soft mood preference: a track whose own words carry the mood
    ranks higher. Library rows carry no genre, so this is a title /
    album keyword match — a soft lean, never a filter that empties
    the pool."""
    if not mood:
        return 0.0
    words = {mood, MOOD_TAGS.get(mood, {}).get("mood", "")}
    hay = _norm_key(track.title + " " + " ".join(track.tags))
    return 1.0 if any(w and w in hay for w in words) else 0.0


def assemble_mix(store: FocusMixStore, *,
                 library: Any = None,
                 feedback_rows: list[dict[str, Any]] | None = None,
                 tags: dict[str, str] | None = None,
                 target_minutes: float = 60.0,
                 title: str | None = None,
                 curator: str = "devon",
                 planner: Callable[..., Any] | None = None,
                 seed: int | None = None) -> FocusMix:
    """Assemble a mix from the owner's library and store it.

    Candidates come from the library (favorites first, then most
    played, then recent history); the owner's taste — liked/kept up,
    skipped down — orders them inside the energy arc, and the
    ``planner`` sequences them (default: the stdlib arc planner;
    pass the DJ-engine planner from the tool / chat layers for
    stem-aware transitions and drop awareness). Raises
    :class:`ValueError` when there is nothing to assemble from —
    the caller renders that honestly instead of inventing tracks.
    """
    tags = dict(tags or {})
    mood = tags.get("mood")
    candidates = _library_candidates(library) if library is not None else []
    if not candidates:
        raise ValueError(
            "no tracks in your library yet — nothing to assemble a "
            "mix from. Play some music first, or refresh the Music "
            "For Programming series and put on an episode.")
    scores = taste_scores(feedback_rows)
    boosted = sorted(
        candidates,
        key=lambda t: (-_mood_boost(t, mood), -_track_score(t, scores),
                       t.title))
    if mood and not any(_mood_boost(t, mood) > 0 for t in candidates):
        note = (f"no {mood} tracks in your library — the mix leans "
                "on what you play most instead")
    else:
        note = ""
    plan = planner or default_planner
    ordered, plan_dict = plan(boosted, float(target_minutes), scores)
    if not ordered:
        raise ValueError("the planner returned no tracks — nothing to play")
    mix_title = title or (
        f"Focus mix — {mood}" if mood else "Focus mix")
    mix = FocusMix(
        id=_slug("fm-", mix_title, str(time.time())),
        title=mix_title,
        curator=curator,
        tracks=list(ordered),
        tags=tags,
        source="library",
        plan=dict(plan_dict or {}),
        notes=[note] if note else [])
    return store.add(mix)


# ── session start: the ledger link ───────────────────────────────────

def start_focus_session_with_mix(
        timer: Any, store: FocusMixStore, mix: FocusMix, *,
        player: Callable[[FocusTrack], dict[str, Any]] | None = None,
        label: str = "") -> dict[str, Any]:
    """Start a focus session with a mix attached and play it.

    The session ledger records the mix (``mix_id`` / ``mix_title``
    on the session row), the mix's ``sessions_played`` ticks up, and
    the first track goes through ``player`` — the caller injects the
    real play path (chat: the play spine; CLI: local playback). With
    no player the mix is planned and the session runs, and the play
    result says so honestly instead of pretending audio went out.
    """
    try:
        session = timer.start(label=label or mix.title,
                              mix_id=mix.id, mix_title=mix.title)
    except ValueError:
        # A session is already running — attach the mix to it instead
        # of failing: the owner asked for music with their work, and
        # the running session is the work.
        active = timer.active() if hasattr(timer, "active") else None
        if active is None:
            raise
        session = timer.attach_mix(None, mix.id, mix.title)
    store.record_played(mix.id)
    first = mix.tracks[0] if mix.tracks else None
    if first is None:
        play = {"status": "empty",
                "detail": "the mix has no tracks — nothing to play"}
    elif player is None:
        play = {"status": "planned",
                "detail": "no player wired — the mix is planned and "
                          "the session is running"}
    else:
        try:
            play = player(first)
        except Exception as exc:  # noqa: BLE001 - the session stands
            play = {"status": "error",
                    "detail": f"couldn't play the first track: {exc}"}
    return {"session": session, "mix": mix, "play": play}


# ── rendering ────────────────────────────────────────────────────────

def _stamp(seconds: float) -> str:
    """``0`` → ``00:00``; ``3661`` → ``1:01:01`` (tracklist stamps)."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_mix(mix: FocusMix) -> str:
    """Timestamped tracklist, e.g. ``00:00 — Artist — Title``.

    When the mix carries a DJ-engine plan, the transition kind rides
    on each line (``[drop_cut]``); otherwise the stamps are the
    cumulative track durations.
    """
    lines = [f"🎧 {mix.title} — {format_duration(mix.duration_s)} · "
             f"{len(mix.tracks)} track(s) · played in "
             f"{mix.sessions_played} session(s)"]
    if mix.tags:
        lines.append("tags: " + ", ".join(
            f"{k}={v}" for k, v in sorted(mix.tags.items())))
    if mix.source == MUSIC_FOR_PROGRAMMING_SOURCE_ID:
        lines.append("seed: Music For Programming — "
                     "musicforprogramming.net")
    transitions = {}
    for step in (mix.plan or {}).get("transitions") or []:
        if isinstance(step, dict) and step.get("to_id"):
            transitions[step["to_id"]] = step
    cursor = 0.0
    for i, track in enumerate(mix.tracks):
        stamp = _stamp(cursor)
        who = f"{track.artist} — " if track.artist else ""
        kind = ""
        if i > 0:
            step = transitions.get(track.id)
            if step:
                kind = f" [{step.get('kind', '')}]"
        lines.append(f"  {stamp} — {who}{track.title}{kind}")
        cursor += track.duration_s
    for note in mix.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def format_mix_reply(mix: FocusMix, session: Any = None,
                     play: dict[str, Any] | None = None) -> str:
    """The chat reply for a mix: tracklist + session + play lines."""
    text = render_mix(mix)
    if session is not None:
        label = getattr(session, "label", "") or mix.title
        text += (f"\nfocus session started ({label}) — "
                 f"{getattr(session, 'id', '')}")
    if play is not None:
        status = str(play.get("status", ""))
        detail = str(play.get("detail", "") or "")
        if status == "sent":
            text += f"\n▶ playing: {detail}"
        elif status == "planned":
            text += f"\n{detail}"
        elif status in ("error", "empty"):
            text += f"\ncouldn't play it yet: {detail}"
        elif detail:
            text += f"\n{detail}"
    return text
