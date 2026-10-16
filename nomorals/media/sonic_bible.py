"""Sonic bible + album consistency — the sound of an artist, written down.

Tier 2. A :class:`SonicBible` is the durable contract for one artist /
one album: key(s), tempo window, sound palette, vocal chain, mastering
targets, reference descriptors, and the rules that make track N sound
like it belongs next to track 1. :func:`check_consistency` scores a
track's descriptor set against the bible and reports drift per axis,
so an album stays coherent without every track sounding identical.

Everything is descriptors (JSON-serializable), never audio: the bible
steers the LLM composer, the rhythm engine, and the mastering stage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["SonicBible", "TrackDescriptor", "check_consistency",
           "album_consistency", "register"]

# Key distance on the circle of fifths (C=0 ...): 0 = same, 1 = neighbour.
_CIRCLE = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6,
           "C#": 7, "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5,
           "Gb": -6, "Cb": -7}


@dataclass
class TrackDescriptor:
    """What a finished (or planned) track sounds like — descriptors only."""

    title: str = ""
    key: str = "C"            # e.g. "C", "F#", "Bb" (+ "m" suffix for minor)
    mode: str = "major"       # "major" | "minor"
    bpm: float = 100.0
    genre: str = "afrobeats"
    energy: float = 0.6       # 0..1
    palette: tuple[str, ...] = ()  # instrument/texture words
    vocal_style: str = ""
    language: str = "en"
    loudness_lufs: float | None = None


@dataclass
class SonicBible:
    """The sound contract for an artist / album / era."""

    artist: str = ""
    album: str = ""
    keys: tuple[str, ...] = ("C",)     # allowed keys (relative major/minor OK)
    mode: str = "any"                  # "major" | "minor" | "any"
    bpm_range: tuple[float, float] = (90.0, 120.0)
    genres: tuple[str, ...] = ("afrobeats",)
    energy_range: tuple[float, float] = (0.3, 0.9)
    palette: tuple[str, ...] = ()      # core textures every track shares
    palette_optional: tuple[str, ...] = ()  # may appear, not required
    vocal_style: str = ""
    languages: tuple[str, ...] = ("en",)
    loudness_lufs: float = -14.0      # streaming target
    rules: tuple[str, ...] = ()       # free-text house rules
    references: tuple[str, ...] = ()  # reference tracks (descriptors, not audio)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("keys", "genres", "palette", "palette_optional",
                  "languages", "rules", "references"):
            d[k] = list(d[k])
        d["bpm_range"] = list(d["bpm_range"])
        d["energy_range"] = list(d["energy_range"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SonicBible":
        d = dict(d or {})
        return cls(
            artist=str(d.get("artist", "") or ""),
            album=str(d.get("album", "") or ""),
            keys=tuple(d.get("keys", ("C",)) or ("C",)),
            mode=str(d.get("mode", "any") or "any"),
            bpm_range=tuple(d.get("bpm_range", (90.0, 120.0)) or (90.0, 120.0)),  # type: ignore[arg-type]
            genres=tuple(d.get("genres", ("afrobeats",)) or ("afrobeats",)),
            energy_range=tuple(d.get("energy_range", (0.3, 0.9)) or (0.3, 0.9)),  # type: ignore[arg-type]
            palette=tuple(d.get("palette", ()) or ()),
            palette_optional=tuple(d.get("palette_optional", ()) or ()),
            vocal_style=str(d.get("vocal_style", "") or ""),
            languages=tuple(d.get("languages", ("en",)) or ("en",)),
            loudness_lufs=float(d.get("loudness_lufs", -14.0)),
            rules=tuple(d.get("rules", ()) or ()),
            references=tuple(d.get("references", ()) or ()),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "SonicBible":
        return cls.from_dict(json.loads(text))


def _key_distance(a: str, b: str) -> int:
    a = (a or "").replace("m", "").strip()
    b = (b or "").replace("m", "").strip()
    if a not in _CIRCLE or b not in _CIRCLE:
        return 99
    return abs(_CIRCLE[a] - _CIRCLE[b])


def check_consistency(bible: SonicBible,
                      track: TrackDescriptor) -> dict[str, Any]:
    """Score one track against the bible. Returns per-axis verdicts + score.

    Axes: key (circle-of-fifths distance to nearest allowed key; a
    relative major/minor swap counts as in-key), mode, bpm window,
    genre family, energy window, palette overlap (core palette words
    present), language, loudness (within ±2 LUFS of target when known).
    Score is the mean of axis scores; drift lists the failing axes.
    """
    axes: dict[str, float] = {}
    notes: list[str] = []
    # key: relative major/minor — same tonic circle position counts
    best_key = min((_key_distance(track.key, k) for k in bible.keys),
                   default=99)
    axes["key"] = 1.0 if best_key == 0 else (0.7 if best_key == 1 else
                                            (0.4 if best_key == 2 else 0.0))
    if axes["key"] < 1.0:
        notes.append(f"key {track.key} is {best_key} fifth(s) from bible keys {list(bible.keys)}")
    # mode
    if bible.mode == "any" or track.mode == bible.mode:
        axes["mode"] = 1.0
    elif {track.mode, bible.mode} == {"major", "minor"}:
        axes["mode"] = 0.7  # relative major/minor swap is coherent
        notes.append(f"mode {track.mode} vs bible {bible.mode} (relative swap — coherent)")
    else:
        axes["mode"] = 0.0
    # bpm
    lo, hi = bible.bpm_range
    axes["bpm"] = 1.0 if lo <= track.bpm <= hi else 0.0
    if axes["bpm"] == 0.0:
        notes.append(f"bpm {track.bpm} outside bible window {lo}–{hi}")
    # genre
    axes["genre"] = 1.0 if track.genre in bible.genres else 0.3
    if axes["genre"] < 1.0:
        notes.append(f"genre {track.genre!r} not in bible genres {list(bible.genres)}")
    # energy
    elo, ehi = bible.energy_range
    axes["energy"] = 1.0 if elo <= track.energy <= ehi else 0.0
    if axes["energy"] == 0.0:
        notes.append(f"energy {track.energy} outside bible window {elo}–{ehi}")
    # palette: fraction of core palette present
    if bible.palette:
        have = {p.lower() for p in track.palette}
        core = {p.lower() for p in bible.palette}
        axes["palette"] = len(have & core) / len(core)
        if axes["palette"] < 1.0:
            notes.append(f"missing core palette: {sorted(core - have)}")
    else:
        axes["palette"] = 1.0
    # language
    axes["language"] = 1.0 if track.language in bible.languages else 0.5
    # loudness
    if track.loudness_lufs is None:
        axes["loudness"] = 1.0  # unknown, not penalised
    else:
        axes["loudness"] = 1.0 if abs(track.loudness_lufs - bible.loudness_lufs) <= 2.0 else 0.0
        if axes["loudness"] == 0.0:
            notes.append(f"loudness {track.loudness_lufs} LUFS vs bible target {bible.loudness_lufs}")
    score = sum(axes.values()) / len(axes) if axes else 1.0
    return {"track": track.title, "score": round(score, 3),
            "axes": {k: round(v, 3) for k, v in axes.items()},
            "drift": [k for k, v in axes.items() if v < 1.0],
            "notes": notes,
            "consistent": score >= 0.7}


def album_consistency(bible: SonicBible,
                      tracks: list[TrackDescriptor]) -> dict[str, Any]:
    """Whole-album view: mean score + per-track results + worst drift."""
    results = [check_consistency(bible, t) for t in tracks]
    mean = (sum(r["score"] for r in results) / len(results)) if results else 1.0
    return {"album": bible.album, "artist": bible.artist,
            "tracks": len(results), "mean_score": round(mean, 3),
            "consistent": mean >= 0.7 and all(r["consistent"] for r in results),
            "results": results}


def register(registry: Any) -> None:
    """Tool registration: ``sonic_bible`` — consistency checking."""
    from ..core.policy import Capability

    @registry.register(
        "sonic_bible",
        description=(
            "Sonic bible + album consistency: action=check (bible JSON, "
            "track JSON → per-axis consistency score) | album (bible JSON, "
            "tracks JSON list → whole-album coherence). The bible is the "
            "artist's sound contract: keys, tempo window, palette, vocal "
            "style, loudness target."
        ),
        capability=Capability.FS_WRITE,
    )
    def sonic_bible(action: str = "check", bible: str = "",
                    track: str = "", tracks: str = "") -> dict:
        import json as _json
        try:
            b = SonicBible.from_dict(_json.loads(bible) if bible else {})
            if action == "album":
                ts = [TrackDescriptor(**t) for t in _json.loads(tracks or "[]")]
                return {"ok": True, **album_consistency(b, ts)}
            t = TrackDescriptor(**(_json.loads(track) if track else {}))
            return {"ok": True, **check_consistency(b, t)}
        except Exception as exc:  # noqa: BLE001
            from ..core.errors import ToolError
            raise ToolError(f"sonic_bible: {exc}")
