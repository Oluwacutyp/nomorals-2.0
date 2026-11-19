"""SoundCraft session model — the .scraft document (lane 107).

Ported from storytold/soundcraft's ``crates/model`` (Session,
Track, Clip, Bus, Source) into Devon's Python. A .scraft file is a
ZIP package: ``session.json`` (canonical JSON, format "soundcraft",
version 1) + ``sources/<id>.wav`` parts + ``meta.json``.

Fidelity discipline, the same habit as the documents organs: an
unedited document writes back byte-identical (the raw input bytes
are kept until the first mutation); every mutation marks the entity
dirty, and ``fidelity_report`` compares every untouched entity
against the bytes it was read from — a changed untouched entity is
reported, never silently rewritten.
"""

from __future__ import annotations

import io
import json
import struct
import wave
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SoundError, SoundErrorKind

__all__ = [
    "Bus",
    "Clip",
    "Note",
    "SessionDocument",
    "Source",
    "Track",
]

FORMAT = "soundcraft"
VERSION = 1
EXTENSION = "scraft"
MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_PART_BYTES = 32 * 1024 * 1024
MAX_CLIPS = 100_000
MAX_TRACKS = 1_024
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def _canonical(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


@dataclass
class Source:
    """An audio source embedded in the session (mono PCM)."""

    id: int
    name: str
    file: str
    sample_rate: int
    samples: np.ndarray = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "file": self.file,
            "sample_rate": self.sample_rate,
            "frames": int(len(self.samples)),
        }


@dataclass
class Clip:
    id: int
    track_id: int
    name: str
    source_id: int | None  # None for note clips (instrument tracks)
    start: int  # session sample the clip starts at
    offset: int  # source sample the clip starts at
    duration: int  # samples of source the clip covers
    gain_db: float = 0.0
    notes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "track_id": self.track_id,
            "name": self.name,
            "source_id": self.source_id,
            "start": self.start,
            "offset": self.offset,
            "duration": self.duration,
            "gain_db": self.gain_db,
        }
        if self.notes:
            data["notes"] = self.notes
        return data


@dataclass
class Track:
    id: int
    name: str
    kind: str  # "audio" | "instrument"
    gain_db: float = 0.0
    pan: float = 0.0  # -1 left .. +1 right
    mute: bool = False
    solo: bool = False
    inserts: list[dict[str, Any]] = field(default_factory=list)
    sends: list[dict[str, Any]] = field(default_factory=list)
    automation: dict[str, list[list[float]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "gain_db": self.gain_db,
            "pan": self.pan,
            "mute": self.mute,
            "solo": self.solo,
            "inserts": self.inserts,
            "sends": self.sends,
            "automation": self.automation,
        }


@dataclass
class Bus:
    id: int
    name: str
    kind: str  # "aux" | "master"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "kind": self.kind}


@dataclass
class Note:
    start: int
    duration: int
    pitch: int  # MIDI note number
    velocity: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "duration": self.duration,
            "pitch": self.pitch,
            "velocity": self.velocity,
        }


def read_wav_mono(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV to mono float64 samples. Stereo is refused honestly."""
    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.getnframes()
            data = wf.readframes(frames)
    except (wave.Error, EOFError, struct.error) as exc:
        raise SoundError(f"not a readable WAV: {exc}",
                         kind=SoundErrorKind.BAD_FILE) from exc
    if channels != 1:
        raise SoundError(
            f"stereo sources are not supported in v1 (got {channels} "
            "channels) — split to mono first",
            kind=SoundErrorKind.UNSUPPORTED)
    if width == 1:
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
        return (arr - 128.0) / 128.0, rate
    if width == 2:
        arr = np.frombuffer(data, dtype="<i2").astype(np.float64)
        return arr / 32768.0, rate
    if width == 3:
        raw3 = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        vals = (raw3[:, 0].astype(np.int64)
                | (raw3[:, 1].astype(np.int64) << 8)
                | (raw3[:, 2].astype(np.int64) << 16))
        vals = np.where(vals >= 1 << 23, vals - (1 << 24), vals)
        return vals.astype(np.float64) / 8388608.0, rate
    if width == 4:
        arr = np.frombuffer(data, dtype="<i4").astype(np.float64)
        return arr / 2147483648.0, rate
    raise SoundError(f"unsupported WAV sample width {width}",
                     kind=SoundErrorKind.UNSUPPORTED)


def write_wav_mono(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float samples as 16-bit PCM WAV (deterministic)."""
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return buf.getvalue()


class SessionDocument:
    """A SoundCraft session: tracks, clips, buses, sources, markers."""

    def __init__(self, data: dict[str, Any],
                 sources: dict[int, Source],
                 raw: bytes | None = None):
        self._data = data
        self._sources = sources
        self._raw = raw
        self._dirty: set[int] = set()
        self._added: set[int] = set()
        self._removed: set[int] = set()
        self._original: dict[str, Any] | None = (
            deepcopy(data) if raw is not None else None)

    # -- construction -------------------------------------------------
    @classmethod
    def create(cls, sample_rate: int = 48000, title: str = "") -> SessionDocument:
        data: dict[str, Any] = {
            "format": FORMAT,
            "version": VERSION,
            "title": title,
            "sample_rate": int(sample_rate),
            "tracks": [],
            "clips": [],
            "buses": [{"id": 0, "name": "Master", "kind": "master"}],
            "sources": [],
            "next_id": 1,
        }
        return cls(data, {})

    @classmethod
    def open(cls, path: str | Path) -> SessionDocument:
        raw = Path(path).read_bytes()
        return cls.from_bytes(raw)

    @classmethod
    def from_bytes(cls, raw: bytes) -> SessionDocument:
        if len(raw) > MAX_PACKAGE_BYTES:
            raise SoundError("session package exceeds 64MB",
                             kind=SoundErrorKind.TOO_LARGE)
        try:
            zin = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as exc:
            raise SoundError(f"not a soundcraft session: {exc}",
                             kind=SoundErrorKind.BAD_FILE) from exc
        try:
            names = zin.namelist()
            if "session.json" not in names:
                raise SoundError("not a soundcraft session: no session.json",
                                 kind=SoundErrorKind.BAD_FILE)
            manifest = zin.read("session.json")
            if len(manifest) > MAX_PART_BYTES:
                raise SoundError("session.json exceeds 32MB",
                                 kind=SoundErrorKind.TOO_LARGE)
            try:
                data = json.loads(manifest.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise SoundError(f"session.json is not valid JSON: {exc}",
                                 kind=SoundErrorKind.BAD_FILE) from exc
        except SoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SoundError(f"not a soundcraft session: {exc}",
                             kind=SoundErrorKind.BAD_FILE) from exc
        if data.get("format") != FORMAT:
            raise SoundError(
                f"not a soundcraft session: format is "
                f"{data.get('format')!r}, expected 'soundcraft'",
                kind=SoundErrorKind.BAD_FORMAT)
        version = data.get("version")
        if version != VERSION:
            raise SoundError(
                f"session file version {version} is not supported "
                f"(this SoundCraft reads version {VERSION})",
                kind=SoundErrorKind.UNSUPPORTED)
        sources: dict[int, Source] = {}
        for entry in data.get("sources", []):
            file = entry["file"]
            try:
                part = zin.read(file)
            except KeyError as exc:
                raise SoundError(f"missing source part {file}",
                                 kind=SoundErrorKind.BAD_FILE) from exc
            samples, sr = read_wav_mono(part)
            sources[int(entry["id"])] = Source(
                id=int(entry["id"]), name=entry["name"], file=file,
                sample_rate=sr, samples=samples)
        return cls(data, sources, raw=raw)

    # -- accessors ----------------------------------------------------
    @property
    def title(self) -> str:
        return str(self._data.get("title", ""))

    @property
    def sample_rate(self) -> int:
        return int(self._data["sample_rate"])

    @property
    def tracks(self) -> list[Track]:
        return [Track(**{k: deepcopy(v) for k, v in t.items()})
                for t in self._data["tracks"]]

    @property
    def clips(self) -> list[Clip]:
        return [Clip(**{k: deepcopy(v) for k, v in c.items()})
                for c in self._data["clips"]]

    @property
    def buses(self) -> list[Bus]:
        return [Bus(**b) for b in self._data["buses"]]

    @property
    def sources(self) -> dict[int, Source]:
        return dict(self._sources)

    def track(self, track_id: int) -> Track:
        for raw in self._data["tracks"]:
            if raw["id"] == track_id:
                return Track(**{k: deepcopy(v) for k, v in raw.items()})
        raise SoundError(f"no track {track_id}",
                         kind=SoundErrorKind.NOT_FOUND)

    def clip(self, clip_id: int) -> Clip:
        for raw in self._data["clips"]:
            if raw["id"] == clip_id:
                return Clip(**{k: deepcopy(v) for k, v in raw.items()})
        raise SoundError(f"no clip {clip_id}",
                         kind=SoundErrorKind.NOT_FOUND)

    def bus(self, bus_id: int) -> Bus:
        for raw in self._data["buses"]:
            if raw["id"] == bus_id:
                return Bus(**raw)
        raise SoundError(f"no bus {bus_id}",
                         kind=SoundErrorKind.NOT_FOUND)

    def source_samples(self, source_id: int) -> np.ndarray:
        source = self._sources.get(source_id)
        if source is None:
            raise SoundError(f"no source {source_id}",
                             kind=SoundErrorKind.NOT_FOUND)
        return source.samples

    def length_samples(self) -> int:
        end = 0
        for clip in self._data["clips"]:
            end = max(end, int(clip["start"]) + int(clip["duration"]))
        return end

    # -- mutation bookkeeping (used by SessionEditor) ------------------
    def _alloc_id(self) -> int:
        nid = int(self._data.get("next_id", 1))
        self._data["next_id"] = nid + 1
        return nid

    def _mark_dirty(self, entity_id: int) -> None:
        self._dirty.add(entity_id)
        self._raw = None

    def _mark_added(self, entity_id: int) -> None:
        self._added.add(entity_id)
        self._raw = None

    def _mark_removed(self, entity_id: int) -> None:
        self._removed.add(entity_id)
        self._dirty.discard(entity_id)
        self._raw = None

    def fidelity_report(self) -> dict[str, Any]:
        """Which entities changed, added, removed — and whether any
        untouched entity drifted from what was read."""
        untouched_changed = 0
        if self._original is not None:
            orig_clips = {c["id"]: c for c in self._original["clips"]}
            orig_tracks = {t["id"]: t for t in self._original["tracks"]}
            for clip in self._data["clips"]:
                if clip["id"] in self._dirty or clip["id"] in self._added:
                    continue
                if orig_clips.get(clip["id"]) != clip:
                    untouched_changed += 1
            for track in self._data["tracks"]:
                if track["id"] in self._dirty or track["id"] in self._added:
                    continue
                if orig_tracks.get(track["id"]) != track:
                    untouched_changed += 1
        return {
            "changed": sorted(self._dirty),
            "added": sorted(self._added),
            "removed": sorted(self._removed),
            "untouched_changed": untouched_changed,
        }

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def to_bytes(self) -> bytes:
        if self._raw is not None:
            return self._raw
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            zout.writestr(
                zipfile.ZipInfo("session.json", _ZIP_EPOCH),
                _canonical(self._data))
            meta = {"created_by": "devon-soundcraft",
                    "format": FORMAT, "version": VERSION}
            zout.writestr(
                zipfile.ZipInfo("meta.json", _ZIP_EPOCH),
                _canonical(meta))
            for entry in self._data["sources"]:
                source = self._sources[int(entry["id"])]
                zout.writestr(
                    zipfile.ZipInfo(source.file, _ZIP_EPOCH),
                    write_wav_mono(source.samples, source.sample_rate))
        return buf.getvalue()

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())

    def inventory(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "sample_rate": self.sample_rate,
            "tracks": len(self._data["tracks"]),
            "clips": len(self._data["clips"]),
            "buses": [b["name"] for b in self._data["buses"]],
            "sources": [s["name"] for s in self._data["sources"]],
            "length_samples": self.length_samples(),
        }
