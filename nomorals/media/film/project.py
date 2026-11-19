"""FilmProject — FilmCraft's non-linear editor model, real impl.

Ports the model of FilmCraft's ``crates/project`` + ``crates/edit``:
a project is a set of tracks (video / audio / title / caption /
composition), each holding clips on a timeline, with Premiere's edit
algebra — overwrite, insert (ripple push on sync-locked tracks),
lift, extract, ripple delete (refusing when a sync-locked track has
material in the gap, naming it), trims (regular / ripple, limited
by media handles), transitions between adjacent clips, named
markers, per-clip colour grade, and keyframes stored in media time.

Everything is JSON-pure: ``to_dict``/``from_dict`` round-trip and
``content_hash`` is the project's identity.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Clip",
    "FilmError",
    "FilmProject",
    "Marker",
    "Track",
    "Transition",
]

TRACK_KINDS = ("video", "audio", "title", "caption", "composition")
CLIP_KINDS = ("video", "still", "solid", "audio", "title", "caption",
              "composition")
TRANSITION_KINDS = ("cross_dissolve", "wipe", "dip_to_black")
TRANSITION_ALIGNS = ("center", "start", "end")
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg",
              ".mpeg"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif",
              ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


class FilmError(ValueError):
    """A project/model problem — overlap, a bad trim, a sync-lock
    conflict, a missing clip. The reason names the track or clip."""


def _num(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FilmError(f"{what} must be a number, got {value!r}")
    return float(value)


@dataclass
class Clip:
    """One media item on a track. ``at`` is timeline time; ``start``
    is media time (the in-point); ``duration`` is the timeline
    length. Keyframes live in media time — the FilmCraft rule."""

    id: str
    kind: str
    source: str
    at: float
    duration: float
    start: float = 0.0
    speed: float = 1.0
    gain: float = 1.0
    grade: dict[str, float] = field(default_factory=lambda: {
        "lift": 0.0, "gamma": 1.0, "gain": 1.0, "saturation": 1.0})
    effects: list[dict[str, Any]] = field(default_factory=list)
    kenburns: dict[str, float] = field(default_factory=dict)
    text: str = ""
    words: list[dict[str, Any]] = field(default_factory=list)
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in CLIP_KINDS:
            raise FilmError(
                f"clip kind must be one of {CLIP_KINDS}, got {self.kind!r}")
        self.at = _num(self.at, "clip at")
        self.duration = _num(self.duration, "clip duration")
        self.start = _num(self.start, "clip start")
        if self.duration <= 0:
            raise FilmError("clip duration must be > 0")
        if self.at < 0:
            raise FilmError("clip at must be >= 0")
        if self.start < 0:
            raise FilmError("clip start must be >= 0")
        if self.speed <= 0:
            raise FilmError("clip speed must be > 0")

    @property
    def end(self) -> float:
        return self.at + self.duration

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "source": self.source,
                "at": self.at, "duration": self.duration,
                "start": self.start, "speed": self.speed,
                "gain": self.gain, "grade": dict(self.grade),
                "effects": list(self.effects),
                "kenburns": dict(self.kenburns), "text": self.text,
                "words": list(self.words), "name": self.name}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Clip:
        return cls(id=str(data["id"]), kind=str(data["kind"]),
                   source=str(data.get("source", "")),
                   at=float(data["at"]), duration=float(data["duration"]),
                   start=float(data.get("start", 0.0)),
                   speed=float(data.get("speed", 1.0)),
                   gain=float(data.get("gain", 1.0)),
                   grade=dict(data.get("grade") or {
                       "lift": 0.0, "gamma": 1.0, "gain": 1.0,
                       "saturation": 1.0}),
                   effects=list(data.get("effects") or []),
                   kenburns=dict(data.get("kenburns") or {}),
                   text=str(data.get("text", "")),
                   words=list(data.get("words") or []),
                   name=str(data.get("name", "")))


@dataclass
class Track:
    """One timeline lane. ``sync_locked`` tracks move with ripple
    edits on other tracks; a ripple delete that would cut through
    their material refuses, naming the track."""

    id: str
    kind: str
    name: str
    items: list[Clip] = field(default_factory=list)
    locked: bool = False
    sync_locked: bool = False
    muted: bool = False

    def __post_init__(self) -> None:
        if self.kind not in TRACK_KINDS:
            raise FilmError(
                f"track kind must be one of {TRACK_KINDS}, "
                f"got {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "name": self.name,
                "items": [c.to_dict() for c in self.items],
                "locked": self.locked,
                "sync_locked": self.sync_locked, "muted": self.muted}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Track:
        return cls(id=str(data["id"]), kind=str(data["kind"]),
                   name=str(data["name"]),
                   items=[Clip.from_dict(c)
                          for c in data.get("items") or []],
                   locked=bool(data.get("locked", False)),
                   sync_locked=bool(data.get("sync_locked", False)),
                   muted=bool(data.get("muted", False)))


@dataclass
class Transition:
    kind: str
    duration: float
    from_id: str
    to_id: str
    align: str = "center"

    def __post_init__(self) -> None:
        if self.kind not in TRANSITION_KINDS:
            raise FilmError(
                f"transition kind must be one of {TRANSITION_KINDS}, "
                f"got {self.kind!r}")
        if self.align not in TRANSITION_ALIGNS:
            raise FilmError(
                f"transition align must be one of {TRANSITION_ALIGNS}, "
                f"got {self.align!r}")
        if self.duration <= 0:
            raise FilmError("transition duration must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "duration": self.duration,
                "from_id": self.from_id, "to_id": self.to_id,
                "align": self.align}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transition:
        return cls(kind=str(data["kind"]),
                   duration=float(data["duration"]),
                   from_id=str(data["from_id"]), to_id=str(data["to_id"]),
                   align=str(data.get("align", "center")))


@dataclass
class Marker:
    time: float
    name: str
    color: str = "yellow"
    kind: str = "comment"

    def to_dict(self) -> dict[str, Any]:
        return {"time": self.time, "name": self.name,
                "color": self.color, "kind": self.kind}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Marker:
        return cls(time=float(data["time"]), name=str(data["name"]),
                   color=str(data.get("color", "yellow")),
                   kind=str(data.get("kind", "comment")))


class FilmProject:
    """A FilmCraft sequence: tracks, clips, transitions, markers,
    grade, ducking — the whole document."""

    def __init__(self, name: str, *, width: int = 1920, height: int = 1080,
                 fps: float = 24.0) -> None:
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        if self.width <= 0 or self.height <= 0:
            raise FilmError("project canvas must be positive")
        if self.fps <= 0:
            raise FilmError("project fps must be > 0")
        self.tracks: list[Track] = []
        self.transitions: list[Transition] = []
        self.markers: list[Marker] = []
        self.ducking: dict[str, Any] = {"enabled": False, "duck_gain": 0.25}
        self._next_clip = itertools.count(1)
        self._next_track = itertools.count(1)

    # -- structure -------------------------------------------------------
    @property
    def duration(self) -> float:
        return max((c.end for t in self.tracks for c in t.items),
                   default=0.0)

    def add_track(self, kind: str, name: str) -> Track:
        track = Track(id=f"t{next(self._next_track)}", kind=kind, name=name)
        self.tracks.append(track)
        return track

    def find_track(self, ref: str | Track) -> Track:
        if isinstance(ref, Track):
            return ref
        for track in self.tracks:
            if track.id == ref or track.name == ref:
                return track
        raise FilmError(
            f"no track {ref!r} — project has: "
            f"{', '.join(t.name for t in self.tracks) or '(none)'}")

    def find_clip(self, clip_id: str) -> tuple[Track, Clip]:
        for track in self.tracks:
            for clip in track.items:
                if clip.id == clip_id:
                    return track, clip
        raise FilmError(f"no clip {clip_id!r} in project {self.name!r}")

    # -- clips -------------------------------------------------------------
    @staticmethod
    def _infer_kind(track_kind: str, source: str) -> str:
        from pathlib import Path
        ext = Path(source).suffix.lower()
        if track_kind == "audio":
            return "audio"
        if track_kind == "title":
            return "title"
        if track_kind == "caption":
            return "caption"
        if track_kind == "composition":
            return "composition"
        if ext in VIDEO_EXTS:
            return "video"
        if ext in IMAGE_EXTS:
            return "still"
        if ext in AUDIO_EXTS:
            return "audio"
        return "video"

    def _check_overlap(self, track: Track, at: float, duration: float,
                       ignore_id: str | None = None) -> None:
        end = at + duration
        for clip in track.items:
            if clip.id == ignore_id:
                continue
            if at < clip.end - 1e-9 and end > clip.at + 1e-9:
                raise FilmError(
                    f"clip at {at}s overlaps {clip.id} "
                    f"({clip.at}s–{clip.end}s) on track {track.name!r}")

    def add_clip(self, track: str | Track, source: str, *, at: float,
                 duration: float, start: float = 0.0,
                 kind: str | None = None, speed: float = 1.0,
                 gain: float = 1.0, **extra: Any) -> Clip:
        tr = self.find_track(track)
        if tr.locked:
            raise FilmError(f"track {tr.name!r} is locked")
        clip_kind = kind or self._infer_kind(tr.kind, source)
        self._check_overlap(tr, float(at), float(duration))
        clip = Clip(id=f"c{next(self._next_clip)}", kind=clip_kind,
                    source=source, at=float(at), duration=float(duration),
                    start=float(start), speed=float(speed),
                    gain=float(gain), **extra)
        tr.items.append(clip)
        tr.items.sort(key=lambda c: c.at)
        return clip

    def insert_clip(self, track: str | Track, source: str, *, at: float,
                    duration: float, start: float = 0.0,
                    kind: str | None = None, **extra: Any) -> Clip:
        """Ripple insert: everything at/after ``at`` on this track
        and on every sync-locked track shifts right by ``duration``."""
        tr = self.find_track(track)
        at = float(at)
        duration = float(duration)
        # a clip spanning the insert point is split there — the
        # Premiere rule: the part before the playhead stays, the
        # part after it is pushed with everything else
        for item in list(tr.items):
            if item.at < at < item.end - 1e-9:
                self.split_clip(item.id, at_time=at)
                break
        affected = [tr] + [t for t in self.tracks
                          if t is not tr and t.sync_locked]
        for t in affected:
            for item in t.items:
                if item.at >= at - 1e-9:
                    item.at += duration
        return self.add_clip(tr, source, at=at, duration=duration,
                             start=start, kind=kind, **extra)

    def split_clip(self, clip_id: str, *, at_time: float) -> tuple[Clip, Clip]:
        track, clip = self.find_clip(clip_id)
        at_time = float(at_time)
        if not clip.at < at_time < clip.end:
            raise FilmError(
                f"split point {at_time}s is outside clip {clip.id} "
                f"({clip.at}s–{clip.end}s)")
        cut = at_time - clip.at
        second = Clip.from_dict(clip.to_dict())
        second.id = f"c{next(self._next_clip)}"
        second.at = at_time
        second.start = clip.start + cut
        second.duration = clip.duration - cut
        clip.duration = cut
        track.items.append(second)
        track.items.sort(key=lambda c: c.at)
        return clip, second

    def lift(self, clip_id: str) -> Clip:
        """Remove the clip, leaving a gap."""
        track, clip = self.find_clip(clip_id)
        track.items.remove(clip)
        self.transitions = [tr for tr in self.transitions
                            if tr.from_id != clip_id
                            and tr.to_id != clip_id]
        return clip

    def extract(self, clip_id: str) -> Clip:
        """Remove the clip and close the gap on its track."""
        return self._remove_and_close(clip_id, sync=False)

    def ripple_delete(self, clip_id: str) -> Clip:
        """Remove the clip and close the gap on its track AND every
        sync-locked track. Refuses — naming the track — when a
        sync-locked track has material inside the gap."""
        return self._remove_and_close(clip_id, sync=True)

    def _remove_and_close(self, clip_id: str, *, sync: bool) -> Clip:
        track, clip = self.find_clip(clip_id)
        if sync:
            for other in self.tracks:
                if other is track or not other.sync_locked:
                    continue
                for item in other.items:
                    if item.at < clip.end - 1e-9 \
                            and item.end > clip.at + 1e-9:
                        raise FilmError(
                            f"cannot ripple-delete {clip.id}: "
                            f"sync-locked track {other.name!r} has "
                            f"material in the gap "
                            f"({item.at}s–{item.end}s)")
        track.items.remove(clip)
        self.transitions = [tr for tr in self.transitions
                            if tr.from_id != clip_id
                            and tr.to_id != clip_id]
        affected = [track] + ([t for t in self.tracks
                               if t is not track and t.sync_locked]
                              if sync else [])
        for t in affected:
            for item in t.items:
                if item.at >= clip.end - 1e-9:
                    item.at -= clip.duration
        return clip

    def trim_clip(self, clip_id: str, edge: str, delta: float,
                  *, ripple: bool = False) -> Clip:
        """Trim ``edge`` ("in" | "out") by ``delta`` seconds.
        Regular trims change the clip only; ripple trims move every
        following clip on the track by the same amount."""
        track, clip = self.find_clip(clip_id)
        if edge not in ("in", "out"):
            raise FilmError(f"trim edge must be 'in' or 'out', got {edge!r}")
        delta = float(delta)
        frame = 1.0 / self.fps
        old_end = clip.end
        if edge == "out":
            new_duration = clip.duration + delta
            if new_duration < frame:
                raise FilmError(
                    f"trim would leave {clip.id} shorter than one frame")
            clip.duration = new_duration
            if ripple:
                for item in track.items:
                    if item is not clip and item.at >= old_end - 1e-9:
                        item.at += delta
        else:
            new_duration = clip.duration - delta
            new_start = clip.start + delta
            if new_duration < frame:
                raise FilmError(
                    f"trim would leave {clip.id} shorter than one frame")
            if new_start < 0:
                raise FilmError(
                    f"trim would move {clip.id} before its media start")
            new_at = clip.at + delta
            if delta < 0:
                self._check_overlap(track, new_at, new_duration,
                                    ignore_id=clip.id)
            clip.at = new_at
            clip.start = new_start
            clip.duration = new_duration
            if ripple:
                for item in track.items:
                    if item is not clip and item.at >= old_end - 1e-9:
                        item.at += delta
        track.items.sort(key=lambda c: c.at)
        return clip

    # -- transitions / markers / grade -------------------------------------
    def add_transition(self, from_id: str, to_id: str, kind: str,
                       duration: float, *, align: str = "center") -> Transition:
        _track_a, clip_a = self.find_clip(from_id)
        track_b, clip_b = self.find_clip(to_id)
        if _track_a is not track_b:
            raise FilmError("a transition joins clips on the same track")
        if abs(clip_b.at - clip_a.end) > 1.0 / self.fps:
            raise FilmError(
                f"clips {from_id} and {to_id} are not adjacent "
                f"({clip_a.end}s vs {clip_b.at}s)")
        if float(duration) > min(clip_a.duration, clip_b.duration):
            raise FilmError(
                "transition duration exceeds the shorter clip")
        tr = Transition(kind=kind, duration=float(duration),
                        from_id=from_id, to_id=to_id, align=align)
        self.transitions.append(tr)
        return tr

    def add_transition_by_index(self, track: str | Track, index: int,
                                kind: str, duration: float, *,
                                align: str = "center") -> Transition:
        tr = self.find_track(track)
        if index < 0 or index + 1 >= len(tr.items):
            raise FilmError(
                f"no adjacent pair at index {index} on track {tr.name!r}")
        return self.add_transition(tr.items[index].id,
                                   tr.items[index + 1].id,
                                   kind, duration, align=align)

    def add_marker(self, time: float, name: str, *, color: str = "yellow",
                   kind: str = "comment") -> Marker:
        marker = Marker(time=float(time), name=name, color=color, kind=kind)
        self.markers.append(marker)
        self.markers.sort(key=lambda m: m.time)
        return marker

    def grade_clip(self, clip_id: str, *, lift: float | None = None,
                   gamma: float | None = None, gain: float | None = None,
                   saturation: float | None = None) -> Clip:
        _track, clip = self.find_clip(clip_id)
        if lift is not None:
            clip.grade["lift"] = float(lift)
        if gamma is not None:
            if float(gamma) <= 0:
                raise FilmError("grade gamma must be > 0")
            clip.grade["gamma"] = float(gamma)
        if gain is not None:
            clip.grade["gain"] = float(gain)
        if saturation is not None:
            clip.grade["saturation"] = float(saturation)
        return clip

    def add_title(self, track: str | Track, text: str, *, at: float,
                  duration: float, font_size: int = 48,
                  color: tuple[int, int, int] = (255, 255, 255)) -> Clip:
        tr = self.find_track(track)
        if tr.kind != "title":
            raise FilmError(
                f"titles go on a title track, not {tr.kind!r} "
                f"({tr.name!r})")
        clip = self.add_clip(tr, "", at=at, duration=duration, kind="title")
        clip.text = text
        clip.kenburns = {"font_size": float(font_size),
                         "r": float(color[0]), "g": float(color[1]),
                         "b": float(color[2])}
        return clip

    def add_caption(self, track: str | Track,
                    words: list[dict[str, Any]]) -> Clip:
        tr = self.find_track(track)
        if tr.kind != "caption":
            raise FilmError(
                f"captions go on a caption track, not {tr.kind!r} "
                f"({tr.name!r})")
        if not words:
            raise FilmError("a caption clip needs words")
        start = min(float(w["start"]) for w in words)
        end = max(float(w["end"]) for w in words)
        clip = self.add_clip(tr, "", at=start, duration=end - start,
                             kind="caption")
        clip.words = [{"text": str(w["text"]), "start": float(w["start"]),
                       "end": float(w["end"])} for w in words]
        return clip

    def set_ducking(self, *, enabled: bool = True, duck_gain: float = 0.25,
                    bed_track: str | None = None,
                    speech_track: str | None = None) -> None:
        self.ducking = {"enabled": bool(enabled),
                        "duck_gain": float(duck_gain),
                        "bed_track": bed_track,
                        "speech_track": speech_track}

    # -- validation / identity ----------------------------------------------
    def validate(self) -> list[str]:
        """Every structural problem, as plain strings ([] = clean).
        Missing files are NOT checked here — the render checks them
        and names the file."""
        problems: list[str] = []
        for track in self.tracks:
            items = sorted(track.items, key=lambda c: c.at)
            for a, b in zip(items, items[1:]):  # noqa: B905 - sliding window: the second iterable is one shorter by construction
                if a.end > b.at + 1e-9:
                    problems.append(
                        f"track {track.name!r}: {a.id} overlaps {b.id}")
            for clip in track.items:
                if clip.kind == "video" and not clip.source:
                    problems.append(
                        f"track {track.name!r}: clip {clip.id} has no source")
        for tr in self.transitions:
            try:
                _ta, ca = self.find_clip(tr.from_id)
                _tb, cb = self.find_clip(tr.to_id)
            except FilmError:
                problems.append(
                    f"transition {tr.kind} references a missing clip")
                continue
            if abs(cb.at - ca.end) > 1.0 / self.fps:
                problems.append(
                    f"transition {tr.kind} clips are no longer adjacent")
            if tr.duration > min(ca.duration, cb.duration):
                problems.append(
                    f"transition {tr.kind} is longer than a clip")
        return problems

    # -- serialisation --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"kind": "devon-film-project", "version": 1,
                "name": self.name, "width": self.width,
                "height": self.height, "fps": self.fps,
                "tracks": [t.to_dict() for t in self.tracks],
                "transitions": [tr.to_dict() for tr in self.transitions],
                "markers": [m.to_dict() for m in self.markers],
                "ducking": dict(self.ducking)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilmProject:
        if data.get("kind") not in (None, "devon-film-project"):
            raise FilmError(
                f"not a film project (kind={data.get('kind')!r})")
        proj = cls(str(data.get("name", "untitled")),
                   width=int(data.get("width", 1920)),
                   height=int(data.get("height", 1080)),
                   fps=float(data.get("fps", 24.0)))
        proj.tracks = [Track.from_dict(t) for t in data.get("tracks") or []]
        proj.transitions = [Transition.from_dict(t)
                            for t in data.get("transitions") or []]
        proj.markers = [Marker.from_dict(m)
                        for m in data.get("markers") or []]
        proj.ducking = dict(data.get("ducking") or
                            {"enabled": False, "duck_gain": 0.25})
        # keep the id counters ahead of anything loaded
        clip_n = 0
        for track in proj.tracks:
            for clip in track.items:
                with contextlib.suppress(ValueError):
                    clip_n = max(clip_n, int(clip.id.lstrip("c")))
        proj._next_clip = itertools.count(clip_n + 1)
        track_n = 0
        for track in proj.tracks:
            with contextlib.suppress(ValueError):
                track_n = max(track_n, int(track.id.lstrip("t")))
        proj._next_track = itertools.count(track_n + 1)
        return proj

    def content_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
