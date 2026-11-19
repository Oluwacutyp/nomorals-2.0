"""SoundCraft session editor — typed commands, guarded steps (lane 107).

Ported from storytold/soundcraft's ``crates/engine`` command
registry: every user-visible action is a command on the session,
each step guarded — a bad step fails with a named reason instead of
corrupting the document. Every mutation marks the entity dirty so
``fidelity_report`` can name exactly what changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import SoundError, SoundErrorKind
from .model import SessionDocument, read_wav_mono

__all__ = ["SessionEditor"]


def _plugin_spec(plugin: Any) -> dict[str, Any]:
    """Turn a plugin object into a serialisable insert spec."""
    info = plugin.info() if hasattr(plugin, "info") else {}
    kind = str(info.get("id", ""))
    params = dict(info.get("params", {})) if isinstance(info.get("params"), dict) else {}
    if kind in ("gain", "parametric-eq", "compressor", "limiter", "tone"):
        return {"kind": kind, "params": params}
    raise SoundError(
        f"plugin {kind!r} has no serialisable insert spec — attach "
        "live plugin instances to the engine, not the document",
        kind=SoundErrorKind.INVALID_ARGUMENT)


class SessionEditor:
    """Edit a session document, one guarded command at a time."""

    def __init__(self, doc: SessionDocument):
        self.doc = doc

    # -- tracks ---------------------------------------------------------
    def add_track(self, name: str, kind: str = "audio") -> int:
        if kind not in ("audio", "instrument"):
            raise SoundError(f"unknown track kind {kind!r}",
                             kind=SoundErrorKind.INVALID_ARGUMENT)
        if len(self.doc._data["tracks"]) >= 1024:
            raise SoundError("too many tracks",
                             kind=SoundErrorKind.TOO_LARGE)
        tid = self.doc._alloc_id()
        self.doc._data["tracks"].append({
            "id": tid, "name": name, "kind": kind, "gain_db": 0.0,
            "pan": 0.0, "mute": False, "solo": False,
            "inserts": [], "sends": [], "automation": {},
        })
        self.doc._mark_added(tid)
        return tid

    def _track_raw(self, track_id: int) -> dict[str, Any]:
        for raw in self.doc._data["tracks"]:
            if raw["id"] == track_id:
                return raw
        raise SoundError(f"no track {track_id}",
                         kind=SoundErrorKind.NOT_FOUND)

    def set_track_gain(self, track_id: int, gain_db: float) -> None:
        raw = self._track_raw(track_id)
        raw["gain_db"] = float(gain_db)
        self.doc._mark_dirty(track_id)

    def set_track_pan(self, track_id: int, pan: float) -> None:
        if not -1.0 <= pan <= 1.0:
            raise SoundError(f"pan {pan} out of range [-1, 1]",
                             kind=SoundErrorKind.INVALID_ARGUMENT)
        raw = self._track_raw(track_id)
        raw["pan"] = float(pan)
        self.doc._mark_dirty(track_id)

    def mute_track(self, track_id: int, mute: bool = True) -> None:
        raw = self._track_raw(track_id)
        raw["mute"] = bool(mute)
        self.doc._mark_dirty(track_id)

    def solo_track(self, track_id: int, solo: bool = True) -> None:
        raw = self._track_raw(track_id)
        raw["solo"] = bool(solo)
        self.doc._mark_dirty(track_id)

    # -- clips ------------------------------------------------------------
    def add_clip(self, track_id: int, name: str, source_path: str,
                 start: int, offset: int = 0) -> int:
        track = self._track_raw(track_id)
        if track["kind"] != "audio":
            raise SoundError("audio clips go on audio tracks",
                             kind=SoundErrorKind.INVALID_ARGUMENT)
        raw = Path(source_path).read_bytes()
        samples, sr = read_wav_mono(raw)
        if sr != self.doc.sample_rate:
            raise SoundError(
                f"source sample rate {sr} does not match session "
                f"{self.doc.sample_rate} — resample first",
                kind=SoundErrorKind.INVALID_ARGUMENT)
        sid = self.doc._alloc_id()
        file = f"sources/{sid}.wav"
        from .model import Source
        self.doc._sources[sid] = Source(id=sid, name=name, file=file,
                                        sample_rate=sr, samples=samples)
        self.doc._data["sources"].append({
            "id": sid, "name": name, "file": file,
            "sample_rate": sr, "frames": int(len(samples)),
        })
        cid = self.doc._alloc_id()
        self.doc._data["clips"].append({
            "id": cid, "track_id": track_id, "name": name,
            "source_id": sid, "start": int(start), "offset": int(offset),
            "duration": int(len(samples) - offset), "gain_db": 0.0,
        })
        self.doc._mark_added(cid)
        return cid

    def add_note(self, track_id: int, start: int, duration: int,
                 pitch: int, velocity: float = 0.8) -> int:
        track = self._track_raw(track_id)
        if track["kind"] != "instrument":
            raise SoundError("notes go on instrument tracks",
                             kind=SoundErrorKind.INVALID_ARGUMENT)
        if not 0 <= pitch <= 127:
            raise SoundError(f"MIDI pitch {pitch} out of range",
                             kind=SoundErrorKind.INVALID_ARGUMENT)
        # One note clip per track: notes accumulate in it.
        clip_raw = None
        for raw in self.doc._data["clips"]:
            if raw["track_id"] == track_id and raw.get("notes") is not None:
                clip_raw = raw
                break
        if clip_raw is None:
            cid = self.doc._alloc_id()
            clip_raw = {
                "id": cid, "track_id": track_id, "name": "Notes",
                "source_id": None, "start": 0, "offset": 0,
                "duration": 0, "gain_db": 0.0, "notes": [],
            }
            self.doc._data["clips"].append(clip_raw)
            self.doc._mark_added(cid)
        note_id = self.doc._alloc_id()
        clip_raw["notes"].append({
            "id": note_id, "start": int(start), "duration": int(duration),
            "pitch": int(pitch), "velocity": float(velocity),
        })
        clip_raw["duration"] = max(
            int(clip_raw["duration"]), int(start) + int(duration))
        self.doc._mark_dirty(int(clip_raw["id"]))
        return note_id

    def _clip_raw(self, clip_id: int) -> dict[str, Any]:
        for raw in self.doc._data["clips"]:
            if raw["id"] == clip_id:
                return raw
        raise SoundError(f"no clip {clip_id}",
                         kind=SoundErrorKind.NOT_FOUND)

    def move_clip(self, clip_id: int, start: int) -> None:
        raw = self._clip_raw(clip_id)
        raw["start"] = int(start)
        self.doc._mark_dirty(clip_id)

    def trim_clip(self, clip_id: int, start: int | None = None,
                  end: int | None = None) -> None:
        raw = self._clip_raw(clip_id)
        if start is not None:
            delta = int(start) - int(raw["start"])
            raw["start"] = int(start)
            raw["offset"] = int(raw["offset"]) + delta
            raw["duration"] = int(raw["duration"]) - delta
        if end is not None:
            raw["duration"] = int(end) - int(raw["start"])
        if raw["duration"] < 0 or raw["offset"] < 0:
            raise SoundError("trim would make a negative-length clip",
                             kind=SoundErrorKind.INVALID_ARGUMENT)
        self.doc._mark_dirty(clip_id)

    def set_clip_gain(self, clip_id: int, gain_db: float) -> None:
        raw = self._clip_raw(clip_id)
        raw["gain_db"] = float(gain_db)
        self.doc._mark_dirty(clip_id)

    def remove_clip(self, clip_id: int) -> None:
        raw = self._clip_raw(clip_id)
        self.doc._data["clips"].remove(raw)
        self.doc._mark_removed(clip_id)

    # -- buses, sends, automation, inserts ------------------------------
    def add_bus(self, name: str) -> int:
        bid = self.doc._alloc_id()
        self.doc._data["buses"].append(
            {"id": bid, "name": name, "kind": "aux"})
        self.doc._mark_added(bid)
        return bid

    def add_send(self, track_id: int, bus_id: int, gain_db: float,
                 pre_fader: bool = False) -> None:
        track = self._track_raw(track_id)
        self.doc.bus(bus_id)  # raises if the bus does not exist
        track["sends"].append({"bus_id": int(bus_id),
                               "gain_db": float(gain_db),
                               "pre_fader": bool(pre_fader)})
        self.doc._mark_dirty(track_id)

    def add_automation(self, track_id: int, param: str,
                       points: list[tuple[int, float]]) -> None:
        track = self._track_raw(track_id)
        if not points:
            raise SoundError("automation needs at least one point",
                             kind=SoundErrorKind.INVALID_ARGUMENT)
        track["automation"][param] = [
            [float(s), float(v)] for s, v in sorted(points)]
        self.doc._mark_dirty(track_id)

    def insert_plugin(self, track_id: int, plugin: Any) -> None:
        track = self._track_raw(track_id)
        spec = plugin if isinstance(plugin, dict) else _plugin_spec(plugin)
        if spec.get("kind") == "plugin":
            raise SoundError(
                "quarantined plugin instances attach to the engine "
                "(MixEngine.insert_plugin), not the document",
                kind=SoundErrorKind.INVALID_ARGUMENT)
        track["inserts"].append(spec)
        self.doc._mark_dirty(track_id)
