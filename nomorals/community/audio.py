"""Audio rooms, second wave: recordings, clips, transcripts, speaker stats.

``audio_rooms.py`` owns the live room (participants, tips, platform voice
chats). This module owns what happens to the *recording* afterwards:

* Recording metadata — title, room, start/end, duration, and an optional
  ``file_ref`` supplied by the caller (a path or URL where the audio
  actually lives). This module never fabricates audio or a file location:
  no ``file_ref`` means "no file attached yet".
* Clips — named time ranges cut from a recording.
* Transcript — segments of speaker-attributed text. The transcript is
  **accepted** from whoever supplies it (manual notes, a platform
  caption export, a wired STT backend). There is no fake STT here: this
  module never generates transcript text from audio, and a recording
  with no supplied transcript simply has no transcript.
* Speaker stats — talk time per speaker, derived ONLY from transcript
  segments. No segments → no stats, never invented numbers.

Isolation: stdlib + ``nomorals.core`` only. No owner-data imports.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id

__all__ = [
    "Recording",
    "Clip",
    "TranscriptSegment",
    "AudioStore",
    "default_audio_dir",
    "speaker_stats",
]


def default_audio_dir() -> Path:
    """Default on-disk home: inside the community dir."""
    return Path.home() / ".devon" / "community" / "audio"


@dataclass
class Recording:
    id: str
    group_key: str
    room_id: str
    title: str
    started_ts: float = 0.0
    ended_ts: float = 0.0
    file_ref: str = ""  # caller-supplied path/URL; "" = no file attached
    format: str = ""  # e.g. "ogg" / "mp3" — caller-supplied, never guessed
    notes: str = ""
    created_ts: float = 0.0

    @property
    def duration_s(self) -> float:
        """Seconds between start and end; 0 while still recording."""
        if self.ended_ts <= self.started_ts:
            return 0.0
        return self.ended_ts - self.started_ts

    @property
    def is_finished(self) -> bool:
        return self.ended_ts > self.started_ts

    def finish(self, ended_ts: float | None = None) -> None:
        end = ended_ts if ended_ts is not None else time.time()
        if end <= self.started_ts:
            raise ValueError("end must be after start")
        self.ended_ts = end

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recording":
        return cls(
            id=str(data["id"]),
            group_key=str(data["group_key"]),
            room_id=str(data.get("room_id", "")),
            title=str(data["title"]),
            started_ts=float(data.get("started_ts", 0.0)),
            ended_ts=float(data.get("ended_ts", 0.0)),
            file_ref=str(data.get("file_ref", "")),
            format=str(data.get("format", "")),
            notes=str(data.get("notes", "")),
            created_ts=float(data.get("created_ts", 0.0)),
        )


@dataclass
class Clip:
    id: str
    recording_id: str
    title: str
    start_s: float
    end_s: float
    created_by_id: str = ""
    created_by_name: str = ""
    created_ts: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Clip":
        return cls(
            id=str(data["id"]),
            recording_id=str(data["recording_id"]),
            title=str(data["title"]),
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            created_by_id=str(data.get("created_by_id", "")),
            created_by_name=str(data.get("created_by_name", "")),
            created_ts=float(data.get("created_ts", 0.0)),
        )


@dataclass
class TranscriptSegment:
    speaker_id: str
    speaker_name: str
    start_s: float
    end_s: float
    text: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            speaker_id=str(data["speaker_id"]),
            speaker_name=str(data.get("speaker_name", "")),
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            text=str(data["text"]),
        )


def speaker_stats(
    segments: list[TranscriptSegment],
) -> dict[str, dict[str, Any]]:
    """Talk time per speaker, derived ONLY from supplied segments.

    Returns ``{speaker_id: {speaker_name, talk_seconds, segments,
    words, share}}`` where ``share`` is talk time / total talk time.
    Empty input → empty dict. Nothing is estimated, interpolated, or
    inferred from audio: no segments, no stats.
    """
    per: dict[str, dict[str, Any]] = {}
    total = 0.0
    for seg in segments:
        entry = per.setdefault(
            seg.speaker_id,
            {
                "speaker_id": seg.speaker_id,
                "speaker_name": seg.speaker_name,
                "talk_seconds": 0.0,
                "segments": 0,
                "words": 0,
            },
        )
        entry["talk_seconds"] = float(entry["talk_seconds"]) + seg.duration_s
        entry["segments"] = int(entry["segments"]) + 1
        entry["words"] = int(entry["words"]) + seg.word_count
        total += seg.duration_s
    for entry in per.values():
        entry["share"] = (float(entry["talk_seconds"]) / total) if total else 0.0
    return per


class AudioStore:
    """JSON persistence for recordings, clips, and transcripts.

    Layout: ``<group>/recordings.json``, ``<group>/clips.json``,
    ``<group>/transcripts/<recording_id>.json``.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else default_audio_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _group_dir(self, group_key: str) -> Path:
        safe = "".join(c for c in group_key if c.isalnum() or c in "-_")
        folder = self.data_dir / (safe or "default")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _load_list(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return data if isinstance(data, list) else []

    # ── recordings ────────────────────────────────────────────────
    def put_recording(self, recording: Recording) -> None:
        folder = self._group_dir(recording.group_key)
        path = folder / "recordings.json"
        items = [i for i in self._load_list(path) if i.get("id") != recording.id]
        items.append(recording.to_dict())
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def get_recording(self, group_key: str, recording_id: str) -> Recording | None:
        for item in self._load_list(self._group_dir(group_key) / "recordings.json"):
            if item.get("id") == recording_id:
                return Recording.from_dict(item)
        return None

    def list_recordings(self, group_key: str) -> list[Recording]:
        return [
            Recording.from_dict(i)
            for i in self._load_list(self._group_dir(group_key) / "recordings.json")
        ]

    # ── clips ─────────────────────────────────────────────────────
    def add_clip(
        self,
        recording: Recording,
        title: str,
        start_s: float,
        end_s: float,
        created_by_id: str = "",
        created_by_name: str = "",
    ) -> Clip:
        """Cut a clip. Bounds are validated against the recording's
        duration when the recording is finished; live recordings accept
        any non-negative range (duration unknown yet)."""
        if start_s < 0 or end_s <= start_s:
            raise ValueError("clip needs 0 <= start < end")
        if recording.is_finished and end_s > recording.duration_s:
            raise ValueError("clip end is past the end of the recording")
        clip = Clip(
            id=new_short_id("clip"),
            recording_id=recording.id,
            title=title,
            start_s=float(start_s),
            end_s=float(end_s),
            created_by_id=created_by_id,
            created_by_name=created_by_name,
            created_ts=time.time(),
        )
        folder = self._group_dir(recording.group_key)
        path = folder / "clips.json"
        items = self._load_list(path)
        items.append(clip.to_dict())
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")
        return clip

    def list_clips(self, group_key: str, recording_id: str = "") -> list[Clip]:
        clips = [
            Clip.from_dict(i)
            for i in self._load_list(self._group_dir(group_key) / "clips.json")
        ]
        if recording_id:
            clips = [c for c in clips if c.recording_id == recording_id]
        return clips

    # ── transcripts (accepted, never generated) ───────────────────
    def attach_transcript(
        self,
        group_key: str,
        recording_id: str,
        segments: list[TranscriptSegment],
        source: str = "manual",
    ) -> None:
        """Store a transcript supplied by the caller.

        ``source`` records where the text came from ("manual",
        "platform-captions", "stt:<backend>") so readers can tell a
        human-typed transcript from a machine one. This function never
        produces transcript text itself — there is no STT here.
        """
        folder = self._group_dir(group_key) / "transcripts"
        folder.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in recording_id if c.isalnum() or c in "-_")
        (folder / f"{safe}.json").write_text(
            json.dumps(
                {
                    "recording_id": recording_id,
                    "source": source,
                    "segments": [s.to_dict() for s in segments],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def get_transcript(
        self, group_key: str, recording_id: str
    ) -> tuple[str, list[TranscriptSegment]]:
        """Return (source, segments). No transcript → ("", [])."""
        safe = "".join(c for c in recording_id if c.isalnum() or c in "-_")
        path = self._group_dir(group_key) / "transcripts" / f"{safe}.json"
        if not path.exists():
            return "", []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return "", []
        return str(data.get("source", "")), [
            TranscriptSegment.from_dict(s) for s in data.get("segments", [])
        ]

    def transcript_text(self, group_key: str, recording_id: str) -> str:
        """Plain-text transcript: one ``Name: text`` line per segment."""
        _, segments = self.get_transcript(group_key, recording_id)
        return "\n".join(
            f"{s.speaker_name or s.speaker_id}: {s.text}" for s in segments
        )

    def speaker_stats_for(
        self, group_key: str, recording_id: str
    ) -> dict[str, dict[str, Any]]:
        """Speaker stats for a recording: only from its stored transcript."""
        _, segments = self.get_transcript(group_key, recording_id)
        return speaker_stats(segments)
