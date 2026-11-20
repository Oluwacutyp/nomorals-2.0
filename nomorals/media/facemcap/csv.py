"""LiveLink CSV facial capture — the parse side.

The bevy-facial-mocap pattern (storytold/bevy-mocap, pattern
only): Unreal's LiveLink Face app writes one CSV row per frame
— a SMPTE timecode, a blendshape count, then one float column
per ARKit morph, in header order. This module parses that
shape, in Python, with the same semantics the Rust crate has:

* Header: ``timecode, blendshapecount, <morph>...`` — the
  second column is skipped, the rest are morph names, trimmed.
* Rows: ``HH:MM:SS:FF[.frac], <count>, <f32>...`` — a row
  whose column count differs from the header is refused by
  name (the Rust csv crate refuses ragged rows; so do we).
* FPS is deduced: the largest frame number seen + 1, since
  frame numbers reset every second.
* Timestamps are SMPTE ``HH:MM:SS:FF`` — segments reversed,
  missing segments default to 0, each segment parsed as a
  float and floored. One difference, named: a segment above
  255 is refused (the clone wraps silently on its u8 cast;
  Devon names it).

Nothing here knows about Bevy or the rig — it is the parse
half. :mod:`mapping` turns a parsed animation into rig
frames; :mod:`loader` ties a manifest to both.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from .errors import FaceMocapError

__all__ = [
    "CSVAnimation",
    "CSVFrame",
    "Timestamp",
]

#: Columns skipped after the timecode before the morphs begin
#: (the blendshape-count column).
_SKIP_COLUMNS_BEFORE_MORPH = 1


@dataclass(frozen=True)
class Timestamp:
    """A SMPTE timecode: hour, minute, second, frame."""

    hour: int
    minute: int
    second: int
    frame: int

    @classmethod
    def parse(cls, text: str) -> Timestamp:
        """Parse ``HH:MM:SS:FF`` — segments may be missing from
        the left (``SS:FF`` is a valid timecode) and a frame may
        carry a fraction (``04.123`` floors to 4). An empty
        string, more than four segments, a non-number, or a
        segment above 255 is refused by name."""
        raw = (text or "").strip()
        if not raw:
            raise FaceMocapError(
                "the timestamp is an empty string",
                kind="bad_timecode")
        segments = raw.split(":")
        if len(segments) > 4:
            raise FaceMocapError(
                f"timestamp {raw!r} has more than 4 segments",
                kind="bad_timecode")
        # Segments start with hours; parse from the frame end so
        # a short timecode still lands in the right slots.
        segments.reverse()

        def _segment(index: int, name: str) -> int:
            if index >= len(segments):
                return 0
            try:
                value = float(segments[index])
            except ValueError as exc:
                raise FaceMocapError(
                    f"timestamp {raw!r}: {name} "
                    f"{segments[index]!r} is not a number",
                    kind="bad_timecode") from exc
            number = int(value // 1)
            if number < 0 or number > 255:
                raise FaceMocapError(
                    f"timestamp {raw!r}: {name} {number} is "
                    f"outside 0..255",
                    kind="bad_timecode")
            return number

        return cls(hour=_segment(3, "hour"),
                   minute=_segment(2, "minute"),
                   second=_segment(1, "second"),
                   frame=_segment(0, "frame"))

    def to_seconds(self, fps: float) -> float:
        """The timecode as seconds from 00:00:00:00, at ``fps``
        frames per second."""
        if fps <= 0:
            raise FaceMocapError(
                f"fps must be positive, got {fps}",
                kind="bad_timecode")
        return (self.hour * 3600 + self.minute * 60
                + self.second + self.frame / fps)


@dataclass(frozen=True)
class CSVFrame:
    """One CSV row: its timestamp and its morph weights, in
    header order."""

    timestamp: Timestamp
    morphs: tuple[float, ...]


@dataclass
class CSVAnimation:
    """A parsed LiveLink CSV: frames, morph names in header
    order, and the deduced fps."""

    frames: list[CSVFrame]
    morph_names: list[str]
    fps: float

    @classmethod
    def parse(cls, text: str) -> CSVAnimation:
        """Parse LiveLink CSV text. An empty file, a header with
        no morphs, a file with no frames, a bad timecode, a bad
        morph value, or a ragged row is refused by name."""
        if not (text or "").strip():
            raise FaceMocapError("empty CSV file",
                                 kind="empty_csv")
        try:
            rows = list(csv.reader(io.StringIO(text),
                                   strict=True))
        except csv.Error as exc:
            raise FaceMocapError(
                f"invalid CSV: {exc}", kind="invalid_csv") from exc
        rows = [row for row in rows if row]
        if not rows:
            raise FaceMocapError("empty CSV file",
                                 kind="empty_csv")
        header = [cell.strip() for cell in rows[0]]
        morph_names = header[_SKIP_COLUMNS_BEFORE_MORPH + 1:]
        if not morph_names:
            raise FaceMocapError(
                "no morphs available in the CSV file",
                kind="no_morphs")
        width = len(header)
        frames: list[CSVFrame] = []
        max_frame = 0
        for line_no, row in enumerate(rows[1:], start=2):
            if len(row) != width:
                raise FaceMocapError(
                    f"CSV row {line_no} has {len(row)} columns, "
                    f"the header has {width}",
                    kind="invalid_csv")
            timestamp = Timestamp.parse(row[0])
            # The blendshape-count column (index 1) is skipped;
            # morphs begin at index 2.
            morphs: list[float] = []
            for index, cell in enumerate(
                    row[_SKIP_COLUMNS_BEFORE_MORPH + 1:]):
                try:
                    morphs.append(float(cell.strip()))
                except ValueError as exc:
                    raise FaceMocapError(
                        f"CSV row {line_no}: morph "
                        f"{morph_names[index]!r} value "
                        f"{cell!r} is not a number",
                        kind="bad_morph") from exc
            if timestamp.frame > max_frame:
                max_frame = timestamp.frame
            frames.append(CSVFrame(timestamp=timestamp,
                                   morphs=tuple(morphs)))
        if not frames:
            raise FaceMocapError("the CSV has 0 frames",
                                 kind="no_frames")
        # Frame numbers start at 0 and reset every second, so
        # the largest frame number + 1 is the fps.
        return cls(frames=frames, morph_names=morph_names,
                   fps=float(max_frame + 1))

    @property
    def duration_s(self) -> float:
        """Seconds from the first frame to the last, at the
        deduced fps."""
        if not self.frames:
            return 0.0
        return (len(self.frames) - 1) / self.fps if self.fps else 0.0
