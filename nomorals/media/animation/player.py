"""Playback state — port of AniCraft ``crates/doc/src/player.rs``.

A :class:`Player` holds the playhead (frame, playing, loop mode) over
a document's scene. Loop modes follow upstream exactly: ``loop``
(wraps), ``play-once`` (stops on the last frame), ``single-frame``
(holds frame 0). The player is pure state — rendering stays in
``render.py``.
"""

from __future__ import annotations

from .errors import AnimationError
from .model import AnimationDocument

__all__ = ["Player"]

LOOP_MODES = ("loop", "play-once", "single-frame")


class Player:
    def __init__(self, doc: AnimationDocument, scene: str | None = None,
                 loop_mode: str = "loop") -> None:
        if loop_mode not in LOOP_MODES:
            raise AnimationError(f"unknown loop mode {loop_mode!r}")
        self.doc = doc
        self.scene = doc.scene(scene)
        self.loop_mode = loop_mode
        self.frame = 0
        self.playing = False

    @property
    def frame_count(self) -> int:
        return max(1, self.scene.timeline.frame_count())

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def seek(self, frame: int) -> None:
        self.frame = max(0, min(self.frame_count - 1, int(frame)))

    def step(self, frames: int = 1) -> int:
        """Advance ``frames`` steps; returns the new frame."""
        if not self.playing and frames == 0:
            return self.frame
        target = self.frame + frames
        if self.loop_mode == "loop":
            self.frame = target % self.frame_count
        elif self.loop_mode == "play-once":
            self.frame = min(target, self.frame_count - 1)
            if self.frame >= self.frame_count - 1:
                self.playing = False
        else:  # single-frame
            self.frame = 0
        return self.frame

    def local_frame(self, offset: int, first: int = 0,
                    length: int | None = None) -> int:
        """Upstream ``LoopMode::local_frame``: a symbol's local frame."""
        total = max(1, length if length is not None
                    else self.frame_count)
        if self.loop_mode == "loop":
            return (first + offset) % total
        if self.loop_mode == "play-once":
            return min(first + offset, total - 1)
        return min(first, total - 1)
