"""FilmCraft port — Devon's non-linear video editor.

A real timeline (multi-track picture/sound, Premiere edit algebra,
transitions, titles, captions, colour grade, audio ducking,
loudness) that renders finished videos of any length through the
edit-engine render queue, wired into her existing systems
(videogen captions, motion-studio frame pipe, effects compositor,
narration/TTS, music). The 15-second cap is dead here by
construction: the render is segmented, so length is a runtime
question, never a design limit.
"""

from __future__ import annotations

from .audio import ducking_envelope, measure_loudness, mix_audio
from .project import Clip, FilmError, FilmProject, Marker, Track, Transition
from .render import export_srt, render_project
from .store import FilmStore, film_path

__all__ = [
    "Clip",
    "FilmError",
    "FilmProject",
    "FilmStore",
    "Marker",
    "Track",
    "Transition",
    "ducking_envelope",
    "export_srt",
    "film_path",
    "measure_loudness",
    "mix_audio",
    "render_project",
]
