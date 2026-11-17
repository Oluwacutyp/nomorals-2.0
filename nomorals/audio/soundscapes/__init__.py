"""Devon's own soundscape machine — the MyNoise organ (L4).

She composes ambient sound herself: layered noise colours, binaural
beats and drones, and synthesised scenes (rain, forest, ocean, fire,
cafe, night) — each on ten sliders, with a hearing calibration, an
animate drift so nothing loops stale, and named mixes the owner
calls back with one word. Everything is synthesised in code; no
recording is ever faked, and a render that cannot be built is
refused in words.
"""

from .engine import (
    BAND_FREQS,
    GREY_CONTOUR_DB,
    MAX_DURATION_S,
    SR,
    animate_sliders,
    apply_eq,
    apply_hearing,
    octave_band_energies,
    render_tone,
    synthesize_noise,
    synthesize_tone,
    write_mono_wav,
    write_stereo_wav,
)
from .generators import GENERATORS, PRESETS, GeneratorSpec, resolve_generator
from .render import render_soundscape
from .stems import STEMS, stem_names
from .store import SoundscapeStore

__all__ = [
    "BAND_FREQS",
    "GENERATORS",
    "GREY_CONTOUR_DB",
    "MAX_DURATION_S",
    "PRESETS",
    "SR",
    "STEMS",
    "GeneratorSpec",
    "SoundscapeStore",
    "animate_sliders",
    "apply_eq",
    "apply_hearing",
    "octave_band_energies",
    "render_soundscape",
    "render_tone",
    "resolve_generator",
    "stem_names",
    "synthesize_noise",
    "synthesize_tone",
    "write_mono_wav",
    "write_stereo_wav",
]
