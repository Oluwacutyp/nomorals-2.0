"""Devon's style-agnostic video edit engine.

Composable primitives with no aesthetic baked in — timeline model,
parameterized effects/transitions, text layers, layered audio, and an
ffmpeg renderer. Styles (phonk, documentary, vlog, …) live in
:mod:`nomorals.media.contentops.styles` and compose these primitives.
"""

from .audio import (
    AudioLayer,
    AudioMix,
    make_music_bed,
    mix_layers,
    normalize_loudness,
)
from .effects import EFFECTS, _effect_filter, list_effects
from .render import (
    HEIGHT,
    WIDTH,
    apply_effect,
    assemble_segments,
    plan_from_cut_points,
    render_timeline,
    render_vertical,
)
from .spec import EditSpec, render, render_report
from .text import (
    TextLayer,
    build_captions,
    burn_text_layers,
    estimate_word_timings,
    layer_to_ass,
    words_to_ass,
)
from .timeline import Clip, Effect, Track, Timeline, Transition, canonical_effect
from .transitions import list_transitions, validate_transition

__all__ = [
    # timeline model
    "Clip", "Effect", "Track", "Timeline", "Transition",
    "canonical_effect",
    # effects / transitions
    "EFFECTS", "list_effects", "list_transitions", "validate_transition",
    # text
    "TextLayer", "words_to_ass", "layer_to_ass", "estimate_word_timings",
    "build_captions", "burn_text_layers",
    # audio
    "AudioLayer", "AudioMix", "mix_layers", "normalize_loudness",
    "make_music_bed",
    # render
    "WIDTH", "HEIGHT", "apply_effect", "assemble_segments",
    "plan_from_cut_points", "render_timeline", "render_vertical",
    # spec
    "EditSpec", "render", "render_report",
    # private-but-stable (tests + compat shims)
    "_effect_filter",
]
