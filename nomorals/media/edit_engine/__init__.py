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
from .genops import GEN_OPS, apply_genop, execute_genop, genop_to_spec, validate_genop
from .document import (
    TRACK_KINDS,
    DocTrack,
    Document,
    UndoStack,
    diff_documents,
    audio_document,
    image_document,
    transcript_document,
    voiceover_document,
)
from .render_queue import (
    PROXY_HEIGHT as RENDER_PROXY_HEIGHT,
    RenderFailure,
    RenderQueue,
    Segment,
    resolve_render_profile,
    segments_for_ranges,
)
from .audio_ops import (
    EMOTIONS,
    EmotionOp,
    LoudnessSpec,
    SpatialParams,
    StemSet,
    ambience_filter,
    check_loudness,
    emotion_op_for_line,
    emotion_timeline,
    loudness_spec_for,
    neural_emotion_available,
    separate_stems,
    spatial_filter,
    stem_mix,
)
from .image_four import (
    finish_pass,
    multi_ref_identity,
    neural_relight_available,
    pose_control,
    pose_control_image,
    relight_after_composite,
    text_fidelity_finish,
)
from .video_four import (
    ASPECTS,
    auto_broll,
    match_cut_score,
    match_cuts,
    reframe,
    reframe_dims,
    tighten,
    tighten_keeps,
)
from .quality_gate import QualityGate, auto_fix, quality_gate
from .pure_render import (
    PROXY_HEIGHT,
    apply_pack,
    canonical_spec_dict,
    proxy_spec,
    render_cache_key,
    render_proxy_then_full,
    render_pure,
    source_fingerprint,
)
from .timeline import Clip, Effect, Track, Timeline, Transition, canonical_effect
from .transcript import compile_transcript_edit, keeps_for
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
    # transcript → spec compiler (the one transcript-edit path)
    "compile_transcript_edit", "keeps_for",
    # pure render: (sources, spec, pack) → cached output
    "canonical_spec_dict", "source_fingerprint", "render_cache_key",
    "proxy_spec", "apply_pack", "render_pure", "render_proxy_then_full",
    "PROXY_HEIGHT",
    # generative ops as spec ops
    "GEN_OPS", "validate_genop", "genop_to_spec", "apply_genop",
    "execute_genop",
    # EDIT_GEN_UPGRADE: one unified document (2.1 + 2.3)
    "TRACK_KINDS", "DocTrack", "Document", "UndoStack", "diff_documents",
    "image_document", "audio_document", "voiceover_document",
    "transcript_document",
    # render queue first-class (2.4)
    "RenderFailure", "RenderQueue", "Segment", "resolve_render_profile",
    "segments_for_ranges", "RENDER_PROXY_HEIGHT",
    # audio missing middle (2.7)
    "EMOTIONS", "EmotionOp", "emotion_op_for_line", "emotion_timeline",
    "neural_emotion_available", "StemSet", "separate_stems", "stem_mix",
    "LoudnessSpec", "loudness_spec_for", "check_loudness",
    "SpatialParams", "spatial_filter", "ambience_filter",
    # image four (2.8)
    "relight_after_composite", "neural_relight_available", "pose_control",
    "pose_control_image", "multi_ref_identity", "text_fidelity_finish",
    "finish_pass",
    # video four (2.9)
    "ASPECTS", "reframe", "reframe_dims", "match_cut_score", "match_cuts",
    "auto_broll", "tighten", "tighten_keeps",
    # quality gate (2.10)
    "QualityGate", "quality_gate", "auto_fix",
    # private-but-stable (tests + compat shims)
    "_effect_filter",
]
