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
from .effects import (
    DEFERRED_EFFECTS,
    EFFECTS,
    EFFECT_PROPERTIES,
    KEYFRAMABLE_PARAMS,
    PROPERTY_EFFECTS,
    ParamSpec,
    _effect_filter,
    effect_properties,
    effect_property_path,
    effect_rung,
    keyframable_family,
    keyframable_params,
    list_effects,
    validate_effect_params,
)
from .frametime import (
    FrameGrid,
    frame_grid,
    quantize_seconds,
    sum_frame_durations,
)
from .host import HostCapabilities, LocalHost, RenderHost, probe_host
from .keyframes import Keyframe, KeyframeTrack
from .render import (
    HEIGHT,
    WIDTH,
    apply_effect,
    assemble_segments,
    plan_from_cut_points,
    render_document,
    render_timeline,
    render_vertical,
)
from .spec import EditSpec, render, render_report
from .text import (
    STTTranscriber,
    TextLayer,
    Transcriber,
    build_captions,
    burn_text_layers,
    captions_from_audio,
    captions_sidecar,
    estimate_word_timings,
    karaoke_layer,
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
from .scene import SCENE_KINDS, Scene, scenes_to_edit_spec
from .scene3d import (
    ANIMATED_PROPS,
    LIGHT_KINDS,
    MATERIAL_KINDS,
    MESH_KINDS,
    NODE_KINDS,
    PRIMITIVES,
    RENDERER_SURFACES,
    AnimKey,
    AnimTrack,
    Camera,
    FrameState,
    Light,
    Material,
    Mesh,
    RendererCapabilities,
    RendererProbe,
    Rotation,
    Scene3D,
    Scene3DError,
    SceneNode,
    SceneRenderer,
    Transform,
    capability_gaps,
    get_renderer,
    register_renderer,
    registered_renderers,
    render_scene,
    renderer_status,
    reset_renderers,
    resolve_renderer,
    scene_requirements,
    scene_state_at,
    scene_state_dict,
    unregister_renderer,
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
from .qc import (
    CONTACT_SHEET_PASSES,
    MEASURER_RUNGS,
    build_contact_sheet,
    measure_render,
    qc_render,
)
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
from .timeline import (
    Clip,
    Effect,
    OverlayTrack,
    Track,
    Timeline,
    Transition,
    canonical_effect,
)
from .transcript import compile_transcript_edit, keeps_for
from .transitions import list_transitions, validate_transition
from .formats import (
    FormatError,
    FormatSpec,
    ResolvedFormat,
    apply_format,
    formats_data,
    get_format,
    list_formats,
    load_format_file,
    pack_for_format,
    register_format,
    resolve_format,
    scaffold_format,
    validate_format,
)

__all__ = [
    # timeline model
    "Clip", "Effect", "Track", "Timeline", "Transition",
    "canonical_effect",
    # rational time (Wave 1.2a) + keyframes (1.2c)
    "FrameGrid", "frame_grid", "quantize_seconds", "sum_frame_durations",
    "Keyframe", "KeyframeTrack", "KEYFRAMABLE_PARAMS",
    "keyframable_params",
    # multi-track model (Wave 1.2b)
    "OverlayTrack",
    # render host seam (Wave 1.2d)
    "HostCapabilities", "RenderHost", "LocalHost", "probe_host",
    # effects / transitions
    "EFFECTS", "list_effects", "list_transitions", "validate_transition",
    # effect property registry (Wave 5 — EffectCraft property paths)
    "EFFECT_PROPERTIES", "PROPERTY_EFFECTS", "DEFERRED_EFFECTS",
    "ParamSpec", "effect_properties", "effect_property_path",
    "effect_rung", "keyframable_family", "validate_effect_params",
    # text
    "TextLayer", "words_to_ass", "layer_to_ass", "estimate_word_timings",
    "build_captions", "burn_text_layers",
    # caption timing provenance (§2.4)
    "Transcriber", "STTTranscriber", "captions_from_audio",
    "captions_sidecar", "karaoke_layer",
    # audio
    "AudioLayer", "AudioMix", "mix_layers", "normalize_loudness",
    "make_music_bed",
    # render
    "WIDTH", "HEIGHT", "apply_effect", "assemble_segments",
    "plan_from_cut_points", "render_timeline", "render_vertical",
    "render_document",
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
    # QC measurers + five-pass contact sheet (§1.5c tail)
    "CONTACT_SHEET_PASSES", "MEASURER_RUNGS", "measure_render",
    "build_contact_sheet", "qc_render",
    # format registry (§1.5b — formats as parameterized data)
    "FormatError", "FormatSpec", "ResolvedFormat", "validate_format",
    "register_format", "get_format", "list_formats", "resolve_format",
    "apply_format", "pack_for_format", "formats_data", "scaffold_format",
    "load_format_file",
    # scene3d spine (§1.5e — the document + frame function + registry
    # both 3D engine rungs plug into)
    "Scene3DError", "Rotation", "Transform", "Camera", "Light",
    "Material", "Mesh", "SceneNode", "AnimKey", "AnimTrack",
    "ANIMATED_PROPS", "Scene3D", "FrameState", "scene_state_at",
    "scene_state_dict", "NODE_KINDS", "MESH_KINDS", "PRIMITIVES",
    "MATERIAL_KINDS", "LIGHT_KINDS", "RENDERER_SURFACES",
    "RendererCapabilities", "RendererProbe", "SceneRenderer",
    "register_renderer", "unregister_renderer", "reset_renderers",
    "registered_renderers", "get_renderer", "renderer_status",
    "scene_requirements", "capability_gaps", "resolve_renderer",
    "render_scene",
    # private-but-stable (tests + compat shims)
    "_effect_filter",
]
