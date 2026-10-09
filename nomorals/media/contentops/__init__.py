"""Short-form content pipeline: niche + topic → rendered MP4.

* ``edit.py`` — the edit engine: ``EditSpec``, ``render``,
  ``detect_beats``, caption/audio helpers (landed).
* ``beats.py`` / ``audio.py`` — beat detection (librosa → numpy fallback),
  music/voiceover ducking + loudness normalization.
* ``niches/`` — ``get_niche`` plugin registry.

``pipeline.py`` imports those contracts defensively and falls back to the
clearly-marked implementations in ``_shims.py`` when they are not importable
yet, so integration is a drop-in the moment the sibling modules land.
"""

try:  # pipeline.py lands with the sibling stream; stay importable until then
    from .pipeline import (  # type: ignore[no-redef]
        STAGES,
        ContentCalendar,
        Job,
        JobStore,
        PlannedPost,
        RunResult,
        ShortPipeline,
        StageError,
    )

    __all__ = [
        "STAGES",
        "ContentCalendar",
        "Job",
        "JobStore",
        "PlannedPost",
        "RunResult",
        "ShortPipeline",
        "StageError",
    ]
except ImportError:  # noqa: BLE001
    __all__: list[str] = []

# niches/ plugin registry (sibling stream) — re-exported for convenience.
from .niches import (  # noqa: E402
    NicheError,
    NichePlugin,
    SceneVisual,
    ScriptResult,
    VisualPlan,
    VoiceSpec,
    all_plugins,
    get_niche,
    list_niches,
    register,
    scaffold,
    validate,
)

__all__ += [
    "NicheError",
    "NichePlugin",
    "SceneVisual",
    "ScriptResult",
    "VisualPlan",
    "VoiceSpec",
    "all_plugins",
    "get_niche",
    "list_niches",
    "register",
    "scaffold",
    "validate",
]

# edit engine (sibling stream: edit.py / beats.py / audio.py) — the
# pipeline's edit contract (EditSpec, render, detect_beats,
# build_captions, make_music_bed) plus the full engine surface.
from .audio import (  # noqa: E402
    mix_audio,
    normalize_loudness,
    voice_segments,
)
from .beats import (  # noqa: E402
    BeatInfo,
    beats_available,
    detect_beats,
    detect_beats_full,
)
from .edit import (  # noqa: E402
    AudioSpec,
    CaptionSpec,
    Clip,
    EditSpec,
    Effect,
    apply_effect,
    assemble_cuts,
    beat_sync_words,
    build_captions,
    burn_beat_captions,
    estimate_word_timings,
    make_music_bed,
    render,
    render_report,
    render_vertical,
    words_to_beat_ass,
)

__all__ += [
    "AudioSpec",
    "CaptionSpec",
    "Clip",
    "EditSpec",
    "Effect",
    "BeatInfo",
    "apply_effect",
    "assemble_cuts",
    "beat_sync_words",
    "beats_available",
    "build_captions",
    "burn_beat_captions",
    "detect_beats",
    "detect_beats_full",
    "estimate_word_timings",
    "make_music_bed",
    "mix_audio",
    "normalize_loudness",
    "render",
    "render_report",
    "render_vertical",
    "voice_segments",
    "words_to_beat_ass",
]
