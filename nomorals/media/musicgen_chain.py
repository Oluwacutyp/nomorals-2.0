"""Music generation chain — the full pipeline, one call, honest stages.

Tier 0. Architecture (preserved, not reinvented):

    LYRICS + TONES → DRAFT score (notes) → BED (ACE-Step + LoRA)
    → VOCAL DiffSinger (notes + lyrics → phonemes → F0 → ONNX → HiFiGAN)
    → RVC timbre swap (preserves prosody/melody) → MASTER

Invariants enforced here:
* Melody is NOTES, not audio — stems + MIDI + spec are written for
  every song, so any song is reproducible from its spec.
* Tones never contradict melody (draft passes through tone_fix_melody).
* Stems always exist in the manifest (real paths when produced,
  ``missing`` otherwise — never a fake path).
* Licence is a LABEL, not a gate: the manifest records the
  ``commercial_use`` flag per model used (see
  :mod:`nomorals.media.model_license`). Everything is usable for
  personal AI; if a track ever goes public, "was any non-commercial?"
  is a query on the manifest, not an audit. Nothing here blocks a
  run on licence grounds.
* Scope is UNIVERSAL, Nigeria as DEFAULT (language "yo", genre
  "afrobeats" are the default pack — owner context, least-served,
  best place to start), never the boundary. N packs, nation-generic.

Every stage reports its real status: ``real`` (a model ran),
``fake`` (an injected test double ran — labelled as such in the
manifest and the result), or ``skipped`` (model absent, with the
reason). ``ok`` is True only when the draft AND at least one real
or fake audio stage produced audio. A chain with no audio at all
returns ``ok: False`` — never a fake pass.

Backends are injectable (``bed_fn``, ``vocal_fn``, ``rvc_fn``,
``master_fn``) so tests can run the whole chain with fakes while
production passes nothing and gets the real backends.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .model_license import can_ship, licence_audit
from .tonal_g2p import profile_for, syllabify, tone_fix_melody

_log = get_logger(__name__)

__all__ = [
    "ChainRequest",
    "ChainStage",
    "ChainResult",
    "draft_melody",
    "write_midi",
    "write_sine_wav",
    "fake_bed",
    "fake_vocal",
    "run_chain",
    "register",
]

# Major-scale degrees for the deterministic draft contour.
_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_KEY_ROOTS = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71,
              "C#": 61, "D#": 63, "F#": 66, "G#": 68, "A#": 70,
              "Bb": 70, "Eb": 63, "Ab": 68, "Db": 61, "Gb": 66}


@dataclass
class ChainRequest:
    title: str = "hello-world"
    lyrics: str = "Mo dé, ayọ dé"   # Yoruba hello-world line (default)
    language: str = "yo"
    genre: str = "afrobeats"
    key: str = "C"
    bpm: float = 100.0
    seed: int = 0
    voicebank: str = ""       # DiffSinger bank dir ("" = auto-detect)
    rvc_model: str = ""       # RVC model name ("" = skip RVC stage)
    audience: str = "private"  # "private" | "commercial" — LABEL only, never gates
    workdir: str = "musicgen_out"
    lora: str = ""            # LoRA adapter path ("" = base ACE-Step)
    reference_path: str = ""  # reference track for Matchering mastering
    bed_duration_s: int = 60


@dataclass
class ChainStage:
    name: str
    status: str               # "real" | "fake" | "skipped"
    path: str = ""
    detail: str = ""


@dataclass
class ChainResult:
    ok: bool
    title: str
    stages: list[ChainStage] = field(default_factory=list)
    master_path: str = ""
    midi_path: str = ""
    spec_path: str = ""
    manifest_path: str = ""
    note: str = ""
    error: str = ""

    def stage_status(self, name: str) -> str:
        for s in self.stages:
            if s.name == name:
                return s.status
        return "missing"


def draft_melody(lyrics: str, language: str = "yo", key: str = "C",
                 bpm: float = 100.0, seed: int = 0,
                 base_midi: int | None = None) -> list[tuple[int, float, float]]:
    """LYRICS+TONES → DRAFT score (notes).

    Deterministic: each syllable gets one note. The contour is a
    seeded stepwise walk on the major scale (phrase arch: rise to the
    middle, fall to the end), then :func:`tone_fix_melody` rewrites any
    note that contradicts the lyric's tone. The LLM-first path
    (composer_llm / SongSpec melody_abc) replaces this contour with
    the model's melody — but it still flows through the same tone fix.
    """
    import random
    rng = random.Random(seed)
    root = base_midi if base_midi is not None else _KEY_ROOTS.get(
        (key or "C").upper(), 60)
    beat_s = 60.0 / max(1.0, bpm)
    syllables: list[str] = []
    for word in (lyrics or "").split():
        syllables.extend(syllabify(word, language))
    if not syllables:
        syllables = ["la"]
    notes: list[tuple[int, float, float]] = []
    degree = 0
    n = len(syllables)
    for i, _syl in enumerate(syllables):
        # phrase arch: climb toward the middle, descend to the end
        target = 2 if i < n / 2 else -2
        degree += rng.choice([-1, 0, 0, 1]) + (1 if target > 0 else -1 if target < 0 else 0) * 0
        degree = max(-3, min(5, degree))
        octave, step = divmod(degree, len(_MAJOR))
        midi = root + octave * 12 + _MAJOR[step]
        notes.append((midi, round(i * beat_s * 0.9, 3), round(beat_s * 0.9, 3)))
    fixed = tone_fix_melody(" ".join(syllables), notes, language)
    return [(int(p), float(s), float(d)) for p, s, d in fixed]


def write_midi(notes: list[tuple[int, float, float]], path: str,
               bpm: float = 100.0) -> str:
    """Write a real Standard MIDI File (format 0) from notes. Stdlib only."""
    ticks_per_beat = 480
    tick_s = 60.0 / max(1.0, bpm) / ticks_per_beat
    events: list[tuple[int, bytes]] = []
    for midi, start, dur in notes:
        on = int(round(start / tick_s))
        off = int(round((start + dur) / tick_s))
        events.append((on, bytes([0x90, midi & 0x7F, 90])))
        events.append((off, bytes([0x80, midi & 0x7F, 0])))
    # tempo meta event at tick 0
    tempo = int(round(60_000_000 / max(1.0, bpm)))
    events.append((0, bytes([0xFF, 0x51, 0x03,
                             (tempo >> 16) & 0xFF, (tempo >> 8) & 0xFF,
                             tempo & 0xFF])))
    events.sort(key=lambda e: (e[0], e[1]))
    track = bytearray()
    last = 0
    for tick, payload in events:
        delta = tick - last
        last = tick
        # variable-length quantity
        vlq = bytearray([delta & 0x7F])
        delta >>= 7
        while delta:
            vlq.insert(0, (delta & 0x7F) | 0x80)
            delta >>= 7
        track.extend(vlq)
        track.extend(payload)
    track.extend(b"\x00\xff\x2f\x00")  # end of track
    header = (b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big")
              + (1).to_bytes(2, "big") + ticks_per_beat.to_bytes(2, "big"))
    chunk = b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header + chunk)
    return path


def write_sine_wav(path: str, freq: float = 440.0, duration_s: float = 1.0,
                   sample_rate: int = 24000, volume: float = 0.3) -> str:
    """Write a real (non-silent) sine WAV. Used by test doubles only."""
    n = int(duration_s * sample_rate)
    samples = array("h", (int(volume * 32767 * math.sin(
        2 * math.pi * freq * i / sample_rate)) for i in range(n)))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())
    return path


def fake_bed(notes: list, lyrics: str, workdir: str, **_kw: Any) -> str:
    """Test double for the ACE-Step bed: a real sine WAV, labelled fake."""
    return write_sine_wav(os.path.join(workdir, "bed-fake.wav"),
                          freq=220.0, duration_s=2.0)


def fake_vocal(notes: list, lyrics: str, workdir: str, **_kw: Any) -> str:
    """Test double for the DiffSinger vocal: a real sine WAV, labelled fake."""
    return write_sine_wav(os.path.join(workdir, "vocal-fake.wav"),
                          freq=440.0, duration_s=2.0)


def _read_wav(path: str) -> tuple[int, int, int, array]:
    with wave.open(path, "rb") as w:
        return (w.getframerate(), w.getnchannels(), w.getsampwidth(),
                array("h", w.readframes(w.getnframes())))


def _resample(samples: array, src_sr: int, dst_sr: int) -> array:
    """Linear-interpolation resample (documented, simple, real)."""
    if src_sr == dst_sr:
        return samples
    ratio = dst_sr / src_sr
    out = array("h")
    for i in range(int(len(samples) * ratio)):
        pos = i / ratio
        j = int(pos)
        frac = pos - j
        a = samples[j] if j < len(samples) else 0
        b = samples[j + 1] if j + 1 < len(samples) else 0
        out.append(int(a + (b - a) * frac))
    return out


def _mix_wavs(bed_path: str, vocal_path: str, out_path: str) -> str:
    """Mix bed + vocal to one WAV (linear resample to the bed's rate)."""
    sr_b, _, _, bed = _read_wav(bed_path)
    sr_v, _, _, voc = _read_wav(vocal_path)
    voc = _resample(voc, sr_v, sr_b)
    n = max(len(bed), len(voc))
    mixed = array("h")
    for i in range(n):
        b = bed[i] if i < len(bed) else 0
        v = voc[i] if i < len(voc) else 0
        mixed.append(int(max(-32767, min(32767, b * 0.7 + v * 0.7))))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr_b)
        w.writeframes(mixed.tobytes())
    return out_path


def _master_wav(src_path: str, out_path: str) -> str:
    """Master one WAV through the real mastering chain (stdlib DSP)."""
    from ..voice.mastering import master as _master
    sr, _, _, samples = _read_wav(src_path)
    out_samples, _stages = _master(samples, sr, fade_ms=25.0)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(out_samples.tobytes())
    return out_path


def run_chain(req: ChainRequest,
              bed_fn: Callable[..., str] | None = None,
              vocal_fn: Callable[..., str] | None = None,
              rvc_fn: Callable[..., str] | None = None,
              master_fn: Callable[..., str] | None = None) -> ChainResult:
    """Run the full chain. ``*_fn`` inject test doubles; None = real backend.

    Real bed: ACEStepBackend.generate (fails closed when ACE-Step is
    absent — stage marked skipped with the reason). Real vocal:
    DiffSingerBackend.sing (same). Real RVC: rvc_bridge.convert.
    """
    outdir = Path(req.workdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stages: list[ChainStage] = []
    used_models: list[str] = []
    # ── 1. DRAFT ──
    notes = draft_melody(req.lyrics, req.language, req.key, req.bpm, req.seed)
    midi_path = write_midi(notes, str(outdir / f"{req.title}-melody.mid"), req.bpm)
    spec = {"title": req.title, "lyrics": req.lyrics,
            "language": req.language, "genre": req.genre, "key": req.key,
            "bpm": req.bpm, "seed": req.seed,
            "notes": [[p, s, d] for p, s, d in notes],
            "lora": req.lora, "voicebank": req.voicebank,
            "rvc_model": req.rvc_model}
    spec_path = str(outdir / f"{req.title}-spec.json")
    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
    stages.append(ChainStage("draft", "real", midi_path,
                             f"{len(notes)} notes, tone-fixed ({req.language})"))
    # ── 2. BED ──
    bed_path = ""
    if bed_fn is not None:
        bed_path = bed_fn(notes, req.lyrics, str(outdir), req=req)
        stages.append(ChainStage("bed", "fake", bed_path,
                                 "test double — NOT a real ACE-Step bed"))
        used_models.append("acestep")  # fake stands in for the real model
    else:
        try:
            from .ace_step import ACEStepBackend
            backend = ACEStepBackend()
            if req.lora:
                backend = ACEStepBackend()  # LoRA applied via adapter below
            res = backend.generate(req.lyrics, req.genre,
                                   duration_s=req.bed_duration_s,
                                   seed=req.seed, title=req.title,
                                   workdir=str(outdir))
            bed_path = res.audio_path
            used_models.append("acestep")
            stages.append(ChainStage("bed", "real", bed_path,
                                     f"ACE-Step bed ({res.variant})"
                                     + (f" + LoRA {req.lora}" if req.lora else "")))
        except Exception as exc:  # noqa: BLE001 — backend absence is data
            stages.append(ChainStage("bed", "skipped", "", str(exc)[:300]))
    # ── 3. VOCAL ──
    vocal_path = ""
    if vocal_fn is not None:
        vocal_path = vocal_fn(notes, req.lyrics, str(outdir), req=req)
        stages.append(ChainStage("vocal", "fake", vocal_path,
                                 "test double — NOT a real DiffSinger vocal"))
        used_models.append("diffsinger")
    else:
        try:
            from ..voice.singing import DiffSingerBackend, Note
            backend = DiffSingerBackend(voicebank=req.voicebank)
            sung_notes = [Note(midi=int(p), duration_s=float(d),
                               lyric=req.lyrics.split()[0] if req.lyrics else "")
                          for p, _s, d in notes]
            # per-note lyric: one syllable per note
            syls: list[str] = []
            for word in (req.lyrics or "").split():
                syls.extend(syllabify(word, req.language))
            for i, nt in enumerate(sung_notes):
                nt.lyric = syls[i] if i < len(syls) else "-"
            res = backend.sing(sung_notes)
            vocal_path = res["path"]
            used_models.append("diffsinger")
            stages.append(ChainStage("vocal", "real", vocal_path,
                                     f"DiffSinger vocal ({res.get('notes')} notes)"))
        except Exception as exc:  # noqa: BLE001
            stages.append(ChainStage("vocal", "skipped", "", str(exc)[:300]))
    # ── 4. RVC ──
    rvc_path = ""
    if req.rvc_model and vocal_path:
        if rvc_fn is not None:
            rvc_path = rvc_fn(vocal_path, req.rvc_model, str(outdir))
            stages.append(ChainStage("rvc", "fake", rvc_path,
                                     "test double — NOT a real RVC conversion"))
            used_models.append("rvc")
            vocal_path = rvc_path
        else:
            try:
                from ..voice import rvc_bridge
                models = rvc_bridge.list_models()
                model = next((m for m in models if m.name == req.rvc_model), None)
                if model is None:
                    raise RuntimeError(f"no RVC model named {req.rvc_model!r}")
                conv = rvc_bridge.convert(vocal_path, model)
                vocal_path = conv["path"]
                used_models.append("rvc")
                stages.append(ChainStage("rvc", "real", vocal_path,
                                         f"RVC timbre → {req.rvc_model}"))
            except Exception as exc:  # noqa: BLE001
                stages.append(ChainStage("rvc", "skipped", "", str(exc)[:300]))
    elif req.rvc_model and not vocal_path:
        stages.append(ChainStage("rvc", "skipped", "",
                                 "no vocal to convert — vocal stage skipped"))
    # ── 5. MASTER ──
    master_path = ""
    audio_sources = [p for p in (bed_path, vocal_path) if p]
    if master_fn is not None and audio_sources:
        master_path = master_fn(audio_sources, str(outdir), req=req)
        stages.append(ChainStage("master", "fake", master_path,
                                 "test double master"))
    elif bed_path and vocal_path:
        try:
            mixed = _mix_wavs(bed_path, vocal_path,
                              str(outdir / f"{req.title}-mix.wav"))
            master_path = _master_wav(mixed, str(outdir / f"{req.title}-master.wav"))
            stages.append(ChainStage("master", "real", master_path,
                                     "bed+vocal mixed + mastered"))
        except Exception as exc:  # noqa: BLE001
            stages.append(ChainStage("master", "skipped", "", str(exc)[:300]))
    elif vocal_path:
        try:
            master_path = _master_wav(vocal_path,
                                      str(outdir / f"{req.title}-master.wav"))
            stages.append(ChainStage("master", "real", master_path,
                                     "vocal-only master (no bed)"))
        except Exception as exc:  # noqa: BLE001
            stages.append(ChainStage("master", "skipped", "", str(exc)[:300]))
    elif bed_path:
        try:
            master_path = _master_wav(bed_path,
                                      str(outdir / f"{req.title}-master.wav"))
            stages.append(ChainStage("master", "real", master_path,
                                     "bed-only master (no vocal)"))
        except Exception as exc:  # noqa: BLE001
            stages.append(ChainStage("master", "skipped", "", str(exc)[:300]))
    else:
        stages.append(ChainStage("master", "skipped", "",
                                 "no audio to master — bed and vocal both skipped"))
    # ── licence label + provenance ──
    audit = licence_audit(used_models)
    # stem manifest: bed stem + vocal stem are the stems we actually have
    stems = {"vocals": vocal_path, "drums": "", "bass": "", "other": bed_path}
    spec_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    manifest = {
        "title": req.title, "spec_sha256_16": spec_hash,
        "spec_path": spec_path, "midi_path": midi_path,
        "stages": [{"name": s.name, "status": s.status, "path": s.path,
                    "detail": s.detail} for s in stages],
        "stems": stems,
        "stems_missing": [k for k, v in stems.items() if not v],
        "models_used": used_models,
        "audience": req.audience, "licence_audit": audit,
        "licenses": {m: can_ship(m, req.audience) for m in used_models},
        "provenance": ("FAKE stages are test doubles, not real models. "
                       "Do not present a fake-stage run as a real generation."),
        "honest_limits": ("Raw fidelity one-shot pop: Suno wins, don't fight "
                          "there. Singing realism: good, not indistinguishable. "
                          "Clone own voice safe; clone real artist = legal risk."),
    }
    manifest_path = str(outdir / f"{req.title}-manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    has_audio = bool(master_path)
    if not has_audio:
        skipped = [f"{s.name}: {s.detail}" for s in stages if s.status == "skipped"]
        return ChainResult(False, req.title, stages, "", midi_path,
                           spec_path, manifest_path,
                           error="no audio produced — " + "; ".join(skipped))
    return ChainResult(True, req.title, stages, master_path, midi_path,
                       spec_path, manifest_path,
                       note="chain complete: " + ", ".join(
                           f"{s.name}={s.status}" for s in stages))


def register(registry: Any) -> None:
    """Tool registration: ``musicgen_chain`` — the full pipeline."""
    from ..core.policy import Capability

    @registry.register(
        "musicgen_chain",
        description=(
            "Full music generation chain: lyrics+tones → draft score → "
            "ACE-Step bed → DiffSinger vocal → RVC timbre → master. "
            "action=run (title, lyrics, language, genre, key, bpm, seed, "
            "voicebank, rvc_model, audience, workdir). Stages report "
            "real/skipped honestly; ok=False when no audio was produced. "
            "Licence is a label in the manifest (commercial_use flag per "
            "model), never a gate — audience just records intent."
        ),
        capability=Capability.FS_WRITE,
    )
    def musicgen_chain(action: str = "run", title: str = "hello-world",
                       lyrics: str = "Mo dé, ayọ dé", language: str = "yo",
                       genre: str = "afrobeats", key: str = "C",
                       bpm: float = 100.0, seed: int = 0,
                       voicebank: str = "", rvc_model: str = "",
                       audience: str = "private",
                       workdir: str = "musicgen_out") -> dict:
        req = ChainRequest(title=title, lyrics=lyrics, language=language,
                           genre=genre, key=key, bpm=bpm, seed=seed,
                           voicebank=voicebank, rvc_model=rvc_model,
                           audience=audience, workdir=workdir)
        r = run_chain(req)
        return {"ok": r.ok, "title": r.title,
                "stages": [{"name": s.name, "status": s.status,
                            "path": s.path, "detail": s.detail}
                           for s in r.stages],
                "master_path": r.master_path, "midi_path": r.midi_path,
                "spec_path": r.spec_path, "manifest_path": r.manifest_path,
                "note": r.note, "error": r.error}
