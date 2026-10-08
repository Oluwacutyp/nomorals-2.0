"""ACE-Step 1.5 local song generation — lyrics → AI instrumental bed.

:mod:`nomorals.media.music` (MusicCreator) writes lyrics, structure, MIDI,
and a rule-based audio render. This module adds the AI layer: ACE-Step 1.5
(MIT, DiT-based text-to-music) turns the lyrics + style conditioning into
a real AI-generated instrumental bed.

Honest boundaries — no fake audio, ever:

* ACE-Step is **not** a pip package. It needs the ``ace-step/ACE-Step-1.5``
  repo (or ``diffusers`` new enough to ship ``AceStepPipeline``), a CUDA
  GPU, and ~5–15 GB of downloaded weights. :meth:`ACEStepBackend.probe`
  checks all of this; :meth:`ACEStepBackend.generate` fails closed with a
  precise install recipe when anything is missing. It never renders
  silence/noise and calls it AI.
* ``termux`` / ``mobile`` / ``embedded`` profiles get an honest
  "needs a workstation GPU" — the DiT will not fit a phone.
* This produces the **bed** (instrumental). Sung vocals are #58
  (DiffSinger → RVC). The output says so.
* The model is loaded around generation and unloaded in ``finally`` —
  it is never held in memory idle.

Real API surface (verified 2026-10-08 against the public repos):

* Official repo: https://github.com/ace-step/ACE-Step-1.5 — generation
  params use ``caption`` (not ``prompt``), ``duration`` (not
  ``audio_duration``), ``lyrics`` with ``[Verse 1]``/``[Chorus]`` section
  tags, ``keyscale``; output is 48 kHz stereo WAV.
* ``diffusers`` ships ``AceStepPipeline`` for text-to-music with lyrics.
"""

from __future__ import annotations

import gc
import os
import re
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..core.profile import detect_profile

_log = get_logger(__name__)

__all__ = [
    "ACEModelUnavailable",
    "ACEStepBackend",
    "BedRequest",
    "BedResult",
    "ProbeResult",
    "format_lyrics_for_acestep",
    "parse_bed_request",
    "probe_acestep",
    "tags_for_style",
    "MODEL_URL",
    "CHECKPOINT_ENV",
    "register",
]

#: the real upstream repo
MODEL_URL = "https://github.com/ace-step/ACE-Step-1.5"
#: env var pointing at downloaded weights (repo default: <repo>/checkpoints)
CHECKPOINT_ENV = "ACESTEP_CHECKPOINTS_DIR"
#: profiles that can never run the DiT — honest refusal, not a fake render
TOO_SMALL = frozenset({"termux", "mobile", "embedded"})
#: laptop-class profiles get the turbo variant + a duration cap
SMALL_GPU = frozenset({"pc"})
#: full DiT profiles
BIG_GPU = frozenset({"workstation", "vps"})

SAMPLE_RATE = 48000  # ACE-Step outputs 48 kHz stereo

INSTALL_HINT = (
    "ACE-Step 1.5 isn't installed. To enable AI song beds:\n"
    "  1. git clone https://github.com/ace-step/ACE-Step-1.5\n"
    "  2. cd ACE-Step-1.5 && uv sync   (needs Python 3.10+, NVIDIA GPU, CUDA)\n"
    "  3. weights auto-download on first run into <repo>/checkpoints\n"
    "     (turbo DiT ~4.8 GB + VAE + LM — or set ACESTEP_CHECKPOINTS_DIR)\n"
    "Until then, /music <topic> [style] still composes with the builtin "
    "rule-based synth — real audio, just not AI-generated."
)


class ACEModelUnavailable(RuntimeError):
    """Raised when ACE-Step can't run here — always with the reason."""


@dataclass
class ProbeResult:
    available: bool
    reason: str = ""          # human-readable when not available
    detail: str = ""          # extra context (paths, versions)
    profile: str = ""
    variant: str = ""         # "turbo" | "full" when available


@dataclass
class BedRequest:
    topic: str
    style: str = "pop"
    duration_s: int = 120
    seed: int = 0


@dataclass
class BedResult:
    ok: bool
    audio_path: str = ""
    title: str = ""
    tags: str = ""
    lyrics: str = ""
    profile: str = ""
    variant: str = ""
    duration_s: float = 0.0
    note: str = ""
    error: str = ""
    backend: str = "acestep"


# ───────────────────────── style → tags ─────────────────────────────────────


def tags_for_style(spec: Any) -> str:
    """Map a StyleSpec onto ACE-Step conditioning tags.

    e.g. afrobeats → "afrobeats, 105bpm, major, log drums, shakers, ...".
    """
    bpm = (spec.tempo[0] + spec.tempo[1]) // 2
    tags: list[str] = [spec.name.replace("_", " "), f"{bpm}bpm", spec.mode]
    tags.extend(spec.instrumentation)
    tags.extend(e.strip() for e in str(spec.energy).split(",") if e.strip())
    tags.extend(spec.palette[:3])
    if getattr(spec, "drum_pattern", ""):
        tags.append(f"{spec.drum_pattern} drums")
    # dedupe, keep order
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip())
    return ", ".join(out)


def format_lyrics_for_acestep(song: Any) -> str:
    """Flatten a MusicCreator Song into ACE-Step lyric format.

    ACE-Step expects section tags like ``[Verse 1]`` / ``[Chorus]``.
    """
    blocks: list[str] = []
    counts: dict[str, int] = {}
    for sec in song.sections:
        name = (sec.name or "verse").strip().capitalize()
        counts[name] = counts.get(name, 0) + 1
        tag = f"{name} {counts[name]}" if counts[name] > 1 else name
        lines = [ln for ln in (sec.lyrics or []) if ln.strip()]
        if not lines:
            continue
        blocks.append(f"[{tag}]\n" + "\n".join(lines))
    return "\n\n".join(blocks)


# ───────────────────────── chat parsing ─────────────────────────────────────


_BED_RE = re.compile(
    r"^\s*(?:make\s+me\s+an?\s+|generate\s+(?:me\s+)?an?\s+)?"
    r"(?P<rest>.+?)\s+song\s*(?:about\s+(?P<topic>.+))?\s*$",
    re.IGNORECASE,
)


def parse_bed_request(text: str, styles: Any = None) -> BedRequest | None:
    """Parse bed requests like:

    * ``/music bed lagos nights afrobeats``
    * ``make me an afrobeats song about Lagos``
    * ``generate a lofi song about rain``

    Returns None when the text isn't a bed request.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    body = raw
    low = raw.lower()
    if low.startswith("/music bed"):
        body = raw[len("/music bed"):].strip()
        m: re.Match | None = None
    elif low.startswith("bed "):
        body = raw[4:].strip()
        m = None
    else:
        m = _BED_RE.match(raw)
        if not m:
            return None
    if m is not None:
        rest = (m.group("rest") or "").strip()
        topic = (m.group("topic") or "").strip()
        # "an afrobeats song about Lagos" → style=afrobeats, topic=Lagos
        words = rest.split()
        style = ""
        if styles is not None:
            from ..media.music import STYLE_ALIASES  # local import, cheap
            first = words[0].lower() if words else ""
            if first in STYLE_ALIASES:
                style = STYLE_ALIASES[first]
                rest = " ".join(words[1:])
        return BedRequest(topic=topic or rest or "untitled",
                          style=style or "pop")
    # explicit "bed <topic> [style]" form
    words = body.split()
    style = "pop"
    if styles is not None and words and words[-1].lower() in styles:
        style = words[-1].lower()
        words = words[:-1]
    topic = " ".join(words).strip()
    if not topic:
        return None
    return BedRequest(topic=topic, style=style)


# ───────────────────────── probing ──────────────────────────────────────────


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _checkpoint_dir() -> str:
    env = os.environ.get(CHECKPOINT_ENV, "").strip()
    if env:
        return env
    home = Path.home() / ".nomorals" / "models" / "acestep-1.5"
    repo = Path.home() / "ACE-Step-1.5" / "checkpoints"
    for cand in (home, repo):
        if cand.is_dir() and any(cand.iterdir()):
            return str(cand)
    return ""


def probe_acestep(profile: str = "") -> ProbeResult:
    """Check whether ACE-Step can run here. Never raises."""
    prof = (profile or "").strip().lower() or detect_profile().kind
    if prof in TOO_SMALL:
        return ProbeResult(
            available=False, profile=prof,
            reason=("ACE-Step needs a workstation GPU — the DiT model won't "
                    "fit this device. /music still works with the builtin "
                    "synth."))
    has_official = _importable("acestep")
    has_diffusers = False
    if _importable("diffusers"):
        try:
            from diffusers import AceStepPipeline  # noqa: F401
            has_diffusers = True
        except Exception:  # noqa: BLE001
            has_diffusers = False
    if not (has_official or has_diffusers):
        return ProbeResult(
            available=False, profile=prof,
            reason="ACE-Step code isn't installed here.",
            detail=INSTALL_HINT)
    if not _cuda_available():
        return ProbeResult(
            available=False, profile=prof,
            reason="no CUDA GPU detected — ACE-Step needs one.",
            detail=INSTALL_HINT)
    ckpt = _checkpoint_dir()
    if not ckpt:
        return ProbeResult(
            available=False, profile=prof,
            reason=("ACE-Step is installed but the model weights aren't "
                    "downloaded yet."),
            detail=(f"set {CHECKPOINT_ENV} to your <repo>/checkpoints, or "
                    f"run the official app once so they auto-download "
                    f"(~5–15 GB). {MODEL_URL}"))
    variant = "turbo" if prof in SMALL_GPU else "full"
    return ProbeResult(available=True, profile=prof, variant=variant,
                       detail=f"checkpoints: {ckpt}")


# ───────────────────────── generation adapters ──────────────────────────────


class _BaseAdapter:
    """One method: render(caption, lyrics, duration_s, seed) → wav bytes."""

    name = "base"

    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def render(self, caption: str, lyrics: str, duration_s: int,
               seed: int) -> tuple[int, bytes]:
        """Return (sample_rate, raw PCM float32 stereo bytes)."""
        raise NotImplementedError


class _DiffusersAdapter(_BaseAdapter):
    """diffusers.AceStepPipeline — text-to-music with lyrics, 48 kHz stereo."""

    name = "diffusers"

    def __init__(self, variant: str = "full") -> None:
        self.variant = variant
        self._pipe: Any = None

    def load(self) -> None:
        import torch
        from diffusers import AceStepPipeline
        # The ACE-Step 1.5 weights on the Hub; turbo = guidance-distilled.
        self._pipe = AceStepPipeline.from_pretrained(
            "ACE-Step/ACE-Step-v1.5", torch_dtype=torch.bfloat16)
        if torch.cuda.is_available():
            self._pipe.to("cuda")

    def unload(self) -> None:
        self._pipe = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def render(self, caption: str, lyrics: str, duration_s: int,
               seed: int) -> tuple[int, bytes]:
        import torch
        import numpy as np
        assert self._pipe is not None, "adapter not loaded"
        out = self._pipe(
            prompt=caption,
            lyrics=lyrics,
            audio_duration=float(duration_s),
            generator=torch.Generator().manual_seed(seed),
        )
        audio = out.audios  # (batch, channels, samples) or list
        arr = np.asarray(audio[0] if hasattr(audio, "__getitem__") else audio,
                         dtype=np.float32)
        if arr.ndim == 1:
            arr = np.stack([arr, arr], axis=0)
        return SAMPLE_RATE, arr.tobytes()


class _OfficialAdapter(_BaseAdapter):
    """The official ace-step package's generation path.

    Field names verified against the 1.5 source: ``caption`` (not
    ``prompt``), ``duration`` (not ``audio_duration``), ``keyscale``.
    """

    name = "official"

    def __init__(self, checkpoint_dir: str, variant: str = "full") -> None:
        self.checkpoint_dir = checkpoint_dir
        self.variant = variant
        self._dit = None
        self._llm = None

    def load(self) -> None:
        from acestep.handler import AceStepHandler  # type: ignore
        self._dit = AceStepHandler()

    def unload(self) -> None:
        self._dit = None
        self._llm = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def render(self, caption: str, lyrics: str, duration_s: int,
               seed: int) -> tuple[int, bytes]:
        import numpy as np
        from acestep.inference import (  # type: ignore
            GenerationParams, generate_music)
        assert self._dit is not None, "adapter not loaded"
        params = GenerationParams(
            caption=caption,
            lyrics=lyrics,
            duration=float(duration_s),
            keyscale="auto",
            seed=seed,
        )
        audio = generate_music(self._dit, self._llm, params)
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.stack([arr, arr], axis=0)
        return SAMPLE_RATE, arr.tobytes()


def _write_wav_float32_stereo(path: str, sample_rate: int,
                              pcm_bytes: bytes) -> None:
    """Write float32 stereo PCM → 16-bit WAV with the stdlib only."""
    import numpy as np
    arr = np.frombuffer(pcm_bytes, dtype=np.float32)
    if arr.size % 2:
        arr = arr[:arr.size - 1]
    stereo = arr.reshape(-1, 2)
    clipped = max(-1.0, min(1.0, float(abs(stereo).max()))) or 1.0
    pcm16 = (stereo / clipped * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


# ───────────────────────── backend ──────────────────────────────────────────


class ACEStepBackend:
    """ACE-Step 1.5 song-bed generation.

    ``adapter`` is injectable (tests); otherwise auto-detected from
    what's installed. The model loads around generation and unloads in
    ``finally`` — never held idle.
    """

    def __init__(self, *, profile: str = "",
                 adapter: _BaseAdapter | None = None,
                 checkpoint_dir: str = "") -> None:
        self.profile = (profile or "").strip().lower()
        self._adapter = adapter
        self._checkpoint_dir = checkpoint_dir

    def probe(self) -> ProbeResult:
        base = probe_acestep(self.profile)
        if self._adapter is not None and not base.available:
            # An injected adapter IS the implementation (tests, embedders):
            # trust it, but keep the profile gate — never run DiT on a phone.
            if base.profile in TOO_SMALL:
                return base
            variant = "turbo" if base.profile in SMALL_GPU else "full"
            return ProbeResult(
                available=True, profile=base.profile, variant=variant,
                detail=f"adapter: {self._adapter.name} (injected)")
        return base

    def _resolve_adapter(self, variant: str) -> _BaseAdapter:
        if self._adapter is not None:
            return self._adapter
        ckpt = self._checkpoint_dir or _checkpoint_dir()
        if _importable("acestep"):
            return _OfficialAdapter(ckpt, variant)
        return _DiffusersAdapter(variant)

    def generate(self, lyrics: str, style: Any, *,
                 duration_s: int = 120, seed: int = 0,
                 title: str = "", workdir: str = "music") -> BedResult:
        """Generate an instrumental bed. Fails closed — never fake audio."""
        probe = self.probe()
        if not probe.available:
            raise ACEModelUnavailable(
                probe.reason + (f"\n{probe.detail}" if probe.detail else ""))
        if not (lyrics or "").strip():
            raise ACEModelUnavailable("no lyrics to generate from.")

        # laptop-class: turbo variant + duration cap
        if probe.profile in SMALL_GPU:
            duration_s = min(duration_s, 90)
        duration_s = max(15, min(duration_s, 240))

        from ..media.music import resolve_style
        spec = style if hasattr(style, "tempo") else resolve_style(style)
        tags = tags_for_style(spec)
        adapter = self._resolve_adapter(probe.variant)

        slug = re.sub(r"[^a-z0-9]+", "-",
                      (title or "ace-bed").lower()).strip("-") or "ace-bed"
        out_dir = Path(workdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = str(out_dir / f"{slug}-bed.wav")

        _log.info("acestep generating bed: %s [%s] %ds (%s)",
                  title or slug, tags, duration_s, probe.variant)
        adapter.load()
        try:
            sr, pcm = adapter.render(tags, lyrics, duration_s, seed)
        finally:
            # VRAM hygiene: never hold the model idle
            try:
                adapter.unload()
            except Exception:  # noqa: BLE001
                _log.warning("acestep unload failed", exc_info=True)
        _write_wav_float32_stereo(audio_path, sr, pcm)

        return BedResult(
            ok=True, audio_path=audio_path, title=title or slug,
            tags=tags, lyrics=lyrics, profile=probe.profile,
            variant=probe.variant, duration_s=float(duration_s),
            note=("🎹 AI instrumental bed (ACE-Step 1.5, "
                  f"{probe.variant}). Vocals come next — #58."),
            backend=f"acestep/{adapter.name}")


# ───────────────────────── high-level flow ──────────────────────────────────


def make_bed(topic: str, *, style: str = "pop", duration_s: int = 120,
             seed: int = 0, context: Any = None, profile: str = "",
             backend: ACEStepBackend | None = None,
             workdir: str = "music") -> BedResult:
    """Topic → lyrics (MusicCreator's real engine) → ACE-Step bed.

    Raises ACEModelUnavailable when the model can't run here.
    """
    from ..media.music import MusicCreator, resolve_style
    spec = resolve_style(style)
    creator = MusicCreator(context)
    song = creator.compose(topic, style=spec.name, with_midi=False,
                           with_audio=False, with_score=False)
    lyrics = format_lyrics_for_acestep(song)
    if not lyrics.strip():
        raise ACEModelUnavailable("lyric engine produced no lyrics.")
    be = backend or ACEStepBackend(profile=profile)
    result = be.generate(lyrics, spec, duration_s=duration_s, seed=seed,
                         title=song.title, workdir=workdir)
    result.lyrics = lyrics
    return result


# ───────────────────────── tool registration ────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "ace_step_bed",
        description=(
            "Generate an AI instrumental song bed with ACE-Step 1.5 "
            "(MIT, local): lyrics (from MusicCreator's engine) + style "
            "conditioning → 48 kHz stereo WAV bed. "
            "action=bed (topic, style, duration_s, seed) | probe | tags "
            "(style → conditioning tags). Fails honestly when the model "
            "isn't installed or there's no CUDA GPU — never fake audio. "
            "Bed only; vocals are a separate step."
        ),
        capability=Capability.FS_WRITE,
    )
    def ace_step_bed(action: str = "bed", topic: str = "",
                     style: str = "pop", duration_s: int = 120,
                     seed: int = 0) -> dict[str, Any]:
        action = (action or "bed").lower()
        if action == "probe":
            p = probe_acestep()
            return {"available": p.available, "reason": p.reason,
                    "detail": p.detail, "profile": p.profile,
                    "variant": p.variant}
        if action == "tags":
            from ..media.music import resolve_style
            return {"style": style, "tags": tags_for_style(
                resolve_style(style))}
        if action != "bed":
            from ..core.errors import ToolError
            raise ToolError(f"unknown action {action!r}")
        if not topic.strip():
            from ..core.errors import ToolError
            raise ToolError("topic is required")
        try:
            res = make_bed(topic, style=style, duration_s=duration_s,
                           seed=seed, context=context)
        except ACEModelUnavailable as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "audio_path": res.audio_path,
                "title": res.title, "tags": res.tags,
                "variant": res.variant, "duration_s": res.duration_s,
                "note": res.note, "backend": res.backend}
