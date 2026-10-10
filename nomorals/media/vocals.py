"""Vocal pipeline: DiffSinger → RVC → mix → master.

Phased vocal pipeline (#58), stages 2–3 — the missing half of song
generation (#57 produces the bed; this produces the voice):

1. **DiffSinger** (open-source) renders lyrics + MIDI melody → sung vocal.
2. **RVC** (MIT) converts that vocal into a target voice.
3. The vocal is mixed over the #57 bed and mastered to a final WAV.

Audience rule (#30), structural not advisory: voices derived from XTTS
(non-commercial license) are allowed for the **private** audience only.
For the **public** audience XTTS-derived voices are excluded — use
MIT/Apache-2.0 voices (e.g. Chatterbox-derived). RVC itself is MIT.

Fail-closed everywhere: missing model → install recipe, never fake
audio. GPU work needs a real machine (profile-gated); the model
loads around generation and unloads in ``finally``.
"""

from __future__ import annotations

import logging
import os
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

__all__ = [
    "VocalModelUnavailable",
    "ProbeResult",
    "VoiceInfo",
    "VocalResult",
    "FullSongResult",
    "NONCOMMERCIAL_SOURCES",
    "INSTALL_DIFFSINGER",
    "INSTALL_RVC",
    "TOO_SMALL",
    "select_voice",
    "probe_diffsinger",
    "probe_rvc",
    "pick_backend",
    "VOCAL_BACKENDS",
    "VocalChain",
    "RVCVoiceRegistry",
    "DEFAULT_VOICE_REGISTRY",
    "parse_full_request",
    "register",
]

INSTALL_DIFFSINGER = (
    "DiffSinger is open-source but not pip-installable as one package:\n"
    "  git clone https://github.com/openvpi/DiffSinger\n"
    "  (needs Python 3.8+, an acoustic model + vocoder checkpoint)"
)
INSTALL_RVC = (
    "pip install rvc-python  (MIT)\n"
    "or: git clone https://github.com/RVC-Project/"
    "Retrieval-based-Voice-Conversion\n"
    "(needs a trained voice model .pth + .index for the target voice)"
)

#: Voice sources that are non-commercial — structurally excluded for
#: the public audience, mirroring nomorals/voice/tts.py.
NONCOMMERCIAL_SOURCES = frozenset({"xtts"})

#: Profiles too small for DiffSinger/RVC inference.
TOO_SMALL = frozenset({"termux", "mobile", "embedded"})

#: Melody note: (midi_pitch, start_beat, duration_beats).
Melody = list


class VocalModelUnavailable(RuntimeError):
    """A vocal stage can't run here — install recipe attached."""


@dataclass
class ProbeResult:
    available: bool
    reason: str = ""
    detail: str = ""
    profile: str = ""


@dataclass
class VoiceInfo:
    """One convertible voice.

    ``source``: where the voice came from (xtts / chatterbox / rvc / ...).
    ``license``: "noncommercial" (XTTS-derived) or "mit" (safe for public).
    ``model_path``: the RVC .pth (may be empty → fail closed at convert).
    """
    voice_id: str
    source: str = "rvc"
    license: str = "mit"
    model_path: str = ""


@dataclass
class VocalResult:
    ok: bool
    audio_path: str = ""
    stage: str = ""          # "render" | "convert"
    voice_id: str = ""
    audience: str = "private"
    error: str = ""
    note: str = ""


@dataclass
class FullSongResult:
    ok: bool
    master_path: str = ""
    bed_path: str = ""
    vocal_path: str = ""
    title: str = ""
    voice_id: str = ""
    audience: str = "private"
    error: str = ""
    note: str = ("AI-generated instrumental + AI vocals — "
                 "label it as such wherever it goes")


def _importable(name: str) -> bool:
    try:
        __import__(name)
    except Exception:  # noqa: BLE001
        return False
    return True


def _detect_profile_kind() -> str:
    try:
        from ..core.profile import detect_profile
        return str(detect_profile().kind).lower()
    except Exception:  # noqa: BLE001
        return ""


def select_voice(voice_id: str, audience: str,
                 registry: dict[str, VoiceInfo] | None = None) -> VoiceInfo:
    """Pick a voice for the audience. Structural XTTS exclusion.

    Raises :class:`VocalModelUnavailable` when the voice is unknown or
    when a non-commercial (XTTS-derived) voice is requested for the
    public audience — never silently downgrades.
    """
    aud = (audience or "private").strip().lower()
    if aud not in ("private", "public"):
        raise VocalModelUnavailable(
            f"audience must be 'private' or 'public', got {audience!r}")
    reg = registry or {}
    vid = (voice_id or "").strip()
    if vid not in reg:
        known = ", ".join(sorted(reg)) or "(no voices registered)"
        raise VocalModelUnavailable(
            f"unknown voice {vid!r}. Known voices: {known}")
    voice = reg[vid]
    if aud == "public" and (
            voice.license == "noncommercial"
            or voice.source.lower() in NONCOMMERCIAL_SOURCES):
        raise VocalModelUnavailable(
            f"voice {vid!r} is {voice.source}-derived (non-commercial) and "
            "cannot serve the public audience — use an MIT/Apache-2.0 "
            "voice (e.g. a Chatterbox-derived one)")
    return voice


def probe_diffsinger(profile: str = "") -> ProbeResult:
    """Check DiffSinger availability. Never raises."""
    prof = (profile or "").strip().lower() or _detect_profile_kind()
    if prof in TOO_SMALL:
        return ProbeResult(
            available=False, profile=prof,
            reason=("DiffSinger needs a real machine — singing synthesis "
                    "won't fit this device."))
    if _importable("diffsinger"):
        return ProbeResult(available=True, profile=prof,
                           detail="diffsinger package importable")
    import shutil
    if shutil.which("diffsinger") or shutil.which("ds-infer"):
        return ProbeResult(available=True, profile=prof,
                           detail="DiffSinger CLI on PATH")
    return ProbeResult(
        available=False, profile=prof,
        reason="DiffSinger isn't installed here.",
        detail=INSTALL_DIFFSINGER)


def probe_rvc(profile: str = "") -> ProbeResult:
    """Check RVC availability. Never raises."""
    prof = (profile or "").strip().lower() or _detect_profile_kind()
    if prof in TOO_SMALL:
        return ProbeResult(
            available=False, profile=prof,
            reason=("RVC voice conversion needs a real machine — "
                    "it won't fit this device."))
    if _importable("rvc_python") or _importable("rvc"):
        return ProbeResult(available=True, profile=prof,
                           detail="rvc package importable")
    import shutil
    if shutil.which("rvc") or shutil.which("infer"):
        return ProbeResult(available=True, profile=prof,
                           detail="RVC tooling on PATH")
    return ProbeResult(
        available=False, profile=prof,
        reason="RVC isn't installed here.",
        detail=INSTALL_RVC)


#: Scored vocal-backend selection (openmontage provider-selector pattern).
#: Each backend: (quality 0..1, latency 0..1 where 1 = fastest).
#: Score = availability_gate × (0.6 × quality + 0.4 × latency).
#: Local-first, honest reasons — never a silent fallback.
VOCAL_BACKENDS: tuple[tuple[str, float, float], ...] = (
    ("diffsinger", 1.0, 0.3),    # best quality, heavy
    ("vocal_lite", 0.55, 1.0),   # offline preview, instant
)


def pick_backend(purpose: str = "sing", profile: str = "",
                 prefer: str = "auto") -> dict:
    """Pick the vocal backend by scored selection. Never raises.

    Returns {"backend", "score", "available", "reason", "trail"} where
    trail is the auditable per-backend scoring. ``prefer`` can force a
    backend name — if it's unavailable the result says so honestly
    instead of silently substituting.
    """
    trail: list[dict] = []
    probes = {"diffsinger": probe_diffsinger(profile),
              "vocal_lite": ProbeResult(available=True, profile=profile or
                                        _detect_profile_kind(),
                                        detail="stdlib preview, always here")}
    for name, quality, latency in VOCAL_BACKENDS:
        probe = probes[name]
        score = round((0.6 * quality + 0.4 * latency)
                      if probe.available else 0.0, 3)
        trail.append({"backend": name, "available": probe.available,
                      "quality": quality, "latency": latency,
                      "score": score,
                      "reason": "" if probe.available else probe.reason})
    if prefer and prefer != "auto":
        entry = next((t for t in trail if t["backend"] == prefer), None)
        if entry is None:
            return {"backend": "", "score": 0.0, "available": False,
                    "reason": f"unknown vocal backend {prefer!r}",
                    "trail": trail}
        return {"backend": prefer, "score": entry["score"],
                "available": entry["available"],
                "reason": entry["reason"] or f"preferred by caller",
                "trail": trail}
    ranked = sorted(trail, key=lambda t: -t["score"])
    best = ranked[0]
    return {"backend": best["backend"], "score": best["score"],
            "available": best["available"],
            "reason": best["reason"] or
            f"scored {best['score']:.2f} (quality+latency)",
            "trail": trail}


def _read_wav(path: str) -> tuple[Any, int, int]:
    """Read a WAV → (mono float64 samples, sample_rate, channels)."""
    import numpy as np
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        width = wf.getsampwidth()
    dtype = {1: np.int8, 2: "<i2", 4: "<i4"}[width]
    arr = np.frombuffer(raw, dtype=np.dtype(dtype)).astype(np.float64)
    scale = float(2 ** (8 * width - 1))
    arr = arr / scale
    if nch > 1:
        arr = arr.reshape(-1, nch)
    mono = arr.mean(axis=1) if arr.ndim > 1 else arr
    return mono, sr, nch


def _write_wav_stereo(path: str, left: Any, right: Any,
                      sample_rate: int) -> bool:
    """Two mono float arrays → 16-bit stereo WAV.

    Returns ``True`` on success, ``False`` (logged, never raises) when the
    frames would exceed :data:`.caps.MAX_AUDIO_WRITE_BYTES` — nothing is
    written on refusal.
    """
    from . import caps

    import numpy as np
    l = np.asarray(left, dtype=np.float64).reshape(-1)
    r = np.asarray(right, dtype=np.float64).reshape(-1)
    n = max(l.size, r.size)
    l = np.pad(l, (0, n - l.size))
    r = np.pad(r, (0, n - r.size))
    stereo = np.stack([l, r], axis=1)
    peak = float(abs(stereo).max()) or 1.0
    pcm16 = (stereo / peak * 32767).astype("<i2")
    frames = pcm16.tobytes()
    ok, reason = caps.check_write_size(
        caps.wav_expected_bytes(len(frames)), caps.MAX_AUDIO_WRITE_BYTES)
    if not ok:
        caps.refuse_write(f"_write_wav_stereo({path})", reason)
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(frames)
    return True


def _resample(mono: Any, src_sr: int, dst_sr: int) -> Any:
    import numpy as np
    mono = np.asarray(mono, dtype=np.float64)
    if src_sr == dst_sr or mono.size == 0:
        return mono
    ratio = dst_sr / src_sr
    n = int(round(mono.size * ratio))
    x_old = np.linspace(0, 1, mono.size)
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, mono)


class _DiffSingerAdapter:
    """DiffSinger lyrics+MIDI → sung vocal. Injectable for tests."""

    name = "diffsinger"

    def __init__(self) -> None:
        self._impl = None

    def load(self) -> None:
        if not (_importable("diffsinger")):
            import shutil
            if not (shutil.which("diffsinger") or shutil.which("ds-infer")):
                raise VocalModelUnavailable(INSTALL_DIFFSINGER)
        self._impl = True  # marker; real impl shells to the package/CLI

    def unload(self) -> None:
        self._impl = None

    def render(self, lyrics: str, melody: list,
               out_path: str) -> str:
        raise VocalModelUnavailable(
            "DiffSinger in-process rendering isn't wired to a checkpoint "
            "here.\n" + INSTALL_DIFFSINGER)


class _RVCAdapter:
    """RVC voice conversion. Injectable for tests."""

    name = "rvc"

    def __init__(self) -> None:
        self._impl = None

    def load(self) -> None:
        if not (_importable("rvc_python") or _importable("rvc")):
            import shutil
            if not shutil.which("rvc"):
                raise VocalModelUnavailable(INSTALL_RVC)
        self._impl = True

    def unload(self) -> None:
        self._impl = None

    def convert(self, vocal_wav: str, model_path: str,
                out_path: str) -> str:
        raise VocalModelUnavailable(
            "RVC in-process conversion isn't wired to a voice model here.\n"
            + INSTALL_RVC)


class VocalChain:
    """Lyrics + melody → sung vocal → converted voice → full song.

    Adapters and the voice registry are injectable (tests); otherwise
    the real DiffSinger/RVC adapters are used and fail closed with
    install recipes. Models load around generation, unload in
    ``finally`` — never held idle.
    """

    def __init__(self, *, profile: str = "",
                 diffsinger: "_DiffSingerAdapter | None" = None,
                 rvc: "_RVCAdapter | None" = None,
                 voices: dict[str, VoiceInfo] | None = None) -> None:
        self.profile = (profile or "").strip().lower()
        self._diffsinger = diffsinger
        self._rvc = rvc
        self._voices = voices or {}

    def _check_profile(self, stage: str) -> None:
        prof = self.profile or _detect_profile_kind()
        if prof in TOO_SMALL:
            raise VocalModelUnavailable(
                f"{stage} needs a real machine — vocal synthesis won't "
                f"fit this device (profile {prof!r}).")

    # -- stage 1: DiffSinger -------------------------------------------------
    def render(self, lyrics: str, melody: list,
               *, workdir: str = "vocals", title: str = "vocal") -> VocalResult:
        """Lyrics + melody → sung vocal WAV. Fails closed."""
        self._check_profile("DiffSinger render")
        if not (lyrics or "").strip():
            raise VocalModelUnavailable("no lyrics to sing")
        if not melody:
            raise VocalModelUnavailable("no melody — need MIDI notes")
        probe = probe_diffsinger(self.profile)
        adapter = self._diffsinger
        if adapter is None:
            if not probe.available:
                raise VocalModelUnavailable(
                    probe.reason + (f"\n{probe.detail}" if probe.detail else ""))
            adapter = _DiffSingerAdapter()
        out = str(Path(workdir) / f"{_slug(title)}_dry.wav")
        try:
            adapter.load()
            path = adapter.render(lyrics, melody, out)
        finally:
            try:
                adapter.unload()
            except Exception:  # noqa: BLE001
                _log.debug("diffsinger unload failed", exc_info=True)
        if not path or not os.path.isfile(path):
            raise VocalModelUnavailable(
                "DiffSinger produced no audio — nothing faked")
        return VocalResult(ok=True, audio_path=path, stage="render",
                           note="DiffSinger sung vocal (dry, unconverted)")

    # -- stage 2: RVC --------------------------------------------------------
    def convert(self, vocal_wav: str, voice_id: str, *,
                audience: str = "private",
                workdir: str = "vocals") -> VocalResult:
        """RVC-convert a vocal into the target voice.

        The #30 audience rule applies: XTTS-derived voices are
        structurally excluded for the public audience.
        """
        self._check_profile("RVC convert")
        if not vocal_wav or not os.path.isfile(vocal_wav):
            raise VocalModelUnavailable(f"no such vocal file: {vocal_wav!r}")
        voice = select_voice(voice_id, audience, self._voices)
        if not voice.model_path or not os.path.isfile(voice.model_path):
            raise VocalModelUnavailable(
                f"voice {voice.voice_id!r} has no RVC model file — "
                "train or download one first, then register its .pth path")
        probe = probe_rvc(self.profile)
        adapter = self._rvc
        if adapter is None:
            if not probe.available:
                raise VocalModelUnavailable(
                    probe.reason + (f"\n{probe.detail}" if probe.detail else ""))
            adapter = _RVCAdapter()
        out = str(Path(workdir)
                  / f"{Path(vocal_wav).stem}_{voice.voice_id}.wav")
        try:
            adapter.load()
            path = adapter.convert(vocal_wav, voice.model_path, out)
        finally:
            try:
                adapter.unload()
            except Exception:  # noqa: BLE001
                _log.debug("rvc unload failed", exc_info=True)
        if not path or not os.path.isfile(path):
            raise VocalModelUnavailable(
                "RVC produced no audio — nothing faked")
        return VocalResult(ok=True, audio_path=path, stage="convert",
                           voice_id=voice.voice_id, audience=audience,
                           note=f"RVC voice conversion → {voice.voice_id} "
                                f"({voice.source}, {audience})")

    # -- stage 3: mix + master ----------------------------------------------
    def mix(self, bed_wav: str, vocal_wav: str, *, out_path: str,
            vocal_gain: float = 1.0, bed_gain: float = 0.8) -> str:
        """Mix bed + vocal → stereo master WAV. Pure DSP, always works."""
        if not bed_wav or not os.path.isfile(bed_wav):
            raise VocalModelUnavailable(f"no such bed file: {bed_wav!r}")
        if not vocal_wav or not os.path.isfile(vocal_wav):
            raise VocalModelUnavailable(f"no such vocal file: {vocal_wav!r}")
        import numpy as np
        bed, bed_sr, _ = _read_wav(bed_wav)
        voc, voc_sr, _ = _read_wav(vocal_wav)
        if voc_sr != bed_sr:
            voc = _resample(voc, voc_sr, bed_sr)
        n = max(bed.size, voc.size)
        bed_p = np.pad(bed, (0, n - bed.size))
        voc_p = np.pad(voc, (0, n - voc.size))
        mixed_l = bed_p * bed_gain + voc_p * vocal_gain
        mixed_r = bed_p * bed_gain + voc_p * vocal_gain * 0.9
        _write_wav_stereo(out_path, mixed_l, mixed_r, bed_sr)
        return out_path

    # -- full chain -----------------------------------------------------------
    def full_song(self, bed_wav: str, lyrics: str, melody: list,
                  voice_id: str, *, audience: str = "private",
                  title: str = "song", workdir: str = "songs") -> FullSongResult:
        """Bed + DiffSinger vocal + RVC voice → mixed master WAV."""
        rendered = self.render(lyrics, melody, workdir=workdir, title=title)
        converted = self.convert(rendered.audio_path, voice_id,
                                 audience=audience, workdir=workdir)
        master = str(Path(workdir) / f"{_slug(title)}_master.wav")
        self.mix(bed_wav, converted.audio_path, out_path=master)
        return FullSongResult(
            ok=True, master_path=master, bed_path=bed_wav,
            vocal_path=converted.audio_path, title=title,
            voice_id=converted.voice_id, audience=audience)


def _slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "untitled").lower()).strip("-")
    return s[:limit] or "untitled"


# ── RVC voice registry ────────────────────────────────────────────────────

#: Default registry path (owner-scoped).
DEFAULT_VOICE_REGISTRY = os.path.expanduser("~/.nomorals/media/rvc_voices.json")


class RVCVoiceRegistry:
    """Persistent registry of RVC-convertible voices.

    Each voice records its ``source`` (xtts / chatterbox / rvc / ...);
    XTTS-derived voices are non-commercial and structurally excluded
    for the public audience (the #30 rule, enforced by
    :func:`select_voice`).
    """

    def __init__(self, path: str = "") -> None:
        self.path = path or DEFAULT_VOICE_REGISTRY
        self._voices: dict[str, VoiceInfo] = {}
        self._load()

    def _load(self) -> None:
        try:
            if os.path.isfile(self.path):
                import json
                with open(self.path, encoding="utf-8") as fh:
                    raw = json.load(fh) or {}
                for vid, v in raw.items():
                    self._voices[vid] = VoiceInfo(
                        voice_id=vid,
                        source=str(v.get("source", "rvc")),
                        license=str(v.get("license", "mit")),
                        model_path=str(v.get("model_path", "")))
        except Exception:  # noqa: BLE001
            _log.debug("voice registry load failed", exc_info=True)

    def _save(self) -> None:
        import json
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({vid: {"source": v.source, "license": v.license,
                             "model_path": v.model_path}
                       for vid, v in self._voices.items()}, fh, indent=2)

    def add(self, voice_id: str, model_path: str,
            source: str = "rvc") -> VoiceInfo:
        """Register a voice. XTTS source → non-commercial license."""
        vid = (voice_id or "").strip()
        if not vid:
            raise VocalModelUnavailable("voice_id is required")
        if not model_path or not os.path.isfile(model_path):
            raise VocalModelUnavailable(
                f"no such RVC model file: {model_path!r}")
        src = (source or "rvc").strip().lower()
        lic = ("noncommercial" if src in NONCOMMERCIAL_SOURCES
               else "mit")
        voice = VoiceInfo(voice_id=vid, source=src, license=lic,
                          model_path=model_path)
        self._voices[vid] = voice
        self._save()
        return voice

    def remove(self, voice_id: str) -> bool:
        if voice_id in self._voices:
            del self._voices[voice_id]
            self._save()
            return True
        return False

    def all(self) -> dict[str, VoiceInfo]:
        return dict(self._voices)

    def default(self) -> VoiceInfo | None:
        """The default voice: the only one, or 'default' if named."""
        if "default" in self._voices:
            return self._voices["default"]
        if len(self._voices) == 1:
            return next(iter(self._voices.values()))
        return None


_FULL_RE = re.compile(
    r"(?i)^(?:make|generate|create)\s+(?:me\s+)?(?:a\s+|an\s+)?"
    r"(?P<rest>.+?)\s+(?:song|track)(?:\s+about\s+(?P<topic>.+))?$"
)


def parse_full_request(text: str, styles: Any = None) -> dict | None:
    """Parse full-song requests: 'make me an afrobeats song about Lagos'.

    Returns {"topic", "style"} or None. Delegates style resolution to
    the #57 bed parser when the text matches its shape.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    low = raw.lower()
    if low.startswith("/music full"):
        body = raw[len("/music full"):].strip()
        if not body:
            return None
        words = body.split()
        style = "pop"
        if styles is not None and words and words[-1].lower() in styles:
            style = words[-1].lower()
            words = words[:-1]
        topic = " ".join(words).strip()
        return {"topic": topic or "untitled", "style": style} if topic else None
    try:
        from .ace_step import parse_bed_request
        bed = parse_bed_request(raw, styles)
    except Exception:  # noqa: BLE001
        bed = None
    if bed is not None:
        return {"topic": bed.topic, "style": bed.style}
    m = _FULL_RE.match(raw)
    if not m:
        return None
    rest = (m.group("rest") or "").strip()
    topic = (m.group("topic") or "").strip()
    style = "pop"
    if styles is not None:
        first = rest.split()[0].lower() if rest.split() else ""
        if first in styles:
            style = first
    return {"topic": topic or rest or "untitled", "style": style}


def make_full_song(topic: str, *, style: str = "pop", voice_id: str = "",
                   audience: str = "private", duration_s: int = 120,
                   seed: int = 0, context: Any = None, profile: str = "",
                   workdir: str = "songs",
                   chain: "VocalChain | None" = None) -> FullSongResult:
    """Topic → bed → vocals → master.

    Vocal strategy, in order:
    1. DiffSinger + RVC (full singing synthesis) when both are available
       and a voice_id is given/registered.
    2. TTS vocal track (sung/hook/hum via vocal_lite) when the heavy
       models aren't available — e.g. on the phone. Real vocals, real
       mix, honest note about the backend.
    3. Honest instrumental when no TTS backend exists either.

    Never fails closed with "no voice registered" — always produces a
    song. Raises VocalModelUnavailable only when nothing at all can run.
    """
    from .ace_step import make_bed, ACEModelUnavailable
    if not (topic or "").strip():
        raise VocalModelUnavailable("topic is required")
    bed = make_bed(topic, style=style, duration_s=duration_s, seed=seed,
                   context=context, profile=profile,
                   workdir=str(Path(workdir) / "bed"))
    from .music import MusicCreator, resolve_style
    spec = resolve_style(style)
    song = MusicCreator(context).compose(topic, style=spec.name,
                                         with_midi=False, with_audio=False,
                                         with_score=False)
    melody = list(song.melody_notes or [])
    title = bed.title or topic

    # Strategy 1: full DiffSinger + RVC when everything's available.
    ds = probe_diffsinger(profile)
    rv = probe_rvc(profile)
    if ds.available and rv.available and (voice_id or "").strip():
        if not melody:
            raise VocalModelUnavailable(
                "the composer produced no melody notes for this song")
        vc = chain or VocalChain(profile=profile,
                                 voices=RVCVoiceRegistry().all())
        select_voice(voice_id, audience, vc._voices)
        return vc.full_song(bed.audio_path, bed.lyrics, melody, voice_id,
                            audience=audience, title=title,
                            workdir=workdir)

    # Strategy 2: TTS vocal track — the phone path. Real lyrics sung/
    # spoken over the bed via the voice catalogue TTS.
    from .vocal_lite import add_vocal_track
    vr = add_vocal_track(song, bed.audio_path, str(Path(workdir) / "tts"),
                         vocal_mode="auto",
                         melody_events=melody or None)
    if vr.get("ok"):
        return FullSongResult(
            ok=True, master_path=str(vr["path"]), bed_path=bed.audio_path,
            vocal_path=str(vr.get("path", "")), title=title,
            voice_id="tts:" + str(vr.get("backend", "auto")),
            audience=audience,
            note=(str(vr.get("note", "")) +
                  " (TTS vocals — DiffSinger/RVC not available here)"))

    # Strategy 3: honest instrumental.
    return FullSongResult(
        ok=True, master_path=bed.audio_path, bed_path=bed.audio_path,
        vocal_path="", title=title, voice_id="", audience=audience,
        note="instrumental — " + str(vr.get("reason", "no vocal backend")))


def register(registry: Any) -> None:
    from ..core.policy import Capability
    context = registry.context

    @registry.register(
        "vocal_chain",
        description=(
            "Full vocal pipeline (DiffSinger → RVC → mix → master), MIT/open "
            "local: action=probe (diffsinger/rvc/demucs status) | separate "
            "(audio_path → 4 Demucs stems) | render (lyrics+melody → sung "
            "vocal) | convert (vocal_wav, voice_id → RVC voice) | full "
            "(topic, style, voice_id → bed + vocals + master). Audience "
            "rule: XTTS-derived voices are private-only, structurally "
            "excluded for public. Fails honestly — never fake audio."
        ),
        capability=Capability.FS_WRITE,
    )
    def vocal_chain(action: str = "probe", audio_path: str = "",
                    lyrics: str = "", melody: str = "",
                    voice_id: str = "", audience: str = "private",
                    topic: str = "", style: str = "pop") -> dict[str, Any]:
        action = (action or "probe").lower()
        if action == "probe":
            from .stems import probe_demucs
            pd = probe_demucs()
            pr = probe_rvc()
            pf = probe_diffsinger()
            return {
                "demucs": {"available": pd.available, "reason": pd.reason},
                "rvc": {"available": pr.available, "reason": pr.reason},
                "diffsinger": {"available": pf.available, "reason": pf.reason},
            }
        if action == "separate":
            from .stems import separate, DemucsUnavailable
            try:
                s = separate(audio_path)
            except DemucsUnavailable as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "stems": s.paths(), "model": s.model}
        if action == "render":
            import json as _json
            try:
                mel = _json.loads(melody) if melody else []
                r = VocalChain(voices=RVCVoiceRegistry().all()).render(
                    lyrics, mel)
            except VocalModelUnavailable as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "audio_path": r.audio_path}
        if action == "convert":
            try:
                r = VocalChain(voices=RVCVoiceRegistry().all()).convert(
                    audio_path, voice_id, audience=audience)
            except VocalModelUnavailable as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "audio_path": r.audio_path,
                    "voice_id": r.voice_id}
        if action == "voices":
            reg = RVCVoiceRegistry()
            return {"voices": [
                {"voice_id": v.voice_id, "source": v.source,
                 "license": v.license,
                 "model": bool(v.model_path and os.path.isfile(v.model_path))}
                for v in reg.all().values()]}
        if action == "voice-add":
            try:
                v = RVCVoiceRegistry().add(
                    voice_id, audio_path, source=style or "rvc")
            except VocalModelUnavailable as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "voice_id": v.voice_id,
                    "license": v.license}
        if action == "full":
            try:
                r = make_full_song(topic, style=style, voice_id=voice_id,
                                   audience=audience, context=context)
            except Exception as exc:  # noqa: BLE001 - both model errors
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "master_path": r.master_path,
                    "note": r.note}
        from ..core.errors import ToolError
        raise ToolError(f"unknown action {action!r}")
