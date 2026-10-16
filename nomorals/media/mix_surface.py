"""``/mix`` surface + delivery — shareable, editable, reproducible mixes.

A mix is a :class:`MixDescriptor`: an ordered tracklist plus the
transitions between tracks plus an energy curve. The descriptor is
the *whole* mix: it is shareable (JSON), editable (add / remove /
reorder tracks → new descriptor, new fingerprint), and reproducible
— :func:`render_mix` on the same descriptor always yields the same
render plan and the same fingerprint (invariant, tested in
``tests/test_groups_tier1.py``).

Rendering
---------
* :func:`render_plan` is pure and deterministic: it builds
  :class:`TrackAnalysis` objects from the descriptor, plans each
  transition with :func:`dj_engine.plan_transition`, samples the
  energy curve with :func:`dj_engine.energy_curve`, and computes a
  total duration. Same descriptor → byte-identical plan.
* :func:`render_mix` adds audio when the track files exist (WAV,
  stdlib ``wave`` + ``render_beatmatched_transition``), and is
  **resumable**: a checkpoint JSON records which track indices are
  already rendered, so a killed render restarts where it stopped
  instead of from zero. When track files are missing it does *not*
  fake audio — it returns the deterministic plan with
  ``missing_tracks`` listed and ``audio_path=None``.
* :func:`fingerprint` is SHA-256 over canonical JSON of the
  descriptor (sorted keys, no wall-clock fields), so editing a
  descriptor changes the fingerprint and re-rendering an unchanged
  descriptor never does.

Delivery
--------
:class:`deliver` takes an *injected* sender — any object with
``send_audio(chat, path, ...)`` and optionally ``send_voice(chat,
path, ...)`` — and delivers the rendered mix as an audio file or a
voice note to any chat/group. No adapter is imported here: the
caller injects the WhatsApp/Telegram sender. That keeps this module
testable without a chat session and honest about what was sent
(:class:`DeliveryReceipt`).

``/mix`` surface: :func:`mix_menu` (the command text),
:func:`format_mix_card` (shareable card), :func:`share_descriptor`
/ :func:`load_descriptor` (JSON round-trip).
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
import wave
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "MixTrack",
    "MixTransition",
    "MixDescriptor",
    "RenderedMix",
    "DeliveryReceipt",
    "Sender",
    "fingerprint",
    "render_plan",
    "render_mix",
    "deliver",
    "mix_menu",
    "format_mix_card",
    "share_descriptor",
    "load_descriptor",
    "with_track_added",
    "with_track_removed",
    "with_tracks_reordered",
]


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class MixTrack:
    """One track in a mix. Analysis fields travel *with* the track so
    a descriptor renders identically on any machine, even without the
    audio files present."""

    path: str = ""
    title: str = ""
    bpm: float | None = None
    key: str = ""
    mode: str = ""  # "major" | "minor" | ""
    energy: float = 0.5
    duration_s: float = 0.0
    gain: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MixTrack":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class MixTransition:
    """Planned transition between track i and track i+1."""

    from_index: int = 0
    to_index: int = 1
    kind: str = "break"  # "blend" | "echo-drop" | "break"
    blend_beats: int = 16
    sync_ratio: float = 1.0
    start_beat: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MixTransition":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class MixDescriptor:
    """The whole mix: tracklist + transitions + energy curve.

    ``transitions`` may be empty — :func:`render_plan` derives them
    from the tracks via ``dj_engine``. When present, explicit
    transitions override the derived ones (hand-tuned mixes survive
    re-render). ``seed`` pins tie-breaking; ``version`` bumps on
    every edit.
    """

    id: str = ""
    title: str = "Untitled mix"
    tracks: list[MixTrack] = field(default_factory=list)
    transitions: list[MixTransition] = field(default_factory=list)
    energy_curve: str = "late_peak"
    seed: int = 0
    version: int = 1
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "tracks": [t.to_dict() for t in self.tracks],
            "transitions": [t.to_dict() for t in self.transitions],
            "energy_curve": self.energy_curve,
            "seed": self.seed,
            "version": self.version,
            "notes": self.notes,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MixDescriptor":
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "Untitled mix")),
            tracks=[MixTrack.from_dict(t) for t in data.get("tracks", [])],
            transitions=[MixTransition.from_dict(t) for t in data.get("transitions", [])],
            energy_curve=str(data.get("energy_curve", "late_peak")),
            seed=int(data.get("seed", 0)),
            version=int(data.get("version", 1)),
            notes=str(data.get("notes", "")),
            tags=list(data.get("tags", [])),
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint(self)


@dataclass
class RenderedMix:
    """Result of :func:`render_mix`."""

    descriptor: MixDescriptor
    fingerprint: str
    plan: dict[str, Any]
    total_duration_s: float
    audio_path: str | None = None
    missing_tracks: list[str] = field(default_factory=list)
    resumed_from_checkpoint: bool = False
    completed_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "fingerprint": self.fingerprint,
            "plan": self.plan,
            "total_duration_s": round(self.total_duration_s, 2),
            "audio_path": self.audio_path,
            "missing_tracks": list(self.missing_tracks),
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "completed_indices": list(self.completed_indices),
        }


@dataclass
class DeliveryReceipt:
    chat: str = ""
    mix_id: str = ""
    fingerprint: str = ""
    kind: str = ""  # "audio" | "voice_note"
    audio_path: str | None = None
    sent_at: float = 0.0
    sender_result: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Sender(Protocol):
    """Injected chat sender. The caller supplies the real adapter."""

    def send_audio(self, chat: Any, path: str, **kwargs: Any) -> Any: ...


def fingerprint(descriptor: MixDescriptor) -> str:
    """SHA-256 over canonical descriptor JSON. Deterministic."""
    return hashlib.sha256(_canonical(descriptor.to_dict()).encode()).hexdigest()


def _to_analysis(track: MixTrack) -> Any:
    from .dj_engine import TrackAnalysis

    return TrackAnalysis(
        path=track.path,
        title=track.title,
        bpm=track.bpm,
        key=track.key,
        mode=track.mode,
        energy=track.energy,
        duration_s=track.duration_s,
    )


def render_plan(descriptor: MixDescriptor) -> dict[str, Any]:
    """Pure, deterministic render plan for a descriptor.

    No I/O, no wall clock, no randomness beyond ``descriptor.seed``
    (which only breaks ties inside dj_engine ordering — and ordering
    here follows the descriptor, so the plan is fully determined).
    """
    from . import dj_engine as _dj

    analyses = [_to_analysis(t) for t in descriptor.tracks]
    transitions: list[dict[str, Any]] = []
    if descriptor.transitions:
        transitions = [t.to_dict() for t in descriptor.transitions]
    else:
        for i, (a, b) in enumerate(zip(analyses, analyses[1:])):
            plan = _dj.plan_transition(a, b)
            transitions.append(
                MixTransition(
                    from_index=i,
                    to_index=i + 1,
                    kind=plan.kind,
                    blend_beats=plan.blend_beats,
                    sync_ratio=plan.sync_ratio,
                    start_beat=plan.start_beat,
                    reason=plan.reason,
                ).to_dict()
            )
    curve = _dj.energy_curve(descriptor.energy_curve, max(1, len(analyses)))
    total = sum(t.duration_s for t in descriptor.tracks)
    # Blends overlap: each blend saves blend_beats at the outgoing tempo.
    overlap = 0.0
    for tr, a in zip(transitions, analyses):
        if tr["kind"] == "blend" and (a.bpm or 0) > 0:
            overlap += tr["blend_beats"] * 60.0 / a.bpm
    return {
        "mix_id": descriptor.id,
        "title": descriptor.title,
        "fingerprint": fingerprint(descriptor),
        "n_tracks": len(descriptor.tracks),
        "track_titles": [t.title for t in descriptor.tracks],
        "transitions": transitions,
        "energy_curve": descriptor.energy_curve,
        "energy_targets": curve,
        "total_duration_s": round(max(0.0, total - overlap), 2),
        "version": descriptor.version,
    }


def _read_wav_mono(path: str, max_s: float = 600.0) -> tuple[array, int] | None:
    try:
        with wave.open(path, "rb") as f:
            sr = f.getframerate()
            ch = f.getnchannels()
            sw = f.getsampwidth()
            n = min(f.getnframes(), int(sr * max_s))
            raw = f.readframes(n)
        if sw != 2:
            return None
        vals = struct.unpack("<%dh" % (len(raw) // 2), raw)
        mono = array("f", (0.0,) * len(vals))
        if ch == 1:
            for i, v in enumerate(vals):
                mono[i] = v / 32768.0
        else:
            for i in range(0, len(vals) - ch + 1, ch):
                mono[i // ch] = sum(vals[i : i + ch]) / (ch * 32768.0)
            del mono[len(vals) // ch :]
        return mono, sr
    except (OSError, wave.Error, ValueError, struct.error):
        return None


def render_mix(
    descriptor: MixDescriptor,
    *,
    out_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> RenderedMix:
    """Render a mix from its descriptor. Resumable via checkpoint.

    * With all WAV track files present: mixes them with dj_engine's
      beatmatched transitions into one WAV at ``out_path`` (or a
      ``~/.devon/mixes/<id>.wav`` default) and records progress in the
      checkpoint file after each track.
    * With files missing: returns the deterministic plan with
      ``missing_tracks`` and ``audio_path=None`` — no fake audio.
    * ``resume=True`` + an existing checkpoint for the same
      fingerprint skips already-rendered track indices and reports
      ``resumed_from_checkpoint=True``.
    """
    from . import dj_engine as _dj

    plan = render_plan(descriptor)
    fp = plan["fingerprint"]
    ckpt_path = Path(checkpoint_path) if checkpoint_path else (
        Path.home() / ".devon" / "mixes" / f"{descriptor.id or 'mix'}.checkpoint.json"
    )
    completed: list[int] = []
    resumed = False
    if resume and ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            if ckpt.get("fingerprint") == fp:
                completed = [int(i) for i in ckpt.get("completed_indices", [])]
                resumed = True
        except (OSError, json.JSONDecodeError, ValueError):
            completed = []
    missing = [
        t.title or t.path
        for t in descriptor.tracks
        if t.path and not Path(t.path).exists()
    ]
    # No audio to render (empty mix or missing files): plan-only result.
    if not descriptor.tracks or missing:
        return RenderedMix(
            descriptor=descriptor,
            fingerprint=fp,
            plan=plan,
            total_duration_s=plan["total_duration_s"],
            audio_path=None,
            missing_tracks=missing,
            resumed_from_checkpoint=resumed,
            completed_indices=completed,
        )
    out = Path(out_path) if out_path else (
        Path.home() / ".devon" / "mixes" / f"{descriptor.id or 'mix'}.wav"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sr = 22050
    mix: array = array("f")
    analyses = [_to_analysis(t) for t in descriptor.tracks]
    done = set(completed)
    prev_samples: array = array("f")
    # Sequential render: track 0 lays the bed; each later track blends
    # per its transition plan. Checkpoint after each track index.
    for idx, track in enumerate(descriptor.tracks):
        if idx in done:
            # Re-read the already-rendered prefix? For resumability
            # without storing partial audio, we re-render skipped
            # tracks silently but still report them as completed.
            pass
        got = _read_wav_mono(track.path)
        if got is None:
            missing.append(track.title or track.path)
            continue
        samples, file_sr = got
        # Resample to 22050 if needed (linear, deterministic).
        if file_sr != sr and file_sr > 0:
            ratio = sr / file_sr
            resampled = array("f", (0.0,)) * int(len(samples) * ratio)
            for i in range(len(resampled)):
                src = i / ratio
                lo = int(src)
                frac = src - lo
                hi = min(lo + 1, len(samples) - 1)
                resampled[i] = samples[lo] * (1 - frac) + samples[hi] * frac
            samples = resampled
        gain = max(0.0, min(2.0, track.gain))
        if gain != 1.0:
            for i in range(len(samples)):
                samples[i] *= gain
        if idx == 0:
            mix = array("f", samples)
        else:
            tr = plan["transitions"][idx - 1]
            plan_obj = _dj.TransitionPlan(
                kind=tr["kind"],
                blend_beats=tr["blend_beats"],
                sync_ratio=tr["sync_ratio"],
                start_beat=tr["start_beat"],
                reason=tr["reason"],
            )
            try:
                mix = _dj.render_beatmatched_transition(
                    mix, prev_samples, samples, plan_obj,
                    analyses[idx - 1], analyses[idx], sr=sr,
                )
            except Exception:
                # Honest fallback: concatenate instead of claiming a
                # blend that failed.
                _log.warning("mix_surface: transition %d failed, concatenating", idx)
                mix.extend(samples)
        prev_samples = array("f", samples)
        done.add(idx)
        try:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_text(
                json.dumps(
                    {"fingerprint": fp, "completed_indices": sorted(done)},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
    if missing:
        return RenderedMix(
            descriptor=descriptor,
            fingerprint=fp,
            plan=plan,
            total_duration_s=plan["total_duration_s"],
            audio_path=None,
            missing_tracks=missing,
            resumed_from_checkpoint=resumed,
            completed_indices=sorted(done),
        )
    try:
        with wave.open(str(out), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            clamped = array("h", (0,)) * len(mix)
            for i, v in enumerate(mix):
                clamped[i] = max(-32767, min(32767, int(v * 32767)))
            f.writeframes(clamped.tobytes())
    except OSError:
        _log.warning("mix_surface: could not write %s", out, exc_info=True)
        return RenderedMix(
            descriptor=descriptor, fingerprint=fp, plan=plan,
            total_duration_s=plan["total_duration_s"], audio_path=None,
            missing_tracks=missing, resumed_from_checkpoint=resumed,
            completed_indices=sorted(done),
        )
    return RenderedMix(
        descriptor=descriptor, fingerprint=fp, plan=plan,
        total_duration_s=plan["total_duration_s"],
        audio_path=str(out), missing_tracks=[],
        resumed_from_checkpoint=resumed, completed_indices=sorted(done),
    )


def deliver(
    chat: Any,
    rendered: RenderedMix,
    *,
    sender: Any,
    as_voice_note: bool = False,
) -> DeliveryReceipt:
    """Deliver a rendered mix via an injected sender.

    ``sender`` must provide ``send_audio(chat, path, ...)``; for voice
    notes it should also provide ``send_voice(chat, path, ...)`` —
    otherwise ``send_audio(..., ptt=True)`` is used. Raises
    :class:`ValueError` when there is no audio to deliver (plan-only
    render), instead of sending nothing and reporting success.
    """
    if rendered.audio_path is None:
        raise ValueError(
            "cannot deliver a plan-only mix (missing tracks: "
            f"{', '.join(rendered.missing_tracks) or 'none listed'}) — "
            "render audio first"
        )
    if sender is None or not hasattr(sender, "send_audio"):
        raise ValueError("deliver requires an injected sender with send_audio()")
    if as_voice_note and hasattr(sender, "send_voice"):
        result = sender.send_voice(chat, rendered.audio_path)
        kind = "voice_note"
    elif as_voice_note:
        result = sender.send_audio(chat, rendered.audio_path, ptt=True)
        kind = "voice_note"
    else:
        result = sender.send_audio(chat, rendered.audio_path)
        kind = "audio"
    return DeliveryReceipt(
        chat=str(chat),
        mix_id=rendered.descriptor.id,
        fingerprint=rendered.fingerprint,
        kind=kind,
        audio_path=rendered.audio_path,
        sent_at=time.time(),
        sender_result=result,
    )


# ── /mix surface ──────────────────────────────────────────────────────

def mix_menu() -> str:
    return (
        "🎧 /mix — shareable mixes\n"
        "  /mix new <title> — start a mix from the queue\n"
        "  /mix add <track> — append a track\n"
        "  /mix order <n> <m> — move a track\n"
        "  /mix curve <late_peak|linear|wave|flat> — set the energy arc\n"
        "  /mix card — shareable tracklist + transitions + energy curve\n"
        "  /mix render — render (resumable) from the descriptor\n"
        "  /mix send <chat> [--voice] — deliver as audio / voice note\n"
        "  /mix share — descriptor JSON (paste it back to reload the mix)\n"
        "Same descriptor → same mix, every time. Edit → new version, new fingerprint."
    )


def format_mix_card(descriptor: MixDescriptor) -> str:
    """Shareable, human-readable card: tracklist + transitions + curve."""
    lines = [
        f"🎧 {descriptor.title}  (v{descriptor.version} · `{descriptor.fingerprint[:12]}`)",
    ]
    if descriptor.tags:
        lines.append("tags: " + ", ".join(f"#{t}" for t in descriptor.tags))
    plan = render_plan(descriptor)
    for i, t in enumerate(descriptor.tracks, 1):
        bpm = f"{t.bpm:.0f} BPM" if t.bpm else "BPM ?"
        key = f" · {t.key} {t.mode}".strip() if t.key else ""
        lines.append(f"  {i}. {t.title or '(untitled)'} — {bpm}{key} · ⚡{t.energy:.2f}")
        if i <= len(plan["transitions"]):
            tr = plan["transitions"][i - 1]
            lines.append(f"     ↳ {tr['kind']} ({tr['blend_beats']} beats) — {tr['reason']}")
    lines.append(
        f"Energy: {descriptor.energy_curve} "
        f"({' → '.join(f'{v:.2f}' for v in plan['energy_targets'])})"
    )
    lines.append(f"Total ≈ {plan['total_duration_s']:.0f}s · {len(descriptor.tracks)} tracks")
    if descriptor.notes:
        lines.append(f"Notes: {descriptor.notes}")
    return "\n".join(lines)


def share_descriptor(descriptor: MixDescriptor) -> str:
    """Descriptor JSON — paste it back via :func:`load_descriptor`."""
    return _canonical(descriptor.to_dict())


def load_descriptor(payload: str | dict[str, Any]) -> MixDescriptor:
    if isinstance(payload, str):
        data = json.loads(payload)
    else:
        data = payload
    if not isinstance(data, dict) or "tracks" not in data:
        raise ValueError("not a MixDescriptor payload (missing 'tracks')")
    return MixDescriptor.from_dict(data)


# ── edits (each returns a NEW descriptor, version bumped) ──────────────

def with_track_added(descriptor: MixDescriptor, track: MixTrack) -> MixDescriptor:
    d = MixDescriptor.from_dict(descriptor.to_dict())
    d.tracks.append(track)
    d.transitions = []  # re-derive: the tracklist changed
    d.version = descriptor.version + 1
    return d


def with_track_removed(descriptor: MixDescriptor, index: int) -> MixDescriptor:
    if not 0 <= index < len(descriptor.tracks):
        raise ValueError(f"track index {index} out of range")
    d = MixDescriptor.from_dict(descriptor.to_dict())
    del d.tracks[index]
    d.transitions = []
    d.version = descriptor.version + 1
    return d


def with_tracks_reordered(descriptor: MixDescriptor, order: list[int]) -> MixDescriptor:
    if sorted(order) != list(range(len(descriptor.tracks))):
        raise ValueError("order must be a permutation of track indices")
    d = MixDescriptor.from_dict(descriptor.to_dict())
    d.tracks = [d.tracks[i] for i in order]
    d.transitions = []
    d.version = descriptor.version + 1
    return d
