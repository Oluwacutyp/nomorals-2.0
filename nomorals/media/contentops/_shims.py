"""SHIM / FALLBACK implementations — clearly marked, temporary.

Two sibling workstreams are building the real modules in parallel:

* ``nomorals/media/contentops/edit.py`` — must expose ``EditSpec``,
  ``render``, ``detect_beats`` (plus caption/audio helpers).
* ``nomorals/media/contentops/niches/`` — exposes ``get_niche(name)``
  returning a ``NichePlugin`` (see ``niches/base.py``): ``.script_prompt(topic)``,
  ``.visual_strategy(script) -> VisualPlan`` (``.render_prompts()``,
  ``.scenes`` with ``.imggen_prompt``/``.motion``/``.duration_s``),
  ``.voice_spec`` (``.resolve(catalogue)`` / ``.candidates()``),
  ``.title_for(topic)``, ``.description_for(topic, script)``,
  ``.tags_for(platform)``, ``.cadence`` (float) / ``.cadence_spec()``,
  ``.generate_script(brain, topic)``.  The registry falls back to this
  file's ``DefaultNiche`` for unregistered names, and the pipeline
  accepts both the full plugin surface and the older duck-typed shape.

``pipeline.py`` tries the real modules first and only uses this file when
they are not importable yet.  Everything here is REAL, working code
(ffmpeg/numpy based) — not a stub — so the pipeline is end-to-end
functional today, and the sibling modules can replace these names
drop-in when they land.

Contract notes for the sibling streams (what ``pipeline.py`` expects):

* ``EditSpec`` — constructible from keyword arguments.  ``pipeline`` passes
  ``scenes`` (list of ``SceneClip``-shaped mappings or objects with
  ``image``/``duration``/``effect`` attributes), ``audio`` (voiceover path),
  ``captions`` (srt path or ""), ``music`` (bed path or ""),
  ``beat_times`` (list[float]), ``output`` (final path), ``width``,
  ``height``, ``fps``.  Defensive construction tries ``EditSpec(**payload)``,
  then ``EditSpec(payload)``; a sibling ``EditSpec`` that accepts at least
  ``scenes``/``audio``/``output`` as kwargs integrates cleanly.
* ``render(spec)`` — renders the spec to the output path carried by the
  spec (``spec.output``) and returns that path as a string.
* ``detect_beats(audio_path)`` — returns a sorted ``list[float]`` of beat/
  onset times in seconds.
* ``get_niche(name)`` — returns a plugin object.  ``pipeline`` calls
  ``plugin.script_prompt(topic) -> str`` and ``plugin.visual_strategy(script)``
  (accepts a list of per-scene visual prompts, a dict with a
  ``"prompts"`` key, or a single string — all normalised).  Attributes read:
  ``voice_style`` (str), ``title_template`` (str), ``description_template``
  (str), ``hashtags`` (list[str]), ``cadence`` (mapping, honours
  ``"posts_per_day"``).
"""

from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from typing import Any

from ...core.logging_setup import get_logger

_log = get_logger(__name__)

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


# ── edit contract ──────────────────────────────────────────────────────


@dataclass
class SceneClip:
    """One still-image segment of the short."""

    image: str
    duration: float
    effect: str = "kenburns"  # "kenburns" | "static"
    zoom_direction: str = "in"  # "in" | "out"


@dataclass
class EditSpec:
    """Beat-synced stills edit.  ``render(spec)`` honours these fields."""

    scenes: list = field(default_factory=list)
    audio: str = ""        # voiceover wav path
    captions: str = ""     # srt path ("" = no burn-in)
    music: str = ""        # music-bed path mixed under voiceover ("" = skip)
    beat_times: list = field(default_factory=list)
    output: str = ""
    width: int = 1080
    height: int = 1920
    fps: int = 30


def _run(cmd: list[str], timeout: int = 600) -> None:
    try:
        subprocess.run(cmd, check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"")[-600:].decode("utf-8", "replace")
        raise RuntimeError(f"ffmpeg failed ({' '.join(cmd[:4])}…): {tail}")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed — the video stages need it "
            "(apt install ffmpeg / pkg install ffmpeg)")


def detect_beats(audio_path: str) -> list[float]:
    """Onset/beat times (seconds) via an energy-envelope peak picker.

    Decodes to mono 22050 Hz, takes RMS energy in 46 ms windows, and picks
    local maxima that stand clearly above their neighbourhood.  Honest
    limits: this finds percussive onsets, not musical downbeats — fine for
    cutting stills on energy, not for bar-accurate choreography.
    """
    if not os.path.exists(audio_path):
        return []
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        _run([_FFMPEG, "-y", "-v", "error", "-i", audio_path,
              "-ac", "1", "-ar", "22050", "-f", "wav", tmp.name])
        with wave.open(tmp.name, "rb") as w:
            n = w.getnframes()
            raw = w.readframes(n)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if not raw:
        return []
    import array
    samples = array.array("h", raw)
    sr = 22050
    win = int(sr * 0.046)  # ~46 ms
    energies: list[float] = []
    for i in range(0, len(samples), win):
        chunk = samples[i:i + win]
        if not chunk:
            break
        energies.append(math.sqrt(sum(s * s for s in chunk) / len(chunk)))
    if not energies:
        return []
    mean_e = sum(energies) / len(energies)
    beats: list[float] = []
    for i in range(2, len(energies) - 2):
        e = energies[i]
        if (e > 1.6 * mean_e
                and e >= energies[i - 1] and e >= energies[i + 1]
                and e > energies[i - 2] and e > energies[i + 2]
                and (not beats or (i * win / sr) - beats[-1] > 0.25)):
            beats.append(round(i * win / sr, 3))
    _log.info("shim detect_beats: %d onsets in %s", len(beats), audio_path)
    return beats


def _scene_segment(clip: SceneClip, width: int, height: int, fps: int,
                   tmpdir: str, idx: int) -> str:
    """Render one still → short mp4 segment with an optional ken-burns."""
    out = os.path.join(tmpdir, f"seg_{idx:03d}.mp4")
    frames = max(1, int(clip.duration * fps))
    if clip.effect == "kenburns":
        zin = clip.zoom_direction != "out"
        # zoompan: 1.0 → 1.12 over the segment (or reverse)
        z0, z1 = (1.0, 1.12) if zin else (1.12, 1.0)
        filt = (
            f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,"
            f"crop={width * 2}:{height * 2},"
            f"zoompan=z='min({z0}+({z1}-{z0})*on/{frames},{z1})'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={width}x{height}:fps={fps}"
        )
    else:
        filt = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height}")
    _run([_FFMPEG, "-y", "-v", "error", "-loop", "1", "-i", clip.image,
          "-vf", filt, "-t", f"{clip.duration:.3f}", "-r", str(fps),
          "-c:v", "libx264", "-pix_fmt", "yuv420p", out])
    return out


def render(spec: "EditSpec") -> str:
    """Render an EditSpec → MP4.  Returns the output path.

    Stills (ken-burns zoompan) → concat → voiceover audio → optional music
    bed mixed underneath → optional SRT burn-in.  Raises RuntimeError with
    the ffmpeg tail on failure.
    """
    if not spec.scenes:
        raise RuntimeError("render: spec has no scenes")
    if not spec.output:
        raise RuntimeError("render: spec.output is empty")
    os.makedirs(os.path.dirname(os.path.abspath(spec.output)), exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="shorts_edit_")
    try:
        segs = [_scene_segment(_as_clip(c), spec.width, spec.height,
                               spec.fps, tmpdir, i)
                for i, c in enumerate(spec.scenes)]
        lst = os.path.join(tmpdir, "concat.txt")
        with open(lst, "w") as f:
            for s in segs:
                f.write(f"file '{s}'\n")
        video = os.path.join(tmpdir, "video.mp4")
        _run([_FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
              "-i", lst, "-c", "copy", video])

        # audio: voiceover + optional bed mixed underneath
        inputs = ["-i", video]
        filters: list[str] = []
        audio_map = "1:a"
        if spec.audio and os.path.exists(spec.audio):
            inputs += ["-i", spec.audio]
            if spec.music and os.path.exists(spec.music):
                inputs += ["-i", spec.music]
                filters = ["[2:a]volume=0.16,apad[bed];"
                           "[1:a][bed]amix=inputs=2:duration=first:dropout_transition=0[a]"]
                audio_map = "[a]"
            else:
                audio_map = "1:a"
            amap = ["-map", "0:v", "-map", audio_map]
        else:
            amap = ["-map", "0:v"]
        mixed = os.path.join(tmpdir, "mixed.mp4")
        cmd = [_FFMPEG, "-y", "-v", "error"] + inputs
        if filters:
            cmd += ["-filter_complex", "".join(filters)]
        cmd += amap + ["-c:v", "copy", "-c:a", "aac", "-shortest", mixed]
        _run(cmd)

        # captions burn-in
        final = spec.output
        if spec.captions and os.path.exists(spec.captions):
            srt = spec.captions.replace("'", r"'\''")
            _run([_FFMPEG, "-y", "-v", "error", "-i", mixed,
                  "-vf", f"subtitles='{srt}':force_style="
                         "'FontSize=22,PrimaryColour=&HFFFFFF&,OutlineColour=&H80000000&,"
                         "BorderStyle=1,Outline=2,MarginV=120,Alignment=2'",
                  "-c:a", "copy", final])
        else:
            shutil.copyfile(mixed, final)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    _log.info("shim render: wrote %s", spec.output)
    return spec.output


def _as_clip(c: Any) -> SceneClip:
    if isinstance(c, SceneClip):
        return c
    if isinstance(c, dict):
        return SceneClip(image=c.get("image", ""),
                         duration=float(c.get("duration", 2.0)),
                         effect=c.get("effect", "kenburns"),
                         zoom_direction=c.get("zoom_direction", "in"))
    return SceneClip(image=str(getattr(c, "image", "")),
                     duration=float(getattr(c, "duration", 2.0)),
                     effect=str(getattr(c, "effect", "kenburns")),
                     zoom_direction=str(getattr(c, "zoom_direction", "in")))


def build_captions(phrases: list[tuple[str, float, float]],
                   out_path: str) -> str:
    """Write an SRT from ``[(text, start_s, end_s), …]``.  Returns path."""
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines: list[str] = []
    for i, (text, a, b) in enumerate(phrases, 1):
        lines += [str(i), f"{ts(a)} --> {ts(b)}", text.strip(), ""]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def make_music_bed(duration_s: float, out_path: str, seed: int = 7) -> str:
    """A deterministic ambient pad bed (numpy → wav).  Returns path.

    Slow Am–F–C–G pad with a breathing lowpass-ish sweep, faded in/out.
    It is a *bed* — quiet by design (−18 dB) — meant to sit under the
    voiceover, not to be listened to alone.
    """
    import numpy as np
    rng = random.Random(seed)
    sr = 22050
    n = max(1, int(duration_s * sr))
    t = np.arange(n) / sr
    # Am F C G roots, 6 s each
    roots = [110.0, 87.31, 130.81, 98.0]
    chord = np.zeros(n)
    seg = 6.0
    for i in range(int(math.ceil(duration_s / seg))):
        f0 = roots[i % len(roots)]
        a, b = int(i * seg * sr), min(n, int((i + 1) * seg * sr))
        tt = t[a:b] - t[a]
        env = np.minimum(1.0, tt / 1.5) * np.minimum(1.0, (b - a) / sr - tt) / 1.0
        env = np.clip(env, 0, 1)
        tone = (np.sin(2 * np.pi * f0 * tt)
                + 0.5 * np.sin(2 * np.pi * f0 * 1.5 * tt)
                + 0.33 * np.sin(2 * np.pi * f0 * 2.0 * tt + 0.7))
        chord[a:b] += tone * env
    # breathing one-pole "lowpass" sweep
    alpha = 0.04 + 0.03 * np.sin(2 * np.pi * t / 12.0)
    y = np.zeros(n)
    acc = 0.0
    for i in range(n):
        acc += alpha[i] * (chord[i] - acc)
        y[i] = acc
    # fades + level
    fade = int(min(2.0, duration_s / 4) * sr)
    ramp = np.ones(n)
    ramp[:fade] = np.linspace(0, 1, fade)
    ramp[-fade:] = np.linspace(1, 0, fade)
    y = y * ramp * 0.14
    peak = np.max(np.abs(y)) or 1.0
    y = (y / peak * 0.5 * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())
    _log.info("shim music bed: %.1fs → %s", duration_s, out_path)
    return out_path


# ── niches contract ──────────────────────────────────────────────────


class _ShimVoiceSpec:
    """Minimal stand-in for niches.base.VoiceSpec."""

    preferred_voice = "narrator"
    fallback_voice = "default"
    pace_wpm = 150

    def candidates(self) -> list:
        return [c for c in (self.preferred_voice, self.fallback_voice) if c]

    def resolve(self, catalogue: Any) -> str:
        try:
            names = set(getattr(catalogue, "voices", {}) or {})
        except Exception:
            names = set()
        for c in self.candidates():
            if c in names:
                return c
        return ""

    def __str__(self) -> str:
        return self.preferred_voice or "narrator"


class DefaultNiche:
    """Generic fallback niche plugin (real, usable — not a stub).

    Mirrors the ``niches.base.NichePlugin`` surface (``voice_spec``,
    ``title_for``/``description_for``/``tags_for``, ``cadence_spec``) so
    the pipeline treats it exactly like a curated sibling plugin.  The
    sibling ``niches/`` registry itself falls back to this class for
    unregistered niche names.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def voice_spec(self) -> _ShimVoiceSpec:
        return _ShimVoiceSpec()

    @property
    def voice_style(self) -> str:  # legacy alias
        return "narrator"

    def title_for(self, topic: str) -> str:
        return self.title_template.format(topic=topic,
                                          niche=self.name).strip()

    def description_for(self, topic: str, script: str = "") -> str:
        return self.description_template.format(
            topic=topic, niche=self.name, hook=script.split("\n")[0],
            script=script,
            hashtags=" ".join(self.hashtags)).strip()

    def tags_for(self, platform: str = "tiktok") -> list[str]:
        return list(self.hashtags)

    def cadence_spec(self) -> dict:
        return dict(self.cadence)

    @property
    def title_template(self) -> str:
        return "{topic} — {niche} in 60 seconds"

    @property
    def description_template(self) -> str:
        return ("{hook}\n\nFull breakdown in 60 seconds. "
                "Follow for daily {niche} shorts.\n\n{hashtags}")

    @property
    def hashtags(self) -> list[str]:
        base = self.name.replace(" ", "").replace("_", "")
        return [f"#{base}", "#shorts", "#learnontiktok", "#fyp"]

    @property
    def cadence(self) -> dict:
        return {"posts_per_day": 1}

    def script_prompt(self, topic: str) -> str:
        return (
            f"You are writing a 60-second vertical short-form video script "
            f"for the niche '{self.name}' about: {topic}\n\n"
            "Rules: punchy hook in the first 3 seconds, one idea per scene, "
            "~120-150 spoken words total, plain conversational English.\n\n"
            "Follow this format EXACTLY:\n"
            "HOOK: <one punchy opening line>\n"
            "[SCENE 1]\n"
            "SAY: <narration, ~20 words>\n"
            "SHOW: <visual description for AI image generation, vivid, vertical composition>\n"
            "[SCENE 2]\n"
            "SAY: ...\n"
            "SHOW: ...\n"
            "(4 to 6 scenes)\n"
            "TITLE: <video title under 60 chars>\n"
            "DESCRIPTION: <two sentences>\n"
        )

    def visual_strategy(self, script: str) -> list[str]:
        """Extract SHOW: lines; fall back to SAY: lines."""
        shows = _extract_field(script, "SHOW")
        if shows:
            return [s + ", vertical 9:16 composition, cinematic lighting"
                    for s in shows]
        says = _extract_field(script, "SAY") or [script[:200]]
        return [s + ", vertical 9:16 composition, cinematic lighting"
                for s in says]


def _extract_field(text: str, field: str) -> list[str]:
    import re
    return [m.strip() for m in
            re.findall(rf"^{field}:\s*(.+)$", text, re.M) if m.strip()]


def get_niche(name: str) -> DefaultNiche:
    """SHIM ``get_niche`` — returns the generic plugin for any name."""
    _log.warning("shim get_niche(%r): sibling niches/ not present; "
                 "using generic fallback plugin", name)
    return DefaultNiche(name or "general")
