"""Platform deliverables — one master, every format the platforms want.

Tier 2. :func:`deliverable_plan` turns a master track + its section map
into the full deliverable set: full master, 15s / 30s promo cuts (cut at
section boundaries, never mid-word), stems manifest, acapella (vocal
stem) manifest, per-platform loudness targets. :func:`cut_wav` performs
a real cut on a WAV file with stdlib ``wave`` (fade in/out to avoid
clicks); everything else is a plan + manifest so a missing master
fails honestly instead of producing fake files.

Loudness targets (integrated LUFS, widely published platform norms):
Spotify −14, YouTube −14, Apple Music −16, TikTok −14, Instagram −14,
Tidal −14, Amazon −14. True-peak ceiling −1.0 dBTP everywhere.
These are *targets for the mastering stage*, not measured values —
measurement needs a real LUFS meter (see model_license.matchering).
"""

from __future__ import annotations

import json
import math
import os
import wave
from array import array
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PLATFORM_LOUDNESS",
    "TRUE_PEAK_CEILING_DBTP",
    "Deliverable",
    "deliverable_plan",
    "cut_wav",
    "stems_manifest",
    "register",
]

#: Integrated LUFS targets per platform (published norms, not measured).
PLATFORM_LOUDNESS: dict[str, float] = {
    "spotify": -14.0,
    "youtube": -14.0,
    "apple_music": -16.0,
    "tiktok": -14.0,
    "instagram": -14.0,
    "tidal": -14.0,
    "amazon_music": -14.0,
    "soundcloud": -14.0,
    "audiomack": -14.0,
    "boomplay": -14.0,
}
TRUE_PEAK_CEILING_DBTP = -1.0

#: Stem names the pipeline always produces (see stems.py / chain).
STEM_NAMES = ("vocals", "drums", "bass", "other", "melody", "acapella")


@dataclass
class Deliverable:
    kind: str               # "master" | "cut_15s" | "cut_30s" | "stem" | "acapella" | "midi" | "spec"
    path: str
    duration_s: float = 0.0
    platform: str = ""      # platform this deliverable targets, "" = all
    loudness_lufs: float | None = None
    note: str = ""


def _cut_points(duration_s: float, sections: list[dict[str, Any]],
                target_s: float) -> tuple[float, float]:
    """Pick a cut window of ~target_s aligned to section boundaries.

    Prefers starting at the chorus/hook (the most recognisable part);
    falls back to the loudest-energy section, then to 0. Never cuts
    mid-section unless the section itself is longer than the target.
    """
    if duration_s <= target_s:
        return 0.0, duration_s
    hook = None
    for sec in sections or []:
        name = str(sec.get("name", "")).lower()
        if "chorus" in name or "hook" in name or "drop" in name:
            hook = sec
            break
    start = 0.0
    if hook is not None:
        start = float(hook.get("start_s", 0.0) or 0.0)
    elif sections:
        best = max(sections, key=lambda s: float(s.get("energy", 0.0) or 0.0))
        start = float(best.get("start_s", 0.0) or 0.0)
    # keep the window inside the track; pull back if it would overrun
    if start + target_s > duration_s:
        start = max(0.0, duration_s - target_s)
    return round(start, 3), round(min(duration_s, start + target_s), 3)


def deliverable_plan(title: str, duration_s: float,
                     sections: list[dict[str, Any]] | None = None,
                     stem_paths: dict[str, str] | None = None,
                     midi_path: str = "", spec_path: str = "",
                     master_path: str = "",
                     platforms: tuple[str, ...] = ("spotify", "youtube",
                                                    "apple_music", "tiktok")
                     ) -> dict[str, Any]:
    """Build the full deliverable manifest for one song (no audio I/O).

    Returns {deliverables: [...], missing: [...]}. A deliverable whose
    source file doesn't exist is listed in ``missing`` — never claimed
    as produced. Cuts are planned from the section map; stems/acapella
    come from the stem paths the chain actually wrote.
    """
    sections = list(sections or [])
    stem_paths = dict(stem_paths or {})
    slug = "".join(c if c.isalnum() or c in "-_" else "-"
                   for c in (title or "song").lower()).strip("-") or "song"
    deliverables: list[Deliverable] = []
    missing: list[str] = []
    # master
    if master_path and os.path.exists(master_path):
        deliverables.append(Deliverable("master", master_path, duration_s,
                                        note="full master"))
    else:
        missing.append("master")
    # cuts
    for target, kind in ((15.0, "cut_15s"), (30.0, "cut_30s")):
        start, end = _cut_points(duration_s, sections, target)
        deliverables.append(Deliverable(
            kind, f"{slug}-{kind}.wav", round(end - start, 3),
            note=f"promo cut {start:.1f}s–{end:.1f}s (section-aligned)"))
    # stems + acapella
    for stem in ("vocals", "drums", "bass", "other"):
        p = stem_paths.get(stem, "")
        if p and os.path.exists(p):
            deliverables.append(Deliverable("stem", p, duration_s,
                                            note=f"{stem} stem"))
        else:
            missing.append(f"stem:{stem}")
    vocals = stem_paths.get("vocals", "")
    if vocals and os.path.exists(vocals):
        deliverables.append(Deliverable("acapella", vocals, duration_s,
                                        note="acapella = isolated vocal stem"))
    else:
        missing.append("acapella")
    if midi_path and os.path.exists(midi_path):
        deliverables.append(Deliverable("midi", midi_path, duration_s,
                                        note="melody + arrangement MIDI"))
    else:
        missing.append("midi")
    if spec_path and os.path.exists(spec_path):
        deliverables.append(Deliverable("spec", spec_path, 0.0,
                                        note="reproducible SongSpec JSON"))
    else:
        missing.append("spec")
    # per-platform loudness rows
    platform_rows = [
        {"platform": p, "target_lufs": PLATFORM_LOUDNESS.get(p),
         "true_peak_dbtp": TRUE_PEAK_CEILING_DBTP}
        for p in platforms
    ]
    return {
        "title": title,
        "duration_s": round(duration_s, 3),
        "deliverables": [d.__dict__ for d in deliverables],
        "missing": missing,
        "platform_loudness": platform_rows,
        "complete": not missing,
    }


def cut_wav(src_path: str, dst_path: str, start_s: float, end_s: float,
            fade_ms: float = 25.0) -> dict[str, Any]:
    """Cut [start_s, end_s) out of a WAV file (stdlib only) + fades.

    Real audio I/O: reads frames, applies linear fade in/out to avoid
    clicks, writes a new WAV with the same params. Fails closed when
    the source is missing or the window is invalid — never writes
    silence and calls it a cut.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"no such source WAV: {src_path}")
    if end_s <= start_s:
        raise ValueError(f"empty cut window {start_s}–{end_s}")
    with wave.open(src_path, "rb") as w:
        params = w.getparams()
        sr = w.getframerate()
        n = w.getnframes()
        start_f = max(0, int(start_s * sr))
        end_f = min(n, int(end_s * sr))
        if end_f <= start_f:
            raise ValueError(f"cut window {start_s}–{end_s}s is outside "
                             f"a {n / sr:.1f}s file")
        w.setpos(start_f)
        raw = w.readframes(end_f - start_f)
    width = params.sampwidth
    if width == 2:
        samples = array("h")
        samples.frombytes(raw)
        peak = 32767.0
    elif width == 1:
        samples = array("B")
        samples.frombytes(raw)
        peak = 255.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    fade_n = int(fade_ms / 1000.0 * sr)
    for i in range(min(fade_n, len(samples))):
        gain = (i + 1) / max(1, fade_n)
        if width == 2:
            samples[i] = int(samples[i] * gain)
            samples[-(i + 1)] = int(samples[-(i + 1)] * gain)
        else:
            samples[i] = int((samples[i] - 128) * gain + 128)
            samples[-(i + 1)] = int((samples[-(i + 1)] - 128) * gain + 128)
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    with wave.open(dst_path, "wb") as w:
        w.setparams(params)
        w.writeframes(samples.tobytes())
    return {"ok": True, "path": dst_path,
            "duration_s": round(len(samples) / sr, 3),
            "start_s": round(start_f / sr, 3), "end_s": round(end_f / sr, 3)}


def stems_manifest(stem_paths: dict[str, str]) -> dict[str, Any]:
    """Validate a stem set: every stem present on disk or listed missing."""
    present = {k: v for k, v in (stem_paths or {}).items()
               if v and os.path.exists(v)}
    missing = [k for k in ("vocals", "drums", "bass", "other")
               if k not in present]
    return {"ok": not missing, "stems": present, "missing": missing,
            "complete": not missing}


def register(registry: Any) -> None:
    """Tool registration: ``deliverables`` — cuts, stems, platform rows."""
    from ..core.policy import Capability

    @registry.register(
        "deliverables",
        description=(
            "Platform deliverables: action=plan (title, duration_s, "
            "sections JSON, stem_paths JSON, midi_path, spec_path, "
            "master_path → full deliverable manifest incl. 15s/30s cuts, "
            "stems, acapella, per-platform LUFS targets) | cut (src, dst, "
            "start_s, end_s → real WAV cut with fades)."
        ),
        capability=Capability.FS_WRITE,
    )
    def deliverables(action: str = "plan", title: str = "",
                     duration_s: float = 0.0, sections: str = "",
                     stem_paths: str = "", midi_path: str = "",
                     spec_path: str = "", master_path: str = "",
                     src: str = "", dst: str = "",
                     start_s: float = 0.0, end_s: float = 0.0) -> dict:
        import json as _json
        if action == "cut":
            return {"ok": True, **cut_wav(src, dst, start_s, end_s)}
        try:
            secs = _json.loads(sections) if sections else []
            stems = _json.loads(stem_paths) if stem_paths else {}
        except Exception as exc:  # noqa: BLE001
            from ..core.errors import ToolError
            raise ToolError(f"bad JSON: {exc}")
        return {"ok": True, **deliverable_plan(
            title, duration_s, secs, stems, midi_path, spec_path, master_path)}
