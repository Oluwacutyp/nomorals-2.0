"""RenderHost — the one host-capability seam (Echelon Wave 1.2d).

FilmCraft's engine never touches the host directly: desktop and web
are the same engine over different ``Services``. Devon's render paths
grew the same need in pieces — ``render.py`` re-probed the profile for
encode knobs (``_PROFILE_PERF``), ``render_queue.py`` re-derived the
profile gate (``resolve_render_profile``), and nothing anywhere stated
what the machine actually offered (ffmpeg? hardware accel? VRAM?).

This module is the named seam those pieces compose into:

* :func:`probe_host` measures the machine once — ffmpeg presence and
  version, the hardware-accelerator list ffmpeg itself reports, VRAM
  when an NVIDIA GPU answers, and the runtime profile — and returns a
  :class:`HostCapabilities` snapshot. Probing never raises: an absent
  capability is data (with the reason), not an exception.
* :class:`LocalHost` is the :class:`RenderHost` over the local
  machine. Its :meth:`LocalHost.plan` is the single place a render
  request becomes a plan: profile gate (from ``render_queue``) +
  encode knobs (from ``render._PROFILE_PERF``) + the accel honesty
  note, in one dict that *says what was degraded and why*.

Honesty shape (EffectCraft's ``accel_note``): a substituted or unused
host capability reports itself. Today's engine encodes with libx264
software on every profile; when the host offers hardware acceleration
the note says it is available-but-unused rather than letting a plan
imply it was used.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...core.profile import detect_profile
from ...media_edit.videos import ffmpeg_path
from .failures import record_failure

__all__ = [
    "HostCapabilities",
    "RenderHost",
    "LocalHost",
    "probe_host",
]


@dataclass
class HostCapabilities:
    """What one host offers the renderer, measured — not assumed."""
    ffmpeg: bool = False
    ffmpeg_version: str = ""
    ffmpeg_error: str = ""
    hwaccels: list[str] = field(default_factory=list)
    vram_mb: int | None = None
    profile: str = "pc"
    accel_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ffmpeg": self.ffmpeg,
                "ffmpeg_version": self.ffmpeg_version,
                "ffmpeg_error": self.ffmpeg_error,
                "hwaccels": list(self.hwaccels),
                "vram_mb": self.vram_mb,
                "profile": self.profile,
                "accel_note": self.accel_note}


def _probe_hwaccels(ffmpeg: str) -> list[str]:
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=15)
    except Exception as exc:  # noqa: BLE001 — capability probe, not a render
        record_failure("host._probe_hwaccels", exc)
        return []
    out: list[str] = []
    for line in (proc.stdout or "").splitlines():
        name = line.strip()
        if name and not name.endswith(":") and name.lower() != "hardware acceleration methods:":
            out.append(name)
    return out


def _probe_vram_mb() -> int | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except Exception as exc:  # noqa: BLE001 — no NVIDIA tooling is normal
        record_failure("host._probe_vram_mb", exc)
        return None
    if proc.returncode != 0:
        return None
    try:
        return int((proc.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def probe_host(profile: str | None = None) -> HostCapabilities:
    """Measure this machine's render capabilities. Never raises."""
    caps = HostCapabilities()
    try:
        caps.profile = (profile or detect_profile().kind or "pc").strip().lower()
    except Exception as exc:  # noqa: BLE001 — profile is a hint, not a gate
        record_failure("host.probe_host:profile", exc)
        caps.profile = (profile or "pc").strip().lower()
    try:
        ffmpeg = str(ffmpeg_path())
        caps.ffmpeg = True
    except Exception as exc:  # noqa: BLE001 — absence is data here
        caps.ffmpeg_error = str(exc)
        caps.accel_note = ("ffmpeg not available on this host — "
                           "no render rung can run")
        return caps
    try:
        proc = subprocess.run([ffmpeg, "-hide_banner", "-version"],
                              capture_output=True, text=True, timeout=15)
        first = (proc.stdout or "").splitlines()
        caps.ffmpeg_version = first[0].strip() if first else ""
    except Exception as exc:  # noqa: BLE001
        record_failure("host.probe_host:version", exc)
    caps.hwaccels = _probe_hwaccels(ffmpeg)
    caps.vram_mb = _probe_vram_mb()
    if caps.hwaccels:
        caps.accel_note = (
            f"hardware acceleration reported by ffmpeg "
            f"({', '.join(caps.hwaccels)}) — available but unused: "
            f"this engine encodes with libx264 (software) on every "
            f"profile")
    else:
        caps.accel_note = ("no hardware acceleration reported by "
                           "ffmpeg — libx264 software encoding")
    return caps


@runtime_checkable
class RenderHost(Protocol):
    """The seam render paths query instead of re-probing the machine."""

    def capabilities(self) -> HostCapabilities:
        ...

    def plan(self, *, width: int, height: int, fps: int,
             profile: str | None = None) -> dict[str, Any]:
        ...


class LocalHost:
    """The :class:`RenderHost` over the local machine."""

    def __init__(self, profile: str | None = None, *,
                 capabilities: HostCapabilities | None = None):
        self._profile = profile
        self._caps = capabilities

    def capabilities(self) -> HostCapabilities:
        if self._caps is None:
            self._caps = probe_host(self._profile)
        return self._caps

    def plan(self, *, width: int, height: int, fps: int,
             profile: str | None = None) -> dict[str, Any]:
        """One render request → one plan, degradations stated.

        Composes the two existing single sources: the profile gate
        (:func:`.render_queue.resolve_render_profile` — height clamp
        with reason) and the encode knobs (``render._perf_for``).
        """
        from .render import _perf_for  # lazy: render imports this seam
        from .render_queue import resolve_render_profile
        caps = self.capabilities()
        prof = (profile or self._profile or caps.profile or "pc")
        gate = resolve_render_profile(profile=prof, height=int(height))
        perf = _perf_for(prof)
        reasons: list[str] = []
        if gate.get("reason"):
            reasons.append(str(gate["reason"]))
        if not caps.ffmpeg:
            reasons.append(caps.accel_note)
        return {
            "profile": prof,
            "width": int(width),
            "height": int(gate["height"]),
            "fps": int(fps),
            "degraded": bool(gate.get("degraded")) or not caps.ffmpeg,
            "reasons": reasons,
            "preset": perf["preset"], "crf": perf["crf"],
            "threads": perf["threads"],
            "accel_note": caps.accel_note,
            "ffmpeg": caps.ffmpeg,
            "hwaccels": list(caps.hwaccels),
            "vram_mb": caps.vram_mb,
        }
