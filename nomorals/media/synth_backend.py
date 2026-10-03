"""Profile-aware MIDI→audio rendering.

The old behavior rendered every composition with the pure-Python synth in
:mod:`nomorals.media.synth` — fine on a phone, but needlessly lo-fi on a
workstation.  This module picks the best backend *automatically* from the
detected environment profile (``nomorals.core.profile``):

* **termux / mobile / embedded** → builtin pure-Python synth (stdlib only,
  light on CPU and RAM — the phone never pays for samples it can't hold).
* **pc / vps / workstation** → FluidSynth with a real GM soundfont when both
  are present; otherwise the builtin synth, plus a one-line note telling
  the owner how to unlock studio quality.

Backend chain, in order:

1. ``NM_SYNTH_BACKEND`` env override (``fluidsynth`` | ``builtin``) — power
   users only; ``fluidsynth`` forced this way fails honestly instead of
   silently downgrading.
2. Automatic: profile + availability probing.

A missing soundfont is never a fatal error: rendering falls back to the
builtin synth and the caller gets a human-readable note with the exact
install command (``nm music soundfont install`` — explicit consent, with
the ~31 MB size stated up front).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..core.logging_setup import get_logger
from ..core.profile import detect_profile

_log = get_logger(__name__)

__all__ = [
    "SynthChoice",
    "choose_synth",
    "render_wav",
    "find_soundfont",
    "soundfont_offer",
    "install_soundfont",
    "SOUNDFONT",
    "LIGHT_PROFILES",
]

#: profiles that always get the lightweight builtin synth — RAM and CPU
#: are precious here, and a 31 MB soundfont is not worth it.
LIGHT_PROFILES = frozenset({"termux", "mobile", "embedded"})

#: The recommended free GM soundfont.  Pinned to an immutable commit of the
#: author's own repository; SHA-256 verified after download.
#: GeneralUser GS v2.0.3, S. Christian Collins — free for any use,
#: redistribution of the unmodified file permitted.
SOUNDFONT = {
    "name": "GeneralUser GS v2.0.3",
    "file": "GeneralUser-GS.sf2",
    "size_mb": 30.8,
    "url": ("https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/"
            "684543d5e5efaef08d02be50dcda8d552478fa60/GeneralUser-GS.sf2"),
    "sha256": ("9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a51"
               "52688cfe"),
    "license": "GeneralUser GS License v2.0 (free, keep unmodified)",
}


@dataclass
class SynthChoice:
    """Which backend was chosen, and why."""

    name: str = "builtin"          # "fluidsynth" | "builtin"
    profile: str = ""             # detected profile kind
    soundfont: str = ""           # soundfont path when fluidsynth
    reason: str = ""              # human-readable why
    note: str = ""                # follow-up for the owner (e.g. soundfont offer)

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "profile": self.profile,
                "soundfont": self.soundfont, "reason": self.reason,
                "note": self.note}


def _profile_kind(context: Any = None) -> str:
    """Environment profile kind: explicit config first, detection second."""
    # 1. runtime settings pin (mirrors agent_loop's resource_profile)
    try:
        settings = getattr(context, "settings", None)
        prof = (getattr(settings, "profile", "") or "").strip().lower()
        if prof:
            return prof
    except Exception:  # noqa: BLE001 - detection is best-effort
        pass
    # 2. termux heuristic (same well-known prefix the agent loop uses)
    try:
        if os.environ.get("PREFIX", "").startswith("/data/data/com.termux"):
            return "termux"
    except Exception:  # noqa: BLE001
        pass
    # 3. full detection
    try:
        return detect_profile().kind
    except Exception:  # noqa: BLE001
        return "pc"


def _fluidsynth_binary() -> str:
    return shutil.which("fluidsynth") or ""


def soundfont_dirs() -> list[Path]:
    """Where a soundfont may already live, in search order."""
    dirs: list[Path] = []
    env = (os.environ.get("NM_SOUNDFONT") or "").strip()
    if env:
        p = Path(env).expanduser()
        dirs.append(p if p.is_dir() else p.parent)
    home = Path.home()
    dirs.append(home / ".nomorals" / "soundfonts")
    dirs.append(home / ".local" / "share" / "soundfonts")
    for sys_dir in ("/usr/share/sounds/sf2", "/usr/share/soundfonts",
                    "/usr/local/share/sounds/sf2"):
        dirs.append(Path(sys_dir))
    return dirs


def find_soundfont() -> str:
    """Path of a usable .sf2/.sf3, or \"\" when none is installed."""
    env = (os.environ.get("NM_SOUNDFONT") or "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return str(p)
    for d in soundfont_dirs():
        try:
            if not d.is_dir():
                continue
            for cand in sorted(d.iterdir()):
                if cand.suffix.lower() in (".sf2", ".sf3") and cand.is_file():
                    return str(cand)
        except OSError:  # noqa: BLE001 - unreadable dir, keep looking
            continue
    return ""


def soundfont_offer() -> dict[str, Any]:
    """The consent payload: what we'd download, how big, from where."""
    return {
        "name": SOUNDFONT["name"],
        "file": SOUNDFONT["file"],
        "size_mb": SOUNDFONT["size_mb"],
        "url": SOUNDFONT["url"],
        "sha256": SOUNDFONT["sha256"],
        "license": SOUNDFONT["license"],
        "install_command": "nm music soundfont install",
        "message": (
            f"studio-quality audio wants the {SOUNDFONT['name']} soundfont "
            f"(~{SOUNDFONT['size_mb']:.0f} MB, {SOUNDFONT['license']}). "
            f"Install it with: nm music soundfont install"
        ),
    }


def install_soundfont(*, dest_dir: str = "",
                      progress: Any = None) -> str:
    """Download + verify the recommended soundfont (explicit consent: the
    caller invoked this, so the size warning was already shown).

    Returns the installed path.  Raises on download/verification failure.
    """
    from urllib.request import urlretrieve

    target_dir = Path(dest_dir).expanduser() if dest_dir else (
        Path.home() / ".nomorals" / "soundfonts")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / SOUNDFONT["file"]
    if target.is_file():
        if _sha256_file(target) == SOUNDFONT["sha256"]:
            return str(target)
        _log.warning("existing soundfont failed checksum — re-downloading")
        target.unlink()

    def _hook(blocks: int, block_size: int, total: int) -> None:
        if progress is not None and total > 0:
            done = min(total, blocks * block_size)
            progress(f"{done / 1048576:.1f} / {total / 1048576:.1f} MB")

    tmp = target.with_suffix(".sf2.download")
    try:
        urlretrieve(SOUNDFONT["url"], str(tmp), reporthook=_hook)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"soundfont download failed: {exc}") from exc
    digest = _sha256_file(tmp)
    if digest != SOUNDFONT["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "soundfont checksum mismatch — refusing to install "
            f"(got {digest[:16]}…, expected {SOUNDFONT['sha256'][:16]}…)")
    tmp.replace(target)
    return str(target)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_synth(context: Any = None) -> SynthChoice:
    """Pick the synth backend.  Automatic; no config the owner must set."""
    forced = (os.environ.get("NM_SYNTH_BACKEND") or "").strip().lower()
    profile = _profile_kind(context)

    if forced == "builtin":
        return SynthChoice(name="builtin", profile=profile,
                           reason="NM_SYNTH_BACKEND=builtin (forced)")
    if forced == "fluidsynth":
        binary = _fluidsynth_binary()
        sf = find_soundfont()
        if not binary:
            raise RuntimeError(
                "NM_SYNTH_BACKEND=fluidsynth but no fluidsynth binary found "
                "(apt: fluidsynth / pkg: fluidsynth)")
        if not sf:
            raise RuntimeError(
                "NM_SYNTH_BACKEND=fluidsynth but no soundfont installed — "
                f"run: {soundfont_offer()['install_command']}")
        return SynthChoice(name="fluidsynth", profile=profile, soundfont=sf,
                           reason="NM_SYNTH_BACKEND=fluidsynth (forced)")

    # automatic path
    if profile in LIGHT_PROFILES:
        return SynthChoice(
            name="builtin", profile=profile,
            reason=(f"profile={profile}: lightweight builtin synth — "
                    "samples stay off the phone"))
    binary = _fluidsynth_binary()
    if not binary:
        return SynthChoice(
            name="builtin", profile=profile,
            reason="fluidsynth not installed — builtin synth "
                   "(apt install fluidsynth for studio quality)")
    sf = find_soundfont()
    if not sf:
        offer = soundfont_offer()
        return SynthChoice(
            name="builtin", profile=profile,
            reason="fluidsynth present but no soundfont installed",
            note=offer["message"])
    return SynthChoice(name="fluidsynth", profile=profile, soundfont=sf,
                       reason=f"fluidsynth + {Path(sf).name}")


def _render_fluidsynth(midi_path: str, out_path: str, soundfont: str,
                       sample_rate: int = 44100,
                       timeout: float = 180.0) -> None:
    """Render a MIDI file to WAV through the fluidsynth CLI."""
    binary = _fluidsynth_binary()
    if not binary:
        raise RuntimeError("fluidsynth binary not found")
    cmd = [binary, "-ni", "-g", "1.0", "-T", "wav", "-F", out_path,
           "-r", str(sample_rate), soundfont, midi_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"fluidsynth timed out after {timeout:.0f}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"fluidsynth failed: {(proc.stderr or '').strip()[-300:]}")
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("fluidsynth produced no output")


def render_wav(parts: dict[str, list], tempo: float, seed: int,
               midi_path: str, out_path: str,
               context: Any = None) -> SynthChoice:
    """Render arranged tracks to a WAV file with the profile-chosen backend.

    Returns the :class:`SynthChoice` (check ``.note`` for a soundfont
    offer to surface).  Falls back to the builtin synth when FluidSynth
    rendering fails — never a silent no-audio.
    """
    choice = choose_synth(context)
    if choice.name == "fluidsynth":
        try:
            _render_fluidsynth(midi_path, out_path, choice.soundfont)
            return choice
        except Exception as exc:  # noqa: BLE001 - graceful fallback
            _log.warning("fluidsynth render failed (%s) — builtin fallback",
                         exc)
            choice = SynthChoice(
                name="builtin", profile=choice.profile,
                reason=f"fluidsynth render failed ({exc}); builtin fallback",
                note=choice.note)
    # builtin path (also the explicit fallback)
    from .synth import write_wav
    write_wav(out_path, parts, tempo, seed=seed)
    return choice
