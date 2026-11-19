"""SoundCraft — Devon's digital-audio-workstation organ (lane 107).

Ported from storytold/soundcraft (dual MIT/Apache-2.0) at upstream
commit 5fabf28. The .scraft format is a ZIP package carrying
``session.json`` (format "soundcraft", version 1) plus WAV source
parts. The organ owns the session model, the mixer, the effects
chain, the realtime callback engine, the quarantined plugin hosts,
automation, and the offline render. See
``nomorals/media/sound-attribution.toml`` for the per-unit SHA-256
record.
"""

from .attribution import verify_attribution
from .automation import AutomationLane
from .edit import SessionEditor
from .effects import Compressor, GainPlugin, Limiter, ParametricEQ, TonePlugin
from .engine import RealtimeEngine
from .errors import SoundError, SoundErrorKind
from .hosts import HostKind, PluginHost, scan_plugins
from .mix import MixEngine, pan_gains
from .model import Bus, Clip, Note, SessionDocument, Source, Track
from .parity import run_parity
from .render import render_session

__all__ = [
    "AutomationLane",
    "Bus",
    "Clip",
    "Compressor",
    "GainPlugin",
    "HostKind",
    "Limiter",
    "MixEngine",
    "Note",
    "ParametricEQ",
    "PluginHost",
    "RealtimeEngine",
    "SessionDocument",
    "SessionEditor",
    "SoundError",
    "SoundErrorKind",
    "Source",
    "TonePlugin",
    "Track",
    "pan_gains",
    "render_session",
    "run_parity",
    "scan_plugins",
    "verify_attribution",
]
