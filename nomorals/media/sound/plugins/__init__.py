"""Test plugins for the SoundCraft hosts (lane 107).

Real plugins in the organ's own plugin shape — the hosts load them
by dotted path into the quarantined worker.
"""

from __future__ import annotations

from .crash import CrashPlugin
from .gain import GainPlugin
from .tone import TonePlugin

__all__ = ["CrashPlugin", "GainPlugin", "TonePlugin"]
