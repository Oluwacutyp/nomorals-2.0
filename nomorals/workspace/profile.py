"""Deprecated shim: environment profiles moved to :mod:`nomorals.core.profile`.

Import from ``nomorals.core.profile`` instead.  This module stays as a
re-export so older branches and plugins keep working.
"""
from ..core.profile import (  # noqa: F401
    EnvironmentProfile,
    cpu_count,
    detect_profile,
    known_profiles,
    memory_mb,
    resolve_profile,
)
