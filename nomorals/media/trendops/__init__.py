"""TrendOps — trend-driven video editing.

Devon catches a trend and the clip herself and edits in that format
(the Roadster GTA edit the owner shared), instead of the owner
downloading clips by hand. Trends are data with expiry dates, built
on the style-agnostic edit engine — never a hardcoded look.
"""

from __future__ import annotations

from .editor import TrendOps
from .formats import (
    CAPTION_STYLES,
    CUT_RULES,
    DEFAULT_EXPIRY_DAYS,
    ROADSTER_SEED,
    TrendFormat,
)
from .library import FormatLibrary

__all__ = [
    "TrendOps",
    "TrendFormat",
    "FormatLibrary",
    "ROADSTER_SEED",
    "CAPTION_STYLES",
    "CUT_RULES",
    "DEFAULT_EXPIRY_DAYS",
]
