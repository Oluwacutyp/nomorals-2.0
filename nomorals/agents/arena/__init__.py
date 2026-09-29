"""The self-improvement arena: autonomous research, knowledge digestion,
and power-gated candidate builds that the owner reviews one by one."""

from .core import Arena, run_cycle_safe
from .topics import CATEGORIES, TOPIC_BANK, sample_topic

__all__ = ["Arena", "run_cycle_safe", "CATEGORIES", "TOPIC_BANK", "sample_topic"]
