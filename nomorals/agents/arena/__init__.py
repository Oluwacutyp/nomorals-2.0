"""The self-improvement arena: autonomous research, knowledge digestion,
and power-gated candidate builds that the owner reviews one by one."""

from .core import Arena, run_cycle_safe
from .topics import (
    CATEGORIES,
    TOPIC_BANK,
    bank_size,
    category_stats,
    category_table,
    sample_topic,
    surprise_topic,
    topics_table,
)

__all__ = [
    "Arena", "run_cycle_safe", "CATEGORIES", "TOPIC_BANK", "sample_topic",
    "surprise_topic", "topics_table", "category_table", "category_stats",
    "bank_size",
]
