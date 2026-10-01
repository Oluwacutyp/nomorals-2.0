"""The self-improvement arena: autonomous research, knowledge digestion,
and power-gated candidate builds that the owner reviews one by one."""

from .core import Arena, run_cycle_safe
from .topics import (
    CATEGORIES,
    DEFAULT_ANTI_REPEAT,
    TOPIC_BANK,
    TopicPack,
    all_categories,
    anti_repeat_window,
    bank_size,
    category_stats,
    category_table,
    clear_recent_topics,
    recent_topics,
    register_topic_pack,
    sample_topic,
    set_anti_repeat_window,
    surprise_topic,
    topic_packs,
    topics_table,
    unregister_topic_pack,
)

__all__ = [
    "Arena", "run_cycle_safe", "CATEGORIES", "DEFAULT_ANTI_REPEAT",
    "TOPIC_BANK", "TopicPack", "all_categories", "anti_repeat_window",
    "sample_topic", "surprise_topic", "topics_table", "category_table",
    "category_stats", "bank_size", "register_topic_pack",
    "unregister_topic_pack", "topic_packs", "recent_topics",
    "clear_recent_topics", "set_anti_repeat_window",
]
