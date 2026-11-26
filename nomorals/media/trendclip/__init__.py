"""TrendClip — find what's hot, cut it into a vertical clip.

The organ is the TrendClip Studio idea, ported into Devon and made
greater: discovery + ingest + Faster Whisper transcription + moment
scoring with BOTH rungs (the deterministic formula and the brain) +
vertical render + validation + an export package + a museum shelf. Chat-
first: ``/trend`` discovers, ``/clip`` analyzes and renders. Manual
publish only — nothing auto-posts.
"""

from __future__ import annotations

from .brain import brain_pick_moments, brain_platform_copy, build_pick_prompt, validate_picks
from .discovery import discover_youtube_trends, inspect_youtube_url, rank_feed_items
from .exhibitor import trendclip_exhibitor
from .jobs import JobStore, TrendclipJob, new_job_id
from .pipeline import (
    TrendclipPipeline,
    prerendered_clip_for_job,
    trendclip_post_copy,
    trendclip_root,
)
from .provision import ensure_faster_whisper, ensure_opencv
from .render import write_ass_captions
from .rights import validate_rights
from .scoring import assemble_drafts, score_feed_item, select_moments, zscore
from .topics import TopicConfig, load_topic, topic_from_dict, topic_from_query
from .transcribe import normalize_words, transcribe_audio

__all__ = [
    "TopicConfig",
    "TrendclipJob",
    "TrendclipPipeline",
    "JobStore",
    "assemble_drafts",
    "brain_pick_moments",
    "brain_platform_copy",
    "build_pick_prompt",
    "discover_youtube_trends",
    "ensure_faster_whisper",
    "ensure_opencv",
    "inspect_youtube_url",
    "load_topic",
    "new_job_id",
    "normalize_words",
    "prerendered_clip_for_job",
    "rank_feed_items",
    "score_feed_item",
    "select_moments",
    "topic_from_dict",
    "topic_from_query",
    "transcribe_audio",
    "trendclip_exhibitor",
    "trendclip_post_copy",
    "trendclip_root",
    "validate_picks",
    "validate_rights",
    "write_ass_captions",
    "zscore",
]
