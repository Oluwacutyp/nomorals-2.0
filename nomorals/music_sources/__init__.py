"""Music sources package: legal registry + enrichment + downloader + health + play.

Surfaces: ``nm music sources|enrich|health|play-registry`` and
``/music sources|enrich|health|registry-play`` in chat (via
music_sources/chat.py). Downloading is per-class and honest:
purchase = checkout plan, pool = membership, open/dataset =
injected-fetcher fetch, metadata = metadata only. Licence is shown
on every result.
"""

from .chat import MUSIC_SOURCES_USAGE, control_music_sources
from .downloader import DownloadPlan, DownloadResult, download, plan_download
from .enrichment import EnrichmentChain, EnrichmentResult
from .health import HealthProber, HealthStatus
from .placements import (
    COMMUNITY_VERIFY_MARGIN,
    SEED_PLACEMENTS,
    Placement,
    PlacementStore,
    default_store_path,
    format_placement_reply,
    format_reverse_reply,
    lookup,
    open_store,
    parse_soundtrack_query,
    placements_for_clip,
    play_handoff,
)
from .play import PlayResult, play_via_registry, registry_play_text
from .registry import (
    DEFAULT_PACK,
    MUSIC_REGISTRY,
    VALID_CLASSES,
    VALID_VOLATILITIES,
    VOLATILITY_HEALTH_HOURS,
    MusicSource,
    all_sources,
    by_class,
    by_id,
    by_licence,
    by_pack,
    health_due,
    validate,
)

__all__ = [
    "EnrichmentChain",
    "EnrichmentResult",
    "MusicSource",
    "MUSIC_REGISTRY",
    "DEFAULT_PACK",
    "VALID_CLASSES",
    "VALID_VOLATILITIES",
    "VOLATILITY_HEALTH_HOURS",
    "validate",
    "all_sources",
    "by_id",
    "by_class",
    "by_pack",
    "by_licence",
    "health_due",
    "DownloadPlan",
    "DownloadResult",
    "plan_download",
    "download",
    "HealthStatus",
    "HealthProber",
    "PlayResult",
    "play_via_registry",
    "registry_play_text",
    "Placement",
    "PlacementStore",
    "SEED_PLACEMENTS",
    "COMMUNITY_VERIFY_MARGIN",
    "default_store_path",
    "open_store",
    "parse_soundtrack_query",
    "lookup",
    "placements_for_clip",
    "format_placement_reply",
    "format_reverse_reply",
    "play_handoff",
    "control_music_sources",
    "MUSIC_SOURCES_USAGE",
]
