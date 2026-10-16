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
    "control_music_sources",
    "MUSIC_SOURCES_USAGE",
]
