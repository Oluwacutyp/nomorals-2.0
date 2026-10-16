"""Legal music-source registry: music-domain data + schema.

Same entry idea as media/source_registry.py (one entry keyed by
pack axes, with provenance-style licence / etiquette labels), but this
module is standalone music-domain data. It does not reuse that
module's machinery and imports nothing from it.

LEGAL REGISTRY ONLY. Every row carries a licence label. A source
without a legal basis does not appear here.

Packs are equal peers. Pack axes: nation, language, tradition, era,
platform, licence, availability_region. All axes are optional peers;
"*" means universal. No nation leads and no nation is the organising
principle. DEFAULT_PACK is the owner's context used as the default
pack -- it is a default, NOT an organising principle.

Tone-language moat, stated generally: tone handling applies to
Mandarin, Cantonese, Vietnamese, Thai, Lao, Hmong, West African tone
languages equally. G2P entries are per language. This is a
general capability, not a fork and not a per-language special case.

Layer 0: stdlib only. No imports from other packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
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
]

# The spec's `class` axis is stored as `source_class` because `class`
# is a reserved word in Python.
VALID_CLASSES = {
    "purchase",
    "pool",
    "open",
    "artist_enabled",
    "metadata",
    "preview",
    "dataset",
}

VALID_VOLATILITIES = {
    "institutional",
    "commercial",
    "indie",
    "platform_api",
}

# Health-check cadence in hours, derived from volatility.
VOLATILITY_HEALTH_HOURS = {
    "institutional": 720,
    "commercial": 168,
    "indie": 24,
    "platform_api": 6,
}

DEFAULT_PACK: Dict[str, str] = {"nation": "*", "language": "*"}


@dataclass
class MusicSource:
    """One entry in the legal music registry."""

    id: str
    name: str
    pack: dict = field(default_factory=dict)
    source_class: str = "metadata"
    auth: str = "none"
    formats: list = field(default_factory=list)
    quality_ceiling: str = ""
    licence: str = ""
    volatility: str = "commercial"
    bulk: bool = False
    api: bool = False
    bot_wall: bool = False
    etiquette: str = ""
    health_check_hours: Optional[int] = None
    notes: str = ""

    def __post_init__(self) -> None:
        validate(self)
        if self.health_check_hours is None:
            self.health_check_hours = VOLATILITY_HEALTH_HOURS[self.volatility]


def validate(source: MusicSource) -> MusicSource:
    """Check a row. Raises ValueError on a missing licence, a bad
    class, or a bad volatility. Returns the source on success."""
    if not isinstance(source.licence, str) or source.licence == "" \
            or source.licence.isspace():
        raise ValueError(
            "MusicSource %r has no licence label" % (getattr(source, "id", "?"),)
        )
    if source.source_class not in VALID_CLASSES:
        raise ValueError(
            "MusicSource %r has invalid class %r"
            % (getattr(source, "id", "?"), source.source_class)
        )
    if source.volatility not in VALID_VOLATILITIES:
        raise ValueError(
            "MusicSource %r has invalid volatility %r"
            % (getattr(source, "id", "?"), source.volatility)
        )
    return source


def _M(
    *,
    id: str,
    name: str,
    source_class: str,
    auth: str,
    formats: List[str],
    quality_ceiling: str,
    licence: str,
    volatility: str,
    bulk: bool,
    api: bool,
    bot_wall: bool,
    etiquette: str,
    notes: str,
    pack: Optional[Dict[str, str]] = None,
) -> MusicSource:
    """Build one row. Pack defaults to DEFAULT_PACK; the pack licence
    axis mirrors the row licence unless the caller sets it."""
    merged: Dict[str, str] = dict(DEFAULT_PACK)
    if pack:
        merged.update(pack)
    merged.setdefault("licence", licence)
    return MusicSource(
        id=id,
        name=name,
        pack=merged,
        source_class=source_class,
        auth=auth,
        formats=list(formats),
        quality_ceiling=quality_ceiling,
        licence=licence,
        volatility=volatility,
        bulk=bulk,
        api=api,
        bot_wall=bot_wall,
        etiquette=etiquette,
        notes=notes,
    )


BUY = "Buy before use; keep the receipt."
POOL_TERMS = "Membership terms govern use; fetch politely, honour rate limits."
PER_TRACK = "Check the per-track licence; cache results; honour rate limits."
API_POLITE = "Honour the API rate limits; cache results; one call per second."
OPEN_POLITE = "Honour the site terms; cache results; fetch politely."

MUSIC_REGISTRY: List[MusicSource] = [
    # ── §1 Purchase ──────────────────────────────────────────────
    _M(id="qobuz", name="Qobuz", source_class="purchase", auth="account",
       formats=["FLAC", "ALAC", "WAV", "AIFF", "DSD", "DXD"],
       quality_ceiling="24/192", licence="DRM-free purchase",
       volatility="commercial", bulk=False, api=False, bot_wall=False,
       etiquette=BUY,
       notes="Hi-res download store; buy once, keep the files."),
    _M(id="bandcamp", name="Bandcamp", source_class="purchase", auth="account",
       formats=["FLAC", "ALAC", "WAV", "AIFF", "MP3", "AAC", "Ogg"],
       quality_ceiling="24/48",
       licence="DRM-free purchase (artist-direct)",
       volatility="commercial", bulk=False, api=False, bot_wall=False,
       etiquette=BUY,
       notes="Artist-direct store; artists set their own prices and terms."),
    _M(id="7digital", name="7digital", source_class="purchase", auth="account",
       formats=["MP3", "FLAC"], quality_ceiling="24-bit FLAC",
       licence="DRM-free purchase", volatility="commercial",
       bulk=False, api=True, bot_wall=False, etiquette=BUY,
       notes="Per-track and album store; buy only the tracks wanted."),
    _M(id="beatport", name="Beatport", source_class="purchase", auth="account",
       formats=["MP3", "WAV", "AIFF"], quality_ceiling="16/44.1",
       licence="DRM-free purchase", volatility="commercial",
       bulk=False, api=True, bot_wall=False, etiquette=BUY,
       notes="DJ store; dance and electronic catalogue with DJ formats."),
    _M(id="juno_download", name="Juno Download", source_class="purchase",
       auth="account", formats=["FLAC", "WAV", "MP3", "ALAC"],
       quality_ceiling="24-bit FLAC", licence="DRM-free purchase",
       volatility="commercial", bulk=False, api=False, bot_wall=False,
       etiquette=BUY,
       notes="Download store with hi-res and standard tiers."),
    _M(id="hdtracks", name="HDtracks", source_class="purchase", auth="account",
       formats=["FLAC", "WAV", "AIFF", "ALAC"], quality_ceiling="24/192",
       licence="DRM-free purchase", volatility="commercial",
       bulk=False, api=False, bot_wall=False, etiquette=BUY,
       notes="Hi-res download store; audiophile catalogue."),
    _M(id="presto", name="Presto Music", source_class="purchase",
       auth="account", formats=["FLAC", "ALAC", "WAV", "MP3"],
       quality_ceiling="24-bit FLAC", licence="DRM-free purchase",
       volatility="commercial", bulk=False, api=False, bot_wall=False,
       etiquette=BUY,
       notes="Classical store; digital booklets with many albums.",
       pack={"tradition": "classical"}),
    _M(id="nativedsd", name="NativeDSD", source_class="purchase",
       auth="account", formats=["DSF", "FLAC", "DXD"],
       quality_ceiling="DSD512", licence="DRM-free purchase",
       volatility="commercial", bulk=False, api=False, bot_wall=False,
       etiquette=BUY,
       notes="DSD specialist store; native DSD and DXD files."),
    _M(id="nugs", name="Nugs.net", source_class="purchase", auth="account",
       formats=["MP3", "FLAC", "ALAC"], quality_ceiling="24-bit",
       licence="DRM-free purchase (official live)",
       volatility="commercial", bulk=False, api=False, bot_wall=False,
       etiquette=BUY,
       notes="Official live recordings from the artists on the bill.",
       pack={"tradition": "live"}),
    _M(id="highresaudio", name="HighResAudio", source_class="purchase",
       auth="account", formats=["FLAC", "ALAC", "DSD", "DXD"],
       quality_ceiling="24/192", licence="DRM-free purchase",
       volatility="commercial", bulk=False, api=False, bot_wall=False,
       etiquette=BUY,
       notes="Hi-res download store; curated audiophile catalogue."),
    _M(id="bleep", name="Bleep", source_class="purchase", auth="account",
       formats=["FLAC", "WAV", "MP3"], quality_ceiling="16/44.1",
       licence="DRM-free purchase", volatility="indie",
       bulk=False, api=False, bot_wall=False, etiquette=BUY,
       notes="Independent store; electronic and left-field catalogue."),
    _M(id="boomkat", name="Boomkat", source_class="purchase", auth="account",
       formats=["FLAC", "WAV", "MP3"], quality_ceiling="16/44.1",
       licence="DRM-free purchase", volatility="indie",
       bulk=False, api=False, bot_wall=False, etiquette=BUY,
       notes="Independent store; experimental and underground catalogue."),
    _M(id="prostudiomasters", name="ProStudioMasters",
       source_class="purchase", auth="account",
       formats=["AIFF", "FLAC", "DSD"], quality_ceiling="24/192",
       licence="DRM-free purchase", volatility="commercial",
       bulk=False, api=False, bot_wall=False, etiquette=BUY,
       notes="Hi-res download store; studio-grade files."),
    _M(id="ototo", name="OTOTOY", source_class="purchase", auth="account",
       formats=["FLAC", "WAV", "MP3", "ALAC"],
       quality_ceiling="24-bit FLAC", licence="DRM-free purchase",
       volatility="commercial", bulk=False, api=False, bot_wall=False,
       etiquette=BUY,
       notes="Japanese download store; domestic catalogue.",
       pack={"nation": "JP", "language": "ja",
             "availability_region": "JP"}),
    _M(id="genie", name="Genie Music", source_class="purchase", auth="account",
       formats=["MP3", "FLAC"], quality_ceiling="16/44.1",
       licence="DRM-free purchase", volatility="commercial",
       bulk=False, api=False, bot_wall=True, etiquette=BUY,
       notes="Korean download store; domestic catalogue.",
       pack={"nation": "KR", "language": "ko",
             "availability_region": "KR"}),
    _M(id="tamilflac", name="TamilFLAC", source_class="purchase",
       auth="account", formats=["FLAC", "MP3"], quality_ceiling="16/44.1",
       licence="DRM-free purchase", volatility="commercial",
       bulk=False, api=False, bot_wall=False, etiquette=BUY,
       notes="Indian download store; Tamil-language catalogue.",
       pack={"nation": "IN", "language": "ta",
             "availability_region": "IN"}),
    _M(id="boomplay", name="Boomplay", source_class="purchase",
       auth="account", formats=["MP3", "AAC"], quality_ceiling="320kbps",
       licence="Licensed purchase / playback under store terms",
       volatility="commercial", bulk=False, api=True, bot_wall=True,
       etiquette="Honour the store terms; fetch politely.",
       notes="Pan-African store and licensed playback; honour store terms.",
       pack={"availability_region": "pan-African"}),

    # ── §2 DJ pools ───────────────────────────────────────────────
    _M(id="djcity", name="DJcity", source_class="pool",
       auth="paid_membership", formats=["MP3", "WAV", "AIFF"],
       quality_ceiling="320kbps", licence="promo pool membership terms",
       volatility="commercial", bulk=True, api=False, bot_wall=False,
       etiquette=POOL_TERMS,
       notes="Promo pool with a wide genre range; membership terms govern use.",
       pack={"availability_region": "US/UK/DE/JP/LATAM/South Asia"}),
    _M(id="bpm_supreme", name="BPM Supreme", source_class="pool",
       auth="paid_membership", formats=["MP3", "WAV"],
       quality_ceiling="320kbps", licence="promo pool membership terms",
       volatility="commercial", bulk=True, api=False, bot_wall=False,
       etiquette=POOL_TERMS,
       notes="Open-format and Latin/Global pool; membership terms govern use."),
    _M(id="zipdj", name="ZIPDJ", source_class="pool",
       auth="paid_membership", formats=["MP3", "WAV"],
       quality_ceiling="320kbps", licence="promo pool membership terms",
       volatility="commercial", bulk=True, api=False, bot_wall=False,
       etiquette=POOL_TERMS,
       notes="Large electronic pool; membership terms govern use."),
    _M(id="beatsource", name="BeatSource", source_class="pool",
       auth="paid_membership", formats=["MP3", "WAV", "AIFF"],
       quality_ceiling="320kbps", licence="promo pool membership terms",
       volatility="commercial", bulk=True, api=False, bot_wall=False,
       etiquette=POOL_TERMS,
       notes="DJ pool for open-format and Latin/Global sets; membership terms govern use."),
    _M(id="digital_dj_pool", name="Digital DJ Pool", source_class="pool",
       auth="paid_membership", formats=["MP3", "WAV"],
       quality_ceiling="320kbps", licence="promo pool membership terms",
       volatility="commercial", bulk=True, api=False, bot_wall=False,
       etiquette=POOL_TERMS,
       notes="DJ pool; files kept under membership terms."),
    _M(id="mymp3pool", name="MyMP3Pool", source_class="pool",
       auth="paid_membership", formats=["MP3", "WAV"],
       quality_ceiling="320kbps", licence="promo pool membership terms",
       volatility="commercial", bulk=True, api=False, bot_wall=False,
       etiquette=POOL_TERMS,
       notes="DJ pool; membership terms govern use."),
    _M(id="dms", name="DMS", source_class="pool",
       auth="paid_membership", formats=["MP3", "WAV"],
       quality_ceiling="320kbps", licence="promo pool membership terms",
       volatility="commercial", bulk=True, api=False, bot_wall=False,
       etiquette=POOL_TERMS,
       notes="Promo pool; membership terms govern use."),
    _M(id="club_killers", name="Club Killers", source_class="pool",
       auth="paid_membership", formats=["MP3", "WAV"],
       quality_ceiling="320kbps", licence="promo pool membership terms",
       volatility="commercial", bulk=True, api=False, bot_wall=False,
       etiquette=POOL_TERMS,
       notes="DJ pool; membership terms govern use."),
    _M(id="vault", name="Vault", source_class="pool",
       auth="paid_membership", formats=["MP3", "WAV"],
       quality_ceiling="320kbps", licence="promo pool membership terms",
       volatility="commercial", bulk=True, api=False, bot_wall=False,
       etiquette=POOL_TERMS,
       notes="Caribbean-focused pool; membership terms govern use.",
       pack={"availability_region": "Caribbean"}),
    _M(id="heavy_hits", name="Heavy Hits", source_class="pool",
       auth="paid_membership", formats=["MP3", "WAV"],
       quality_ceiling="320kbps", licence="promo pool membership terms",
       volatility="commercial", bulk=True, api=False, bot_wall=False,
       etiquette=POOL_TERMS,
       notes="DJ pool; membership terms govern use."),
    _M(id="crate_connect", name="Crate Connect", source_class="pool",
       auth="paid_membership", formats=["MP3", "WAV"],
       quality_ceiling="320kbps", licence="promo pool membership terms",
       volatility="commercial", bulk=True, api=False, bot_wall=False,
       etiquette=POOL_TERMS,
       notes="DJ pool; membership terms govern use."),
    _M(id="promo_only", name="Promo Only", source_class="pool",
       auth="paid_membership", formats=["MP4", "MP3"],
       quality_ceiling="1080p video", licence="promo pool membership terms",
       volatility="commercial", bulk=True, api=False, bot_wall=False,
       etiquette=POOL_TERMS,
       notes="Promo pool with video content; membership terms govern use."),
    _M(id="traxsource", name="Traxsource", source_class="pool",
       auth="account", formats=["MP3", "WAV", "AIFF"],
       quality_ceiling="16/44.1", licence="DRM-free purchase (DJ store)",
       volatility="commercial", bulk=False, api=False, bot_wall=False,
       etiquette=BUY,
       notes="Store with per-track buying; house and soulful focus."),

    # ── §3 Open / CC / public domain ───────────────────────────────
    _M(id="free_music_archive", name="Free Music Archive",
       source_class="open", auth="none",
       formats=["MP3", "FLAC", "Ogg"], quality_ceiling="varies by track",
       licence="CC per-track (see track page)", volatility="institutional",
       bulk=False, api=True, bot_wall=False, etiquette=PER_TRACK,
       notes="Curated free archive; check the per-track licence before use."),
    _M(id="jamendo", name="Jamendo", source_class="open", auth="none",
       formats=["MP3", "FLAC"], quality_ceiling="varies by track",
       licence="CC per-track; paid commercial tier separate",
       volatility="commercial", bulk=False, api=True, bot_wall=False,
       etiquette=PER_TRACK,
       notes="Creative Commons catalogue plus a separate paid commercial tier."),
    _M(id="ccmixter", name="ccMixter", source_class="open", auth="none",
       formats=["MP3", "FLAC", "WAV"], quality_ceiling="varies by track",
       licence="CC per-track (remix-friendly)", volatility="indie",
       bulk=False, api=False, bot_wall=False, etiquette=PER_TRACK,
       notes="Remixable stems and songs; per-track licence governs reuse."),
    _M(id="internet_archive", name="Internet Archive", source_class="open",
       auth="none", formats=["MP3", "FLAC", "Ogg"],
       quality_ceiling="varies by item",
       licence="Public domain / CC per-item", volatility="institutional",
       bulk=True, api=True, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Live Music Archive plus 78s and public-domain items; per-item rights vary.",
       pack={"tradition": "live", "era": "78s era"}),
    _M(id="musopen", name="Musopen", source_class="open", auth="none",
       formats=["MP3", "FLAC"], quality_ceiling="varies by recording",
       licence="Public domain", volatility="institutional",
       bulk=False, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Public-domain classical recordings and sheet music.",
       pack={"tradition": "classical"}),
    _M(id="freesound", name="Freesound", source_class="open", auth="account",
       formats=["WAV", "MP3", "FLAC"], quality_ceiling="varies by file",
       licence="CC per-file (CC0 / CC-BY / CC-BY-NC)",
       volatility="institutional", bulk=False, api=True, bot_wall=False,
       etiquette=PER_TRACK,
       notes="Sound effects and field recordings; per-file licence governs reuse."),
    _M(id="youtube_audio_library", name="YouTube Audio Library",
       source_class="open", auth="account", formats=["MP3", "WAV"],
       quality_ceiling="varies by track",
       licence="Royalty-free under YouTube Audio Library terms",
       volatility="platform_api", bulk=False, api=False, bot_wall=True,
       etiquette="Honour the stated terms; fetch politely.",
       notes="Royalty-free music and effects; Google account needed; honour the stated terms.",
       pack={"platform": "YouTube"}),
    _M(id="pixabay", name="Pixabay", source_class="open", auth="none",
       formats=["MP3", "WAV"], quality_ceiling="varies by track",
       licence="Royalty-free under Pixabay terms", volatility="commercial",
       bulk=False, api=True, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Royalty-free music and effects; per-item licence as shown."),
    _M(id="audionautix", name="Audionautix", source_class="open", auth="none",
       formats=["MP3"], quality_ceiling="varies by track",
       licence="Royalty-free (CC-BY as stated)", volatility="indie",
       bulk=False, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Royalty-free music; per-track terms as shown on the site."),
    _M(id="soundclick", name="SoundClick", source_class="open", auth="none",
       formats=["MP3"], quality_ceiling="varies by track",
       licence="Per-track licence as stated by rights holder",
       volatility="commercial", bulk=False, api=False, bot_wall=False,
       etiquette=PER_TRACK,
       notes="Artist uploads; free and paid tiers with per-track terms set by the rights holder."),
    _M(id="imslp", name="IMSLP", source_class="open", auth="none",
       formats=["PDF", "MP3"], quality_ceiling="as scanned",
       licence="Public domain scores", volatility="institutional",
       bulk=False, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Public-domain scores; check the per-score copyright status for your region.",
       pack={"tradition": "classical"}),
    _M(id="soundcloud", name="SoundCloud", source_class="artist_enabled",
       auth="none", formats=["MP3", "AAC"], quality_ceiling="varies by track",
       licence="Per-track licence as set by uploader",
       volatility="platform_api", bulk=False, api=True, bot_wall=True,
       etiquette=API_POLITE,
       notes="Artist uploads; use only the tracks whose licence allows it.",
       pack={"platform": "SoundCloud"}),
    _M(id="audiomack", name="Audiomack", source_class="artist_enabled",
       auth="none", formats=["MP3", "AAC"], quality_ceiling="varies by track",
       licence="Per-track licence as set by uploader",
       volatility="platform_api", bulk=False, api=True, bot_wall=True,
       etiquette=API_POLITE,
       notes="Artist uploads; use only the tracks whose licence allows it.",
       pack={"platform": "Audiomack"}),

    # ── §4 Metadata / preview ──────────────────────────────────────
    _M(id="musicbrainz", name="MusicBrainz", source_class="metadata",
       auth="none", formats=[], quality_ceiling="metadata only",
       licence="Open (CC0 metadata)", volatility="platform_api",
       bulk=False, api=True, bot_wall=False, etiquette=API_POLITE,
       notes="Open music encyclopedia; community metadata under open terms."),
    _M(id="acoustid", name="AcoustID", source_class="metadata",
       auth="api_key", formats=[], quality_ceiling="metadata only",
       licence="Open (MIT-style)", volatility="platform_api",
       bulk=False, api=True, bot_wall=False, etiquette=API_POLITE,
       notes="Open fingerprint service; Chromaprint lookups map audio to MusicBrainz IDs."),
    _M(id="acousticbrainz", name="AcousticBrainz", source_class="metadata",
       auth="none", formats=[], quality_ceiling="metadata only",
       licence="Open (CC0)", volatility="platform_api",
       bulk=False, api=True, bot_wall=False, etiquette=API_POLITE,
       notes="Open acoustic features: BPM, key, scale, mood."),
    _M(id="deezer", name="Deezer", source_class="preview", auth="api_key",
       formats=["preview MP3"], quality_ceiling="30s preview",
       licence="API terms; previews only", volatility="platform_api",
       bulk=False, api=True, bot_wall=True, etiquette=API_POLITE,
       notes="Streaming catalogue; 30-second previews under API terms."),
    _M(id="itunes", name="iTunes", source_class="preview", auth="none",
       formats=["preview AAC"], quality_ceiling="30s preview",
       licence="API terms; previews only", volatility="platform_api",
       bulk=False, api=True, bot_wall=False, etiquette=API_POLITE,
       notes="Previews plus high-resolution artwork under API terms."),
    _M(id="lrclib", name="LRCLIB", source_class="metadata", auth="none",
       formats=[], quality_ceiling="metadata only",
       licence="Open (MIT-style)", volatility="platform_api",
       bulk=False, api=True, bot_wall=False, etiquette=API_POLITE,
       notes="Open synced-lyrics service; community contributions."),
    _M(id="cover_art_archive", name="Cover Art Archive",
       source_class="metadata", auth="none", formats=["JPEG", "PNG"],
       quality_ceiling="full artwork", licence="Open (CC0)",
       volatility="platform_api", bulk=False, api=True, bot_wall=False,
       etiquette=API_POLITE,
       notes="Open cover-art archive tied to MusicBrainz IDs."),
    _M(id="theaudiodb", name="TheAudioDB", source_class="metadata",
       auth="api_key", formats=[], quality_ceiling="metadata only",
       licence="API terms", volatility="platform_api",
       bulk=False, api=True, bot_wall=False, etiquette=API_POLITE,
       notes="Music metadata and artwork under API terms."),
    _M(id="lastfm", name="Last.fm", source_class="metadata", auth="api_key",
       formats=[], quality_ceiling="metadata only",
       licence="API terms", volatility="platform_api",
       bulk=False, api=True, bot_wall=False, etiquette=API_POLITE,
       notes="Listening data and metadata under API terms."),
    _M(id="listenbrainz", name="ListenBrainz", source_class="metadata",
       auth="api_key", formats=[], quality_ceiling="metadata only",
       licence="Open (CC0)", volatility="platform_api",
       bulk=False, api=True, bot_wall=False, etiquette=API_POLITE,
       notes="Open listening history and metadata; user data stays with the user."),
    _M(id="discogs", name="Discogs", source_class="metadata", auth="api_key",
       formats=[], quality_ceiling="metadata only",
       licence="API terms", volatility="platform_api",
       bulk=False, api=True, bot_wall=False, etiquette=API_POLITE,
       notes="Release database and marketplace metadata under API terms."),
    _M(id="spotify", name="Spotify", source_class="metadata", auth="oauth",
       formats=[], quality_ceiling="metadata only",
       licence="API terms; metadata only, no audio use without a licensing deal",
       volatility="platform_api", bulk=False, api=True, bot_wall=True,
       etiquette=API_POLITE,
       notes="Metadata only here; audio use needs a licensing deal."),
    _M(id="musixmatch", name="Musixmatch", source_class="metadata",
       auth="api_key", formats=[], quality_ceiling="metadata only",
       licence="API terms", volatility="platform_api",
       bulk=False, api=True, bot_wall=False, etiquette=API_POLITE,
       notes="Lyrics under API terms; honour the display terms."),
    _M(id="acrcloud", name="ACRCloud", source_class="metadata",
       auth="api_key", formats=[], quality_ceiling="metadata only",
       licence="Paid API terms", volatility="platform_api",
       bulk=False, api=True, bot_wall=False, etiquette=API_POLITE,
       notes="Audio recognition under paid API terms."),
    _M(id="mubert", name="Mubert", source_class="preview", auth="api_key",
       formats=["WAV", "MP3"], quality_ceiling="as generated",
       licence="API terms; generated audio under Mubert licence",
       volatility="platform_api", bulk=False, api=True, bot_wall=False,
       etiquette=API_POLITE,
       notes="Generative music under API terms; generated audio under the Mubert licence."),

    # ── §5 Training datasets ───────────────────────────────────────
    _M(id="mtg_jamendo", name="MTG-Jamendo", source_class="dataset",
       auth="none", formats=["MP3"], quality_ceiling="as released",
       licence="CC per-track", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="55,609 tracks with 195 tags; Creative Commons."),
    _M(id="fma_dataset", name="FMA Dataset", source_class="dataset",
       auth="none", formats=["MP3"], quality_ceiling="as released",
       licence="CC per-track", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="106,574 tracks; Creative Commons per-track."),
    _M(id="msd", name="Million Song Dataset", source_class="dataset",
       auth="none", formats=[], quality_ceiling="as released",
       licence="Metadata under open terms", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="505,216 tracks of metadata; audio via the linked sources."),
    _M(id="maestro", name="MAESTRO", source_class="dataset", auth="none",
       formats=["WAV", "MIDI"], quality_ceiling="as released",
       licence="Open (CC-BY-NC-SA as released)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="200+ hours of piano with tight alignment.",
       pack={"tradition": "classical piano"}),
    _M(id="musicnet", name="MusicNet", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open (CC-BY)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Labelled classical recordings for research on note-level labels.",
       pack={"tradition": "classical"}),
    _M(id="medleydb", name="MedleyDB", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open (CC-BY-NC-SA)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Multitrack recordings with melody and instrument labels."),
    _M(id="musdb18", name="MUSDB18", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open (CC-BY-NC-SA)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="150 tracks with isolated stems for separation research."),
    _M(id="musdb_xl", name="MUSDB-XL", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open (CC-BY-NC-SA)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Extended MUSDB with more tracks and stems."),
    _M(id="slakh2100", name="Slakh2100", source_class="dataset", auth="none",
       formats=["WAV", "MIDI"], quality_ceiling="as released",
       licence="Open (CC-BY)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="2,100 MIDI-aligned tracks for mixture research."),
    _M(id="lakh_midi", name="Lakh MIDI", source_class="dataset", auth="none",
       formats=["MIDI"], quality_ceiling="as released",
       licence="Open (CC-BY)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Large MIDI collection aligned to the Million Song Dataset."),
    _M(id="giantsteps", name="GiantSteps", source_class="dataset",
       auth="none", formats=["MP3"], quality_ceiling="as released",
       licence="Open (CC-BY)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Electronic dance music tempo and key data."),
    _M(id="magnatagatune", name="MagnaTagATune", source_class="dataset",
       auth="none", formats=["MP3"], quality_ceiling="as released",
       licence="Open (CC-BY-NC)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Tagging data with clips and labels."),
    _M(id="nsynth", name="NSynth", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open (CC-BY)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Musical notes from many instruments for synthesis research."),
    _M(id="irmas", name="IRMAS", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Instrument recognition data with single-note samples."),
    _M(id="urmp", name="URMP", source_class="dataset", auth="none",
       formats=["WAV", "MP4"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Multitrack classical performances with video.",
       pack={"tradition": "classical"}),
    _M(id="jamendomaxcaps", name="JamendoMaxCaps", source_class="dataset",
       auth="none", formats=["MP3"], quality_ceiling="as released",
       licence="CC per-track", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="362,238 captioned tracks for caption research."),
    _M(id="disco_10m", name="Disco-10M", source_class="dataset", auth="none",
       formats=["MP3"], quality_ceiling="as released",
       licence="CC per-track", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Large captioned music collection for research."),
    _M(id="musan", name="MUSAN", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open (CC-BY)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Speech, music, and noise data across 12 languages."),
    _M(id="gtzan", name="GTZAN", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Genre classification set; small and widely used."),
    _M(id="tabla_solo", name="Tabla Solo", source_class="dataset",
       auth="none", formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Solo tabla recordings for percussion research.",
       pack={"tradition": "Hindustani classical"}),

    # ── §5 Speech packs (equal peers) ──────────────────────────────
    _M(id="common_voice", name="Common Voice", source_class="dataset",
       auth="none", formats=["MP3"], quality_ceiling="as released",
       licence="Open (CC0)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Crowd-sourced speech in many languages under open terms."),
    _M(id="voxceleb", name="VoxCeleb", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Speaker recognition data from public interviews."),
    _M(id="librispeech", name="LibriSpeech", source_class="dataset",
       auth="none", formats=["FLAC"], quality_ceiling="as released",
       licence="Open (CC-BY)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Read English speech from public-domain audiobooks.",
       pack={"language": "en"}),
    _M(id="libritts", name="LibriTTS", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open (CC-BY)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Read English speech tuned for synthesis research.",
       pack={"language": "en"}),
    _M(id="emilia", name="Emilia", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open (CC-BY-NC)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Multilingual speech from in-the-wild recordings."),
    _M(id="waxal", name="Waxal", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="21 African languages, 11,000+ hours of speech.",
       pack={"availability_region": "Africa"}),
    _M(id="naijavoices", name="NaijaVoices", source_class="dataset",
       auth="none", formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="1,800+ hours, 5,000+ speakers; Nigerian languages.",
       pack={"nation": "NG"}),
    _M(id="lacuna", name="Lacuna", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Speech data for under-served languages."),
    _M(id="masakhane", name="Masakhane", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="African language speech and text efforts.",
       pack={"availability_region": "Africa"}),
    _M(id="sadilar", name="SADiLaR", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="South African language resources.",
       pack={"nation": "ZA"}),
    _M(id="indic_superb", name="Indic SUPERB", source_class="dataset",
       auth="none", formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Indian language speech benchmark.",
       pack={"nation": "IN"}),
    _M(id="aishell", name="AISHELL", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Mandarin speech data for recognition research.",
       pack={"nation": "CN", "language": "zh"}),
    _M(id="jvs", name="JVS Corpus", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Japanese speech corpus for synthesis research.",
       pack={"nation": "JP", "language": "ja"}),
    _M(id="arabic_speech", name="Arabic Speech", source_class="dataset",
       auth="none", formats=["WAV"], quality_ceiling="as released",
       licence="Open for research", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Arabic speech data across dialects.",
       pack={"language": "ar"}),
    _M(id="mls", name="Multilingual LibriSpeech", source_class="dataset",
       auth="none", formats=["FLAC"], quality_ceiling="as released",
       licence="Open (CC-BY)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="Multilingual read speech across many languages."),
    _M(id="voxpopuli", name="VoxPopuli", source_class="dataset", auth="none",
       formats=["WAV"], quality_ceiling="as released",
       licence="Open (CC0)", volatility="institutional",
       bulk=True, api=False, bot_wall=False, etiquette=OPEN_POLITE,
       notes="European Parliament speech across many languages."),
]


def all_sources() -> List[MusicSource]:
    """Every registered source, in registry order."""
    return list(MUSIC_REGISTRY)


def by_id(source_id: str) -> Optional[MusicSource]:
    """One source by id, or None."""
    for source in MUSIC_REGISTRY:
        if source.id == source_id:
            return source
    return None


def by_class(source_class: str) -> List[MusicSource]:
    """All sources in one class (purchase, pool, open, ...)."""
    return [s for s in MUSIC_REGISTRY if s.source_class == source_class]


def by_pack(axis: str, value: str) -> List[MusicSource]:
    """All sources whose pack matches ``axis == value``.

    "*" on either side matches any value: a query of "*" returns
    every row, and a row whose pack value is "*" (universal) matches
    every query. A row missing the axis is treated as universal.
    """
    out: List[MusicSource] = []
    for source in MUSIC_REGISTRY:
        row_value = source.pack.get(axis, "*")
        if value == "*" or row_value == "*" or row_value == value:
            out.append(source)
    return out


def by_licence(substring: str) -> List[MusicSource]:
    """All sources whose licence label contains ``substring``
    (case-insensitive)."""
    needle = substring.lower()
    return [s for s in MUSIC_REGISTRY if needle in s.licence.lower()]


def health_due(source: MusicSource, last_check_ts: Optional[float],
               now: float) -> bool:
    """True when ``source`` is due a health check.

    ``last_check_ts`` of None means never checked, so due. Otherwise
    due once ``now - last_check_ts`` reaches the source's
    ``health_check_hours`` (derived from volatility).
    """
    if last_check_ts is None:
        return True
    hours = source.health_check_hours or 0
    return (now - last_check_ts) >= hours * 3600
