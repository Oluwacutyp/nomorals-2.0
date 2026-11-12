"""Per-site download descriptors — the download chain as data (B4).

B1–B3 hard-coded the SoundCloud path inside the download engine:
``_stage_connector`` carried a ``soundcloud.com`` host check, and
``download()`` carried the ladder order as a tuple literal.  Adding
a site meant editing the engine.

This module is the data that replaces that: one
:class:`SiteDescriptor` per site — URL patterns, the chain of rungs
that applies to the site in order, the connector id that resolves
its streams, the block vocabulary the site can produce, the auth
the site needs, and whether the site serves downloads at all.  The
engine (:mod:`nomorals.media.downloader`) reads descriptors; a new
site is a new descriptor, never an engine edit.

Honest-failure shape (the contract the engine keeps for every
descriptor): a connector refusal carrying a class from the
descriptor's ``block_classes`` is journaled WITH its class and the
chain moves on — the block is evidence, never a dead end, and the
download never raises.  True DRM is named, never chased (the
rendered stage reports it from the log's license traffic and
downloads nothing).

Auth: ``auth="none"`` means the public path (SoundCloud, YouTube
and TikTok today).
``"account"`` / ``"vault"`` name a requirement the authenticated
rung (B5, :mod:`nomorals.media.site_auth`) honours: a descriptor
with ``auth != "none"`` resolves its session before the chain
runs, and a missing session fails the download as found-but-
blocked, class ``login``, instead of running anonymously.  A
descriptor with ``auth == "none"`` may still declare
``auth_domains`` / ``auth_cookie_names`` — the shape of the
site's session — so a classified ``login`` block from the
site's own connector can retry the direct stage authenticated
through the estate's cookies file.

Validation is at LOAD, never mid-download: a malformed descriptor
fails in :meth:`SiteDescriptor.validate` / the registry's
constructor with a :class:`SiteDescriptorError` naming the site
and the field — a bad pattern, an unknown rung, or a missing
required field can never surface as a failed download.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..core.logging_setup import get_logger
from .downloader_failures import record_failure

_log = get_logger(__name__)

__all__ = [
    "CHAIN_STAGES",
    "BLOCK_CLASSES",
    "AUTH_KINDS",
    "SiteDescriptorError",
    "SiteDescriptor",
    "SiteDescriptorRegistry",
    "SOUNDCLOUD_DESCRIPTOR",
    "YOUTUBE_DESCRIPTOR",
    "TIKTOK_DESCRIPTOR",
    "DEFAULT_SITE_DESCRIPTORS",
]

#: The chain's stage vocabulary, in the canonical ladder order
#: (B2/B3 pinned: direct, kit, connector, proxy, rendered, browser).
#: A descriptor's ``chain`` is a subsequence of this, in this order
#: unless the site genuinely needs a different order.
CHAIN_STAGES: tuple[str, ...] = (
    "direct", "kit", "connector", "proxy", "rendered", "browser",
)

#: The structured block vocabulary a site may produce — the same
#: classes the SoundCloud connector speaks (``connectors.soundcloud``
#: ``BLOCK_*``): preview / drm / region / login / private.  A
#: connector refusal carrying one of these is journaled with its
#: class; a class outside the vocabulary is never invented.
BLOCK_CLASSES: tuple[str, ...] = (
    "preview", "drm", "region", "login", "private",
)

#: Auth requirements a descriptor may declare.  ``"none"`` is the
#: public path; ``"account"`` needs the owner's signed-in account
#: on the site; ``"vault"`` needs vault-held credentials.  B5
#: (:mod:`nomorals.media.site_auth`) is the rung that honours the
#: non-``"none"`` requirements — the engine resolves the session
#: before the chain and never runs anonymously for such a site.
AUTH_KINDS: tuple[str, ...] = ("none", "account", "vault")


class SiteDescriptorError(ValueError):
    """A site descriptor is malformed.  The message always names
    the site and the field, so a bad descriptor is fixable at its
    definition instead of surfacing as a failed download."""

    def __init__(self, site: str, field_name: str, detail: str) -> None:
        self.site = str(site or "")
        self.field_name = str(field_name or "")
        super().__init__(
            f"site descriptor for {self.site!r}: field "
            f"{self.field_name!r}: {detail}")


@dataclass(frozen=True)
class SiteDescriptor:
    """One site's download behaviour, as data.

    ``site``: the site's name (``"soundcloud"``).  ``url_patterns``:
    regexes matched against the URL (same convention as
    :class:`nomorals.media.download_kit.ExtractorDescriptor`).
    ``chain``: the rungs that apply to this site, in order — a
    subsequence of :data:`CHAIN_STAGES`.  ``connector``: the
    connector id that resolves this site's streams (``"soundcloud"``),
    "" when the site has no connector rung; required whenever
    ``"connector"`` is in ``chain``.  ``block_classes``: the block
    vocabulary this site can produce.  ``auth``: the auth the site
    needs (see :data:`AUTH_KINDS`).  ``auth_domains``: the cookie
    domains the site's session lives under (required whenever
    ``auth`` is not ``"none"``; also the domains a ``login``-block
    retry consults for an ``auth == "none"`` site).
    ``auth_cookie_names``: the session cookie names that prove a
    signed-in session (``("auth_token",)`` for SoundCloud) — when
    given, a live cookie of any other name does not count as a
    session.  ``downloadable``: False when the site serves no
    downloads at all.
    """

    site: str
    url_patterns: tuple[str, ...]
    chain: tuple[str, ...] = CHAIN_STAGES
    connector: str = ""
    block_classes: tuple[str, ...] = ()
    auth: str = "none"
    auth_domains: tuple[str, ...] = ()
    auth_cookie_names: tuple[str, ...] = ()
    downloadable: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        # Frozen dataclass: validate on construction so a bad
        # descriptor can never exist to be registered.
        self.validate()

    # -- matching ------------------------------------------------------

    def matches(self, url: str) -> bool:
        try:
            return any(re.search(p, url or "", re.I)
                       for p in self.url_patterns)
        except Exception as e:
            record_failure("site_descriptor.matches", e)
            return False

    @property
    def auth_needed(self) -> bool:
        return self.auth != "none"

    # -- validation (at LOAD, never mid-download) ----------------------

    def validate(self) -> None:
        """Raise :class:`SiteDescriptorError` naming site + field
        when this descriptor is malformed."""
        site = str(self.site or "")
        if not site.strip():
            raise SiteDescriptorError(
                site, "site", "site name is empty")
        patterns = self.url_patterns
        if not patterns:
            raise SiteDescriptorError(
                site, "url_patterns",
                "at least one URL pattern is required")
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                raise SiteDescriptorError(
                    site, "url_patterns",
                    "URL patterns must be non-empty strings")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise SiteDescriptorError(
                    site, "url_patterns",
                    f"pattern {pattern!r} does not compile: {exc}"
                    ) from exc
        chain = tuple(self.chain or ())
        if not chain:
            raise SiteDescriptorError(
                site, "chain", "chain is empty — at least one rung "
                "is required")
        for stage in chain:
            if stage not in CHAIN_STAGES:
                raise SiteDescriptorError(
                    site, "chain",
                    f"unknown rung {stage!r} (want one of "
                    f"{', '.join(CHAIN_STAGES)})")
        if "connector" in chain and not str(self.connector or "").strip():
            raise SiteDescriptorError(
                site, "connector",
                "chain includes the connector rung but no connector "
                "id is declared")
        for block in self.block_classes or ():
            if block not in BLOCK_CLASSES:
                raise SiteDescriptorError(
                    site, "block_classes",
                    f"unknown block class {block!r} (want one of "
                    f"{', '.join(BLOCK_CLASSES)})")
        if self.auth not in AUTH_KINDS:
            raise SiteDescriptorError(
                site, "auth",
                f"unknown auth kind {self.auth!r} (want one of "
                f"{', '.join(AUTH_KINDS)})")
        for field_name, values in (
                ("auth_domains", self.auth_domains),
                ("auth_cookie_names", self.auth_cookie_names)):
            for value in values or ():
                if not isinstance(value, str) or not value.strip():
                    raise SiteDescriptorError(
                        site, field_name,
                        f"{field_name} entries must be non-empty "
                        "strings")
        if self.auth != "none" and not tuple(self.auth_domains or ()):
            raise SiteDescriptorError(
                site, "auth_domains",
                f"auth {self.auth!r} requires at least one "
                "auth_domains entry — the cookie domains the "
                "session lives under")

    # -- data round-trip ------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SiteDescriptor:
        """Build + validate from a plain dict (a JSON/YAML-shaped
        descriptor).  Unknown keys and missing required fields fail
        naming the field."""
        if not isinstance(data, dict):
            raise SiteDescriptorError(
                "", "descriptor", "a descriptor must be a mapping")
        known = {"site", "url_patterns", "chain", "connector",
                 "block_classes", "auth", "auth_domains",
                 "auth_cookie_names", "downloadable", "notes"}
        for key in data:
            if key not in known:
                raise SiteDescriptorError(
                    str(data.get("site") or ""), str(key),
                    f"unknown descriptor field {key!r}")
        for required in ("site", "url_patterns"):
            if required not in data:
                raise SiteDescriptorError(
                    str(data.get("site") or ""), required,
                    f"required field {required!r} is missing")
        return cls(
            site=data["site"],
            url_patterns=tuple(data["url_patterns"]),
            chain=tuple(data.get("chain") or CHAIN_STAGES),
            connector=str(data.get("connector") or ""),
            block_classes=tuple(data.get("block_classes") or ()),
            auth=str(data.get("auth") or "none"),
            auth_domains=tuple(data.get("auth_domains") or ()),
            auth_cookie_names=tuple(
                data.get("auth_cookie_names") or ()),
            downloadable=bool(data.get("downloadable", True)),
            notes=str(data.get("notes") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "site": self.site,
            "url_patterns": list(self.url_patterns),
            "chain": list(self.chain),
            "connector": self.connector,
            "block_classes": list(self.block_classes),
            "auth": self.auth,
            "auth_domains": list(self.auth_domains),
            "auth_cookie_names": list(self.auth_cookie_names),
            "downloadable": self.downloadable,
            "notes": self.notes,
        }


class SiteDescriptorRegistry:
    """The site descriptors the engine reads: a validated list,
    specific first.  Construction validates every descriptor, so a
    malformed one fails at LOAD — never at download time."""

    def __init__(
        self, descriptors: list[SiteDescriptor] | None = None
    ) -> None:
        self._descs: list[SiteDescriptor] = []
        for desc in descriptors or []:
            # __post_init__ already validated; re-validating here
            # keeps a hand-built (frozen, bypassed) instance honest.
            desc.validate()
            self._descs.append(desc)

    def register(self, desc: SiteDescriptor) -> None:
        """Add (or replace, by site) one validated descriptor."""
        desc.validate()
        self._descs = [d for d in self._descs if d.site != desc.site]
        self._descs.append(desc)

    def get(self, site: str) -> SiteDescriptor | None:
        for desc in self._descs:
            if desc.site == site:
                return desc
        return None

    def all(self) -> list[SiteDescriptor]:
        return list(self._descs)

    def for_url(self, url: str) -> SiteDescriptor | None:
        """The first descriptor whose patterns match *url* (None
        when no site descriptor claims it — the engine then runs
        the canonical chain)."""
        try:
            for desc in self._descs:
                if desc.matches(url or ""):
                    return desc
        except Exception as e:
            record_failure("site_descriptors.for_url", e)
        return None


#: SoundCloud — the FIRST site descriptor (B4): the B1–B3 behaviour
#: as data.  Public api-v2, no account: progressive MP3 preferred,
#: HLS fallback, the track's ``track_authorization`` forwarded on
#: the transcoding call (the connector does that — see
#: :mod:`nomorals.connectors.soundcloud`); blocks carry the full
#: structured vocabulary; a blocked track resolves found-but-blocked
#: and the chain moves on.  The canonical ladder order is the
#: chain: direct, kit, connector, proxy, rendered, browser.
SOUNDCLOUD_DESCRIPTOR = SiteDescriptor(
    site="soundcloud",
    url_patterns=(
        r"https?://([a-z0-9-]+\.)*soundcloud\.com/",
        r"https?://on\.soundcloud\.com/",
    ),
    chain=CHAIN_STAGES,
    connector="soundcloud",
    block_classes=BLOCK_CLASSES,
    auth="none",
    auth_domains=("soundcloud.com",),
    auth_cookie_names=("auth_token",),
    downloadable=True,
    notes="SoundCloud public api-v2: no account; progressive MP3 "
          "preferred, HLS fallback; track_authorization forwarded "
          "on transcoding calls; blocks are preview/drm/region/"
          "login/private and resolve found-but-blocked; a login-"
          "gated track retries the direct stage authenticated "
          "through the owner's soundcloud.com session (B5)",
)

#: YouTube — the SECOND site descriptor: public video by default,
#: no account, resolved by the kit rung (yt-dlp) — there is no
#: YouTube connector in :mod:`nomorals.connectors`, so the chain
#: is the canonical ladder minus the connector rung.  The resolver
#: already routes YouTube URLs through yt-dlp (kind ``"youtube"``,
#: with a single-video-id check); this descriptor is what puts the
#: download engine on the same path.  What needs a login, plainly:
#: public videos need nothing; age-restricted, sign-in-required
#: and members-only videos need the owner's signed-in youtube.com
#: session — the B5 rung resolves it (estate first, then vault)
#: when a login block is classified, and the retry runs
#: authenticated through the estate's cookies file.  The up-front
#: gate does NOT apply (``auth`` is ``"none"``): the chain runs
#: anonymously first and only a login block asks for the session.
#: The session cookies are Google's own: ``SID`` is the primary
#: session cookie, ``SAPISID`` the cookie yt-dlp uses for
#: authenticated innertube calls — an anonymous clearance cookie
#: is not a login.  Blocks: ``region`` (geo-blocked), ``login``
#: (age / sign-in wall), ``private`` (private, deleted or
#: otherwise unavailable).  No ``preview`` class — YouTube has no
#: preview-only streams — and no ``drm`` on standard video
#: streams; rented / purchased movie content is DRM-gated and
#: out of the chain's reach, named as a site error, never chased.
YOUTUBE_DESCRIPTOR = SiteDescriptor(
    site="youtube",
    url_patterns=(
        r"https?://([a-z0-9-]+\.)*youtube\.com/"
        r"(watch|shorts|live|embed|v|playlist)\b",
        r"https?://([a-z0-9-]+\.)*youtube-nocookie\.com/embed/",
        r"https?://youtu\.be/[A-Za-z0-9_-]+",
    ),
    chain=("direct", "kit", "proxy", "rendered", "browser"),
    connector="",
    block_classes=("region", "login", "private"),
    auth="none",
    auth_domains=("youtube.com",),
    auth_cookie_names=("SID", "SAPISID"),
    downloadable=True,
    notes="YouTube public video: no account, kit (yt-dlp) resolves; "
          "age-restricted / sign-in-required / members-only videos "
          "retry authenticated through the owner's youtube.com "
          "session (B5) when a login block is classified; blocks "
          "are region/login/private; no connector rung",
)

#: TikTok — the THIRD site descriptor: public video by default,
#: no account, resolved by the direct stage through yt-dlp (its
#: TikTok extractor covers video, short links and live; the kit
#: stage only fetches direct-rule URLs, so it stays silent here —
#: the chain's real yt-dlp work happens in the direct stage).
#: There is no TikTok connector in :mod:`nomorals.connectors`,
#: so the chain is the canonical ladder minus the connector
#: rung.  What needs a login, plainly: public videos need
#: nothing; a login wall (TikTok redirects the page to /login)
#: needs the owner's signed-in tiktok.com session — the B5 rung
#: resolves it (estate first, then vault) when a login block is
#: classified, and the retry runs authenticated through the
#: estate's cookies file.  The up-front gate does NOT apply
#: (``auth`` is ``"none"``): the chain runs anonymously first
#: and only a login block asks for the session.  The session
#: cookie is ``sid_tt`` — the one cookie yt-dlp's own TikTok
#: extractor reads and forwards to the API host
#: (yt_dlp/extractor/tiktok.py, ``_call_api_impl``, master,
#: fetched 2026-11-12); TikTok web's ``sessionid`` login cookie
#: is NOT consumed by yt-dlp's extractor and is therefore not
#: named, and ``msToken`` is an anti-bot token the extractor
#: never reads either.  Honest gap: no TikTok connector exists
#: to classify a login block today, so the live login-block
#: retry for TikTok is unproven in this lane — the declared
#: shape is what the B5 resolver answers (``require=True``) and
#: what a future classified block would retry with.  Blocks:
#: ``region`` (geo-blocked), ``login`` (login wall), ``private``
#: (private, deleted or otherwise unavailable).  No ``preview``
#: class — TikTok has no preview-only streams — and no ``drm``
#: on standard video streams.  Photo posts (``/photo/<id>``)
#: are genuine TikTok post URLs and are claimed, but honestly:
#: yt-dlp lands only the slideshow's background audio on a
#: photo post, never the images — the images need the
#: rendered / browser rungs (page extraction), never chased
#: through the kit.
TIKTOK_DESCRIPTOR = SiteDescriptor(
    site="tiktok",
    url_patterns=(
        r"https?://([a-z0-9-]+\.)*tiktok\.com/@[\w.-]+/"
        r"(video|photo)/\d+\b",
        r"https?://([a-z0-9-]+\.)*tiktok\.com/@[\w.-]+/live\b",
        r"https?://([a-z0-9-]+\.)*tiktok\.com/"
        r"(embed/\d+|share/video/\d+)",
        r"https?://www\.tiktokv?\.com/(embed|share/video)/\d+",
        r"https?://([a-z0-9-]+\.)*tiktok\.com/t/\w+",
        r"https?://(vm|vt)\.tiktok\.com/\w+",
        r"https?://m\.tiktok\.com/share/live/\d+",
    ),
    chain=("direct", "kit", "proxy", "rendered", "browser"),
    connector="",
    block_classes=("region", "login", "private"),
    auth="none",
    auth_domains=("tiktok.com",),
    auth_cookie_names=("sid_tt",),
    downloadable=True,
    notes="TikTok public video: no account, direct stage via "
          "yt-dlp; photo posts land audio-only via yt-dlp, "
          "images need rendered/browser; login wall retries "
          "authenticated through the owner's tiktok.com session "
          "(B5) when a login block is classified; blocks are "
          "region/login/private; no connector rung",
)

#: The descriptors the engine reads by default.
DEFAULT_SITE_DESCRIPTORS = SiteDescriptorRegistry(
    [SOUNDCLOUD_DESCRIPTOR, YOUTUBE_DESCRIPTOR, TIKTOK_DESCRIPTOR])
