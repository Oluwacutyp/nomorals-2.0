"""Pass #88 — music sources go open (the owner's open-sources law).

Pass #86's report named the remaining CLOSED-SOURCE class verbatim:
"Music search sources (``media/sources.py`` chains) … still don't
discover/grow".  The music organ resolved/searched/downloaded through
hardcoded source chains that never recorded an outcome, never
health-checked, never admitted a new source, and forgot everything
on restart.

Survey truth this suite pins:

* The registry's model FITS the music class without distortion —
  every chain source is one host serving search (the same
  (nation, language, domain) key, the same outcome/health/candidate
  vocabulary) — so music sources join the SAME ``SourceRegistry``
  (option (a) in the brief): one registry, one persistence, one
  health vocabulary.  No parallel stats store.
* Several chain sources were NOT in the registry at all (bandcamp,
  archive.org, musicbrainz, openverse, wikimedia, itunes, deezer) and
  the chain's NetNaija host is ``netnaija.com`` while the registry
  seed was ``thenetnaija.net`` — a bandcamp download therefore
  recorded a *candidate* instead of crediting the serving source.
* Day-one pin: with NO learned state, the resolver's chain order is
  today's ``MUSIC_SEARCH_SOURCES`` order, byte-for-byte.
* Outcome taxonomy at the resolver seam (the sources' ``search()``
  never raises, so hit / no-hit is what is observable there):
  a hit credits the source; an operational failure (adapter/API
  error) records a failure WITH error class and feeds the circuit
  breaker; a no-result search is recorded as weak evidence
  (``no_result``) that sinks the source in score ordering but never
  stands it down — a healthy niche source must not be punished for
  not carrying an obscure track.  The health loop is the dead-source
  arbiter, exactly as in #86.
* Discovery channels that genuinely exist on the music path:
  download URLs / redirect targets and browser-page embedded hosts
  (already intaken by the downloader, #86), plus NEW here: direct
  media hosts carried by search candidates themselves (Openverse
  results point at arbitrary creator hosts).  Channels that do NOT
  exist: no chart/discovery listing is fetched by any music consumer,
  and scrape post pages' embedded hosts only surface through the
  download path's existing intake (sources.py is context-free and
  registry-free by design).

All fetches are stubbed — no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


# ── helpers ───────────────────────────────────────────────────────────


def _kv(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    return KVStore(db), db


def _seeded(store=None):
    from nomorals.media.source_registry import _seed

    return _seed(store=store)


def _ctx(tmp_path, db=None):
    return SimpleNamespace(
        db=db,
        settings=SimpleNamespace(home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={}, tools=None)


def _ok_fetcher(url):
    return 200, b"<html><body>a real page with real content</body></html>"


def _dead_fetcher(url):
    return 404, b"gone"


def _cand(source, title="Song", artist="Singer", url="", direct_url=""):
    from nomorals.media.sources import SourceCandidate

    return SourceCandidate(source=source, title=title, artist=artist,
                           url=url, direct_url=direct_url)


#: every sources.py search class the resolver drives, patched to
#: "no results" — the chain then walks its full length.
_SEARCH_CLASSES = (
    "NetNaijaSource", "NotJustOkSource", "TooXclusiveSource",
    "BandcampSource", "ArchiveOrgSource", "MusicBrainzSource",
    "OpenverseSource", "WikimediaSource", "ITunesSource",
    "DeezerSource", "BoomplaySource",
)


def _patch_all_searches(**overrides):
    """Patch every chain search to [] except the named overrides
    (class name -> return value).  Returns a list of patchers."""
    patchers = []
    for cls_name in _SEARCH_CLASSES:
        ret = overrides.get(cls_name, [])
        patchers.append(mock.patch(
            f"nomorals.media.sources.{cls_name}.search",
            return_value=ret))
    return patchers


class _start_all:
    def __init__(self, patchers):
        self._patchers = patchers

    def __enter__(self):
        for p in self._patchers:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patchers):
            p.stop()
        return False


def _resolver(ctx, reg):
    from nomorals.media.resolver import SourceResolver

    return SourceResolver(ctx, source_registry=reg)


def _attempt_names(result):
    return [a.split(" — ")[0] for a in result.attempts]


# ── (1) the music class joins the SAME registry ───────────────────────


def test_every_music_chain_source_is_a_registered_seed():
    from nomorals.media.sources import (
        MUSIC_SEARCH_SOURCES, MUSIC_SOURCE_DOMAINS)

    reg = _seeded()
    st = reg.status()
    origins = {s["domain"]: s["origin"] for s in st["sources"]}
    for strategy in MUSIC_SEARCH_SOURCES:
        domain = MUSIC_SOURCE_DOMAINS[strategy]
        if domain is None:
            continue  # workspace-scan: the owner's own files, no host
        desc = reg.for_domain(domain)
        assert desc is not None, f"{strategy} ({domain}) not registered"
        assert origins[domain] == "seed", (strategy, origins[domain])


def test_music_source_domains_cover_the_whole_chain():
    from nomorals.media.sources import (
        MUSIC_SEARCH_SOURCES, MUSIC_SOURCE_DOMAINS, SOURCE_CHAIN)

    assert set(MUSIC_SOURCE_DOMAINS) >= set(MUSIC_SEARCH_SOURCES)
    for chain_name, _inst in SOURCE_CHAIN:
        assert chain_name in MUSIC_SOURCE_DOMAINS
    # the map's domains are the hosts the sources actually fetch
    # (evidence: _BASE / API hosts in sources.py), incl. the census
    # correction — the chain's NetNaija is netnaija.com, not the
    # download-blog seed thenetnaija.net.
    assert MUSIC_SOURCE_DOMAINS["netnaija-search"] == "netnaija.com"
    assert MUSIC_SOURCE_DOMAINS["deezer-search"] == "api.deezer.com"
    assert MUSIC_SOURCE_DOMAINS["openverse-search"] == "api.openverse.org"
    assert MUSIC_SOURCE_DOMAINS["wikimedia-search"] == \
        "commons.wikimedia.org"


def test_music_seeds_do_not_change_the_extractor_view():
    # The new seeds declare NO extraction facts, so the #76/#87
    # extractor view (a pure derivation) is byte-identical.
    from nomorals.media.download_kit import descriptors_for_registry

    names = [d.name for d in descriptors_for_registry(_seeded())]
    assert names == [
        "netnaija", "audiomack", "boomplay", "notjustok", "tooxclusive",
        "youtube", "soundcloud", "spotify", "tiktok", "instagram",
        "x-twitter", "direct-media", "ytdlp-generic"], names


# ── (2) day-one pin: no learned state → today's chain order ───────────


def test_day_one_chain_order_is_byte_for_byte(tmp_path):
    from nomorals.media.sources import MUSIC_SEARCH_SOURCES

    reg = _seeded()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="/nonexistent"))
    with _start_all(_patch_all_searches()), \
            mock.patch("nomorals.media.resolver.probe",
                       side_effect=RuntimeError("no yt-dlp")), \
            mock.patch(
                "nomorals.media.playback.PlaybackEngine"
                "._youtube_search_id",
                side_effect=RuntimeError("no yt-dlp")):
        out = _resolver(ctx, reg).resolve("xyqz totally unknown 999")
    assert not out.ok
    names = _attempt_names(out)
    chain = [n for n in names if n in MUSIC_SEARCH_SOURCES]
    assert chain == list(MUSIC_SEARCH_SOURCES), names


# ── (3) outcome recording on the real search/resolve paths ────────────


def test_search_hit_credits_the_serving_source(tmp_path):
    reg = _seeded()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="/nonexistent"))
    hit = _cand("bandcamp", url="https://someartist.bandcamp.com/track/x")
    with _start_all(_patch_all_searches(BandcampSource=[hit])):
        out = _resolver(ctx, reg).resolve("some song")
    assert out.ok and out.kind == "bandcamp"
    bandcamp = reg.for_domain("bandcamp.com")
    st = reg.stats_for(bandcamp)
    assert st["ok"] == 1 and st["fail"] == 0, st
    # sources tried before the hit recorded a no-result — weak
    # evidence only: counted, but no health damage, no stand-down.
    netnaija = reg.for_domain("netnaija.com")
    st = reg.stats_for(netnaija)
    assert st["no_result"] == 1 and st["fail"] == 0, st
    assert reg.health_for(netnaija) == "unknown"
    pristine = _seeded()
    assert reg.score(netnaija) < pristine.score(
        pristine.for_domain("netnaija.com"))


def test_operational_failure_records_error_class_and_stands_down(
        tmp_path):
    from nomorals.media.sources import MUSIC_SEARCH_SOURCES

    reg = _seeded()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="/nonexistent"))
    broken_sc = SimpleNamespace(
        search_tracks=mock.Mock(
            side_effect=RuntimeError("api-v2 rotated client id")))
    resolver = SourceResolver = None
    from nomorals.media.resolver import SourceResolver
    resolver = SourceResolver(ctx, soundcloud=broken_sc,
                              source_registry=reg)
    with _start_all(_patch_all_searches()), \
            mock.patch(
                "nomorals.media.playback.PlaybackEngine"
                "._youtube_search_id",
                side_effect=RuntimeError("no yt-dlp")):
        for _ in range(3):
            out = resolver.resolve("some song")
            assert not out.ok
    sc = reg.for_domain("soundcloud.com")
    st = reg.stats_for(sc)
    assert st["fail"] == 3 and st["last_error_class"] == "search_failed", st
    assert reg.health_for(sc) == "stood-down"
    # stood down = demoted out of the active ordering: on the next
    # resolve it is attempted only after every active source.
    with _start_all(_patch_all_searches()), \
            mock.patch(
                "nomorals.media.playback.PlaybackEngine"
                "._youtube_search_id",
                side_effect=RuntimeError("no yt-dlp")):
        out = resolver.resolve("some song")
    names = [n for n in _attempt_names(out) if n in MUSIC_SEARCH_SOURCES]
    assert names.index("soundcloud-search") > names.index("deezer-search")
    assert names[-1] == "soundcloud-search" or \
        names.index("soundcloud-search") > names.index("spotify-search")


def test_sweep_records_outcomes_per_source(tmp_path):
    reg = _seeded()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="/nonexistent"))
    hit = _cand("bandcamp", url="https://a.bandcamp.com/track/x")
    with _start_all(_patch_all_searches(BandcampSource=[hit])), \
            mock.patch(
                "nomorals.media.playback.PlaybackEngine"
                "._youtube_search_many", return_value=[]):
        cands, attempts = _resolver(ctx, reg).search_candidates_detailed(
            "some song")
    assert [c.kind for c in cands] == ["bandcamp"]
    st = reg.stats_for(reg.for_domain("bandcamp.com"))
    assert st["ok"] == 1, st
    st = reg.stats_for(reg.for_domain("netnaija.com"))
    assert st["no_result"] == 1 and st["fail"] == 0, st


def test_music_download_credits_the_chain_source_not_a_candidate(
        tmp_path):
    # Before #88 a bandcamp download intaked a CANDIDATE (the host
    # was unknown to the registry); now the serving chain source is
    # registered, so the download credits it (#86 path, by domain).
    from nomorals.media.downloader import MediaDownloader

    reg = _seeded()
    dl = MediaDownloader(_ctx(tmp_path), source_registry=reg)

    def landed(url, dest, **kw):
        p = Path(dest) / "song.mp3"
        p.write_bytes(b"ID3" + b"y" * 32)
        return {"url": url, "path": str(p),
                "bytes": p.stat().st_size, "title": "Song",
                "extractor": "fake", "seconds": 0.1}

    with mock.patch("nomorals.tools.media.download",
                    side_effect=landed):
        report = dl.download("https://someartist.bandcamp.com/track/x",
                             dest_dir=tmp_path)
    assert report.ok, report.summary()
    st = reg.stats_for(reg.for_domain("bandcamp.com"))
    assert st["ok"] == 1, st
    assert reg.candidates() == []


# ── (4) discovery intake on the music path ────────────────────────────


def test_search_candidate_hosts_are_noted_as_untrusted_candidates(
        tmp_path):
    reg = _seeded()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="/nonexistent"))
    hit = _cand("openverse", title="Jam", artist="Indie",
                url="https://media.rarelabel.example/jam.mp3",
                direct_url="https://media.rarelabel.example/jam.mp3")
    with _start_all(_patch_all_searches(OpenverseSource=[hit])), \
            mock.patch(
                "nomorals.media.playback.PlaybackEngine"
                "._youtube_search_many", return_value=[]):
        cands, _attempts = _resolver(
            ctx, reg).search_candidates_detailed("jam")
    assert cands and cands[0].kind == "openverse"
    cand_domains = [c.domain for c in reg.candidates()]
    assert cand_domains == ["media.rarelabel.example"], cand_domains
    # untrusted: it is NOT a source and serves nothing
    assert reg.for_domain("media.rarelabel.example") is None
    meta = reg.status()["candidates"]
    assert meta and meta[0]["via"] == "music-search-result", meta


def test_candidate_serves_nothing_until_promoted_then_grown_serves(
        tmp_path):
    from nomorals.media import sources as sources_mod

    reg = _seeded()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="/nonexistent"))
    grown_hit = _cand("grown:freshbeats.example.org", title="Fresh",
                      url="https://freshbeats.example.org/posts/1")
    assert reg.note_candidate("https://freshbeats.example.org/song",
                              via="test") is True
    resolver = _resolver(ctx, reg)
    # not yet promoted: even though the grown source COULD answer,
    # the chain never consults it — candidates serve nothing.
    with _start_all(_patch_all_searches()), \
            mock.patch.object(sources_mod.DiscoveredBlogSource, "search",
                              return_value=[grown_hit]), \
            mock.patch(
                "nomorals.media.playback.PlaybackEngine"
                "._youtube_search_id",
                side_effect=RuntimeError("no yt-dlp")):
        out = resolver.resolve("fresh beats")
    assert not out.ok
    assert not any(n.startswith("grown:") for n in _attempt_names(out))
    # health loop promotes it → the chain grows by one serving source
    summary = reg.verify_health(fetcher=_ok_fetcher)
    assert summary["promoted"] == 1, summary
    with _start_all(_patch_all_searches()), \
            mock.patch.object(sources_mod.DiscoveredBlogSource, "search",
                              return_value=[grown_hit]), \
            mock.patch(
                "nomorals.media.playback.PlaybackEngine"
                "._youtube_search_id",
                side_effect=RuntimeError("no yt-dlp")):
        out = resolver.resolve("fresh beats")
    assert out.ok and out.kind == "grown:freshbeats.example.org", out
    assert "grown:freshbeats.example.org" in _attempt_names(out)
    # and the grown source earned the credit for serving
    st = reg.stats_for(reg.for_domain("freshbeats.example.org"))
    assert st["ok"] == 1, st


# ── (5) health loop + the existing scheduler job cover music ──────────


def test_health_loop_verifies_the_music_class():
    reg = _seeded()
    summary = reg.verify_health(fetcher=_ok_fetcher)
    assert summary["checked"] >= len(reg)
    music = reg.status()["music"]
    assert music["chain"], music
    assert all(row["health"] == "verified-live"
               for row in music["chain"]), music["chain"]
    # and the dead-fetcher loop stands music sources down by the
    # same vocabulary (3 consecutive failures → stood-down)
    reg2 = _seeded()
    for _ in range(3):
        reg2.verify_health(fetcher=_dead_fetcher)
    music2 = reg2.status()["music"]
    bandcamp_row = [r for r in music2["chain"]
                    if r["domain"] == "bandcamp.com"][0]
    assert bandcamp_row["health"] in ("stood-down", "disabled",
                                      "dead"), bandcamp_row


def _sched_ctx(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.tools.registry import ToolRegistry

    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(
        db=db,
        settings=SimpleNamespace(home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={}, tools=None)
    registry = ToolRegistry(ctx)
    registry.register_builtins()
    ctx.tools = registry
    return ctx, registry


def test_existing_health_job_covers_music_sources(tmp_path, monkeypatch):
    # No new scheduler binding: the #86 "download source health" job
    # binds download_sources(action=health) over the ONE registry —
    # registering the music class in that registry extends the job.
    from nomorals.agents.scheduler import Scheduler
    from nomorals.media import source_registry as sr
    from nomorals.media.source_jobs import ensure_source_health_jobs

    monkeypatch.setattr(sr, "_default_health_fetch", _ok_fetcher)
    ctx, _registry = _sched_ctx(tmp_path)
    sched = Scheduler(ctx)
    sched.notifier = mock.MagicMock()
    ensure_source_health_jobs(sched)
    row = [j for j in sched.list_jobs()
           if j["name"] == "download source health"][0]
    payload = json.loads(ctx.db.query_one(
        "SELECT payload FROM schedule_jobs WHERE name = ?",
        ("download source health",))["payload"] or "{}")
    assert payload == {"tool": "download_sources",
                       "args": {"action": "health"}}
    out = sched.run_now(row["id"])
    assert out["ok"] is True, out
    from nomorals.media.downloader import MediaDownloader
    music = MediaDownloader(ctx).source_registry().status()["music"]
    assert music["chain"] and all(
        r["health"] == "verified-live" for r in music["chain"]), music


# ── (6) persistence + status surface ──────────────────────────────────


def test_music_outcomes_persist_across_restart(tmp_path):
    kv, _db = _kv(tmp_path)
    reg = _seeded(store=kv)
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="/nonexistent"))
    hit = _cand("bandcamp", url="https://a.bandcamp.com/track/x")
    with _start_all(_patch_all_searches(BandcampSource=[hit])):
        out = _resolver(ctx, reg).resolve("some song")
    assert out.ok
    reg2 = _seeded(store=kv)  # restart: fresh registry, same store
    st = reg2.stats_for(reg2.for_domain("bandcamp.com"))
    assert st["ok"] == 1, st
    st = reg2.stats_for(reg2.for_domain("netnaija.com"))
    assert st["no_result"] == 1, st
    music = reg2.status()["music"]
    row = [r for r in music["chain"] if r["domain"] == "bandcamp.com"]
    assert row and row[0]["ok"] == 1 and row[0]["origin"] == "seed", row


def test_status_surface_shows_the_music_class(tmp_path):
    reg = _seeded()
    reg.note_candidate("https://freshbeats.example.org/song", via="test")
    st = reg.status()
    music = st["music"]
    assert music["candidates_pending"] == 1
    strategies = {r["strategy"] for r in music["chain"]}
    assert "bandcamp-search" in strategies
    assert "netnaija-search" in strategies
    from nomorals.media.source_registry import render_status_text
    text = render_status_text(st)
    assert "music sources" in text.lower(), text
    assert "bandcamp.com" in text, text
    # the chat surface (/download status) renders the same registry
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    ctx, _ = _sched_ctx(tmp_path)
    obj = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    obj.context = ctx
    chat_text = obj._control_download("status")
    assert "music sources" in chat_text.lower(), chat_text
