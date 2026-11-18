"""Chat control for the music registry surface.

Wired into ``/music`` as ``/music sources …`` / ``/music enrich …`` /
``/music health …`` / ``/music registry-play …`` and into ``nm music``
as the ``sources`` / ``enrich`` / ``health`` / ``play-registry``
actions — one control, both surfaces. Never raises.

    /music sources [class <class>] [pack <axis> <value>] [licence <text>]
    /music sources <id> — one source, full row
    /music enrich <file-or-query> — enrichment chain (missing steps recorded)
    /music health [id] — probe one / all due sources (unknown without a pinger)
    /music registry-play <query> [--source <id>] — registry play path

Licence is shown on every result. Purchase rows resolve to a
checkout plan, never a claimed download.
"""

from __future__ import annotations

from typing import Any

import logging

_log = logging.getLogger(__name__)

__all__ = ["control_music_sources", "MUSIC_SOURCES_USAGE"]


def MUSIC_SOURCES_USAGE() -> str:
    from .registry import MUSIC_REGISTRY

    rows = len(MUSIC_REGISTRY)
    return (
        "🎵 /music sources — legal music registry (licence always shown)\n"
        f"  /music sources — all {rows} rows, grouped by class\n"
        "  /music sources class <purchase|pool|open|artist_enabled|metadata|preview|dataset>\n"
        "  /music sources pack <axis> <value> — e.g. pack nation JP\n"
        "  /music sources licence <text>\n"
        "  /music sources <id> — one source, full row\n"
        "  /music enrich <file-or-query> — fingerprint → AcoustID → MusicBrainz → one hop\n"
        "  /music health [id] — probe one / all due sources\n"
        "  /music registry-play <query> [--source <id>] — search → download → enrich\n"
        "  /music radio <place> [genre] — live stations for a place\n"
        "  /music radio play <station> | /music radio play <place> [genre]\n"
        "  /music radio globe — one dot per city, sized by broadcasters\n"
        "  /music radio grow — pull the most-voted stations in\n"
        "  /music radio health [station] | /music radio status"
    )


def control_music_sources(tail: str, context: Any = None, chat: Any = None,
                          sender_id: str = "", sender: str = "",
                          chat_key: str = "", deliver_fn: Any = None) -> str:
    """Registry surface. Never raises.

    ``deliver_fn(path, caption) -> (ok, error)`` is the host surface's
    media transport when it has one (the partner runtime passes its
    gateway sender): a fetched registry track is then DELIVERED, not
    just described.  Without one this surface returns text — the
    fetched file is named as saved on the server, never claimed sent.
    """
    try:
        return _control_music_sources(tail, context, deliver_fn)
    except Exception as exc:  # noqa: BLE001 — never raise from chat
        return f"Music sources hit an error ({exc})."


def _control_music_sources(tail: str, context: Any = None,
                           deliver_fn: Any = None) -> str:
    from . import play as play_mod
    from . import registry as reg

    raw = (tail or "").strip()
    if not raw or raw.lower() == "help":
        return MUSIC_SOURCES_USAGE()
    verb, _, rest = raw.partition(" ")
    verb = verb.lower()
    rest = rest.strip()
    if verb == "sources":
        return _sources(reg, rest)
    if verb == "enrich":
        return _enrich(rest)
    if verb == "health":
        return _health(rest, context)
    if verb == "radio":
        return _radio(rest, context, deliver_fn)
    if verb in ("registry-play", "play"):
        return _registry_play(play_mod, rest, context, deliver_fn)
    return MUSIC_SOURCES_USAGE()


def _fmt_row(source: Any) -> str:
    return (
        f"  • {source.id} — {source.name} [{source.source_class}] "
        f"licence: {source.licence} · quality: {source.quality_ceiling or '—'} · "
        f"formats: {', '.join(source.formats) or '—'} · "
        f"check every {source.health_check_hours}h"
    )


def _sources(reg: Any, rest: str) -> str:
    from .registry import all_sources, by_class, by_id, by_licence, by_pack

    if not rest:
        lines = [f"🎵 music sources — legal registry "
                 f"({len(reg.MUSIC_REGISTRY)} rows):"]
        for cls in ("purchase", "pool", "open", "artist_enabled",
                    "metadata", "preview", "dataset"):
            rows = by_class(cls)
            lines.append(f"\n{cls} ({len(rows)}):")
            lines += [_fmt_row(s) for s in rows]
        lines.append("\nLicence is shown on every row. Purchase = buy, never auto-downloaded.")
        return "\n".join(lines)
    tokens = rest.split()
    if tokens[0].lower() == "class" and len(tokens) >= 2:
        rows = by_class(tokens[1].lower())
        if not rows:
            return f"no sources in class {tokens[1]!r}."
        return "\n".join([f"🎵 {tokens[1].lower()} ({len(rows)}):"] + [_fmt_row(s) for s in rows])
    if tokens[0].lower() == "pack" and len(tokens) >= 3:
        rows = by_pack(tokens[1].lower(), tokens[2])
        if not rows:
            return f"no sources with pack {tokens[1]}={tokens[2]!r}."
        return "\n".join([f"🎵 pack {tokens[1]}={tokens[2]} ({len(rows)}):"] + [_fmt_row(s) for s in rows])
    if tokens[0].lower() == "licence" and len(tokens) >= 2:
        rows = by_licence(" ".join(tokens[1:]))
        if not rows:
            return f"no sources with licence matching {' '.join(tokens[1:])!r}."
        return "\n".join([f"🎵 licence {' '.join(tokens[1:])} ({len(rows)}):"] + [_fmt_row(s) for s in rows])
    source = by_id(tokens[0].lower())
    if source is None:
        return f"unknown source {tokens[0]!r}.\n" + MUSIC_SOURCES_USAGE()
    lines = [
        f"🎵 {source.name} ({source.id})",
        f"  class: {source.source_class}",
        f"  licence: {source.licence}",
        f"  quality: up to {source.quality_ceiling or '—'}",
        f"  formats: {', '.join(source.formats) or '—'}",
        f"  auth: {source.auth} · bulk: {source.bulk} · api: {source.api}",
        f"  volatility: {source.volatility} · health check every {source.health_check_hours}h",
        f"  etiquette: {source.etiquette}",
        f"  pack: {source.pack}",
        f"  notes: {source.notes}",
    ]
    return "\n".join(lines)


def _enrich(query: str) -> str:
    if not query:
        return "usage: /music enrich <file-or-query>"
    from .enrichment import EnrichmentChain

    chain = EnrichmentChain()
    result = chain.resolve(query)
    missing = ", ".join(result.missing) or "none"
    return (
        f"🎵 enrichment for {query!r}\n"
        f"  steps: {len(result.steps)} recorded, missing: {missing}\n"
        "  No enrichment callables wired — wire fingerprint/AcoustID/"
        "MusicBrainz callables for live resolution."
    )


def _health(rest: str, context: Any = None) -> str:
    from .health import HealthProber
    from .registry import by_id

    prober = HealthProber()
    if rest:
        source = by_id(rest.split()[0].lower())
        if source is None:
            return f"unknown source {rest.split()[0]!r}."
        status = prober.probe(source, force=True)
        lines = [
            f"🎵 health {source.id}: {status.status}",
            f"  detail: {status.detail}",
            f"  next check in ~{status.next_due_hours}h (volatility-driven)",
        ]
        # AA-57: the ONE registry's persisted health for this row,
        # when a registry is in reach — the health playback obeys.
        from . import registry_bridge
        reg = registry_bridge.music_registry_for_context(context)
        if reg is not None:
            registry_bridge.ensure_music_sources(reg)
            view = registry_bridge.source_health(reg, source.id)
            if view is not None:
                stats = view["stats"]
                lines.append(
                    f"  source registry: {view['health']} "
                    f"(score {view['score']:.2f}, served "
                    f"{stats.get('ok', 0)}, failed "
                    f"{stats.get('fail', 0)}) — playback consults this")
        return "\n".join(lines)
    results = prober.probe_all(force=True)
    ups = sum(1 for r in results if r.status == "up")
    downs = sum(1 for r in results if r.status == "down")
    unknowns = sum(1 for r in results if r.status == "unknown")
    return (
        f"🎵 health: probed {len(results)} sources — "
        f"{ups} up, {downs} down, {unknowns} unknown (no pinger wired).\n"
        "  Wire a pinger for live up/down; without one nothing is fabricated."
    )


def _radio_home(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _radio(rest: str, context: Any = None,
           deliver_fn: Any = None) -> str:
    """ /music radio — the Radio.Garden surface: stations by place,
    live samples through the play spine's delivery. Never raises."""
    from .radio import (
        StationStore,
        capture_sample,
        check_all,
        check_station,
        default_store_path,
        format_radio_reply,
        globe_dots,
        grow_from_directory,
        search_stations,
        station_by_id,
        station_by_name,
    )

    home = _radio_home(context)
    store = (StationStore(default_store_path(home)) if home
             else StationStore())
    raw = (rest or "").strip()
    words = raw.split()
    verb = words[0].lower() if words else ""
    if verb == "status":
        summary = store.health_summary()
        return (f"📻 radio index: {store.count()} stations — "
                f"{summary.get('up', 0)} up, {summary.get('down', 0)} down, "
                f"{summary.get('unchecked', 0)} unchecked, "
                f"{summary.get('unknown', 0)} unknown.\n"
                f"  store: {store.path}")
    if verb == "globe":
        dots = globe_dots(store)
        lines = [f"📻 radio globe — {len(dots)} cities, "
                 f"{store.count()} stations:"]
        for dot in dots:
            lines.append(f"  • {dot['city']}, {dot['country']} "
                         f"({dot['lat']:.2f}, {dot['lon']:.2f}) — "
                         f"{dot['count']} station(s)")
        return "\n".join(lines)
    if verb == "grow":
        result = grow_from_directory(store)
        if result.get("error"):
            return (f"📻 the radio directory did not answer "
                    f"({result['error']}) — the index is unchanged.")
        return (f"📻 grew the radio index: {result['added']} new "
                f"station(s) from {result['seen']} seen — "
                f"{store.count()} stations now.")
    if verb == "health":
        ref = " ".join(words[1:]).strip()
        if ref:
            station = (station_by_id(store, ref)
                       or station_by_name(store, ref))
            if station is None:
                return f"📻 no station matches {ref!r}."
            result = check_station(store, station.id)
            if result is None:
                return f"📻 no station matches {ref!r}."
            status, detail = result
            return (f"📻 health {station.name}: {status}\n"
                    f"  detail: {detail}")
        result = check_all(store)
        lines = [f"📻 radio health: {result['up']} up, "
                 f"{result['down']} down."]
        lines += [f"  • {failure}" for failure in result["failures"]]
        return "\n".join(lines)
    if verb == "play":
        query = " ".join(words[1:]).strip()
        if not query:
            return "usage: /music radio play <station> | <place> [genre]"
        station = (station_by_id(store, query)
                   or station_by_name(store, query))
        candidates = [station] if station is not None else None
        if candidates is None:
            parts = query.split()
            place = query
            genre = None
            if len(parts) > 1:
                tried = search_stations(
                    store, place=parts[0],
                    genre=" ".join(parts[1:]), limit=3)
                if not tried:
                    tried = search_stations(
                        store, place=" ".join(parts[1:]),
                        genre=parts[0], limit=3)
                if tried:
                    candidates = tried
            if candidates is None:
                candidates = search_stations(store, place=place,
                                             genre=genre, limit=3)
        if not candidates:
            return format_radio_reply([], place=query)
        errors: list[str] = []
        for candidate in candidates:
            from pathlib import Path as _Path

            sample = capture_sample(
                candidate, dest_dir=(
                    str(_Path(home) / "radio_samples")
                    if home else None))
            if not sample.ok:
                errors.append(f"{candidate.name}: {sample.error}")
                continue
            caption = (f"🔴 LIVE {candidate.name} — {candidate.city}, "
                       f"{candidate.country}\n"
                       f"{sample.seconds}s live sample — the full live "
                       f"stream is at {candidate.stream_url}")
            if deliver_fn is None:
                return (f"{caption}\n"
                        f"saved on the server at {sample.path} — this "
                        f"surface returns text; /play {sample.path} "
                        f"sends it here.")
            try:
                ok, err = deliver_fn(sample.path, caption)
            except Exception as exc:  # noqa: BLE001 - delivery failure is data
                return f"📻 the sample was captured but sending it failed ({exc})."
            if ok:
                return caption
            return (f"📻 the sample was captured but sending it failed "
                    f"({err}).\n{caption}")
        return ("📻 every station tried was down — "
                + "; ".join(errors))
    if verb == "list":
        rows = store.all()
        lines = [f"📻 radio index — {len(rows)} station(s):"]
        lines += [f"  • {s.name} — {s.city}, {s.country}"
                  for s in rows]
        return "\n".join(lines)
    # Anything else is "<place> [genre]".
    place = raw
    genre = None
    found = search_stations(store, place=place, limit=20)
    if not found and len(words) > 1:
        found = search_stations(store, place=" ".join(words[1:]),
                                genre=words[0], limit=20)
        if found:
            place = " ".join(words[1:])
            genre = words[0]
    return format_radio_reply(found, place=place, genre=genre or "")


def _registry_play(play_mod: Any, rest: str, context: Any = None,
                   deliver_fn: Any = None) -> str:
    if not rest:
        return "usage: /music registry-play <query> [--source <id>]"
    source_id = None
    query = rest
    if "--source" in rest:
        query, _, source_id = rest.partition("--source")
        query = query.strip()
        source_id = (source_id.strip().split() or [None])[0]
    # AA-57: play selection consults the ONE registry's persisted
    # health when a registry is in reach (db-carrying context).
    from . import registry_bridge
    reg = registry_bridge.music_registry_for_context(context)
    # Doc 7: the REAL search/download steps are wired — this branch
    # runs the same resolver/downloader pipeline as ordinary /play
    # instead of stopping at no_search / plan_only.
    from .live import live_collaborators
    search_fn, download_fn = live_collaborators(context)
    result = play_mod.play_via_registry(
        query, search_fn=search_fn, download_fn=download_fn,
        source_id=source_id, registry=reg)
    text = play_mod.registry_play_text(result)
    payload = result.download.payload \
        if result.download is not None else None
    if result.status != "ready" or not isinstance(payload, dict) \
            or not payload.get("path"):
        return text
    path = str(payload["path"])
    title = str(payload.get("title") or query)
    artist = str(payload.get("artist") or "")
    line = f"🎵 {title}"
    if artist and artist.lower() not in title.lower():
        line += f" — {artist}"
    line += (f"\nvia {result.source_name} · licence: {result.licence}"
             f" · fetched via {payload.get('served_via') or 'a wired source'}")
    if payload.get("verify_state") == "unverified":
        line += (f"\n(full-track length unverified — "
                 f"{payload.get('verify_why') or 'not probed'})")
    if deliver_fn is None:
        # This surface returns text: the file was fetched and saved
        # on the server — say exactly that, never claim a delivery.
        return (f"{text}\n{line}\n"
                f"saved on the server at {path} — this surface "
                f"returns text; /play {path} sends it here.")
    try:
        ok, err = deliver_fn(path, line)
    except Exception as exc:  # noqa: BLE001 — delivery failure is data
        ok, err = False, str(exc)
    if ok:
        return f"{line}\n(source: {result.source_name} · licence: {result.licence})"
    return (f"{text}\n{line}\n"
            f"had the file but couldn't send it: {err} "
            f"(it's saved on the server at {path})")
