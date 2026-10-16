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
    return (
        "🎵 /music sources — legal music registry (licence always shown)\n"
        "  /music sources — all 94 rows, grouped by class\n"
        "  /music sources class <purchase|pool|open|artist_enabled|metadata|preview|dataset>\n"
        "  /music sources pack <axis> <value> — e.g. pack nation JP\n"
        "  /music sources licence <text>\n"
        "  /music sources <id> — one source, full row\n"
        "  /music enrich <file-or-query> — fingerprint → AcoustID → MusicBrainz → one hop\n"
        "  /music health [id] — probe one / all due sources\n"
        "  /music registry-play <query> [--source <id>] — search → download → enrich"
    )


def control_music_sources(tail: str, context: Any = None, chat: Any = None,
                          sender_id: str = "", sender: str = "",
                          chat_key: str = "") -> str:
    """Registry surface. Never raises."""
    try:
        return _control_music_sources(tail)
    except Exception as exc:  # noqa: BLE001 — never raise from chat
        return f"Music sources hit an error ({exc})."


def _control_music_sources(tail: str) -> str:
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
        return _health(rest)
    if verb in ("registry-play", "play"):
        return _registry_play(play_mod, rest)
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
        lines = ["🎵 music sources — legal registry (94 rows):"]
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


def _health(rest: str) -> str:
    from .health import HealthProber
    from .registry import by_id

    prober = HealthProber()
    if rest:
        source = by_id(rest.split()[0].lower())
        if source is None:
            return f"unknown source {rest.split()[0]!r}."
        status = prober.probe(source, force=True)
        return (
            f"🎵 health {source.id}: {status.status}\n"
            f"  detail: {status.detail}\n"
            f"  next check in ~{status.next_due_hours}h (volatility-driven)"
        )
    results = prober.probe_all(force=True)
    ups = sum(1 for r in results if r.status == "up")
    downs = sum(1 for r in results if r.status == "down")
    unknowns = sum(1 for r in results if r.status == "unknown")
    return (
        f"🎵 health: probed {len(results)} sources — "
        f"{ups} up, {downs} down, {unknowns} unknown (no pinger wired).\n"
        "  Wire a pinger for live up/down; without one nothing is fabricated."
    )


def _registry_play(play_mod: Any, rest: str) -> str:
    if not rest:
        return "usage: /music registry-play <query> [--source <id>]"
    source_id = None
    query = rest
    if "--source" in rest:
        query, _, source_id = rest.partition("--source")
        query = query.strip()
        source_id = (source_id.strip().split() or [None])[0]
    result = play_mod.play_via_registry(query, source_id=source_id)
    return play_mod.registry_play_text(result)
