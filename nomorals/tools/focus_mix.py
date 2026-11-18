"""Focus-mix tool: the MusicForProgramming rung for the brain.

``"put on a coding mix"`` over the focus-mix organ — not a
standalone player. The organ (:mod:`nomorals.music_sources.focus_mixes`)
owns the mixes, the series RSS refresh, the assembly from the
owner's library and the session link; this module is only the
surface between the brain and the organ, plus the DJ-engine planner
adapter — the one place allowed to reach down into ``media/``
(layer 4) for stem-aware, drop-aware sequencing
(``plan_long_mix``) and the owner's taste log
(``dj_engine.read_feedback_log``).

The taste model is the owner's: ``dj_planner`` re-ranks candidates
from the same feedback rows the DJ engine learns from — liked /
kept tracks count for, skipped count against — and the mix records
where its tracks came from, never invented ones.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..music_sources.focus_mixes import (
    MOOD_TAGS,
    FocusMix,
    FocusMixStore,
    FocusTrack,
    assemble_mix,
    find_mix,
    format_mix_reply,
    refresh_from_series,
    render_mix,
    start_focus_session_with_mix,
)

__all__ = ["dj_planner", "read_owner_taste", "register"]


def _norm_key(text: str) -> str:
    """Normalise a title/artist for best-effort taste joins."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _store_of(context: Any) -> FocusMixStore:
    home = _home_of(context)
    return FocusMixStore(home) if home else FocusMixStore()


def _timer_of(context: Any) -> Any:
    from ..focus.sessions import FocusTimer

    home = _home_of(context)
    if home is None:
        return FocusTimer()
    return FocusTimer(str(Path(home) / "data" / "focus.json"))


def _library_of(context: Any) -> Any | None:
    """The owner's music library, or None when no database is wired."""
    if context is None or getattr(context, "db", None) is None:
        return None
    try:
        from ..media.library import MusicLibrary

        return MusicLibrary(context)
    except Exception:  # noqa: BLE001 - no library is a fact, not a crash
        return None


def read_owner_taste(feedback_path: str | Path | None = None
                     ) -> list[dict[str, Any]]:
    """The owner's DJ feedback rows (liked / kept / skipped / played)."""
    from ..media.dj_engine import read_feedback_log

    return read_feedback_log(feedback_path)


def dj_planner(tracks: list[FocusTrack], target_minutes: float,
               feedback_rows: list[dict[str, Any]] | None = None
               ) -> tuple[list[FocusTrack], dict[str, Any]]:
    """Sequence tracks with the real DJ engine (``plan_long_mix``).

    Converts to :class:`LongTrack` descriptors, re-ranks with a
    :class:`TasteProfile` built from the owner's feedback rows
    (``kept`` / ``liked`` count for, ``skipped`` against), plans the
    long mix — energy arc, stem-aware transitions, drop-aware —
    and maps the plan back onto the input tracks. Returns
    ``(ordered_tracks, plan_dict)``; empty input plans nothing.
    """
    from ..media.dj_long import TasteProfile, plan_long_mix

    if not tracks:
        return [], {}
    profile = TasteProfile()
    # Feedback rows name a track by id *or* by title (the DJ log
    # carries whichever the caller passed) — resolve either to the
    # candidate's id, so a liked title actually steers the plan.
    by_title: dict[str, str] = {}
    for track in tracks:
        by_title.setdefault(_norm_key(track.title), track.id)
        by_title.setdefault(
            _norm_key(f"{track.artist} {track.title}"), track.id)
    by_id = {t.id: t for t in tracks}
    for row in (feedback_rows or []):
        if not isinstance(row, dict):
            continue
        event = str(row.get("event", "")).strip().lower()
        if event not in ("like", "liked", "kept", "skip", "skipped",
                         "rewind"):
            continue
        track_id = str(row.get("track_id", "") or "")
        if track_id not in by_id:
            resolved = by_title.get(_norm_key(track_id), "")
            if not resolved:
                resolved = by_title.get(
                    _norm_key(str(row.get("title", "") or "")), "")
            track_id = resolved
        if not track_id:
            continue
        if event in ("like", "liked", "kept"):
            profile.record("like", track_id)
        elif event in ("skip", "skipped"):
            profile.record("skip", track_id)
        else:
            profile.record("rewind", track_id)
    descriptors = [{
        "id": t.id, "title": t.title, "artist": t.artist,
        "bpm": t.bpm, "key": t.key, "energy": t.energy,
        "duration_s": t.duration_s, "tags": list(t.tags),
    } for t in tracks]
    plan = plan_long_mix(descriptors, target_minutes=float(target_minutes),
                         taste=profile)
    ordered = [by_id[t.id] for t in plan.tracks if t.id in by_id]
    return ordered, plan.to_dict()


def _parse_create_tracks(raw: str) -> list[FocusTrack]:
    """``"Title — Artist; Title2 — Artist2"`` → FocusTracks.

    The owner's own curation: durations default to 3 minutes and
    are honest placeholders — the track is identified by title and
    artist, never by an invented length.
    """
    tracks: list[FocusTrack] = []
    for chunk in str(raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if " — " in chunk:
            title, artist = chunk.split(" — ", 1)
        elif " - " in chunk:
            title, artist = chunk.split(" - ", 1)
        else:
            title, artist = chunk, ""
        title, artist = title.strip(), artist.strip()
        if not title:
            continue
        tracks.append(FocusTrack(
            id="ft-custom-" + hashlib.sha1(
                f"{title}|{artist}|{len(tracks)}".encode()
            ).hexdigest()[:12],
            title=title, artist=artist, source_id="custom"))
    return tracks


def register(registry: Any) -> None:
    """Attach the ``focus_mix`` tool."""
    context = registry.context

    @registry.register(
        "focus_mix",
        description=(
            "Focus mixes for deep work (MusicForProgramming core): "
            "action=list lists the mixes she holds; action=show takes "
            "id=<mix id, title or 'episode N'> and renders the "
            "tracklist; action=play takes id=<mix> and starts a focus "
            "session with that mix attached, recording the session ↔ "
            "mix link; action=assemble builds a new mix from the "
            "owner's library and listening history (mood=<darker | "
            "lighter | deep | driving | calm | ambient>, "
            "minutes=<target length>) sequenced by the DJ engine; "
            "action=refresh pulls the live Music For Programming RSS "
            "feed and records new episodes as mixes; action=create "
            "takes title=<name> and tracks=<\"Title — Artist; ...\"> "
            "and stores an owner-curated mix; action=remove takes "
            "id=<mix>; action=status reports the running focus "
            "session and its mix, if any."
        ),
        capability=Capability.MEM_WRITE,
        result_protocol="dict_ok",
    )
    def _focus_mix_tool(**kwargs: Any) -> dict[str, Any]:
        action = str(kwargs.get("action", "list") or "list").lower()
        store = _store_of(context)
        if action == "list":
            rows = [{
                "id": m.id, "title": m.title, "curator": m.curator,
                "tracks": len(m.tracks),
                "duration": m.duration_s, "source": m.source,
                "sessions_played": m.sessions_played,
            } for m in store.list()]
            return {"ok": True, "mixes": rows, "count": len(rows)}
        if action == "show":
            mix = find_mix(store, str(kwargs.get("id", "") or ""))
            if mix is None:
                return {"ok": False,
                        "error": "no focus mix matches that id or title"}
            return {"ok": True, "mix": mix.to_dict(),
                    "reply": render_mix(mix)}
        if action == "play":
            mix = find_mix(store, str(kwargs.get("id", "") or ""))
            if mix is None:
                return {"ok": False,
                        "error": "no focus mix matches that id or title"}
            try:
                timer = _timer_of(context)
                result = start_focus_session_with_mix(
                    timer, store, mix)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True,
                    "session": result["session"].to_dict(),
                    "mix": mix.to_dict(),
                    "play": result["play"],
                    "reply": format_mix_reply(
                        mix, result["session"], result["play"])}
        if action == "assemble":
            mood = str(kwargs.get("mood", "") or "").strip().lower()
            tags = dict(MOOD_TAGS[mood]) if mood in MOOD_TAGS else {}
            if mood and mood not in MOOD_TAGS:
                return {"ok": False,
                        "error": f"unknown mood {mood!r} — darker | "
                                 "lighter | deep | driving | calm | "
                                 "ambient"}
            try:
                minutes = float(kwargs.get("minutes", 60) or 60)
            except (TypeError, ValueError):
                return {"ok": False,
                        "error": "assemble minutes=<n> must be a number"}
            try:
                mix = assemble_mix(
                    store, library=_library_of(context),
                    feedback_rows=read_owner_taste(),
                    tags=tags, target_minutes=minutes,
                    title=str(kwargs.get("title", "") or "") or None,
                    planner=dj_planner)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "mix": mix.to_dict(),
                    "reply": render_mix(mix)}
        if action == "refresh":
            result = refresh_from_series(store)
            if result.get("error"):
                return {"ok": False, "error": result["error"],
                        "added": 0, "total": result.get("total", 0)}
            return {"ok": True, "added": result["added"],
                    "total": result["total"]}
        if action == "create":
            title = str(kwargs.get("title", "") or "").strip()
            if not title:
                return {"ok": False,
                        "error": "create needs title=<name>"}
            tracks = _parse_create_tracks(
                str(kwargs.get("tracks", "") or ""))
            if not tracks:
                return {"ok": False,
                        "error": "create needs tracks=<\"Title — "
                                 "Artist; ...\">"}
            mix = FocusMix(
                id="fm-custom-" + hashlib.sha1(
                    title.encode()).hexdigest()[:12],
                title=title, curator="owner", tracks=tracks,
                source="custom")
            store.add(mix)
            return {"ok": True, "mix": mix.to_dict(),
                    "reply": render_mix(mix)}
        if action == "remove":
            mix_id = str(kwargs.get("id", "") or "").strip()
            if not mix_id or not store.remove(mix_id):
                return {"ok": False,
                        "error": "no focus mix matches that id"}
            return {"ok": True, "removed": mix_id}
        if action == "status":
            timer = _timer_of(context)
            active = timer.active()
            if active is None:
                return {"ok": True, "running": False,
                        "mixes": len(store.list())}
            return {"ok": True, "running": True,
                    "session": active.to_dict(),
                    "mix_id": active.mix_id,
                    "mix_title": active.mix_title,
                    "mixes": len(store.list())}
        return {"ok": False,
                "error": "unknown action — list | show | play | "
                         "assemble | refresh | create | remove | status"}
