"""Soundtrack tool: ``soundtrack`` — the brain's soundtrack index.

Devon asks what song plays in a film or TV scene and gets the song,
the artist, the scene it plays in and the way to hear it — straight
into her own /play chain. The tool is a thin surface over
:mod:`nomorals.music_sources.placements`; the organ owns the
catalogue, the trust model and the honest refusal when a work or a
song is not recorded. A clip can be identified first through her own
fingerprint memory, then answered from the catalogue.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability

__all__ = ["register"]


def _identify_clip(path: str) -> dict[str, Any]:
    """The production clip leg: Devon's own fingerprint memory.

    Lazy import — the tool and the audio organ are both L4, so this
    stays inside the layer.
    """
    from ..audio.fingerprint import identify

    return identify(path)


def _home_from_context(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def register(registry: Any) -> None:
    """Attach the ``soundtrack`` tool to a registry."""
    context = registry.context

    @registry.register(
        "soundtrack",
        description=(
            "What song plays in a film or TV scene — the soundtrack "
            "index. Ask for a work and an episode and get the songs, "
            "the artists, the scene each plays in, its provenance and "
            "confidence, and the /play command that hears it. A clip "
            "can be identified first through Devon's own fingerprint "
            "memory, then answered from the catalogue. An unknown work "
            "or song is refused honestly, never invented. "
            "action=<lookup|song|clip|add|vote>; lookup takes "
            "query=<work [S1E2]>; song takes track=<title> and "
            "artist=<artist>; clip takes path=<audio file>; add takes "
            "work=<title>, kind=<tv|film|game>, season, episode, "
            "scene=<scene note>, track=<title>, artist=<artist>; vote "
            "takes work=<title>, track=<title>, confirm=<true|false>. "
            "Returns ok, work, kind, season, episode, placements "
            "(each with track_title, track_artist, scene_note, "
            "provenance, confidence, verified, trust_label and "
            "play), or error."
        ),
        capability=Capability.DB_READ,
    )
    def soundtrack_tool(
        query: str = "",
        action: str = "lookup",
        track: str = "",
        artist: str = "",
        path: str = "",
        work: str = "",
        kind: str = "tv",
        season: Any = None,
        episode: Any = None,
        scene: str = "",
        confirm: bool = True,
    ) -> dict[str, Any]:
        from ..music_sources.placements import (
            Placement,
            format_placement_reply,
            format_reverse_reply,
            lookup,
            open_store,
            placements_for_clip,
        )

        store = open_store(_home_from_context(context))
        act = (action or "lookup").strip().lower()
        if act == "song":
            if not track or not str(track).strip():
                raise ValueError("soundtrack song needs a track title")
            works = store.works_for_track(
                str(track).strip(), str(artist).strip() or None)
            if not works:
                return {"ok": False, "track": str(track).strip(),
                        "artist": str(artist).strip(),
                        "error": (f"no placements recorded for "
                                  f"“{track.strip()}” yet")}
            return {"ok": True, "track": str(track).strip(),
                    "artist": str(artist).strip(), "placements": works,
                    "reply": format_reverse_reply(works)}
        if act == "clip":
            if not path or not str(path).strip():
                raise ValueError("soundtrack clip needs an audio path")
            ans = placements_for_clip(store, str(path).strip(),
                                        identify_fn=_identify_clip)
            ans["reply"] = (format_reverse_reply(ans["placements"])
                            if ans.get("ok")
                            else format_placement_reply(ans))
            return ans
        if act == "add":
            if not work or not str(work).strip() or not track:
                raise ValueError(
                    "soundtrack add needs work=<title> and track=<title>")
            try:
                season_n = int(season) if season is not None else -1
                episode_n = int(episode) if episode is not None else -1
            except (TypeError, ValueError) as err:
                raise ValueError(
                    "soundtrack add needs numeric season/episode") from err
            placement = Placement(
                work_kind=(kind or "tv").strip().lower(),
                work_title=str(work).strip(),
                season=season_n, episode=episode_n,
                scene_note=str(scene or "").strip(),
                track_title=str(track).strip(),
                track_artist=str(artist or "").strip(),
                provenance="owner", confidence=0.95)
            added = store.add(placement)
            return {"ok": True, "added": added,
                    "placement": placement.to_dict(),
                    "reply": (f"recorded: {placement.track_title} — "
                              f"{placement.track_artist} in "
                              f"{placement.work_title}"
                              + (" — already recorded" if not added else ""))}
        if act == "vote":
            if not work or not track:
                raise ValueError(
                    "soundtrack vote needs work=<title> and track=<title>")
            resolved = store.resolve_work(str(work).strip())
            if not resolved:
                return {"ok": False,
                        "error": f"“{work.strip()}” is not in the catalogue"}
            updated = store.vote(resolved[0]["work_id"], str(track).strip(),
                                 confirm=bool(confirm))
            if updated is None:
                return {"ok": False,
                        "error": (f"no community placement for "
                                  f"“{track.strip()}” in "
                                  f"{resolved[0]['work_title']} to vote on")}
            return {"ok": True, "placement": updated.to_dict()}
        if not query or not str(query).strip():
            raise ValueError(
                "soundtrack needs a query — a show or film, optionally S1E2")
        ans = lookup(store, str(query).strip())
        ans["reply"] = format_placement_reply(ans)
        return ans
