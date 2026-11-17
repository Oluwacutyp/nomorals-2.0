"""``nm soundtrack`` — what song plays in a film or TV scene.

    nm soundtrack Stranger Things S4E4
    nm soundtrack Pulp Fiction [--json]
    nm soundtrack song "Come and Get Your Love"
    nm soundtrack clip /path/to/clip.wav
    nm soundtrack add --work Severance --kind tv --season 2 --episode 3 \\
        --track "Unknown Song" --artist "Unknown Artist" --scene "closing credits"
    nm soundtrack vote --work "The Wire" --track "Way Down in the Hole" confirm

The same catalogue chat /soundtrack uses: seeded from public
soundtrack listings (each entry labelled with its provenance and
confidence — curated entries are never presented as verified), grown
by the owner's own watching and community votes. An unknown work or
song prints the honest reason and exits 1 — never an invented entry.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

__all__ = ["_cmd_soundtrack"]


def _home(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def _print(answer: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(answer, indent=2, default=str))
    else:
        print(answer.get("reply") or answer.get("error") or "")


def _cmd_soundtrack(args: argparse.Namespace, context: Any) -> int:
    from ...music_sources.placements import (
        Placement,
        format_placement_reply,
        format_reverse_reply,
        lookup,
        open_store,
        placements_for_clip,
    )

    store = open_store(_home(context))
    words = list(getattr(args, "soundtrack_query", None) or [])
    as_json = bool(getattr(args, "json", False))
    verb = (words[0].lower() if words else "")
    try:
        if verb == "song":
            track = " ".join(words[1:]).strip()
            artist = getattr(args, "artist", None) or ""
            if not track:
                print("soundtrack: give me a track — "
                      "nm soundtrack song <track>", file=sys.stderr)
                return 2
            works = store.works_for_track(track, artist.strip() or None)
            if not works:
                print(f"no placements recorded for “{track}” yet",
                      file=sys.stderr)
                return 1
            _print({"ok": True, "track": track, "artist": artist,
                    "placements": works,
                    "reply": format_reverse_reply(works)}, as_json)
            return 0
        if verb == "clip":
            if len(words) < 2:
                print("soundtrack: give me an audio path — "
                      "nm soundtrack clip <path>", file=sys.stderr)
                return 2
            from ...audio.fingerprint import identify as _identify

            ans = placements_for_clip(store, words[1],
                                      identify_fn=_identify)
            ans["reply"] = (format_reverse_reply(ans["placements"])
                            if ans.get("ok")
                            else format_placement_reply(ans))
            _print(ans, as_json)
            return 0 if ans.get("ok") else 1
        if verb == "add":
            work = getattr(args, "work", None) or ""
            track = getattr(args, "track", None) or ""
            if not work or not track:
                print("soundtrack: add needs --work and --track",
                      file=sys.stderr)
                return 2
            placement = Placement(
                work_kind=(getattr(args, "kind", None) or "tv").strip().lower(),
                work_title=str(work).strip(),
                season=int(getattr(args, "season", None) or -1),
                episode=int(getattr(args, "episode", None) or -1),
                scene_note=str(getattr(args, "scene", None) or "").strip(),
                track_title=str(track).strip(),
                track_artist=str(getattr(args, "artist", None) or "").strip(),
                provenance="owner", confidence=0.95)
            added = store.add(placement)
            _print({"ok": True, "added": added,
                    "placement": placement.to_dict(),
                    "reply": (f"recorded: {placement.track_title} — "
                              f"{placement.track_artist} in "
                              f"{placement.work_title}"
                              + (" — already recorded" if not added else ""))},
                   as_json)
            return 0
        if verb == "vote":
            work = getattr(args, "work", None) or ""
            track = getattr(args, "track", None) or ""
            choice = (words[-1].lower() if words else "")
            if not work or not track or choice not in ("confirm", "deny"):
                print("soundtrack: vote needs --work, --track and "
                      "confirm|deny", file=sys.stderr)
                return 2
            resolved = store.resolve_work(str(work).strip())
            if not resolved:
                print(f"“{work.strip()}” is not in the catalogue",
                      file=sys.stderr)
                return 1
            updated = store.vote(resolved[0]["work_id"], str(track).strip(),
                                 confirm=(choice == "confirm"))
            if updated is None:
                print(f"no community placement for “{track.strip()}” in "
                      f"{resolved[0]['work_title']} to vote on",
                      file=sys.stderr)
                return 1
            _print({"ok": True, "placement": updated.to_dict(),
                    "reply": (f"{updated.track_title} — "
                              f"{updated.trust_label()}")}, as_json)
            return 0
        query = " ".join(words).strip()
        if not query:
            print("soundtrack: give me a show or a film — "
                  "nm soundtrack <show|film> [S1E2]", file=sys.stderr)
            return 2
        ans = lookup(store, query)
        ans["reply"] = format_placement_reply(ans)
        _print(ans, as_json)
        return 0 if ans.get("ok") else 1
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        print(f"soundtrack: {exc}", file=sys.stderr)
        return 1
