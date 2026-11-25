"""``nm museum`` — the Museum census from the console.

The console twin of chat /museum (``agents/partner/runtime_meta.py:
_control_museum``) over the same ``nomorals.museum.Museum``: bare
``nm museum`` prints the census, ``nm museum <shelf>`` prints one
shelf, ``nm museum find <id>`` prints one card. The Museum is built
from the same roots as the ``museum_census`` tool (``books/tools.py:
_museum``) — ApkForge root, library, character + learn stores,
TrendOps root, downloads root, archives root, museum store root — so
the console census and the chat census can never disagree about what
exists. The character/learn stores are rooted explicitly at the
context home (the tool's defaults resolve to the same
``<home>/characters`` when the home is the Devon home), so a test
home stays a test home. ``nm museum send <id>`` is refused in words
with chat's exact wording: sending needs a live chat. ``nm museum
search/zip/unzip/attach/companions/cite`` are the console twins of
the same chat verbs, over the same Museum.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["_cmd_museum"]


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


def _museum(context: Any) -> Any:
    """The one Museum construction site on the console — the twin of
    ``books/tools.py: _museum``."""
    from ...agents.apkforge.project import apkforge_dir
    from ...books.library import Library
    from ...characters.learn import LearnStore
    from ...characters.store import CharacterStore
    from ...media.trendops.editor import DEFAULT_ROOT as _trendops_root
    from ...museum import Museum

    home = _home_path(context)
    chars = home / "characters" if home is not None else None
    settings = getattr(context, "settings", None)
    resolve = getattr(settings, "resolve", None)
    if callable(resolve):
        downloads = resolve("data/downloads")
        museum_root = resolve("museum")
        archives = (Path(getattr(settings, "workspace_dir", "")
                         or resolve("workspace")) / "archives")
    elif home is not None:
        downloads = home / "data" / "downloads"
        museum_root = home / "museum"
        archives = home / "workspace" / "archives"
    else:
        downloads = museum_root = archives = None
    return Museum(
        apk_root=str(apkforge_dir(context)),
        library=Library(context),
        char_store=(CharacterStore(data_dir=chars)
                    if chars is not None else CharacterStore()),
        learn_store=(LearnStore(data_dir=chars)
                     if chars is not None else LearnStore()),
        learn_dir=(chars / "learn") if chars is not None else None,
        trendops_root=_trendops_root,
        downloads_root=downloads,
        archives_root=archives,
        museum_root=museum_root)


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _cmd_museum(args: argparse.Namespace, context: Any) -> int:
    """``nm museum [census|<shelf>|find <id>|search <words>|send <id>|
    zip <id>|unzip <id>|attach <slug> <id> [note]|companions <slug>|
    cite <slug>]``."""
    action = getattr(args, "museum_action", None)
    rest = list(getattr(args, "museum_rest", None) or [])
    as_json = bool(getattr(args, "json", False))
    try:
        museum = _museum(context)
    except Exception as exc:  # noqa: BLE001 - museum path is the whole command
        return _usage_error(f"museum unavailable: {exc}")

    if action is None or action == "census":
        res = museum.census()
        if as_json:
            print(json.dumps(res, indent=2, default=str))
            return 0
        print(museum.render(res))
        return 0

    if action == "find":
        if not rest:
            return _usage_error(
                "usage: nm museum find <id> — nm museum lists the cards "
                "and their ids")
        card = museum.find(rest[0])
        if card is None:
            return _usage_error(
                f"no single card matches {rest[0]!r} — nm museum lists "
                "them, use the full id")
        if as_json:
            print(json.dumps({"card": card}, indent=2, default=str))
            return 0
        sub = {"artifacts": [card], "by_shelf": {card["shelf"]: 1},
               "count": 1, "errors": {}}
        print(museum.render(sub))
        return 0

    if action == "search":
        if not rest:
            return _usage_error(
                "usage: nm museum search <words> — find artifacts by "
                "name, kind, or date")
        res = museum.search(" ".join(rest))
        if as_json:
            print(json.dumps(res, indent=2, default=str))
            return 0
        print(res.get("card_text") or "no match")
        return 0

    if action == "zip":
        if not rest:
            return _usage_error(
                "usage: nm museum zip <id> — pack one card into the "
                "museum store")
        try:
            out = museum.zip_card(rest[0])
        except ValueError as exc:
            return _usage_error(f"museum zip: {exc}")
        if as_json:
            print(json.dumps(out, indent=2, default=str))
            return 0
        print(f"zipped {out.get('title')} [{out.get('card')}] — "
              f"{out.get('files')} file(s) → {out.get('zip')}")
        return 0

    if action == "unzip":
        if not rest:
            return _usage_error(
                "usage: nm museum unzip <id> — unpack a zip card into "
                "the museum store")
        try:
            out = museum.unzip_card(rest[0])
        except ValueError as exc:
            return _usage_error(f"museum unzip: {exc}")
        if as_json:
            print(json.dumps(out, indent=2, default=str))
            return 0
        text = (f"unzipped {out.get('card')} — {out.get('files')} "
                f"file(s) → {out.get('destination')}")
        if out.get("skipped_count"):
            text += (f" ({out.get('skipped_count')} unsafe member(s) "
                     "skipped)")
        print(text)
        return 0

    if action == "attach":
        if len(rest) < 2:
            return _usage_error(
                "usage: nm museum attach <book slug> <artifact id> "
                "[note]")
        try:
            out = museum.attach_companion(
                rest[0], rest[1], note=" ".join(rest[2:]))
        except ValueError as exc:
            return _usage_error(f"museum attach: {exc}")
        if as_json:
            print(json.dumps(out, indent=2, default=str))
            return 0
        text = (f"attached {out.get('artifact')} to "
                f"{out.get('book')}")
        if out.get("note"):
            text += f" — {out.get('note')}"
        print(text)
        return 0

    if action == "companions":
        if not rest:
            return _usage_error(
                "usage: nm museum companions <book slug>")
        comps = museum.companions_for_book(rest[0])
        if as_json:
            print(json.dumps({"book": rest[0], "companions": comps},
                             indent=2, default=str))
            return 0
        if not comps:
            print(f"{rest[0]} has no companion artifacts yet")
            return 0
        lines = [f"companions of {rest[0]}:"]
        for comp in comps:
            card = comp.get("card") or {}
            mark = ("missing from the museum" if comp.get("missing")
                    else card.get("title", comp.get("artifact")))
            lines.append(f"  {comp.get('artifact')} — {mark}"
                         + (f" — {comp['note']}"
                            if comp.get("note") else ""))
        print("\n".join(lines))
        return 0

    if action == "cite":
        if not rest:
            return _usage_error("usage: nm museum cite <book slug>")
        cited = museum.cite(rest[0])
        if as_json:
            print(json.dumps(cited, indent=2, default=str))
            return 0
        arts = cited.get("artifacts") or []
        missing = cited.get("missing") or []
        if not arts and not missing:
            print(f"{rest[0]} cites no artifacts yet")
            return 0
        lines = [f"{rest[0]} cites:"]
        for card in arts:
            lines.append(f"  [{card.get('id')}] {card.get('title')} — "
                         f"{card.get('kind')}")
        for ref in missing:
            lines.append(f"  {ref} — missing from the museum")
        print("\n".join(lines))
        return 0

    if action == "send":
        if not rest:
            return _usage_error(
                "usage: nm museum send <id> — nm museum lists the "
                "cards and their ids")
        print("send needs a live chat — run /museum send from "
              "a chat, not the console")
        return 0

    # Anything else is a shelf name, as in chat.
    shelf = str(action).strip().lower()
    res = museum.census()
    by_shelf = res.get("by_shelf") or {}
    if shelf not in by_shelf:
        known = ", ".join(sorted(by_shelf))
        text = (f"no shelf {shelf!r} in the museum"
                + (f" — shelves: {known}" if known
                   else " — the museum is empty"))
        if as_json:
            print(json.dumps({**res, "shelf": shelf,
                              "unknown_shelf": True}, indent=2, default=str))
            return 0
        print(text)
        return 0
    filtered = [c for c in res["artifacts"] if c.get("shelf") == shelf]
    sub = {"artifacts": filtered, "by_shelf": {shelf: len(filtered)},
           "count": len(filtered), "errors": {}}
    if as_json:
        print(json.dumps({**sub, "shelf": shelf}, indent=2, default=str))
        return 0
    print(museum.render(sub))
    return 0
