"""``nm museum`` — the Museum census from the console.

The console twin of chat /museum (``agents/partner/runtime_meta.py:
_control_museum``) over the same ``nomorals.museum.Museum``: bare
``nm museum`` prints the census, ``nm museum <shelf>`` prints one
shelf, ``nm museum find <id>`` prints one card. The Museum is built
from the same roots as the ``museum_census`` tool (``books/tools.py:
_museum``) — ApkForge root, library, character + learn stores,
TrendOps root — so the console census and the chat census can never
disagree about what exists. The character/learn stores are rooted
explicitly at the context home (the tool's defaults resolve to the
same ``<home>/characters`` when the home is the Devon home), so a
test home stays a test home. ``nm museum send <id>`` is refused in
words with chat's exact wording: sending needs a live chat.
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
    return Museum(
        apk_root=str(apkforge_dir(context)),
        library=Library(context),
        char_store=(CharacterStore(data_dir=chars)
                    if chars is not None else CharacterStore()),
        learn_store=(LearnStore(data_dir=chars)
                     if chars is not None else LearnStore()),
        learn_dir=(chars / "learn") if chars is not None else None,
        trendops_root=_trendops_root)


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _cmd_museum(args: argparse.Namespace, context: Any) -> int:
    """``nm museum [census|<shelf>|find <id>|send <id>]``."""
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
