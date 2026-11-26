"""The file class, exhibited in the Museum.

Shelved files join the Museum census through the same exhibitor seam
every other system uses: ``files_exhibitor`` returns one card per
shelved file (a File's own card — same shape, same sendable rule), and
``museum_with_files`` builds the Museum every other surface builds
(ApkForge root, library, character + learn stores, TrendOps root)
with the files exhibitor registered alongside. The central
``books/tools.py: _museum`` is NOT touched here — that is the parent's
one-line fold-in at ship time; every file-class surface (tool, chat,
CLI) already reads the census through this helper, so the connection
is live in this lane, not a note about a future one.
"""

from __future__ import annotations

from typing import Any

from .store import FileStore

__all__ = ["files_exhibitor", "museum_with_files"]


def files_exhibitor(store: FileStore) -> list[dict[str, Any]]:
    """One card per shelved file. A raising file contributes nothing
    and never kills the shelf — the census core's own rule."""
    cards: list[dict[str, Any]] = []
    for f in store.all_files():
        try:
            cards.append(f.to_card())
        except Exception:  # noqa: BLE001 — one bad file, not the shelf
            continue
    return cards


def _home_path(context: Any):
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    if not home:
        return None
    from pathlib import Path

    return Path(home)


def museum_with_files(context: Any, store: FileStore | None = None,
                      ) -> Any:
    """The one Museum construction site for the file class — the twin
    of ``cmdline/commands/museum.py: _museum`` and
    ``books/tools.py: _museum``, plus the files exhibitor."""
    from ..agents.apkforge.project import apkforge_dir
    from ..books.library import Library
    from ..characters.learn import LearnStore
    from ..characters.store import CharacterStore
    from ..media.trendops.editor import DEFAULT_ROOT as _trendops_root
    from ..museum import Museum

    own = store or FileStore(context)
    home = _home_path(context)
    chars = home / "characters" if home is not None else None
    museum = Museum(
        apk_root=str(apkforge_dir(context)),
        library=Library(context),
        char_store=(CharacterStore(data_dir=chars)
                    if chars is not None else CharacterStore()),
        learn_store=(LearnStore(data_dir=chars)
                     if chars is not None else LearnStore()),
        learn_dir=(chars / "learn") if chars is not None else None,
        trendops_root=_trendops_root)
    museum.register_exhibitor(
        "files", lambda: files_exhibitor(own))
    return museum
