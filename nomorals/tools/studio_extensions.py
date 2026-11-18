"""Studio-extensions tool: ``studio_extensions`` — the brain's
answer to "install this theme" / "what extensions are on".

One tool, action-dispatched, over :mod:`nomorals.studio`; the organ
owns the contract, the store and the unpacking — this module is only
the surface.

``action=list`` lists what is installed, what is switched on for
the Studio, and which theme the Studio is wearing.
``action=install`` installs a package from a file (``path``) or a
URL (``url`` with a pinned ``sha256`` and ``size`` — the bytes are
checked against the digest before they count).
``action=enable`` / ``action=disable`` switches an extension on or
off for the Studio. ``action=uninstall`` removes one everywhere.
``action=theme`` chooses the Studio's look (``ext_id``, or
``"none"`` to return to its own look). ``action=themes`` lists the
installed themes; ``action=panels`` lists the panels enabled for
one Studio page (``page``, default ``photo``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..studio.panel import PanelFile
from ..studio.store import StudioStore

__all__ = ["parse_studio_request", "register"]

_ZIP_RE = re.compile(r"([^\s\"']+\.zip)", re.IGNORECASE)


def parse_studio_request(
        text: str, known: list[tuple[str, str]]
        ) -> tuple[str, str] | None:
    """Natural phrasing -> (verb, ref) for the Studio extensions,
    or None.

    Matches only when the text names the Studio's extensions or a
    theme, or an extension package to install — a bare "install
    this" with nothing to install it from is not a match, so the
    hook never steals an unrelated question.
    """
    low = (text or "").lower()
    if "studio" not in low and "extension" not in low \
            and "theme" not in low:
        return None
    match = _ZIP_RE.search(text or "")
    if match is not None and ("install" in low or "extension" in low):
        return ("install", match.group(1))
    for ident, title in known:
        for name in (title, ident):
            if name and name.lower() in low:
                if "theme" in low:
                    return ("theme", ident)
                return ("show", ident)
    if "studio theme" in low or "studio extensions" in low:
        return ("list", "")
    return None


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _store(context: Any) -> StudioStore:
    return StudioStore(home=_home_of(context))


def _describe(installed: Any, state: Any) -> dict:
    return {"id": str(installed.id),
            "version": str(installed.version),
            "kind": str(installed.kind),
            "name": installed.name,
            "tier": str(installed.tier),
            "apps": list(installed.apps),
            "enabled_for_studio": state.is_enabled(
                "studio", str(installed.id))}


def register(registry: Any) -> None:
    """Attach the ``studio_extensions`` tool."""
    context = registry.context

    @registry.register(
        "studio_extensions",
        description=(
            "Devon Studio's extensions (the craft-extensions "
            "contract): install, list, enable, disable and remove "
            "themes, plug-ins and resource packs for the Studio, "
            "and choose the Studio's look. action=list lists what "
            "is installed and which theme the Studio wears. "
            "action=install installs a package from a file (path) "
            "or a URL (url with a pinned sha256 and size). "
            "action=enable|disable switches an extension on or off "
            "for the Studio. action=uninstall removes one "
            "everywhere. action=theme chooses the look (ext_id, or "
            "'none' for the Studio's own look). action=themes lists "
            "the installed themes; action=panels lists the panels "
            "enabled for one Studio page (page, default photo). "
            "Packages live under <home>/studio/extensions/."
        ),
        capability=Capability.MEDIA,
    )
    def studio_extensions_tool(action: str = "list", ext_id: str = "",
                               path: str = "", url: str = "",
                               sha256: str = "", size: int = 0,
                               page: str = "") -> dict[str, Any]:
        store = _store(context)
        verb = (action or "list").strip().lower()
        if verb == "list":
            state = store.state()
            active = store.active_theme("studio")
            return {"ok": True,
                    "installed": [_describe(item, state)
                                  for item in state.installed],
                    "count": len(state.installed),
                    "active_theme": (active.id if active is not None
                                     else None),
                    "generation": state.generation}
        if verb == "install":
            src = (path or "").strip()
            if src:
                record = store.install_file(Path(src).expanduser())
            else:
                link = (url or "").strip()
                if not link:
                    raise ValueError(
                        "studio_extensions install needs path=<zip> "
                        "or url=<zip url> with sha256= and size=")
                if not sha256 or not size:
                    raise ValueError(
                        "studio_extensions install from a URL needs "
                        "the pinned sha256= and size=")
                record = store.install_url(link, sha256=sha256,
                                           size=int(size))
            state = store.state()
            result = _describe(record, state)
            result["ok"] = True
            return result
        if verb in ("enable", "disable"):
            ident = (ext_id or "").strip()
            if not ident:
                raise ValueError(
                    f"studio_extensions {verb} needs ext_id=<id>")
            store.set_enabled("studio", ident, verb == "enable")
            return {"ok": True, "id": ident, "enabled": verb == "enable"}
        if verb == "uninstall":
            ident = (ext_id or "").strip()
            if not ident:
                raise ValueError(
                    "studio_extensions uninstall needs ext_id=<id>")
            store.uninstall(ident)
            return {"ok": True, "id": ident, "uninstalled": True}
        if verb == "theme":
            ident = (ext_id or "").strip()
            if not ident:
                raise ValueError(
                    "studio_extensions theme needs ext_id=<id> or "
                    "'none'")
            if ident.lower() == "none":
                store.set_active_theme("studio", None)
                return {"ok": True, "active_theme": None}
            store.set_active_theme("studio", ident)
            return {"ok": True, "active_theme": ident}
        if verb == "themes":
            found = store.installed_for("studio", "theme")
            return {"ok": True,
                    "themes": [{"id": str(f.installed.id),
                                "name": f.installed.name,
                                "version": str(f.installed.version)}
                               for f in found]}
        if verb == "panels":
            wanted = (page or "photo").strip().lower()
            found = store.installed_for("studio", "plugin")
            panels = []
            for entry in found:
                parsed = _read_panel(entry.dir / "panel.json")
                if parsed is None or parsed.page != wanted:
                    continue
                panels.append({"id": str(entry.installed.id),
                               "title": parsed.title,
                               "slot": parsed.slot,
                               "blocks": [dict(b) for b in parsed.blocks]})
            return {"ok": True, "page": wanted, "panels": panels}
        raise ValueError(f"unknown studio_extensions action {verb!r}")


def _read_panel(path: Path):
    try:
        data = path.read_bytes()
    except OSError:
        return None
    try:
        return PanelFile.parse(data)
    except ValueError:
        return None
