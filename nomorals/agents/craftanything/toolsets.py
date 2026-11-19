"""Groups of tools.

Port of upstream ``Toolset`` (``crates/engine/src/tool.rs``): agents see
``Core`` plus whatever toolsets they enable, which keeps their context
small; every tool is always reachable from the CLI and UI. The built-in
groups are listed here; a caller adds its own with
:meth:`Toolset.custom` (upstream's ``Toolset::Custom``), and
:func:`list_toolsets` lists every group in use. Serialized as its id.

Devon's groups differ from upstream's (upstream groups by target
platform — native, Apple, Android — because it inspects software;
Devon groups by what the tool does), but the shape is upstream's:
``id`` / ``title`` / ``from_id`` / ``ALL`` / ``Custom``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Toolset", "toolset_for", "list_toolsets"]


class Toolset:
    """A group of tools, serialized as its id."""

    _BY_ID: dict[str, Toolset] = {}
    _CUSTOM: dict[str, Toolset] = {}

    def __init__(self, toolset_id: str, title: str) -> None:
        self.id = toolset_id
        self.title = title

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Toolset({self.id!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Toolset) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_id(cls, toolset_id: str) -> Toolset | None:
        """A built-in or previously-made custom group by id."""
        if toolset_id in cls._BY_ID:
            return cls._BY_ID[toolset_id]
        return cls._CUSTOM.get(toolset_id)

    @classmethod
    def custom(cls, toolset_id: str) -> Toolset:
        """A group added at run time (lower-case letters, digits, ``_``)."""
        toolset_id = (toolset_id or "").strip().lower()
        if not toolset_id or any(
                ch not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                for ch in toolset_id):
            raise ValueError(
                f"custom toolset id must be lower-case letters, digits "
                f"or '_'; got {toolset_id!r}")
        if toolset_id not in cls._CUSTOM:
            cls._CUSTOM[toolset_id] = cls(toolset_id, toolset_id)
        return cls._CUSTOM[toolset_id]


def _register(toolset_id: str, title: str) -> Toolset:
    toolset = Toolset(toolset_id, title)
    Toolset._BY_ID[toolset_id] = toolset
    return toolset


#: Triage, overview, search, lists, evidence, and the tools that do not
#: belong to a narrower group. Agents always see Core.
Toolset.CORE = _register("core", "Essentials")
#: The craft organs and the other media tools: design, sound, CAD,
#: materials, textures, animation, film, effects, artist, word, sheets.
Toolset.MEDIA = _register("media", "Media and design")
#: The document organs: word, sheets, decks, PDF.
Toolset.DOCUMENTS = _register("documents", "Documents")
#: Anything that reaches the network: fetch, search, browser, social.
Toolset.NETWORK = _register("network", "Network")
#: Anything that runs code or the shell, or spawns agents/missions.
Toolset.SYSTEM = _register("system", "Running programs")
#: The agent surface itself: goals, projects, skills, planning.
Toolset.AGENTS = _register("agents", "Agents")
#: Money: finance, trading, commerce, deals.
Toolset.COMMERCE = _register("commerce", "Money and commerce")

Toolset.ALL: tuple[Toolset, ...] = (
    Toolset.CORE, Toolset.MEDIA, Toolset.DOCUMENTS, Toolset.NETWORK,
    Toolset.SYSTEM, Toolset.AGENTS, Toolset.COMMERCE,
)

#: Tool names that belong to a narrower group than their capability
#: alone would suggest. The media organs first: their capability is
#: ``fs.write`` like a hundred other tools, but they are media tools.
_MEDIA_TOOLS = frozenset({
    "artcraft", "anicraft", "cadcraft", "deckcraft", "designcraft",
    "effectcraft", "figcraft", "filmcraft", "gridcraft", "lightcraft",
    "materialcraft", "pdfcraft", "photocraft", "soundcraft",
    "texturecraft", "vectorcraft", "wordcraft",
    "media", "media_edit", "media_gen", "media_director", "media_pipeline",
    "bgremove", "board", "studio", "music", "visuals", "voice_live",
})
#: The document organs (word/sheets/decks live under documents/ but
#: their tool names are the craft names, already in _MEDIA_TOOLS; these
#: are the non-craft document tools).
_DOCUMENT_TOOLS = frozenset({
    "documents", "doc", "docx", "pdf", "sheets", "decks",
})
#: Capabilities that put a tool in the System group.
_SYSTEM_CAPABILITIES = frozenset({
    "exec.shell", "exec.code", "exec.install", "agent.spawn",
    "mission.start", "train.run", "train.gpu",
})
#: Capabilities that put a tool in the Network group.
_NETWORK_CAPABILITIES = frozenset({
    "net.out", "net.download", "net.browser",
    "social.read", "social.post", "social.dm", "social.bulk",
    "social.play",
})
#: Capabilities that put a tool in the Commerce group.
_COMMERCE_CAPABILITIES = frozenset({
    "finance", "trading",
})


def toolset_for(name: str, spec: Any) -> Toolset:
    """The toolset a tool belongs to, from its name and spec.

    Explicit family table first (the media/document organs), then the
    capability: exec/agent tools are System, network/social tools are
    Network, money tools are Commerce, everything else is Core. Every
    tool lands in exactly one group — there is no unassigned tool.
    """
    if name in _MEDIA_TOOLS:
        return Toolset.MEDIA
    if name in _DOCUMENT_TOOLS:
        return Toolset.DOCUMENTS
    capability = getattr(spec, "capability", "") or ""
    if capability in _SYSTEM_CAPABILITIES:
        return Toolset.SYSTEM
    if capability in _NETWORK_CAPABILITIES:
        return Toolset.NETWORK
    if capability in _COMMERCE_CAPABILITIES:
        return Toolset.COMMERCE
    kind = (getattr(spec, "kind", "") or "").lower()
    if kind in ("net", "web", "external"):
        return Toolset.NETWORK
    if kind == "exec":
        return Toolset.SYSTEM
    return Toolset.CORE


def list_toolsets(registry: Any) -> list[dict[str, Any]]:
    """Every toolset in use, with its tools (upstream ``Registry::toolsets``)."""
    members: dict[str, list[str]] = {t.id: [] for t in Toolset.ALL}
    for name, spec in registry._tools.items():
        members[toolset_for(name, spec).id].append(name)
    return [
        {"id": toolset.id, "title": toolset.title,
         "count": len(members[toolset.id]),
         "tools": sorted(members[toolset.id])}
        for toolset in Toolset.ALL
    ]
