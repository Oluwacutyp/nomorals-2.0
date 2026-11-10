"""map — Devon Atlas as an agent tool (docs 19+21 M7).

ONE tool with an ``operation`` dispatch — the registry's
multi-op precedent (``decoder_agent``'s action dispatch) over a
single organ.  Every operation returns the MapAgent's TICKET:
the structured result plus the MapReceipt id(s) that evidence
it (doc 21 §6).  The tool owns no map logic: it parses the
operation's parameters and forwards to
:class:`nomorals.agents.map.MapAgent` verbatim, so tool results
and agent results are the same dicts by construction.  The
module is pinned L5 in the layering map because it composes
the L5 agent (the tools.weather / tools.braineval precedent).

    map(operation="find", query="Ilawe")
    map(operation="near", point="7.6211,5.2214")
    map(operation="route", start="Ilawe", end="Ado Ekiti")
    map(operation="view", point="7.6211,5.2214", zoom=13)
    map(operation="overlay", overlay="delivery stops")

Operations: ``packs`` (the verified shelf) · ``find`` /
``near`` (the local gazetteer — local first; federation is
owner policy and off by default) · ``route`` (the local road
graph; a no-route answers with its reason) · ``view`` (a static
map rendered from a local pack, answered with the PNG's file
path; with ``query=`` naming one of the owner's marks and no
``point=``, the mark's view — the basemap centered on it, its
marker drawn) · ``overlays`` (the owner's saved overlay
documents) · ``overlay`` (render one named overlay over its
pack) · ``marks`` (the owner's personal marks, M9).

M9: the owner's marks are first-class places on this plane —
``find`` resolves a mark ahead of the gazetteer, ``route``
accepts a mark label as either endpoint, and ``marks`` lists
them.  Mark CAPTURE is not a tool operation: it is the armed
``/map mark`` chat flow (the M6 posture — the owner's location
moves only through chat-owner surfaces, never agent calls).
"""

from __future__ import annotations

from typing import Any

from ..core.errors import ToolError
from ..core.policy import Capability

_OPERATIONS = ("packs", "find", "near", "route", "view",
               "overlays", "overlay", "marks")


def _point(point: str, op: str) -> tuple[float, float]:
    raw = (point or "").strip()
    try:
        nums = [float(v) for v in raw.split(",")]
    except ValueError:
        nums = []
    if len(nums) != 2:
        raise ToolError(
            "map operation " + repr(op) + " needs point= as "
            "\"lat,lon\", got " + repr(point))
    return nums[0], nums[1]


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "map",
        description=(
            "Devon Atlas — Devon's own local maps, as tickets "
            "with their MapReceipt ids. operation: packs (the "
            "installed, hash-pinned basemap packs + verify "
            "state) | find (place name -> ranked local "
            "gazetteer candidates) | near (point -> named "
            "places with distances) | route (start/end as "
            "place names or lat,lon -> distance, duration "
            "estimate, steps, geometry, over the local road "
            "graph) | view (point -> a rendered PNG, answered "
            "with its file path; query= may name one of the "
            "owner's marks instead) | overlays (the owner's "
            "saved overlay documents) | overlay (overlay= "
            "name -> that overlay rendered over its pack as a "
            "PNG) | marks (the owner's personal marks). Find "
            "resolves the owner's marks first and route accepts "
            "a mark label as an endpoint. Everything is local: "
            "no provider is ever "
            "consulted silently, and an empty local answer is "
            "returned as outcome 'unavailable' with its reason."
        ),
        capability=Capability.FS_READ,
        annotations={"destructive": False, "idempotent": True,
                     "read_only": True, "open_world": False},
        examples=[
            "map(operation=\"find\", query=\"Ilawe\")",
            "map(operation=\"route\", start=\"Ilawe\", "
            "end=\"Ado Ekiti\")",
            "map(operation=\"view\", point=\"7.6211,5.2214\")",
        ],
    )
    def map_tool(operation: str, query: str = "", point: str = "",
                 start: str = "", end: str = "", zoom: int = 0,
                 overlay: str = "") -> dict[str, Any]:
        from ..agents.map import MapAgent

        op = (operation or "").strip().lower()
        if op not in _OPERATIONS:
            raise ToolError(
                "unknown map operation " + repr(operation)
                + " — " + " | ".join(_OPERATIONS))
        agent = MapAgent(context)
        try:
            if op == "packs":
                return agent.packs()
            if op == "find":
                if not (query or "").strip():
                    raise ToolError(
                        "map operation 'find' needs query= (a "
                        "place name)")
                return agent.find(query)
            if op == "near":
                lat, lon = _point(point, op)
                return agent.near(lat, lon)
            if op == "route":
                if not (start or "").strip() \
                        or not (end or "").strip():
                    raise ToolError(
                        "map operation 'route' needs start= and "
                        "end= (place names or lat,lon)")
                return agent.route(start, end)
            if op == "view":
                if not (point or "").strip() \
                        and (query or "").strip():
                    return agent.view_mark(
                        query, zoom=int(zoom) or None)
                lat, lon = _point(point, op)
                return agent.view(lat, lon,
                                  zoom=int(zoom) or None)
            if op == "marks":
                return agent.marks()
            if op == "overlays":
                return agent.overlays()
            # op == "overlay"
            if not (overlay or "").strip():
                raise ToolError(
                    "map operation 'overlay' needs overlay= "
                    "(the overlay's name)")
            return agent.overlay_render(
                overlay, zoom=int(zoom) or None)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
