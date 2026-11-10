"""MapAgent — Devon Atlas as an agent (docs 19+21 M7).

The agency-facing organ over the shipped Atlas planes: packs
(M1), the local geocoder (M4), the local router (M5), the
renderers (M3/M5), and the owner's overlays
(:mod:`nomorals.maps.overlays`).  The agent owns no map logic of
its own — every method assembles the context (packs directory,
receipt store, settings posture) and calls the same cores the
/map chat surface calls, returning the operation's TICKET: the
structured result plus the MapReceipt id(s) that evidence it
(doc 21 §6 — agents get tickets with receipts, never bare
strings where structure exists).

The thin ``tools/map.py`` tool forwards to these methods
verbatim, and ``agency/wiring.py`` composes the agent beside
its peers (the NewsAgent precedent), so the brain, the tool
plane and the chat surface all read ONE implementation of each
operation.  Malformed input raises ValueError (the tool plane
translates it to a ToolError); operational outcomes — no pack
installed, no covering pack, no route, an empty gazetteer —
return as tickets with outcome ``"unavailable"`` and the reason
in ``detail``, never as exceptions and never dressed up.

Receipt ids ride the ticket because the agent layer is where
evidence is cited (the translate tool's precedent); chat
surfaces, which never print internal ids, render their own text
from the same cores.

M9 adds the owner's personal marks
(:mod:`nomorals.maps.marks`) to the same plane: find resolves
his marks FIRST (a mark is his own named place — it outranks
any gazetteer hit of the same name), route endpoints accept a
mark by label, and a mark renders as a gold marker through the
shipped overlay seam.  Capture itself is a chat-surface act
(the armed /map mark flow), never a tool operation — the same
posture as M6's live share.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..location.primitives import LatLon
from ..maps.coords import (
    BBox,
    TileCoord,
    tile_x_to_lon,
    tile_y_to_lat,
    validate_lat,
    validate_lon,
)
from ..maps.manifest import PackError, load_manifest
from ..maps.marks import Mark, MarkStore
from ..maps.overlays import (
    KIND_MARKER,
    Overlay,
    OverlayFeature,
    OverlayStore,
    overlay_bbox,
    overlay_to_map_overlay,
)
from ..maps.packs import verify_pack_dir
from ..maps.receipts import MapReceiptStore
from ..maps.settings import map_settings

__all__ = ["MapAgent"]


class MapAgent:
    """Atlas operations over one context, as ticket dicts."""

    def __init__(self, context: Any) -> None:
        self.context = context

    # ── context assembly (the /map surface's own resolution) ──

    def _home(self) -> str:
        settings = getattr(self.context, "settings", None)
        home = getattr(settings, "home_path", None)
        return str(home) if home else ""

    def packs_dir(self) -> Path:
        """Where Atlas packs live: ``NM_MAPS_PACKS_DIR``, else
        ``<home>/maps/packs`` (the M1 settings' own shape)."""
        configured = map_settings().packs_dir
        if configured:
            return Path(configured).expanduser()
        home = self._home()
        base = Path(home) if home else Path.home() / ".nomorals"
        return base / "maps" / "packs"

    def receipt_store(self) -> MapReceiptStore:
        return MapReceiptStore(
            db=getattr(self.context, "db", None), home=self._home())

    def overlay_store(self) -> OverlayStore:
        return OverlayStore(
            db=getattr(self.context, "db", None), home=self._home())

    def mark_store(self) -> MarkStore:
        return MarkStore(
            db=getattr(self.context, "db", None), home=self._home())

    def _installed(self):
        """Installed packs as ``(dir, manifest|None, error)``
        triples — a directory whose manifest will not load is
        carried as a named corrupt entry, never silently
        skipped (the surface's discipline, shared)."""
        root = self.packs_dir()
        found = []
        if not root.is_dir():
            return found
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            try:
                found.append((entry, load_manifest(entry), ""))
            except PackError as exc:
                found.append((entry, None, str(exc)))
        return found

    def _shelf(self):
        """The loadable packs: ``(dir, manifest)`` pairs."""
        return [(d, m) for d, m, err in self._installed()
                if m is not None]

    def _gazetteer(self):
        from ..maps.gazetteer import Gazetteer

        return Gazetteer(
            self.packs_dir(), store=self.receipt_store(),
            federation=map_settings().geocode_federation)

    def _router(self):
        from ..maps.routing import Router

        return Router(
            self.packs_dir(), store=self.receipt_store(),
            federation=map_settings().routing_federation)

    # ── packs ────────────────────────────────────────────────

    def packs(self) -> dict[str, Any]:
        """The shelf, verified: one pack_verify receipt per
        loadable pack rides the ticket's receipt list."""
        store = self.receipt_store()
        ticket: dict[str, Any] = {
            "operation": "packs", "outcome": "ok",
            "packs_dir": str(self.packs_dir()), "packs": [],
            "receipt_ids": [], "detail": ""}
        installed = self._installed()
        if not installed:
            ticket["outcome"] = "unavailable"
            ticket["detail"] = (
                "no map packs installed — Atlas packs live in "
                + str(self.packs_dir()) + " (built by the basemap "
                "pipeline; each pack is a directory holding a "
                "manifest.json)")
            return ticket
        for entry, manifest, error in installed:
            if manifest is None:
                ticket["packs"].append({
                    "dir": entry.name, "corrupt": error})
                continue
            verdict, receipt = verify_pack_dir(entry, store=store)
            ticket["receipt_ids"].append(receipt.receipt_id)
            ticket["packs"].append({
                "pack_id": manifest.pack_id,
                "name": manifest.name,
                "tile_format": manifest.tile_format,
                "zoom_min": manifest.zoom_min,
                "zoom_max": manifest.zoom_max,
                "bbox": {"west": manifest.bbox.west,
                         "south": manifest.bbox.south,
                         "east": manifest.bbox.east,
                         "north": manifest.bbox.north},
                "attribution": manifest.attribution,
                "verify": {
                    "ok": verdict.ok,
                    "signature_state": verdict.signature_state,
                    "missing": list(verdict.missing),
                    "mismatched": list(verdict.mismatched)}})
        return ticket

    # ── personal marks (M9) ──────────────────────────────────

    def place_hint(self, lat: float, lon: float) -> str:
        """The gazetteer's name for a spot, when it has one.

        Reads an EXISTING index only: a mark capture must never
        trigger the (potentially minutes-long) index build, so
        when no index file exists the hint is honestly empty and
        the coordinates are the answer.  Garnish: never raises.
        """
        try:
            gazetteer = self._gazetteer()
            if not gazetteer.index_path.is_file():
                return ""
            result = gazetteer.reverse(float(lat), float(lon),
                                       quiet=True, limit=1)
            if result.candidates:
                return result.candidates[0].name
        except Exception:  # noqa: BLE001 - a hint is garnish
            return ""
        return ""

    def _fill_hint(self, mark: Mark) -> Mark:
        """A mark with its place hint filled in, when the local
        gazetteer covers the spot (lazily, at read — capture
        never blocked on it).  The fill persists best-effort."""
        if mark.place_hint:
            return mark
        hint = self.place_hint(mark.lat, mark.lon)
        if not hint:
            return mark
        self.mark_store().set_hint(mark.mark_id, hint)
        return replace(mark, place_hint=hint)

    def mark_for(self, label: str) -> Mark | None:
        """The owner's mark under ``label`` (hint filled), or
        None.  The one lookup every surface shares."""
        mark = self.mark_store().get(label)
        return self._fill_hint(mark) if mark is not None else None

    @staticmethod
    def _mark_candidate(mark: Mark) -> dict[str, Any]:
        return {"name": mark.label, "kind": "mark",
                "lat": mark.lat, "lon": mark.lon, "pack_id": "",
                "score": None, "matched_as": "mark",
                "place_hint": mark.place_hint,
                "platform": mark.platform,
                "marked_at": mark.updated_at}

    def marks(self) -> dict[str, Any]:
        """The owner's marks, as rows (hints filled lazily)."""
        rows = []
        for mark in self.mark_store().list():
            mark = self._fill_hint(mark)
            rows.append({
                "label": mark.label, "lat": mark.lat,
                "lon": mark.lon, "place_hint": mark.place_hint,
                "platform": mark.platform, "method": mark.method,
                "accuracy": mark.accuracy,
                "created_at": mark.created_at,
                "updated_at": mark.updated_at})
        return {"operation": "marks", "outcome": "ok",
                "marks": rows}

    def view_mark(self, label: str, *,
                  zoom: int | None = None) -> dict[str, Any]:
        """Render the basemap centered on one of the owner's
        marks, the mark drawn as a gold marker through the
        shipped MapOverlay seam — a transient one-marker overlay
        over ``view``'s ±1-tile box."""
        store = self.mark_store()
        ticket: dict[str, Any] = {
            "operation": "view", "outcome": "unavailable",
            "mark": None, "image_path": "", "notes": [],
            "receipt_id": "", "detail": ""}
        mark = store.get(label)
        if mark is None:
            existing = [m.label for m in store.list()]
            ticket["detail"] = (
                "no mark named " + repr(label) + " — your marks: "
                + (", ".join(existing) if existing else "none yet"))
            return ticket
        mark = self._fill_hint(mark)
        if zoom is None:
            ceilings = []
            for _entry, manifest in self._shelf():
                box = manifest.bbox
                if box.crosses_antimeridian:
                    continue
                if box.south <= mark.lat <= box.north \
                        and box.west <= mark.lon <= box.east:
                    ceilings.append(manifest.zoom_max)
            zoom = min(15, max(ceilings)) if ceilings else 15
        zoom = int(zoom)
        tile = TileCoord.from_latlon(mark.lat, mark.lon, zoom)
        n = 1 << zoom
        x0, x1 = max(0, tile.x - 1), min(n - 1, tile.x + 1)
        y0, y1 = max(0, tile.y - 1), min(n - 1, tile.y + 1)
        bbox = BBox(west=tile_x_to_lon(x0, zoom),
                    south=tile_y_to_lat(y1 + 1, zoom),
                    east=tile_x_to_lon(x1 + 1, zoom),
                    north=tile_y_to_lat(y0, zoom))
        overlay = Overlay(
            overlay_id="mark-" + mark.mark_id, name=mark.label,
            created_at=mark.updated_at,
            features=(OverlayFeature(
                KIND_MARKER, ((mark.lat, mark.lon),), mark.label),))
        rendered = self._render(bbox, zoom, overlay=overlay)
        rendered["operation"] = "view"
        rendered["mark"] = {
            "label": mark.label, "lat": mark.lat, "lon": mark.lon,
            "place_hint": mark.place_hint,
            "platform": mark.platform, "method": mark.method,
            "marked_at": mark.updated_at}
        return rendered

    # ── geocoder ─────────────────────────────────────────────

    def find(self, query: str, limit: int = 8) -> dict[str, Any]:
        """Forward geocode over the local gazetteer (M4).  The
        index builds itself on first use, like the surface.

        M9: the OWNER'S MARKS resolve FIRST — a mark with this
        label is prepended (kind ``"mark"``) ahead of any
        gazetteer hit, and answers even on an empty shelf: his
        own saved place needs no pack to be found again."""
        query = (query or "").strip()
        if not query:
            raise ValueError("find needs a place name")
        mark = self.mark_for(query)
        shelf = self._shelf()
        if not shelf:
            if mark is not None:
                return {"operation": "find", "outcome": "ok",
                        "query": query,
                        "candidates": [self._mark_candidate(mark)],
                        "posture": "local", "provider": "",
                        "note": "", "receipt_id": ""}
            return {"operation": "find", "outcome": "unavailable",
                    "query": query, "candidates": [],
                    "receipt_id": "",
                    "detail": "no map packs installed — the local "
                              "gazetteer indexes installed packs "
                              "only, and the shelf is empty"}
        gazetteer = self._gazetteer()
        built = gazetteer.ensure(shelf)
        result = gazetteer.search(query, limit=int(limit))
        candidates = [
            {"name": c.name, "kind": c.kind, "lat": c.lat,
             "lon": c.lon, "pack_id": c.pack_id,
             "score": c.score, "matched_as": c.matched_as}
            for c in result.candidates]
        if mark is not None:
            candidates.insert(0, self._mark_candidate(mark))
        ticket: dict[str, Any] = {
            "operation": "find",
            "outcome": "ok" if candidates else "unavailable",
            "query": query,
            "candidates": candidates,
            "posture": result.posture, "provider": result.provider,
            "note": result.note,
            "receipt_id": (result.receipt.receipt_id
                           if result.receipt else "")}
        if built.built_any:
            ticket["indexed"] = {
                "entries": built.total_entries,
                "packs": len(shelf)}
        if not candidates:
            ticket["detail"] = (
                "no place named " + repr(query) + " in the local "
                "Atlas index — " + result.note)
        return ticket

    def near(self, lat: float, lon: float,
             limit: int = 8) -> dict[str, Any]:
        """Reverse geocode: named places near a point, with
        their distances stated (M4's ladder)."""
        validate_lat(float(lat))
        validate_lon(float(lon))
        shelf = self._shelf()
        if not shelf:
            return {"operation": "near", "outcome": "unavailable",
                    "lat": float(lat), "lon": float(lon),
                    "candidates": [], "receipt_id": "",
                    "detail": "no map packs installed — the local "
                              "gazetteer indexes installed packs "
                              "only, and the shelf is empty"}
        gazetteer = self._gazetteer()
        gazetteer.ensure(shelf)
        result = gazetteer.reverse(float(lat), float(lon),
                                   limit=int(limit))
        ticket: dict[str, Any] = {
            "operation": "near",
            "outcome": "ok" if result.candidates else "unavailable",
            "lat": result.lat, "lon": result.lon,
            "candidates": [
                {"name": c.name, "kind": c.kind, "lat": c.lat,
                 "lon": c.lon, "pack_id": c.pack_id,
                 "distance_km": c.distance_km}
                for c in result.candidates],
            "searched_km": result.searched_km,
            "rung_km": result.rung_km,
            "posture": result.posture, "provider": result.provider,
            "note": result.note,
            "receipt_id": (result.receipt.receipt_id
                           if result.receipt else "")}
        if not result.candidates:
            ticket["detail"] = (
                "no named place within "
                + str(int(result.searched_km)) + " km — "
                + result.note)
        return ticket

    # ── routing ──────────────────────────────────────────────

    def _resolve_endpoint(self, text: str, gazetteer):
        """One route end as ``(LatLon, label)`` — or ``(None,
        error)``.  Raw "lat,lon" wins first; then the OWNER'S
        MARKS by label (M9 — his own named places outrank the
        index, as in find); otherwise the gazetteer's top hit
        (the surface's rule, shared)."""
        raw = (text or "").strip()
        nums: list[float] = []
        try:
            nums = [float(v) for v in raw.split(",")]
        except ValueError:
            nums = []
        if len(nums) == 2:
            lat, lon = nums
            try:
                validate_lat(lat)
                validate_lon(lon)
            except ValueError as exc:
                return None, "bad point " + repr(raw) + ": " + str(exc)
            return LatLon(lat, lon), f"{lat:.4f},{lon:.4f}"
        mark = self.mark_store().get(raw)
        if mark is not None:
            return LatLon(mark.lat, mark.lon), mark.label
        result = gazetteer.search(raw)
        if not result.candidates:
            return None, ("no place named " + repr(raw) + " in the "
                          "local Atlas index — " + result.note)
        top = result.candidates[0]
        return LatLon(top.lat, top.lon), top.name

    def route(self, from_text: str, to_text: str) -> dict[str, Any]:
        """Route between two places (names or "lat,lon") over
        one installed vector pack's road graph (M5)."""
        if not (from_text or "").strip() \
                or not (to_text or "").strip():
            raise ValueError(
                "route needs both ends: a start and an end "
                "(place names or lat,lon)")
        shelf = self._shelf()
        ticket: dict[str, Any] = {
            "operation": "route", "outcome": "unavailable",
            "from_label": "", "to_label": "", "routed": False,
            "steps": [], "geometry": [], "receipt_id": ""}
        if not shelf:
            ticket["detail"] = (
                "no map packs installed — routing runs on "
                "installed packs only; there is no provider "
                "fallback")
            return ticket
        gazetteer = self._gazetteer()
        gazetteer.ensure(shelf)
        start, from_label = self._resolve_endpoint(from_text,
                                                   gazetteer)
        if start is None:
            ticket["detail"] = from_label
            return ticket
        end, to_label = self._resolve_endpoint(to_text, gazetteer)
        if end is None:
            ticket["detail"] = to_label
            return ticket

        def _covers(manifest, point) -> bool:
            box = manifest.bbox
            if box.crosses_antimeridian:
                return False
            return (box.south <= point.lat <= box.north
                    and box.west <= point.lon <= box.east)

        vector = [(d, m) for d, m in shelf
                  if m.tile_format == "mvt"]
        both = [(d, m) for d, m in vector
                if _covers(m, start) and _covers(m, end)]
        if not both:
            if not vector:
                ticket["detail"] = (
                    "routing needs a vector (mvt) pack — the "
                    "installed packs are raster, which carry no "
                    "road graph. Nothing was fetched from a "
                    "provider.")
            else:
                covering = [m.name for _d, m in vector
                            if _covers(m, start) or _covers(m, end)]
                if covering:
                    ticket["detail"] = (
                        "no single installed vector pack covers "
                        "both ends — " + ", ".join(covering)
                        + " cover one end only. Routing is "
                        "per-pack and local; nothing was fetched "
                        "from a provider.")
                else:
                    ticket["detail"] = (
                        "no installed vector pack covers both "
                        "ends of this route. Routing is per-pack "
                        "and local; nothing was fetched from a "
                        "provider.")
            return ticket
        entry, manifest = sorted(
            both, key=lambda dm: (-dm[1].zoom_max, dm[1].name))[0]
        result = self._router().route(
            entry, manifest, start, end,
            from_label=from_label, to_label=to_label)
        ticket.update({
            "outcome": "ok" if result.ok else "unavailable",
            "routed": bool(result.ok),
            "from_label": result.from_label,
            "to_label": result.to_label,
            "distance_m": result.distance_m,
            "duration_s": result.duration_s,
            "steps": [
                {"name": s.name, "road_class": s.road_class,
                 "length_m": s.length_m, "named": s.named}
                for s in result.steps],
            "geometry": [[p[0], p[1]] for p in result.geometry],
            "from_snap_m": result.from_snap_m,
            "to_snap_m": result.to_snap_m,
            "reason": result.reason, "detail": result.detail,
            "pack_id": manifest.pack_id,
            "posture": result.posture, "provider": result.provider,
            "receipt_id": (result.receipt.receipt_id
                           if result.receipt else "")})
        return ticket

    # ── rendering ────────────────────────────────────────────

    @staticmethod
    def _zoom_for_bbox(bbox: BBox) -> int:
        """The deepest zoom whose tile count for ``bbox`` stays
        modest (the /map view surface's rule, shared)."""
        import math

        from ..maps.coords import lat_to_y_float, lon_to_x_float

        for z in range(0, 19):
            n = 1 << z
            cols = lon_to_x_float(bbox.east, z) \
                - lon_to_x_float(bbox.west, z)
            if bbox.crosses_antimeridian:
                cols += n
            rows = lat_to_y_float(bbox.south, z) \
                - lat_to_y_float(bbox.north, z)
            if math.ceil(cols) * math.ceil(rows) > 64:
                return max(0, z - 1)
        return 18

    def _pick_pack(self, bbox: BBox, zoom: int, *,
                   prefer_vector: bool = False):
        """The surface's pack discipline: zoom in range, then
        contains, then intersects, deepest zoom_max first (name
        tiebreak).  Returns (entry, manifest) or None."""

        def _contains(outer, inner) -> bool:
            if outer.crosses_antimeridian \
                    or inner.crosses_antimeridian:
                return False
            return (outer.south <= inner.south
                    and inner.north <= outer.north
                    and outer.west <= inner.west
                    and inner.east <= outer.east)

        def _intersects(outer, inner) -> bool:
            if inner.south > outer.north \
                    or inner.north < outer.south:
                return False
            if outer.crosses_antimeridian \
                    or inner.crosses_antimeridian:
                return True
            return not (inner.east < outer.west
                        or inner.west > outer.east)

        pool = self._shelf()
        if prefer_vector:
            vector = [(d, m) for d, m in pool
                      if m.tile_format == "mvt"]
            if vector:
                pool = vector
        in_zoom = [(d, m) for d, m in pool
                   if m.zoom_min <= zoom <= m.zoom_max]
        full = [(d, m) for d, m in in_zoom
                if _contains(m.bbox, bbox)]
        partial = [(d, m) for d, m in in_zoom
                   if _intersects(m.bbox, bbox)]
        for group in (full, partial):
            if group:
                return sorted(
                    group,
                    key=lambda dm: (-dm[1].zoom_max, dm[1].name))[0]
        return None

    def _render(self, bbox: BBox, zoom: int, *,
                overlay: Overlay | None = None,
                extra_notes: tuple[str, ...] = (),
                ) -> dict[str, Any]:
        """Render one view to a durable PNG under the maps home
        and return its ticket.  The file is the hand-on ref the
        agent/tool layer passes around (the report tools'
        discipline: a built artifact answers with its path)."""
        from ..maps.render import FixtureRenderer
        from ..maps.vector_render import VectorRenderer

        ticket: dict[str, Any] = {
            "outcome": "unavailable", "image_path": "",
            "pack_id": "", "pack_name": "", "tile_format": "",
            "zoom": int(zoom),
            "bbox": {"west": bbox.west, "south": bbox.south,
                     "east": bbox.east, "north": bbox.north},
            "attribution": "", "missing_tiles": [],
            "notes": list(extra_notes), "receipt_id": "",
            "detail": ""}
        picked = self._pick_pack(bbox, zoom,
                                 prefer_vector=overlay is not None)
        if picked is None:
            shelf = self._shelf()
            if not shelf:
                ticket["detail"] = (
                    "no map packs installed — Atlas renders "
                    "installed packs only; there is no provider "
                    "fallback")
            else:
                names = "; ".join(
                    m.name + " (z" + str(m.zoom_min) + "–"
                    + str(m.zoom_max) + ")" for _d, m in shelf)
                ticket["detail"] = (
                    "no installed pack covers this view at z"
                    + str(zoom) + " — installed: " + names
                    + ". Atlas is local-first: nothing was "
                    "fetched from a provider.")
            return ticket
        entry, manifest = picked
        ticket.update({"pack_id": manifest.pack_id,
                       "pack_name": manifest.name,
                       "tile_format": manifest.tile_format,
                       "attribution": manifest.attribution})
        store = self.receipt_store()
        map_overlay = None
        if overlay is not None:
            map_overlay, conv_notes = overlay_to_map_overlay(
                overlay)
            ticket["notes"].extend(conv_notes)
        if manifest.tile_format == "mvt":
            renderer = VectorRenderer(entry, manifest, store=store)
            result = renderer.render(bbox, zoom, size=(768, 768),
                                     overlay=map_overlay)
        elif manifest.tile_format == "png":
            if overlay is not None:
                ticket["notes"].append(
                    "overlay not drawn: the covering pack is "
                    "raster (png) and the shipped raster "
                    "renderer has no overlay seam — the base "
                    "view is returned without the overlay")
            renderer = FixtureRenderer(entry, manifest, store=store)
            result = renderer.render(bbox, zoom, size=(768, 768))
        else:
            ticket["detail"] = (
                "pack " + manifest.name + " uses tile format "
                + repr(manifest.tile_format) + ", which no "
                "shipped renderer draws yet (mvt and png packs "
                "render)")
            return ticket
        ticket["receipt_id"] = result.receipt.receipt_id
        ticket["missing_tiles"] = list(result.missing_tiles)
        if result.image is None:
            ticket["detail"] = (
                "Atlas could not render this view from "
                + manifest.name + ": " + result.receipt.detail)
            return ticket
        home = self._home()
        base = Path(home) if home else Path.home() / ".nomorals"
        renders = base / "maps" / "renders"
        renders.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(
            (manifest.pack_id + "|" + repr(bbox) + "|"
             + str(zoom) + "|"
             + (overlay.overlay_id if overlay else "")).encode(
                 "utf-8")).hexdigest()[:10]
        path = renders / (manifest.pack_id + "-z" + str(zoom)
                          + "-" + key + ".png")
        path.write_bytes(result.image)
        ticket.update({
            "outcome": "ok", "image_path": str(path),
            "image_bytes": len(result.image),
            "width": result.width, "height": result.height})
        return ticket

    def view(self, lat: float | None = None,
             lon: float | None = None, *,
             bbox: BBox | None = None,
             zoom: int | None = None) -> dict[str, Any]:
        """A static map view as a PNG ticket: a point (the
        surface's ±1-tile box at the zoom) or an explicit box."""
        if bbox is None:
            if lat is None or lon is None:
                raise ValueError(
                    "view needs a point (lat, lon) or a bbox")
            validate_lat(float(lat))
            validate_lon(float(lon))
            zoom = 13 if zoom is None else int(zoom)
            tile = TileCoord.from_latlon(float(lat), float(lon),
                                         zoom)
            n = 1 << zoom
            x0, x1 = max(0, tile.x - 1), min(n - 1, tile.x + 1)
            y0, y1 = max(0, tile.y - 1), min(n - 1, tile.y + 1)
            bbox = BBox(west=tile_x_to_lon(x0, zoom),
                        south=tile_y_to_lat(y1 + 1, zoom),
                        east=tile_x_to_lon(x1 + 1, zoom),
                        north=tile_y_to_lat(y0, zoom))
        else:
            zoom = self._zoom_for_bbox(bbox) if zoom is None \
                else int(zoom)
        ticket = self._render(bbox, int(zoom))
        ticket["operation"] = "view"
        return ticket

    # ── overlays ─────────────────────────────────────────────

    def overlays(self) -> dict[str, Any]:
        """The owner's overlay documents, as summary rows."""
        rows = []
        for overlay in self.overlay_store().list():
            rows.append({
                "overlay_id": overlay.overlay_id,
                "name": overlay.name,
                "created_at": overlay.created_at,
                "feature_count": len(overlay.features),
                "kinds": sorted({f.kind for f in overlay.features}),
                "features": [
                    {"kind": f.kind,
                     "points": [[p[0], p[1]] for p in f.points],
                     "label": f.label}
                    for f in overlay.features]})
        return {"operation": "overlays", "outcome": "ok",
                "overlays": rows}

    def overlay_render(self, name: str, *,
                       zoom: int | None = None) -> dict[str, Any]:
        """Render one named overlay over the best covering pack.

        The conversion's honesty notes (polygon outlines) ride
        the ticket's ``notes`` — the caller shows them, the
        overlay is never silently degraded."""
        store = self.overlay_store()
        overlay = store.get(name)
        ticket: dict[str, Any] = {
            "operation": "overlay_render",
            "outcome": "unavailable", "overlay": None,
            "image_path": "", "notes": [], "receipt_id": "",
            "detail": ""}
        if overlay is None:
            existing = [o.name for o in store.list()]
            ticket["detail"] = (
                "no overlay named " + repr(name) + " — stored "
                "overlays: "
                + (", ".join(existing) if existing else "none"))
            return ticket
        box = overlay_bbox(overlay)
        pad_lat = max(0.01, 0.15 * (box.north - box.south))
        pad_lon = max(0.01, 0.15 * (box.east - box.west))
        try:
            padded = BBox(west=max(-180.0, box.west - pad_lon),
                          south=max(-85.0, box.south - pad_lat),
                          east=min(180.0, box.east + pad_lon),
                          north=min(85.0, box.north + pad_lat))
        except ValueError as exc:
            ticket["detail"] = "overlay box is not renderable: " \
                + str(exc)
            return ticket
        zoom = self._zoom_for_bbox(padded) if zoom is None \
            else int(zoom)
        # The bbox rule can ask deeper than any pack goes (a
        # point overlay wants the deepest zoom its pack
        # offers): clamp to the deepest ceiling among packs
        # intersecting the box, so a renderable overlay renders.
        ceilings = []
        for _entry, manifest in self._shelf():
            box = manifest.bbox
            if padded.south > box.north or padded.north < box.south:
                continue
            if not (box.crosses_antimeridian
                    or padded.east < box.west
                    or padded.west > box.east):
                ceilings.append(manifest.zoom_max)
        if ceilings:
            zoom = min(zoom, max(ceilings))
        rendered = self._render(padded, zoom, overlay=overlay)
        rendered["operation"] = "overlay_render"
        rendered["overlay"] = {
            "name": overlay.name,
            "overlay_id": overlay.overlay_id,
            "feature_count": len(overlay.features),
            "kinds": sorted({f.kind for f in overlay.features})}
        return rendered
