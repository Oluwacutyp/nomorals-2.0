"""RuntimeLiveMixin: PartnerRuntime command group (live)."""

from __future__ import annotations
from .failures import record_failure


class RuntimeLiveMixin:
    """RuntimeLiveMixin for :class:`PartnerRuntime`."""


    # ── weather awareness + USA situations + timezones ───────────────────
    def _control_weather(self, tail: str) -> str:
        """/weather [place] | usa | forecast <place> | alerts <place> — keyless live weather."""
        from ..weather import Weather, USASituations, owner_tz, tz_note

        w = Weather()
        sub = (tail or "").strip()
        if not sub or sub.lower() == "now":
            return w.now()["text"]
        low = sub.lower()
        if low == "usa":
            return USASituations(w).overview()["text"]
        if low.startswith("forecast"):
            place = sub[8:].strip() or None
            return w.forecast(place, days=3)["text"]
        if low.startswith("alerts"):
            place = sub[6:].strip() or None
            res = w.now(place)
            alerts = res.get("alerts") or []
            if not alerts:
                return f"no active alerts — {res['text']}"
            return "\n".join(f"• [{a.get('severity','?')}] {a.get('title','')}"
                             for a in alerts[:8])
        if low.startswith("tz"):
            return tz_note(owner_tz())
        return w.now(sub)["text"]

    def _control_tz(self, tail: str) -> str:
        """/tz [YYYY-MM-DD HH:MM [from-zone] [to-zone]] — convert time zones."""
        from ..weather import convert_time, owner_tz, tz_note

        toks = (tail or "").strip().split()
        if not toks:
            return tz_note(owner_tz())
        # optional "YYYY-MM-DD HH:MM" glued at the front
        raw = " ".join(toks[:2]) if len(toks) > 1 and toks[1].count(":") else toks[0]
        rest = toks[2:] if len(toks) > 1 and toks[1].count(":") else toks[1:]
        from_zone = rest[0] if len(rest) > 0 else owner_tz()
        to_zone = rest[1] if len(rest) > 1 else owner_tz()
        res = convert_time(raw, from_zone, to_zone)
        return res.get("text") or f"couldn't parse that: {res.get('error','')}"

    # ── Devon Atlas ─────────────────────────────────────────────────────
    def _map_packs_dir(self):
        """Where Atlas packs live: ``NM_MAPS_PACKS_DIR``, else
        ``<home>/maps/packs`` (the M1 settings' own shape)."""
        from pathlib import Path

        from ...maps.settings import map_settings

        configured = map_settings().packs_dir
        if configured:
            return Path(configured).expanduser()
        settings = getattr(self.context, "settings", None)
        home = getattr(settings, "home_path", None)
        base = Path(home) if home else Path.home() / ".nomorals"
        return base / "maps" / "packs"

    def _map_receipt_store(self):
        from ...maps.receipts import MapReceiptStore

        settings = getattr(self.context, "settings", None)
        home = getattr(settings, "home_path", None)
        return MapReceiptStore(
            db=getattr(self.context, "db", None),
            home=str(home) if home else "")

    def _map_installed(self):
        """Installed packs as ``(dir, manifest|None, error)`` triples —
        a directory whose manifest will not load is carried as a
        named corrupt entry, never silently skipped."""
        from ...maps.manifest import PackError, load_manifest

        root = self._map_packs_dir()
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

    def _control_map(self, tail: str, chat_key: str = "",
                     message=None) -> str:
        """/map packs | /map view <lat,lon> [zoom] | /map view <west,south,east,north> [zoom] | /map find <place name> | /map near <lat,lon> | /map route <place or lat,lon> to <place or lat,lon>

        Devon Atlas on chat: renders the owner's installed packs —
        Devon's own basemap data — into a photo, the attribution
        riding the caption (it is manifest data, a license fact).

        Gating: owner-only.  A render spends the owner's host on
        the owner's installed data, and every other Atlas surface
        is an owner surface; the platform distinguishes the owner
        here (:meth:`_is_operator`), and a message-less internal
        call runs as the owner, exactly as /decode's gate reads
        it.  There is no provider fallback behind this verb: when
        no installed pack covers the view, the reply names what is
        installed and what is missing — local-first is the Atlas
        doctrine (doc 21 S1), providers are owner-policy
        federation territory (M4/M5), never a silent substitute.

        /map find and /map near (M4) answer from the local
        gazetteer indexed from those same packs — the index builds
        itself on first use and rebuilds only when a pack's digest
        moves.  Federation (doc 21 S3) is owner policy
        (NM_MAPS_GEOCODE_FEDERATION) and off by default: an empty
        local answer is the final answer unless the owner turns
        federation on and a provider is wired.

        /map route (M5) routes over a road graph extracted from
        one installed vector pack's own tiles — endpoints resolve
        through the gazetteer (or raw coordinates), the answer is
        a text summary plus the route drawn gold on the pack's
        map as a photo, and a no-route is answered honestly with
        its reason (snap failure, disconnected graph, corridor
        bound) — never a provider substitute; routing federation
        is the same owner policy (NM_MAPS_ROUTING_FEDERATION),
        off by default, and no provider ships.

        /map share (M6) runs the LiveLocationBroker over
        location/'s ConsentStore: the owner's command IS the
        opt-in grant (scope "live-share"), a session is a
        time-boxed share of the owner's own SUPPLIED location
        updates to one chat at a declared precision, the
        recipient gets one card naming the precision + TTL up
        front, and every supplied update is evaluated — and its
        decision recorded — by the broker.  A second owner
        policy (NM_MAPS_LIVE_SHARE) gates the whole surface,
        off by default: Atlas on is not share on.

        /map overlay (M7) is the owner's own layer over those
        packs: named marker/polyline/polygon documents stored
        as data (nomorals.maps.overlays), rendered through the
        same overlay seam routes draw through.  Polygons ride
        as outlines — the seam has no fill — and the show
        caption says so whenever a polygon is in the overlay,
        never silently.  The verbs here are the chat-thin
        paths (one-marker add); the MapAgent + map tool carry
        the same store and renderer to the brain."""
        from ...maps.settings import map_settings

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /map packs  |  /map view <lat,lon> [zoom]  |  "
                    "/map view <west,south,east,north> [zoom]  |  "
                    "/map find <place name>  |  /map near <lat,lon>  |  "
                    "/map route <place or lat,lon> to <place or lat,lon>  |  "
                    "/map share <chat> <minutes> [exact|neighborhood|city]  |  "
                    "/map share stop [chat]  |  /map share status  |  "
                    "/map overlay list  |  /map overlay show <name>  |  "
                    "/map overlay add <name> <lat,lon> [label]  |  "
                    "/map overlay drop <name>")
        is_owner = True if message is None else self._is_operator(message)
        if not is_owner:
            return ("owner-only: Atlas renders the owner's installed "
                    "map packs on the owner's host.")
        if not map_settings().enabled:
            return ("Devon Atlas is off on this host — set "
                    "NM_MAPS_ENABLED=1 to light the map surfaces "
                    "(packs live under NM_MAPS_PACKS_DIR, or "
                    "<home>/maps/packs).")
        verb, _, rest = tail.partition(" ")
        verb = verb.lower()
        if verb == "packs":
            return self._map_packs_text()
        if verb == "view":
            return self._map_view(rest.strip(), chat_key)
        if verb == "find":
            return self._map_find(rest.strip(), chat_key)
        if verb == "near":
            return self._map_near(rest.strip())
        if verb == "route":
            return self._map_route(rest.strip(), chat_key)
        if verb == "share":
            return self._map_share(rest.strip())
        if verb == "overlay":
            return self._map_overlay(rest.strip(), chat_key)
        return (f"unknown /map verb {verb!r} — /map packs | "
                "/map view <lat,lon> [zoom] | /map find <place name> | "
                "/map near <lat,lon> | "
                "/map route <place or lat,lon> to <place or lat,lon> | "
                "/map share <chat> <minutes> [precision] | "
                "/map share stop [chat] | /map share status | "
                "/map overlay list | /map overlay show <name> | "
                "/map overlay add <name> <lat,lon> [label] | "
                "/map overlay drop <name>")

    def _map_packs_text(self) -> str:
        from ...maps.packs import verify_pack_dir

        installed = self._map_installed()
        root = self._map_packs_dir()
        if not installed:
            return (f"no map packs installed — Atlas packs live in "
                    f"{root} (built by the basemap pipeline; each pack "
                    "is a directory holding a manifest.json).")
        store = self._map_receipt_store()
        lines = [f"🗺 Devon Atlas — {len(installed)} pack(s):"]
        for entry, manifest, error in installed:
            if manifest is None:
                lines.append(f"  ✗ {entry.name} — corrupt: manifest "
                             f"unreadable ({error})")
                continue
            verdict, _receipt = verify_pack_dir(entry, store=store)
            state = "✓ verified" if verdict.ok else "✗ corrupt"
            lines.append(
                f"  {state} · {manifest.name} ({manifest.pack_id}) · "
                f"z{manifest.zoom_min}–{manifest.zoom_max} · "
                f"{manifest.tile_format} · signature "
                f"{verdict.signature_state}")
            if not verdict.ok:
                bad = list(verdict.missing) + list(verdict.mismatched)
                lines.append("      defective members: "
                             + ", ".join(bad[:6]))
            lines.append(f"      {manifest.attribution}")
        return "\n".join(lines)

    def _map_gazetteer(self):
        """(gazetteer, installed) — the local geocoder (M4) over
        the shelf, federation posture from settings."""
        from ...maps.gazetteer import Gazetteer
        from ...maps.settings import map_settings

        gazetteer = Gazetteer(
            self._map_packs_dir(), store=self._map_receipt_store(),
            federation=map_settings().geocode_federation)
        installed = [(d, m) for d, m, err in self._map_installed()
                     if m is not None]
        return gazetteer, installed

    def _map_router(self):
        """The local router (M5) over the shelf, federation
        posture from settings."""
        from ...maps.routing import Router
        from ...maps.settings import map_settings

        return Router(
            self._map_packs_dir(), store=self._map_receipt_store(),
            federation=map_settings().routing_federation)

    def _map_agent(self):
        """The MapAgent (M7) over this runtime's context — the
        overlay verbs ride the agent, so the chat surface, the
        tool plane and the brain share one store + renderer."""
        from ..map import MapAgent

        return MapAgent(self.context)

    def _map_overlay(self, spec: str, chat_key: str = "") -> str:
        """/map overlay list | show <name> | add <name> <lat,lon>
        [label] | drop <name> — the owner's own layers (M7)."""
        from ...maps.coords import validate_lat, validate_lon
        from ...maps.overlays import OverlayError, OverlayFeature

        usage = ("usage: /map overlay list  |  /map overlay "
                 "show <name>  |  /map overlay add <name> "
                 "<lat,lon> [label]  |  /map overlay drop <name>")
        toks = (spec or "").split()
        if not toks:
            return usage
        sub = toks[0].lower()
        agent = self._map_agent()
        if sub == "list":
            rows = agent.overlays()["overlays"]
            if not rows:
                return ("no overlays yet — /map overlay add "
                        "<name> <lat,lon> [label] saves a marker "
                        "overlay; richer documents are authored "
                        "through the MapAgent.")
            lines = [f"🗺 {len(rows)} overlay(s):"]
            for row in rows:
                kinds = ", ".join(row["kinds"])
                lines.append(
                    f"  • {row['name']} — "
                    f"{row['feature_count']} feature(s) ({kinds})")
            return "\n".join(lines)
        if sub == "add":
            if len(toks) < 3:
                return ("usage: /map overlay add <name> <lat,lon> "
                        "[label]")
            name = toks[1]
            try:
                nums = [float(v) for v in toks[2].split(",")]
            except ValueError:
                nums = []
            if len(nums) != 2:
                return (f"couldn't parse coordinates {toks[2]!r} "
                        "— /map overlay add <name> <lat,lon> "
                        "[label]")
            lat, lon = nums
            try:
                validate_lat(lat)
                validate_lon(lon)
            except ValueError as exc:
                return f"bad marker point: {exc}"
            label = " ".join(toks[3:])
            try:
                agent.overlay_store().create(
                    name, [OverlayFeature(
                        "marker", ((lat, lon),), label)])
            except OverlayError as exc:
                return f"couldn't save overlay {name!r}: {exc}"
            return (f"📌 overlay {name!r} saved — a marker at "
                    f"{lat:.4f},{lon:.4f}. /map overlay show "
                    f"{name} renders it over its pack.")
        if sub == "show":
            name = " ".join(toks[1:]).strip()
            if not name:
                return "usage: /map overlay show <name>"
            ticket = agent.overlay_render(name)
            if ticket["outcome"] != "ok":
                return ("🗺 " + (ticket["detail"]
                                 or "that overlay couldn't be "
                                    "rendered"))
            caption = (f"🗺 overlay {ticket['overlay']['name']} · "
                       f"{ticket['pack_name']} · "
                       f"z{ticket['zoom']}")
            for note in ticket["notes"]:
                caption += f"\n{note}"
            caption += f"\n{ticket['attribution']}"
            chat = self._ref_from_key(chat_key)
            try:
                res = self.gateway.send_file(
                    chat.platform, chat, ticket["image_path"],
                    caption=caption, kind="image",
                    name=f"overlay-{ticket['overlay']['name']}.png")
                if getattr(res, "ok", False):
                    return ""
            except Exception as exc:  # noqa: BLE001
                record_failure("runtime_live._map_overlay", exc)
            return (f"(the overlay rendered from "
                    f"{ticket['pack_name']} but the photo "
                    "couldn't be sent on this surface)")
        if sub == "drop":
            name = " ".join(toks[1:]).strip()
            if not name:
                return "usage: /map overlay drop <name>"
            if agent.overlay_store().delete(name):
                return f"overlay {name!r} dropped."
            return f"no overlay named {name!r}."
        return (f"unknown /map overlay verb {sub!r} — list | "
                "show <name> | add <name> <lat,lon> [label] | "
                "drop <name>")

    @staticmethod
    def _map_find_winner(candidates):
        """A clear winner, or None: a sole candidate found exactly
        or by prefix, or an exact top hit the runner-up does not
        tie.  Anything less is a list, not a verdict."""
        if not candidates:
            return None
        top = candidates[0]
        if len(candidates) == 1:
            return top if top.matched_as in ("exact", "prefix") \
                else None
        if top.matched_as == "exact" \
                and candidates[1].matched_as != "exact":
            return top
        return None

    @staticmethod
    def _map_find_zoom(kind: str) -> int:
        """The view zoom a found place earns, by its kind: a
        country is not viewed at street depth."""
        return {"continent": 4, "country": 5, "state": 7,
                "city": 12, "town": 13, "village": 14}.get(kind, 13)

    def _map_find(self, query: str, chat_key: str = "") -> str:
        if not query:
            return "usage: /map find <place name>"
        gazetteer, installed = self._map_gazetteer()
        if not installed:
            return ("no map packs installed — /map packs shows the "
                    "shelf, and it is empty. The geocoder indexes "
                    "installed packs only; there is no provider "
                    "fallback.")
        names = {m.pack_id: m.name for _d, m in installed}
        lines: list[str] = []
        built = gazetteer.ensure(installed)
        if built.built_any:
            lines.append(
                f"(indexed the local gazetteer first: "
                f"{built.total_entries} named places from "
                f"{len(installed)} pack(s) — it rebuilds only when "
                "a pack changes.)")
        result = gazetteer.search(query)
        if not result.candidates:
            lines.append(
                f"no place named {query!r} in the local Atlas "
                "index — " + result.note + ".")
            return "\n".join(lines)
        lines.append(
            f"🔎 {len(result.candidates)} place(s) matching "
            f"{query!r} (local Atlas index):")
        for cand in result.candidates:
            lines.append(
                f"  • {cand.name} — {cand.kind} · "
                f"{cand.lat:.4f},{cand.lon:.4f} · "
                + names.get(cand.pack_id, cand.pack_id))
        winner = self._map_find_winner(result.candidates)
        if winner is not None:
            # A clear winner also gets its map photo; the list is
            # the reply either way (a failed photo send never
            # rewrites the answer into a claim).
            self._map_view(
                f"{winner.lat:.6f},{winner.lon:.6f} "
                f"{self._map_find_zoom(winner.kind)}", chat_key)
        return "\n".join(lines)

    def _map_near(self, spec: str) -> str:
        usage = "usage: /map near <lat,lon>"
        toks = spec.split()
        if not toks:
            return usage
        try:
            nums = [float(v) for v in toks[0].split(",")]
        except ValueError:
            return f"couldn't parse coordinates {toks[0]!r} — " + usage
        if len(nums) != 2:
            return usage
        lat, lon = nums
        from ...maps.coords import validate_lat, validate_lon

        try:
            validate_lat(lat)
            validate_lon(lon)
        except ValueError as exc:
            return f"bad point: {exc}"
        gazetteer, installed = self._map_gazetteer()
        if not installed:
            return ("no map packs installed — /map packs shows the "
                    "shelf, and it is empty. The geocoder indexes "
                    "installed packs only; there is no provider "
                    "fallback.")
        names = {m.pack_id: m.name for _d, m in installed}
        lines: list[str] = []
        built = gazetteer.ensure(installed)
        if built.built_any:
            lines.append(
                f"(indexed the local gazetteer first: "
                f"{built.total_entries} named places from "
                f"{len(installed)} pack(s) — it rebuilds only when "
                "a pack changes.)")
        result = gazetteer.reverse(lat, lon)
        if not result.candidates:
            lines.append(result.note + ".")
            return "\n".join(lines)
        lines.append(
            f"📍 named places near {lat:.4f},{lon:.4f} "
            "(local Atlas index):")
        for cand in result.candidates:
            if cand.distance_km >= 1.0:
                dist = f"{cand.distance_km:.1f} km"
            else:
                dist = f"{cand.distance_km * 1000:.0f} m"
            lines.append(
                f"  • {cand.name} — {cand.kind} · {dist} · "
                + names.get(cand.pack_id, cand.pack_id))
        return "\n".join(lines)

    @staticmethod
    def _map_distance_text(meters: float) -> str:
        if meters >= 1000:
            return f"{meters / 1000:.1f} km"
        return f"{meters:.0f} m"

    @staticmethod
    def _map_duration_text(seconds: float) -> str:
        """A duration as an estimate reads: "~35 min"."""
        total = int(round(seconds))
        if total < 60:
            return f"~{total} sec"
        minutes = (total + 30) // 60
        if minutes < 60:
            return f"~{minutes} min"
        return f"~{minutes // 60} h {minutes % 60:02d} min"

    def _map_route_endpoint(self, text, gazetteer):
        """One route end as ``(point, label)`` — or ``(None,
        error)``.  Raw "lat,lon" wins first; otherwise the
        gazetteer's top hit, its display name the label the
        summary answers with."""
        from ...location.primitives import LatLon
        from ...maps.coords import validate_lat, validate_lon

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
                return None, f"bad point {raw!r}: {exc}"
            return LatLon(lat, lon), f"{lat:.4f},{lon:.4f}"
        result = gazetteer.search(raw)
        if not result.candidates:
            return None, (f"no place named {raw!r} in the local "
                          "Atlas index — " + result.note)
        top = result.candidates[0]
        return LatLon(top.lat, top.lon), top.name

    def _map_route(self, spec: str, chat_key: str = "") -> str:
        import re

        usage = ("usage: /map route <place or lat,lon> to "
                 "<place or lat,lon>")
        parts = re.split(r"\s+to\s+", (spec or "").strip(),
                         flags=re.IGNORECASE)
        if len(parts) < 2 or not parts[0].strip() \
                or not parts[-1].strip():
            return usage
        from_text = parts[0].strip()
        to_text = " to ".join(parts[1:]).strip()
        gazetteer, installed = self._map_gazetteer()
        if not installed:
            return ("no map packs installed — /map packs shows the "
                    "shelf, and it is empty. Routing runs on "
                    "installed packs only; there is no provider "
                    "fallback.")
        lines: list[str] = []
        built = gazetteer.ensure(installed)
        if built.built_any:
            lines.append(
                f"(indexed the local gazetteer first: "
                f"{built.total_entries} named places from "
                f"{len(installed)} pack(s) — it rebuilds only when "
                "a pack changes.)")
        start, from_label = self._map_route_endpoint(
            from_text, gazetteer)
        if start is None:
            lines.append(from_label + ".")
            return "\n".join(lines)
        end, to_label = self._map_route_endpoint(to_text, gazetteer)
        if end is None:
            lines.append(to_label + ".")
            return "\n".join(lines)

        def _covers(manifest, point) -> bool:
            box = manifest.bbox
            if box.crosses_antimeridian:
                return False
            return (box.south <= point.lat <= box.north
                    and box.west <= point.lon <= box.east)

        vector = [(d, m) for d, m in installed
                  if m.tile_format == "mvt"]
        both = [(d, m) for d, m in vector
                if _covers(m, start) and _covers(m, end)]
        if not both:
            if not vector:
                lines.append(
                    "routing needs a vector (mvt) pack — the "
                    "installed packs are raster, which carry no "
                    "road graph. Nothing was fetched from a "
                    "provider.")
            else:
                covering = [m.name for _d, m in vector
                            if _covers(m, start) or _covers(m, end)]
                if covering:
                    lines.append(
                        "no single installed vector pack covers "
                        "both ends — " + ", ".join(covering)
                        + " cover one end only. Routing is per-pack "
                        "and local; nothing was fetched from a "
                        "provider.")
                else:
                    lines.append(
                        "no installed vector pack covers both ends "
                        "of this route. Routing is per-pack and "
                        "local; nothing was fetched from a provider.")
            return "\n".join(lines)
        entry, manifest = sorted(
            both, key=lambda dm: (-dm[1].zoom_max, dm[1].name))[0]
        result = self._map_router().route(
            entry, manifest, start, end,
            from_label=from_label, to_label=to_label)
        if not result.ok:
            lines.append(
                f"🧭 no route {from_label} → {to_label} — "
                + result.detail)
            return "\n".join(lines)
        lines.append(
            f"🧭 {from_label} → {to_label} — "
            f"{self._map_distance_text(result.distance_m)} · "
            f"{self._map_duration_text(result.duration_s)} by road "
            f"(local Atlas routing · {manifest.name})")
        for step in result.steps:
            lines.append(
                f"  • {step.name} — "
                f"{self._map_distance_text(step.length_m)}")
        snaps = []
        if result.from_snap_m is not None \
                and result.from_snap_m >= 10:
            snaps.append(
                f"starts {result.from_snap_m:.0f} m from the "
                "nearest road")
        if result.to_snap_m is not None and result.to_snap_m >= 10:
            snaps.append(
                f"ends {result.to_snap_m:.0f} m from the nearest "
                "road")
        if snaps:
            lines.append("  (" + "; ".join(snaps) + ")")
        if not self._map_route_photo(entry, manifest, result,
                                     chat_key):
            lines.append(
                "(route found; its map photo couldn't be rendered "
                "or sent on this surface.)")
        return "\n".join(lines)

    def _map_route_photo(self, entry, manifest, result,
                         chat_key) -> bool:
        """Send the route drawn over the pack as a photo.

        A side effect in the /map find winner's shape: the text
        summary is the answer either way — a photo that cannot
        render or send changes nothing but its own presence (the
        caller notes it, in one line)."""
        import os
        import tempfile
        from pathlib import Path

        from ...maps.coords import BBox
        from ...maps.vector_render import MapOverlay, VectorRenderer

        if not result.geometry:
            return False
        lats = [p[0] for p in result.geometry]
        lons = [p[1] for p in result.geometry]
        pad_lat = max(0.01, 0.15 * (max(lats) - min(lats)))
        pad_lon = max(0.01, 0.15 * (max(lons) - min(lons)))
        try:
            bbox = BBox(west=max(-180.0, min(lons) - pad_lon),
                        south=max(-85.0, min(lats) - pad_lat),
                        east=min(180.0, max(lons) + pad_lon),
                        north=min(85.0, max(lats) + pad_lat))
        except ValueError:
            return False
        zoom = max(manifest.zoom_min,
                   min(manifest.zoom_max, self._map_zoom_for(bbox)))
        overlay = MapOverlay(
            polylines=(tuple(result.geometry),),
            markers=(result.geometry[0], result.geometry[-1]))
        renderer = VectorRenderer(
            entry, manifest, store=self._map_receipt_store())
        rendered = renderer.render(bbox, zoom, size=(768, 768),
                                   overlay=overlay)
        if rendered.image is None:
            return False
        caption = (f"🧭 {result.from_label} → {result.to_label} · "
                   f"{self._map_distance_text(result.distance_m)} · "
                   f"{self._map_duration_text(result.duration_s)}\n"
                   f"{manifest.attribution}")
        chat = self._ref_from_key(chat_key)
        tmp_path = ""
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="atlas-route-",
                                            suffix=".png")
            os.close(fd)
            Path(tmp_path).write_bytes(rendered.image)
            res = self.gateway.send_file(
                chat.platform, chat, tmp_path, caption=caption,
                kind="image", name=f"{manifest.pack_id}-route.png")
            return bool(getattr(res, "ok", False))
        except Exception as exc:  # noqa: BLE001
            record_failure("runtime_live._map_route_photo", exc)
            return False
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError as exc:  # transient send copy
                    record_failure(
                        "runtime_live._map_route_photo.unlink", exc)

    # ── live location sharing (M6) ──────────────────────────────

    def _live_broker(self):
        """The LiveLocationBroker over the location root, wired to
        the platform adapters as its sender.  State lives under
        ``<home>/location`` when a home is configured, else the
        location foundation's own root."""
        from pathlib import Path

        from ...location.live import LiveLocationBroker
        from ...location.privacy import ConsentStore, default_location_root

        broker = getattr(self, "_live_broker_obj", None)
        if broker is not None:
            return broker
        settings = getattr(self.context, "settings", None)
        home = getattr(settings, "home_path", None)
        root = (Path(home) / "location") if home \
            else default_location_root()
        broker = LiveLocationBroker(
            root, consent=ConsentStore(root),
            sender=self._live_share_send)
        self._live_broker_obj = broker
        return broker

    def _live_share_send(self, chat_key: str, loc) -> bool:
        """The broker's sender: one location pin to one chat via
        its platform adapter.  False when no adapter serves the
        platform or the send does not land — the broker records
        the failure; a pin is never claimed sent unproven."""
        from ...social.chat.base import ChatRef

        platform, _, _ = chat_key.partition(":")
        adapter = self._adapter_for(platform)
        if adapter is None:
            return False
        try:
            chat = ChatRef.parse(chat_key)
        except ValueError:
            chat = self._ref_from_key(chat_key)
        result = adapter.send_location(chat, loc.lat, loc.lon)
        return bool(getattr(result, "ok", result))

    def _live_share_card(self, chat_key: str, session) -> bool:
        """The recipient's one up-front card: that a live share
        started, at what precision, and when it stops — visible
        on the share itself, never a silent feed."""
        import time as _time

        platform, _, _ = chat_key.partition(":")
        adapter = self._adapter_for(platform)
        if adapter is None:
            return False
        from ...social.chat.base import ChatRef

        try:
            chat = ChatRef.parse(chat_key)
        except ValueError:
            chat = self._ref_from_key(chat_key)
        minutes = max(1, round(
            (session.expires_at - session.started_at) / 60))
        until = _time.strftime(
            "%H:%M", _time.localtime(session.expires_at))
        card = (
            "📍 live location sharing started — the owner is "
            "sharing their live location with this chat for the "
            f"next {minutes} min (until {until}).\n"
            f"precision: {session.precision} · updates arrive "
            "here as location pins when they move · the share "
            "stops on its own at the end of the window, or the "
            "moment the owner stops it.")
        result = adapter.send(chat, card)
        return bool(getattr(result, "ok", result))

    def _map_share(self, spec: str) -> str:
        """/map share <chat> <minutes> [precision] | stop [chat] |
        status — the owner verbs over the LiveLocationBroker."""
        import time as _time

        from ...location.live import PRECISIONS, SCOPE_LIVE_SHARE, LiveLocationError
        from ...maps.settings import map_settings

        usage = ("usage: /map share <chat> <minutes> "
                 "[exact|neighborhood|city]  |  /map share stop "
                 "[chat]  |  /map share status")
        if not map_settings().live_share_enabled:
            return ("live location sharing is off on this host — "
                    "set NM_MAPS_LIVE_SHARE=1 to light it (Atlas "
                    "being on is not share being on; a share also "
                    "needs your consent grant, which the start "
                    "command itself records).")
        toks = spec.split()
        if not toks:
            return usage
        head = toks[0].lower()
        broker = self._live_broker()
        if head == "status":
            return self._map_share_status(broker)
        if head == "stop":
            return self._map_share_stop(broker, toks[1:])
        # start: <chat-key> <minutes> [precision]
        if len(toks) < 2:
            return usage
        chat_key = toks[0]
        try:
            minutes = float(toks[1])
        except ValueError:
            return (f"minutes must be a number, got {toks[1]!r} — "
                    + usage)
        precision = toks[2].lower() if len(toks) > 2 else "exact"
        if precision not in PRECISIONS:
            return ("precision must be one of "
                    + "|".join(PRECISIONS)
                    + f", got {toks[2]!r}")
        from ...location.privacy import ConsentStore

        # The owner's own start command IS the opt-in act: record
        # the grant the session rides, then open the session.  A
        # ConsentStore over the broker's root writes the same
        # consent.json the broker's own store reads per call.
        ConsentStore(broker.root).grant("owner", SCOPE_LIVE_SHARE)
        try:
            session = broker.start_session(
                "owner", (chat_key,), ttl_s=minutes * 60.0,
                precision=precision)
        except LiveLocationError as exc:
            return f"couldn't start the share: {exc}"
        card_ok = self._live_share_card(chat_key, session)
        until = _time.strftime(
            "%H:%M", _time.localtime(session.expires_at))
        lines = [
            f"📍 live share started → {chat_key} · "
            f"{precision} precision · until {until} "
            f"({minutes:g} min).",
            "updates you send Devon as a location share are "
            "forwarded as pins; every decision is recorded "
            "(/map share status).",
        ]
        if not card_ok:
            lines.append(
                "(the recipient's share card couldn't be sent on "
                "that surface — the share itself is live; they "
                "will simply start seeing pins.)")
        return "\n".join(lines)

    def _map_share_stop(self, broker, args) -> str:
        """stop [chat] — end the owner's live sessions (all, or
        the ones feeding one chat).  The stop is durable: ended
        sessions never resume.  (A consent revoke recorded in the
        ConsentStore by any surface halts forwarding just as
        finally — the broker re-checks consent on every update —
        it just leaves the session record to expire on its own
        clock rather than ending it here.)"""
        targets = [s for s in broker.sessions("owner")]
        if args:
            chat_key = args[0]
            targets = [s for s in targets
                       if chat_key in s.recipients]
            if not targets:
                return (f"no live share is feeding {chat_key} — "
                        "nothing to stop.")
        if not targets:
            return "no live share is running — nothing to stop."
        stopped = []
        for session in targets:
            broker.stop_session(session.id)
            stopped.extend(session.recipients)
        return ("⏹ live share stopped → "
                + ", ".join(sorted(set(stopped)))
                + ". Ended shares stay ended; a new share is a "
                  "new /map share command.")

    def _map_share_status(self, broker) -> str:
        """Active shares (recipients + precision + time left) and
        the broker's most recent decisions."""
        import time as _time

        now = _time.time()
        active = broker.sessions("owner")
        lines: list[str] = []
        if active:
            lines.append("📍 live shares running:")
            for session in active:
                left = max(0, int(session.expires_at - now))
                until = _time.strftime(
                    "%H:%M", _time.localtime(session.expires_at))
                lines.append(
                    f"  • → {', '.join(session.recipients)} · "
                    f"{session.precision} · {left // 60}m "
                    f"{left % 60:02d}s left (until {until})")
        else:
            lines.append("📍 no live share is running.")
        recent = broker.recent_decisions(limit=5, subject="owner")
        if recent:
            lines.append("recent decisions:")
            for rec in recent:
                when = _time.strftime(
                    "%H:%M", _time.localtime(float(rec.get("at", 0))))
                who = ", ".join(rec.get("recipients") or []) or "—"
                lines.append(
                    f"  • {when} {rec.get('decision', '')} → {who}"
                    f" ({rec.get('reason', '')})")
        return "\n".join(lines)

    def _maybe_live_share_update(self, message) -> None:
        """The supplied-update intake for the LiveLocationBroker.

        Runs only when ALL of these hold: live sharing is enabled
        (NM_MAPS_LIVE_SHARE — the hook is dark otherwise, and the
        message flows on untouched), the message is the OWNER's
        (only the owner's supplied position is ever brokered —
        anyone else's location share is their own business, left
        to ordinary chat), and meta carries a location an adapter
        surfaced from the message itself.  The broker evaluates;
        this hook never replies and never raises (the caller also
        guards it)."""
        from ...location.primitives import LatLon
        from ...maps.settings import map_settings

        settings = map_settings()
        if not settings.enabled or not settings.live_share_enabled:
            return
        if not self._is_operator(message):
            return
        supplied = (message.meta or {}).get("location") or {}
        try:
            loc = LatLon(lat=float(supplied["lat"]),
                         lon=float(supplied["lon"]))
        except (KeyError, TypeError, ValueError):
            return  # meta claimed a location it cannot back
        platform = str(getattr(message.chat, "platform", "") or "")
        self._live_broker().evaluate(
            "owner", loc, ts=float(getattr(message, "ts", 0.0) or 0.0)
            or None, source=platform)

    @staticmethod
    def _map_zoom_for(bbox) -> int:
        """The deepest zoom whose tile count for ``bbox`` stays
        modest — the /map view default when no zoom is given."""
        import math

        from ...maps.coords import lat_to_y_float, lon_to_x_float

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

    def _map_view(self, spec: str, chat_key: str) -> str:
        import os
        import tempfile
        from pathlib import Path

        from ...maps.coords import (
            BBox,
            TileCoord,
            tile_x_to_lon,
            tile_y_to_lat,
        )
        from ...maps.render import FixtureRenderer
        from ...maps.vector_render import VectorRenderer

        usage = ("usage: /map view <lat,lon> [zoom]  |  "
                 "/map view <west,south,east,north> [zoom]")
        toks = spec.split()
        if not toks:
            return usage
        try:
            nums = [float(v) for v in toks[0].split(",")]
        except ValueError:
            return (f"couldn't parse coordinates {toks[0]!r} — "
                    + usage)
        zoom = None
        if len(toks) > 1:
            try:
                zoom = int(toks[1])
            except ValueError:
                return f"zoom must be a whole number, got {toks[1]!r}"
        if len(nums) == 2:
            lat, lon = nums
            zoom = 13 if zoom is None else zoom
            try:
                tile = TileCoord.from_latlon(lat, lon, zoom)
            except ValueError as exc:
                return f"bad view point: {exc}"
            n = 1 << zoom
            x0, x1 = max(0, tile.x - 1), min(n - 1, tile.x + 1)
            y0, y1 = max(0, tile.y - 1), min(n - 1, tile.y + 1)
            bbox = BBox(west=tile_x_to_lon(x0, zoom),
                        south=tile_y_to_lat(y1 + 1, zoom),
                        east=tile_x_to_lon(x1 + 1, zoom),
                        north=tile_y_to_lat(y0, zoom))
            view_desc = f"{lat:.4f},{lon:.4f}"
        elif len(nums) == 4:
            try:
                bbox = BBox(west=nums[0], south=nums[1],
                            east=nums[2], north=nums[3])
            except ValueError as exc:
                return f"bad view box: {exc}"
            if zoom is None:
                zoom = self._map_zoom_for(bbox)
            view_desc = (f"{nums[0]:.2f},{nums[1]:.2f} → "
                         f"{nums[2]:.2f},{nums[3]:.2f}")
        else:
            return usage

        installed = [(d, m) for d, m, err in self._map_installed()
                     if m is not None]
        if not installed:
            return ("no map packs installed — /map packs shows the "
                    "shelf, and it is empty. Atlas renders installed "
                    "packs only; there is no provider fallback.")

        def _contains(outer, inner) -> bool:
            if outer.crosses_antimeridian or inner.crosses_antimeridian:
                return False
            return (outer.south <= inner.south
                    and inner.north <= outer.north
                    and outer.west <= inner.west
                    and inner.east <= outer.east)

        def _intersects(outer, inner) -> bool:
            if inner.south > outer.north or inner.north < outer.south:
                return False
            if outer.crosses_antimeridian or inner.crosses_antimeridian:
                return True
            return not (inner.east < outer.west
                        or inner.west > outer.east)

        def _pick(pool):
            return sorted(pool, key=lambda dm: (-dm[1].zoom_max,
                                                dm[1].name))[0]

        in_zoom = [(d, m) for d, m in installed
                   if m.zoom_min <= zoom <= m.zoom_max]
        full = [(d, m) for d, m in in_zoom if _contains(m.bbox, bbox)]
        partial = [(d, m) for d, m in in_zoom
                   if _intersects(m.bbox, bbox)]
        if full:
            entry, manifest = _pick(full)
        elif partial:
            entry, manifest = _pick(partial)
        else:
            shelf = "; ".join(
                f"{m.name} (z{m.zoom_min}–{m.zoom_max}, "
                f"{m.bbox.west:.1f},{m.bbox.south:.1f} → "
                f"{m.bbox.east:.1f},{m.bbox.north:.1f})"
                for _d, m in installed)
            return (f"no installed pack covers {view_desc} at "
                    f"z{zoom} — installed: {shelf}. Atlas is "
                    "local-first: nothing was fetched from a "
                    "provider.")
        store = self._map_receipt_store()
        if manifest.tile_format == "mvt":
            renderer = VectorRenderer(entry, manifest, store=store)
        elif manifest.tile_format == "png":
            renderer = FixtureRenderer(entry, manifest, store=store)
        else:
            return (f"pack {manifest.name} uses tile format "
                    f"{manifest.tile_format!r}, which no shipped "
                    "renderer draws yet (mvt and png packs render).")
        result = renderer.render(bbox, zoom, size=(768, 768))
        if result.image is None:
            return (f"Atlas could not render {view_desc} from "
                    f"{manifest.name}: {result.receipt.detail}")
        caption = f"🗺 {manifest.name} · z{zoom}"
        if result.missing_tiles:
            caption += (f" · {len(result.missing_tiles)} tile(s) "
                        "with no data")
        if "labels skipped" in result.receipt.detail:
            caption += " · place labels unavailable on this host"
        caption += f"\n{manifest.attribution}"
        chat = self._ref_from_key(chat_key)
        tmp_path = ""
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="atlas-",
                                            suffix=".png")
            os.close(fd)
            Path(tmp_path).write_bytes(result.image)
            res = self.gateway.send_file(
                chat.platform, chat, tmp_path, caption=caption,
                kind="image", name=f"{manifest.pack_id}-z{zoom}.png")
            if getattr(res, "ok", False):
                # The photo landing IS the confirmation — no
                # narration appended (owner no-meta-narration rule).
                return ""
            return (f"(the map rendered from {manifest.name} but "
                    "the photo couldn't be sent on this surface)")
        except Exception as exc:  # noqa: BLE001
            record_failure("runtime_live._control_map", exc)
            return (f"(the map rendered from {manifest.name} but "
                    "the photo couldn't be sent on this surface)")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError as exc:  # transient send copy
                    record_failure(
                        "runtime_live._control_map.unlink", exc)

    # ── translate ────────────────────────────────────────────────────
    def _control_translate(self, tail: str, message=None) -> str:
        """/translate [to <lang>] <text> | <url> | (as a reply) |
        lang <code> | status — the translation organ, on demand.

        News and briefings already arrive in the owner's language on
        their own; this is the explicit form: text, a replied-to
        message, or a link.  The ladder's marked identity fallback is
        surfaced as-is — original text is never dressed up as a
        translation.
        """
        from ...llm.languages import language_name, normalize_code
        from ...llm.translate import translator_for

        translator = translator_for(self.context)
        owner_lang = translator.owner_language()

        def _replied_text() -> str:
            if message is None:
                return ""
            meta = getattr(message, "meta", None) or {}
            for key in ("reply_to_text", "quoted_text", "reply_text"):
                val = meta.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return ""

        def _render(outcome, source_text: str) -> str:
            if outcome.rung == "same-language":
                return (f"that's already in "
                        f"{language_name(outcome.target_lang)} — "
                        f"nothing to translate.")
            if not outcome.translated:
                return (f"{source_text}\n"
                        f"⚠ {outcome.note}")
            footer = "— translated"
            if outcome.source_lang:
                footer += f" from {language_name(outcome.source_lang)}"
            if outcome.target_lang != owner_lang:
                footer += f" to {language_name(outcome.target_lang)}"
            return f"{outcome.text}\n{footer}"

        text = (tail or "").strip()
        words = text.split()
        first = words[0].lower() if words else ""

        # /translate lang [code] — the owner-language setting itself.
        if first in ("lang", "language"):
            if len(words) < 2:
                return (f"your language: "
                        f"{language_name(owner_lang)} ({owner_lang}) — "
                        f"news, briefings and /translate land in it. "
                        f"/translate lang <code> changes it.")
            try:
                code = translator.set_owner_language(words[1])
            except ValueError as exc:
                return str(exc)
            return (f"your language is now "
                    f"{language_name(code)} ({code}) — deliveries and "
                    f"/translate will use it.")

        # /translate status — the ladder, honestly.
        if first == "status" and len(words) == 1:
            lines = [f"translation ladder — your language: "
                     f"{language_name(owner_lang)} ({owner_lang})"]
            for rung in translator.ladder_status():
                mark = "✓" if rung["available"] else "✗"
                lines.append(f"  {mark} {rung['rung']}: {rung['detail']}")
            return "\n".join(lines)

        # Target override: "to <lang>" prefix, or a leading language
        # token (/tl es …).  A bare code with nothing after it applies
        # to a replied-to message, or asks for the text.
        target = ""
        rest = text
        if first == "to" and len(words) >= 2 and normalize_code(words[1]):
            target = normalize_code(words[1])
            rest = text.split(None, 2)[2] if len(words) > 2 else ""
        elif first and normalize_code(first) and first not in ("http",):
            code = normalize_code(first)
            # a leading code only counts as a target when something
            # follows it or a reply supplies the text — a one-word
            # message that happens to be a code is likelier text.
            if len(words) > 1 or _replied_text():
                target = code
                rest = text.split(None, 1)[1] if len(words) > 1 else ""

        rest = rest.strip()
        if rest.lower().startswith(("http://", "https://")):
            url = rest.split()[0]
            res = translator.translate_url(url, target=target)
            if not res.get("ok"):
                return f"couldn't translate that page — {res.get('error')}"
            lines = []
            if res.get("title"):
                lines.append(str(res["title"]))
                if (res.get("translated")
                        and res.get("title_original")
                        and res["title_original"] != res["title"]):
                    lines.append(f"  ◦ original: {res['title_original']}")
            lines.append(str(res.get("text") or ""))
            if res.get("translated"):
                foot = "— translated"
                if res.get("source_lang"):
                    foot += f" from {language_name(res['source_lang'])}"
                foot += (" · full text" if res.get("full") else
                         f" · gist of a {res.get('words', 0)}-word page "
                         f"— the link has the whole thing")
                lines.append(foot)
            else:
                lines.append(f"⚠ {res.get('note') or 'untranslated'}")
            return "\n".join(lines)
        if rest:
            outcome = translator.translate(rest, target=target)
            return _render(outcome, rest)
        replied = _replied_text()
        if replied:
            outcome = translator.translate(replied, target=target)
            return _render(outcome, replied)
        if message is not None and getattr(message, "reply_to", ""):
            return ("I can see you replied to a message, but this "
                    "surface didn't hand me its text — paste the text "
                    "and I'll translate it.")
        return ("nothing to translate — give me text, a link, or "
                "reply to a message with /translate. "
                f"Your language: {language_name(owner_lang)} "
                f"({owner_lang}); /translate lang <code> changes it.")

    # ── news ─────────────────────────────────────────────────────────────────
    def _control_news(self, tail: str) -> str:
        from ..features import feature_enabled
        from ..news import NewsAgent, publisher_of

        if not feature_enabled(self.context, "news"):
            return "news is off. /features news on"
        # The command reply IS the delivery — NewsAgent is constructed
        # with NO notifier on purpose. NewsAgent.run() publishes through
        # a notifier when given one, and attaching it here pushed the
        # full digest to every owner channel on EVERY /news run: a
        # duplicate inside the owner's own chat, and worse, an
        # owner-channel push triggerable by any non-owner (/news is a
        # PUBLIC command).
        agent = NewsAgent(self.context)
        parts = tail.split()
        verb = parts[0].lower() if parts else "run"
        if verb == "status":
            rows = agent.recent(8)
            if not rows:
                from ...social.chat.tgbot_buttons import actionable

                return actionable(
                    "no news stored yet.",
                    [[("🗞️ Fetch the feeds", "news run")]])
            lines = ["recent news:"]
            for row in rows:
                # credit the publisher when the row carries attribution
                # (pass #83); legacy rows show their feed name as before
                lines.append(f"  [{publisher_of(row)}] {row['title'][:70]}")
            # Delivery-state census + interest health (pass #85): the
            # degraded no-brain state is VISIBLE here, never silent.
            try:
                st = agent.status()
                lines.append(
                    f"delivery: {st['undelivered']} undelivered, "
                    f"{st['pushed_today']} pushed today "
                    f"(cap {st['push_cap_per_day']}/day), "
                    f"{st['archived']} archived")
                lines.append(
                    f"interest: {st['interest']}"
                    + (" (quiet hours — pushes held)"
                       if st.get("quiet_now") else ""))
                lines.append(
                    "beats: " + ", ".join(st["beats"])
                    + (f" | pins: {', '.join(st['pins'])}"
                       if st["pins"] else ""))
                tr = st.get("translation") or {}
                if tr.get("enabled"):
                    lines.append(
                        "translation: on → "
                        f"{tr.get('effective_target') or '?'} "
                        "(stories translated before summarizing)")
                else:
                    lines.append("translation: off")
            except Exception as exc:  # noqa: BLE001 - status extras are best-effort
                record_failure("runtime_live._control_news", exc)
            return "\n".join(lines)
        if verb == "poll":
            # No notifier on this surface (see the run path's note):
            # a public command must not push to owner channels.  The
            # poll still fetches/stores/scores; high-tier stories are
            # reported held, and the scheduled news-poll job (which
            # owns a notifier) is what pushes them.
            report = agent.poll()
            return (f"news poll — {report['polled']} new stories stored "
                    f"({report['held']} undelivered, "
                    f"{report['archived']} archived low-interest); "
                    f"interest: {report['interest']}; "
                    f"sources: {len(report.get('sources_ok') or [])}/"
                    f"{report.get('sources_tried', 0)} feeds answered")
        if verb == "brief":
            report = agent.brief()
            if not report["ok"]:
                return "nothing undelivered — the digest is empty."
            return (f"news digest — {report['items']} items\n\n"
                    f"{report['digest']}")
        if verb in ("pin", "unpin"):
            topic = tail.split(None, 1)[1].strip() if len(parts) > 1 else ""
            if not topic:
                return f"usage: /news {verb} <topic>"
            profile = (agent.pin(topic) if verb == "pin"
                       else agent.unpin(topic))
            pins = profile["pins"]
            return (f"interest pins: {', '.join(pins) if pins else '(none)'}")
        if verb == "interest":
            profile = agent.interest_profile()
            return ("interest profile\nbeats: " + profile["beats_text"]
                    + "\npins: "
                    + (", ".join(profile["pins"]) or "(none)"))
        if verb == "target":
            # Docs 19+21 (N1): the digest's translation posture.
            # /news target           → inspect
            # /news target <code>    → on, translating into <code>
            # /news target on|off    → flip, keeping the stored target
            from ...llm.languages import language_name, normalize_code

            sub = parts[1].strip().lower() if len(parts) > 1 else ""
            if sub == "off":
                agent.set_news_translation(enabled=False)
                return ("news translation: off — digests compose from "
                        "the source text again.")
            if sub == "on":
                st = agent.set_news_translation(enabled=True)
                code = st["effective_target"]
                return (f"news translation: on → {language_name(code)} "
                        f"({code}) — stories are translated before "
                        "they're summarized.")
            if sub:
                code = normalize_code(sub)
                if not code:
                    return (f"unknown language: {parts[1]} — use a code "
                            "or name like yo, yoruba, fr or en.")
                agent.set_news_translation(enabled=True, target=code)
                return (f"news translation: on → {language_name(code)} "
                        f"({code}) — stories are translated before "
                        "they're summarized.")
            st = agent.news_translation()
            if not st["enabled"]:
                return ("news translation: off — /news target <code> "
                        "turns it on (e.g. /news target yo).")
            code = st["effective_target"]
            return (f"news translation: on → {language_name(code)} "
                    f"({code}) — stories are translated before they're "
                    "summarized. /news target off turns it off.")
        if verb == "run" or verb == "briefing":
            cap = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            report = agent.run(cap)
            if not report["ok"]:
                errs = "; ".join(report["errors"][:3])
                return f"news run got nothing readable ({report['items']} items)" + (f" — {errs}" if errs else "")
            # No seen-marking here (F6): agent.run() digests through
            # brief(), which already records these stories as
            # delivered in the news ledger — and the briefing slots
            # read that same ledger, so this run's stories can't
            # repeat as "fresh news" in the next slot.  (The retired
            # pulse seen-marker used to need a manual sync here.)
            tried = report.get("sources_tried", 0)
            ok_srcs = report.get("sources_ok") or []
            src_line = (f"sources: {len(ok_srcs)}/{tried} feeds answered "
                        f"({', '.join(ok_srcs[:12])}"
                        f"{'…' if len(ok_srcs) > 12 else ''})")
            stood = report.get("stood_down") or []
            stood_line = ("\n(standing down unhealthy feeds, skipped this "
                          f"run: {', '.join(stood)})" if stood else "")
            # FULL digest — no [:3000] cut; the gateway chunks it losslessly.
            return (f"news digest — {report['items']} new items (of {report['fresh']} fresh)\n{src_line}{stood_line}\n\n{report['digest']}")
        return ("usage: /news [run [n]|briefing|poll|brief|status|"
                "interest|pin <topic>|unpin <topic>|"
                "target [<code>|on|off]]")

    # ── pulse / briefings ──────────────────────────────────────────────
    def _control_pulse(self, tail: str) -> str:
        """Briefing schedule + on-demand runs. Never raises.

        /pulse                → status (all four slots)
/pulse status [slot]   → schedules + last runs
        /pulse config [slot]  → times / enabled / voice
        /pulse run [slot|all] → deliver the briefing now
        /pulse catchup [slot] → run undelivered slots
        /pulse stories [batch] [page] → page a delivered story list
        Slots: morning · afternoon · evening · midnight.
        """
        try:
            from ..morning_pulse import (
                BRIEFING_SLOTS,
                _read_slot_marker,
                _slot_prefs,
                check_slot_catchup,
                last_pulse_run,
                run_slot,
                slot_enabled,
                slot_time,
                slot_voice,
            )
            parts = (tail or "").split()
            verb = parts[0].lower() if parts else "status"
            slot = parts[1].lower() if len(parts) > 1 else "all"

            def _last_run_text(s: str) -> str:
                try:
                    m = (last_pulse_run(self.context) if s == "morning"
                         else _read_slot_marker(self.context, s))
                    ts = float(m.get("ts") or 0)
                    if not ts:
                        return "never"
                    from datetime import datetime
                    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                except Exception as exc:  # noqa: BLE001
                    record_failure("runtime_live._last_run_text", exc)
                    return "?"

            def _status_line(s: str) -> str:
                spec = BRIEFING_SLOTS[s]
                tz = str(_slot_prefs(self.context, s).get("timezone") or "")
                return (f"• {s} — {slot_time(self.context, s)} {tz}, "
                        f"{'on' if slot_enabled(self.context, s) else 'off'}, "
                        f"voice {'on' if slot_voice(self.context, s) else 'off'}, "
                        f"last run: {_last_run_text(s)} "
                        f"({spec['title']})")

            if verb == "run":
                if slot in ("all", ""):
                    lines = []
                    for s in BRIEFING_SLOTS:
                        r = run_slot(self.context, s)
                        lines.append(
                            f"• {s}: {'delivered' if r.get('delivered_text') else 'not delivered'}"
                            + (f" (text+audio)" if r.get("delivered_audio") else ""))
                    return "briefings run:\n" + "\n".join(lines)
                if slot not in BRIEFING_SLOTS:
                    return (f"unknown slot {slot!r} — "
                            f"slots: {', '.join(BRIEFING_SLOTS)}")
                r = run_slot(self.context, slot)
                if r.get("ok"):
                    return (f"{slot} briefing: "
                            f"{'delivered' if r.get('delivered_text') else 'not delivered'}"
                            + (f" (text+audio)" if r.get("delivered_audio") else "")
                            + f" — {r.get('elapsed_s', 0)}s")
                return f"{slot} briefing failed: {r.get('error', '?')}"
            if verb == "status":
                if slot in BRIEFING_SLOTS:
                    return _status_line(slot)
                if slot in ("all", ""):
                    return ("briefing schedule:\n" + "\n".join(
                        _status_line(s) for s in BRIEFING_SLOTS))
                return (f"unknown slot {slot!r} — "
                        f"slots: {', '.join(BRIEFING_SLOTS)}")
            if verb == "config":
                if slot in BRIEFING_SLOTS:
                    return _status_line(slot)
                if slot in ("all", ""):
                    return ("briefing config:\n" + "\n".join(
                        _status_line(s) for s in BRIEFING_SLOTS))
                return (f"unknown slot {slot!r} — "
                        f"slots: {', '.join(BRIEFING_SLOTS)}")
            if verb == "catchup":
                if slot in BRIEFING_SLOTS:
                    r = check_slot_catchup(self.context, slot)
                    return f"{slot} catch-up: {r}"
                if slot in ("all", ""):
                    from ..morning_pulse import check_all_catchup
                    return f"catch-up: {check_all_catchup(self.context)}"
                return (f"unknown slot {slot!r} — "
                        f"slots: {', '.join(BRIEFING_SLOTS)}")
            if verb == "stories":
                # The delivered story list's pager: /pulse stories
                # [batch] [page] — the buttons on the delivered list
                # ride exactly this grammar, and the render edits
                # the ONE live message in place (session
                # "pulse-stories").  No batch = the latest list.
                from ...social.chat.tgbot_buttons import actionable
                from ..morning_pulse import story_page_render

                batch = "latest"
                page = 1
                for tok in parts[1:]:
                    if tok.isdigit():
                        page = int(tok)
                    elif tok != "stories":
                        batch = tok
                render = story_page_render(self.context, batch, page)
                if render is None:
                    return ("no pageable story list — either none was "
                            "delivered in the last week, or that list "
                            "has expired. The next briefing brings a "
                            "fresh one.")
                return actionable(render["text"], render["buttons"],
                                  session="pulse-stories",
                                  session_edit=True)
            return ("usage: /pulse [status|config|run|catchup|stories] "
                    "[morning|afternoon|evening|midnight|all]")
        except Exception as exc:  # noqa: BLE001 — command never raises
            return f"pulse error: {exc}"

    # ── always-on research ───────────────────────────────────────────────────
    def _control_research(self, tail: str, chat_key: str = "",
                          message: Any = None) -> str:
        from ..features import feature_enabled
        from ..news import NewsAgent  # noqa: F401 - keeps imports local & symmetric
        from ..notifier import Notifier
        from ..researcher import DOMAINS, ResearchAgent

        if not feature_enabled(self.context, "research"):
            return "research is off. /features research on"
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "status"
        if verb in {"from", "ask", "docs", "done", "clear"}:
            return self._research_grounded(verb, parts[1:], chat_key, message)
        agent = ResearchAgent(self.context, notifier=Notifier(self.context, self.gateway))
        if verb == "status":
            counts: dict[str, int] = {}
            try:
                from ...storage.rowstore import attached_rows
                log = attached_rows(self.context.db, "research_log")
                if log is not None:
                    for r in log.all():
                        key = str(r.get("domain"))
                        counts[key] = counts.get(key, 0) + 1
                else:
                    for row in self.context.db.query("SELECT domain, COUNT(*) AS n FROM research_log GROUP BY domain"):
                        counts[str(row.get("domain"))] = int(row.get("n", 0))
            except Exception as exc:  # noqa: BLE001
                record_failure("runtime_live._control_research", exc)
            lines = [f"research feature: on — domains: {', '.join(sorted(DOMAINS))}"]
            lines.append(f"stored digests: {sum(counts.values())}" +
                         (f" ({', '.join(f'{d} {n}' for d, n in sorted(counts.items()))})" if counts else ""))
            lines.append(f"background loop: {'running' if agent.loop_running() else 'idle'}")
            return "\n".join(lines)
        if verb == "run":
            domain = parts[1] if len(parts) > 1 and parts[1] in DOMAINS else None
            result = agent.run_cycle(domain)
            if not result.get("ok"):
                return f"research failed: {result.get('error')}"
            return (
                f"researched ({result['domain']} / {result.get('researcher', '?')}): {result['topic']}\n"
                f"suggestion: {result['suggestion']}\n"
                f"score: {result.get('score', 0):.2f} → {result.get('status', '?')}\n"
                f"({result.get('seconds', 0)}s — {self._status_note(result.get('status'))})"
            )
        if verb in {"ideas", "queue", "pending"}:
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            rows = agent.list_proposals(limit=n)
            from ...social.chat.tgbot_buttons import actionable

            if not rows:
                return actionable(
                    "no proposals yet.",
                    [[("🔬 Run research", "research run")]])
            lines = [f"top proposals ({len(rows)}):"]
            for row in rows:
                lines.append(
                    f"· {row['id'][:8]}  {row['score']:.2f}  [{row['status']}] "
                    f"{row['domain']} — {str(row['topic'])[:52]}\n"
                    f"  {str(row['suggestion'])[:120]}"
                )
            # Approve/deny ride as DECLARED buttons (presentation
            # architecture) — the latest proposal is the buttons'
            # target; a specific one still takes its printed id
            # typed, exactly as the rows show it.
            return actionable(
                "\n".join(lines),
                [[("✅ Approve latest", "research approve latest"),
                  ("❌ Deny latest", "research deny latest")]])
        if verb == "approve" and len(parts) > 1:
            return agent.approve(parts[1])
        if verb == "deny" and len(parts) > 1:
            return agent.deny(parts[1])
        if verb == "history":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            if self.context.db is None:
                return "no database attached — research history only lives in the full runtime"
            from ...storage.rowstore import attached_rows
            log = attached_rows(self.context.db, "research_log")
            if log is not None:
                all_rows = log.all()
                all_rows.sort(key=lambda r: r.get("created_at") or 0,
                              reverse=True)
                cols = ("id", "domain", "topic", "score", "status",
                        "created_at")
                rows = [{k: r.get(k) for k in cols}
                        for r in all_rows[:max(1, min(n, 20))]]
            else:
                rows = self.context.db.query(
                    "SELECT id, domain, topic, score, status, created_at FROM research_log "
                    "ORDER BY created_at DESC LIMIT ?", (max(1, min(n, 20)),))
            if not rows:
                return "no research history yet"
            lines = ["recent research:"]
            for row in rows:
                lines.append(
                    f"· {row['id'][:8]}  {float(row.get('score') or 0):.2f}  "
                    f"[{row.get('status', 'pending')}] {row['domain']} — "
                    f"{str(row.get('topic'))[:56]}"
                )
            return "\n".join(lines)
        return ("usage: /research [run [lifestyle|tech|cyber]|status|ideas [n]|approve <id|latest>|deny <id|latest>|history [n]]\n"
                "grounded: /research from [paths] | /research ask <question> | /research docs | /research done")

    def _control_web_research(self, tail: str, *, chat_key: str = "",
                              chat: Any = None) -> str:
        """``/research`` — ad-hoc web research + loop management.

        Bare topic (or ``run <topic>``) → one-shot swarm research on the
        topic, brief back in chat. Loop-management verbs (status, ideas,
        run <domain>, approve, …) delegate to the research loop handler.
        """
        from ..features import feature_enabled
        if not feature_enabled(self.context, "research"):
            return "research is off. /features research on"
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        loop_verbs = {"status", "ideas", "queue", "pending", "approve",
                      "deny", "history", "from", "ask", "docs", "done",
                      "clear"}
        if verb in loop_verbs:
            return self._control_research(tail, chat_key=chat_key)
        if verb == "run":
            # "run <domain>" → loop cycle; "run <free topic>" → ad-hoc.
            from ..researcher import DOMAINS
            if not rest or rest.lower() in DOMAINS:
                return self._control_research(tail, chat_key=chat_key)
            topic = rest
        elif verb:
            topic = (tail or "").strip()
        else:
            return ("research what? /research <topic> — e.g. "
                    "/research agent coding")
        try:
            from ..research_digest import ResearchPipeline
            result = ResearchPipeline.run(topic, self.context)
        except Exception as exc:  # noqa: BLE001 - honest failure, not silence
            return f"research failed: {type(exc).__name__}: {exc}"
        brief = (result.get("brief") or "").strip()
        if brief:
            return self._send_long_checked(
                chat.platform, chat, brief[:6000]) if chat is not None else brief
        n_find = len((result.get("report") or {}).get("findings") or [])
        if n_find:
            return f"researched {topic!r}: {n_find} findings, brief empty."
        return (f"no solid findings on {topic!r} — the swarm came back "
                f"empty. Try rephrasing or a narrower topic.")

    # ── source-grounded research (per-chat document sessions) ──────────────

    #: mime types treated as groundable documents (besides kind == "document"
    #: attachments and text/*)
    _GROUNDED_DOC_MIMES = frozenset({
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.oasis.opendocument.text",
        "application/epub+zip",
        "application/json",
        "application/rtf",
        "text/csv",
    })

    #: extensions treated as groundable documents when the mime is missing
    _GROUNDED_DOC_EXTS = frozenset({
        ".pdf", ".txt", ".md", ".markdown", ".rst", ".rtf",
        ".doc", ".docx", ".odt", ".epub",
        ".html", ".htm", ".csv", ".tsv", ".json",
    })

    def _grounded_store(self):
        """Chat-bound grounded session store, created lazily per runtime."""
        store = getattr(self, "_grounded_sessions", None)
        if store is None:
            from pathlib import Path

            from ...research.grounded_store import GroundedSessionStore

            settings = getattr(self.context, "settings", None)
            resolve = getattr(settings, "resolve", None)
            root = resolve("grounded") if callable(resolve) else Path("data/grounded")
            store = GroundedSessionStore(root)
            self._grounded_sessions = store
        return store

    @staticmethod
    def _is_groundable_media(m: Any) -> bool:
        path = getattr(m, "path", None)
        if not path:
            return False
        if getattr(m, "kind", "") == "document":
            return True
        mime = (getattr(m, "mime", "") or "").lower()
        if mime.startswith("text/") or mime in RuntimeLiveMixin._GROUNDED_DOC_MIMES:
            return True
        from pathlib import Path

        return Path(str(path)).suffix.lower() in RuntimeLiveMixin._GROUNDED_DOC_EXTS

    def _research_grounded(self, verb: str, args: list[str],
                           chat_key: str, message: Any) -> str:
        """/research from|ask|docs|done — source-grounded Q&A over chat documents."""
        from ...research.grounded import GroundedError

        store = self._grounded_store()
        if verb == "from":
            return self._grounded_from(store, args, chat_key, message)
        if verb in ("done", "clear"):
            store.drop(chat_key)
            return "grounded session cleared — documents removed."
        session = store.get(chat_key)
        if session is None:
            return ("no grounded session for this chat yet — "
                    "attach documents with /research from first.")
        if verb == "docs":
            docs = store.list_docs(chat_key)
            if not docs:
                return "grounded session is empty — /research from to add documents."
            lines = [f"grounded documents ({len(docs)}):"]
            lines.extend(f"· {d['doc_id']} — {d['title']}" for d in docs)
            return "\n".join(lines)
        # ask
        question = " ".join(args).strip()
        if not question:
            return "usage: /research ask <question>"
        try:
            answer = session.ask(question, context=self.context)
        except GroundedError as exc:
            return f"grounded research failed: {exc}"
        return answer.render()

    def _grounded_from(self, store: Any, args: list[str],
                       chat_key: str, message: Any) -> str:
        """Ingest the triggering message's document attachments and/or tail
        file paths into the chat-bound grounded session."""
        from pathlib import Path

        from ...research.grounded import GroundedError

        media_docs = []
        if message is not None:
            for m in (getattr(message, "media", None) or []):
                if self._is_groundable_media(m):
                    media_docs.append(m)
        path_args = [a for a in args if Path(a).is_file()]
        if not media_docs and not path_args:
            return ("nothing to ground on — attach a document to the message "
                    "or pass file paths: /research from <path> ...")

        titles: list[str] = []
        errors: list[str] = []
        jobs: list[tuple[str, str, str]] = []  # (filename, source path, mime)
        for m in media_docs:
            jobs.append((getattr(m, "name", "") or Path(str(m.path)).name,
                         str(m.path), getattr(m, "mime", "") or ""))
        for p in path_args:
            jobs.append((Path(p).name, p, ""))

        for filename, source, mime in jobs:
            try:
                data = Path(source).read_bytes()
            except OSError as exc:
                errors.append(f"{filename}: unreadable ({exc})")
                continue
            try:
                store.add_upload(chat_key, data, filename, mime=mime)
            except (GroundedError, ValueError) as exc:
                errors.append(f"{filename}: {exc}")
                continue
            titles.append(filename)

        if not titles:
            detail = "; ".join(errors[:3])
            return "couldn't ingest any documents" + (f": {detail}" if detail else ".")
        reply = (f"grounded on {len(titles)} document(s): {', '.join(titles)}. "
                 f"Ask with /research ask <question>")
        if errors:
            reply += f" ({len(errors)} failed: " + "; ".join(errors[:2]) + ")"
        return reply

    # ── finance: the native-TA FinancialExpert ─────────────────────────────
    def _control_finance(self, tail: str, chat_key: str) -> str:
        """Conversational finance over free market data.

        /finance quote <symbol> [market]
        /finance analyze <symbol> [market] [timeframe]
        /finance signal <symbol> [market]
        /finance idea <symbol> [market] [--profile default|aggressive|conservative]
        /finance backtest <symbol> [market] [--profile P] [--strategy NAME]
        /finance strategies — the native strategy zoo
        /finance compare <sym1,sym2,..> [market]
        /finance watch <symbol> <above|below> <price> [market]
        /finance doctor
        """
        from ..financial_expert import FinancialExpert
        from ...integrations import sentinel_bridge as bridge

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        def _usage() -> str:
            return (
                "/finance quote <symbol> [market] — spot price, no engine\n"
                "/finance analyze <symbol> [market] [timeframe] — regime + bias\n"
                "/finance signal <symbol> [market] — directional call\n"
                "/finance idea <symbol> [market] [--profile P] — full trade plan\n"
                "/finance backtest <symbol> [market] [--profile P] [--strategy NAME]\n"
                "/finance strategies — list the strategy zoo\n"
                "/finance compare <s1,s2,..> [market]\n"
                "/finance watch <symbol> <above|below> <price> [market] — price alert\n"
                "/finance doctor — integration health\n"
                "markets: crypto (default) | forex | stocks")

        if verb in ("", "help"):
            return _usage()

        def _market(args: list[str], default: str = "crypto") -> tuple[str, list[str]]:
            if args and args[-1].lower() in ("crypto", "forex", "stocks"):
                return args[-1].lower(), args[:-1]
            return default, args

        try:
            if verb == "doctor":
                return bridge.doctor().summary_text()

            if verb == "strategies":
                rows = FinancialExpert(self.context).strategies()
                lines = ["strategy zoo (native, no submodule needed):"]
                for r in rows:
                    params = ", ".join(f"{k}={v}"
                                       for k, v in r["params"].items())
                    lines.append(f"  {r['name']:15s} [{r['kind']:9s}] "
                                 f"{r['blurb']} ({params})")
                return "\n".join(lines)

            if verb == "quote":
                toks = rest.split()
                if not toks:
                    return "usage: /finance quote <symbol> [market]"
                market, toks = _market(toks)
                q = FinancialExpert(self.context).quote(toks[0], market)
                if not q:
                    return f"no quote for {toks[0]} [{market}]"
                chg = q.get("change_pct_24h")
                chg_s = f" ({chg:+.2f}% 24h)" if isinstance(chg, (int, float)) else ""
                return (f"{q['symbol']} {q['price']:,.2f} "
                        f"{q.get('currency', '')}{chg_s} — via {q['source']}")

            if verb in ("analyze", "signal", "backtest"):
                toks = rest.split()
                if not toks:
                    return f"usage: /finance {verb} <symbol> [market] [timeframe]"
                market, toks = _market(toks)
                symbol = toks[0]
                expert = FinancialExpert(self.context)
                if verb == "analyze":
                    tf = toks[1] if len(toks) > 1 else "1h"
                    return expert.analyze(symbol, market, timeframe=tf).summary_text()
                if verb == "signal":
                    return expert.signal(symbol, market).summary_text()
                profile = "default"
                strategy = None
                for flag, slot in (("--profile", "profile"),
                                   ("--strategy", "strategy")):
                    if flag in toks:
                        try:
                            val = toks[toks.index(flag) + 1]
                        except IndexError:
                            val = None
                        if slot == "profile":
                            profile = val or profile
                        else:
                            strategy = val
                return expert.backtest(symbol, market, strategy=strategy,
                                       profile=profile).summary_text()

            if verb == "idea":
                toks = rest.split()
                if not toks:
                    return "usage: /finance idea <symbol> [market] [--profile P]"
                profile = "default"
                if "--profile" in toks:
                    i = toks.index("--profile")
                    try:
                        profile = toks[i + 1]
                    except IndexError:  # noqa: E103 - missing --profile value keeps default
                        pass
                    toks = toks[:i] + toks[i + 2:]
                market, toks = _market(toks)
                if not toks:
                    return "usage: /finance idea <symbol> [market] [--profile P]"
                return FinancialExpert(self.context).trade_idea(
                    toks[0], market, profile=profile).summary_text()

            if verb == "compare":
                toks = rest.split()
                if not toks:
                    return "usage: /finance compare <s1,s2,..> [market]"
                market, toks = _market(toks)
                symbols = [s.strip() for s in " ".join(toks).split(",") if s.strip()]
                if not symbols:
                    return "usage: /finance compare <s1,s2,..> [market]"
                return FinancialExpert(self.context).compare(symbols, market).summary_text()

            if verb == "watch":
                toks = rest.split()
                # /finance watch BTC below 60000 [crypto]
                if len(toks) < 3:
                    return "usage: /finance watch <symbol> <above|below> <price> [market]"
                market, toks = _market(toks)
                symbol, direction, price_s = toks[0], toks[1].lower(), toks[2]
                if direction not in ("above", "below"):
                    return "usage: /finance watch <symbol> <above|below> <price> [market]"
                try:
                    price = float(price_s.replace(",", ""))
                except ValueError:
                    return f"not a price: {price_s!r}"
                op = "gt" if direction == "above" else "lt"
                res = FinancialExpert(self.context).watch_price(
                    symbol, market,
                    condition={"op": op, "field": "value", "value": price},
                    name=f"{symbol.upper()} {direction} {price:,.4g}")
                if not res.get("ok"):
                    return f"couldn't create the alert: {res}"
                w = res["watcher"]
                return (f"watching {symbol.upper()} [{market}] — alert when "
                        f"price goes {direction} {price:,.4g} "
                        f"(watcher {w.get('id')}). {res.get('echo', '')}".strip())
        except bridge.SentinelUnavailable as exc:
            return str(exc)
        except bridge.SentinelError as exc:
            return f"finance error: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat must never traceback
            return f"finance error: {exc}"
        return f"unknown /finance verb {verb!r}\n{_usage()}"

    @staticmethod
    def _status_note(status: str) -> str:
        return {
            "notified": "delivered",
            "notified-unlogged": "delivered but not journaled — /research ideas will not show it",
            "pending": "valuable but over the daily cap — queued (see /research ideas)",
            "skipped": "below the quality bar — logged, not delivered",
        }.get(status, "stored")
