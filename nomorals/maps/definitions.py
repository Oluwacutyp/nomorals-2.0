"""Pack + source definitions: the data the M2 pipeline runs on.

Doc 21 §3 (superseding doc 19's per-source TileStore, S2): the
basemap is a set of Devon-built *releases*.  This module is the
registry of what can be built — sources (where bytes come from,
under whose attribution and licence) and pack definitions (which
source, which builder, which extent and zooms) — as **data**, so
the pipeline in :mod:`nomorals.maps.pipeline` stays generic over
them and a new pack is a new row, never new code.

Hash discipline, stated honestly: a source whose publisher ships
a stable checksum gets ``sha256`` pinned here and the pipeline
refuses a mismatch by name.  Rolling extracts (Geofabrik's
``-latest``) publish only an MD5 sidecar that changes daily, so
they are recorded with ``sha256 == ""`` — the pipeline pins the
digest it actually fetched into the build record and the pack's
release-registry entry (pin-after-first-fetch), and verifies the
publisher MD5 sidecar when one is declared.  Nothing here claims
a pinned hash that was never pinned.
"""

from __future__ import annotations

from dataclasses import dataclass

from .coords import BBox

__all__ = [
    "BUILDER_NE_RASTER", "BUILDER_PLANETILER", "PACKS",
    "PLANETILER_AUX_SOURCE_IDS", "SOURCES",
    "MapSource", "PackDefinition", "get_pack", "get_source",
]

#: Builder ids the pipeline dispatches on (see pipeline.BUILDERS).
BUILDER_PLANETILER = "planetiler-mbtiles"
BUILDER_NE_RASTER = "ne-raster"

#: Planetiler's own auxiliary inputs, as registered sources: the
#: pipeline acquires them into the build's download directory
#: (through the proxy-honouring fetcher) before invoking the JVM,
#: because a JVM on a proxied host cannot fetch them itself.
#: The filenames are the exact names Planetiler expects in its
#: download directory.
PLANETILER_AUX_SOURCE_IDS = (
    "planetiler-aux-lakelines",
    "planetiler-aux-water",
    "planetiler-aux-naturalearth",
    "planetiler-aux-tileweights",
)


@dataclass(frozen=True)
class MapSource:
    """One downloadable input: where, whose, under what terms."""

    source_id: str
    name: str
    url: str
    attribution: str
    license: str
    #: Pinned SHA-256 of the artifact, or "" when the publisher
    #: ships a rolling file (pin-after-first-fetch, see module doc).
    sha256: str = ""
    #: Optional publisher checksum sidecar (MD5) the acquire stage
    #: verifies when the artifact itself is unpinned.
    md5_url: str = ""
    filename: str = ""

    def artifact_name(self) -> str:
        """The local filename this source's download lands as."""
        if self.filename:
            return self.filename
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class PackDefinition:
    """One buildable pack: source(s) + builder + extent + zooms.

    ``zoom_max`` is what a standard build produces;
    ``stretch_zoom_max`` names the deeper zoom the doc asks for
    beyond the standard build (doc 21 §3: Pack A z0–z15/16,
    Pack B z0–z7/8) — recorded here so the gap between a bounded
    build and the full ask is data, never silently dropped.
    """

    pack_id: str
    name: str
    source_id: str
    builder: str
    tile_format: str
    bbox: BBox
    zoom_min: int
    zoom_max: int
    stretch_zoom_max: int = 0
    extra_source_ids: tuple[str, ...] = ()
    notes: str = ""

    def source_ids(self) -> tuple[str, ...]:
        return (self.source_id,) + tuple(self.extra_source_ids)


_NIGERIA_BBOX = BBox(2.6, 4.2, 14.7, 13.9)
_LAGOS_BBOX = BBox(2.65, 6.35, 4.35, 6.95)
_WORLD_BBOX = BBox(-180.0, -85.05112878, 180.0, 85.05112878)

SOURCES: tuple[MapSource, ...] = (
    MapSource(
        source_id="geofabrik-nigeria",
        name="Geofabrik Nigeria extract (OpenStreetMap)",
        url="https://download.geofabrik.de/africa/nigeria-latest.osm.pbf",
        attribution="© OpenStreetMap contributors",
        license="ODbL 1.0",
        md5_url=("https://download.geofabrik.de/africa/"
                 "nigeria-latest.osm.pbf.md5"),
        filename="nigeria-latest.osm.pbf"),
    MapSource(
        source_id="ne-land",
        name="Natural Earth 10m land (GeoJSON)",
        url=("https://raw.githubusercontent.com/nvkelso/"
             "natural-earth-vector/master/geojson/ne_10m_land.geojson"),
        attribution="Natural Earth",
        license="Public domain (Natural Earth terms)"),
    MapSource(
        source_id="ne-lakes",
        name="Natural Earth 10m lakes (GeoJSON)",
        url=("https://raw.githubusercontent.com/nvkelso/"
             "natural-earth-vector/master/geojson/ne_10m_lakes.geojson"),
        attribution="Natural Earth",
        license="Public domain (Natural Earth terms)"),
    MapSource(
        source_id="ne-admin0-lines",
        name="Natural Earth 10m admin-0 boundary lines (GeoJSON)",
        url=("https://raw.githubusercontent.com/nvkelso/"
             "natural-earth-vector/master/geojson/"
             "ne_10m_admin_0_boundary_lines_land.geojson"),
        attribution="Natural Earth",
        license="Public domain (Natural Earth terms)"),
    MapSource(
        source_id="planetiler-jar",
        name="Planetiler tile builder (single jar)",
        url=("https://github.com/onthegomap/planetiler/releases/"
             "download/v0.9.0/planetiler.jar"),
        attribution="Planetiler contributors (Apache-2.0)",
        license="Apache-2.0",
        sha256=("b75619e4960459db056971b5d20d9414c8940e5c0e1521"
                "b01d43b9d6af3a8795"),
        filename="planetiler.jar"),
    MapSource(
        source_id="planetiler-aux-lakelines",
        name="OSM lake centerlines (Planetiler auxiliary input)",
        url=("https://github.com/acalcutt/osm-lakelines/releases/"
             "download/v12/lake_centerline.shp.zip"),
        attribution="© OpenStreetMap contributors",
        license="ODbL 1.0",
        filename="lake_centerline.shp.zip"),
    MapSource(
        source_id="planetiler-aux-water",
        name="OSM water polygons (Planetiler auxiliary input)",
        url=("https://osmdata.openstreetmap.de/download/"
             "water-polygons-split-3857.zip"),
        attribution="© OpenStreetMap contributors",
        license="ODbL 1.0",
        filename="water-polygons-split-3857.zip"),
    MapSource(
        source_id="planetiler-aux-naturalearth",
        name="Natural Earth vector sqlite (Planetiler auxiliary input)",
        url=("https://naciscdn.org/naturalearth/packages/"
             "natural_earth_vector.sqlite.zip"),
        attribution="Natural Earth",
        license="Public domain (Natural Earth terms)",
        filename="natural_earth_vector.sqlite.zip"),
    MapSource(
        source_id="planetiler-aux-tileweights",
        name="OSM tile weights (Planetiler auxiliary input)",
        url=("https://raw.githubusercontent.com/onthegomap/"
             "planetiler/main/layerstats/top_osm_tiles.tsv.gz"),
        attribution="© OpenStreetMap contributors",
        license="ODbL 1.0",
        filename="tile_weights.tsv.gz"),
)

PACKS: tuple[PackDefinition, ...] = (
    PackDefinition(
        pack_id="pack-a-nigeria",
        name="Devon Atlas — Nigeria (Pack A)",
        source_id="geofabrik-nigeria",
        builder=BUILDER_PLANETILER,
        tile_format="mvt",
        bbox=_NIGERIA_BBOX,
        zoom_min=0, zoom_max=14, stretch_zoom_max=16,
        notes=("Regional basemap, doc 21 §3 Pack A (z0–z15/16): "
               "the standard build is z0–z14 country-wide; z15–16 "
               "land as bounded urban addenda (see pack-a-lagos) "
               "because a country-wide z15 MVT build multiplies "
               "size/time for detail only cities carry.")),
    PackDefinition(
        pack_id="pack-a-lagos",
        name="Devon Atlas — Lagos detail (Pack A addendum)",
        source_id="geofabrik-nigeria",
        builder=BUILDER_PLANETILER,
        tile_format="mvt",
        bbox=_LAGOS_BBOX,
        zoom_min=15, zoom_max=15, stretch_zoom_max=16,
        notes=("Bounded urban addendum over the Lagos metro bbox "
               "— the trip/region-pack pattern applied to depth. "
               "The doc asks z15–16; Planetiler 0.9 hard-caps "
               "maxzoom at 15 (\"Max zoom must be <= 15\"), so the "
               "build is z15 and z16 is the named stretch, served "
               "by overzooming the z15 vector tiles — the "
               "standard MVT rendering practice.")),
    PackDefinition(
        pack_id="pack-b-world",
        name="Devon Atlas — World overview (Pack B)",
        source_id="ne-land",
        builder=BUILDER_NE_RASTER,
        tile_format="png",
        bbox=_WORLD_BBOX,
        zoom_min=0, zoom_max=7, stretch_zoom_max=8,
        extra_source_ids=("ne-lakes", "ne-admin0-lines"),
        notes=("World overview, doc 21 §3 Pack B (z0–z7/8): "
               "Devon-rendered raster from Natural Earth 10m "
               "vectors (land, lakes, admin-0 boundaries) — PNG "
               "tiles the M1 renderer already composites; z8 is "
               "the named stretch beyond the standard build.")),
)


def get_source(source_id: str) -> MapSource:
    """The registered source, or a KeyError naming the miss."""
    for source in SOURCES:
        if source.source_id == source_id:
            return source
    raise KeyError("unknown map source: " + repr(source_id))


def get_pack(pack_id: str) -> PackDefinition:
    """The registered pack definition, or a KeyError naming it."""
    for pack in PACKS:
        if pack.pack_id == pack_id:
            return pack
    raise KeyError("unknown map pack: " + repr(pack_id))
