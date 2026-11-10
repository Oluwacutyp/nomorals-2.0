"""MBTiles reading: the packaging bridge from builders to packs.

Planetiler (and most tile toolchains) emit MBTiles — one SQLite
file whose ``tiles`` table stores tile bytes in **TMS** rows
(``tile_row`` counts north from the Mercator south edge).  A Devon
pack is a directory in **XYZ** layout (``tiles/{z}/{x}/{y}.mvt``,
row counting south from the north edge — the layout
:func:`nomorals.maps.manifest.tile_relpath` pins and the
renderers read).  This module is the whole conversion: stream the
table with the stdlib ``sqlite3``, flip each row, write each tile
byte-for-byte.  MVT payloads stay exactly as the builder emitted
them (gzip-compressed protobuf, the MBTiles convention) — the
pack manifest hashes what is on disk, and M3's renderer gunzips
at the render boundary.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

from .manifest import PackError, tile_relpath

__all__ = ["extract_mbtiles", "iter_mbtiles", "mbtiles_metadata"]


def _connect(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise PackError("mbtiles file does not exist: " + str(path))
    return sqlite3.connect("file:" + str(path) + "?mode=ro",
                           uri=True)


def mbtiles_metadata(path: Path | str) -> dict[str, str]:
    """The MBTiles ``metadata`` table as a plain dict (may be {})."""
    conn = _connect(Path(path))
    try:
        try:
            rows = conn.execute(
                "SELECT name, value FROM metadata").fetchall()
        except sqlite3.Error:
            return {}
        return {str(name): str(value) for name, value in rows}
    finally:
        conn.close()


def iter_mbtiles(path: Path | str) -> Iterator[tuple[int, int, int, bytes]]:
    """Yield ``(z, x, y_xyz, tile_bytes)`` for every stored tile.

    Rows are MBTiles/TMS on disk; the yielded ``y`` is already
    flipped to the XYZ row the pack layout uses.
    """
    conn = _connect(Path(path))
    try:
        cursor = conn.execute(
            "SELECT zoom_level, tile_column, tile_row, tile_data "
            "FROM tiles")
        for z, x, row_tms, data in cursor:
            z = int(z)
            y = (1 << z) - 1 - int(row_tms)
            yield z, int(x), y, bytes(data)
    finally:
        conn.close()


def extract_mbtiles(path: Path | str, out_dir: Path | str,
                    tile_format: str = "mvt") -> int:
    """Write every tile of an MBTiles file into a pack layout.

    Returns the number of tiles written.  An MBTiles with no
    ``tiles`` table, or with zero tiles, is a named PackError —
    an empty basemap is a failed build, never a pack.
    """
    root = Path(out_dir)
    count = 0
    try:
        for z, x, y, data in iter_mbtiles(path):
            dest = root / tile_relpath(z, x, y, tile_format)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            count += 1
    except sqlite3.Error as exc:
        raise PackError(
            "mbtiles unreadable: " + str(path) + ": " + str(exc)
        ) from exc
    if count == 0:
        raise PackError("mbtiles holds no tiles: " + str(path))
    return count
