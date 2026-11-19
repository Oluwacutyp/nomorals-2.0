"""Sparse copy-on-write 64x64 tiles (TextureCraft port, census 3A.15).

Faithful port of upstream ``crates/texel/src/tile.rs`` (commit
37ad2112): a texture is a grid of 64x64 tiles; a tile that holds no
non-zero texel is not allocated at all, so a texture that is mostly
empty costs almost nothing. Tiles are immutable byte strings shared
between clones; a write to a shared tile copies it first
(copy-on-write), so a write to one copy never mutates the other.

Components are u16 (``[u16; 4]`` per texel, unused lanes zero). All
u16 conversions, the bilinear resampler and the remapper follow the
upstream file exactly, including the half-pixel (0.5, 0.5) sampling
origins and the sparsity-preserving rule that an empty source region
stays empty in the result.
"""
from __future__ import annotations

import struct
from collections.abc import Callable, Sequence

TILE: int = 64
TILE_AREA: int = TILE * TILE
MAX_COMPS: int = 4

_U16 = struct.Struct("<H")


def to_u16(v: float) -> int:
    """Unorm 0..1 -> u16 (NaN maps to 0, like upstream)."""
    if v != v:  # NaN
        return 0
    if v <= 0.0:
        return 0
    if v >= 1.0:
        return 65535
    return int(v * 65535.0 + 0.5)


def from_u16(v: int) -> float:
    return v / 65535.0


def to_u16_signed(v: float) -> int:
    """Signed -1..1 -> u16 (NaN maps to 0)."""
    if v != v:
        return 0
    if v <= -1.0:
        return 0
    if v >= 1.0:
        return 65535
    return int((v + 1.0) * 32767.5 + 0.5)


def from_u16_signed(v: int) -> float:
    return v / 32767.5 - 1.0


def _tile_bytes(comps: int) -> bytes:
    return bytes(TILE_AREA * comps * 2)


class TileImage:
    """A sparse image of 64x64 u16 tiles.

    ``tiles`` holds ``Optional[bytes]`` per tile slot; ``None`` means
    the tile is entirely zero and unallocated. Cloning shares tile
    bytes objects; :meth:`tile_mut` / :meth:`set` copy a tile before
    writing it when it is shared with another image.
    """

    __slots__ = ("width", "height", "comps", "tiles_x", "tiles_y",
                 "tiles", "_shared", "_owned")

    def __init__(self, width: int, height: int, comps: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        if comps < 1 or comps > MAX_COMPS:
            raise ValueError("comps must be 1..=4")
        self.width = int(width)
        self.height = int(height)
        self.comps = int(comps)
        self.tiles_x = (self.width + TILE - 1) // TILE
        self.tiles_y = (self.height + TILE - 1) // TILE
        self.tiles: list[bytes | None] = [None] * (self.tiles_x * self.tiles_y)
        self._shared: set[int] = set()
        self._owned: set[int] = set()

    # -- construction helpers -------------------------------------------------
    @classmethod
    def from_tiles(cls, width: int, height: int, comps: int,
                   tiles: Sequence[bytes | None]) -> TileImage:
        img = cls(width, height, comps)
        if len(tiles) != len(img.tiles):
            raise ValueError("tile count does not match dimensions")
        img.tiles = list(tiles)
        return img

    def clone(self) -> TileImage:
        """Share every allocated tile (copy-on-write)."""
        other = TileImage.__new__(TileImage)
        other.width = self.width
        other.height = self.height
        other.comps = self.comps
        other.tiles_x = self.tiles_x
        other.tiles_y = self.tiles_y
        other.tiles = list(self.tiles)
        other._shared = set()
        other._owned = set()
        for i, t in enumerate(self.tiles):
            if t is not None:
                self._shared.add(i)
                self._owned.discard(i)
                other._shared.add(i)
        return other

    # -- access ---------------------------------------------------------------
    def _slot(self, x: int, y: int) -> int | None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return None
        return (y // TILE) * self.tiles_x + (x // TILE)

    def is_allocated(self, tx: int, ty: int) -> bool:
        if tx < 0 or ty < 0 or tx >= self.tiles_x or ty >= self.tiles_y:
            return False
        return self.tiles[ty * self.tiles_x + tx] is not None

    def allocated(self) -> list[tuple[int, int]]:
        return [(i % self.tiles_x, i // self.tiles_x)
                for i, t in enumerate(self.tiles) if t is not None]

    def allocated_count(self) -> int:
        return sum(1 for t in self.tiles if t is not None)

    def bytes(self) -> int:
        """Allocated storage in bytes (upstream ``bytes()``)."""
        return self.allocated_count() * TILE_AREA * self.comps * 2

    def tile_handle(self, tx: int, ty: int) -> bytes | None:
        """The raw tile bytes object (identity is meaningful: clones
        share the same object until a write un-shares it)."""
        if tx < 0 or ty < 0 or tx >= self.tiles_x or ty >= self.tiles_y:
            return None
        return self.tiles[ty * self.tiles_x + tx]

    def tile(self, tx: int, ty: int) -> bytes | None:
        return self.tile_handle(tx, ty)

    def get(self, x: int, y: int) -> list[int]:
        """Texel as ``[u16; 4]`` (unused lanes zero), like upstream."""
        slot = self._slot(x, y)
        out = [0, 0, 0, 0]
        if slot is None:
            return out
        t = self.tiles[slot]
        if t is None:
            return out
        off = ((y % TILE) * TILE + (x % TILE)) * self.comps * 2
        for c in range(self.comps):
            out[c] = _U16.unpack_from(t, off + c * 2)[0]
        return out

    def set(self, x: int, y: int, values: Sequence[int]) -> None:
        slot = self._slot(x, y)
        if slot is None:
            return
        tile = self._ensure_mut(slot)
        off = ((y % TILE) * TILE + (x % TILE)) * self.comps * 2
        for c in range(min(self.comps, len(values))):
            _U16.pack_into(tile, off + c * 2, int(values[c]) & 0xFFFF)

    def set_tile(self, tx: int, ty: int, data: bytes) -> None:
        if tx < 0 or ty < 0 or tx >= self.tiles_x or ty >= self.tiles_y:
            return
        if len(data) != TILE_AREA * self.comps * 2:
            raise ValueError("tile data has the wrong length")
        self.tiles[ty * self.tiles_x + tx] = bytes(data)
        self._shared.discard(ty * self.tiles_x + tx)
        self._owned.discard(ty * self.tiles_x + tx)

    def tile_mut(self, tx: int, ty: int,
                 fill: Sequence[int] | None = None) -> bytearray:
        """Mutable working copy of a tile (copy-on-write). The returned
        bytearray is stored back immediately, so edits to it are the
        tile's contents; when ``fill`` is given the tile is filled
        with that per-texel pattern first."""
        if tx < 0 or ty < 0 or tx >= self.tiles_x or ty >= self.tiles_y:
            raise ValueError("tile out of range")
        slot = ty * self.tiles_x + tx
        tile = self._ensure_mut(slot)
        if fill is not None:
            vals = [int(v) & 0xFFFF for v in fill]
            vals = (vals + [0] * self.comps)[: self.comps]
            for i in range(TILE_AREA):
                for c in range(self.comps):
                    _U16.pack_into(tile, (i * self.comps + c) * 2, vals[c])
        return tile

    def _ensure_mut(self, slot: int) -> bytearray:
        """Return a mutable, privately-owned tile for ``slot``.

        A tile already owned by this image is returned as-is (no
        copy); a shared or immutable tile is copied first, so a
        write here never mutates a clone's tile."""
        t = self.tiles[slot]
        if t is None:
            tile = bytearray(_tile_bytes(self.comps))
            self.tiles[slot] = tile  # type: ignore[assignment]
            self._owned.add(slot)
            return tile
        if slot in self._owned and isinstance(t, bytearray):
            return t
        tile = bytearray(t)
        self.tiles[slot] = tile  # type: ignore[assignment]
        self._shared.discard(slot)
        self._owned.add(slot)
        return tile

    def prune_empty(self) -> None:
        """Free tiles that hold only zeroes."""
        for i, t in enumerate(self.tiles):
            if t is not None and not any(t):
                self.tiles[i] = None
                self._shared.discard(i)
                self._owned.discard(i)

    def any_allocated_in(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Whether a bilinear read anywhere in the texel box can touch
        an allocated tile (upstream ``any_allocated_in``)."""
        def span(lo: float, hi: float, n: int) -> tuple[int, int]:
            a = max(0.0, lo - 1.0)
            b = max(0.0, hi + 1.0)
            return (int(a) // TILE, min(int(b) // TILE, max(0, n - 1)))
        ax, bx = span(min(x0, x1), max(x0, x1), self.tiles_x)
        ay, by = span(min(y0, y1), max(y0, y1), self.tiles_y)
        for sy in range(ay, by + 1):
            for sx in range(ax, bx + 1):
                if self.is_allocated(sx, sy):
                    return True
        return False

    # -- resampling -----------------------------------------------------------
    def resampled(self, width: int, height: int,
                  empty: Sequence[int] | None = None) -> TileImage:
        """Bilinear resample (upstream ``resampled``), sparsity
        preserving: a target tile is allocated only when the source
        region it samples from holds data. ``empty`` is the per-
        component fill for unallocated source regions."""
        out = TileImage(width, height, self.comps)
        if self.allocated_count() == 0 or width <= 0 or height <= 0:
            return out
        for ty in range(out.tiles_y):
            for tx in range(out.tiles_x):
                tile = self.resampled_tile(width, height, tx, ty, empty)
                if tile is not None:
                    out.set_tile(tx, ty, tile)
        return out

    def resampled_tile(self, width: int, height: int, tx: int, ty: int,
                       empty: Sequence[int] | None = None
                       ) -> bytes | None:
        """One target tile of a bilinear resample (upstream
        ``resampled_tile``)."""
        empty_vals = [0] * self.comps if empty is None else \
            [int(v) & 0xFFFF for v in empty]
        empty_vals = (empty_vals + [0] * self.comps)[: self.comps]
        sw, sh = float(self.width), float(self.height)
        dw, dh = max(1, width), max(1, height)
        x0 = tx * TILE * sw / dw
        y0 = ty * TILE * sh / dh
        x1 = (tx + 1) * TILE * sw / dw
        y1 = (ty + 1) * TILE * sh / dh
        if not self.any_allocated_in(x0, y0, x1, y1):
            return None
        tile = bytearray(_tile_bytes(self.comps))
        for ly in range(TILE):
            gy = ty * TILE + ly
            if gy >= dh:
                break
            fy = (gy + 0.5) * sh / dh
            for lx in range(TILE):
                gx = tx * TILE + lx
                if gx >= dw:
                    break
                fx = (gx + 0.5) * sw / dw
                vals = bilinear(self, fx, fy, empty_vals)
                off = (ly * TILE + lx) * self.comps * 2
                for c in range(self.comps):
                    _U16.pack_into(tile, off + c * 2, vals[c])
        return bytes(tile)

    def remapped_tile(self, width: int, height: int, tx: int, ty: int,
                      empty: Sequence[int],
                      mapping: Callable[[float, float],
                                        tuple[float, float] | None]
                      ) -> bytes | None:
        """One target tile of a remap (upstream ``remapped_tile``):
        the centre of destination texel (x, y) reads this image
        bilinearly at ``mapping(x + 0.5, y + 0.5)``; ``None`` when no
        texel of the tile reads an allocated source texel."""
        if self.allocated_count() == 0 or self.width == 0 or self.height == 0:
            return None
        empty_vals = [int(v) & 0xFFFF for v in empty]
        empty_vals = (empty_vals + [0] * self.comps)[: self.comps]
        tile = bytearray(_tile_bytes(self.comps))
        any_read = False
        for ly in range(min(TILE, max(0, height - ty * TILE))):
            for lx in range(min(TILE, max(0, width - tx * TILE))):
                off = (ly * TILE + lx) * self.comps * 2
                src = mapping(float(tx * TILE + lx) + 0.5,
                              float(ty * TILE + ly) + 0.5)
                if src is None or not _finite(src[0]) or not _finite(src[1]):
                    for c in range(self.comps):
                        _U16.pack_into(tile, off + c * 2, empty_vals[c])
                    continue
                vals, hit = bilinear(self, src[0], src[1], empty_vals,
                                     want_any=True)
                any_read = any_read or hit
                for c in range(self.comps):
                    _U16.pack_into(tile, off + c * 2, vals[c])
        if not any_read:
            return None
        return bytes(tile)


def _finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def bilinear(img: TileImage, fx: float, fy: float,
             empty: Sequence[int] | None = None,
             want_any: bool = False
             ) -> list[int] | tuple[list[int], bool]:
    """Bilinear sample at fractional texel coordinates (upstream
    ``bilinear``): the sample position is shifted by -0.5 first, the
    four neighbours are read with unallocated tiles yielding
    ``empty``, and the result rounds with +0.5. Returns the values,
    or ``(values, any_allocated_read)`` when ``want_any`` is set."""
    empty_vals = [0] * img.comps if empty is None else \
        [int(v) & 0xFFFF for v in empty]
    empty_vals = (empty_vals + [0] * img.comps)[: img.comps]
    fx = max(0.0, fx - 0.5)
    fy = max(0.0, fy - 0.5)
    ix = int(fx // 1)
    iy = int(fy // 1)
    wx = fx - ix
    wy = fy - iy
    any_read = False

    def get(x: int, y: int) -> list[int]:
        nonlocal any_read
        x = min(max(0, x), img.width - 1)
        y = min(max(0, y), img.height - 1)
        if img.is_allocated(x // TILE, y // TILE):
            any_read = True
            return img.get(x, y)
        return (empty_vals + [0, 0, 0, 0])[:4]

    a = get(ix, iy)
    b = get(ix + 1, iy)
    c = get(ix, iy + 1)
    d = get(ix + 1, iy + 1)
    out: list[int] = []
    for k in range(img.comps):
        top = a[k] * (1.0 - wx) + b[k] * wx
        bot = c[k] * (1.0 - wx) + d[k] * wx
        out.append(max(0, min(65535, int(top * (1.0 - wy) + bot * wy + 0.5))))
    if want_any:
        return out, any_read
    return out
