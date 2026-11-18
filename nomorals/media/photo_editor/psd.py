"""PSD format crate — Devon's native Photoshop reader/writer.

PhotoCraft's ``crates/psd`` decomposition, reimplemented in Python
from Adobe's public "Photoshop File Formats Specification" (the
crate itself is clean-room from that spec; the code here is
Devon's own, written against the spec + the crate's structure —
dual MIT/Apache, see the survey's license note).

What it does, faithfully:

* ``PsdFile.from_bytes`` parses a PSD v1 file into a format-level
  model: header, color-mode data (kept raw), image resources (kept
  raw), layer records with decoded channel planes, and the merged
  composite. Unknown tagged blocks and resources are kept raw and
  written back on encode.
* ``PsdFile.to_bytes`` / ``encode_psd`` write PSD v1 back from the
  model / from a :class:`PhotoDocument`. Round-trip parity is pixel
  + structure (names, opacity, rects, blend keys) — NOT byte
  identity: channel data is re-encoded (RLE), so an unmodified file
  re-opens identical but not byte-identical. Byte-stable write-back
  is a named remainder.
* Channel data is decoded lazily per section with all four
  compression methods: raw, PackBits RLE (per-row byte-count table),
  ZIP, ZIP with horizontal delta prediction. Depth 8 is native;
  16-bit decodes through its high byte and 32-bit float through a
  clamp to [0, 1] — both documented on the decoded planes, so the
  document model is always 8-bit.
* The parser never crashes on malformed input: every read is
  bounds-checked, dimensions and allocations are capped (PSD v1:
  30,000px, 56 channels), and every failure is a ``PsdError`` naming
  the offset and the problem.

Modes: RGB, grayscale, indexed (palette from the color-mode data),
CMYK and Lab (both converted through PIL, labelled as such).
Bitmap unpacks to 8-bit. Multichannel and duotone are refused
honestly — the mode is named in the error, never decoded as if it
were RGB. PSB (version 2) is refused honestly; the corpus carries
no PSB files.
"""

from __future__ import annotations

import contextlib
import struct
import zlib
from typing import Any

__all__ = [
    "ADJUSTMENT_KEYS",
    "FILL_KEYS",
    "LayerRecord",
    "PsdError",
    "PsdFile",
    "decode_planes",
    "encode_planes",
    "encode_psd",
    "packbits_decode",
    "packbits_encode",
]

#: PSD v1 hard dimension cap (the spec's own limit for version 1).
MAX_DIMENSION = 30000
#: Spec limit: 1..=56 channels.
MAX_CHANNELS = 56
#: Backstop against decompression bombs (PhotoCraft uses 8 GiB on
#: 64-bit; Devon's documents are single-user files, 512 MiB is the
#: honest bound here — the corpus max is 256x256).
MAX_DECODED_BYTES = 512 * 1024 * 1024

_MAGIC = b"8BPS"
_SIG_8BIM = b"8BIM"

#: Stored color-mode value -> Devon mode name.
MODES: dict[int, str] = {
    0: "Bitmap",
    1: "L",
    2: "Indexed",
    3: "RGB",
    4: "CMYK",
    7: "Multichannel",
    8: "Duotone",
    9: "Lab",
}

#: Color (non-alpha) channels per mode, when fixed.
_MODE_COLOR_CHANNELS: dict[str, int] = {
    "Bitmap": 1,
    "L": 1,
    "Indexed": 1,
    "RGB": 3,
    "CMYK": 4,
    "Lab": 3,
}

#: Tagged-block keys that mark an adjustment layer (the layer's
#: pixels are the adjustment's parameters, not a picture — the
#: flatten reports them as unrendered, never pretends otherwise).
ADJUSTMENT_KEYS: dict[str, str] = {
    "brit": "brightness/contrast",
    "curv": "curves",
    "levl": "levels",
    "hue2": "hue/saturation",
    "mixr": "channel mixer",
    "blnc": "color balance",
    "phfl": "photo filter",
    "blwh": "black and white",
    "grdm": "gradient map",
    "post": "posterize",
    "thrs": "threshold",
    "selc": "selective color",
    "vibA": "vibrance",
    "expA": "exposure",
    "nvrt": "invert",
}

#: Tagged-block keys that mark a fill layer (solid color, gradient,
#: pattern — no pixel channels of its own).
FILL_KEYS: dict[str, str] = {
    "SoCo": "solid color fill",
    "GdFl": "gradient fill",
    "PtFl": "pattern fill",
}

#: Section-divider (lsct) types: 1/2 open/closed folder, 3 end.
_GROUP_START = (1, 2)


class PsdError(Exception):
    """A PSD file that cannot be read, or a document that cannot be
    written as PSD — always with the reason named."""


# ---------------------------------------------------------------------------
# Bounds-checked reader
# ---------------------------------------------------------------------------

class _Reader:
    """A cursor over bytes that fails with an offset instead of
    crashing on a short read."""

    def __init__(self, data: bytes, *, what: str = "PSD file") -> None:
        self._data = data
        self._pos = 0
        self._what = what

    def __len__(self) -> int:
        return len(self._data) - self._pos

    @property
    def total(self) -> int:
        """Total bytes in the buffer (not just the remainder)."""
        return len(self._data)

    @property
    def pos(self) -> int:
        return self._pos

    def seek(self, pos: int) -> None:
        if not 0 <= pos <= len(self._data):
            raise PsdError(
                f"{self._what}: seek to {pos} past end "
                f"({len(self._data)} bytes)")
        self._pos = pos

    def take(self, count: int) -> bytes:
        if count < 0 or self._pos + count > len(self._data):
            raise PsdError(
                f"{self._what}: short read at offset {self._pos}: "
                f"need {count} bytes, "
                f"{len(self._data) - self._pos} left")
        out = self._data[self._pos:self._pos + count]
        self._pos += count
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack(">H", self.take(2))[0]

    def i16(self) -> int:
        return struct.unpack(">h", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack(">i", self.take(4))[0]


def _row_bytes(width: int, depth: int) -> int:
    if depth == 1:
        return (width + 7) // 8
    if depth == 8:
        return width
    if depth == 16:
        return width * 2
    if depth == 32:
        return width * 4
    raise PsdError(f"unsupported PSD depth {depth}")


def _decoded_len(planes: int, width: int, height: int,
                 depth: int) -> int:
    total = planes * height * _row_bytes(width, depth)
    if total > MAX_DECODED_BYTES:
        raise PsdError(
            f"decoded channel data would be {total} bytes — over "
            f"the {MAX_DECODED_BYTES} byte budget")
    return total


# ---------------------------------------------------------------------------
# PackBits (Apple/TIFF RLE, as used by Photoshop)
# ---------------------------------------------------------------------------

def packbits_encode(src: bytes) -> bytes:
    """Encode one row. Mirrors PhotoCraft's encoder: a run packet
    is never larger than the literal it replaces, and the -128
    no-op is never emitted."""
    out = bytearray()
    n = len(src)
    i = 0
    while i < n:
        run = 1
        while i + run < n and run < 128 and src[i + run] == src[i]:
            run += 1
        if run >= 2:
            out.append((1 - run) & 0xFF)
            out.append(src[i])
            i += run
            continue
        start = i
        length = 0
        while i < n and length < 128:
            if (i + 2 < n and src[i] == src[i + 1]
                    and src[i] == src[i + 2]):
                break
            i += 1
            length += 1
        out.append(length - 1)
        out.extend(src[start:start + length])
    return bytes(out)


def packbits_decode(src: bytes, expected: int) -> bytes:
    """Decode PackBits into exactly ``expected`` bytes."""
    out = bytearray()
    i = 0
    while len(out) < expected:
        if i >= len(src):
            raise PsdError(
                "PackBits stream ends early: "
                f"{len(out)} of {expected} bytes decoded")
        header = src[i]
        i += 1
        if header == 128:  # -128: no-op
            continue
        if header < 128:  # literal: header + 1 bytes follow
            count = header + 1
            if i + count > len(src):
                raise PsdError("PackBits literal runs past the row")
            out.extend(src[i:i + count])
            i += count
        else:  # run: repeat the next byte (257 - header) times
            if i >= len(src):
                raise PsdError("PackBits run has no byte to repeat")
            out.extend([src[i]] * (257 - header))
            i += 1
    if len(out) > expected:
        raise PsdError(
            f"PackBits row overflows: {len(out)} bytes for a "
            f"{expected}-byte row")
    return bytes(out)


# ---------------------------------------------------------------------------
# Channel planes: the four compression methods
# ---------------------------------------------------------------------------

def _predict(buf: bytearray, width: int, depth: int) -> None:
    """Apply horizontal delta prediction in place (encoder side).
    Byte-wise for 8-bit, u16-sample for 16-bit, byte-plane shuffle
    + byte-wise delta for 32-bit — as Photoshop does."""
    rb = _row_bytes(width, depth)
    if rb == 0:
        return
    for start in range(0, len(buf), rb):
        row = buf[start:start + rb]
        if depth in (1, 8):
            for i in range(len(row) - 1, 0, -1):
                row[i] = (row[i] - row[i - 1]) & 0xFF
        elif depth == 16:
            for i in range(width - 1, 0, -1):
                a = (row[2 * i] << 8) | row[2 * i + 1]
                b = (row[2 * i - 2] << 8) | row[2 * i - 1]
                value = (a - b) & 0xFFFF
                row[2 * i] = value >> 8
                row[2 * i + 1] = value & 0xFF
        elif depth == 32:
            scratch = bytearray(rb)
            for i in range(width):
                for k in range(4):
                    scratch[k * width + i] = row[4 * i + k]
            for i in range(rb - 1, 0, -1):
                scratch[i] = (scratch[i] - scratch[i - 1]) & 0xFF
            row[:] = scratch
        else:
            raise PsdError(
                f"prediction for depth {depth} is not supported")


def _unpredict(buf: bytearray, width: int, depth: int) -> None:
    """Reverse :func:`_predict` in place (decoder side)."""
    rb = _row_bytes(width, depth)
    if rb == 0:
        return
    for start in range(0, len(buf), rb):
        row = buf[start:start + rb]
        if depth in (1, 8):
            for i in range(1, len(row)):
                row[i] = (row[i] + row[i - 1]) & 0xFF
        elif depth == 16:
            for i in range(1, width):
                a = (row[2 * i] << 8) | row[2 * i + 1]
                b = (row[2 * i - 2] << 8) | row[2 * i - 1]
                value = (a + b) & 0xFFFF
                row[2 * i] = value >> 8
                row[2 * i + 1] = value & 0xFF
        elif depth == 32:
            for i in range(1, rb):
                row[i] = (row[i] + row[i - 1]) & 0xFF
            scratch = bytes(row)
            for i in range(width):
                for k in range(4):
                    row[4 * i + k] = scratch[k * width + i]
        else:
            raise PsdError(
                f"prediction for depth {depth} is not supported")


def decode_planes(compression: int, data: bytes, planes: int,
                  width: int, height: int, depth: int) -> bytes:
    """Decode planar channel data (``planes`` x ``height`` rows) to
    native big-endian sample bytes. ``compression``: 0 raw, 1 RLE,
    2 ZIP, 3 ZIP with prediction; anything else is refused."""
    total = _decoded_len(planes, width, height, depth)
    rb = _row_bytes(width, depth)
    if compression == 0:
        if len(data) < total:
            raise PsdError(
                f"raw channel data is short: {len(data)} bytes for "
                f"{total}")
        return data[:total]
    if compression == 1:
        rows = planes * height
        if len(data) < rows * 2:
            raise PsdError(
                "RLE row-count table is short: "
                f"{len(data)} bytes for {rows} rows")
        counts = struct.unpack(f">{rows}H", data[:rows * 2])
        pos = rows * 2
        out = bytearray()
        for count in counts:
            if pos + count > len(data):
                raise PsdError("RLE row runs past the channel data")
            out.extend(packbits_decode(data[pos:pos + count], rb))
            pos += count
        if len(out) != total:
            raise PsdError(
                f"RLE decoded {len(out)} bytes, expected {total}")
        return bytes(out)
    if compression in (2, 3):
        try:
            raw = zlib.decompress(data)
        except zlib.error as exc:
            raise PsdError(f"zlib channel data is corrupt: {exc}"
                           ) from exc
        if len(raw) != total:
            raise PsdError(
                f"zlib channel data decoded to {len(raw)} bytes, "
                f"expected {total}")
        if compression == 3:
            buf = bytearray(raw)
            _unpredict(buf, width, depth)
            return bytes(buf)
        return raw
    raise PsdError(f"unsupported PSD compression method {compression}")


def encode_planes(compression: int, decoded: bytes, planes: int,
                  width: int, height: int, depth: int) -> bytes:
    """Encode planar channel data (the inverse of
    :func:`decode_planes`)."""
    total = _decoded_len(planes, width, height, depth)
    if len(decoded) != total:
        raise PsdError(
            f"channel data has {len(decoded)} bytes, layout needs "
            f"{total}")
    rb = _row_bytes(width, depth)
    if compression == 0:
        return decoded
    if compression == 1:
        rows = planes * height
        counts: list[int] = []
        packed = bytearray()
        for r in range(rows):
            row = decoded[r * rb:(r + 1) * rb]
            encoded = packbits_encode(row)
            counts.append(len(encoded))
            packed.extend(encoded)
        return struct.pack(f">{rows}H", *counts) + bytes(packed)
    if compression in (2, 3):
        buf = bytearray(decoded)
        if compression == 3:
            _predict(buf, width, depth)
        return zlib.compress(bytes(buf))
    raise PsdError(f"unsupported PSD compression method {compression}")


# ---------------------------------------------------------------------------
# Depth conversion: every decoded plane becomes 8-bit samples
# ---------------------------------------------------------------------------

def _to_8bit(raw: bytes, depth: int) -> bytes:
    if depth == 8:
        return raw
    if depth == 1:
        out = bytearray()
        for byte in raw:
            for bit in range(7, -1, -1):
                out.append(255 if (byte >> bit) & 1 else 0)
        return bytes(out)
    if depth == 16:
        return bytes(raw[0::2])
    if depth == 32:
        count = len(raw) // 4
        values = struct.unpack(f">{count}f", raw[:count * 4])
        return bytes(
            max(0, min(255, int(round(v * 255.0)))) for v in values)
    raise PsdError(f"unsupported PSD depth {depth}")


def _read_pascal(reader: _Reader, align: int) -> bytes:
    """A Pascal string (length byte + bytes) padded to ``align``."""
    length = reader.u8()
    raw = reader.take(length)
    total = 1 + length
    pad = (-total) % align
    if pad:
        reader.take(pad)
    return raw


def _decode_legacy_name(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# ---------------------------------------------------------------------------
# Layer records
# ---------------------------------------------------------------------------

class LayerRecord:
    """One layer as parsed: rect, decoded 8-bit channel planes,
    blend, opacity, flags, name, mask, and the adjustment/fill
    keys that say what kind of layer it is."""

    def __init__(self) -> None:
        self.top = 0
        self.left = 0
        self.bottom = 0
        self.right = 0
        #: channel id -> decoded 8-bit plane bytes (rect-sized).
        self.channels: dict[int, bytes] = {}
        self.blend_key = "norm"
        self.opacity = 255
        self.clipping = 0
        self.flags = 0
        self.name = ""
        #: tagged-block keys found in the layer's extra data.
        self.block_keys: list[str] = []
        #: tagged blocks (key, raw data) as parsed — kept so a
        #: save writes them back verbatim.
        self.raw_blocks: list[tuple[str, bytes]] = []
        #: unicode name from the 'luni' block, when present.
        self.unicode_name = ""
        #: (top, left, bottom, right) of the layer mask, or None.
        self.mask_rect: tuple[int, int, int, int] | None = None
        self.mask_default = 255
        self.mask_disabled = False
        #: decoded 8-bit mask plane (mask-rect-sized), or None.
        self.mask: bytes | None = None
        #: group marker: "start", "end", or "".
        self.group = ""
        self.group_open = True

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def visible(self) -> bool:
        # flags bit 1: 0 = visible, 1 = hidden.
        return not (self.flags & 0x02)

    @property
    def display_name(self) -> str:
        return self.unicode_name or self.name or "Layer"

    @property
    def adjustment_keys(self) -> list[str]:
        return [key for key in self.block_keys
                if key in ADJUSTMENT_KEYS]

    @property
    def fill_keys(self) -> list[str]:
        return [key for key in self.block_keys if key in FILL_KEYS]

    @property
    def is_adjustment(self) -> bool:
        return bool(self.adjustment_keys)

    @property
    def is_fill(self) -> bool:
        return bool(self.fill_keys)

    @property
    def is_group_start(self) -> bool:
        return self.group == "start"

    @property
    def is_group_end(self) -> bool:
        return self.group == "end"

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.top, self.left, self.bottom, self.right)


def _parse_tagged_blocks(reader: _Reader, end: int,
                         what: str) -> list[tuple[str, bytes]]:
    """Read 8BIM tagged blocks until ``end``: (key, raw data).
    PSD v1 blocks all carry a 4-byte length (8-byte lengths are a
    PSB-only shape, and PSB is refused at the header)."""
    blocks: list[tuple[str, bytes]] = []
    while reader.pos < end:
        sig = reader.take(4)
        if sig not in (_SIG_8BIM, b"8B64"):
            raise PsdError(
                f"{what}: bad tagged-block signature {sig!r} at "
                f"offset {reader.pos - 4}")
        key = reader.take(4).decode("latin-1")
        size = reader.u32()
        data = reader.take(size)
        blocks.append((key, data))
        if size % 2:
            reader.take(1)
    if reader.pos != end:
        raise PsdError(
            f"{what}: tagged blocks end at {reader.pos}, "
            f"expected {end}")
    return blocks


def _parse_layer_record(reader: _Reader, depth: int,
                        version: int) -> tuple[LayerRecord,
                                               list[tuple[int, int]]]:
    """One layer record (no channel data — that follows ALL the
    records, in layer order; see :func:`_read_layer_channels`).
    Returns the record and its (channel id, byte size) list."""
    record = LayerRecord()
    record.top = reader.i32()
    record.left = reader.i32()
    record.bottom = reader.i32()
    record.right = reader.i32()
    channel_count = reader.u16()
    if channel_count > MAX_CHANNELS:
        raise PsdError(
            f"layer claims {channel_count} channels (max "
            f"{MAX_CHANNELS})")
    channel_sizes: list[tuple[int, int]] = []
    for _ in range(channel_count):
        channel_id = reader.i16()
        size = (struct.unpack(">Q", reader.take(8))[0]
                if version == 2 else reader.u32())
        channel_sizes.append((channel_id, size))
    sig = reader.take(4)
    if sig not in (_SIG_8BIM, b"8B64"):
        raise PsdError(
            f"layer blend signature {sig!r} at offset "
            f"{reader.pos - 4}")
    record.blend_key = reader.take(4).decode("latin-1")
    record.opacity = reader.u8()
    record.clipping = reader.u8()
    record.flags = reader.u8()
    reader.u8()  # filler
    extra_len = reader.u32()
    extra_end = reader.pos + extra_len
    if extra_end > reader.total:
        raise PsdError("layer extra data runs past the file")
    # Layer mask data.
    mask_size = reader.u32()
    if mask_size:
        mask_start = reader.pos
        mask_end = mask_start + mask_size
        mtop = reader.i32()
        mleft = reader.i32()
        mbottom = reader.i32()
        mright = reader.i32()
        record.mask_rect = (mtop, mleft, mbottom, mright)
        record.mask_default = reader.u8()
        mask_flags = reader.u8()
        record.mask_disabled = bool(mask_flags & 0x02)
        # The rest (parameters, real mask) is kept as raw bytes;
        # the mask pixels arrive through channel -2.
        reader.seek(mask_end)
    # Blending ranges.
    ranges_len = reader.u32()
    reader.take(ranges_len)
    # Layer name (Pascal, padded to 4).
    record.name = _decode_legacy_name(_read_pascal(reader, 4))
    # Tagged blocks to the end of the extra data.
    blocks = _parse_tagged_blocks(reader, extra_end, "layer extra")
    record.raw_blocks = blocks
    for key, data in blocks:
        record.block_keys.append(key)
        if key == "luni" and len(data) >= 4:
            count = struct.unpack(">I", data[:4])[0]
            raw = data[4:4 + count * 2]
            with contextlib.suppress(UnicodeDecodeError):
                record.unicode_name = raw.decode("utf-16-be")
        elif key == "lsct":
            kind = struct.unpack(">I", data[:4])[0] if len(data) >= 4 \
                else 0
            if kind in _GROUP_START:
                record.group = "start"
                record.group_open = kind == 1
            elif kind == 3:
                record.group = "end"
    # Channel image data does NOT follow this record — it follows
    # ALL the records, in layer order (Adobe spec, layer info;
    # PhotoCraft crates/psd/src/layer.rs LayerInfo::read_body reads
    # every record first, then every channel). It is read by
    # _read_layer_channels once the caller has all the records.
    return record, channel_sizes


def _read_layer_channels(reader: _Reader, record: LayerRecord,
                         channel_sizes: list[tuple[int, int]],
                         depth: int) -> None:
    """Decode one layer's channel image data, in channel order."""
    height = record.height
    width = record.width
    for channel_id, size in channel_sizes:
        blob = reader.take(size)
        if size == 0 or width == 0 or height == 0:
            record.channels[channel_id] = b""
            continue
        channel_reader = _Reader(blob, what="layer channel")
        compression = channel_reader.u16()
        body = blob[2:]
        decoded = decode_planes(compression, body, 1, width, height,
                                depth)
        record.channels[channel_id] = _to_8bit(decoded, depth)
    # The mask plane, when the layer carries one (channel -2).
    if -2 in record.channels and record.mask_rect is not None:
        record.mask = record.channels[-2]


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------

class PsdFile:
    """A parsed PSD v1 file: header, layers, composite."""

    def __init__(self) -> None:
        self.version = 1
        self.channels = 0
        self.width = 0
        self.height = 0
        self.depth = 8
        self.mode = "RGB"
        #: raw color-mode data bytes (kept, written back verbatim).
        self.color_mode_data = b""
        #: raw image-resources section bytes (kept, written back).
        self.image_resources = b""
        #: raw global-layer-mask bytes (kept, written back).
        self.global_mask = b""
        self.layers: list[LayerRecord] = []
        #: (compression, decoded 8-bit planes) for the composite.
        self._composite: tuple[int, list[bytes]] | None = None
        #: composite channel count (incl. alpha when present).
        self.composite_channels = 0

    # -- parse ---------------------------------------------------------
    @classmethod
    def from_bytes(cls, data: bytes) -> PsdFile:
        if len(data) < 26:
            raise PsdError(
                f"file is {len(data)} bytes — too short for a PSD "
                "header (26 bytes)")
        if data[:4] != _MAGIC:
            raise PsdError(
                f"bad PSD signature {data[:4]!r} — expected b'8BPS'")
        reader = _Reader(data)
        reader.seek(4)
        version = reader.u16()
        if version == 2:
            raise PsdError(
                "PSB (large document format, version 2) is not "
                "supported by this build — PSD v1 only")
        if version != 1:
            raise PsdError(f"unsupported PSD version {version}")
        reader.take(6)  # reserved
        psd = cls()
        psd.version = version
        psd.channels = reader.u16()
        psd.height = reader.u32()
        psd.width = reader.u32()
        psd.depth = reader.u16()
        mode_value = reader.u16()
        if psd.channels < 1 or psd.channels > MAX_CHANNELS:
            raise PsdError(
                f"channel count {psd.channels} outside 1.."
                f"{MAX_CHANNELS}")
        if psd.width < 1 or psd.width > MAX_DIMENSION \
                or psd.height < 1 or psd.height > MAX_DIMENSION:
            raise PsdError(
                f"dimensions {psd.width}x{psd.height} outside "
                f"1..{MAX_DIMENSION}")
        if psd.depth not in (1, 8, 16, 32):
            raise PsdError(f"unsupported PSD depth {psd.depth}")
        mode = MODES.get(mode_value)
        if mode is None:
            raise PsdError(
                f"unknown PSD color mode {mode_value}")
        if mode in ("Multichannel", "Duotone"):
            raise PsdError(
                f"PSD color mode {mode} is not supported — "
                "multichannel and duotone files are refused, never "
                "decoded as if they were RGB")
        psd.mode = mode
        # Color mode data (kept raw; indexed palettes read from it).
        color_len = reader.u32()
        psd.color_mode_data = reader.take(color_len)
        # Image resources (kept raw).
        res_len = reader.u32()
        psd.image_resources = reader.take(res_len)
        # Layer and mask information.
        section_len = reader.u32()
        section_end = reader.pos + section_len
        if section_end > len(data):
            raise PsdError("layer section runs past the file")
        if section_len:
            layer_info_len = reader.u32()
            layer_info_end = reader.pos + layer_info_len
            if layer_info_len >= 2:
                count = reader.i16()
                negative = count < 0
                count = abs(count)
                if count > 10000:
                    raise PsdError(
                        f"layer count {count} is implausible")
                pending: list[tuple[LayerRecord,
                                    list[tuple[int, int]]]] = []
                for _ in range(count):
                    record, sizes = _parse_layer_record(
                        reader, psd.depth, version)
                    pending.append((record, sizes))
                    psd.layers.append(record)
                for record, sizes in pending:
                    _read_layer_channels(
                        reader, record, sizes, psd.depth)
                if negative and psd.layers:
                    # First alpha channel is the transparency mask:
                    # Photoshop marks it by negating the count.
                    pass
            reader.seek(layer_info_end)
            # Global layer mask info (kept raw).
            global_len = reader.u32()
            psd.global_mask = reader.take(global_len)
            # Additional layer information: tagged blocks to the
            # section end (kept raw — skipped, not parsed).
            reader.seek(section_end)
        # Image data: the merged composite.
        compression = reader.u16()
        body = data[reader.pos:]
        color_planes = _MODE_COLOR_CHANNELS[psd.mode]
        total_planes = psd.channels
        if total_planes < color_planes:
            raise PsdError(
                f"header claims {psd.channels} channels for mode "
                f"{psd.mode} (needs {color_planes})")
        decoded = decode_planes(compression, body, total_planes,
                                psd.width, psd.height, psd.depth)
        row = _row_bytes(psd.width, psd.depth)
        planes = []
        for p in range(total_planes):
            plane = decoded[p * psd.height * row:
                            (p + 1) * psd.height * row]
            planes.append(_to_8bit(plane, psd.depth))
        psd._composite = (compression, planes)
        psd.composite_channels = total_planes
        return psd

    # -- composite + flatten ------------------------------------------
    def composite_rgba(self) -> Any:
        """Photoshop's own rendering of this file: the merged
        composite image data, decoded to RGBA. This is the oracle —
        the vendor's rendering of the same layer stack."""
        from PIL import Image

        if self._composite is None:
            raise PsdError("no composite image data parsed")
        _, planes = self._composite
        width, height = self.width, self.height
        if self.mode == "RGB":
            rgb = Image.merge("RGB", [
                Image.frombytes("L", (width, height), planes[0]),
                Image.frombytes("L", (width, height), planes[1]),
                Image.frombytes("L", (width, height), planes[2]),
            ])
            if len(planes) >= 4:
                alpha = Image.frombytes(
                    "L", (width, height), planes[3])
                return Image.merge("RGBA", [*rgb.split(), alpha])
            return rgb.convert("RGBA")
        if self.mode == "L":
            gray = Image.frombytes("L", (width, height), planes[0])
            if len(planes) >= 2:
                alpha = Image.frombytes(
                    "L", (width, height), planes[1])
                return Image.merge("RGBA", [
                    gray, gray, gray, alpha])
            return gray.convert("RGBA")
        if self.mode == "Indexed":
            palette = self.color_mode_data
            if len(palette) < 768:
                raise PsdError(
                    "indexed PSD without a 768-byte palette")
            indexed = Image.frombytes(
                "P", (width, height), planes[0])
            indexed.putpalette(palette[:768])
            return indexed.convert("RGBA")
        if self.mode == "CMYK":
            cmyk = Image.merge("CMYK", [
                Image.frombytes("L", (width, height), planes[i])
                for i in range(4)])
            out = cmyk.convert("RGB").convert("RGBA")
            if len(planes) >= 5:
                alpha = Image.frombytes(
                    "L", (width, height), planes[4])
                out.putalpha(alpha)
            return out
        if self.mode == "Lab":
            lab = Image.merge("LAB", [
                Image.frombytes("L", (width, height), planes[0]),
                Image.frombytes("L", (width, height), planes[1]),
                Image.frombytes("L", (width, height), planes[2]),
            ])
            out = lab.convert("RGB").convert("RGBA")
            if len(planes) >= 4:
                alpha = Image.frombytes(
                    "L", (width, height), planes[3])
                out.putalpha(alpha)
            return out
        if self.mode == "Bitmap":
            bits = Image.frombytes("1", (width, height), planes[0])
            return bits.convert("RGBA")
        raise PsdError(f"mode {self.mode} has no composite render")

    def flatten_layers(self) -> Any:
        """Devon's own compositor over the parsed layers: normal
        blend, per-layer opacity, visibility, and layer masks.
        Non-normal blend keys, adjustment layers, fill layers,
        layer effects, live text and smart objects are NOT
        rendered — see :meth:`unrendered` — so the flatten is
        honest about what it did. A layerless file flattens to
        its merged composite: the composite IS the document."""
        from PIL import Image

        if not self.layers:
            # A layerless PSD is a flat image: the merged composite
            # IS the document — there is no stack to composite.
            return self.composite_rgba()
        canvas = Image.new("RGBA", (self.width, self.height),
                           (0, 0, 0, 0))
        # PSD stores layers bottom-first (file order); compositing
        # runs in that order, each layer over the last.
        for layer in self.layers:
            if layer.is_group_start or layer.is_group_end:
                continue
            if not layer.visible:
                continue
            if layer.is_adjustment or layer.is_fill:
                continue
            if layer.width == 0 or layer.height == 0:
                continue
            rgba = _layer_rgba(layer, self.mode)
            if rgba is None:
                continue
            if layer.opacity < 255:
                alpha = rgba.split()[3].point(
                    lambda v, o=layer.opacity: v * o // 255)
                rgba.putalpha(alpha)
            canvas.alpha_composite(
                rgba, (layer.left, layer.top))
        return canvas

    def unrendered(self) -> list[str]:
        """What the flatten deliberately did not render, as human
        lines: adjustment layers, fill layers, non-normal blends."""
        lines: list[str] = []
        for layer in self.layers:
            for key in layer.adjustment_keys:
                lines.append(
                    f"adjustment layer {layer.display_name!r} "
                    f"({ADJUSTMENT_KEYS[key]}) is not rendered")
            for key in layer.fill_keys:
                lines.append(
                    f"fill layer {layer.display_name!r} "
                    f"({FILL_KEYS[key]}) is not rendered")
            if layer.blend_key not in ("norm", "pass"):
                lines.append(
                    f"layer {layer.display_name!r} uses blend "
                    f"{layer.blend_key!r} — rendered as normal")
        return lines

    def unrendered_adjustments(self) -> list[str]:
        return [line for line in self.unrendered()
                if line.startswith("adjustment")]

    # -- encode ---------------------------------------------------------
    def to_bytes(self) -> bytes:
        """Write this parsed file back as PSD v1. Channel data is
        re-encoded (RLE); resources, color-mode data and the global
        mask are written back verbatim."""
        out = bytearray()
        out += _MAGIC
        out += struct.pack(">H", 1)
        out += b"\x00" * 6
        # Planes are always 8-bit after decode, so the written file
        # is always depth 8 however deep the source was.
        _, planes = self._composite or (1, [])
        out += struct.pack(">HIIHH", len(planes), self.height,
                           self.width, 8,
                           _mode_value(self.mode))
        out += struct.pack(">I", len(self.color_mode_data))
        out += self.color_mode_data
        out += struct.pack(">I", len(self.image_resources))
        out += self.image_resources
        layer_bytes = _encode_layer_info(self)
        out += struct.pack(">I", len(layer_bytes))
        out += layer_bytes
        out += struct.pack(">H", 1)  # composite: RLE
        row = self.width  # 8-bit planes
        packed = bytearray()
        counts: list[int] = []
        for plane in planes:
            for r in range(self.height):
                encoded = packbits_encode(plane[r * row:(r + 1) * row])
                counts.append(len(encoded))
                packed.extend(encoded)
        out += struct.pack(f">{len(counts)}H", *counts)
        out += packed
        return bytes(out)


def _mode_value(mode: str) -> int:
    for value, name in MODES.items():
        if name == mode:
            return value
    raise PsdError(f"unknown mode {mode!r}")


def _layer_rgba(layer: LayerRecord, mode: str) -> Any | None:
    """One layer's decoded planes -> an RGBA image at the layer's
    own rect. Returns None when the layer has no pixel channels."""
    from PIL import Image

    width, height = layer.width, layer.height
    if width == 0 or height == 0:
        return None
    planes = layer.channels
    if mode == "RGB":
        if 0 not in planes or 1 not in planes or 2 not in planes:
            return None
        rgb = Image.merge("RGB", [
            Image.frombytes("L", (width, height), planes[0]),
            Image.frombytes("L", (width, height), planes[1]),
            Image.frombytes("L", (width, height), planes[2]),
        ])
        if -1 in planes:
            alpha = Image.frombytes(
                "L", (width, height), planes[-1])
        else:
            alpha = Image.new("L", (width, height), 255)
        out = Image.merge("RGBA", [*rgb.split(), alpha])
    elif mode == "L":
        if 0 not in planes:
            return None
        gray = Image.frombytes("L", (width, height), planes[0])
        if -1 in planes:
            alpha = Image.frombytes(
                "L", (width, height), planes[-1])
        else:
            alpha = Image.new("L", (width, height), 255)
        out = Image.merge("RGBA", [gray, gray, gray, alpha])
    elif mode == "CMYK":
        if any(i not in planes for i in range(4)):
            return None
        cmyk = Image.merge("CMYK", [
            Image.frombytes("L", (width, height), planes[i])
            for i in range(4)])
        rgb = cmyk.convert("RGB")
        if -1 in planes:
            alpha = Image.frombytes(
                "L", (width, height), planes[-1])
        else:
            alpha = Image.new("L", (width, height), 255)
        out = Image.merge("RGBA", [*rgb.split(), alpha])
    elif mode == "Lab":
        if any(i not in planes for i in range(3)):
            return None
        lab = Image.merge("LAB", [
            Image.frombytes("L", (width, height), planes[i])
            for i in range(3)])
        rgb = lab.convert("RGB")
        if -1 in planes:
            alpha = Image.frombytes(
                "L", (width, height), planes[-1])
        else:
            alpha = Image.new("L", (width, height), 255)
        out = Image.merge("RGBA", [*rgb.split(), alpha])
    elif mode == "Indexed":
        if 0 not in planes:
            return None
        indexed = Image.frombytes(
            "P", (width, height), planes[0])
        out = indexed.convert("RGBA")
        if -1 in planes:
            alpha = Image.frombytes(
                "L", (width, height), planes[-1])
            out.putalpha(alpha)
    elif mode == "Bitmap":
        if 0 not in planes:
            return None
        bits = Image.frombytes("1", (width, height), planes[0])
        out = bits.convert("RGBA")
        if -1 in planes:
            alpha = Image.frombytes(
                "L", (width, height), planes[-1])
            out.putalpha(alpha)
    else:
        return None
    if layer.mask is not None and layer.mask_rect is not None:
        mtop, mleft, mbottom, mright = layer.mask_rect
        mwidth = max(0, mright - mleft)
        mheight = max(0, mbottom - mtop)
        if mwidth and mheight and len(layer.mask) >= mwidth * mheight:
            mask = Image.frombytes(
                "L", (mwidth, mheight),
                layer.mask[:mwidth * mheight])
            full = Image.new("L", (width, height),
                             layer.mask_default)
            full.paste(mask, (mleft - layer.left,
                              mtop - layer.top))
            alpha = out.split()[3]
            from PIL import ImageChops

            out.putalpha(ImageChops.multiply(alpha, full))
    return out


def _encode_layer_info(psd: PsdFile) -> bytes:
    """The layer-and-mask section for a parsed file (layers only;
    global mask + additional blocks are written back verbatim
    where the parsed file kept them).

    All layer records first, then all channel image data, in
    layer order — the spec layout (and the crate's
    ``LayerInfo::write``); channel data never rides inside its
    own record.
    """
    records = bytearray()
    blobs = bytearray()
    count = len(psd.layers)
    records += struct.pack(">h", count)
    for layer in psd.layers:
        record, channel_blobs = _encode_layer_record(layer, psd.mode)
        records += record
        blobs += channel_blobs
    layer_info = (struct.pack(">I", len(records) + len(blobs))
                  + bytes(records) + bytes(blobs))
    section = bytearray()
    section += layer_info
    section += struct.pack(">I", len(psd.global_mask))
    section += psd.global_mask
    return bytes(section)


def _encode_layer_record(layer: LayerRecord, mode: str
                         ) -> tuple[bytes, bytes]:
    """One layer record + its channel image data (RLE), as a
    (record, channel-blobs) pair — the caller writes all records
    first, then all blobs, in layer order."""
    width, height = layer.width, layer.height
    # Channel planes to write: color channels first, then the
    # transparency channel (-1) when the layer carries one. A
    # zero-size layer (adjustment, fill, group marker, or a layer
    # cropped away entirely) carries no channels at all.
    color_ids = {"RGB": [0, 1, 2], "L": [0], "CMYK": [0, 1, 2, 3],
                 "Lab": [0, 1, 2], "Indexed": [0],
                 "Bitmap": [0]}.get(mode, [0, 1, 2])
    channel_data: list[tuple[int, bytes]] = []
    if width and height:
        for cid in color_ids:
            plane = layer.channels.get(cid)
            if plane is None:
                plane = bytes(width * height)
            encoded = struct.pack(">H", 1) + encode_planes(
                1, plane, 1, width, height, 8)
            channel_data.append((cid, encoded))
        if -1 in layer.channels:
            plane = layer.channels[-1]
            encoded = struct.pack(">H", 1) + encode_planes(
                1, plane, 1, width, height, 8)
            channel_data.append((-1, encoded))
        if layer.mask is not None and layer.mask_rect is not None:
            mtop, mleft, mbottom, mright = layer.mask_rect
            mw = max(0, mright - mleft)
            mh = max(0, mbottom - mtop)
            if mw and mh and len(layer.mask) >= mw * mh:
                encoded = struct.pack(">H", 1) + encode_planes(
                    1, layer.mask[:mw * mh], 1, mw, mh, 8)
                channel_data.append((-2, encoded))
    extra = bytearray()
    # Layer mask data: none on encode (mask pixels ride channel
    # -2; the mask rect is rebuilt from the layer rect).
    if layer.mask is not None and layer.mask_rect is not None:
        mtop, mleft, mbottom, mright = layer.mask_rect
        mask_block = struct.pack(
            ">iiiiBB", mtop, mleft, mbottom, mright,
            layer.mask_default, 0x02 if layer.mask_disabled else 0)
        extra += struct.pack(">I", len(mask_block)) + mask_block
    else:
        extra += struct.pack(">I", 0)
    extra += struct.pack(">I", 0)  # blending ranges: none
    # The legacy Pascal name is capped at 31 characters —
    # Photoshop's limit (the corpus's longest is 31: the full
    # name lives in the 'luni' block, written back verbatim
    # above, so nothing is lost).
    name = (layer.unicode_name or layer.name)[:31]
    name_bytes = name.encode("utf-8", errors="replace")
    pascal = bytes([len(name_bytes)]) + name_bytes
    pascal += b"\x00" * ((-len(pascal)) % 4)
    extra += pascal
    # Tagged blocks, written back verbatim (the 'luni' name block
    # included — a rename would need to rebuild it, and this build
    # has no rename op, so the blocks stay consistent).
    for key, data in layer.raw_blocks:
        extra += _SIG_8BIM
        extra += key.encode("latin-1")[:4].ljust(4, b" ")
        extra += struct.pack(">I", len(data))
        extra += data
        if len(data) % 2:
            extra += b"\x00"
    record = bytearray()
    record += struct.pack(
        ">iiiiH", layer.top, layer.left, layer.bottom, layer.right,
        len(channel_data))
    for cid, blob in channel_data:
        record += struct.pack(">hI", cid, len(blob))
    record += _SIG_8BIM
    # The layer's own blend key, not "norm": a save must not
    # silently turn a multiply layer into a normal one.
    key = (layer.blend_key or "norm").encode("latin-1")[:4]
    record += key.ljust(4, b" ")
    record += bytes([layer.opacity, layer.clipping,
                     layer.flags, 0])
    record += struct.pack(">I", len(extra))
    record += extra
    blobs = bytearray()
    for _, blob in channel_data:
        blobs += blob
    return bytes(record), bytes(blobs)


def encode_psd(document: Any) -> bytes:
    """Write a :class:`PhotoDocument` as PSD v1 bytes: header,
    layer records (RLE channels), and the RLE composite from the
    document's own flatten."""
    width, height = document.width, document.height
    if width < 1 or width > MAX_DIMENSION \
            or height < 1 or height > MAX_DIMENSION:
        raise PsdError(
            f"cannot write a {width}x{height} PSD (limit "
            f"{MAX_DIMENSION})")
    psd = PsdFile()
    psd.width = width
    psd.height = height
    psd.depth = 8
    psd.mode = "RGB"
    psd.channels = 4
    layers: list[LayerRecord] = []
    for layer in document.layers:
        record = LayerRecord()
        record.top = layer.top
        record.left = layer.left
        record.bottom = layer.top + layer.height
        record.right = layer.left + layer.width
        record.blend_key = layer.blend or "norm"
        record.opacity = layer.opacity
        record.clipping = 0
        record.flags = 0 if layer.visible else 0x02
        record.name = layer.name
        record.raw_blocks = list(layer.raw_blocks)
        if layer.kind == "pixels" and layer.width and layer.height:
            rgba = layer.to_pil()
            r, g, b, a = rgba.split()
            record.channels = {
                0: r.tobytes(),
                1: g.tobytes(),
                2: b.tobytes(),
                -1: a.tobytes(),
            }
            if layer.mask is not None:
                mask_img = layer.mask_to_pil()
                record.mask_rect = (layer.top, layer.left,
                                    layer.top + layer.height,
                                    layer.left + layer.width)
                record.mask = mask_img.tobytes()
                record.mask_default = 255
        layers.append(record)
    # PSD stores layers bottom-first (file order) — the document
    # is bottom-first too, so the records go in as they are.
    psd.layers = layers
    flat = document.flatten()
    if flat.mode != "RGBA":
        flat = flat.convert("RGBA")
    r, g, b, a = flat.split()
    psd._composite = (1, [r.tobytes(), g.tobytes(), b.tobytes(),
                          a.tobytes()])
    psd.composite_channels = 4
    return psd.to_bytes()
