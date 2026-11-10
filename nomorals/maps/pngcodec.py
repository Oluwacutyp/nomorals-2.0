"""A small, honest PNG codec for map tiles — stdlib only.

The repo's PNG idiom is zlib+struct (``media_edit/cv_video.py``'s
writer); Atlas needs a *reader* as well, because M2's pipeline will
emit tiles with external tooling whose encoders use every filter
type.  So this codec decodes real-world PNGs — color types 0/2/3/4/6
at 8-bit depth, all five scanline filters — and encodes plain
filter-0 RGB.  Bounds are stated, not hidden: 16-bit sources and
Adam7 interlacing are refused with a named error rather than
silently mis-decoded.  Alpha is dropped on decode (tiles composite
opaquely); grayscale and palette expand to RGB.
"""

from __future__ import annotations

import struct
import zlib

__all__ = ["PngError", "decode_png", "encode_rgb_png"]

_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


class PngError(ValueError):
    """A PNG defect, named in the message."""


def _chunk(kind: bytes, data: bytes) -> bytes:
    head = struct.pack(">I", len(data)) + kind + data
    return head + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_rgb_png(width: int, height: int, rgb: bytes) -> bytes:
    """Encode an RGB buffer (``width*height*3`` bytes) as a PNG."""
    if width <= 0 or height <= 0:
        raise PngError("PNG dimensions must be positive")
    if len(rgb) != width * height * 3:
        raise PngError("RGB buffer does not match the dimensions")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(
        b"\x00" + rgb[y * width * 3:(y + 1) * width * 3]
        for y in range(height))
    return (_SIGNATURE + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(rows, 6))
            + _chunk(b"IEND", b""))


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(raw: bytes, width: int, height: int,
              bpp: int) -> bytes:
    """Reconstruct scanlines (filter types 0–4) into packed pixels."""
    stride = width * bpp
    out = bytearray(height * stride)
    prev = bytearray(stride)
    pos = 0
    for y in range(height):
        if pos >= len(raw):
            raise PngError("PNG pixel data truncated")
        kind = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        if len(line) != stride:
            raise PngError("PNG pixel data truncated")
        pos += stride
        if kind == 0:
            pass
        elif kind == 1:  # Sub
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif kind == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif kind == 3:  # Average
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + (left + prev[i]) // 2) & 0xFF
        elif kind == 4:  # Paeth
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                up_left = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(left, prev[i], up_left)) & 0xFF
        else:
            raise PngError("unknown PNG filter type: " + str(kind))
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return bytes(out)


def decode_png(data: bytes) -> tuple[int, int, bytes]:
    """Decode a PNG to ``(width, height, rgb_bytes)``.

    Supports 8-bit color types 0 (gray), 2 (RGB), 3 (palette),
    4 (gray+alpha) and 6 (RGBA), non-interlaced.  Anything else is
    a named :class:`PngError`, never a guess.
    """
    if not data.startswith(_SIGNATURE):
        raise PngError("not a PNG (bad signature)")
    pos = len(_SIGNATURE)
    ihdr: tuple | None = None
    palette: bytes = b""
    idat = bytearray()
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if len(body) != length:
            raise PngError("PNG chunk truncated")
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"PLTE":
            palette = body
        elif kind == b"IDAT":
            idat.extend(body)
        elif kind == b"IEND":
            break
        pos += 8 + length + 4
    if ihdr is None:
        raise PngError("PNG has no IHDR")
    width, height, depth, color_type, _comp, _filt, interlace = ihdr
    if depth != 8:
        raise PngError("only 8-bit PNGs are supported, got depth "
                       + str(depth))
    if interlace != 0:
        raise PngError("interlaced (Adam7) PNGs are not supported")
    if color_type not in _CHANNELS:
        raise PngError("unsupported PNG color type: " + str(color_type))
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise PngError("PNG pixel stream is corrupt: " + str(exc)) from exc
    channels = _CHANNELS[color_type]
    pixels = _unfilter(raw, width, height, channels)
    rgb = bytearray(width * height * 3)
    if color_type == 2:
        rgb[:] = pixels
    elif color_type == 6:
        for i in range(width * height):
            rgb[i * 3:i * 3 + 3] = pixels[i * 4:i * 4 + 3]
    elif color_type == 0:
        for i in range(width * height):
            rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = pixels[i]
    elif color_type == 4:
        for i in range(width * height):
            gray = pixels[i * 2]
            rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = gray
    else:  # palette
        if not palette:
            raise PngError("palette PNG has no PLTE chunk")
        for i in range(width * height):
            entry = pixels[i] * 3
            if entry + 2 >= len(palette):
                raise PngError("palette index out of range")
            rgb[i * 3:i * 3 + 3] = palette[entry:entry + 3]
    return width, height, bytes(rgb)
