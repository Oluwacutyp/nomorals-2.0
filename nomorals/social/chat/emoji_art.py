"""The emoji-art organ: text-native art she makes herself
(expression completion, 2026-11-07; spec mined from
emojicombos.com — see ~/workspace/emoji-combos-mining/REPORT.md).

The medium is TEXT: emoji/symbol combos, dot art, ASCII art and
emoji mosaics live inside ordinary messages, so they work on
every platform she runs on — no backend, no premium, no upload
to a third-party site.  The load-bearing detail from the
mining: dot art is drawn in **Braille Patterns U+2800–U+28FF**
(a 2x4 dot matrix per character) and padded with **U+2800
braille blanks** — plain spaces collapse in chat renderers and
naive text art falls apart; braille blanks hold the grid.

What lives here (transport-free, like the rest of the
expression family):

* converters — image bytes -> braille dot art / ASCII ramp art /
  emoji mosaic, sized per surface so a piece never wraps;
* piece transforms — invert / mirror / rescale / restyle, the
  conversational remix verbs;
* the fancy-text styler — Unicode font variants + decorated
  forms for names, bios, captions, group titles;
* her OWN combo library — seeded with pieces drawn here, grown
  by her generations and the owner's saves, searchable by
  topic/tag.  (The mined site's corpus is user-submitted and
  its terms are silent on reuse — it is a spec, never a source.)
* describe -> art: a library piece when she has one, else her
  studio renders the described design and the converter draws
  it in dots — generation the reference site cannot do.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import random
import time
from pathlib import Path
from typing import Any

__all__ = [
    "BRAILLE_BLANK",
    "image_to_braille",
    "image_to_ascii",
    "image_to_mosaic",
    "matrix_to_braille",
    "piece_to_matrix",
    "invert_piece",
    "mirror_piece",
    "scale_piece",
    "restyle_piece",
    "FANCY_STYLES",
    "fancy_text",
    "ComboLibrary",
    "describe_art",
    "width_for_platform",
]

BRAILLE_BLANK = "⠀"

#: braille dot bit for (column, row) — the Unicode layout:
#: dots 1,2,3,7 down the left column, 4,5,6,8 down the right.
_DOT_BITS = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04,
             (0, 3): 0x40, (1, 0): 0x08, (1, 1): 0x10,
             (1, 2): 0x20, (1, 3): 0x80}

_ASCII_RAMP = " .:-=+*#%@"


def width_for_platform(platform: str) -> int:
    """Default art width (characters) per surface — sized so
    the piece never wraps on the surface it lands on."""
    platform = str(platform or "")
    if platform.startswith("whatsapp"):
        return 28
    if platform.startswith("telegram"):
        return 34
    return 32


def _load_gray(data: bytes):  # type: ignore[no-untyped-def]
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    return img.convert("L"), img.convert("RGB")


def _matrix_from_gray(gray, width_chars: int, *, mode: str,  # type: ignore[no-untyped-def]
                      invert: bool = False) -> list[list[bool]]:
    """Resample a grayscale image to a boolean dot matrix for
    braille: ``width_chars`` cells across (2 dots each), rows
    in whole cells (4 dots each), aspect preserved."""
    from PIL import Image

    w, h = gray.size
    dots_w = max(2, width_chars * 2)
    dots_h = max(4, int(round(dots_w * (h / max(1, w)))))
    dots_h -= dots_h % 4
    small = gray.resize((dots_w, dots_h), Image.LANCZOS)
    px = small.load()
    if mode == "dither":
        work = [[float(px[x, y]) for x in range(dots_w)]
                for y in range(dots_h)]
        matrix = [[False] * dots_w for _ in range(dots_h)]
        for y in range(dots_h):
            for x in range(dots_w):
                on = work[y][x] < 128.0
                matrix[y][x] = on
                err = work[y][x] - (0.0 if on else 255.0)
                for dx, dy, share in ((1, 0, 7 / 16), (-1, 1, 3 / 16),
                                      (0, 1, 5 / 16), (1, 1, 1 / 16)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < dots_w and 0 <= ny < dots_h:
                        work[ny][nx] += err * share
    else:
        total = sum(px[x, y] for y in range(dots_h)
                    for x in range(dots_w))
        threshold = total / max(1, dots_w * dots_h)
        matrix = [[px[x, y] < threshold for x in range(dots_w)]
                  for y in range(dots_h)]
    if invert:
        matrix = [[not v for v in row] for row in matrix]
    return matrix


def matrix_to_braille(matrix: list[list[bool]]) -> str:
    """Render a boolean dot matrix as braille text — every
    line full width, padded with U+2800 blanks (never spaces),
    so the grid holds its shape in any chat renderer."""
    if not matrix:
        return ""
    height = len(matrix)
    width = len(matrix[0])
    lines = []
    for cy in range(0, height - height % 4, 4):
        chars = []
        for cx in range(0, width - width % 2, 2):
            bits = 0
            for (dx, dy), bit in _DOT_BITS.items():
                if matrix[cy + dy][cx + dx]:
                    bits |= bit
            chars.append(chr(0x2800 + bits))
        lines.append("".join(chars))
    return "\n".join(lines)


def piece_to_matrix(text: str) -> list[list[bool]] | None:
    """Decode a braille piece back to its dot matrix (None
    when the text is not braille art)."""
    lines = [ln for ln in str(text or "").splitlines() if ln]
    if not lines:
        return None
    if any(not (0x2800 <= ord(ch) <= 0x28FF)
           for ln in lines for ch in ln):
        return None
    matrix: list[list[bool]] = []
    for ln in lines:
        rows: list[list[bool]] = [[], [], [], []]
        for ch in ln:
            bits = ord(ch) - 0x2800
            for dy in range(4):
                for dx in range(2):
                    rows[dy].append(
                        bool(bits & _DOT_BITS[(dx, dy)]))
        matrix.extend(rows)
    return matrix


def image_to_braille(data: bytes, *, width: int = 32,
                     mode: str = "mean",
                     invert: bool = False) -> str:
    """Photo bytes -> braille dot art."""
    gray, _rgb = _load_gray(data)
    return matrix_to_braille(
        _matrix_from_gray(gray, width, mode=mode, invert=invert))


def image_to_ascii(data: bytes, *, width: int = 40,
                   invert: bool = False) -> str:
    """Photo bytes -> character-ramp ASCII art (characters are
    ~twice as tall as wide, so rows sample at half rate)."""
    from PIL import Image

    gray, _rgb = _load_gray(data)
    w, h = gray.size
    rows = max(1, int(round(width * (h / max(1, w)) * 0.5)))
    small = gray.resize((width, rows), Image.LANCZOS)
    px = small.load()
    ramp = _ASCII_RAMP[::-1] if invert else _ASCII_RAMP
    lines = []
    for y in range(rows):
        lines.append("".join(
            ramp[min(len(ramp) - 1,
                     px[x, y] * len(ramp) // 256)]
            for x in range(width)))
    return "\n".join(lines)


def _mosaic_cell(r: int, g: int, b: int) -> str:
    luma = (r * 299 + g * 587 + b * 114) // 1000
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 28:  # near-gray: value only
        return "⬜" if luma > 150 else "⬛"
    if r >= g and r >= b:
        return "🟧" if g > 110 and b < 110 else "🟥"
    if g >= r and g >= b:
        return "🟨" if r > 120 else "🟩"
    if b >= r and b >= g:
        return "🟪" if r > 110 else "🟦"
    return "🟫"


def image_to_mosaic(data: bytes, *, width: int = 24) -> str:
    """Photo bytes -> an emoji mosaic (palette-matched emoji
    pixels — the conversion the reference site can't do)."""
    from PIL import Image

    _gray, rgb = _load_gray(data)
    w, h = rgb.size
    rows = max(1, int(round(width * (h / max(1, w)) * 0.55)))
    small = rgb.resize((width, rows), Image.LANCZOS)
    px = small.load()
    return "\n".join(
        "".join(_mosaic_cell(*px[x, y][:3]) for x in range(width))
        for y in range(rows))


# ── piece transforms (the remix verbs) ───────────────────────

def invert_piece(text: str) -> str | None:
    """Flip a piece: braille bits invert; ASCII walks the
    ramp backwards; a mosaic swaps its dark/light squares."""
    matrix = piece_to_matrix(text)
    if matrix is not None:
        return matrix_to_braille(
            [[not v for v in row] for row in matrix])
    lines = str(text or "").splitlines()
    if lines and all(ch in _ASCII_RAMP for ln in lines
                     for ch in ln):
        flip = {c: _ASCII_RAMP[len(_ASCII_RAMP) - 1 - i]
                for i, c in enumerate(_ASCII_RAMP)}
        return "\n".join("".join(flip.get(ch, ch) for ch in ln)
                         for ln in lines)
    if "⬛" in text or "⬜" in text:
        swap = {"⬛": "⬜", "⬜": "⬛", "🟥": "🟦", "🟦": "🟥",
                "🟩": "🟪", "🟪": "🟩", "🟨": "🟧", "🟧": "🟨"}
        return "".join(swap.get(ch, ch) for ch in text)
    return None


def mirror_piece(text: str) -> str | None:
    """Mirror a piece horizontally — for braille the dot
    matrix reverses (each cell's dot columns land correctly);
    text pieces reverse per line."""
    matrix = piece_to_matrix(text)
    if matrix is not None:
        return matrix_to_braille(
            [list(reversed(row)) for row in matrix])
    lines = str(text or "").splitlines()
    if not lines:
        return None
    return "\n".join(ln[::-1] for ln in lines)


def scale_piece(text: str, factor: float) -> str | None:
    """Resample a braille piece by ``factor`` (0.5 smaller,
    2.0 bigger) through its dot matrix."""
    matrix = piece_to_matrix(text)
    if matrix is None or not matrix:
        return None
    h, w = len(matrix), len(matrix[0])
    nh = max(4, int(round(h * factor)) // 4 * 4)
    nw = max(2, int(round(w * factor)) // 2 * 2)
    out = []
    for y in range(nh):
        row = []
        for x in range(nw):
            sx, sy = int(x / factor), int(y / factor)
            row.append(bool(matrix[min(h - 1, sy)][min(w - 1, sx)]))
        out.append(row)
    return matrix_to_braille(out)


def restyle_piece(text: str, target: str) -> str | None:
    """Re-render a braille piece as ASCII half-blocks or an
    emoji mosaic (dark on / light off); braille->braille is
    itself."""
    matrix = piece_to_matrix(text)
    if matrix is None:
        return None
    if target == "braille":
        return matrix_to_braille(matrix)
    if target == "mosaic":
        return "\n".join(
            "".join("⬛" if v else "⬜" for v in row)
            for row in matrix)
    if target == "ascii":
        lines = []
        for y in range(0, len(matrix) - 1, 2):
            chars = []
            for x in range(len(matrix[0])):
                top, bot = matrix[y][x], matrix[y + 1][x]
                chars.append("█" if top and bot else
                             "▀" if top else
                             "▄" if bot else " ")
            lines.append("".join(chars))
        return "\n".join(lines)
    return None


# ── fancy text ───────────────────────────────────────────────

def _variant(base_upper: int, base_lower: int, base_digit: int,
             exceptions: dict[str, str] | None = None
             ) -> dict[str, str]:
    table = dict(exceptions or {})
    for i in range(26):
        table.setdefault(chr(ord("A") + i), chr(base_upper + i))
        table.setdefault(chr(ord("a") + i), chr(base_lower + i))
    if base_digit:
        for i in range(10):
            table.setdefault(str(i), chr(base_digit + i))
    return table


_FANCY_TABLES: dict[str, dict[str, str]] = {
    "bold": _variant(0x1D400, 0x1D41A, 0x1D7CE),
    "italic": _variant(0x1D434, 0x1D44E, 0, {"h": "ℎ"}),
    "bolditalic": _variant(0x1D468, 0x1D482, 0),
    "script": _variant(0x1D49C, 0x1D4B6, 0, {
        "B": "ℬ", "E": "ℰ", "F": "ℱ", "H": "ℋ", "I": "ℐ",
        "L": "ℒ", "M": "ℳ", "R": "ℛ", "e": "ℯ", "g": "ℊ",
        "o": "ℴ"}),
    "fraktur": _variant(0x1D504, 0x1D51E, 0, {
        "C": "ℭ", "H": "ℌ", "I": "ℑ", "R": "ℜ", "Z": "ℨ"}),
    "doublestruck": _variant(0x1D538, 0x1D552, 0x1D7D8, {
        "C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ",
        "R": "ℝ", "Z": "ℤ"}),
    "monospace": _variant(0x1D670, 0x1D68A, 0x1D7F6),
    "sans": _variant(0x1D5A0, 0x1D5BA, 0x1D7E2),
    "sansbold": _variant(0x1D5D4, 0x1D5EE, 0x1D7EC),
    "circled": _variant(0x24B6, 0x24D0, 0, {"0": "⓪"}),
    "fullwidth": {chr(c): chr(c + 0xFEE0)
                  for c in range(0x21, 0x7F)},
    "smallcaps": dict(zip(
        "abcdefghijklmnopqrstuvwxyz",
        "ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘǫʀsᴛᴜᴠᴡxʏᴢ", strict=True)),
}
for _i in range(1, 10):
    _FANCY_TABLES["circled"][str(_i)] = chr(0x2460 + _i - 1)

_DECORATIONS: dict[str, tuple[str, str]] = {
    "sparkle": ("✨ ", " ✨"),
    "stars": ("⋆★ ", " ★⋆"),
    "hearts": ("♡ ", " ♡"),
    "frame": ("【 ", " 】"),
    "quotes": ("『 ", " 』"),
    "waves": ("˗ˏˋ ", " ˎˊ˗"),
}

FANCY_STYLES: tuple[str, ...] = tuple(_FANCY_TABLES) \
    + tuple(_DECORATIONS)


def fancy_text(text: str, style: str) -> str | None:
    """Restyle text: a Unicode font variant (bold, script,
    circled, smallcaps, ...) or a decoration (sparkle, frame,
    hearts, ...).  Characters a variant doesn't carry pass
    through unchanged — the text is never eaten."""
    style = str(style or "").lower().strip()
    text = str(text or "")
    if style in _FANCY_TABLES:
        table = _FANCY_TABLES[style]
        return "".join(table.get(ch, ch) for ch in text)
    if style in _DECORATIONS:
        pre, post = _DECORATIONS[style]
        return f"{pre}{text}{post}"
    return None


# ── her combo library ────────────────────────────────────────

def _grid(rows: tuple[str, ...]) -> list[list[bool]]:
    return [[ch == "#" for ch in row] for row in rows]


#: Pieces drawn HERE, as dot grids — her seed shelf.  Drawn,
#: not copied: the mined site's corpus is user-submitted with
#: silent terms, so her library starts with her own hand.
_SEED_GRIDS: dict[str, tuple[str, ...]] = {
    "heart": ("........",
              ".##..##.",
              "########",
              "########",
              ".######.",
              "..####..",
              "...##...",
              "........"),
    "star": ("....#...",
             "....#...",
             ".######.",
             "..####..",
             ".######.",
             "..#..#..",
             ".#....#.",
             "........"),
    "cat": (".#....#.",
            ".##..##.",
            "########",
            "#.#..#.#",
            "########",
            "##.##.##",
            ".######.",
            "........"),
    "moon": ("...####.",
             ".##.....",
             "##......",
             "##......",
             "##......",
             ".##.....",
             "...####.",
             "........"),
    "rocket": ("...##...",
               "..####..",
               "..####..",
               ".######.",
               "########",
               "#.####.#",
               "...##...",
               "..#..#.."),
    "flower": (".#....#.",
               "..#..#..",
               "...##...",
               "########",
               "...##...",
               "..#..#..",
               ".#....#.",
               "........"),
    "crown": ("........",
              "#..##..#",
              "#..##..#",
              "########",
              "########",
              ".######.",
              "........",
              "........"),
    "coffee": ("........",
               ".######.",
               ".######.",
               ".#####.#",
               ".######.",
               "..####..",
               "########",
               "........"),
}

_SEED_COMBOS: tuple[dict[str, Any], ...] = (
    {"name": "cat kaomoji", "tags": ["cat", "cute", "kaomoji"],
     "kind": "kaomoji", "text": "ᓚ₍ ^. .^₎"},
    {"name": "bear hug", "tags": ["bear", "hug", "kaomoji"],
     "kind": "kaomoji", "text": "ʕっ•ᴥ•ʔっ"},
    {"name": "shrug", "tags": ["shrug", "whatever", "kaomoji"],
     "kind": "kaomoji", "text": "¯\\_(ツ)_/¯"},
    {"name": "table flip", "tags": ["rage", "funny", "kaomoji"],
     "kind": "kaomoji", "text": "(╯°□°)╯︵ ┻━┻"},
    {"name": "sparkle heart", "tags": ["heart", "cute", "combo"],
     "kind": "combo", "text": "˗ˏˋ ♡ ˎˊ˗"},
    {"name": "paw trail", "tags": ["cat", "paw", "combo"],
     "kind": "combo", "text": "~° * ⋆˚🐾˖°"},
    {"name": "moon dust", "tags": ["moon", "night", "combo"],
     "kind": "combo", "text": "☾⋆⁺₊✧"},
    {"name": "soft bow", "tags": ["cute", "bow", "combo"],
     "kind": "combo", "text": "꒰ᐢ. .ᐢ꒱₊˚⊹"},
    {"name": "star wand", "tags": ["star", "magic", "combo"],
     "kind": "combo", "text": "𖤐 ‧₊˚ ⋅"},
)


def _piece_id(name: str) -> str:
    return "ea_" + hashlib.sha256(
        f"{name}-{time.time()}-{random.random()}".encode()
    ).hexdigest()[:10]


class ComboLibrary:
    """Her combo library: one JSON document (atomic saves,
    corrupt files quarantined — the expression library's own
    discipline).  Pieces carry name/tags/kind/text/provenance;
    the seed shelf is written on first load, marked as hers."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {"pieces": []}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(
                    self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) \
                        and isinstance(raw.get("pieces"), list):
                    self.data = raw
                    return
                raise ValueError("library root is not an object")
            except (OSError, ValueError):
                with contextlib.suppress(OSError):
                    self.path.replace(self.path.with_suffix(
                        f".corrupt-{int(time.time())}"))
        self.data = {"pieces": [self._seed_piece(p)
                                for p in self._seeds()]}
        self.save()

    @staticmethod
    def _seeds() -> list[dict[str, Any]]:
        pieces = [dict(p) for p in _SEED_COMBOS]
        for name, rows in _SEED_GRIDS.items():
            pieces.append({
                "name": f"{name} dot art",
                "tags": [name, "dotart", "drawn"],
                "kind": "dotart",
                "text": matrix_to_braille(_grid(rows))})
        return pieces

    @staticmethod
    def _seed_piece(piece: dict[str, Any]) -> dict[str, Any]:
        return {"id": _piece_id(str(piece["name"])),
                "name": str(piece["name"]),
                "tags": [str(t).lower() for t in piece["tags"]],
                "kind": str(piece["kind"]),
                "text": str(piece["text"]),
                "source": "seed:self", "created": time.time()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self.data.get("pieces") or [])

    def pieces(self) -> list[dict[str, Any]]:
        return list(self.data.get("pieces") or [])

    def search(self, query: str) -> list[dict[str, Any]]:
        # Tokens under 2 chars never search: "a" is a substring
        # of nearly every name/tag and would match the whole
        # shelf (it once swallowed every described subject).
        words = [w for w in str(query or "").lower().split()
                 if len(w) >= 2]
        if not words:
            return self.pieces()
        out = []
        for piece in self.data.get("pieces") or []:
            hay = (str(piece.get("name") or "").lower() + " "
                   + " ".join(piece.get("tags") or []))
            if any(w in hay for w in words):
                out.append(piece)
        return out

    def add(self, *, name: str, text: str, kind: str = "combo",
            tags: Any = (), source: str = "owner"
            ) -> dict[str, Any]:
        piece = {"id": _piece_id(name), "name": str(name),
                 "tags": [str(t).lower() for t in tags or ()
                          if str(t).strip()],
                 "kind": str(kind or "combo"),
                 "text": str(text), "source": str(source),
                 "created": time.time()}
        self.data.setdefault("pieces", []).append(piece)
        self.save()
        return piece


def describe_art(subject: str, library: ComboLibrary | None,
                 *, width: int = 30) -> tuple[str, str] | None:
    """Subject words -> (art, how): her library's own piece
    first; else the studio draws the described design and the
    braille converter renders it — generation, in dots."""
    subject = " ".join(str(subject or "").split()).strip()
    if not subject:
        return None
    if library is not None:
        for piece in library.search(subject):
            if piece.get("kind") in ("dotart", "mosaic"):
                return str(piece["text"]), "her library"
    from . import emoji_studio

    spec = emoji_studio.parse_design(subject)
    img = emoji_studio.render_master(spec, size=400)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    art = image_to_braille(buf.getvalue(), width=width,
                           mode="dither")
    return art, f"drawn for you — {spec.get('shape') or 'face'}"
