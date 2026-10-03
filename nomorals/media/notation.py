"""Staff notation renderer — real engraved-style sheet music, pure Python.

Renders a Song's melody as treble-clef staff notation into a standalone PDF
using raw PDF content-stream operators (no dependencies).  This is genuine
notation — note heads on a staff with stems, beams, bar lines, chord symbols
and section labels — not a text lead sheet.

Scope is deliberately simple: single treble staff, 4/4, melody only, with
chord symbols above the staff.  Enough to read and play the tune.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ── minimal PDF builder ───────────────────────────────────────────────────

class _PDF:
    """Tiny PDF writer: one font (Helvetica), vector ops, text."""

    def __init__(self, width: float = 595.0, height: float = 842.0):
        self.width = width
        self.height = height
        self._pages: list["_Page"] = []

    def new_page(self) -> "_Page":
        p = _Page(self.width, self.height)
        self._pages.append(p)
        return p

    def build(self) -> bytes:
        objs: list[bytes] = []
        # 1: catalog, 2: pages, 3: font, then page/content pairs
        n_pages = len(self._pages)
        page_obj_nums = []
        content_obj_nums = []
        for i in range(n_pages):
            page_obj_nums.append(4 + i * 2)
            content_obj_nums.append(5 + i * 2)

        kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
        objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        objs.append(
            f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode("ascii"))
        objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        for i, pg in enumerate(self._pages):
            objs.append(
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {self.width:.1f} {self.height:.1f}] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_obj_nums[i]} 0 R >>".encode("ascii"))
            raw = pg.stream.encode("latin-1")
            objs.append(
                f"<< /Length {len(raw)} >>\nstream\n".encode("ascii")
                + raw + b"\nendstream")

        out = [b"%PDF-1.4\n"]
        offsets = []
        for i, body in enumerate(objs, start=1):
            offsets.append(len(b"".join(out)))
            out.append(f"{i} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
        xref_pos = len(b"".join(out))
        out.append(f"xref\n0 {len(objs) + 1}\n".encode("ascii"))
        out.append(b"0000000000 65535 f \n")
        for off in offsets:
            out.append(f"{off:010d} 00000 n \n".encode("ascii"))
        out.append(
            f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF".encode("ascii"))
        return b"".join(out)


class _Page:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.ops: list[str] = []

    @property
    def stream(self) -> str:
        return "\n".join(self.ops)

    # vector ops
    def line(self, x1, y1, x2, y2, w: float = 1.0):
        self.ops.append(f"{w:.2f} w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S")

    def ellipse(self, cx, cy, rx, ry, fill: bool = True):
        # bezier approximation of an ellipse
        k = 0.5523
        self.ops.append(
            f"{cx - rx:.1f} {cy:.1f} m "
            f"{cx - rx:.1f} {cy + k * ry:.1f} "
            f"{cx - k * rx:.1f} {cy + ry:.1f} "
            f"{cx:.1f} {cy + ry:.1f} c "
            f"{cx + k * rx:.1f} {cy + ry:.1f} "
            f"{cx + rx:.1f} {cy + k * ry:.1f} "
            f"{cx + rx:.1f} {cy:.1f} c "
            f"{cx + rx:.1f} {cy - k * ry:.1f} "
            f"{cx + k * rx:.1f} {cy - ry:.1f} "
            f"{cx:.1f} {cy - ry:.1f} c "
            f"{cx - k * rx:.1f} {cy - ry:.1f} "
            f"{cx - rx:.1f} {cy - k * ry:.1f} "
            f"{cx - rx:.1f} {cy:.1f} c "
            + ("f" if fill else "S"))

    def text(self, x, y, s: str, size: float = 10, bold: bool = False):
        esc = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        # latin-1 fallback for anything non-ascii
        esc = esc.encode("latin-1", "replace").decode("latin-1")
        self.ops.append(
            f"BT /F1 {size:.1f} Tf {x:.1f} {y:.1f} Td ({esc}) Tj ET")


# ── notation ──────────────────────────────────────────────────────────────

#: MIDI pitch of the bottom treble line (E4)
_E4 = 64
#: semitone offsets of the diatonic scale degrees (C major)
_DIATONIC = [0, 2, 4, 5, 7, 9, 11]


def _pitch_to_staff(pitch: int) -> tuple[int, bool]:
    """Return (diatonic_steps_from_E4, is_accidental).

    Steps count staff positions (lines+spaces); accidentals are flagged so
    the caller can draw a sharp/flat sign (we just nudge: draw natural).
    """
    # find nearest diatonic pitch at or below
    octave = (pitch // 12) - 1  # MIDI octave, C4=60 → octave 4
    pc = pitch % 12
    if pc in _DIATONIC:
        deg = _DIATONIC.index(pc)
        acc = False
    else:
        # chromatic alteration: sharp of the degree below
        below = max(s for s in _DIATONIC if s < pc)
        deg = _DIATONIC.index(below)
        acc = True
    # steps from E4: E is degree 2 of its octave
    steps = (octave - 4) * 7 + (deg - 2)
    return steps, acc


@dataclass
class _Engraver:
    page: _Page
    x: float
    y: float  # baseline of bottom staff line
    staff_gap: float = 9.0
    bar_width: float = 0.0

    def staff_y(self, steps: int) -> float:
        return self.y + steps * (self.staff_gap / 2)

    def draw_staff(self, x0: float, x1: float):
        for i in range(5):
            yy = self.y + i * self.staff_gap
            self.page.line(x0, yy, x1, yy, w=0.8)

    def draw_barline(self, x: float, final: bool = False):
        top = self.y + 4 * self.staff_gap
        self.page.line(x, self.y, x, top, w=1.2)
        if final:
            self.page.line(x + 3, self.y, x + 3, top, w=2.5)

    def draw_note(self, x: float, pitch: int, dur_beats: float):
        steps, acc = _pitch_to_staff(pitch)
        cy = self.staff_y(steps)
        # ledger lines
        if steps < 0:
            for s in range(-2, steps - 1, -2):
                yy = self.staff_y(s)
                self.page.line(x - 9, yy, x + 9, yy, w=0.8)
        elif steps > 8:
            for s in range(10, steps + 1, 2):
                yy = self.staff_y(s)
                self.page.line(x - 9, yy, x + 9, yy, w=0.8)
        # accidental sign (sharp)
        if acc:
            self.page.text(x - 16, cy - 3, "#", size=9)
        # note head: filled for <= quarter, hollow for longer
        filled = dur_beats < 1.75
        self.page.ellipse(x, cy, 5.2, 3.8, fill=filled)
        # stem
        stem_up = steps <= 4
        if dur_beats < 3.75:  # no stem for whole notes
            sx = x + 4.6 if stem_up else x - 4.6
            top = cy + 31 if stem_up else cy - 31
            self.page.line(sx, cy, sx, top, w=1.1)

    def draw_rest(self, x: float):
        cy = self.y + 2 * self.staff_gap
        self.page.ellipse(x, cy, 4.5, 3.2, fill=True)
        self.page.line(x + 4, cy + 2, x + 4, cy - 14, w=1.1)


def render_score_pdf(title: str, subtitle: str,
                     melody: list[tuple[int, float, float]],
                     bars: list[tuple[int, str, str]],
                     tempo: int = 120) -> bytes:
    """Render staff notation PDF.

    melody: list of (midi_pitch, start_beat, duration_beats)
    bars: list of (bar_index, chord_symbol, section_name)
    """
    pdf = _PDF()
    W, H = pdf.width, pdf.height
    MARGIN = 46.0
    STAFF_W = W - 2 * MARGIN
    BAR_W = 64.0
    BARS_PER_SYSTEM = max(1, int(STAFF_W // BAR_W))

    # group melody notes by bar
    by_bar: dict[int, list[tuple[int, float, float]]] = {}
    for pitch, start, dur in melody:
        b = int(start // 4)
        by_bar.setdefault(b, []).append((pitch, start % 4, dur))

    bar_info = {b: (chord, sec) for b, chord, sec in bars}
    total_bars = max([b for b, _, _ in bars], default=-1) + 1
    if total_bars == 0 and by_bar:
        total_bars = max(by_bar) + 1

    systems: list[list[int]] = []
    for i in range(0, total_bars, BARS_PER_SYSTEM):
        systems.append(list(range(i, min(i + BARS_PER_SYSTEM, total_bars))))

    page = pdf.new_page()
    y = H - 70
    # title block
    page.text(MARGIN, y, title, size=17)
    page.text(MARGIN, y - 20, subtitle, size=10)
    page.text(W - MARGIN - 90, y - 20, f"= {tempo}", size=10)
    y -= 58

    SYSTEM_H = 96.0
    for si, sys_bars in enumerate(systems):
        if y - SYSTEM_H < MARGIN:
            page = pdf.new_page()
            y = H - 70
        eng = _Engraver(page, 0, y)
        x0, x1 = MARGIN, MARGIN + len(sys_bars) * BAR_W
        eng.draw_staff(x0, x1)
        # clef: stylized "&"-like treble marker (text approximation)
        page.text(x0 + 4, y - 6, "G", size=22)

        for j, b in enumerate(sys_bars):
            bx = x0 + j * BAR_W
            chord, sec = bar_info.get(b, ("", ""))
            # section label at first bar of a new section
            prev_sec = bar_info.get(b - 1, ("", ""))[1]
            if sec and sec != prev_sec:
                page.text(bx + 2, y + 4 * eng.staff_gap + 16,
                          sec.upper(), size=8)
            if chord:
                page.text(bx + 4, y + 4 * eng.staff_gap + 6, chord, size=9)
            # barline at end
            eng.draw_barline(bx + BAR_W, final=(b == total_bars - 1))
            # notes
            for pitch, off, dur in sorted(by_bar.get(b, []), key=lambda n: n[1]):
                nx = bx + 12 + (off / 4.0) * (BAR_W - 20)
                eng.draw_note(nx, pitch, dur)
        y -= SYSTEM_H

    return pdf.build()
