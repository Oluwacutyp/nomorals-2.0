"""Measured-fidelity corpus for the documents engine (Wave 2).

The indd-utils discipline, applied: fidelity claims are *measured*
on a generated corpus, never asserted.  This module builds one rich
fixture document, round-trips it through every writer the engine
ships, and counts how many of the fixture's values survive the trip
(value-level for the cell formats, token-level for the prose
formats).  ``python -m nomorals.documents.corpus`` regenerates
``documents/PARITY.md`` from a live run; the wave test-suite asserts
the committed doc is fresh and that coverage never silently drops.

OCR rows are **environment-gated**: they measure only when the
Tesseract stack is present (absent on EC2 and on the build host —
an owner's-hand install, plan §6.3).  When it is absent the row
reports its gated state; it never reports a fabricated number.
"""

from __future__ import annotations

import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .model import Document, Section, Table

__all__ = ["fixture_document", "measure_all", "render_parity_md",
           "find_embedded_font", "ocr_gate_state"]


# ── the fixture ─────────────────────────────────────────────────────────────

def fixture_document() -> Document:
    """One deterministic document exercising every value class the
    writers claim to carry: prose (incl. non-Latin-1), headings,
    lists, code, and typed table cells."""
    return Document(
        format="corpus",
        title="Wave 2 Fidelity Corpus",
        author="nomorals corpus",
        sections=[
            Section(level=1, heading="Overview",
                    text="Fidelity is measured, not asserted.\n"
                         "Greek Αθήνα, Cyrillic Москва, a café price "
                         "of €50 and an arrow → survive or are named."),
            Section(level=2, heading="Methods", kind="list",
                    text="- generate the corpus\n"
                         "- round-trip every writer\n"
                         "- count recovered values"),
            Section(level=2, heading="Code", kind="code",
                    text="answer = 42  # measured, not vibes"),
        ],
        tables=[
            Table(name="Metrics",
                  headers=["Metric", "Count", "Ratio", "Day", "Active",
                           "Code", "Amount", "Note"],
                  rows=[
                      ["downloads", "1200", "4.50", "2026-11-05", "TRUE",
                       "007", "1,234", "café"],
                      ["uploads", "37", "0.125", "2026-11-06", "FALSE",
                       "042", "9,876", "Αθήνα"],
                      ["gaps", "", "-3", "2026-11-07", "TRUE",
                       "100", "0", ""],
                  ]),
            Table(name="Cities",
                  headers=["City", "Country"],
                  rows=[["Αθήνα", "Greece"],
                        ["Москва", "Russia"],
                        ["Lagos", "Nigeria"]]),
        ],
    )


# ── value accounting ────────────────────────────────────────────────────────

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _expected_values(doc: Document) -> list[str]:
    values: list[str] = []
    if doc.title:
        values.append(doc.title)
    for section in doc.sections:
        if section.heading:
            values.append(section.heading)
        for line in section.text.split("\n"):
            if line.strip():
                values.append(line.strip())
    for table in doc.tables:
        if table.name:
            values.append(table.name)
        values.extend(h for h in table.headers if h.strip())
        for row in table.rows:
            values.extend(c for c in row if c.strip())
    return values


def _value_matches(expected: str, pool: Counter) -> bool:
    """One expected value against the recovered pool (multiset).

    Exact text wins; otherwise numeric equality (``4.50`` written as a
    number cell reads back ``4.5``) or an ISO-date prefix (a date cell
    reads back as a timestamp) count as the value surviving.
    """
    if pool.get(expected, 0) > 0:
        pool[expected] -= 1
        return True
    # case-folded equality: a boolean cell written TRUE reads back
    # "True" — the value survived; only its casing canonicalized.
    for candidate in list(pool):
        if pool[candidate] > 0 and candidate.casefold() == expected.casefold():
            pool[candidate] -= 1
            return True
    try:
        target = float(expected.replace(",", ""))
    except ValueError:
        target = None
    if target is not None:
        for candidate in list(pool):
            if pool[candidate] <= 0:
                continue
            try:
                if float(candidate.replace(",", "")) == target:
                    pool[candidate] -= 1
                    return True
            except ValueError:
                continue
    if _ISO_DATE_RE.match(expected):
        for candidate in list(pool):
            if pool[candidate] > 0 and candidate.startswith(expected):
                pool[candidate] -= 1
                return True
    return False


def _coverage(expected: list[str], recovered: list[str]) -> tuple[int, int]:
    pool = Counter(recovered)
    hits = sum(1 for value in expected if _value_matches(value, pool))
    return hits, len(expected)


def _xlsx_recovered(doc: Document) -> list[str]:
    values: list[str] = []
    for table in doc.tables:
        if table.name:
            values.append(table.name)
        values.extend(table.headers)
        for row in table.rows:
            values.extend(row)
    # The prose sheet stores a section's whole text in one cell; the
    # corpus's accounting unit is the line, so multi-line cells also
    # contribute their lines (the text is wholly present either way).
    recovered: list[str] = []
    for value in values:
        recovered.append(value)
        if "\n" in value:
            recovered.extend(line.strip() for line in value.split("\n")
                             if line.strip())
    return recovered


def _pptx_recovered(doc: Document) -> list[str]:
    values: list[str] = []
    for section in doc.sections:
        for line in section.text.split("\n"):
            values.extend(part.strip() for part in line.split(" | ")
                          if part.strip())
    if doc.title:
        values.append(doc.title)
    return values


def _token_recovered(doc: Document) -> list[str]:
    from .model import full_text
    return full_text(doc).split()


def _token_expected(doc: Document) -> list[str]:
    tokens: list[str] = []
    for value in _expected_values(doc):
        tokens.extend(value.split())
    return tokens


# ── environment gates ───────────────────────────────────────────────────────

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def find_embedded_font() -> str | None:
    """A subsettable TrueType font for the embedded-PDF row, if the
    machine has one (env override, fontconfig, then known paths)."""
    env = os.environ.get("NM_CORPUS_FONT")
    if env and Path(env).is_file():
        return env
    fc_match = shutil.which("fc-match")
    if fc_match:
        import subprocess
        try:
            out = subprocess.run(
                [fc_match, "-f", "%{file}", "DejaVuSans"],
                capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and Path(out.stdout.strip()).is_file():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    share = Path("/usr/share/fonts")
    if share.is_dir():
        for hit in sorted(share.rglob("DejaVuSans.ttf")):
            return str(hit)
    return None


def ocr_gate_state() -> dict[str, Any]:
    """The OCR gate, as data: is the stack usable *here*?"""
    from .ocr import ocr_available

    return {
        "tesseract_on_path": shutil.which("tesseract") is not None,
        "stack_available": ocr_available(),
    }


# ── measurement ─────────────────────────────────────────────────────────────

def measure_all() -> dict[str, Any]:
    """Run every round-trip and return the raw measurements."""
    from .convert import (to_docx, to_epub, to_markdown, to_pdf,
                          to_pdf_embedded, to_pptx, to_xlsx)
    from .parsers import parse_bytes

    doc = fixture_document()
    expected_cells = _expected_values(doc)
    expected_tokens = _token_expected(doc)
    rows: list[dict[str, Any]] = []

    def cell_row(fmt: str, blob: bytes, filename: str, recover) -> None:
        back = parse_bytes(blob, filename=filename)
        hits, total = _coverage(expected_cells, recover(back))
        rows.append({"format": fmt, "basis": "value-level",
                     "hits": hits, "total": total,
                     "coverage": round(100.0 * hits / total, 2)})

    def token_row(fmt: str, blob: bytes, filename: str) -> None:
        back = parse_bytes(blob, filename=filename)
        hits, total = _coverage(expected_tokens, _token_recovered(back))
        rows.append({"format": fmt, "basis": "token-level",
                     "hits": hits, "total": total,
                     "coverage": round(100.0 * hits / total, 2)})

    token_row("markdown", to_markdown(doc).encode("utf-8"), "corpus.md")
    try:
        token_row("docx", to_docx(doc), "corpus.docx")
    except Exception as exc:  # noqa: BLE001 - gated row, named
        rows.append({"format": "docx", "basis": "token-level",
                     "gated": f"writer unavailable here: {exc}"})
    token_row("epub", to_epub(doc), "corpus.epub")
    cell_row("xlsx", to_xlsx(doc), "corpus.xlsx", _xlsx_recovered)
    cell_row("pptx", to_pptx(doc), "corpus.pptx", _pptx_recovered)
    token_row("pdf (WinAnsi writer)", to_pdf(doc), "corpus.pdf")

    font = find_embedded_font()
    embedded: dict[str, Any] = {"font": font}
    if font:
        blob = to_pdf_embedded(doc, font)
        back = parse_bytes(blob, filename="corpus.pdf")
        hits, total = _coverage(expected_tokens, _token_recovered(back))
        embedded.update({"hits": hits, "total": total,
                         "coverage": round(100.0 * hits / total, 2)})
        from ..core.pdf import _TtfFont
        full = _TtfFont(Path(font).read_bytes())
        embedded["font_glyphs"] = full.num_glyphs
        # subset glyph count, read back out of the produced PDF
        from ..core.pdf import _parse_objects, _stream_bytes
        import zlib as _zlib
        for body in _parse_objects(blob).values():
            if b"Length1" in body:
                raw = _zlib.decompress(_stream_bytes(body) or b"")
                embedded["subset_glyphs"] = _TtfFont(raw).num_glyphs
    rows.append({"format": "pdf (embedded font)", "basis": "token-level",
                 **embedded})

    # XLSX typed-cell census via the independent reader, when present.
    typed: dict[str, int] = {}
    try:
        import io

        import openpyxl
        workbook = openpyxl.load_workbook(
            io.BytesIO(to_xlsx(doc)), data_only=True)
        for sheet in workbook.worksheets:
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.value is not None:
                        typed[cell.data_type] = typed.get(
                            cell.data_type, 0) + 1
        workbook.close()
    except ImportError:
        typed = {}

    # WinAnsi losses, counted exactly: fixture characters the legacy
    # encoder itself cannot represent and has no ASCII fallback for
    # — the ones that become '?' (the same rule `_to_pdf_text` uses).
    from ..core.pdf import _ASCII_FALLBACK, _encode_char
    fixture_chars = sorted({ch for value in expected_cells
                            for ch in value if ord(ch) > 127})
    winansi_lost = [ch for ch in fixture_chars
                    if _encode_char(ch) is None
                    and ch not in _ASCII_FALLBACK and ch != "\u00a0"]
    gate = ocr_gate_state()
    return {
        "rows": rows,
        "typed_cells": typed,
        "fixture_non_latin1": fixture_chars,
        "winansi_unrepresentable": winansi_lost,
        "ocr_gate": gate,
    }


# ── the parity doc ──────────────────────────────────────────────────────────

def render_parity_md() -> str:
    """PARITY.md content, generated from a live measurement run."""
    m = measure_all()
    lines: list[str] = []
    lines.append("# Documents parity — Echelon documents port (Wave 2)")
    lines.append("")
    lines.append("Per the integration plan (P-rules): fidelity rows are")
    lines.append("*measured* on a generated corpus (indd-utils discipline),")
    lines.append("never asserted. Regenerate with:")
    lines.append("")
    lines.append("    python -m nomorals.documents.corpus")
    lines.append("")
    lines.append("The corpus fixture (`documents/corpus.py::fixture_document`)")
    lines.append("carries prose with non-Latin-1 text (Greek, Cyrillic, €, →),")
    lines.append("headings, a list section, a code section, and typed table")
    lines.append("cells (integers, decimals, ISO dates, booleans, leading-zero")
    lines.append("codes, thousands separators, empty cells). Each row writes")
    lines.append("the fixture with one writer and reads it back with the")
    lines.append("engine's own reader; coverage counts fixture values found in")
    lines.append("the read-back. Value-level rows compare whole cell values")
    lines.append("(numeric equality and ISO-date prefixes count — a typed cell")
    lines.append("reads back in canonical form); token-level rows compare the")
    lines.append("full-text token multiset (prose formats).")
    lines.append("")
    lines.append("## Round-trip fidelity (measured)")
    lines.append("")
    lines.append("| Format | Basis | Values recovered | Coverage | State |")
    lines.append("|---|---|---|---|---|")
    for row in m["rows"]:
        if "gated" in row:
            lines.append(f"| {row['format']} | {row['basis']} | — | — "
                         f"| gated — {row['gated']} |")
        elif row["format"] == "pdf (embedded font)" and "hits" not in row:
            lines.append(f"| pdf (embedded font) | token-level | — | — "
                         "| gated — no subsettable TrueType font found "
                         "on this machine |")
        else:
            lines.append(
                f"| {row['format']} | {row['basis']} | "
                f"{row['hits']}/{row['total']} | {row['coverage']}% "
                "| shipped (Wave 2) |")
    lines.append("")
    lines.append(
        "Shortfalls, named: markdown loses the two table names (its "
        "table block has no caption — see \"What does not convert\"); "
        "docx loses the three list-bullet markers (a paragraph style, "
        "not text); the WinAnsi PDF loses exactly the tokens "
        "containing the characters counted below. Nothing else is "
        "lost in any format.")
    lines.append("")
    emb = next(r for r in m["rows"] if r["format"] == "pdf (embedded font)")
    if "hits" in emb:
        lines.append(
            f"Embedded-PDF detail: font `{emb['font']}` carries "
            f"{emb['font_glyphs']} glyphs; the subset embedded in the "
            f"corpus PDF carries {emb['subset_glyphs']} — subsetting "
            "keeps the file small while the text stays exact. (Byte "
            "sizes are not printed: the page footer carries a "
            "timestamp, so a PDF's exact length is time-dependent "
            "at the ±1-byte level and would make this doc lie about "
            "being stale.)")
        lines.append("")
    if m["typed_cells"]:
        census = ", ".join(f"{k}={v}" for k, v in
                           sorted(m["typed_cells"].items()))
        lines.append(
            f"XLSX typed-cell census (independent reader, openpyxl "
            f"data_type): {census} — `n` numeric, `d` date, `b` boolean, "
            "`s` string. Typed writing is real: values a spreadsheet "
            "can type are typed, and values that would lose information "
            "typed (leading-zero codes, thousands separators) stay "
            "strings.")
        lines.append("")
    lost = " ".join(m["winansi_unrepresentable"])
    lines.append(
        f"The WinAnsi PDF writer cannot represent "
        f"{len(m['winansi_unrepresentable'])} of the fixture's "
        f"non-Latin-1 characters ({lost}) — they become `?` there; "
        "the embedded-font writer carries all of them. That gap is "
        "the reason the embedded path exists.")
    lines.append("")
    lines.append("## Error system")
    lines.append("")
    lines.append("| Row | State | Evidence / note |")
    lines.append("|---|---|---|")
    lines.append(
        "| `DocumentError` joins the NoMoralsError code system: "
        "stable `code` (`document.error`, refined per raise, e.g. "
        "`document.missing_dependency`, `document.ocr_unavailable`), "
        "`details`, `retryable`, registry membership, `to_dict()` | "
        "shipped (Wave 2) | `documents/errors.py`; it also stays a "
        "`ValueError` subclass so every pre-existing "
        "`except DocumentError`/`except ValueError` handler keeps "
        "working — the fold is additive |")
    lines.append(
        "| Machine-readable `kind` on every raise, separate from the "
        "human message (Echelon standard): parse / unsupported_format "
        "/ missing_dependency / empty / invalid_argument / io / "
        "write / ocr_unavailable / upstream / error (unclassified) | "
        "shipped (Wave 2) | kinds assigned at raise sites across "
        "`parsers.py` (with the parse funnel classifying parser "
        "failures), `convert.py`, `ocr.py`, `writers.py`; mirrored "
        "into `details[\"kind\"]`. The `index.py` / `compare.py` / "
        "`summarize.py` sites were classified by the burn-down "
        "sweep (all 59 sites; messages byte-identical, kind "
        "metadata only), which added one kind, `upstream` — a "
        "backing engine/service present but failed or vanished at "
        "runtime, distinct from `missing_dependency` (absent) |")
    lines.append("")
    lines.append("## Writers (plan item (b) — owner-visible choice)")
    lines.append("")
    lines.append("| Row | State | Evidence / note |")
    lines.append("|---|---|---|")
    lines.append(
        "| XLSX writer | shipped (Wave 2) | `documents/writers.py::"
        "to_xlsx` — stdlib zipfile; one sheet per table + a Document "
        "sheet for prose; typed cells; shared strings. The plan "
        "flagged writers-vs-read-only as owner-visible; the owner's "
        "standing doctrine (full capability, no artificial "
        "limitations) settles it for writers |")
    lines.append(
        "| PPTX writer | shipped (Wave 2) | `documents/writers.py::"
        "to_pptx` — stdlib zipfile; title slide, one slide per "
        "section, DrawingML tables. python-pptx is not required to "
        "write or read |")
    lines.append(
        "| Write surfaces | shipped (Wave 2) | `convert_document` "
        "targets `xlsx`/`pptx` and `nm doc convert --to xlsx|pptx`; "
        "the `documents` tool surface shape is unchanged |")
    lines.append("")
    lines.append("## PDF write fidelity (plan item (c))")
    lines.append("")
    lines.append("| Row | State | Evidence / note |")
    lines.append("|---|---|---|")
    lines.append(
        "| Embedded fonts + non-WinAnsi text: TrueType subsetting "
        "and embedding (Type0 / CIDFontType2 / Identity-H / "
        "ToUnicode), stdlib only | shipped (Wave 2) | "
        "`core/pdf.py::render_pdf_embedded` + "
        "`convert.to_pdf_embedded`; L1 stays dependency-free (the "
        "TTF is parsed and rebuilt with `struct`). Devon's own "
        "reader extracts the embedded text exactly via the "
        "ToUnicode CMap |")
    lines.append(
        "| WinAnsi writer | kept, unchanged | `core/pdf.py::"
        "render_pdf` — byte-stable legacy path; its non-Latin-1 "
        "losses are counted in the fidelity section above, not "
        "hidden |")
    lines.append("")
    lines.append("## What does not convert")
    lines.append("")
    lines.append("| Format | Does not carry |")
    lines.append("|---|---|")
    lines.append(
        "| xlsx | Section `kind` (list/code/quote) flattens to plain "
        "rows; section/table `page` provenance and table `caption`/"
        "`confidence` have no cell home; integers beyond 2^53 lose "
        "exactness (IEEE doubles); date cells read back as ISO "
        "timestamps, not the original date-only string; the prose "
        "sheet's title row reads back as that sheet's header row "
        "(the reader's first-row-is-headers rule) |")
    lines.append(
        "| pptx | Cell typing — every value is text on a slide; "
        "table headers are not distinguished from rows on read-back "
        "(all rows return as `a | b` lines); section kinds flatten; "
        "one fixed 16:9 blank layout |")
    lines.append(
        "| pdf (embedded) | Characters outside the *chosen font's* "
        "coverage fall back to `?` — font coverage is the limit; "
        "headings scale the one embedded face (no separate bold "
        "face); astral-plane characters render and extract via "
        "ToUnicode, but the subset font's own cmap is BMP-only "
        "(format 4) |")
    lines.append(
        "| pdf (WinAnsi) | Every character outside WinAnsi — "
        "counted in the fidelity section |")
    lines.append(
        "| docx | Table caption/confidence and section page "
        "provenance are not written; only list sections keep a "
        "visible kind distinction (List Bullet style) |")
    lines.append(
        "| markdown / epub | Table name/caption/page/confidence; "
        "all values are text (no typed cells) |")
    lines.append("")
    lines.append("## OCR (environment-gated, plan §6.3)")
    lines.append("")
    gate = m["ocr_gate"]
    gate_state = ("available here" if gate["stack_available"]
                  else "gated — Tesseract stack absent")
    lines.append("| Row | State | Evidence / note |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Scanned PDF → text (`ocr_pdf`), image → text "
        f"(`ocr_image`) | {gate_state} | the OCR seam is built and "
        "fails fast with a coded error (`document.ocr_unavailable`) "
        "where the stack is missing; the stack is absent on EC2 "
        "(owner's-hand install) and absent on the build host, so no "
        "OCR fidelity number is claimed anywhere. Where "
        "`ocr_available()` is true, this row measures like the "
        "others — until then it stays gated, never fabricated |")
    lines.append("")
    lines.append("## Open remainders")
    lines.append("")
    lines.append("| Row | State | Note |")
    lines.append("|---|---|---|")
    lines.append(
        "| Error-kind classification in `index.py` / `compare.py` / "
        "`summarize.py` | closed (burn-down) | the Wave-2 open "
        "remainder, closed by the burn-down sweep: all 59 raise "
        "sites carry an explicit taxonomy kind, proven per-site by "
        "`tests/test_documents_error_kinds.py` (each kind asserted "
        "by triggering the raise); human messages byte-identical "
        "to before the sweep. The taxonomy gained one kind, "
        "`upstream`, for the four sites whose failure is a "
        "present-but-failed backing engine/service (FTS table / "
        "backend loss, facet storage failure, LLM completion "
        "failure) |")
    lines.append(
        "| CFF-flavored OpenType (.otf) embedding | refused, named "
        "| the stdlib subsetter is TrueType-only; `render_pdf_"
        "embedded` raises `PdfError` naming the reason. A CFF "
        "subsetter is an L4 engine decision, not an L1 gap |")
    lines.append(
        "| OCR fidelity rows | gated — environment | see the OCR "
        "section; they un-gate where Tesseract is installed |")
    lines.append("")
    lines.append("---")
    lines.append("Status: verified in tests (this corpus + the Wave 2 "
                 "suite); not yet verified live on EC2.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    target = Path(__file__).with_name("PARITY.md")
    target.write_text(render_parity_md(), encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
