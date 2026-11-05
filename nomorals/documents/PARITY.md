# Documents parity — Echelon documents port (Wave 2)

Per the integration plan (P-rules): fidelity rows are
*measured* on a generated corpus (indd-utils discipline),
never asserted. Regenerate with:

    python -m nomorals.documents.corpus

The corpus fixture (`documents/corpus.py::fixture_document`)
carries prose with non-Latin-1 text (Greek, Cyrillic, €, →),
headings, a list section, a code section, and typed table
cells (integers, decimals, ISO dates, booleans, leading-zero
codes, thousands separators, empty cells). Each row writes
the fixture with one writer and reads it back with the
engine's own reader; coverage counts fixture values found in
the read-back. Value-level rows compare whole cell values
(numeric equality and ISO-date prefixes count — a typed cell
reads back in canonical form); token-level rows compare the
full-text token multiset (prose formats).

## Round-trip fidelity (measured)

| Format | Basis | Values recovered | Coverage | State |
|---|---|---|---|---|
| markdown | token-level | 86/88 | 97.73% | shipped (Wave 2) |
| docx | token-level | 85/88 | 96.59% | shipped (Wave 2) |
| epub | token-level | 88/88 | 100.0% | shipped (Wave 2) |
| xlsx | value-level | 50/50 | 100.0% | shipped (Wave 2) |
| pptx | value-level | 50/50 | 100.0% | shipped (Wave 2) |
| pdf (WinAnsi writer) | token-level | 81/88 | 92.05% | shipped (Wave 2) |
| pdf (embedded font) | token-level | 88/88 | 100.0% | shipped (Wave 2) |

Shortfalls, named: markdown loses the two table names (its table block has no caption — see "What does not convert"); docx loses the three list-bullet markers (a paragraph style, not text); the WinAnsi PDF loses exactly the tokens containing the characters counted below. Nothing else is lost in any format.

Embedded-PDF detail: font `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` carries 6253 glyphs; the subset embedded in the corpus PDF carries 74 — subsetting keeps the file small while the text stays exact. (Byte sizes are not printed: the page footer carries a timestamp, so a PDF's exact length is time-dependent at the ±1-byte level and would make this doc lie about being stale.)

XLSX typed-cell census (independent reader, openpyxl data_type): b=3, d=3, n=7, s=34 — `n` numeric, `d` date, `b` boolean, `s` string. Typed writing is real: values a spreadsheet can type are typed, and values that would lose information typed (leading-zero codes, thousands separators) stay strings.

The WinAnsi PDF writer cannot represent 11 of the fixture's non-Latin-1 characters (Α ή α θ ν М а в к о с) — they become `?` there; the embedded-font writer carries all of them. That gap is the reason the embedded path exists.

## Error system

| Row | State | Evidence / note |
|---|---|---|
| `DocumentError` joins the NoMoralsError code system: stable `code` (`document.error`, refined per raise, e.g. `document.missing_dependency`, `document.ocr_unavailable`), `details`, `retryable`, registry membership, `to_dict()` | shipped (Wave 2) | `documents/errors.py`; it also stays a `ValueError` subclass so every pre-existing `except DocumentError`/`except ValueError` handler keeps working — the fold is additive |
| Machine-readable `kind` on every raise, separate from the human message (Echelon standard): parse / unsupported_format / missing_dependency / empty / invalid_argument / io / write / ocr_unavailable / error (unclassified) | shipped (Wave 2) | kinds assigned at raise sites across `parsers.py` (with the parse funnel classifying parser failures), `convert.py`, `ocr.py`, `writers.py`; mirrored into `details["kind"]`. `index.py` / `compare.py` / `summarize.py` raise sites still carry the unclassified kind — see open remainders |

## Writers (plan item (b) — owner-visible choice)

| Row | State | Evidence / note |
|---|---|---|
| XLSX writer | shipped (Wave 2) | `documents/writers.py::to_xlsx` — stdlib zipfile; one sheet per table + a Document sheet for prose; typed cells; shared strings. The plan flagged writers-vs-read-only as owner-visible; the owner's standing doctrine (full capability, no artificial limitations) settles it for writers |
| PPTX writer | shipped (Wave 2) | `documents/writers.py::to_pptx` — stdlib zipfile; title slide, one slide per section, DrawingML tables. python-pptx is not required to write or read |
| Write surfaces | shipped (Wave 2) | `convert_document` targets `xlsx`/`pptx` and `nm doc convert --to xlsx|pptx`; the `documents` tool surface shape is unchanged |

## PDF write fidelity (plan item (c))

| Row | State | Evidence / note |
|---|---|---|
| Embedded fonts + non-WinAnsi text: TrueType subsetting and embedding (Type0 / CIDFontType2 / Identity-H / ToUnicode), stdlib only | shipped (Wave 2) | `core/pdf.py::render_pdf_embedded` + `convert.to_pdf_embedded`; L1 stays dependency-free (the TTF is parsed and rebuilt with `struct`). Devon's own reader extracts the embedded text exactly via the ToUnicode CMap |
| WinAnsi writer | kept, unchanged | `core/pdf.py::render_pdf` — byte-stable legacy path; its non-Latin-1 losses are counted in the fidelity section above, not hidden |

## What does not convert

| Format | Does not carry |
|---|---|
| xlsx | Section `kind` (list/code/quote) flattens to plain rows; section/table `page` provenance and table `caption`/`confidence` have no cell home; integers beyond 2^53 lose exactness (IEEE doubles); date cells read back as ISO timestamps, not the original date-only string; the prose sheet's title row reads back as that sheet's header row (the reader's first-row-is-headers rule) |
| pptx | Cell typing — every value is text on a slide; table headers are not distinguished from rows on read-back (all rows return as `a | b` lines); section kinds flatten; one fixed 16:9 blank layout |
| pdf (embedded) | Characters outside the *chosen font's* coverage fall back to `?` — font coverage is the limit; headings scale the one embedded face (no separate bold face); astral-plane characters render and extract via ToUnicode, but the subset font's own cmap is BMP-only (format 4) |
| pdf (WinAnsi) | Every character outside WinAnsi — counted in the fidelity section |
| docx | Table caption/confidence and section page provenance are not written; only list sections keep a visible kind distinction (List Bullet style) |
| markdown / epub | Table name/caption/page/confidence; all values are text (no typed cells) |

## OCR (environment-gated, plan §6.3)

| Row | State | Evidence / note |
|---|---|---|
| Scanned PDF → text (`ocr_pdf`), image → text (`ocr_image`) | gated — Tesseract stack absent | the OCR seam is built and fails fast with a coded error (`document.ocr_unavailable`) where the stack is missing; the stack is absent on EC2 (owner's-hand install) and absent on the build host, so no OCR fidelity number is claimed anywhere. Where `ocr_available()` is true, this row measures like the others — until then it stays gated, never fabricated |

## Open remainders

| Row | State | Note |
|---|---|---|
| Error-kind classification in `index.py` / `compare.py` / `summarize.py` | open remainder | their raise sites carry the unclassified kind (`error`) today; the code system membership already applies to them. A classification sweep of those three organs is ranked below the remaining waves |
| CFF-flavored OpenType (.otf) embedding | refused, named | the stdlib subsetter is TrueType-only; `render_pdf_embedded` raises `PdfError` naming the reason. A CFF subsetter is an L4 engine decision, not an L1 gap |
| OCR fidelity rows | gated — environment | see the OCR section; they un-gate where Tesseract is installed |

---
Status: verified in tests (this corpus + the Wave 2 suite); not yet verified live on EC2.
