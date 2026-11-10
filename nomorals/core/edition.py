"""Branded HTML editions — ONE renderer for every file edition
(Doc 16, Pass 3).

Every long document Devon ships as a file — an authored book, a
followed story, a parsed document, a presentation Doc, a report —
renders through this one module, over the Pass-1 manifest contract
(:mod:`nomorals.core.longdoc`) and the Pass-2 content-resolver seam
(a callable mapping a manifest segment to its readable text).
Producers keep their public shapes (``books.forge.html_book``,
``documents.convert.to_html``, ``tools.filesend.md_to_html``) and
delegate here, so the three historical HTML producers can no longer
drift apart.

Home: core (L1), beside :mod:`nomorals.core.pdf` — the only layer
every producer can reach downward (documents/social/tools are L4,
books is L5), stdlib only.

The brand (the typography law, extended from chat text to file
editions): an edition carries the Peacethefirst mark structurally —
a brand line in the cover header, the closing footer, and a subtle
print-only mark in the page's bottom safe area.  It never lands
inside content: code blocks and prose stay byte-faithful, and the
screen rendering carries no watermark over the text at all.
``brand_policy`` follows the manifest's
:class:`~nomorals.core.longdoc.PresentationHints` and mirrors
``social.chat.typography.resolve_style`` semantics exactly —
``default`` stamps the house mark, ``off`` stamps nothing (no brand
string anywhere in the file), ``custom`` stamps the caller's custom
text and falls back to the house mark when none is supplied, the
same way a group ``stamp_text`` of "" resolves to the default.
The constants are mirrored here because core cannot import social;
a test pins them equal so the two homes cannot drift.

Self-containment: an edition is one HTML file — embedded CSS, one
small trusted inline script (theme / text-size / measure controls
and a reading-progress rail; every control degrades to a static,
fully readable page with scripts off or storage blocked), system
font stacks only.  No external asset is ever referenced.

Honesty rules, enforced by construction and pinned by tests:

* content is escaped into the templates — untrusted source markup
  renders as visible text, never as live markup;
* absent segments are declared absent, in the contents list and in
  the body — never invented, never silently dropped;
* the front matter states the declared completeness ("3 of 12
  chapters") and, when the source has a revision counter, the
  revision number.  Hash-form revision identities never print —
  a hash is an internal id, not an owner-facing label.
"""

from __future__ import annotations

import html as _html
import re
from collections.abc import Callable, Mapping
from typing import Any

from .longdoc import LongDocumentManifest, ManifestSegment
from .text import slugify

__all__ = [
    "BRAND_NAME", "BRAND_TEXT", "THEMES", "TREATMENTS",
    "Resolver", "completeness_line", "render_edition",
    "render_edition_from_map", "treatment_for",
]

#: The brand, mirrored from social.chat.typography (whose
#: DEFAULT_STAMP_TEXT is the canonical chat form) — core cannot
#: import social, and a test pins the two equal.
BRAND_NAME = "Peacethefirst"
BRAND_TEXT = "Powered by peacethefirst"

#: A content resolver: manifest segment → its readable text, or
#: None when the segment's content is not held (the Pass-2 seam —
#: books/documents/social.chat manifest modules ship one per
#: producer).
Resolver = Callable[[ManifestSegment], "str | None"]

#: Edition colour themes.  light/dark/sepia are the reader themes
#: of the handoff spec; print and minimal are the documents
#: engine's historical variants, kept as first-class themes so its
#: callers lose nothing in the consolidation.
THEMES = ("light", "dark", "sepia", "print", "minimal")

#: Editorial treatments, chosen from the manifest's source kind
#: (never guessed from prose): book (serif, drop caps, chapter
#: openers), report (editorial sans, accent rule), technical
#: (compact sans, restrained data styling).
TREATMENTS = ("report", "book", "technical")

_SOURCE_TREATMENTS = {
    "forge-book": "book",
    "story": "book",
    "document": "technical",
    "present-doc": "report",
    "text": "report",
}

_SOURCE_LABELS = {
    "story": "Followed story",
    "forge-book": "Authored book",
    "document": "Document",
    "present-doc": "Report",
    "text": "Text",
}

#: Source kinds whose segment content is markdown-authored (forge
#: chapters are manuscript markdown).  Everything else resolves to
#: plain text and renders as faithful prose.
_MARKDOWN_SOURCES = frozenset({"forge-book"})


def treatment_for(manifest: LongDocumentManifest) -> str:
    """The treatment a manifest's source kind maps to."""
    return _SOURCE_TREATMENTS.get(manifest.source_kind, "report")


# ── brand / theme / completeness ────────────────────────────────────


def _brand_for(manifest: LongDocumentManifest, brand_policy: str,
               brand_text: str) -> str:
    """Resolve the brand line for one edition ("" = no brand).

    Mirrors typography.resolve_style: an unknown policy is data,
    not an error, and reads as the default; ``custom`` with no
    custom text falls back to the house mark, exactly as a group
    ``stamp_text`` of "" does.
    """
    policy = (brand_policy or manifest.presentation.brand_policy
              or "default").strip().lower()
    if policy == "off":
        return ""
    if policy == "custom":
        custom = (brand_text or "").strip()
        if custom:
            return custom
    return BRAND_TEXT


def _theme_for(manifest: LongDocumentManifest, theme: str) -> str:
    name = (theme or manifest.presentation.theme or "light")
    name = name.strip().lower()
    return name if name in THEMES else "light"


def _unit_noun(manifest: LongDocumentManifest) -> str:
    kinds = {seg.kind for seg in manifest.segments}
    if "chapter" in kinds:
        return "chapters"
    if "section" in kinds:
        return "sections"
    if kinds == {"table"}:
        return "tables"
    return "parts"


def _missing_label(manifest: LongDocumentManifest) -> str:
    """The declared-but-absent ranges in owner words, e.g.
    "chapters 4, 6–10" ("" when nothing is declared missing)."""
    ranges = manifest.completeness.missing_ranges
    if not ranges:
        return ""
    parts = []
    for start, end in ranges:
        parts.append(str(start) if start == end else f"{start}–{end}")
    return f"{_unit_noun(manifest)} " + ", ".join(parts)


def completeness_line(manifest: LongDocumentManifest) -> str:
    """The edition's declared completeness, in owner words — the
    line that stops a partial edition posing as the whole work."""
    comp = manifest.completeness
    unit = _unit_noun(manifest)
    present = comp.present_count
    total = comp.declared_total
    if comp.status == "partial":
        line = f"Partial edition — {present} of {total} {unit} present."
        missing = _missing_label(manifest)
        if missing:
            line += f" Missing: {missing}."
        return line
    if comp.status == "open":
        if total:
            return (f"Ongoing edition — {present} of {total} declared "
                    f"{unit} so far; the work is still growing.")
        return (f"Ongoing edition — {present} {unit} so far; "
                f"the work is still growing.")
    if total:
        return f"Complete edition — {present} of {total} {unit}."
    return f"Complete edition — {present} {unit}."


def _esc(text: Any) -> str:
    return _html.escape(str(text if text is not None else ""),
                        quote=True)


# ── inline + block rendering ────────────────────────────────────────
#
# Two content modes.  "markdown" is the honest subset the shipped
# producers already speak (headings, fences, lists, quotes, pipe
# tables, bold/italic/code/links) — filesend.md_to_html's grammar,
# extended with tables and heading anchors.  "plain" is faithful
# prose: escaped paragraphs, nothing reinterpreted.  Both escape
# FIRST; markup is only ever built from the escaped text, so source
# HTML is inert by construction.

_SAFE_HREF = re.compile(r"^(https?://|mailto:|#)[^\s\"'<>]+$")
_CODE_SPAN = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_UL_ITEM = re.compile(r"^[-*+]\s+(.*)$")
_OL_ITEM = re.compile(r"^\d+[.)]\s+(.*)$")
_TABLE_SEP = re.compile(r"^\|?[\s:|-]+\|?$")
_TAG = re.compile(r"<[^>]+>")


def _inline(text: str) -> str:
    """Markdown inline forms over ESCAPED text (filesend's order:
    code, bold, italic, links).  A link whose target is not a safe
    web/mail/anchor reference renders as its label text only."""
    out = _html.escape(text, quote=False)
    out = _CODE_SPAN.sub(r"<code>\1</code>", out)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _ITALIC.sub(r"<em>\1</em>", out)

    def _link(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        if not _SAFE_HREF.match(url):
            return label
        return ('<a href="' + _html.escape(url, quote=True) + '">'
                + label + "</a>")

    return _LINK.sub(_link, out)


def _heading_slug(raw: str, seen: dict[str, int]) -> str:
    """Anchor id for an in-content markdown heading: the heading's
    plain text (tags and markdown marks stripped) through the same
    slug rule the report organ uses (core slugify, limit 60,
    "section" fallback), deduped with -2, -3 … suffixes."""
    plain = _TAG.sub("", raw)
    plain = _LINK.sub(lambda m: m.group(1), plain)
    plain = plain.replace("**", "").replace("`", "").replace("*", "")
    base = slugify(plain, limit=60, fallback="section") or "section"
    count = seen.get(base, 0) + 1
    seen[base] = count
    return base if count == 1 else f"{base}-{count}"


def _split_cells(line: str) -> list[str]:
    cells = line.strip()
    if cells.startswith("|"):
        cells = cells[1:]
    if cells.endswith("|"):
        cells = cells[:-1]
    return [cell.strip() for cell in cells.split("|")]


def _table_html(rows: list[list[str]], *, header: bool,
                inline: bool) -> str:
    fmt = _inline if inline else _esc
    parts = ["<table>"]
    body = rows
    if header and rows:
        parts.append("<thead><tr>" + "".join(
            f"<th>{fmt(cell)}</th>" for cell in rows[0]) + "</tr></thead>")
        body = rows[1:]
    parts.append("<tbody>")
    for row in body:
        parts.append("<tr>" + "".join(
            f"<td>{fmt(cell)}</td>" for cell in row) + "</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _render_markdown(text: str,
                     slugs: dict[str, int]) -> str:
    lines = (text or "").split("\n")
    out: list[str] = []
    para: list[str] = []
    fence: list[str] | None = None
    list_tag: str | None = None
    quote: list[str] = []

    def flush_para() -> None:
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    def flush_quote() -> None:
        if quote:
            out.append("<blockquote>" + "".join(
                f"<p>{_inline(q)}</p>" for q in quote) + "</blockquote>")
            quote.clear()

    def flush_all() -> None:
        flush_para()
        flush_list()
        flush_quote()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if fence is not None:
            if stripped.startswith("```"):
                out.append("<pre><code>" + _esc("\n".join(fence))
                           + "</code></pre>")
                fence = None
            else:
                fence.append(line)
            i += 1
            continue
        if stripped.startswith("```"):
            flush_all()
            fence = []
            i += 1
            continue
        if not stripped:
            flush_all()
            i += 1
            continue
        heading = _HEADING.match(stripped)
        if heading:
            flush_all()
            level = len(heading.group(1))
            raw = heading.group(2).strip()
            slug = _heading_slug(raw, slugs)
            out.append(f'<h{level} id="{slug}">{_inline(raw)}</h{level}>')
            i += 1
            continue
        if stripped in ("---", "***", "___"):
            flush_all()
            out.append("<hr>")
            i += 1
            continue
        if "|" in stripped and i + 1 < len(lines) \
                and _TABLE_SEP.match(lines[i + 1].strip()) \
                and "-" in lines[i + 1]:
            flush_all()
            rows = [_split_cells(stripped)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_cells(lines[i]))
                i += 1
            out.append(_table_html(rows, header=True, inline=True))
            continue
        if stripped.startswith(">"):
            flush_para()
            flush_list()
            quote.append(stripped.lstrip(">").strip())
            i += 1
            continue
        item = _UL_ITEM.match(stripped)
        tag = "ul"
        if item is None:
            item = _OL_ITEM.match(stripped)
            tag = "ol"
        if item is not None:
            flush_para()
            flush_quote()
            if list_tag != tag:
                flush_list()
                out.append(f"<{tag}>")
                list_tag = tag
            out.append(f"<li>{_inline(item.group(1))}</li>")
            i += 1
            continue
        flush_list()
        flush_quote()
        para.append(stripped)
        i += 1
    if fence is not None:
        out.append("<pre><code>" + _esc("\n".join(fence))
                   + "</code></pre>")
    flush_all()
    return "\n".join(out)


def _paragraphs(text: str) -> list[str]:
    """Faithful plain-prose blocks: blank-line paragraphs, single
    newlines kept as line breaks, everything escaped."""
    blocks: list[str] = []
    for chunk in re.split(r"\n\s*\n", (text or "").strip()):
        chunk = chunk.strip()
        if chunk:
            blocks.append("<p>" + _esc(chunk).replace("\n", "<br>\n")
                          + "</p>")
    return blocks


_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _render_plain(segment: ManifestSegment, text: str) -> str:
    """Plain-mode segment body.  A documents section's declared
    kind (carried in the segment metadata by its derivation) keeps
    its shipped rendering: code as a code block, a quote as a
    quote, a list as a list; a table segment's pipe grid becomes a
    real table; everything else is faithful paragraphs."""
    kind = str(segment.metadata.get("kind") or "")
    if segment.kind == "table":
        rows = [_split_cells(ln) for ln in (text or "").split("\n")
                if ln.strip()]
        if rows:
            return _table_html(rows, header=len(rows) > 1, inline=False)
        return ""
    if kind == "code":
        return ("<pre><code>" + _esc((text or "").strip())
                + "</code></pre>")
    if kind == "quote":
        quoted = "<br>\n".join(
            _esc(ln.strip().lstrip(">").strip())
            for ln in (text or "").split("\n") if ln.strip())
        return f"<blockquote>{quoted}</blockquote>"
    if kind == "list":
        items = []
        for ln in (text or "").split("\n"):
            item = _BULLET.sub("", ln.strip())
            if item:
                items.append(f"<li>{_esc(item)}</li>")
        return "<ul>\n" + "\n".join(items) + "\n</ul>" if items else ""
    return "\n".join(_paragraphs(text))


# ── anchors + segment rendering ─────────────────────────────────────


def _sanitize_id(segment_id: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", segment_id.lower()).strip("-")
    return cleaned or "segment"


def _anchors_for(manifest: LongDocumentManifest,
                 anchors: Mapping[str, str] | None,
                 ) -> dict[str, str]:
    """Anchor per segment: the caller's map wins (a producer's
    published anchor scheme is its contract — documents' heading
    slugs, forge's ch<n>); the default is the slugified segment
    title, deduped, with the sanitized segment id as fallback."""
    out: dict[str, str] = {}
    seen: set[str] = set()
    for seg in manifest.segments:
        chosen = ""
        if anchors and seg.segment_id in anchors:
            chosen = str(anchors[seg.segment_id])
        if not chosen:
            base = slugify(seg.title, limit=60, fallback="")
            chosen = base or _sanitize_id(seg.segment_id)
            candidate, n = chosen, 2
            while candidate in seen:
                candidate = f"{chosen}-{n}"
                n += 1
            chosen = candidate
        seen.add(chosen)
        out[seg.segment_id] = chosen
    return out


def _segment_label(segment: ManifestSegment) -> str:
    if segment.title:
        return segment.title
    n = segment.ordinal + 1
    if segment.kind == "chapter":
        return f"Chapter {n}"
    if segment.kind == "section":
        return f"Section {n}"
    if segment.kind == "table":
        return f"Table {n}"
    if segment.kind == "summary":
        return "Summary"
    if segment.kind == "overflow":
        return "Overflow"
    return f"Part {n}"


def _heading_level(segment: ManifestSegment) -> int | None:
    """The heading level a segment's title renders at (None = no
    heading: summary/text/overflow segments carry their content
    without a manufactured title).  A documents section keeps the
    level its source declared."""
    if not segment.title:
        return None
    if segment.kind in ("summary", "text", "overflow"):
        return None
    level = segment.metadata.get("level")
    if isinstance(level, int) and level > 0:
        return min(max(level, 1), 6)
    return 2


def _render_segment(segment: ManifestSegment, text: str | None,
                    anchor: str, anchor_mode: str,
                    compat_anchor: bool, mode: str,
                    slugs: dict[str, int]) -> str:
    classes = f"segment kind-{segment.kind}"
    opening = f'<section class="{classes}"'
    if anchor_mode == "section":
        opening += f' id="{_esc(anchor)}"'
    opening += ">"
    parts = [opening]
    if anchor_mode == "compat" and compat_anchor:
        parts.append(f'<a id="{_esc(anchor)}"></a>')
    if not segment.present or text is None:
        level = _heading_level(segment)
        if level is not None:
            parts.append(f"<h{level}>{_esc(segment.title)}</h{level}>")
        parts.append(
            '<p class="absent-note">Not yet available — this part is '
            "declared by the source but is not held in this "
            "edition.</p>")
        parts.append("</section>")
        return "\n".join(parts)
    level = _heading_level(segment)
    if level is not None:
        parts.append(f"<h{level}>{_esc(segment.title)}</h{level}>")
    body = (_render_markdown(text, slugs) if mode == "markdown"
            else _render_plain(segment, text))
    if segment.kind == "summary":
        body = f'<div class="lead">{body}</div>'
    if segment.kind == "table":
        body = f'<div class="table-wrap">{body}</div>'
    parts.append(body)
    parts.append("</section>")
    return "\n".join(parts)


# ── the edition frame ───────────────────────────────────────────────

_CSS = """
:root{--bg:#fdfbf7;--ink:#23232b;--muted:#6b6b76;--accent:#8a5a13;
--panel:#f4efe4;--rule:#ddd5c2;--code-bg:#f4efe4;--measure:46rem}
html[data-theme="dark"]{--bg:#1a1a2e;--ink:#e9e9f0;--muted:#a0a0b4;
--accent:#c9a86a;--panel:#23233c;--rule:#3a3a58;--code-bg:#23233c}
html[data-theme="sepia"]{--bg:#f4ecd8;--ink:#433422;--muted:#857252;
--accent:#8a6d3b;--panel:#ece0c4;--rule:#d8c9a3;--code-bg:#ece0c4}
html[data-theme="print"]{--bg:#ffffff;--ink:#000000;--muted:#444444;
--accent:#000000;--panel:#f2f2f2;--rule:#999999;--code-bg:#f2f2f2}
html[data-theme="minimal"]{--bg:#ffffff;--ink:#222222;--muted:#777777;
--accent:#555555;--panel:#fafafa;--rule:#e2e2e2;--code-bg:#fafafa}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
line-height:1.65}
.page{max-width:var(--measure);margin:0 auto;padding:0 1.25rem 4rem}
body.wide .page{max-width:64rem}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.progress{position:fixed;top:0;left:0;right:0;height:3px;z-index:40;
background:transparent}
.progress span{display:block;height:100%;width:0;background:var(--accent)}
.controlbar{position:sticky;top:0;z-index:30;display:flex;gap:.4rem;
flex-wrap:wrap;align-items:center;padding:.5rem 0;
background:var(--bg);border-bottom:1px solid var(--rule)}
.controlbar .control-label{font-size:.78rem;letter-spacing:.14em;
text-transform:uppercase;color:var(--muted);margin-right:.4rem}
.controlbar button{font:inherit;font-size:.85rem;padding:.28rem .7rem;
border:1px solid var(--rule);border-radius:999px;background:var(--panel);
color:var(--ink);cursor:pointer}
.cover{padding:3.2rem 0 1.6rem;border-bottom:2px solid var(--rule)}
.brand-line{font-size:.8rem;letter-spacing:.22em;text-transform:uppercase;
color:var(--accent);margin:0 0 1.2rem}
.doc-title{font-size:2.3rem;line-height:1.15;margin:.2rem 0 .6rem}
.subtitle{font-size:1.15rem;color:var(--muted);font-style:italic;
margin:.2rem 0}
.author{color:var(--muted);margin:.3rem 0}
.edition-meta{color:var(--muted);font-size:.92rem;margin:.6rem 0 .2rem}
.completeness{font-weight:600;margin:.4rem 0 .2rem}
.completeness-note{color:var(--muted);font-size:.92rem;margin:.2rem 0}
nav.toc{background:var(--panel);border:1px solid var(--rule);
border-radius:10px;padding:1rem 1.4rem;margin:1.6rem 0}
nav.toc h2{font-size:1.05rem;margin:.1rem 0 .6rem}
nav.toc ol,nav.toc ul{margin:0;padding-left:1.4rem}
nav.toc li{margin:.28rem 0}
nav.toc a{text-decoration:none}
nav.toc a:hover{text-decoration:underline}
nav.toc .absent{color:var(--muted)}
nav.toc .absent-note{font-style:italic;font-weight:400}
main{padding-top:.6rem}
.segment{margin:2rem 0}
.segment h2{line-height:1.3}
.absent-note{color:var(--muted);font-style:italic;
border:1px dashed var(--rule);border-radius:8px;padding:.8rem 1rem}
.lead{font-size:1.08rem}
.table-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.95rem}
th,td{border:1px solid var(--rule);padding:.5rem .75rem;text-align:left}
th{background:var(--panel);font-weight:600}
pre{background:var(--code-bg);border:1px solid var(--rule);
border-radius:8px;padding:1rem;overflow-x:auto}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
font-size:.9em}
blockquote{border-left:4px solid var(--rule);margin:1rem 0;
padding:.25rem 1rem;color:var(--muted)}
hr{border:none;border-top:1px solid var(--rule);margin:2rem 0}
.edition-footer{margin-top:3.5rem;padding-top:1.2rem;
border-top:1px solid var(--rule);color:var(--muted);
font-size:.88rem;text-align:center}
.print-mark{display:none}
.treatment-book body,.treatment-book{font-family:Georgia,
"Times New Roman",serif}
.treatment-book .cover{text-align:center;padding-top:14vh}
.treatment-book .segment p{text-align:justify}
.treatment-book .kind-chapter>p:first-of-type::first-letter{
font-size:3.1em;float:left;line-height:.85;padding:.05em .12em 0 0;
color:var(--accent)}
.treatment-book .doc-title{font-size:2.7rem}
.treatment-report .cover{border-bottom:3px solid var(--accent)}
.treatment-report .segment h2{padding-bottom:.35rem;
border-bottom:2px solid var(--rule)}
.treatment-technical{line-height:1.55}
.treatment-technical .doc-title{font-size:1.9rem}
.treatment-technical .segment{margin:1.5rem 0}
.treatment-technical th,.treatment-technical td{padding:.4rem .6rem}
@media print{
@page{margin:18mm}
.controlbar,.progress{display:none}
.page{max-width:none;padding:0}
.treatment-book .kind-chapter{page-break-before:always}
.print-mark{display:block;position:fixed;bottom:0;left:0;right:0;
text-align:center;font-size:8pt;color:#888;letter-spacing:.18em;
text-transform:uppercase}
}
@media (prefers-reduced-motion:reduce){
*{transition:none !important;scroll-behavior:auto !important}
}
"""

_SCRIPT = """
(function(){
var d=document.documentElement;
function save(k,v){try{localStorage.setItem("edition-"+k,v);}catch(e){}}
function load(k){try{return localStorage.getItem("edition-"+k);}
catch(e){return null;}}
var savedTheme=load("theme");
if(savedTheme){d.setAttribute("data-theme",savedTheme);}
var scale=parseInt(load("scale")||"100",10);
if(isNaN(scale)){scale=100;}
function applyScale(){d.style.fontSize=scale+"%";}
applyScale();
var buttons=document.querySelectorAll("[data-theme-set]");
for(var i=0;i<buttons.length;i++){
buttons[i].addEventListener("click",function(){
var v=this.getAttribute("data-theme-set");
d.setAttribute("data-theme",v);save("theme",v);});}
var down=document.getElementById("font-down");
var up=document.getElementById("font-up");
if(down){down.addEventListener("click",function(){
scale=Math.max(80,scale-10);applyScale();save("scale",String(scale));});}
if(up){up.addEventListener("click",function(){
scale=Math.min(140,scale+10);applyScale();save("scale",String(scale));});}
var wide=document.getElementById("width-toggle");
if(wide){wide.addEventListener("click",function(){
document.body.classList.toggle("wide");});}
var bar=document.getElementById("reading-progress");
function prog(){var h=document.documentElement.scrollHeight-window.innerHeight;
var y=window.scrollY||0;var p=h>0?Math.round(100*y/h):0;
if(bar){bar.style.width=p+"%";}}
window.addEventListener("scroll",prog,{passive:true});prog();
})();
"""

_CONTROLBAR = (
    '<header class="controlbar" role="toolbar" '
    'aria-label="Reading controls">\n'
    '<span class="control-label">Reading</span>\n'
    '<button type="button" data-theme-set="light">Light</button>\n'
    '<button type="button" data-theme-set="dark">Dark</button>\n'
    '<button type="button" data-theme-set="sepia">Sepia</button>\n'
    '<button type="button" id="font-down" '
    'aria-label="Smaller text">A−</button>\n'
    '<button type="button" id="font-up" '
    'aria-label="Larger text">A+</button>\n'
    '<button type="button" id="width-toggle">Width</button>\n'
    "</header>"
)


def render_edition(manifest: LongDocumentManifest, resolve: Resolver,
                   *, theme: str = "", treatment: str = "",
                   brand_policy: str = "", brand_text: str = "",
                   toc: bool = True,
                   content_mode: str = "",
                   anchors: Mapping[str, str] | None = None,
                   anchor_mode: str = "section",
                   subtitle: str = "",
                   chrome: bool = True) -> str:
    """Render one self-contained branded HTML edition of a manifest.

    ``resolve`` is the producer's Pass-2 content resolver (segment →
    readable text or None).  ``theme``/``treatment``/``brand_policy``
    override the manifest's hints; ``content_mode`` is "markdown" or
    "plain" (defaulted from the source kind); ``anchors`` maps
    segment ids to a producer's published anchor scheme, rendered on
    the section wrapper (``anchor_mode="section"``, the default) or
    as a leading anchor element (``"compat"``, the documents
    engine's historical form); ``chrome=False`` renders the themed
    body only — no cover, contents, footer, brand or controls — for
    callers that wrap the body in their own chrome.
    """
    brand = _brand_for(manifest, brand_policy, brand_text)
    theme_name = _theme_for(manifest, theme)
    treatment_name = (treatment or "").strip().lower()
    if treatment_name not in TREATMENTS:
        treatment_name = treatment_for(manifest)
    mode = (content_mode or "").strip().lower()
    if mode not in ("markdown", "plain"):
        mode = ("markdown" if manifest.source_kind in _MARKDOWN_SOURCES
                else "plain")
    anchor_map = _anchors_for(manifest, anchors)
    slugs: dict[str, int] = {}

    body_parts: list[str] = []
    if chrome:
        body_parts.append(
            '<div class="progress" aria-hidden="true">'
            '<span id="reading-progress"></span></div>')
        body_parts.append(_CONTROLBAR)
    cover: list[str] = []
    if chrome:
        cover.append('<header class="cover">')
        if brand:
            cover.append(f'<p class="brand-line">{_esc(brand)}</p>')
        title = manifest.title or "Untitled document"
        cover.append(f'<h1 class="doc-title">{_esc(title)}</h1>')
        if subtitle:
            cover.append(f'<p class="subtitle">{_esc(subtitle)}</p>')
        if manifest.author:
            cover.append(
                f'<p class="author">Author: {_esc(manifest.author)}</p>')
        meta = [_SOURCE_LABELS.get(manifest.source_kind, "Document")]
        if manifest.revision is not None:
            meta.append(f"Revision {manifest.revision}")
        cover.append(
            f'<p class="edition-meta">{_esc(" · ".join(meta))}</p>')
        cover.append(
            f'<p class="completeness">'
            f'{_esc(completeness_line(manifest))}</p>')
        note = manifest.completeness.note
        if note:
            cover.append(
                f'<p class="completeness-note">{_esc(note)}</p>')
        cover.append("</header>")
        body_parts.extend(cover)
        if toc and manifest.segments:
            nav = ['<nav class="toc" aria-label="Edition contents">',
                   "<h2>Contents</h2>", "<ol>"]
            for seg in manifest.segments:
                label = _esc(_segment_label(seg))
                if seg.present:
                    href = _esc(anchor_map[seg.segment_id])
                    nav.append(f'<li><a href="#{href}">{label}</a></li>')
                else:
                    nav.append(
                        f'<li class="absent"><span>{label}</span> '
                        '<span class="absent-note">— not yet '
                        "available</span></li>")
            nav.append("</ol>")
            nav.append("</nav>")
            body_parts.append("\n".join(nav))
    body_parts.append("<main>")
    provided = set(anchors or {})
    for seg in manifest.segments:
        text = resolve(seg) if seg.present else None
        body_parts.append(_render_segment(
            seg, text, anchor_map[seg.segment_id], anchor_mode,
            seg.segment_id in provided, mode, slugs))
    body_parts.append("</main>")
    if chrome:
        footer = ['<footer class="edition-footer">']
        if brand:
            footer.append(f"<p>{_esc(brand)}</p>")
        footer.append(
            f"<p>{_esc(completeness_line(manifest))}</p>")
        footer.append("</footer>")
        body_parts.append("\n".join(footer))
        if brand:
            body_parts.append(
                f'<div class="print-mark" aria-hidden="true">'
                f"{_esc(brand)}</div>")
    lang = manifest.presentation.locale or "en"
    direction = manifest.presentation.direction or "auto"
    dir_attr = f' dir="{_esc(direction)}"' if direction in (
        "ltr", "rtl", "auto") else ""
    parts = [
        "<!DOCTYPE html>",
        f'<html lang="{_esc(lang)}" data-theme="{theme_name}"{dir_attr}>',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, '
        'initial-scale=1">',
        f"<title>{_esc(manifest.title or 'Document')}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        f'<body class="treatment-{treatment_name}">',
        '<div class="page">',
        "\n".join(body_parts),
        "</div>",
    ]
    if chrome:
        parts.append(f"<script>{_SCRIPT}</script>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts) + "\n"


def render_edition_from_map(manifest: LongDocumentManifest,
                            contents: Mapping[str, str],
                            **kwargs: Any) -> str:
    """Render an edition from a ``content_ref → text`` map — the
    same shape :meth:`LongDocumentManifest.verify` consumes, so a
    caller can verify and render from one content supply."""
    def resolve(segment: ManifestSegment) -> str | None:
        if not segment.present:
            return None
        return contents.get(segment.content_ref)

    return render_edition(manifest, resolve, **kwargs)
