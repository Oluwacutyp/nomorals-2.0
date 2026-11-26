"""Picture-to-document: one step from an image to an editable document.

Flow: extract (``tools.vision.read_text`` — tesseract first, vision model
second) → correct (the model/brain path — spelling, grammar, OCR errors
fixed where confident, uncertain regions flagged, never invented) → the
BRAIN chooses the document shape at runtime from the live tool listing →
a builder tool renders an editable file → the file is sent back with the
reply.

Two-call shape, and the shape is the point. ``vision_document`` is the
extract-and-correct step only: it returns the corrected text, content
observations, and a listing of the builder tools built LIVE from the
registry (``document_tool_listing`` — never a pasted static list). It
makes NO file. The brain then calls one of the ``document_*`` builder
tools with the content shaped for it — the brain chooses which tool to
call and how to shape the document. The chat handler (``/vision
document``, and read-text delivery) does both steps for the user in one
message: ``picture_to_document`` extracts and corrects, asks the model
to choose and shape (``plan_document``), then invokes the chosen builder
THROUGH the registry (``registry.call``) — the tool is genuinely called,
never bypassed by a direct render call.

The format/shape decision is deliberately NOT a lookup table. There is
no content-kind → format mapping anywhere in this module: no
"letter → Word", no "receipt → spreadsheet". ``content_signals`` reports
observations about the text (line count, table-like rows, salutation,
amounts) and nothing more — it carries no format field and no
recommendation. The only lookup in this file, ``FORMATS``, maps the
USER's own explicit format words ("word", "html", "pdf" ...) to tools —
the user's instruction winning, not a content rule.

Deterministic parts stay code: extraction, the correction guards
(numbers must survive verbatim, ``[illegible]`` markers must survive),
file writing, sending. The CHOICE of document shape is the brain's.
When no model is available to choose, the flow degrades honestly to a
plain-text document and says so — that is a capability fallback (the
brain cannot choose when there is no brain), not a content rule.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..documents import convert
from ..documents import writers as _doc_writers
from ..documents.errors import DocumentError
from ..documents.model import Document, Section, Table, new_document
from ..llm.base import LLMResponse
from .vision import (
    _audit,
    _parse_json_loose,
    read_text,
)

_log = logging.getLogger("nomorals.vision_document")

# Builders the brain can choose from. The listing handed to the model is
# built live from the registry at call time (``document_tool_listing``);
# this tuple is only the set of names to look up there.
DOCUMENT_TOOL_NAMES = (
    "document_word",
    "document_html",
    "document_text",
    "document_pdf",
    "document_sheet",
)

# The USER's own format words → tool. This is the user-specified format
# winning over any choice, never a content rule: no content kind is ever
# keyed here.
FORMATS = {
    "docx": "document_word",
    "word": "document_word",
    "html": "document_html",
    "htm": "document_html",
    "txt": "document_text",
    "text": "document_text",
    "pdf": "document_pdf",
    "xlsx": "document_sheet",
    "sheet": "document_sheet",
    "spreadsheet": "document_sheet",
}

_ILLEGIBLE = "[illegible]"
_NUMBER_RE = re.compile(r"\d[\d\s,]*(?:\.\d+)?")
_TABLE_ROW_RE = re.compile(r"\s{2,}|\t|\|")

_CORRECT_PROMPT = """\
You are correcting text extracted by OCR from a picture. Fix spelling,
grammar, punctuation, and line breaks so the text reads cleanly.

Rules:
- Do NOT change any fact, name, number, date, or amount. Numbers must
  come back exactly as written in the OCR text.
- Text marked [illegible] must stay exactly [illegible] — never guess
  what it says. If a word is uncertain, keep the OCR reading and add
  [?] after it.
- Keep paragraph breaks. Do not add new content, headings, or
  commentary.
- Return ONLY the corrected text, nothing else.
"""

_CHOICE_PROMPT = """\
You are choosing how to turn corrected text from a picture into an
editable document.

The user's instruction: {instruction}
Content observations: {signals}
Corrected text:
{text}

Available document tools (live):
{listing}

Reply with ONLY a JSON object:
{{"tool": "<one of the tool names above>", "reason": "<one plain \
sentence saying why this shape fits the content and the instruction>",
"title": "<a short document title>", "sections": [{{"heading": "",
"level": 1, "text": "...", "kind": "text"}}], "tables": [{{"name": "",
"headers": ["..."], "rows": [["..."]]}}]}}

Rules:
- Every number, name, and date in sections and tables must come from
  the corrected text — invent nothing.
- Keep [illegible] markers wherever the corrected text has them.
- Use tables ONLY for genuinely tabular content (rows with the same
  columns); otherwise use sections.
- Keep the content complete: do not drop paragraphs or rows.
"""

_SHAPE_PROMPT = """\
You are shaping corrected text from a picture into a document for the
{tool} tool ({tool_desc}).

The user's instruction: {instruction}
Content observations: {signals}
Corrected text:
{text}

Reply with ONLY a JSON object:
{{"title": "<a short document title>", "sections": [{{"heading": "",
"level": 1, "text": "...", "kind": "text"}}], "tables": [{{"name": "",
"headers": ["..."], "rows": [["..."]]}}]}}

Rules:
- Every number, name, and date in sections and tables must come from
  the corrected text — invent nothing.
- Keep [illegible] markers wherever the corrected text has them.
- Use tables ONLY for genuinely tabular content (rows with the same
  columns); otherwise use sections.
- Keep the content complete: do not drop paragraphs or rows.
"""


def _numbers(text: str) -> Counter[str]:
    """Digit groups, normalised (commas/spaces stripped) — the guard
    that proves a correction did not touch any number."""
    return Counter(m.group(0).replace(",", "").replace(" ", "")
                   for m in _NUMBER_RE.finditer(text or ""))


def _is_mock(response: Any) -> bool:
    return getattr(response, "provider", "") == "mock" or bool(
        getattr(response, "error", ""))


def correct_text(context: Any, raw_text: str, *, source: str = "") -> dict[str, Any]:
    """Correct OCR text through the model/brain path, guarded.

    Confident corrections are applied; uncertain regions stay flagged;
    numbers and ``[illegible]`` markers must survive verbatim. When no
    model is available, or the model is the mock, or a guard fails, the
    raw OCR text is returned uncorrected with a note naming why — never
    a silent "correction".
    """
    raw = (raw_text or "").strip()
    if not raw:
        return {"text": "", "corrected": False,
                "note": "no text to correct"}
    router = getattr(context, "router", None) if context is not None else None
    if router is None:
        return {"text": raw, "corrected": False,
                "note": "no model available — text is uncorrected OCR"}
    try:
        response: LLMResponse = router.complete(
            _CORRECT_PROMPT + "\nOCR text:\n" + raw)
    except Exception as exc:  # noqa: BLE001 - correction is best-effort
        return {"text": raw, "corrected": False,
                "note": f"correction unavailable ({exc}) — "
                        "text is uncorrected OCR"}
    if _is_mock(response):
        return {"text": raw, "corrected": False,
                "note": "no real model available (mock) — "
                        "text is uncorrected OCR"}
    corrected = (getattr(response, "text", "") or "").strip()
    if not corrected:
        return {"text": raw, "corrected": False,
                "note": "the model returned no correction — "
                        "text is uncorrected OCR"}
    if _numbers(corrected) != _numbers(raw):
        return {"text": raw, "corrected": False,
                "note": "correction rejected: it changed a number — "
                        "text is uncorrected OCR"}
    if corrected.count(_ILLEGIBLE) < raw.count(_ILLEGIBLE):
        return {"text": raw, "corrected": False,
                "note": "correction rejected: it dropped an [illegible] "
                        "marker — text is uncorrected OCR"}
    note = "corrected by the model"
    if getattr(response, "degraded", False):
        note += " (served by a fallback provider)"
    return {"text": corrected, "corrected": True, "note": note,
            "provider": getattr(response, "provider", ""),
            "model": getattr(response, "model", ""),
            "usage": getattr(response, "usage", {}) or {}}


def content_signals(text: str) -> dict[str, Any]:
    """Observations about the text — descriptive facts only.

    Deliberately carries NO format field and NO recommendation: it is
    evidence for the brain's runtime choice, never a decision. Any
    content-kind → format mapping built on these signals would be the
    encoded direction the owner rejected.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    words = (text or "").split()
    table_like = 0
    for ln in lines:
        cells = [c for c in _TABLE_ROW_RE.split(ln) if c.strip()]
        if len(cells) >= 2 and any(ch.isdigit() for ch in ln):
            table_like += 1
    first = lines[0].strip().lower() if lines else ""
    salutation = bool(re.match(
        r"(dear|hello|hi|to whom|respected|sir|madam)\b", first))
    has_list = any(re.match(r"\s*([-*•]|\d+[.)])\s", ln) for ln in lines)
    return {
        "lines": len(lines),
        "words": len(words),
        "table_like_rows": table_like,
        "has_salutation": salutation,
        "has_list_markers": has_list,
        "has_amounts": bool(re.search(r"[₦$€£]\s*\d|\d+\.\d{2}\b",
                                      text or "")),
        "illegible_markers": (text or "").count(_ILLEGIBLE),
    }


def document_tool_listing(registry: Any) -> str:
    """The builder tools, listed LIVE from the registry — never a
    pasted static list. Only tools actually registered are offered, so
    the brain can only choose a tool that exists."""
    lines = []
    for name in DOCUMENT_TOOL_NAMES:
        spec = registry.get(name) if registry is not None else None
        if spec is None:
            continue
        desc = (getattr(spec, "description", "") or "").strip()
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _shaped_numbers_ok(shaped: dict[str, Any], text: str) -> bool:
    """Every number in the shaped output must already exist in the
    corrected text; every [illegible] marker must survive."""
    parts: list[str] = []
    for section in shaped.get("sections") or []:
        if isinstance(section, dict):
            parts.append(str(section.get("heading", "")))
            parts.append(str(section.get("text", "")))
    for table in shaped.get("tables") or []:
        if isinstance(table, dict):
            parts.append(str(table.get("name", "")))
            parts.extend(str(h) for h in table.get("headers") or [])
            for row in table.get("rows") or []:
                parts.extend(str(c) for c in row)
    joined = "\n".join(parts)
    for number, count in _numbers(joined).items():
        if _numbers(text)[number] < count:
            return False
    return joined.count(_ILLEGIBLE) >= text.count(_ILLEGIBLE)


def _plain_fallback(text: str, title: str, reason: str) -> dict[str, Any]:
    """Honest degradation: plain text, whole content, reason stated."""
    return {
        "tool": "document_text",
        "reason": reason,
        "title": title or "Document from picture",
        "sections": [{"heading": "", "level": 1, "text": text,
                      "kind": "text"}],
        "tables": [],
        "fallback": True,
    }


def plan_document(context: Any, registry: Any, *, text: str,
                  instruction: str = "", tool: str | None = None,
                  title: str = "") -> dict[str, Any]:
    """Let the brain choose (and shape) the document at runtime.

    With ``tool`` given (the user's explicit format), the model only
    shapes the content for that tool. Without it, the model also picks
    the tool from the live listing. On any failure — no model, mock,
    unparseable reply, an invented number, a dropped [illegible] — the
    plan falls back to a plain-text document carrying the whole text,
    and says why.
    """
    clean = (text or "").strip()
    signals = content_signals(clean)
    title = title or "Document from picture"
    if not clean:
        raise ToolError("no corrected text to shape into a document")
    router = getattr(context, "router", None) if context is not None else None
    if router is None:
        return _plain_fallback(
            clean, title,
            "no model available to choose the document shape — plain text")
    listing = document_tool_listing(registry)
    if tool is not None:
        if tool not in DOCUMENT_TOOL_NAMES:
            raise ToolError(f"unknown document tool {tool!r}")
        spec = registry.get(tool) if registry is not None else None
        prompt = _SHAPE_PROMPT.format(
            tool=tool, tool_desc=(getattr(spec, "description", "") or ""),
            instruction=instruction or "(none — make a good document)",
            signals=signals, text=clean)
    else:
        prompt = _CHOICE_PROMPT.format(
            instruction=instruction or "(none — make a good document)",
            signals=signals, text=clean, listing=listing)
    try:
        response: LLMResponse = router.complete(prompt)
    except Exception as exc:  # noqa: BLE001 - choice is best-effort
        return _plain_fallback(
            clean, title,
            f"document choice unavailable ({exc}) — plain text")
    if _is_mock(response):
        return _plain_fallback(
            clean, title,
            "no real model available (mock) to choose the document "
            "shape — plain text")
    plan, ok = _parse_json_loose(getattr(response, "text", "") or "")
    if not ok or plan is None:
        return _plain_fallback(
            clean, title,
            "the model did not return a usable document plan — plain text")
    chosen = plan.get("tool", tool) if tool is None else tool
    if chosen not in DOCUMENT_TOOL_NAMES:
        return _plain_fallback(
            clean, title,
            f"the model chose an unknown document tool {chosen!r} — "
            "plain text")
    shaped = {
        "title": str(plan.get("title", "") or title),
        "sections": plan.get("sections") or [],
        "tables": plan.get("tables") or [],
    }
    if not isinstance(shaped["sections"], list):
        shaped["sections"] = []
    if not isinstance(shaped["tables"], list):
        shaped["tables"] = []
    if not shaped["sections"] and not shaped["tables"]:
        return _plain_fallback(
            clean, title,
            "the model returned an empty document plan — plain text")
    if not _shaped_numbers_ok(shaped, clean):
        return _plain_fallback(
            clean, title,
            "the model's document plan invented a number or dropped an "
            "[illegible] marker — plain text")
    reason = str(plan.get("reason", "") or "")
    if not reason:
        reason = ("you asked for this format" if tool is not None
                  else "chosen by the model")
    return {
        "tool": chosen,
        "reason": reason,
        "title": shaped["title"],
        "sections": shaped["sections"],
        "tables": shaped["tables"],
        "fallback": False,
    }


# ── structure → Document → bytes ─────────────────────────────────────────────


def structure_to_document(structure: dict[str, Any]) -> Document:
    """A normalised structure (title/sections/tables) → the documents
    organ's Document. Section/Table payloads are validated by their own
    ``from_dict`` constructors — a bad payload raises, never renders
    half a document."""
    doc = new_document(format="md",
                       title=str(structure.get("title", "") or ""),
                       source="vision_document")
    for raw_section in structure.get("sections") or []:
        if not isinstance(raw_section, dict):
            raise DocumentError("bad section payload: not an object")
        doc.sections.append(Section.from_dict(raw_section))
    for raw_table in structure.get("tables") or []:
        if not isinstance(raw_table, dict):
            raise DocumentError("bad table payload: not an object")
        doc.tables.append(Table.from_dict(raw_table))
    if not doc.sections and not doc.tables:
        raise DocumentError("document has no content to render",
                            kind="empty")
    return doc


def render_word(structure: dict[str, Any]) -> bytes:
    """Real .docx: real tables for tabular content, headings and
    paragraphs for prose (python-docx via the documents organ)."""
    return convert.to_docx(structure_to_document(structure))


def render_html(structure: dict[str, Any]) -> str:
    """A styled, print-friendly HTML page (the documents organ's print
    theme — serif, print CSS, self-contained)."""
    return convert.to_html(structure_to_document(structure), theme="print",
                           toc=False)


def render_text(structure: dict[str, Any]) -> str:
    """Plain text: title, headings, prose, tables as ``a | b`` lines."""
    doc = structure_to_document(structure)
    lines: list[str] = []
    if doc.title:
        lines.append(doc.title)
        lines.append("")
    for section in doc.sections:
        if section.heading:
            lines.append("#" * max(1, min(6, section.level))
                         + " " + section.heading)
            lines.append("")
        if section.text and section.text.strip():
            lines.append(section.text)
            lines.append("")
    for table in doc.tables:
        if table.name:
            lines.append(table.name)
        if table.headers:
            lines.append(" | ".join(table.headers))
        for row in table.rows:
            lines.append(" | ".join(row))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_pdf(structure: dict[str, Any]) -> bytes:
    """PDF via the documents organ's pure-Python writer."""
    return convert.to_pdf(structure_to_document(structure))


def render_sheet(structure: dict[str, Any]) -> bytes:
    """Spreadsheet via the documents organ's stdlib xlsx writer — one
    worksheet per table; prose rides a leading Document sheet."""
    return _doc_writers.to_xlsx(structure_to_document(structure))


_RENDERERS = {
    "document_word": (render_word, "docx"),
    "document_html": (render_html, "html"),
    "document_text": (render_text, "txt"),
    "document_pdf": (render_pdf, "pdf"),
    "document_sheet": (render_sheet, "xlsx"),
}


def structure_from_parts(title: str = "", text: str = "",
                         sections_json: str = "",
                         tables_json: str = "") -> dict[str, Any]:
    """Builder-tool inputs → a normalised structure. ``text`` alone
    becomes one prose section; ``sections_json``/``tables_json`` are
    JSON lists of Section/Table payloads."""
    sections: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    if sections_json:
        try:
            parsed = json.loads(sections_json)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ToolError(
                f"sections_json is not valid JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise ToolError("sections_json must be a JSON list")
        sections = parsed
    if tables_json:
        try:
            parsed = json.loads(tables_json)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ToolError(
                f"tables_json is not valid JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise ToolError("tables_json must be a JSON list")
        tables = parsed
    if not sections and (text or "").strip():
        sections = [{"heading": "", "level": 1, "text": text.strip(),
                     "kind": "text"}]
    if not sections and not tables:
        raise ToolError("a document needs text or a table to build")
    return {"title": title or "Document", "sections": sections,
            "tables": tables}


def _stage_dir(context: Any) -> Path:
    """Where built documents land: the settings home's data dir, the
    workspace dir, or a temp dir — in that order."""
    settings = getattr(context, "settings", None) if context is not None else None
    home = getattr(settings, "home_path", None) if settings is not None else None
    if home is not None:
        base = Path(home) / "data" / "vision_docs"
    else:
        work = getattr(settings, "workspace_dir", "") if settings is not None else ""
        base = (Path(work) / "vision_docs") if work else Path(
            tempfile.mkdtemp(prefix="vision_docs_"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _safe_filename(title: str, ext: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (title or "document").strip())
    slug = slug.strip("._") or "document"
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{slug[:60]}-{stamp}.{ext}"


def stage_document(context: Any, data: bytes | str,
                   filename: str) -> Path:
    """Write built document bytes into the staging dir; returns the
    path. Never writes outside the staging dir."""
    name = Path(filename).name
    if not name:
        raise ToolError("cannot stage a document with no filename")
    path = _stage_dir(context) / name
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode, encoding=None if isinstance(data, bytes) else "utf-8") as fh:
        fh.write(data)
    return path


def build_document_file(context: Any, tool: str, title: str = "",
                        text: str = "", sections_json: str = "",
                        tables_json: str = "") -> dict[str, Any]:
    """The builder tools' shared implementation: structure → render →
    stage. ``tool`` is one of the live builder tool names."""
    if tool not in _RENDERERS:
        raise ToolError(f"unknown document tool {tool!r}")
    structure = structure_from_parts(title, text, sections_json, tables_json)
    render, ext = _RENDERERS[tool]
    try:
        data = render(structure)
    except DocumentError as exc:
        raise ToolError(f"could not build the document: {exc}") from exc
    filename = _safe_filename(structure.get("title", ""), ext)
    path = stage_document(context, data, filename)
    size = len(data) if isinstance(data, (bytes, str)) else 0
    _log.info("vision_document build tool=%s file=%s bytes=%d",
              tool, filename, size)
    return {"path": str(path), "filename": filename, "format": ext,
            "tool": tool, "bytes": size,
            "sections": len(structure["sections"]),
            "tables": len(structure["tables"])}


def combine_texts(parts: list[dict[str, Any]]) -> str:
    """Per-image corrected texts → one combined text for the choice
    prompt, one marked block per picture, in order."""
    blocks = []
    for i, part in enumerate(parts, 1):
        source = part.get("source", "") or f"picture {i}"
        blocks.append(f"--- Picture {i} ({source}) ---\n{part['text']}")
    return "\n\n".join(blocks)


def picture_to_document(context: Any, registry: Any, *,
                        images: list[tuple[bytes, str]],
                        instruction: str = "",
                        format: str | None = None) -> dict[str, Any]:
    """One step: pictures → editable document file.

    ``images`` is a list of ``(bytes, source)`` pairs, in order —
    several pictures go into ONE document, one section per picture in
    the combined text the brain shapes. ``format`` is the user's
    explicit format word when given (it wins); otherwise the brain
    chooses at runtime. Raises :class:`ToolError` — never a fake
    document — when the text cannot be read (both OCR rungs missing),
    when no text is found, or when the document cannot be built.
    """
    if not images:
        raise ToolError("no image to turn into a document")
    from ..vision.seer import UNTRUSTED_VISION_PREFIX

    tool = None
    if format:
        tool = FORMATS.get(format.strip().lower())
        if tool is None:
            raise ToolError(
                f"unknown document format {format!r} "
                f"(want one of: {', '.join(sorted(FORMATS))})")
    parts: list[dict[str, Any]] = []
    methods: list[str] = []
    sources: list[str] = []
    for data, source in images:
        result = read_text(context, data, source=source or "bytes")
        text = (result.get("text") or "").strip()
        if text.startswith(UNTRUSTED_VISION_PREFIX):
            text = text[len(UNTRUSTED_VISION_PREFIX):].strip()
        if text.startswith("[vision unavailable"):
            raise ToolError(
                "could not read the image: tesseract OCR is not "
                "available and the vision model is unavailable "
                f"({text}) — install tesseract or configure a vision "
                "model; no document was made")
        if not text:
            continue
        corrected = correct_text(context, text, source=source or "bytes")
        _audit("document", source or "bytes", len(data),
               len(corrected["text"]),
               corrected.get("provider", "") or result.get("provider", ""),
               corrected.get("model", "") or result.get("model", ""),
               corrected.get("usage", {}) or {}, context)
        parts.append({"source": source or "bytes",
                      "text": corrected["text"],
                      "corrected": corrected["corrected"],
                      "note": corrected["note"]})
        methods.append(str(result.get("method", "?")))
        sources.append(source or "bytes")
    if not parts:
        raise ToolError(
            "no text found in the image(s) — nothing to turn into a "
            "document; no empty file was sent")
    combined = combine_texts(parts)
    plan = plan_document(context, registry, text=combined,
                         instruction=instruction, tool=tool)
    # The builder is invoked THROUGH the registry — the tool is genuinely
    # called, never bypassed by a direct render call. This is the same
    # call the brain makes in the tool loop, and the same call the
    # document_* tools expose to it.
    outcome = registry.call(
        plan["tool"],
        title=plan["title"],
        sections_json=json.dumps(plan["sections"]),
        tables_json=json.dumps(plan["tables"]),
    )
    if not outcome.ok:
        raise ToolError(
            f"the {plan['tool']} builder failed: "
            f"{getattr(outcome.error, 'message', outcome.error) or 'unknown error'}"
            " — no document was made")
    built = outcome.value or {}
    if not built.get("path"):
        raise ToolError(
            f"the {plan['tool']} builder returned no file — "
            "no document was made")
    uncertain = sum(part["text"].count(_ILLEGIBLE) for part in parts)
    return {
        "path": built["path"],
        "filename": built.get("filename", ""),
        "format": built.get("format", ""),
        "tool": plan["tool"],
        "reason": plan["reason"],
        "corrected": all(part["corrected"] for part in parts),
        "correction_notes": [part["note"] for part in parts],
        "method": "+".join(sorted(set(methods))) or "?",
        "sources": sources,
        "uncertain": uncertain,
        "fallback": plan.get("fallback", False),
        "text": combined,
        "text_preview": combined[:400],
    }
