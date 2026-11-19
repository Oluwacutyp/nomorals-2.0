"""WordCraft: open, read, edit and save real .docx files (WordCraft
port of storytold/wordcraft — a clean-room word processor).

A thin surface over :mod:`nomorals.documents.word` — the organ owns
the OOXML package, the document model and the fidelity-preserving
editor; the tool only routes arguments and shapes the reply. Pure
local work, stdlib only, no upload, no retention. Writes output
files, so FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..documents.word import (
    WordDocument,
    WordEditor,
)
from ..documents.word import (
    run_parity as run_wordcraft_parity,
)
from ..documents.word import (
    verify_attribution as verify_wordcraft_attribution,
)

__all__ = ["register"]

_ACTIONS = ("inspect", "create", "replace", "style", "add", "table",
            "metadata", "parity", "attribution")


def _split_list(value: str | list[str] | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_rows(value: str | list[Any] | None) -> list[list[str]]:
    if not value:
        return []
    if isinstance(value, list):
        return [[str(cell) for cell in row] for row in value]
    rows: list[list[str]] = []
    for row in value.split(";"):
        rows.append([cell.strip() for cell in row.split("|")])
    return rows


def register(registry: Any) -> None:
    """Attach the wordcraft family to a registry."""
    @registry.register(
        "wordcraft",
        description=(
            "Open, read, edit and save real .docx word-processor "
            "documents, local and uncapped — inspect paragraphs, runs, "
            "styles, headings, lists, tables, images, headers and "
            "footers; replace text, restyle runs and paragraphs, add "
            "headings, paragraphs, list items and tables, set document "
            "metadata, or create a blank document. Edits preserve the "
            "untouched parts byte-for-byte. WordCraft actions "
            "(document model + fidelity-preserving edits): "
            "inspect=<read the document inventory, sources=[path]>; "
            "create=<blank document, output=<path>, title=<t>, "
            "author=<a>>; replace=<replace text, text=<old>, "
            "new_text=<new>, first_only=<true|false>>; style=<restyle, "
            "para=<n>, run=<n>, bold=<true|false>, italic=<true|false>, "
            "color=<hex>, size=<half-points>, style_name=<paragraph "
            "style>>; add=<add a block, kind=<heading|paragraph|list>, "
            "text=<t>, level=<n>, ordered=<true|false>>; "
            "table=<append a table, headers=<a,b>, rows=<1|2;3|4>>; "
            "metadata=<set title/author, title=<t>, author=<a>>; "
            "parity=<run the machine-checked parity TOML>; "
            "attribution=<verify the SHA-256 attribution DB>. "
            "action=<inspect|create|replace|style|add|table|metadata|"
            "parity|attribution>; sources=<list of .docx paths>; "
            "output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def wordcraft_tool(action: str = "inspect",
                       sources: list[str] | None = None,
                       output: str = "", text: str = "",
                       new_text: str = "", first_only: bool = False,
                       para: int = 0, run: int = 0,
                       bold: bool | None = None,
                       italic: bool | None = None,
                       color: str = "", size: int = 0,
                       style_name: str = "", kind: str = "paragraph",
                       level: int = 1, ordered: bool = False,
                       headers: str | list[str] | None = None,
                       rows: str | list[Any] | None = None,
                       title: str = "", author: str = ""
                       ) -> dict[str, Any]:
        action = (action or "inspect").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown wordcraft action {action!r}; use "
                             f"one of {', '.join(_ACTIONS)}")
        sources = list(sources or [])
        if action == "inspect":
            if not sources:
                raise ValueError("wordcraft inspect needs a source .docx")
            summary = WordDocument.from_bytes(
                Path(sources[0]).read_bytes()).to_dict()
            summary["reply"] = (
                f"{sources[0]}: .docx, "
                f"{len(summary['paragraphs'])} paragraph(s), "
                f"{len(summary['tables'])} table(s), "
                f"{len(summary['images'])} image(s), "
                f"{len(summary['headers'])} header(s), "
                f"{len(summary['footers'])} footer(s).")
            return summary
        if action == "parity":
            result = run_wordcraft_parity()
            failed = [c["id"] for c in result["cases"]
                      if c["status"] == "fail"]
            result["reply"] = (
                f"wordcraft parity: {result['passed']} passed, "
                f"{result['failed']} failed"
                + (f" ({', '.join(failed)})" if failed else "") + ".")
            return result
        if action == "attribution":
            result = verify_wordcraft_attribution()
            result["reply"] = (
                f"wordcraft attribution: {result['checked']} verified, "
                f"{result['failed']} failed, "
                f"{result['skipped']} skipped (clone not on disk).")
            return result
        if action == "create":
            if not output:
                raise ValueError("wordcraft create needs output=<path>")
            editor = WordEditor.create(title=title or None,
                                       author=author or None)
            saved = editor.save()
            out_path = Path(output).expanduser()
            out_path.write_bytes(saved)
            return {"ok": True, "output": str(out_path),
                    "paragraphs": len(editor.document.paragraphs),
                    "size_after": len(saved),
                    "reply": f"created blank .docx at {out_path}."}
        if not sources:
            raise ValueError(f"wordcraft {action} needs a source .docx")
        if not output:
            raise ValueError(f"wordcraft {action} needs output=<path>")
        original = Path(sources[0]).read_bytes()
        editor = WordEditor.from_bytes(original)
        if action == "replace":
            if not text:
                raise ValueError("wordcraft replace needs text=<old>")
            replaced = editor.replace_text(text, new_text,
                                           first_only=first_only)
            detail = f"replaced {replaced} occurrence(s) of " \
                     f"{text!r} with {new_text!r}"
        elif action == "style":
            if style_name:
                editor.set_paragraph_style(para, style_name)
                detail = (f"paragraph {para} restyled to "
                          f"{style_name!r}")
            else:
                editor.set_run_format(
                    para, run, bold=bold, italic=italic,
                    color=color or None,
                    size_half_points=size or None)
                detail = f"run {run} of paragraph {para} restyled"
        elif action == "add":
            kind = (kind or "paragraph").lower()
            if kind == "heading":
                editor.add_heading(text, level)
            elif kind == "list":
                editor.add_list_item(text, ordered=ordered)
            elif kind == "paragraph":
                editor.add_paragraph(text)
            else:
                raise ValueError(
                    f"wordcraft add kind {kind!r}; use heading, "
                    f"paragraph, or list")
            detail = f"added {kind} {text!r}"
        elif action == "table":
            head = _split_list(headers)
            body = _split_rows(rows)
            if not head and not body:
                raise ValueError(
                    "wordcraft table needs headers=<a,b> and/or "
                    "rows=<1|2;3|4>")
            editor.add_table(head, body)
            detail = (f"added table {len(head)} column(s), "
                      f"{len(body)} row(s)")
        elif action == "metadata":
            fields: dict[str, str] = {}
            if title:
                fields["title"] = title
            if author:
                fields["author"] = author
            if not fields:
                raise ValueError(
                    "wordcraft metadata needs title=<t> and/or "
                    "author=<a>")
            editor.set_metadata(**fields)
            detail = "metadata set: " + ", ".join(
                f"{key}={value!r}" for key, value in fields.items())
        else:  # pragma: no cover - guarded by _ACTIONS above
            raise ValueError(f"unknown wordcraft action {action!r}")
        saved = editor.save()
        out_path = Path(output).expanduser()
        out_path.write_bytes(saved)
        result: dict[str, Any] = {
            "ok": True, "output": str(out_path),
            "paragraphs": len(editor.document.paragraphs),
            "tables": len(editor.document.tables),
            "size_before": len(original),
            "size_after": len(saved),
            "reply": f"{detail}; saved to {out_path}."}
        if action == "replace":
            result["replacements"] = replaced
        return result
