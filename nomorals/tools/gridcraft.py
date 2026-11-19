"""GridCraft: open, read, edit and recalculate real .xlsx files
(GridCraft port of storytold/gridcraft — a clean-room spreadsheet).

A thin surface over :mod:`nomorals.documents.sheets` — the organ
owns the workbook model, the formula engine, the calc graph and
the fidelity-preserving editor; the tool only routes arguments and
shapes the reply. Pure local work, stdlib only, no upload, no
retention. Writes output files, so FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..documents.sheets import (
    SheetEditor,
    Workbook,
)
from ..documents.sheets import (
    run_parity as run_gridcraft_parity,
)
from ..documents.sheets import (
    verify_attribution as verify_gridcraft_attribution,
)
from ..documents.sheets.calc import recalc

__all__ = ["register"]

_ACTIONS = ("inspect", "create", "set", "formula", "style", "add",
            "recalc", "metadata", "parity", "attribution")


def _parse_text(text: str) -> Any:
    """A set value: int, then float, then the raw string."""
    stripped = (text or "").strip()
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    return text


def _cell_map(wb: Workbook) -> dict[str, Any]:
    """Every formula cell as "Sheet!Ref" → cached value."""
    from ..documents.sheets.model import CellError

    out: dict[str, Any] = {}
    for ws in wb.sheets:
        for cell in ws.cells.values():
            if cell.formula is None:
                continue
            cached = cell.cached
            if isinstance(cached, CellError):
                cached = cached.display
            out[f"{ws.name}!{cell.ref}"] = cached
    return out


def register(registry: Any) -> None:
    """Attach the gridcraft family to a registry."""
    @registry.register(
        "gridcraft",
        description=(
            "Open, read, edit and recalculate real .xlsx spreadsheets, "
            "local and uncapped — inspect sheets, cells, formulas, "
            "styles and named ranges; set values and formulas; restyle "
            "cells; add sheets; insert rows and columns; recalculate "
            "the whole workbook; or run the parity TOML and the "
            "attribution check. Edits preserve the untouched parts "
            "byte-for-byte. GridCraft actions (spreadsheet model + "
            "calc engine + fidelity-preserving edits): "
            "inspect=<read the workbook inventory, sources=[path]>; "
            "create=<blank workbook, output=<path>, title=<t>, "
            "author=<a>>; set=<set a value, sheet=<s>, ref=<A1>, "
            "text=<value>>; formula=<set a formula, sheet=<s>, "
            "ref=<A1>, text=<formula>>; style=<restyle a cell, "
            "sheet=<s>, ref=<A1>, bold=<true|false>, color=<hex>, "
            "number_format=<fmt>>; add=<add a sheet, text=<name>>; "
            "recalc=<recalculate, sources=[path]>; metadata=<set "
            "title/author, title=<t>, author=<a>>; parity=<run the "
            "machine-checked parity TOML>; attribution=<verify the "
            "SHA-256 attribution DB>. action=<inspect|create|set|"
            "formula|style|add|recalc|metadata|parity|attribution>; "
            "sources=<list of .xlsx paths>; output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def gridcraft_tool(action: str = "inspect",
                       sources: list[str] | None = None,
                       output: str = "", text: str = "",
                       sheet: str = "", ref: str = "",
                       bold: bool | None = None,
                       italic: bool | None = None,
                       color: str = "", number_format: str = "",
                       title: str = "", author: str = ""
                       ) -> dict[str, Any]:
        action = (action or "inspect").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown gridcraft action {action!r}; use "
                             f"one of {', '.join(_ACTIONS)}")
        sources = list(sources or [])
        if action == "inspect":
            if not sources:
                raise ValueError("gridcraft inspect needs a source .xlsx")
            wb = Workbook.from_bytes(Path(sources[0]).read_bytes())
            summary = SheetEditor(wb).summary()
            summary["reply"] = (
                f"{sources[0]}: .xlsx, "
                f"{len(summary['sheets'])} sheet(s), "
                f"{sum(s['cells'] for s in summary['sheets'])} cell(s), "
                f"{sum(s['formulas'] for s in summary['sheets'])} "
                f"formula(s): "
                + ", ".join(s["name"] for s in summary["sheets"])
                + ".")
            return summary
        if action == "parity":
            result = run_gridcraft_parity()
            failed = result["failed"]
            result["reply"] = (
                f"gridcraft parity: {result['passed']} passed, "
                f"{len(failed)} failed"
                + (f" ({', '.join(failed)})" if failed else "") + ".")
            return result
        if action == "attribution":
            result = verify_gridcraft_attribution()
            result["reply"] = (
                f"gridcraft attribution: {result['checked']} verified, "
                f"{len(result['failed'])} failed, "
                f"{len(result['skipped'])} skipped (clone not on disk).")
            return result
        if action == "recalc":
            if not sources:
                raise ValueError("gridcraft recalc needs a source .xlsx")
            wb = Workbook.from_bytes(Path(sources[0]).read_bytes())
            result = recalc(wb)
            cells = _cell_map(wb)
            return {
                "ok": True,
                "cells": cells,
                "changed": result.changed,
                "cycles": result.cycles,
                "errors": result.errors,
                "reply": (
                    f"recalculated {sources[0]}: "
                    f"{len(cells)} formula cell(s), "
                    f"{len(result.cycles)} cycle(s)"
                    + (f" ({'; '.join(','.join(c) for c in result.cycles)})"
                       if result.cycles else "") + "."),
            }
        if action == "create":
            if not output:
                raise ValueError("gridcraft create needs output=<path>")
            wb = Workbook.blank()
            if title or author:
                editor = SheetEditor(wb)
                editor.set_metadata(title=title or "",
                                    author=author or "")
            saved = wb.to_bytes()
            out_path = Path(output).expanduser()
            out_path.write_bytes(saved)
            return {"ok": True, "output": str(out_path),
                    "sheets": wb.sheet_names,
                    "size_after": len(saved),
                    "reply": f"created blank .xlsx at {out_path}."}
        if not sources:
            raise ValueError(f"gridcraft {action} needs a source .xlsx")
        if not output:
            raise ValueError(f"gridcraft {action} needs output=<path>")
        original = Path(sources[0]).read_bytes()
        wb = Workbook.from_bytes(original)
        editor = SheetEditor(wb)
        if action == "set":
            if not sheet or not ref:
                raise ValueError(
                    "gridcraft set needs sheet=<s> and ref=<A1>")
            value = _parse_text(text)
            editor.set_value(sheet, ref, value)
            detail = f"set {sheet}!{ref} to {value!r}"
            extra: dict[str, Any] = {"value": value}
        elif action == "formula":
            if not sheet or not ref:
                raise ValueError(
                    "gridcraft formula needs sheet=<s> and ref=<A1>")
            editor.set_formula(sheet, ref, text)
            detail = f"set {sheet}!{ref} to {text!r}"
            extra = {}
        elif action == "style":
            if not sheet or not ref:
                raise ValueError(
                    "gridcraft style needs sheet=<s> and ref=<A1>")
            kwargs: dict[str, Any] = {}
            if bold is not None:
                kwargs["bold"] = bold
            if italic is not None:
                kwargs["italic"] = italic
            if color:
                kwargs["color"] = color
            if number_format:
                kwargs["number_format"] = number_format
            if not kwargs:
                raise ValueError(
                    "gridcraft style needs at least one of bold, "
                    "italic, color, number_format")
            editor.restyle(sheet, ref, **kwargs)
            detail = f"restyled {sheet}!{ref}"
            extra = {}
        elif action == "add":
            if not text:
                raise ValueError(
                    "gridcraft add needs text=<new sheet name>")
            editor.add_sheet(text)
            detail = f"added sheet {text!r}"
            extra = {}
        elif action == "metadata":
            fields: dict[str, str] = {}
            if title:
                fields["title"] = title
            if author:
                fields["author"] = author
            if not fields:
                raise ValueError(
                    "gridcraft metadata needs title=<t> and/or "
                    "author=<a>")
            editor.set_metadata(**fields)
            detail = "metadata set: " + ", ".join(
                f"{key}={value!r}" for key, value in fields.items())
            extra = {}
        else:  # pragma: no cover - guarded by _ACTIONS above
            raise ValueError(f"unknown gridcraft action {action!r}")
        recalc(wb)
        saved = editor.to_bytes()
        out_path = Path(output).expanduser()
        out_path.write_bytes(saved)
        result: dict[str, Any] = {
            "ok": True, "output": str(out_path),
            "sheets": wb.sheet_names,
            "cells": _cell_map(wb),
            "size_before": len(original),
            "size_after": len(saved),
            "reply": f"{detail}; saved to {out_path}."}
        result.update(extra)
        return result
