"""Pagewise: merge/split/rotate/stamp/compress/repair/protect/unlock/
archive/scan/sign over PDFs (Smallpdf #48 + ILovePDF #49 core).

A thin surface over :mod:`nomorals.documents.pagewise` — the organ
owns the object-graph surgery; the tool only routes arguments and
shapes the reply. Pure local work, no upload, no retention, no daily
cap. Writes output files, so FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..documents.pagewise import (
    ArchiveOp,
    CompressOp,
    CropOp,
    DeleteOp,
    MergeOp,
    PagewiseJob,
    ProtectOp,
    ReorderOp,
    RepairOp,
    RotateOp,
    ScanOp,
    SelectOp,
    SignOp,
    StampOp,
    UnlockOp,
    format_pagewise_reply,
    pdf_info,
    run_pagewise,
)
from ..documents.pdfcraft import PdfDocument, PdfEditor
from ..documents.pdfcraft import run_parity as run_pdfcraft_parity
from ..documents.pdfcraft import (
    verify_attribution as verify_pdfcraft_attribution,
)

__all__ = ["register"]

_ACTIONS = ("merge", "select", "delete", "reorder", "rotate", "crop",
            "stamp", "compress", "repair", "protect", "unlock",
            "archive", "scan", "sign", "info",
            # PdfCraft port (documents/pdfcraft/): object-model reads and
            # incremental edits ride the same tool, not a second one.
            "inspect", "annotate", "metadata", "parity", "attribution")


def _parse_box(spec: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError("box needs four numbers: x0,y0,x1,y1")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"bad box {spec!r}") from None
    return (nums[0], nums[1], nums[2], nums[3])


def _parse_order(spec: str) -> list[int]:
    try:
        return [int(p.strip()) for p in spec.split(",") if p.strip()]
    except ValueError:
        raise ValueError(f"bad order {spec!r}; use 3,1,2") from None


def _build_op(action: str, args: dict[str, Any]) -> Any:
    if action == "merge":
        return MergeOp()
    if action == "select":
        return SelectOp(args["ranges"] or "all")
    if action == "delete":
        return DeleteOp(args["pages"] or args["ranges"])
    if action == "reorder":
        return ReorderOp(_parse_order(args["order"]))
    if action == "rotate":
        return RotateOp(args["pages"] or "all", int(args["degrees"]))
    if action == "crop":
        return CropOp(args["pages"] or "all", _parse_box(args["box"]))
    if action == "stamp":
        return StampOp(args["text"], args["pages"] or "all",
                       position=args["position"])
    if action == "compress":
        return CompressOp()
    if action == "repair":
        return RepairOp()
    if action == "protect":
        return ProtectOp(args["password"],
                         owner_password=args["owner_password"] or None)
    if action == "unlock":
        return UnlockOp(args["password"])
    if action == "archive":
        return ArchiveOp(level=args["level"])
    if action == "scan":
        return ScanOp(images=tuple(args["images"] or []))
    if action == "sign":
        images = args["images"] or []
        if not images:
            raise ValueError("sign needs an image (images=[path])")
        return SignOp(image=images[0], page=int(args["page"] or 1),
                      rect=_parse_box(args["rect"])
                      if args["rect"] else (400, 700, 120, 40))
    raise ValueError(f"unknown pagewise action {action!r}; "
                     f"use one of {', '.join(_ACTIONS)}")


def register(registry: Any) -> None:
    """Attach the pagewise family to a registry."""
    @registry.register(
        "pagewise",
        description=(
            "Page surgery over PDFs, local and uncapped — merge, split, "
            "extract/delete/reorder pages, rotate, crop, stamp page "
            "numbers or a watermark, compress, repair a damaged PDF, "
            "protect/unlock with a password, write the PDF/A archival "
            "shape, scan images into a PDF, or place a signature image. "
            "PdfCraft actions (object model + incremental edits): "
            "inspect=<read the catalog, pages, fonts, images, "
            "annotations, metadata>; annotate=<append a sticky-note "
            "annotation, text=<contents>, page=<n>, rect=<x0,y0,x1,y1>>; "
            "metadata=<set the title/author incrementally, title=<t>, "
            "author=<a>>; parity=<run the machine-checked parity TOML>; "
            "attribution=<verify the SHA-256 attribution DB>. "
            "action=<merge|select|delete|reorder|rotate|crop|stamp|"
            "compress|repair|protect|unlock|archive|scan|sign|info|"
            "inspect|annotate|metadata|parity|attribution>; "
            "sources=<list of PDF paths>; output=<path to write>; "
            "ranges=<1-based pages like '2-5,7'>; pages=<1-based pages>; "
            "order=<new order like '3,1,2'>; degrees=<multiple of 90>; "
            "box=<x0,y0,x1,y1 crop box>; text=<stamp text, {n} and "
            "{total} expand>; position=<bottom-center|bottom-right|"
            "bottom-left|top-center|top-right|top-left|center>; "
            "password=<password>; images=<list of image paths for scan "
            "or sign>; rect=<x,y,w,h for the signature>; level=<PDF/A "
            "level, default 1b>."
        ),
        capability=Capability.FS_WRITE,
    )
    def pagewise_tool(action: str = "info",
                      sources: list[str] | None = None,
                      output: str = "", ranges: str = "",
                      pages: str = "", order: str = "",
                      degrees: int = 0, box: str = "",
                      text: str = "", position: str = "bottom-center",
                      password: str = "", owner_password: str = "",
                      level: str = "1b", images: list[str] | None = None,
                      rect: str = "", title: str = "",
                      page: int = 1, author: str = "") -> dict[str, Any]:
        action = (action or "info").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown pagewise action {action!r}; use "
                             f"one of {', '.join(_ACTIONS)}")
        sources = list(sources or [])
        if action == "inspect":
            if not sources:
                raise ValueError("pagewise inspect needs a source PDF")
            summary = PdfDocument.from_bytes(
                Path(sources[0]).read_bytes()).to_dict()
            summary["reply"] = (
                f"{sources[0]}: PDF {summary['version']}, "
                f"{summary['page_count']} page(s), "
                f"{len(summary['fonts'])} font(s), "
                f"{len(summary['images'])} image(s), "
                f"{len(summary['annotations'])} annotation(s)"
                f"{', repaired xref' if summary['repaired'] else ''}.")
            return summary
        if action == "parity":
            result = run_pdfcraft_parity()
            failed = [c["id"] for c in result["cases"]
                      if c["status"] == "fail"]
            result["reply"] = (
                f"pdfcraft parity: {result['passed']} passed, "
                f"{result['failed']} failed"
                + (f" ({', '.join(failed)})" if failed else "") + ".")
            return result
        if action == "attribution":
            result = verify_pdfcraft_attribution()
            result["reply"] = (
                f"pdfcraft attribution: {result['checked']} verified, "
                f"{result['failed']} failed, "
                f"{result['skipped']} skipped (clone not on disk).")
            return result
        if action in ("annotate", "metadata"):
            if not sources:
                raise ValueError(
                    f"pagewise {action} needs a source PDF")
            if not output:
                raise ValueError(
                    f"pagewise {action} needs output=<path to write>")
            original = Path(sources[0]).read_bytes()
            editor = PdfEditor.from_bytes(original)
            if action == "annotate":
                if not text:
                    raise ValueError(
                        "pagewise annotate needs text=<contents>")
                box = _parse_box(rect) if rect else (72, 700, 200, 720)
                editor.add_text_annotation(page, box, text)
            else:
                fields: dict[str, str] = {}
                if title:
                    fields["title"] = title
                if author:
                    fields["author"] = author
                if not fields:
                    raise ValueError(
                        "pagewise metadata needs title=<t> and/or "
                        "author=<a>")
                editor.set_info(**fields)
            saved = editor.save_incremental()
            out_path = Path(output).expanduser()
            out_path.write_bytes(saved)
            row: dict[str, Any] = {
                "output": str(out_path),
                "pages": editor.document.page_count,
                "size_before": len(original),
                "size_after": len(saved),
                "incremental": True,
                "prefix_preserved": saved[:len(original)] == original,
            }
            if action == "annotate":
                row["annotations"] = len(editor.document.annotations())
                row["reply"] = (
                    f"annotated page {page} of {sources[0]} "
                    f"(incremental, {len(saved) - len(original)} bytes "
                    f"appended).")
            else:
                row["title"] = editor.document.info.get("Title", "")
                row["author"] = editor.document.info.get("Author", "")
                row["reply"] = (
                    f"metadata set on {sources[0]} (incremental, "
                    f"{len(saved) - len(original)} bytes appended).")
            return row
        if action == "info":
            if not sources:
                raise ValueError("pagewise info needs a source PDF")
            info = pdf_info(Path(sources[0]).read_bytes())
            info["reply"] = (
                f"{sources[0]}: {info['pages']} page(s), "
                f"{'encrypted' if info['encrypted'] else 'not encrypted'}, "
                f"{info['size_bytes']} bytes.")
            return info
        args = {"ranges": ranges, "pages": pages, "order": order,
                "degrees": degrees, "box": box, "text": text,
                "position": position, "password": password,
                "owner_password": owner_password, "level": level,
                "images": images, "rect": rect, "page": page}
        op = _build_op(action, args)
        if action == "scan":
            job_sources: list[Any] = []
        else:
            if not sources:
                raise ValueError(f"pagewise {action} needs a source PDF")
            job_sources = sources
        result = run_pagewise(PagewiseJob(sources=job_sources, ops=[op],
                                          title=title or None))
        if not output:
            raise ValueError("pagewise needs output=<path to write>")
        out_path = Path(output).expanduser()
        out_path.write_bytes(result.data)
        row: dict[str, Any] = {
            "output": str(out_path),
            "pages": result.pages,
            "size_before": result.size_before,
            "size_after": result.size_after,
            "measurements": result.measurements,
            "labels": result.labels,
        }
        row["reply"] = format_pagewise_reply(result)
        return row
