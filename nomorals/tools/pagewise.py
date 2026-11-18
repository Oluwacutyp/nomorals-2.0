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

__all__ = ["register"]

_ACTIONS = ("merge", "select", "delete", "reorder", "rotate", "crop",
            "stamp", "compress", "repair", "protect", "unlock",
            "archive", "scan", "sign", "info")


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
            "action=<merge|select|delete|reorder|rotate|crop|stamp|"
            "compress|repair|protect|unlock|archive|scan|sign|info>; "
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
                      page: int = 1) -> dict[str, Any]:
        action = (action or "info").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown pagewise action {action!r}; use "
                             f"one of {', '.join(_ACTIONS)}")
        sources = list(sources or [])
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
