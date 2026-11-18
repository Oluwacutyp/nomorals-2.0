"""``nm doc`` — universal document engine: parse, convert, search documents."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _cmd_doc(args: Any, context: Any) -> int:
    """Route ``nm doc <parse|convert|search|show|diff|summarize|ocr|pagewise>``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm doc parse <file> [--json]\n"
              "       nm doc convert <file> --to md|html|txt|pdf|csv|xlsx|pptx [--out PATH]\n"
              "       nm doc search <query> --dir DIR [--limit N] [--json]\n"
              "       nm doc show <file>\n"
              "       nm doc diff <file-a> <file-b> [--json]\n"
              "       nm doc summarize <file> [--sentences N] [--json]\n"
              "       nm doc ocr <scanned.pdf> [--lang eng] [--dpi 200] [--json]\n"
              "       nm doc pagewise <merge|select|delete|reorder|rotate|crop|stamp|compress|repair|protect|unlock|archive|scan|sign|info> <files...> --out PATH [--ranges R] [--pages P] [--order 3,1,2] [--degrees 90] [--box x0,y0,x1,y1] [--text T] [--position POS] [--password PW] [--level 1b]",
              file=sys.stderr)
        return 2
    verb = words[0]
    if verb == "parse":
        return _doc_parse(args, words[1:])
    if verb == "convert":
        return _doc_convert(args, words[1:])
    if verb == "search":
        return _doc_search(args, words[1:])
    if verb == "show":
        return _doc_show(args, words[1:])
    if verb == "diff":
        return _doc_diff(args, words[1:])
    if verb == "summarize":
        return _doc_summarize(args, words[1:])
    if verb == "ocr":
        return _doc_ocr(args, words[1:])
    if verb == "pagewise":
        return _doc_pagewise(args, words[1:])
    print(f"unknown doc verb: {verb}", file=sys.stderr)
    return 2


def _doc_parse(args: Any, rest: list[str]) -> int:
    from ...documents.expose import read_document

    if not rest:
        print("usage: nm doc parse <file> [--json]", file=sys.stderr)
        return 2
    try:
        doc = read_document(path=rest[0])
    except Exception as exc:  # noqa: BLE001 - fail fast with the real error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(doc.to_dict(), indent=2, default=str))
        return 0
    print(f"format:  {doc.format}")
    print(f"title:   {doc.title or '(none)'}")
    print(f"source:  {doc.source}")
    print(f"sections: {len(doc.sections)}   tables: {len(doc.tables)}")
    for sec in doc.sections[:20]:
        head = f"  h{sec.level} {sec.heading}" if sec.heading else "  (body)"
        preview = sec.text.strip().replace("\n", " ")[:80]
        print(f"{head} — {preview}")
    for i, tbl in enumerate(doc.tables[:10]):
        print(f"  table[{i}] {tbl.name}: {len(tbl.headers)} cols x {len(tbl.rows)} rows")
    return 0


def _doc_convert(args: Any, rest: list[str]) -> int:
    from ...documents.expose import convert_document, read_document

    if not rest:
        print("usage: nm doc convert <file> --to "
              "md|html|txt|pdf|csv|xlsx|pptx [--out PATH]",
              file=sys.stderr)
        return 2
    target = (getattr(args, "to", "") or "").lower()
    # CLI spellings → the facade's convert targets. The facade owns the
    # dispatch table; the CLI only maps its short names onto it.
    targets = {"md": "markdown", "html": "html", "txt": "text",
               "pdf": "pdf", "csv": "csv", "xlsx": "xlsx", "pptx": "pptx"}
    facade_target = targets.get(target)
    if facade_target is None:
        print(f"unknown target format: {target or '(none)'}", file=sys.stderr)
        return 2
    try:
        doc = read_document(path=rest[0])
        out = convert_document(doc, facade_target)
    except Exception as exc:  # noqa: BLE001 - fail fast with the real error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    out_path = getattr(args, "out", "") or ""
    if isinstance(out, bytes) or out_path:
        if not out_path:
            print("error: --out PATH is required for binary output", file=sys.stderr)
            return 2
        data = out if isinstance(out, bytes) else out.encode("utf-8")
        Path(out_path).expanduser().write_bytes(data)
        print(f"wrote {len(data)} bytes -> {out_path}")
        return 0
    print(out if isinstance(out, str) else out.decode("utf-8", "replace"))
    return 0


def _doc_search(args: Any, rest: list[str]) -> int:
    from ...documents import DocumentIndex, parse_path

    if not rest:
        print("usage: nm doc search <query> --dir DIR [--limit N] [--json]",
              file=sys.stderr)
        return 2
    query = " ".join(rest)
    root = Path(getattr(args, "dir", "") or ".").expanduser()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2
    index = DocumentIndex()
    indexed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {
                ".pdf", ".docx", ".xlsx", ".pptx", ".html", ".htm",
                ".md", ".markdown", ".csv", ".tsv", ".txt",
                ".rtf", ".epub", ".odt", ".ods"}:
            continue
        try:
            index.add(parse_path(str(path)))
            indexed += 1
        except Exception:  # noqa: BLE001 - skip unreadable files, keep going
            continue
    limit = getattr(args, "limit", 10) or 10
    hits = index.search(query, limit=limit)
    if getattr(args, "json", False):
        print(json.dumps({"indexed": indexed, "hits": hits}, indent=2))
        return 0
    print(f"indexed {indexed} document(s) under {root}")
    for hit in hits:
        print(f"[{hit['score']:.2f}] {hit['title'] or hit['doc_id']} — {hit['snippet']}")
    if not hits:
        print("no matches")
    return 0


def _doc_show(args: Any, rest: list[str]) -> int:
    from ...documents import full_text
    from ...documents.expose import read_document

    if not rest:
        print("usage: nm doc show <file>", file=sys.stderr)
        return 2
    try:
        doc = read_document(path=rest[0])
    except Exception as exc:  # noqa: BLE001 - fail fast with the real error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(full_text(doc))
    return 0


def _doc_diff(args: Any, rest: list[str]) -> int:
    from ...documents.expose import compare_document_pair, read_document

    if len(rest) < 2:
        print("usage: nm doc diff <file-a> <file-b> [--json]", file=sys.stderr)
        return 2
    try:
        first = read_document(path=rest[0])
        second = read_document(path=rest[1])
        result = compare_document_pair(first, second)
    except Exception as exc:  # noqa: BLE001 - fail fast with the real error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0
    print(f"summary: {result.summary}")
    for label, items in (("sections added", result.sections_added),
                         ("sections removed", result.sections_removed),
                         ("sections changed", result.sections_changed),
                         ("tables added", result.tables_added),
                         ("tables removed", result.tables_removed),
                         ("tables changed", result.tables_changed)):
        for item in items:
            detail = result.stats.get("table_details", {}).get(item, "")
            print(f"  {label}: {item}" + (f" ({detail})" if detail else ""))
    if result.unified_diff:
        print("--- unified diff ---")
        print(result.unified_diff)
    return 0


def _doc_summarize(args: Any, rest: list[str]) -> int:
    from ...documents import keywords
    from ...documents.expose import read_document, summarize_document

    if not rest:
        print("usage: nm doc summarize <file> [--sentences N] [--json]",
              file=sys.stderr)
        return 2
    try:
        count = int(getattr(args, "sentences", 5) or 5)
    except (TypeError, ValueError):
        print("error: --sentences must be an integer", file=sys.stderr)
        return 2
    try:
        doc = read_document(path=rest[0])
        lines = summarize_document(doc, sentences=count)
        words = keywords("\n".join(lines), top_n=10)
    except Exception as exc:  # noqa: BLE001 - fail fast with the real error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps({"sentences": lines, "keywords": words}, indent=2))
        return 0
    for line in lines:
        print(f"- {line}")
    print(f"\nkeywords: {', '.join(words)}")
    return 0


def _doc_ocr(args: Any, rest: list[str]) -> int:
    from ...documents import ocr_pdf

    if not rest:
        print("usage: nm doc ocr <scanned.pdf> [--lang eng] [--dpi 200] [--json]",
              file=sys.stderr)
        return 2
    target = Path(rest[0]).expanduser()
    if not target.is_file():
        print(f"error: no such file: {target}", file=sys.stderr)
        return 1
    try:
        dpi = int(getattr(args, "dpi", 200) or 200)
    except (TypeError, ValueError):
        print("error: --dpi must be an integer", file=sys.stderr)
        return 2
    try:
        doc = ocr_pdf(target.read_bytes(),
                      lang=getattr(args, "lang", "eng") or "eng", dpi=dpi)
    except Exception as exc:  # noqa: BLE001 - fail fast with the real error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(doc.to_dict(), indent=2, default=str))
        return 0
    print(f"pages: {doc.metadata.get('pages')}")
    print(f"mean confidence: {doc.metadata.get('ocr_mean_confidence', 'n/a')}")
    for sec in doc.sections:
        preview = sec.text.strip().replace("\n", " ")[:100]
        print(f"{sec.heading} — {preview}")
    return 0


def _doc_pagewise(args: Any, rest: list[str]) -> int:
    from ...documents.pagewise import (
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

    def _box(spec: str) -> tuple[float, float, float, float]:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 4:
            raise ValueError("box needs four numbers: x0,y0,x1,y1")
        nums = [float(p) for p in parts]
        return (nums[0], nums[1], nums[2], nums[3])

    if not rest:
        print("usage: nm doc pagewise "
              "<merge|select|delete|reorder|rotate|crop|stamp|compress|"
              "repair|protect|unlock|archive|scan|sign|info> <files...> "
              "--out PATH [--ranges R] [--pages P] [--order 3,1,2] "
              "[--degrees 90] [--box x0,y0,x1,y1] [--text T] "
              "[--position POS] [--password PW] [--level 1b]",
              file=sys.stderr)
        return 2
    action = rest[0]
    files = rest[1:]
    out_path = getattr(args, "out", "") or ""
    try:
        if action == "info":
            if not files:
                print("usage: nm doc pagewise info <file>",
                      file=sys.stderr)
                return 2
            info = pdf_info(Path(files[0]).expanduser().read_bytes())
            if getattr(args, "json", False):
                print(json.dumps(info, indent=2, default=str))
                return 0
            print(f"pages:     {info['pages']}")
            print(f"encrypted: {info['encrypted']}")
            print(f"size:      {info['size_bytes']} bytes")
            if info.get("title"):
                print(f"title:     {info['title']}")
            if info["corruption"]:
                print(f"corruption: {', '.join(info['corruption'])}")
            return 0
        if action == "merge":
            ops = [MergeOp()]
            sources: list[Any] = files
        elif action == "select":
            ops = [SelectOp(getattr(args, "ranges", "") or "all")]
            sources = files[:1]
        elif action == "delete":
            ops = [DeleteOp(getattr(args, "pages", "")
                            or getattr(args, "ranges", ""))]
            sources = files[:1]
        elif action == "reorder":
            order = [int(p.strip()) for p in
                     (getattr(args, "order", "") or "").split(",")
                     if p.strip()]
            ops = [ReorderOp(order)]
            sources = files[:1]
        elif action == "rotate":
            ops = [RotateOp(getattr(args, "pages", "") or "all",
                            int(getattr(args, "degrees", 0) or 0))]
            sources = files[:1]
        elif action == "crop":
            ops = [CropOp(getattr(args, "pages", "") or "all",
                          _box(getattr(args, "box", "") or ""))]
            sources = files[:1]
        elif action == "stamp":
            ops = [StampOp(getattr(args, "text", "") or "",
                           getattr(args, "pages", "") or "all",
                           position=getattr(args, "position",
                                            "bottom-center")
                           or "bottom-center")]
            sources = files[:1]
        elif action == "compress":
            ops = [CompressOp()]
            sources = files[:1]
        elif action == "repair":
            ops = [RepairOp()]
            sources = files[:1]
        elif action == "protect":
            ops = [ProtectOp(getattr(args, "password", "") or "")]
            sources = files[:1]
        elif action == "unlock":
            ops = [UnlockOp(getattr(args, "password", "") or "")]
            sources = files[:1]
        elif action == "archive":
            ops = [ArchiveOp(level=getattr(args, "level", "1b")
                             or "1b")]
            sources = files[:1]
        elif action == "scan":
            ops = [ScanOp(images=tuple(files))]
            sources = []
        elif action == "sign":
            if len(files) < 2:
                print("usage: nm doc pagewise sign <pdf> <image> "
                      "--out PATH [--pages N] [--rect x,y,w,h]",
                      file=sys.stderr)
                return 2
            rect = _box(getattr(args, "rect", "") or "") \
                if getattr(args, "rect", "") else (400, 700, 120, 40)
            ops = [SignOp(image=files[1],
                          page=int(getattr(args, "pages", "") or 1),
                          rect=rect)]
            sources = files[:1]
        else:
            print(f"unknown pagewise action: {action}", file=sys.stderr)
            return 2
        if not sources and action != "scan":
            print(f"error: pagewise {action} needs a source PDF",
                  file=sys.stderr)
            return 2
        if not out_path:
            print("error: --out PATH is required (pagewise writes a "
                  "fresh PDF)", file=sys.stderr)
            return 2
        result = run_pagewise(PagewiseJob(sources=sources, ops=ops))
    except Exception as exc:  # noqa: BLE001 - fail fast with the real error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    Path(out_path).expanduser().write_bytes(result.data)
    print(format_pagewise_reply(result))
    print(f"wrote {len(result.data)} bytes -> {out_path}")
    return 0
