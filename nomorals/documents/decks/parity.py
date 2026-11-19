"""Parity runner — the machine-checked parity TOML.

Port of the *design* of the wordcraft parity runner (lane 100): a
TOML file listing real fixture .pptx files, the operations applied
to each, and the expected outcomes — and a runner that EXECUTES
them against the real engine and reports pass/fail per case with
measured values. Parity is measured, never asserted: a case whose
expectation is wrong fails loudly, a missing fixture fails (never
a silent skip), and an operation that raises unexpectedly fails
with the exception named.

Schema (version 1):

    schema_version = 1

    [[cases]]
    id = "simple-inspect"
    fixture = "simple.pptx"
    ops = ["inspect"]
    [cases.expect]
    slides = 2
    title = "Simple Fixture"

Ops: inspect, replace, restyle-run, add-slide, add-table,
add-image, remove-slide, reorder, set-notes, set-metadata,
set-transition, open-refused. Expect keys: slides, shapes
(total), tables, images, charts, title, author, first_slide_text,
text_contains (list — every entry must appear in the resulting
text), parts_preserved (bool — every part except the ones the op
may touch stays byte-identical), untouched_slides_changed (int),
transition, notes, error_kind.
"""

from __future__ import annotations

import io
import tomllib
from pathlib import Path
from typing import Any

from PIL import Image

from ..errors import DocumentError
from .edit import DeckEditor
from .model import DeckDocument

__all__ = ["DEFAULT_PARITY_PATH", "load_parity", "run_parity"]

DEFAULT_PARITY_PATH = Path(__file__).resolve().parent.parent \
    / "deckcraft-parity.toml"

_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent \
    / "tests" / "fixtures" / "deckcraft"

#: Parts an edit is allowed to touch; everything else must survive
#: byte-identical for ``parts_preserved`` to hold.
_EDITABLE_PARTS = frozenset({
    "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels",
    "ppt/slides/slide1.xml", "ppt/slides/slide2.xml",
    "ppt/slides/slide3.xml", "ppt/slides/slide4.xml",
    "ppt/slides/_rels/slide1.xml.rels",
    "ppt/slides/_rels/slide2.xml.rels",
    "ppt/slides/_rels/slide3.xml.rels",
    "ppt/slides/_rels/slide4.xml.rels",
    "ppt/notesSlides/notesSlide1.xml",
    "ppt/notesSlides/notesSlide2.xml",
    "docProps/core.xml", "[Content_Types].xml",
})


def load_parity(path: str | Path | None = None) -> dict[str, Any]:
    """Load a parity TOML file (stdlib tomllib)."""
    path = Path(path) if path is not None else DEFAULT_PARITY_PATH
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise DocumentError(f"cannot read parity TOML {path}: {exc}",
                            kind="io") from exc
    except tomllib.TOMLDecodeError as exc:
        raise DocumentError(f"bad parity TOML {path}: {exc}",
                            kind="parse") from exc
    if data.get("schema_version") != 1:
        raise DocumentError(
            f"parity TOML {path} is not schema version 1",
            kind="unsupported_format")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise DocumentError(f"parity TOML {path} has no [[case]] rows",
                            kind="parse")
    return data


def _sample_png() -> bytes:
    img = Image.new("RGB", (40, 30), (10, 200, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _apply_op(op: str, editor: DeckEditor,
              args: dict[str, Any]) -> None:
    if op == "inspect":
        return
    if op == "replace":
        editor.replace_text(str(args["old"]), str(args["new"]))
    elif op == "restyle-run":
        editor.set_run_format(
            int(args["slide"]), int(args["shape"]), int(args["para"]),
            int(args["run"]),
            bold=args.get("bold"), italic=args.get("italic"),
            color=args.get("color"),
            size_half_points=args.get("size_half_points"))
    elif op == "add-slide":
        editor.add_slide(layout=str(args.get("layout", "content")),
                         title=args.get("title"))
    elif op == "add-table":
        editor.add_table(int(args.get("slide", 0)),
                         list(args["headers"]),
                         [list(r) for r in args["rows"]])
    elif op == "add-image":
        editor.add_image(int(args.get("slide", 0)), _sample_png())
    elif op == "remove-slide":
        editor.remove_slide(int(args["slide"]))
    elif op == "reorder":
        editor.move_slide(int(args["from"]), int(args["to"]))
    elif op == "set-notes":
        editor.set_notes(int(args["slide"]), str(args["text"]))
    elif op == "set-metadata":
        editor.set_metadata(**{k: str(v)
                               for k, v in args.items()})
    elif op == "set-transition":
        editor.set_transition(int(args["slide"]), str(args["kind"]))
    elif op == "open-refused":
        return
    else:
        raise DocumentError(f"unknown parity op {op!r}",
                            kind="invalid_argument")


def _run_case(case: dict[str, Any], fixtures_dir: Path
              ) -> dict[str, Any]:
    """Execute one [[case]]; returns {id, status, measured, expected,
    detail}. A failure is a reported row, never a raised exception."""
    case_id = str(case.get("id", "?"))
    expected = case.get("expect", {})
    measured: dict[str, Any] = {}
    try:
        fixture = case.get("fixture", "")
        fixture_path = fixtures_dir / str(fixture)
        if not fixture_path.is_file():
            return {"id": case_id, "status": "fail",
                    "measured": {"fixture_missing": str(fixture)},
                    "expected": expected,
                    "detail": f"fixture {fixture!r} is not on disk — "
                              f"a missing fixture is a failure, never "
                              f"a silent skip"}
        original = fixture_path.read_bytes()
        ops = case.get("ops", ["inspect"])
        args = case.get("op_args", {})
        if "open-refused" in ops:
            try:
                DeckDocument.from_bytes(original)
            except DocumentError as exc:
                measured["error_kind"] = exc.kind
                if expected.get("error_kind") == exc.kind:
                    return {"id": case_id, "status": "pass",
                            "measured": measured, "expected": expected,
                            "detail": "refused honestly with kind "
                                      f"{exc.kind!r}"}
                return {"id": case_id, "status": "fail",
                        "measured": measured, "expected": expected,
                        "detail": f"refused with kind {exc.kind!r}, "
                                  f"expected {expected.get('error_kind')!r}"}
            return {"id": case_id, "status": "fail",
                    "measured": measured, "expected": expected,
                    "detail": "expected a refusal, the file opened"}
        editor = DeckEditor.from_bytes(original)
        for op in ops:
            _apply_op(op, editor, args)
        saved = editor.save()
        doc = DeckDocument.from_bytes(saved)
        measured["slides"] = len(doc.slides)
        measured["shapes"] = sum(len(s.shapes) for s in doc.slides)
        measured["tables"] = sum(
            1 for s in doc.slides for sh in s.shapes
            if sh.kind == "table")
        measured["images"] = len(doc.images)
        measured["charts"] = sum(
            1 for s in doc.slides for sh in s.shapes
            if sh.kind == "chart")
        measured["title"] = doc.core_properties.get("title", "")
        measured["author"] = doc.core_properties.get("author", "")
        measured["first_slide_text"] = doc.slides[0].text \
            if doc.slides else ""
        measured["transition"] = doc.slides[0].transition \
            if doc.slides else None
        measured["notes"] = doc.slides[0].notes \
            if doc.slides else ""
        text = doc.plain_text()
        for want in expected.get("text_contains", []):
            measured.setdefault("text_contains", {})[want] = want in text
        if expected.get("parts_preserved") is not None:
            report = editor.document.fidelity_report(original)
            changed = set(report["parts_changed"])
            measured["parts_preserved"] = (
                changed <= _EDITABLE_PARTS
                and report["untouched_slides_changed"] == 0)
            measured["parts_changed"] = sorted(changed)
        if "untouched_slides_changed" in expected:
            report = editor.document.fidelity_report(original)
            measured["untouched_slides_changed"] = report[
                "untouched_slides_changed"]
        if "error_kind" in expected:
            measured["error_kind"] = None
    except DocumentError as exc:
        measured["error_kind"] = exc.kind
        if expected.get("error_kind") == exc.kind:
            return {"id": case_id, "status": "pass",
                    "measured": measured, "expected": expected,
                    "detail": f"refused honestly with kind {exc.kind!r}"}
        return {"id": case_id, "status": "fail",
                "measured": measured, "expected": expected,
                "detail": f"raised DocumentError kind {exc.kind!r}: "
                          f"{exc}"}
    except Exception as exc:  # noqa: BLE001 - a case must report, not raise
        return {"id": case_id, "status": "fail",
                "measured": measured, "expected": expected,
                "detail": f"raised {type(exc).__name__}: {exc}"}
    failures: list[str] = []
    for key, want in expected.items():
        if key == "text_contains":
            for text_want, found in measured.get(
                    "text_contains", {}).items():
                if not found:
                    failures.append(
                        f"text {text_want!r} not in result")
        elif key == "first_slide_text":
            if want not in measured.get(key, ""):
                failures.append(
                    f"first_slide_text does not contain {want!r}")
        elif key in ("parts_preserved", "untouched_slides_changed",
                     "slides", "shapes", "tables", "images", "charts",
                     "title", "author", "transition", "notes",
                     "error_kind"):
            got = measured.get(key)
            if got != want:
                failures.append(f"{key}: measured {got!r}, "
                                f"expected {want!r}")
    if failures:
        return {"id": case_id, "status": "fail",
                "measured": measured, "expected": expected,
                "detail": "; ".join(failures)}
    return {"id": case_id, "status": "pass",
            "measured": measured, "expected": expected,
            "detail": "all expectations met"}


def run_parity(path: str | Path | None = None, *,
               fixtures_dir: str | Path | None = None
               ) -> dict[str, Any]:
    """Run every case in the parity TOML; returns {passed, failed,
    cases}. A missing fixture file fails its case — never a skip."""
    data = load_parity(path)
    base = Path(fixtures_dir) if fixtures_dir is not None \
        else _FIXTURES_DIR
    cases = [_run_case(case, base) for case in data["cases"]]
    passed = sum(1 for c in cases if c["status"] == "pass")
    return {"passed": passed, "failed": len(cases) - passed,
            "cases": cases}
