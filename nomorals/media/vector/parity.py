"""Parity runner — the machine-checked parity TOML.

Port of the *design* of the pdfcraft/wordcraft parity runners
(lanes 99/100): a TOML file listing real fixture SVGs, the
operations applied to each, and the expected outcomes — and a
runner that EXECUTES them against the real engine and reports
pass/fail per case with measured values. Parity is measured, never
asserted: a case whose expectation is wrong fails loudly, a
missing fixture fails (never a silent skip), and an operation
that raises unexpectedly fails with the exception named.

Schema (version 1):

    schema_version = 1

    [[cases]]
    id = "simple-inspect"
    fixture = "simple.svg"
    ops = ["inspect"]
    [cases.expect]
    nodes = 3
    title = "Simple Fixture"

Ops: inspect, add-rect, set-fill, transform, group,
boolean-union, boolean-difference, render, open-refused.
Expect keys: nodes, layers, title, gradients, text_contains
(list — every entry must appear in the resulting text),
untouched_nodes_changed (int), changed_ids (int), area (float,
±1.0), render_png (bool), error_kind.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .edit import VectorEditor
from .errors import VectorError
from .geom import Affine
from .model import VectorDocument
from .pathops import polygon_area
from .render import render_document

__all__ = ["DEFAULT_PARITY_PATH", "load_parity", "run_parity"]

DEFAULT_PARITY_PATH = Path(__file__).resolve().parent.parent \
    / "vector-parity.toml"

_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent \
    / "tests" / "fixtures" / "vectorcraft"


def load_parity(path: str | Path | None = None) -> dict[str, Any]:
    """Load a parity TOML file (stdlib tomllib)."""
    path = Path(path) if path is not None else DEFAULT_PARITY_PATH
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise VectorError(f"cannot read parity TOML {path}: {exc}",
                          kind="io") from exc
    except tomllib.TOMLDecodeError as exc:
        raise VectorError(f"bad parity TOML {path}: {exc}",
                          kind="parse") from exc
    if data.get("schema_version") != 1:
        raise VectorError(
            f"parity TOML {path} schema_version is not 1",
            kind="parse")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise VectorError(
            f"parity TOML {path} has no [[cases]] rows",
            kind="parse")
    return data


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("id", "?"))
    fixture = str(case.get("fixture", ""))
    path = _FIXTURES_DIR / fixture
    if not path.exists():
        return {"id": case_id, "status": "fail",
                "detail": f"fixture missing: {fixture}"}
    original = path.read_bytes()
    try:
        before = VectorDocument.from_bytes(original)
    except VectorError as exc:
        if "open-refused" in case.get("ops", []):
            expect = case.get("expect", {})
            if expect.get("error_kind") in (None, exc.kind):
                return {"id": case_id, "status": "pass",
                        "measured": {"error_kind": exc.kind}}
            return {"id": case_id, "status": "fail",
                    "detail": f"error kind {exc.kind}, expected "
                              f"{expect.get('error_kind')}"}
        return {"id": case_id, "status": "fail",
                "detail": f"open failed: {exc} (kind {exc.kind})"}
    editor = VectorEditor.from_bytes(original)
    params = case.get("params", {})
    measured: dict[str, Any] = {}
    try:
        for op in case.get("ops", []):
            _apply_op(editor, op, params)
    except VectorError as exc:
        return {"id": case_id, "status": "fail",
                "detail": f"op failed: {exc} (kind {exc.kind})"}
    except (ValueError, KeyError, IndexError) as exc:
        return {"id": case_id, "status": "fail",
                "detail": f"op failed: {exc}"}
    after = VectorDocument.from_bytes(editor.save())
    report = before.fidelity_report(after, touched=editor.touched)
    measured["nodes"] = len(after.all_nodes())
    measured["layers"] = len(after.layers)
    measured["title"] = after.title
    measured["gradients"] = len(after.defs.gradients)
    measured["text_contains"] = after.plain_text()
    measured["untouched_nodes_changed"] = report["untouched_changed"]
    measured["changed_ids"] = report["changed_ids"]
    if "render" in case.get("ops", []):
        png = render_document(after)
        measured["render_png"] = png.startswith(b"\x89PNG") \
            and len(png) > 100
    if any(op.startswith("boolean-") for op in case.get("ops", [])):
        paths = [n for n in after.all_nodes() if n.kind == "path"]
        measured["area"] = sum(
            abs(polygon_area(n.path)) for n in paths)
    expect = case.get("expect", {})
    for key, wanted in expect.items():
        if key == "text_contains":
            for entry in wanted:
                if entry not in measured["text_contains"]:
                    return {"id": case_id, "status": "fail",
                            "detail": f"text {entry!r} not in "
                                      "document text",
                            "measured": measured}
            continue
        if key == "area":
            if abs(measured.get("area", 0.0) - float(wanted)) > 1.0:
                return {"id": case_id, "status": "fail",
                        "detail": f"area {measured.get('area')} != "
                                  f"{wanted}", "measured": measured}
            continue
        if measured.get(key) != wanted:
            return {"id": case_id, "status": "fail",
                    "detail": f"{key} {measured.get(key)!r} != "
                              f"{wanted!r}", "measured": measured}
    return {"id": case_id, "status": "pass", "measured": measured}


def _apply_op(editor: VectorEditor, op: str,
              params: dict[str, Any]) -> None:
    doc = editor.document
    if op == "inspect":
        return
    if op == "render":
        return
    if op == "add-rect":
        rect = params.get("rect", [10, 10, 50, 40, "#123456"])
        editor.add_rect(float(rect[0]), float(rect[1]),
                        float(rect[2]), float(rect[3]),
                        fill=str(rect[4]) if len(rect) > 4
                        else "#000000")
        return
    if op == "set-fill":
        node_id = str(params.get("node", ""))
        if not node_id:
            node_id = doc.all_nodes()[0].node_id
        editor.set_fill(node_id, str(params.get("fill", "#123456")))
        return
    if op == "transform":
        node_id = str(params.get("node", ""))
        if not node_id:
            node_id = doc.all_nodes()[0].node_id
        editor.transform(
            node_id,
            Affine.translate(float(params.get("dx", 0.0)),
                             float(params.get("dy", 0.0))))
        return
    if op == "group":
        ids = [str(n) for n in params.get("nodes", [])]
        if len(ids) < 2:
            ids = [n.node_id for n in doc.all_nodes()[:2]]
        editor.group(ids, name=str(params.get("name", "Group")))
        return
    if op in ("boolean-union", "boolean-difference"):
        kind = op.split("-", 1)[1]
        a = str(params.get("a", ""))
        b = str(params.get("b", ""))
        editor.apply_boolean(kind, a, b)
        return
    if op == "open-refused":
        return
    raise VectorError(f"unknown parity op {op!r}",
                      kind="invalid_argument")


def run_parity(path: str | Path | None = None) -> dict[str, Any]:
    """Run every parity case; returns {passed, failed, cases}."""
    data = load_parity(path)
    cases = [_run_case(case) for case in data["cases"]]
    passed = sum(1 for c in cases if c["status"] == "pass")
    return {"passed": passed,
            "failed": len(cases) - passed,
            "cases": cases}
