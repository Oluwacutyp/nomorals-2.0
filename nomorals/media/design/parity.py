"""FigCraft parity — machine-checked design parity (lane 105).

Reads ``nomorals/media/design-parity.toml``: each case names a
real fixture and an operation with expected outcomes. Missing
fixture = fail, never skip. ``run_parity`` returns
``{"passed": <count>, "failed": [<case ids>],
"passed_ids": [...], "failures": [{"id", "reason"}]}``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .components import resolve_instance
from .errors import DesignError
from .layout import compute_layout
from .model import DesignDocument
from .proto import flows
from .render import render_frame
from .session import SessionClient, SessionServer

__all__ = ["run_parity"]

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "media" / "design-parity.toml"


def _fixture_path(name: str) -> Path:
    return (Path(__file__).resolve().parents[3] / "tests" / "fixtures"
            / "figcraft" / name)


def _load(name: str) -> DesignDocument:
    path = _fixture_path(name)
    if not path.exists():
        raise DesignError(f"missing fixture {name}", kind="not_found")
    return DesignDocument.from_bytes(path.read_bytes())


def _run_case(case: dict[str, Any]) -> None:
    operation = case.get("operation")
    if operation is None:
        ops = case.get("ops") or []
        operation = ops[0] if ops else None
    fixture = case.get("fixture", "")
    expect = case.get("expect", {})
    if operation == "inspect":
        doc = _load(fixture)
        if "nodes" in expect and len(doc.nodes) != int(expect["nodes"]):
            raise AssertionError(
                f"nodes: got {len(doc.nodes)}, want {expect['nodes']}")
        return
    if operation == "roundtrip":
        doc = _load(fixture)
        if doc.to_bytes() != _fixture_path(fixture).read_bytes():
            raise AssertionError("round trip bytes differ")
    elif operation == "layout":
        doc = _load(fixture)
        layout = compute_layout(doc, int(case["root"]))
        for nid, wanted in expect.items():
            got = layout.get(int(nid))
            if got is None:
                raise AssertionError(f"node {nid} missing from layout")
            for key, value in wanted.items():
                if abs(float(got[key]) - float(value)) > 0.01:
                    raise AssertionError(
                        f"node {nid} {key}: got {got[key]}, want {value}")
    elif operation == "instance":
        doc = _load(fixture)
        resolved = resolve_instance(doc, int(case["node"]))
        if resolved.get("exact") != expect.get("exact"):
            raise AssertionError(f"exact: got {resolved.get('exact')}")
        tree = resolved["tree"]
        root_name = tree[0].get("name") if tree else None
        if root_name != expect.get("root_name"):
            raise AssertionError(f"root name: got {root_name}")
    elif operation == "flows":
        doc = _load(fixture)
        got = flows(doc, int(case["start"]))
        if got != [int(v) for v in expect.get("flows", [])]:
            raise AssertionError(f"flows: got {got}, want {expect.get('flows')}")
    elif operation == "fidelity":
        raw = _fixture_path(fixture).read_bytes()
        doc = DesignDocument.from_bytes(raw)
        edit = case.get("edit", {})
        if edit.get("kind") == "set_text":
            doc.node(int(edit["node"]))["characters"] = str(edit["value"])
        elif edit.get("kind") == "set_fill":
            node = doc.node(int(edit["node"]))
            node["fills"] = [{"type": "solid",
                              "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
                              "opacity": 1.0, "visible": True, "blend": "normal"}]
        report = doc.fidelity_report(raw, touched={int(edit["node"])})
        if len(report["changed_ids"]) != int(expect.get("changed_count", 0)):
            raise AssertionError(
                f"changed nodes: got {report['changed_ids']}, "
                f"want {expect.get('changed_count')}")
    elif operation == "two_editors":
        doc = _load(fixture)
        server = SessionServer(doc)
        alice = SessionClient(server, "alice")
        bob = SessionClient(server, "bob")
        alice.connect()
        bob.connect()
        edit_a = case.get("edit_a", {})
        edit_b = case.get("edit_b", {})
        alice.set_text(int(edit_a["node"]), str(edit_a["value"]))
        bob.set_fill(int(edit_b["node"]), str(edit_b["value"]))
        alice.receive()
        bob.receive()
        alice.receive()
        bob.receive()
        assert alice.doc is not None and bob.doc is not None
        if alice.doc.node(int(edit_a["node"])).get("characters") != edit_a["value"]:
            raise AssertionError("alice lost her own edit")
        if bob.doc.node(int(edit_a["node"])).get("characters") != edit_a["value"]:
            raise AssertionError("bob never got alice's edit")
        fill = bob.doc.node(int(edit_b["node"])).get("fills") or [{}]
        if float(fill[0].get("color", {}).get("r", 0.0)) != 1.0:
            raise AssertionError("alice never got bob's edit")
        if alice.doc.node(int(edit_b["node"])).get("fills") != bob.doc.node(int(edit_b["node"])).get("fills"):
            raise AssertionError("editors diverged")
    elif operation == "render":
        doc = _load(fixture)
        _, info = render_frame(doc, int(case["node"]))
        if info["width"] != int(expect.get("width", 0)):
            raise AssertionError(f"width: got {info['width']}")
        if info["height"] != int(expect.get("height", 0)):
            raise AssertionError(f"height: got {info['height']}")
        if "unrendered" in expect and info["unrendered"] != expect["unrendered"]:
            raise AssertionError(f"unrendered: got {info['unrendered']}")
        needle = expect.get("unrendered_contains")
        if needle and not any(needle in u for u in info["unrendered"]):
            raise AssertionError(f"unrendered lacks {needle!r}: "
                                 f"got {info['unrendered']}")
    elif operation == "open_fails":
        path = _fixture_path(fixture)
        if not path.exists():
            raise AssertionError(f"missing fixture {fixture}")
        try:
            DesignDocument.from_bytes(path.read_bytes())
        except DesignError:
            return
        raise AssertionError("broken fixture opened without error")
    else:
        raise AssertionError(f"unknown operation {operation!r}")


def run_parity(path: str | Path | None = None) -> dict[str, Any]:
    toml_path = Path(path) if path is not None else _DEFAULT_PATH
    if not toml_path.exists():
        return {"passed": 0, "failed": ["<toml>"], "passed_ids": [],
                "failures": [{"id": "<toml>",
                              "reason": f"missing parity file {toml_path}"}]}
    with toml_path.open("rb") as handle:
        data = tomllib.load(handle)
    passed: list[str] = []
    failed: list[dict[str, str]] = []
    for case in data.get("cases", []):
        case_id = str(case.get("id", "<unnamed>"))
        try:
            _run_case(case)
        except Exception as exc:  # noqa: BLE001 - parity must name the failure
            failed.append({"id": case_id, "reason": str(exc)})
        else:
            passed.append(case_id)
    return {"passed": len(passed), "failed": [f["id"] for f in failed],
            "passed_ids": passed, "failures": failed}
