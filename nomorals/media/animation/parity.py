"""Machine-checked parity for the animation organ.

A parity TOML lists real fixtures, the operations to run on them
and the expected outcomes. ``run_parity`` executes every case and
fails on any mismatch; a missing fixture FAILS its case, never a
silent skip. The TOML lives at ``nomorals/media/animation-parity.toml``
and fixtures at ``tests/fixtures/anicraft/``.
"""

from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path

from .easing import Easing
from .errors import AnimationError
from .eval import evaluate
from .model import AnimationDocument
from .publish import publish
from .xfl import import_xfl

TOML_PATH = Path(__file__).resolve().parent.parent / "animation-parity.toml"
FIXTURES = (Path(__file__).resolve().parent.parent.parent.parent
            / "tests" / "fixtures" / "anicraft")


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) <= tol


def _run_case(case: dict, fixtures: Path) -> dict:
    case_id = case.get("id", "?")
    result: dict = {"id": case_id, "ok": True, "details": {}}
    ops = case.get("ops", [])
    expect = case.get("expect", {})
    doc: AnimationDocument | None = None
    fixture_name = case.get("fixture")
    if fixture_name:
        fixture = fixtures / fixture_name
        if not fixture.exists():
            return {"id": case_id, "ok": False,
                    "details": {"missing_fixture": fixture_name}}
        if fixture.is_dir():
            doc = import_xfl(fixture)
        else:
            doc = AnimationDocument.parse(fixture.read_bytes())
    for op in ops:
        if op == "inspect":
            if doc is None:
                raise AnimationError("inspect needs a fixture")
            scene = doc.scenes[0]
            result["details"]["layers"] = len(scene.timeline.layers)
            result["details"]["keyframes"] = sum(
                len(ly.keyframes) for ly in scene.timeline.layers)
            result["details"]["elements"] = sum(
                len(kf.elements) for ly in scene.timeline.layers
                for kf in ly.keyframes)
        elif op == "roundtrip":
            if doc is None or fixture_name is None:
                raise AnimationError("roundtrip needs a fixture")
            raw = (fixtures / fixture_name).read_bytes()
            result["details"]["byte_identical"] = doc.to_bytes() == raw
        elif op == "easing":
            name = case.get("easing", "linear")
            t = float(case.get("t", 0.0))
            result["details"]["value"] = Easing.named(name).apply(t)
        elif op == "classic":
            strength = float(case.get("strength", 0.0))
            t = float(case.get("t", 0.0))
            result["details"]["value"] = Easing.classic(strength).apply(t)
        elif op == "eval":
            if doc is None:
                raise AnimationError("eval needs a fixture")
            frame = int(case.get("frame", 0))
            dl = evaluate(doc, frame)
            if not dl.items:
                return {"id": case_id, "ok": False,
                        "details": {"no_items": True}}
            item = dl.items[0]
            result["details"]["x"] = item.transform.x
            result["details"]["y"] = item.transform.y
            result["details"]["scale_x"] = item.transform.scale_x
            result["details"]["scale_y"] = item.transform.scale_y
            result["details"]["rotation"] = item.transform.rotation
            result["details"]["items"] = len(dl.items)
        elif op == "publish":
            if doc is None:
                raise AnimationError("publish needs a fixture")
            fmt = case.get("format", "gif")
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / f"out.{fmt}"
                report = publish(doc, fmt, out)
                result["details"]["frame_count"] = report["frame_count"]
                result["details"]["exists"] = out.is_file()
        elif op == "xfl":
            if doc is None:
                raise AnimationError("xfl needs a fixture")
            scene = doc.scenes[0]
            result["details"]["layers"] = len(scene.timeline.layers)
            result["details"]["keyframes_layer0"] = len(
                scene.timeline.layers[0].keyframes) \
                if scene.timeline.layers else 0
            result["details"]["first_element_kind"] = (
                scene.timeline.layers[0].keyframes[0].elements[0].kind
                if scene.timeline.layers
                and scene.timeline.layers[0].keyframes
                and scene.timeline.layers[0].keyframes[0].elements
                else "")
        elif op == "undo":
            if doc is None:
                raise AnimationError("undo needs a fixture")
            from .edit import AnimationEditor

            ed = AnimationEditor(doc)
            layer = doc.scenes[0].timeline.layers[0]
            before = len(layer.keyframes)
            ed.add_keyframe(layer.id, 48, 12)
            ed.undo()
            result["details"]["keyframes_after_undo"] = len(
                doc.scenes[0].timeline.layers[0].keyframes)
            result["details"]["keyframes_before"] = before
        else:
            return {"id": case_id, "ok": False,
                    "details": {"unknown_op": op}}
    for key, want in expect.items():
        got = result["details"].get(key)
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            ok = _close(got, want)
        else:
            ok = got == want
        if not ok:
            result["ok"] = False
            result["details"][f"mismatch_{key}"] = {"got": got,
                                                    "want": want}
    return result


def run_parity(toml_path: Path | str | None = None,
               fixtures: Path | str | None = None) -> dict:
    """Run every parity case; returns passed/failed/total/results."""
    data = _load_toml(Path(toml_path) if toml_path else TOML_PATH)
    fix = Path(fixtures) if fixtures else FIXTURES
    results = []
    for case in data.get("cases", []):
        try:
            results.append(_run_case(case, fix))
        except AnimationError as exc:
            results.append({"id": case.get("id", "?"), "ok": False,
                            "details": {"error": str(exc)}})
    failed = [r["id"] for r in results if not r["ok"]]
    return {"total": len(results), "passed": len(results) - len(failed),
            "failed": failed, "results": results}
