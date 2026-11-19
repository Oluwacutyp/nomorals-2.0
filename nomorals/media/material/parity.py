"""Machine-checked parity for the materials organ.

A parity TOML lists real fixtures, the operations to run on them
and the expected outcomes. ``run_parity`` executes every case and
fails on any mismatch; a missing fixture FAILS its case, never a
silent skip. The TOML lives at ``nomorals/media/material-parity.toml``
and fixtures at ``tests/fixtures/materialcraft/``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from . import library
from .errors import MaterialError, MaterialErrorKind
from .eval import CookCache, cook
from .model import MaterialDocument

TOML_PATH = Path(__file__).resolve().parent.parent / "material-parity.toml"
FIXTURES = (Path(__file__).resolve().parent.parent.parent.parent
            / "tests" / "fixtures" / "materialcraft")


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _run_case(case: dict, fixtures: Path, cache: CookCache) -> dict:
    case_id = case.get("id", "?")
    fixture_name = case.get("fixture")
    ops = case.get("ops", [])
    expect = case.get("expect", {})
    result: dict = {"id": case_id, "ok": True, "details": {}}
    if fixture_name:
        fixture = fixtures / fixture_name
        if not fixture.is_file():
            return {"id": case_id, "ok": False,
                    "details": {"missing_fixture": fixture_name}}
        raw = fixture.read_bytes()
        if fixture.suffix.lower() == ".mcraft":
            doc = MaterialDocument.parse(raw)
        else:
            doc = MaterialDocument.parse(raw)
    else:
        doc = MaterialDocument.blank("parity")
    graph = doc.graph
    cooked = None
    for op in ops:
        if op == "inspect":
            result["details"]["nodes"] = len(graph.nodes)
            result["details"]["connections"] = len(graph.connections)
            result["details"]["outputs"] = len(graph.outputs)
        elif op == "roundtrip":
            result["details"]["byte_identical"] = doc.to_bytes() == raw
        elif op == "cook":
            cooked = cook(graph, cache)
            result["details"]["channels"] = sorted(cooked.outputs.keys())
            result["details"]["errors"] = len(cooked.errors)
            result["details"]["nodes_cooked"] = cooked.nodes_cooked
        elif op == "cook_twice":
            first = cook(graph, cache)
            before_hits = cache.hits
            second = cook(graph, cache)
            result["details"]["cache_hits_second"] = cache.hits - before_hits
            result["details"]["byte_identical"] = (
                first.outputs.get("albedo") is not None
                and second.outputs.get("albedo") is not None
                and first.outputs["albedo"].data.tobytes()
                == second.outputs["albedo"].data.tobytes())
            cooked = second
        elif op == "cook_pixel":
            cooked = cook(graph, cache)
            albedo = cooked.outputs.get("albedo")
            if albedo is not None:
                px = albedo.data[0, 0]
                result["details"]["pixel_0_0"] = [round(float(v), 4)
                                                  for v in px[:3]]
                result["details"]["mean"] = [round(float(v), 4)
                                             for v in albedo.mean()[:3]]
        elif op == "cook_gray":
            cooked = cook(graph, cache)
            gray = cooked.outputs.get("height") or cooked.outputs.get("albedo")
            if gray is not None:
                result["details"]["mean"] = round(float(gray.mean()[0]), 4)
                result["details"]["pixel_0_0"] = round(float(gray.data[0, 0, 0]), 4)
        elif op == "cycle":
            try:
                cooked = cook(graph, cache)
                result["details"]["cycle"] = False
            except MaterialError as exc:
                result["details"]["cycle"] = exc.kind.value == "cycle"
                result["details"]["message"] = str(exc)
        elif op == "preset":
            name = case.get("preset", "metal")
            preset_graph = library.build_preset(name)
            cooked = cook(preset_graph, cache)
            result["details"]["channels"] = sorted(cooked.outputs.keys())
            albedo = cooked.outputs.get("albedo")
            if albedo is not None:
                result["details"]["mean"] = [round(float(v), 4)
                                             for v in albedo.mean()[:3]]
        elif op == "normal_vector":
            cooked = cook(graph, cache)
            normal = cooked.outputs.get("normal")
            if normal is not None:
                px = normal.data[0, 0]
                result["details"]["normal_0_0"] = [round(float(v), 4)
                                                   for v in px[:3]]
        elif op == "invalidate":
            first = cook(graph, cache)
            first_keys = {nid: info["key"]
                          for nid, info in first.per_node.items()}
            # change one upstream param; only downstream re-cooks
            for node in graph.nodes.values():
                if node.def_ == "perlin_noise":
                    node.params["scale"] = 8.0
                    break
            doc.mark_dirty()
            second = cook(graph, cache)
            recooked = [nid for nid, info in second.per_node.items()
                        if not info.get("cached") and nid in first_keys]
            result["details"]["recooked"] = sorted(recooked)
            result["details"]["nodes_cooked"] = second.nodes_cooked
            cooked = second
        else:
            raise MaterialError(MaterialErrorKind.invalid_argument,
                                f"unknown parity op {op!r}")
    for key, wanted in expect.items():
        got = result["details"].get(key)
        if isinstance(wanted, float) and isinstance(got, (int, float)):
            if abs(float(got) - wanted) > 1e-3:
                result["ok"] = False
                result["details"][f"mismatch_{key}"] = (got, wanted)
        elif (isinstance(wanted, list) and isinstance(got, list)
                and all(isinstance(x, (int, float)) for x in wanted)
                and all(isinstance(x, (int, float)) for x in got)):
            # numeric list (pixel values): compare with tolerance
            if len(got) != len(wanted) or any(
                    abs(float(a) - float(b)) > 1e-3
                    for a, b in zip(got, wanted, strict=True)):
                result["ok"] = False
                result["details"][f"mismatch_{key}"] = (got, wanted)
        elif got != wanted:
            result["ok"] = False
            result["details"][f"mismatch_{key}"] = (got, wanted)
    return result


def run_parity(toml_path: Path | str | None = None,
               fixtures: Path | str | None = None) -> dict:
    """Run every parity case; returns {passed, failed, cases}."""
    data = _load_toml(Path(toml_path) if toml_path else TOML_PATH)
    fixture_dir = Path(fixtures) if fixtures else FIXTURES
    cache = CookCache()
    cases = []
    for case in data.get("cases", []):
        try:
            cases.append(_run_case(case, fixture_dir, cache))
        except MaterialError as exc:
            cases.append({"id": case.get("id", "?"), "ok": False,
                          "details": {"error": str(exc)}})
    passed = sum(1 for c in cases if c["ok"])
    return {"passed": passed, "failed": len(cases) - passed,
            "total": len(cases), "cases": cases,
            "cache": cache.stats()}
