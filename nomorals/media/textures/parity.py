"""TextureCraft parity — machine-checked cases (lane 108).

``nomorals/media/textures-parity.toml`` lists real fixtures +
operations + expected outcomes. ``run_parity`` runs every case and
fails on any mismatch. Missing fixture = fail, never skip.

The loop lives in the shared library now
(``nomorals/craftlibs/parity.py``); this module keeps the
textures case runner and the textures result shape.
"""
from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

from nomorals.craftlibs.parity import run_cases

from .blend import BlendMode, blend1, blend3
from .compose import composite
from .dilate import NONE, DilationMap
from .errors import TextureError
from .mipmaps import FilterMode, WrapMode, mipmap_chain, sample
from .model import TextureDocument
from .noise import fbm, perlin, value
from .normal import height_to_normal
from .patterns import ProceduralKind
from .patterns import raw as pattern_raw
from .tiles import TileImage

__all__ = ["run_parity"]

_DEFAULT_PATH = (Path(__file__).resolve().parents[2] / "media"
                 / "textures-parity.toml")
_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "textures"
_TOL = 1e-6


def _close(a: float, b: float, tol: float = _TOL) -> bool:
    return abs(a - b) <= tol


def _run_case(case: dict[str, Any]) -> None:
    op = str(case.get("operation", ""))
    expect = case.get("expect", {}) or {}
    fixture = case.get("fixture")
    fixture_path = _FIXTURES / str(fixture) if fixture else None
    if fixture_path is not None and not fixture_path.exists():
        raise AssertionError(f"missing fixture: {fixture}")
    if op == "roundtrip":
        assert fixture_path is not None
        raw = fixture_path.read_bytes()
        doc = TextureDocument.from_bytes(raw)
        if doc.to_bytes() != raw:
            raise AssertionError("unedited save is not byte-identical")
    elif op == "cow":
        img = TileImage(64, 64, 1)
        img.set(0, 0, [5])
        clone = img.clone()
        if img.tile_handle(0, 0) is not clone.tile_handle(0, 0):
            raise AssertionError("clone does not share its tile")
        img.set(1, 0, [6])
        if img.tile_handle(0, 0) is clone.tile_handle(0, 0):
            raise AssertionError("write did not un-share the tile")
        if clone.get(1, 0)[0] != 0 or img.get(1, 0)[0] != 6:
            raise AssertionError("copy-on-write leaked a write")
        if img.bytes() != 8192:
            raise AssertionError(f"tile accounting: {img.bytes()}")
    elif op == "noise":
        points = [(0.0, 0.0, 0.0), (0.5, 0.25, 0.125),
                  (1.3, 2.7, 0.4), (3.7, 1.1, 9.2),
                  (10.5, 10.5, 10.5), (-2.25, 4.75, -0.5)]
        want_value = [0.044569507, 0.186984763, 0.705094278, 0.278636873,
                      0.391551822, 0.538408279]
        want_perlin = [0.5, 0.502421975, 0.460757077, 0.308367729,
                       0.4375, 0.511600018]
        want_fbm = [0.178656384, 0.337487936, 0.552097023, 0.401274621,
                    0.409147024, 0.560437799]
        for p, wv, wp, wf in zip(points, want_value, want_perlin, want_fbm,
                                strict=True):
            if not _close(value(p, 7), wv):
                raise AssertionError(f"value{p} = {value(p, 7)} want {wv}")
            if not _close(perlin(p, 7), wp):
                raise AssertionError(f"perlin{p} = {perlin(p, 7)} want {wp}")
            if not _close(fbm(p, 5, 7, value), wf):
                raise AssertionError(f"fbm{p} = {fbm(p, 5, 7, value)} want {wf}")
    elif op == "pattern":
        cases = {"clouds": 0.525642514, "perlin": 0.605365098,
                 "checker": 1.0, "stripes": 1.0, "gradient": 0.25,
                 "wood": 0.861425757, "bricks": 0.945120573,
                 "tiles": 0.917647302, "voronoi": 0.033640295}
        for name, want in cases.items():
            kind = ProceduralKind.parse(name)
            assert kind is not None
            got = pattern_raw(kind, 0.25, 0.75, 0.0, 3)
            if not _close(got, want, 2e-6):
                raise AssertionError(f"{name} = {got} want {want}")
    elif op == "blend":
        modes = list(BlendMode.all())
        if len(modes) != 32:
            raise AssertionError(f"blend menu has {len(modes)} modes")
        for mode in modes:
            for a in (0.0, 0.25, 0.5, 1.0):
                for b in (0.0, 0.3, 0.7, 1.0):
                    if not math.isfinite(blend1(mode, a, b)):
                        raise AssertionError(f"{mode} not finite")
                    out = blend3(mode, (a, a, a), (b, b, b))
                    if not all(math.isfinite(v) for v in out):
                        raise AssertionError(f"{mode} blend3 not finite")
        checks = [("multiply", 0.5, 0.7, 0.349999994),
                  ("screen", 0.5, 0.7, 0.850000024),
                  ("overlay", 0.5, 0.7, 0.699999988)]
        for name, a, b, want in checks:
            mode = BlendMode.parse(name)
            assert mode is not None
            if not _close(blend1(mode, a, b), want, 2e-6):
                raise AssertionError(f"{name} mismatch")
    elif op == "composite":
        assert fixture_path is not None
        doc = TextureDocument.open(fixture_path)
        out = composite(doc, "Body", "base_color")
        for texel in expect.get("texels", []):
            x, y = int(texel["x"]), int(texel["y"])
            got = out.get(x, y)
            want = [int(v) for v in texel["values"]]
            tol = int(texel.get("tolerance", 0))
            for g, w in zip(got, want, strict=True):
                if abs(g - w) > tol:
                    raise AssertionError(
                        f"composite({x},{y}) = {got} want {want} ±{tol}")
    elif op == "mipmap":
        img = TileImage(128, 128, 1)
        for y in range(64):
            for x in range(64):
                img.set(x, y, [30000])
        chain = mipmap_chain(img)
        if [c.width for c in chain] != [128, 64, 32, 16, 8, 4, 2, 1]:
            raise AssertionError("mipmap chain sizes wrong")
        if chain[1].get(16, 16)[0] != 30000:
            raise AssertionError("mipmap level 1 lost the flat value")
        if chain[1].get(60, 60)[0] != 0:
            raise AssertionError("mipmap level 1 has data outside the fill")
    elif op == "sample":
        img = TileImage(64, 64, 1)
        img.set(0, 0, [0])
        img.set(1, 0, [65535])
        img.set(0, 1, [65535])
        img.set(1, 1, [65535])
        v = sample([img], 1.0 / 64.0, 1.0 / 64.0, 0.0,
                   WrapMode.CLAMP, FilterMode.BILINEAR)
        if abs(v[0] - 0.75 * 65535) > 2000:
            raise AssertionError(f"bilinear sample = {v[0]}")
        img2 = TileImage(64, 64, 1)
        img2.set(32, 32, [65535])
        chain = mipmap_chain(img2)
        v0 = sample(chain, 32.5 / 64, 32.5 / 64, 0.0,
                    WrapMode.CLAMP, FilterMode.NEAREST)[0]
        v1 = sample(chain, 32.5 / 64, 32.5 / 64, 1.0,
                    WrapMode.CLAMP, FilterMode.TRILINEAR)[0]
        if v0 != 65535 or not 0 < v1 < 65535:
            raise AssertionError(f"trilinear: {v0} {v1}")
    elif op == "normal":
        flat = TileImage(64, 64, 1)
        n = height_to_normal(flat, strength=1.0)
        if n.get(32, 32)[:3] != [32768, 32768, 65535]:
            raise AssertionError(f"flat normal = {n.get(32, 32)}")
        slope = TileImage(64, 64, 1)
        for x in range(64):
            for y in range(64):
                slope.set(x, y, [x * 1024])
        n2 = height_to_normal(slope, strength=1.0)
        r, g, b = n2.get(32, 32)[:3]
        if not (r < 32768 and g == 32768 and b > 32768):
            raise AssertionError(f"slope normal = {(r, g, b)}")
    elif op == "dilate":
        covered = [False] * 64
        covered[3 * 8 + 3] = True
        m = DilationMap(covered, 8, 8, max_dist=2)
        for y in range(8):
            for x in range(8):
                s = m.src[y * 8 + x]
                if 1 <= x <= 5 and 1 <= y <= 5:
                    if s != 27:
                        raise AssertionError(f"dilate({x},{y}) = {s}")
                elif s != NONE:
                    raise AssertionError(f"dilate({x},{y}) = {s}")
    elif op == "open_rejected":
        assert fixture_path is not None
        try:
            TextureDocument.open(fixture_path)
        except TextureError:
            return
        raise AssertionError("broken fixture opened without an error")
    else:
        raise AssertionError(f"unknown parity operation: {op!r}")


def run_parity(path: str | Path | None = None) -> dict[str, Any]:
    toml_path = Path(path) if path is not None else _DEFAULT_PATH
    if not toml_path.exists():
        return {"passed": 0, "failed": ["<toml>"], "passed_ids": [],
                "failures": [{"id": "<toml>",
                              "reason": f"missing {toml_path}"}],
                "skipped": 0}
    with toml_path.open("rb") as handle:
        data = tomllib.load(handle)
    result = run_cases(data.get("cases", []), run_case=_run_case)
    return {"passed": result["passed"],
            "failed": result["failed"],
            "passed_ids": result["passed_ids"],
            "failures": result["failures"],
            "skipped": 0}
