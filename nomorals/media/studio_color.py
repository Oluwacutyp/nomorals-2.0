"""Devon Studio — professional color grading (rebuilt engine, §1.1c).

One canonical color engine, four thin surfaces:

- **LUT engine** — real ``.cube`` parsing *and* generation (the file is
  no longer passed opaquely to ffmpeg): :class:`CubeLUT` parses
  TITLE / LUT_1D_SIZE / LUT_3D_SIZE / DOMAIN_MIN / DOMAIN_MAX, applies
  with trilinear interpolation, serializes back to standard ``.cube``
  text. Lattice order is the .cube standard — red varies fastest
  (row index = ``r + N*g + N^2*b``), verified against ffmpeg's lut3d.
- **ASC-CDL** — the true ASC CDL v1.2 math (slope/offset/power per
  channel + saturation around Rec.709 luma), replacing the old
  ``colorbalance`` approximation that wore the name. ``color_wheels``
  keeps its signature and now *is* a CDL: slope=gain, offset=lift,
  power=gamma.
- **Scopes** — waveform / RGB parade / vectorscope / histogram
  computed from real frame data (numpy), exposed as data structures
  for the brain/QC. Nothing here renders a picture of a scope; a
  missing rung (numpy/PIL/ffmpeg) returns a named reason instead.
- **One grade representation** — a grade (CDL and/or curves) bakes to
  a :class:`CubeLUT`; video applies that LUT through ffmpeg ``lut3d``
  (the single render path the edit_engine chain also uses), stills
  apply it through numpy/PIL, and ``media_edit``'s PIL grade tables
  are built by this module's :func:`channel_tables`.

Style-agnostic: these are primitives. Looks live in presets (LOOKS).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .failures import record_failure


class StudioColorError(RuntimeError):
    """A named color-engine failure (bad LUT, bad CDL, missing rung)."""


def _ffmpeg() -> str | None:
    from shutil import which
    return which("ffmpeg") or which("/usr/bin/ffmpeg")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")  # noqa: S603


def _np():
    """numpy, or a StudioColorError naming the missing rung."""
    try:
        import numpy as np
    except ImportError as exc:
        raise StudioColorError(
            "numpy is not installed — the color engine's LUT/CDL/scope "
            "math needs it (pip install numpy)") from exc
    return np


# ── .cube LUT engine ───────────────────────────────────────────────

_MAX_LUT_SIZE = 256


@dataclass
class CubeLUT:
    """A parsed/generated 3D LUT (+ optional 1D pre-LUT).

    ``data`` is a float32 array of shape ``(size**3, 3)`` in .cube row
    order (red fastest). ``lut1d`` is ``(m, 3)`` or None, applied first.
    Values live in the LUT's domain (default 0..1 in, unbounded out —
    clamping happens in the uint8/image apply paths, as lut3d does).
    """

    size: int
    data: Any  # np.ndarray (size**3, 3) float32
    domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    title: str = ""
    lut1d: Any = None  # np.ndarray (m, 3) float32 or None

    # -- construction ------------------------------------------------

    @classmethod
    def identity(cls, size: int = 17) -> "CubeLUT":
        return cls.from_function(size, lambda r, g, b: (r, g, b),
                                 title="identity")

    @classmethod
    def from_function(cls, size: int,
                      fn: Callable[[Any, Any, Any], tuple[Any, Any, Any]],
                      *, title: str = "") -> "CubeLUT":
        """Sample ``fn(r, g, b) -> (r, g, b)`` on the lattice.

        ``fn`` receives numpy arrays of lattice coordinates (domain
        0..1) and returns three arrays — vectorized by construction.
        """
        np = _np()
        if size < 2 or size > _MAX_LUT_SIZE:
            raise StudioColorError(
                f"LUT size {size} out of range 2..{_MAX_LUT_SIZE}")
        axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
        # Grid axes are (b, g, r): C-order ravel then makes red vary
        # fastest — the .cube row order (index = r + N*g + N^2*b),
        # which is also how sample() reshapes the data back.
        bb, gg, rr = np.meshgrid(axis, axis, axis, indexing="ij")
        fr, fg, fb = fn(rr, gg, bb)
        data = np.empty((size ** 3, 3), dtype=np.float64)
        data[:, 0] = np.asarray(fr, dtype=np.float64).ravel()
        data[:, 1] = np.asarray(fg, dtype=np.float64).ravel()
        data[:, 2] = np.asarray(fb, dtype=np.float64).ravel()
        return cls(size=size, data=data.astype(np.float32), title=title)

    @classmethod
    def from_tables(cls, tables: Sequence[Sequence[float]],
                    size: int = 32) -> "CubeLUT":
        """A separable grade (three 256-entry 1D tables) as a 3D LUT."""
        np = _np()
        tabs = [np.asarray(t, dtype=np.float64) for t in tables]
        if len(tabs) != 3 or any(t.shape != (256,) for t in tabs):
            raise StudioColorError(
                "from_tables needs three 256-entry channel tables")

        def fn(r, g, b):
            def look(tab, v):
                return np.interp(v, np.linspace(0.0, 1.0, 256), tab)
            return look(tabs[0], r), look(tabs[1], g), look(tabs[2], b)
        return cls.from_function(size, fn)

    # -- parsing / serialization --------------------------------------

    @classmethod
    def parse(cls, text: str) -> "CubeLUT":
        """Parse real .cube text. Raises StudioColorError naming the
        exact defect (bad keyword, wrong row count, bad number)."""
        np = _np()
        size: int | None = None
        size1d: int | None = None
        dmin = (0.0, 0.0, 0.0)
        dmax = (1.0, 1.0, 1.0)
        title = ""
        rows: list[tuple[float, float, float]] = []
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            head = line.split(None, 1)[0].upper()
            if head == "TITLE":
                rest = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
                title = rest.strip().strip('"')
                continue
            if head in ("LUT_3D_SIZE", "LUT_1D_SIZE"):
                parts = line.split()
                try:
                    n = int(parts[1])
                except (IndexError, ValueError) as exc:
                    raise StudioColorError(
                        f"{head} needs an integer size (line {lineno})") from exc
                if n < 2 or n > _MAX_LUT_SIZE:
                    raise StudioColorError(
                        f"{head} {n} out of range 2..{_MAX_LUT_SIZE} "
                        f"(line {lineno})")
                if head == "LUT_3D_SIZE":
                    size = n
                else:
                    size1d = n
                continue
            if head in ("DOMAIN_MIN", "DOMAIN_MAX"):
                parts = line.split()
                try:
                    vals = tuple(float(p) for p in parts[1:4])
                except (IndexError, ValueError) as exc:
                    raise StudioColorError(
                        f"{head} needs three floats (line {lineno})") from exc
                if len(vals) != 3:
                    raise StudioColorError(
                        f"{head} needs three floats (line {lineno})")
                if head == "DOMAIN_MIN":
                    dmin = vals  # type: ignore[assignment]
                else:
                    dmax = vals  # type: ignore[assignment]
                continue
            if head.isalpha() and not _looks_like_data(line):
                raise StudioColorError(
                    f"unknown .cube keyword {head!r} (line {lineno})")
            parts = line.split()
            try:
                vals = tuple(float(p) for p in parts)
            except ValueError as exc:
                raise StudioColorError(
                    f"bad LUT data row (line {lineno}): {line!r}") from exc
            if len(vals) != 3:
                raise StudioColorError(
                    f"LUT data rows need exactly 3 values "
                    f"(line {lineno}): {line!r}")
            rows.append(vals)
        if size is None and size1d is None:
            raise StudioColorError(
                "not a .cube LUT: no LUT_3D_SIZE or LUT_1D_SIZE found")
        expect = (size1d or 0) + (size ** 3 if size else 0)
        if len(rows) != expect:
            raise StudioColorError(
                f".cube row count mismatch: expected {expect} rows "
                f"(1D {size1d or 0} + 3D {size ** 3 if size else 0}), "
                f"found {len(rows)}")
        arr = np.asarray(rows, dtype=np.float32) if rows else \
            np.zeros((0, 3), dtype=np.float32)
        lut1d = arr[:size1d] if size1d else None
        data = arr[size1d:] if size else np.zeros((0, 3), dtype=np.float32)
        return cls(size=size or 0, data=data, domain_min=dmin,
                   domain_max=dmax, title=title, lut1d=lut1d)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "CubeLUT":
        p = Path(path)
        if not p.exists():
            raise StudioColorError(f"no such LUT file: {p}")
        try:
            return cls.parse(p.read_text(encoding="utf-8", errors="replace"))
        except StudioColorError:
            raise
        except OSError as exc:
            raise StudioColorError(f"cannot read LUT {p}: {exc}") from exc

    def to_cube_text(self) -> str:
        lines: list[str] = []
        if self.title:
            lines.append(f'TITLE "{self.title}"')
        if self.lut1d is not None:
            lines.append(f"LUT_1D_SIZE {len(self.lut1d)}")
        if self.size:
            lines.append(f"LUT_3D_SIZE {self.size}")
        if tuple(self.domain_min) != (0.0, 0.0, 0.0):
            lines.append("DOMAIN_MIN " + " ".join(
                f"{v:.6f}" for v in self.domain_min))
        if tuple(self.domain_max) != (1.0, 1.0, 1.0):
            lines.append("DOMAIN_MAX " + " ".join(
                f"{v:.6f}" for v in self.domain_max))
        rows = []
        if self.lut1d is not None:
            rows.extend(self.lut1d)
        if self.size:
            rows.extend(self.data)
        for row in rows:
            lines.append(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
        return "\n".join(lines) + "\n"

    def write(self, path: str | os.PathLike[str]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_cube_text(), encoding="utf-8")
        return p

    # -- application ---------------------------------------------------

    def _apply_1d(self, arr: Any) -> Any:
        np = _np()
        if self.lut1d is None:
            return arr
        m = len(self.lut1d)
        xs = np.linspace(0.0, 1.0, m)
        out = np.empty_like(arr)
        for c in range(3):
            out[..., c] = np.interp(arr[..., c], xs, self.lut1d[:, c])
        return out

    def sample(self, rgb: Any) -> Any:
        """Trilinear sample. ``rgb``: (..., 3) floats in domain units.

        Returns raw interpolated values (unclamped), shape (..., 3).
        """
        np = _np()
        arr = np.asarray(rgb, dtype=np.float64)
        if arr.shape == () or arr.shape[-1] != 3:
            raise StudioColorError(
                f"sample needs (..., 3) RGB data, got shape {arr.shape}")
        flat = arr.reshape(-1, 3)
        flat = self._apply_1d(flat)
        if not self.size:
            return flat.reshape(arr.shape)
        dmin = np.asarray(self.domain_min, dtype=np.float64)
        dmax = np.asarray(self.domain_max, dtype=np.float64)
        span = np.where(dmax > dmin, dmax - dmin, 1.0)
        u = (flat - dmin) / span
        n = self.size
        grid = self.data.reshape(n, n, n, 3)  # [b][g][r] — red fastest
        pos = np.clip(u, 0.0, 1.0) * (n - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, n - 1)
        frac = pos - lo
        # grid axes are [b, g, r]; input columns are (r, g, b).
        r_lo, g_lo, b_lo = lo[:, 0], lo[:, 1], lo[:, 2]
        r_hi, g_hi, b_hi = hi[:, 0], hi[:, 1], hi[:, 2]
        fr, fg, fb = frac[:, 0], frac[:, 1], frac[:, 2]
        c000 = grid[b_lo, g_lo, r_lo]
        c100 = grid[b_lo, g_lo, r_hi]
        c010 = grid[b_lo, g_hi, r_lo]
        c110 = grid[b_lo, g_hi, r_hi]
        c001 = grid[b_hi, g_lo, r_lo]
        c101 = grid[b_hi, g_lo, r_hi]
        c011 = grid[b_hi, g_hi, r_lo]
        c111 = grid[b_hi, g_hi, r_hi]
        wr = fr[:, None]
        wg = fg[:, None]
        wb = fb[:, None]
        out = (
            c000 * (1 - wr) * (1 - wg) * (1 - wb)
            + c100 * wr * (1 - wg) * (1 - wb)
            + c010 * (1 - wr) * wg * (1 - wb)
            + c110 * wr * wg * (1 - wb)
            + c001 * (1 - wr) * (1 - wg) * wb
            + c101 * wr * (1 - wg) * wb
            + c011 * (1 - wr) * wg * wb
            + c111 * wr * wg * wb
        )
        return out.reshape(arr.shape)

    def apply(self, arr: Any) -> Any:
        """Apply to a float (..., 3) array in 0..1. Output clamped."""
        np = _np()
        out = self.sample(arr)
        return np.clip(out, 0.0, 1.0)

    def apply_uint8(self, arr: Any) -> Any:
        """Apply to a uint8 (..., 3) array. Identity LUTs round-trip
        bit-identically (trilinear reproduces linear maps exactly)."""
        np = _np()
        a = np.asarray(arr)
        if a.dtype != np.uint8:
            raise StudioColorError(
                f"apply_uint8 needs a uint8 array, got {a.dtype}")
        out = self.apply(a.astype(np.float64) / 255.0)
        return np.rint(out * 255.0).astype(np.uint8)

    def apply_to_image(self, img: Any) -> Any:
        """Apply to a PIL image (RGB), returning a new PIL image."""
        np = _np()
        try:
            from PIL import Image
        except ImportError as exc:
            raise StudioColorError(
                "Pillow is not installed — still-image LUT application "
                "needs it (pip install Pillow)") from exc
        arr = np.asarray(img.convert("RGB"))
        return Image.fromarray(self.apply_uint8(arr), "RGB")


def _looks_like_data(line: str) -> bool:
    try:
        parts = line.split()
        [float(p) for p in parts]
        return len(parts) == 3
    except ValueError:
        return False


def parse_cube(text: str) -> CubeLUT:
    return CubeLUT.parse(text)


def load_cube(path: str | os.PathLike[str]) -> CubeLUT:
    return CubeLUT.load(path)


def generate_cube(size: int,
                  fn: Callable[[Any, Any, Any], tuple[Any, Any, Any]],
                  *, path: str | os.PathLike[str] | None = None,
                  title: str = "") -> CubeLUT:
    """Generate a LUT by sampling ``fn`` on the lattice; optionally
    write it as a .cube file. Returns the CubeLUT."""
    lut = CubeLUT.from_function(size, fn, title=title)
    if path is not None:
        lut.write(path)
    return lut


# ── ASC-CDL (the real math) ────────────────────────────────────────

#: Rec.709 luma coefficients — the ASC CDL specification's saturation
#: step is defined against these, not BT.601.
_CDL_LUMA = (0.2126, 0.7152, 0.0722)


def _vec3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    try:
        v = tuple(float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise StudioColorError(f"CDL {name} must be three numbers") from exc
    if len(v) != 3:
        raise StudioColorError(
            f"CDL {name} must have exactly 3 components, got {len(v)}")
    return v  # type: ignore[return-value]


@dataclass
class CDL:
    """One ASC-CDL decision: SOP (slope/offset/power) + saturation.

    The standard math (ASC CDL v1.2), per channel::

        sop = (in * slope + offset) ** power      (base clamped at 0)
        out = luma + saturation * (sop - luma)    (Rec.709 luma)
        out = clamp(out, 0, 1)
    """

    slope: tuple[float, float, float] = (1.0, 1.0, 1.0)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    power: tuple[float, float, float] = (1.0, 1.0, 1.0)
    saturation: float = 1.0

    def __post_init__(self) -> None:
        self.slope = _vec3(self.slope, "slope")
        self.offset = _vec3(self.offset, "offset")
        self.power = _vec3(self.power, "power")
        if any(p <= 0 for p in self.power):
            raise StudioColorError(
                f"CDL power must be > 0 per channel, got {self.power}")
        self.saturation = float(self.saturation)

    def is_identity(self) -> bool:
        return (self.slope == (1.0, 1.0, 1.0)
                and self.offset == (0.0, 0.0, 0.0)
                and self.power == (1.0, 1.0, 1.0)
                and self.saturation == 1.0)

    def apply_array(self, arr: Any) -> Any:
        """Apply the CDL to a float (..., 3) array (values nominally
        0..1). Pure math — bit-identical output for an identity CDL."""
        np = _np()
        a = np.asarray(arr, dtype=np.float64)
        if a.shape == () or a.shape[-1] != 3:
            raise StudioColorError(
                f"CDL apply needs (..., 3) RGB data, got {a.shape}")
        if self.is_identity():
            # An identity decision must be bit-identical — bypass the
            # pow entirely (vectorized pow(x, 1.0) is ~1ulp off).
            return np.array(a, dtype=np.float64, copy=True)
        slope = np.asarray(self.slope)
        offset = np.asarray(self.offset)
        power = np.asarray(self.power)
        sop = np.power(np.maximum(a * slope + offset, 0.0), power)
        if self.saturation != 1.0:
            luma = sop @ np.asarray(_CDL_LUMA)
            sop = luma[..., None] + self.saturation * (sop - luma[..., None])
        return np.clip(sop, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {"slope": list(self.slope), "offset": list(self.offset),
                "power": list(self.power), "saturation": self.saturation}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CDL":
        return cls(slope=_vec3(data.get("slope", (1, 1, 1)), "slope"),
                   offset=_vec3(data.get("offset", (0, 0, 0)), "offset"),
                   power=_vec3(data.get("power", (1, 1, 1)), "power"),
                   saturation=float(data.get("saturation", 1.0)))


def apply_cdl_array(arr: Any, cdl: CDL) -> Any:
    return cdl.apply_array(arr)


def cdl_from_wheels(lift: Sequence[float], gamma: Sequence[float],
                    gain: Sequence[float], *,
                    saturation: float = 1.0) -> CDL:
    """The color-wheel vocabulary as a true CDL: gain is slope, lift
    is offset, gamma is power. (The pre-rebuild wheels were an ffmpeg
    ``colorbalance`` approximation — additive shadow/mid/highlight
    terms wearing CDL's name.)"""
    return CDL(slope=_vec3(gain, "gain"), offset=_vec3(lift, "lift"),
               power=_vec3(gamma, "gamma"), saturation=saturation)


# ── curves ─────────────────────────────────────────────────────────

def parse_curve_points(spec: str) -> list[tuple[float, float]]:
    """Parse ffmpeg-style curve points: ``'0/0 0.5/0.6 1/1'``."""
    pts: list[tuple[float, float]] = []
    for tok in str(spec).split():
        if "/" not in tok:
            raise StudioColorError(f"bad curve point {tok!r}")
        xs, ys = tok.split("/", 1)
        try:
            pts.append((float(xs), float(ys)))
        except ValueError as exc:
            raise StudioColorError(f"bad curve point {tok!r}") from exc
    if len(pts) < 2:
        raise StudioColorError("a curve needs at least two points")
    pts.sort()
    return pts


def evaluate_curve(points: Sequence[tuple[float, float]], xs: Any) -> Any:
    """Catmull-Rom spline through the control points, clamped to
    [0, 1] on output; inputs outside the point range take the end
    values. Deterministic and smooth — the curve shape Resolve-style
    tools draw."""
    np = _np()
    pts = sorted((float(x), float(y)) for x, y in points)
    if len(pts) < 2:
        raise StudioColorError("a curve needs at least two points")
    px = np.asarray([p[0] for p in pts])
    py = np.asarray([p[1] for p in pts])
    x = np.asarray(xs, dtype=np.float64)
    out = np.interp(x, px, py)  # base + end clamping
    # Catmull-Rom refinement inside each segment.
    idx = np.clip(np.searchsorted(px, x) - 1, 0, len(pts) - 2)
    x0 = px[idx]
    x1 = px[idx + 1]
    t = np.where(x1 > x0, (x - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0)
    inside = (x >= px[0]) & (x <= px[-1])
    if len(pts) >= 3 and bool(inside.any()):
        p0 = py[np.clip(idx - 1, 0, len(pts) - 1)]
        p1 = py[idx]
        p2 = py[idx + 1]
        p3 = py[np.clip(idx + 2, 0, len(pts) - 1)]
        t2 = t * t
        t3 = t2 * t
        cr = (0.5 * ((2.0 * p1) + (-p0 + p2) * t
                     + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                     + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3))
        out = np.where(inside, cr, out)
    return np.clip(out, 0.0, 1.0)


def channel_tables(fn: Callable[[int, float], float]) -> list[int]:
    """The shared 1D LUT representation for stills: 768 ints
    (256 per channel, PIL ``Image.point`` layout) built from
    ``fn(channel, x) -> y`` with x, y in 0..1. ``media_edit``'s grade
    builder delegates here so stills and the video engine share one
    table builder."""
    table: list[int] = []
    for c in range(3):
        for i in range(256):
            v = fn(c, i / 255.0)
            table.append(max(0, min(255, int(round(v * 255)))))
    return table


# ── grade baking: CDL + curves → one CubeLUT ───────────────────────

def bake_grade_lut(size: int = 17, *, cdl: CDL | None = None,
                   curves: dict[str, Any] | None = None,
                   title: str = "") -> CubeLUT:
    """Bake a grade (curves, then CDL) into a CubeLUT — the single
    representation every surface applies.

    ``curves`` maps master/red/green/blue to point lists or ffmpeg
    point strings. Order is curves first (per-channel shaping), then
    the CDL decision — documented because order is audible in the
    result.
    """
    np = _np()
    specs: dict[str, list[tuple[float, float]]] = {}
    if curves:
        for name, spec in curves.items():
            if spec is None:
                continue
            specs[name] = (parse_curve_points(spec) if isinstance(spec, str)
                           else [(float(x), float(y)) for x, y in spec])

    def fn(r, g, b):
        chans = [r, g, b]
        if "master" in specs:
            chans = [evaluate_curve(specs["master"], c) for c in chans]
        for i, name in enumerate(("red", "green", "blue")):
            if name in specs:
                chans[i] = evaluate_curve(specs[name], chans[i])
        arr = np.stack(chans, axis=-1)
        if cdl is not None:
            arr = cdl.apply_array(arr)
        return arr[..., 0], arr[..., 1], arr[..., 2]
    return CubeLUT.from_function(size, fn, title=title or "baked grade")


def lut_cache_dir() -> Path:
    env = os.environ.get("NM_DEVON_LUT_CACHE")
    return Path(env) if env else Path.home() / ".devon" / "lut_cache"


def bake_cdl_cube_file(cdl: CDL, *, size: int = 17,
                       cache_dir: str | os.PathLike[str] | None = None,
                       curves: dict[str, Any] | None = None) -> Path:
    """Bake a CDL (+optional curves) to a content-addressed .cube on
    disk — the file ffmpeg's lut3d consumes in the render graph.
    Same grade ⇒ same path ⇒ renders stay reproducible."""
    key_src = json.dumps(
        {"cdl": cdl.to_dict(), "curves": curves or {}, "size": size},
        sort_keys=True)
    digest = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]
    d = Path(cache_dir) if cache_dir else lut_cache_dir()
    path = d / f"grade_{digest}.cube"
    if not path.exists():
        lut = bake_grade_lut(size, cdl=cdl, curves=curves,
                             title=f"CDL {digest}")
        tmp = path.with_suffix(".cube.tmp")
        lut.write(tmp)
        os.replace(tmp, path)
    return path


# ── scopes (computed from real frame data, exposed as data) ────────

#: BT.601 luma — scopes read the frame the way classic scope hardware
#: did; the choice is stated here because it is part of the reading.
_SCOPE_LUMA = (0.299, 0.587, 0.114)


def _frame_float(arr: Any):
    """Normalize an image input to a float64 (H, W, 3) array in 0..1.

    Accepts numpy arrays (uint8 or float) and PIL images."""
    np = _np()
    if hasattr(arr, "convert") and hasattr(arr, "size"):  # PIL image
        arr = np.asarray(arr.convert("RGB"))
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise StudioColorError(
            f"scope input must be an (H, W, 3) RGB frame, got {a.shape}")
    if a.dtype == np.uint8:
        return a.astype(np.float64) / 255.0
    return np.clip(a.astype(np.float64), 0.0, 1.0)


def _luma(a: Any) -> Any:
    return (a[..., 0] * _SCOPE_LUMA[0] + a[..., 1] * _SCOPE_LUMA[1]
            + a[..., 2] * _SCOPE_LUMA[2])


def histogram_data(arr: Any, *, bins: int = 256) -> dict[str, list[int]]:
    """Per-channel + luma histograms (counts)."""
    np = _np()
    a = _frame_float(arr)
    out: dict[str, list[int]] = {}
    for i, name in enumerate(("r", "g", "b")):
        h, _ = np.histogram(a[..., i], bins=bins, range=(0.0, 1.0))
        out[name] = [int(v) for v in h]
    h, _ = np.histogram(_luma(a), bins=bins, range=(0.0, 1.0))
    out["luma"] = [int(v) for v in h]
    return out


def _waveform_of(values: Any, width: int, levels: int) -> list[list[int]]:
    np = _np()
    h, w = values.shape
    cols = (np.arange(w) * width) // max(1, w)
    lvl = np.clip((values * levels).astype(np.int64), 0, levels - 1)
    idx = (lvl * width + cols[None, :]).ravel()
    counts = np.bincount(idx, minlength=levels * width)
    return [[int(v) for v in row]
            for row in counts.reshape(levels, width)]


def waveform_data(arr: Any, *, width: int = 128,
                  levels: int = 256) -> list[list[int]]:
    """Luma waveform: ``levels`` rows (row 0 = black) × ``width``
    columns of pixel counts — the scope, as data."""
    a = _frame_float(arr)
    return _waveform_of(_luma(a), width, levels)


def rgb_parade_data(arr: Any, *, width: int = 128,
                    levels: int = 256) -> dict[str, list[list[int]]]:
    """Waveform per channel — the RGB parade, as data."""
    a = _frame_float(arr)
    return {name: _waveform_of(a[..., i], width, levels)
            for i, name in enumerate(("r", "g", "b"))}


def chroma_cbcr(arr: Any) -> tuple[Any, Any]:
    """BT.601 (Cb, Cr) per pixel: Cb = (B−Y)/1.772, Cr = (R−Y)/1.402,
    each in ≈ [−0.5, 0.5]."""
    a = _frame_float(arr)
    y = _luma(a)
    cb = (a[..., 2] - y) / 1.772
    cr = (a[..., 0] - y) / 1.402
    return cb, cr


def vectorscope_data(arr: Any, *, size: int = 128) -> list[list[int]]:
    """Vectorscope: 2D counts over the (Cb, Cr) plane. Row 0 is the
    TOP of the scope (+Cr), columns run −Cb → +Cb."""
    np = _np()
    cb, cr = chroma_cbcr(arr)
    xi = np.clip(((cb + 0.5) * size).astype(np.int64), 0, size - 1)
    yi = np.clip(((cr + 0.5) * size).astype(np.int64), 0, size - 1)
    idx = ((size - 1 - yi) * size + xi).ravel()
    counts = np.bincount(idx, minlength=size * size)
    return [[int(v) for v in row] for row in counts.reshape(size, size)]


def vectorscope_centroid(arr: Any) -> tuple[float, float]:
    """Mean (Cb, Cr) — where the frame's chroma sits, as one point.
    A neutral frame sits at (0, 0); pure red at ≈ (−0.169, +0.500)."""
    cb, cr = chroma_cbcr(arr)
    return float(cb.mean()), float(cr.mean())


def measure_frame(arr: Any) -> dict[str, Any]:
    """Numeric frame measurements for the brain/QC: per-channel and
    luma ranges, percentiles, clipping, mean chroma/saturation."""
    np = _np()
    a = _frame_float(arr)
    luma = _luma(a)
    cb, cr = chroma_cbcr(a)
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
    chans: dict[str, Any] = {}
    for i, name in enumerate(("r", "g", "b")):
        ch = a[..., i]
        chans[name] = {"min": float(ch.min()), "max": float(ch.max()),
                       "mean": float(ch.mean())}
    return {
        "width": int(a.shape[1]), "height": int(a.shape[0]),
        "channels": chans,
        "luma": {"min": float(luma.min()), "max": float(luma.max()),
                 "mean": float(luma.mean()),
                 "p1": float(np.percentile(luma, 1)),
                 "p50": float(np.percentile(luma, 50)),
                 "p99": float(np.percentile(luma, 99))},
        "clipped_black_pct": float((luma <= 1 / 255).mean() * 100.0),
        "clipped_white_pct": float((luma >= 254 / 255).mean() * 100.0),
        "mean_chroma": {"cb": float(cb.mean()), "cr": float(cr.mean())},
        "mean_saturation": float(sat.mean()),
    }


def scopes(arr: Any, *, waveform_width: int = 128,
           vectorscope_size: int = 128) -> dict[str, Any]:
    """All scopes + measurements for one frame, as data."""
    try:
        a = _frame_float(arr)
    except StudioColorError as exc:
        return {"ok": False, "reason": str(exc)}
    return {
        "ok": True,
        "measurements": measure_frame(a),
        "histogram": histogram_data(a),
        "waveform": waveform_data(a, width=waveform_width),
        "rgb_parade": rgb_parade_data(a, width=waveform_width),
        "vectorscope": vectorscope_data(a, size=vectorscope_size),
    }


def _grab_frame(src: str, at: float = 1.0) -> tuple[Any, str | None]:
    """One frame of a media file as a float (H, W, 3) array.

    Returns (array, None) or (None, named-reason)."""
    np = _np()
    ff = _ffmpeg()
    if not ff:
        return None, "ffmpeg not found — cannot extract a frame to scope"
    if not Path(src).exists():
        return None, f"no such media file: {src}"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        frame = tf.name
    try:
        proc = _run([ff, "-hide_banner", "-y", "-ss", str(at), "-i", src,
                     "-frames:v", "1", frame])
        if proc.returncode != 0 or not Path(frame).exists():
            return None, (f"ffmpeg could not extract a frame at {at}s: "
                          f"{proc.stderr[-300:]}")
        try:
            from PIL import Image
        except ImportError:
            return None, ("Pillow is not installed — cannot decode the "
                          "extracted frame (pip install Pillow)")
        with Image.open(frame) as im:
            arr = np.asarray(im.convert("RGB"))
        return arr.astype(np.float64) / 255.0, None
    finally:
        try:
            os.unlink(frame)
        except OSError as exc:
            record_failure("studio_color._grab_frame", exc)


def scopes_for_media(src: str | os.PathLike[str], *,
                     at: float = 1.0) -> dict[str, Any]:
    """Scopes for a frame of a media file (video or still image).

    Failure is a dict with a named reason — never a fabricated scope.
    """
    try:
        _np()
    except StudioColorError as exc:
        return {"ok": False, "reason": str(exc)}
    try:
        from PIL import Image
        with Image.open(str(src)) as im:  # still image fast path
            return scopes(im.convert("RGB"))
    except Exception as _exc:
        record_failure("studio_color.scopes_for_media", _exc)
    arr, reason = _grab_frame(str(src), at)
    if arr is None:
        return {"ok": False, "reason": reason}
    return scopes(arr)


# ── frame histogram (legacy helper, kept for auto_grade/match) ─────

def _frame_histogram(src: str, at: float = 1.0) -> list[list[int]] | None:
    """Per-channel 64-bin histogram of one frame. Needs numpy + PIL."""
    try:
        _np()
    except StudioColorError:
        return None
    arr, _reason = _grab_frame(src, at)
    if arr is None:
        return None
    np = _np()
    img = (arr * 255.0).astype(np.uint8)
    hists = []
    for ch in range(3):
        h, _ = np.histogram(img[:, :, ch], bins=64, range=(0, 256))
        hists.append(h.tolist())
    return hists


# ── LUT application to media (ffmpeg lut3d over the parsed LUT) ────

def _apply_lut_file(src: str, lut_file: Path, out_p: Path,
                    strength: float) -> dict[str, Any]:
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    vf = f"lut3d='{lut_file}'"
    if strength < 1.0:
        vf = (f"split[a][b];[a]lut3d='{lut_file}'[g];"
              f"[b][g]blend=all_mode=normal:all_opacity={strength}")
    proc = _run([ff, "-hide_banner", "-y", "-i", str(src), "-vf", vf,
                 "-c:a", "copy", str(out_p)])
    if proc.returncode != 0 or not out_p.exists():
        return {"ok": False, "reason": proc.stderr[-500:]}
    return {"ok": True, "path": str(out_p)}


def apply_lut(src: str | os.PathLike[str], lut_path: str | os.PathLike[str], *,
              out: str | os.PathLike[str] | None = None,
              strength: float = 1.0) -> dict[str, Any]:
    """Apply a 3D .cube LUT to media. The LUT is parsed and validated
    by the real engine first — a malformed .cube fails here with the
    parser's reason, not as opaque ffmpeg stderr — then applied via
    ffmpeg lut3d (the canonical render path)."""
    lut = Path(lut_path)
    if not lut.exists() or lut.suffix.lower() != ".cube":
        return {"ok": False, "reason": f"not a .cube LUT: {lut_path}"}
    try:
        parsed = CubeLUT.load(lut)
    except StudioColorError as exc:
        return {"ok": False, "reason": f"invalid .cube LUT: {exc}"}
    out_p = Path(out) if out else Path(tempfile.mkdtemp()) / "graded.mp4"
    r = _apply_lut_file(str(src), lut, out_p, strength)
    if r.get("ok"):
        r["lut"] = str(lut)
        r["strength"] = strength
        r["lut_info"] = {"size": parsed.size, "title": parsed.title,
                         "has_1d": parsed.lut1d is not None}
    return r


def list_luts(search_dirs: list[str] | None = None) -> list[str]:
    """Find .cube LUTs in common locations."""
    dirs = search_dirs or [
        str(Path.home() / ".devon" / "luts"),
        "/usr/share/devon/luts",
        "assets/luts",
    ]
    found: list[str] = []
    for d in dirs:
        p = Path(d)
        if p.is_dir():
            found.extend(str(f) for f in p.glob("*.cube"))
    return sorted(found)


# ── curves ─────────────────────────────────────────────────────────

def apply_curves(src: str | os.PathLike[str], *,
                 master: str | None = None,
                 red: str | None = None,
                 green: str | None = None,
                 blue: str | None = None,
                 out: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """RGB curves via ffmpeg. Points like '0/0 0.5/0.6 1/1'.

    Real curves — the same math as Resolve/Premiere. (The same point
    specs also bake into LUTs via :func:`bake_grade_lut`.)"""
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    parts: list[str] = []
    for name, pts in (("m", master), ("r", red), ("g", green), ("b", blue)):
        if pts:
            # validate shape loosely
            for tok in pts.split():
                if "/" not in tok:
                    return {"ok": False,
                            "reason": f"bad curve point {tok!r} in {name}"}
            parts.append(f"{name}='{pts}'")
    if not parts:
        return {"ok": False, "reason": "no curves given"}
    out_p = Path(out) if out else Path(tempfile.mkdtemp()) / "curved.mp4"
    proc = _run([ff, "-hide_banner", "-y", "-i", str(src),
                 "-vf", f"curves={':'.join(parts)}",
                 "-c:a", "copy", str(out_p)])
    if proc.returncode != 0 or not out_p.exists():
        return {"ok": False, "reason": proc.stderr[-500:]}
    return {"ok": True, "path": str(out_p), "curves": parts}


# ── color wheels = true ASC-CDL ────────────────────────────────────

def color_wheels(src: str | os.PathLike[str], *,
                 lift: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 gamma: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 out: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Lift/gamma/gain wheels — now a true ASC-CDL (slope=gain,
    offset=lift, power=gamma), baked to a .cube and applied through
    the same lut3d path as every other grade. Replaces the old
    colorbalance approximation."""
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    try:
        cdl = cdl_from_wheels(lift, gamma, gain)
    except StudioColorError as exc:
        return {"ok": False, "reason": str(exc)}
    out_p = Path(out) if out else Path(tempfile.mkdtemp()) / "wheeled.mp4"
    with tempfile.TemporaryDirectory() as td:
        cube = bake_grade_lut(32, cdl=cdl, title="wheels").write(
            Path(td) / "wheels.cube")
        r = _apply_lut_file(str(src), cube, out_p, 1.0)
    if not r.get("ok"):
        return r
    return {"ok": True, "path": str(out_p),
            "lift": lift, "gamma": gamma, "gain": gain,
            "cdl": cdl.to_dict(), "engine": "asc-cdl"}


def apply_cdl(src: str | os.PathLike[str], *,
              slope: tuple[float, float, float] = (1.0, 1.0, 1.0),
              offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
              power: tuple[float, float, float] = (1.0, 1.0, 1.0),
              saturation: float = 1.0,
              out: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Apply a full ASC-CDL decision (SOP + saturation) to media."""
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    try:
        cdl = CDL(slope=slope, offset=offset, power=power,
                  saturation=saturation)
    except StudioColorError as exc:
        return {"ok": False, "reason": str(exc)}
    out_p = Path(out) if out else Path(tempfile.mkdtemp()) / "cdl.mp4"
    with tempfile.TemporaryDirectory() as td:
        cube = bake_grade_lut(32, cdl=cdl, title="cdl").write(
            Path(td) / "cdl.cube")
        r = _apply_lut_file(str(src), cube, out_p, 1.0)
    if not r.get("ok"):
        return r
    return {"ok": True, "path": str(out_p), "cdl": cdl.to_dict(),
            "engine": "asc-cdl"}


# ── auto-grade + shot matching ─────────────────────────────────────

def auto_grade(src: str | os.PathLike[str], *,
               out: str | os.PathLike[str] | None = None,
               target_luma: float = 0.5) -> dict[str, Any]:
    """One-click balance: auto white-balance + exposure normalize.

    Measures the frame, computes correction, applies it. Real analysis,
    not a fixed preset.
    """
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    src = str(src)
    hist = _frame_histogram(src)
    if hist is None:
        # fallback: ffmpeg's own auto levels
        out_p = Path(out) if out else Path(tempfile.mkdtemp()) / "autograde.mp4"
        proc = _run([ff, "-hide_banner", "-y", "-i", src,
                     "-vf", "whitebalance=auto,eq=brightness=0:saturation=1.1",
                     "-c:a", "copy", str(out_p)])
        if proc.returncode != 0 or not out_p.exists():
            return {"ok": False, "reason": proc.stderr[-500:]}
        return {"ok": True, "path": str(out_p), "method": "ffmpeg-auto"}
    import statistics
    # mean per channel → gray-world white balance gains
    means = [statistics.mean(h[i] * i for i in range(64)) / max(1, sum(h))
             for h in hist]
    gmean = statistics.mean(means) or 1.0
    gains = [gmean / (m or 1.0) for m in means]
    gains = [min(2.0, max(0.5, g)) for g in gains]
    # exposure: shift mean luma toward target
    luma = 0.299 * means[0] + 0.587 * means[1] + 0.114 * means[2]
    exposure = (target_luma - luma / 255.0) * 0.8
    exposure = max(-0.3, min(0.3, exposure))
    vf = (f"colorbalance=rs={gains[0] - 1:.3f}:gs={gains[1] - 1:.3f}:"
          f"bs={gains[2] - 1:.3f},eq=brightness={exposure:.3f}")
    out_p = Path(out) if out else Path(tempfile.mkdtemp()) / "autograde.mp4"
    proc = _run([ff, "-hide_banner", "-y", "-i", src, "-vf", vf,
                 "-c:a", "copy", str(out_p)])
    if proc.returncode != 0 or not out_p.exists():
        return {"ok": False, "reason": proc.stderr[-500:]}
    return {"ok": True, "path": str(out_p), "method": "gray-world",
            "gains": [round(g, 3) for g in gains],
            "exposure": round(exposure, 3)}


def cdf_match_tables(src_arr: Any, ref_arr: Any) -> list[list[float]]:
    """Per-channel CDF-matching tables (256 floats each): applying
    these to the source makes its per-channel CDFs ≈ the reference's —
    full histogram transfer, not a median-shift approximation."""
    np = _np()
    src8 = np.rint(np.clip(src_arr, 0.0, 1.0) * 255.0).astype(np.int64)
    ref8 = np.rint(np.clip(ref_arr, 0.0, 1.0) * 255.0).astype(np.int64)
    tables: list[list[float]] = []
    for c in range(3):
        hs = np.bincount(src8[..., c].ravel(), minlength=256).astype(np.float64)
        hr = np.bincount(ref8[..., c].ravel(), minlength=256).astype(np.float64)
        cs = np.cumsum(hs) / max(1.0, hs.sum())
        cr = np.cumsum(hr) / max(1.0, hr.sum())
        idx = np.searchsorted(cr, cs, side="left")
        tables.append([float(min(255, int(i))) / 255.0 for i in idx])
    return tables


def match_shot(src: str | os.PathLike[str],
               reference: str | os.PathLike[str], *,
               out: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Match src's color to a reference frame via histogram transfer.

    Full per-channel CDF matching, baked to a LUT and applied through
    lut3d — two angles, one look.
    """
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    try:
        _np()
    except StudioColorError:
        return {"ok": False,
                "reason": "need numpy+PIL for histogram matching"}
    a_src, reason = _grab_frame(str(src))
    a_ref, reason_ref = _grab_frame(str(reference))
    if a_src is None or a_ref is None:
        return {"ok": False,
                "reason": (reason if a_src is None else reason_ref) or
                          "need numpy+PIL for histogram matching"}
    tables = cdf_match_tables(a_src, a_ref)
    lut = CubeLUT.from_tables(tables, size=32)
    out_p = Path(out) if out else Path(tempfile.mkdtemp()) / "matched.mp4"
    with tempfile.TemporaryDirectory() as td:
        cube = lut.write(Path(td) / "match.cube")
        r = _apply_lut_file(str(src), cube, out_p, 1.0)
    if not r.get("ok"):
        return r
    return {"ok": True, "path": str(out_p), "method": "histogram-transfer"}


# ── looks (presets — styles live here, engines stay agnostic) ───────

LOOKS: dict[str, dict[str, Any]] = {
    "cinematic": {"curves": {"m": "0/0 0.5/0.45 1/0.95"},
                  "wheels": {"lift": (-0.03, -0.02, 0.0),
                             "gain": (1.05, 1.0, 0.95)}},
    "teal_orange": {"wheels": {"lift": (0.0, 0.02, 0.05),
                               "gain": (1.08, 0.98, 0.9)}},
    "noir": {"curves": {"m": "0/0.05 0.5/0.45 1/0.9"}},
    "vibrant": {"curves": {"m": "0/0 0.5/0.55 1/1"}},
    "phonk": {"wheels": {"lift": (0.02, 0.0, 0.05),
                         "gain": (1.1, 0.95, 1.05)}},
}


def apply_look(src: str | os.PathLike[str], look: str, *,
               out: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Apply a named look. Preset composes primitives — engines untouched."""
    spec = LOOKS.get(look)
    if spec is None:
        return {"ok": False,
                "reason": f"unknown look {look!r}; pick from {sorted(LOOKS)}"}
    cur = str(src)
    if "curves" in spec:
        c = spec["curves"]
        r = apply_curves(cur, master=c.get("m"), red=c.get("r"),
                         green=c.get("g"), blue=c.get("b"))
        if not r.get("ok"):
            return r
        cur = r["path"]
    if "wheels" in spec:
        w = spec["wheels"]
        r = color_wheels(cur, lift=w.get("lift", (0, 0, 0)),
                         gamma=w.get("gamma", (1, 1, 1)),
                         gain=w.get("gain", (1, 1, 1)))
        if not r.get("ok"):
            return r
        cur = r["path"]
    if out and cur != str(out):
        os.replace(cur, out)
        cur = str(out)
    return {"ok": True, "path": cur, "look": look}


def list_looks() -> list[str]:
    return sorted(LOOKS)
