"""Devon Studio — professional color grading.

LUTs (3D .cube), curves, color wheels (lift/gamma/gain), auto-grade,
and shot matching via histogram transfer. Real ffmpeg/OpenCV ops.

Style-agnostic: these are primitives. Looks live in presets.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _ffmpeg() -> str | None:
    from shutil import which
    return which("ffmpeg") or which("/usr/bin/ffmpeg")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603


# ── LUTs ───────────────────────────────────────────────────────────

def apply_lut(src: str | os.PathLike[str], lut_path: str | os.PathLike[str], *,
              out: str | os.PathLike[str] | None = None,
              strength: float = 1.0) -> dict[str, Any]:
    """Apply a 3D .cube LUT via ffmpeg lut3d. Real LUT pipeline."""
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    lut = Path(lut_path)
    if not lut.exists() or lut.suffix.lower() != ".cube":
        return {"ok": False, "reason": f"not a .cube LUT: {lut_path}"}
    out_p = Path(out) if out else Path(tempfile.mkdtemp()) / "graded.mp4"
    vf = f"lut3d='{lut}'"
    if strength < 1.0:
        # blend graded with original
        vf = (f"split[a][b];[a]lut3d='{lut}'[g];"
              f"[b][g]blend=all_mode=normal:all_opacity={strength}")
    proc = _run([ff, "-hide_banner", "-y", "-i", str(src), "-vf", vf,
                 "-c:a", "copy", str(out_p)])
    if proc.returncode != 0 or not out_p.exists():
        return {"ok": False, "reason": proc.stderr[-500:]}
    return {"ok": True, "path": str(out_p), "lut": str(lut),
            "strength": strength}


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

    Real curves — the same math as Resolve/Premiere.
    """
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


# ── color wheels: lift / gamma / gain ───────────────────────────────

def color_wheels(src: str | os.PathLike[str], *,
                 lift: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 gamma: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 out: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """ASC-CDL style wheels via ffmpeg colorbalance + eq.

    lift shifts shadows, gamma the mids, gain the highlights — per channel.
    """
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    # colorbalance handles shadows/mids/highlights per channel
    cb = (f"colorbalance=rs={lift[0]:.3f}:gs={lift[1]:.3f}:bs={lift[2]:.3f}:"
          f"rm={gamma[0] - 1.0:.3f}:gm={gamma[1] - 1.0:.3f}:bm={gamma[2] - 1.0:.3f}:"
          f"rh={gain[0] - 1.0:.3f}:gh={gain[1] - 1.0:.3f}:bh={gain[2] - 1.0:.3f}")
    out_p = Path(out) if out else Path(tempfile.mkdtemp()) / "wheeled.mp4"
    proc = _run([ff, "-hide_banner", "-y", "-i", str(src),
                 "-vf", cb, "-c:a", "copy", str(out_p)])
    if proc.returncode != 0 or not out_p.exists():
        return {"ok": False, "reason": proc.stderr[-500:]}
    return {"ok": True, "path": str(out_p),
            "lift": lift, "gamma": gamma, "gain": gain}


# ── auto-grade + shot matching ─────────────────────────────────────

def _frame_histogram(src: str, at: float = 1.0) -> list[list[int]] | None:
    """Per-channel 64-bin histogram of one frame. Needs numpy + cv2/PIL."""
    try:
        import numpy as np
    except ImportError:
        return None
    ff = _ffmpeg()
    if not ff:
        return None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        frame = tf.name
    proc = _run([ff, "-hide_banner", "-y", "-ss", str(at), "-i", src,
                 "-frames:v", "1", frame])
    if proc.returncode != 0:
        return None
    try:
        from PIL import Image
        img = np.asarray(Image.open(frame).convert("RGB"))
    except ImportError:
        return None
    finally:
        try:
            os.unlink(frame)
        except OSError:
            pass
    hists = []
    for ch in range(3):
        h, _ = np.histogram(img[:, :, ch], bins=64, range=(0, 256))
        hists.append(h.tolist())
    return hists


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


def match_shot(src: str | os.PathLike[str],
               reference: str | os.PathLike[str], *,
               out: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Match src's color to a reference frame via histogram transfer.

    The money shot-matching primitive: two angles, one look.
    """
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    hs = _frame_histogram(str(src))
    hr = _frame_histogram(str(reference))
    if hs is None or hr is None:
        return {"ok": False,
                "reason": "need numpy+PIL for histogram matching"}
    import statistics
    # per-channel CDF matching → approximate with gain/offset per channel
    vf_parts: list[str] = []
    for ch, (sh, rh) in enumerate(zip(hs, hr)):
        def cdf(h: list[int]) -> list[float]:
            tot = sum(h) or 1
            acc, out_c = 0.0, []
            for v in h:
                acc += v / tot
                out_c.append(acc)
            return out_c
        cs, cr = cdf(sh), cdf(rh)
        # find median shift: where each CDF hits 0.5
        def med(c: list[float]) -> float:
            for i, v in enumerate(c):
                if v >= 0.5:
                    return i / 63.0
            return 0.5
        ms, mr = med(cs), med(cr)
        gain = (mr + 0.05) / (ms + 0.05)
        gain = max(0.5, min(2.0, gain))
        offset = (mr - ms * gain) * 0.5
        ch_name = "rgb"[ch]
        vf_parts.append(f"{ch_name}='{ch_name}*({gain:.3f})+{offset * 255:.1f}'")
    vf = f"geq={':'.join(vf_parts)}"
    out_p = Path(out) if out else Path(tempfile.mkdtemp()) / "matched.mp4"
    proc = _run([ff, "-hide_banner", "-y", "-i", str(src),
                 "-vf", vf, "-c:a", "copy", str(out_p)])
    if proc.returncode != 0 or not out_p.exists():
        return {"ok": False, "reason": proc.stderr[-500:]}
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
