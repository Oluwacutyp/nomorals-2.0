"""Temporal-consistency pass for chained clips.

Neural clips are generated independently: even with a shared style
suffix, the last frame of clip A and the first frame of clip B differ
in white balance, exposure and texture — the cut pops. This module
measures the boundary mismatch and grades the opening frames of each
later clip toward the previous clip's closing frame (Reinhard color
transfer with a decaying weight), so the cut reads as one scene.

Color + structure are both measured on the boundary frames:

- color_shift: per-channel mean/std distance (the white-balance pop)
- hist_corr: per-channel histogram correlation (the palette pop)
- edge_sim: gradient-magnitude correlation (the texture/structure pop)

The combined score is 0..1 (1 = seamless). The pass reports before/after
scores per boundary; it never touches audio, duration or the rest of
the clip.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

PIL_AVAILABLE = Image is not None


def _require_pil() -> None:
    if Image is None:
        raise RuntimeError(
            "nomorals.media.videogen.consistency is unavailable: Pillow is not installed "
            "(pip install Pillow)")

import importlib.util as _importlib_util

try:
    NUMPY_AVAILABLE = _importlib_util.find_spec("numpy") is not None
except ImportError:
    # a meta-path blocker (or a broken install) raises instead of None
    NUMPY_AVAILABLE = False


def _require_numpy() -> None:
    if not NUMPY_AVAILABLE:
        raise RuntimeError(
            "nomorals.media.videogen.consistency is unavailable: numpy is not installed "
            "(pip install numpy)")


from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "boundary_frames",
    "boundary_metric",
    "reinhard_match",
    "consistency_pass",
    "CONSISTENCY_HEAD_FRAMES",
]

#: how many opening frames of each later clip get graded toward the
#: previous clip's closing frame
CONSISTENCY_HEAD_FRAMES = 8


def _ffmpeg_bin() -> str:
    from shutil import which
    ff = which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found — consistency pass needs it")
    return ff


def _grab_frame(path: str | os.PathLike, t: float, *,
                width: int, out: str) -> str:
    """Single frame at timestamp t as PNG (extract_frames writes jpg,
    which some ffmpeg builds reject at encode time)."""
    import subprocess
    subprocess.run(
        [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(path), "-ss", f"{max(0.0, t):.3f}",
         "-frames:v", "1", "-vf", f"scale={width}:-2", out],
        check=True, capture_output=True, timeout=120)
    return out


def boundary_frames(a_path: str | os.PathLike,
                    b_path: str | os.PathLike,
                    *, width: int = 320) -> tuple[Image.Image, Image.Image]:
    """Last frame of clip A and first frame of clip B (downscaled for
    the metric — grading itself runs at full resolution)."""
    from ...media_edit.videos import video_probe
    from ..motion_studio._core import probe_duration
    wd = Path(tempfile.mkdtemp(prefix="consistency_"))
    dur_a = max(0.1, probe_duration(a_path))
    try:
        fps_a = float(video_probe(a_path).get("fps") or 24.0)
    except Exception:  # noqa: BLE001 - probe is best-effort
        fps_a = 24.0
    # stay a frame or two inside the end: the exact last timestamp has
    # no decodable frame on short clips
    t_last = max(0.0, dur_a - 2.0 / max(1.0, fps_a))
    fa = _grab_frame(a_path, t_last, width=width, out=str(wd / "a.png"))
    fb = _grab_frame(b_path, 0.0, width=width, out=str(wd / "b.png"))
    _require_pil()
    return (Image.open(fa).convert("RGB"), Image.open(fb).convert("RGB"))


def _hist_corr(a: "np.ndarray", b: "np.ndarray") -> float:
    """Per-channel histogram correlation, averaged (0..1)."""
    _require_numpy()
    import numpy as np
    corrs = []
    for c in range(3):
        ha, _ = np.histogram(a[..., c], bins=32, range=(0, 255))
        hb, _ = np.histogram(b[..., c], bins=32, range=(0, 255))
        ha = ha.astype(np.float64)
        hb = hb.astype(np.float64)
        denom = ha.std() * hb.std()
        if denom < 1e-9:
            corrs.append(1.0 if ha.std() < 1e-9 and hb.std() < 1e-9 else 0.0)
        else:
            corrs.append(float(((ha - ha.mean()) * (hb - hb.mean())
                                ).mean() / denom))
    return float(np.clip(np.mean(corrs), 0.0, 1.0))


def _edge_sim(a: "np.ndarray", b: "np.ndarray") -> float:
    """Gradient-magnitude correlation — do the edges line up?

    Sobel via numpy slicing (no scipy): keeps the metric on the
    dependency-free spine.
    """
    _require_numpy()
    import numpy as np
    _require_pil()
    from PIL import Image as _I

    def grad_mag(g: "np.ndarray") -> "np.ndarray":
        _require_numpy()
        p = np.pad(g, 1, mode="edge")
        gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]
              - p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2])
        gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
              - p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:])
        return np.hypot(gx, gy)

    ga = np.asarray(_I.fromarray(a).convert("L"), dtype=np.float64)
    gb = np.asarray(_I.fromarray(b).convert("L"), dtype=np.float64)
    ma = grad_mag(ga).ravel()
    mb = grad_mag(gb).ravel()
    denom = ma.std() * mb.std()
    if denom < 1e-9:
        return 1.0
    return float(np.clip(((ma - ma.mean()) * (mb - mb.mean())
                          ).mean() / denom, 0.0, 1.0))


def boundary_metric(a: Image.Image, b: Image.Image) -> dict[str, float]:
    """Color + structure mismatch between two boundary frames.

    Returns ``{"color_shift", "hist_corr", "edge_sim", "score"}`` —
    ``score`` 0..1, 1 = seamless.
    """
    _require_numpy()
    import numpy as np
    size = (160, 160)
    aa = np.asarray(a.resize(size).convert("RGB"), dtype=np.float64)
    bb = np.asarray(b.resize(size).convert("RGB"), dtype=np.float64)
    # color: normalized per-channel mean + std distance
    mean_d = np.abs(aa.mean(axis=(0, 1)) - bb.mean(axis=(0, 1))).mean() / 255.0
    std_d = np.abs(aa.std(axis=(0, 1)) - bb.std(axis=(0, 1))).mean() / 128.0
    color_shift = float(np.clip(0.6 * mean_d + 0.4 * std_d, 0.0, 1.0))
    hist = _hist_corr(aa, bb)
    try:
        edge = _edge_sim(aa, bb)
    except Exception:  # noqa: BLE001 - scipy missing: degrade the metric,
        # not the pass (histogram + color still carry the signal)
        edge = hist
    score = float(np.clip(0.4 * hist + 0.3 * edge +
                          0.3 * (1.0 - color_shift), 0.0, 1.0))
    return {"color_shift": color_shift, "hist_corr": hist,
            "edge_sim": edge, "score": score}


def reinhard_match(src: Image.Image, ref: Image.Image,
                   *, strength: float = 1.0) -> Image.Image:
    """Reinhard color transfer: match ``src``'s per-channel mean/std to
    ``ref``'s. ``strength`` 0..1 blends between original and matched."""
    _require_numpy()
    import numpy as np
    s = np.clip(float(strength), 0.0, 1.0)
    if s <= 0.0:
        return src.copy()
    sa = np.asarray(src.convert("RGB"), dtype=np.float64)
    ra = np.asarray(ref.convert("RGB"), dtype=np.float64)
    out = np.empty_like(sa)
    for c in range(3):
        sm, ss = sa[..., c].mean(), sa[..., c].std()
        rm, rs = ra[..., c].mean(), ra[..., c].std()
        ch = sa[..., c] - sm
        if ss > 1e-6:
            ch = ch * (rs / ss)
        out[..., c] = ch + (rm * s + sm * (1.0 - s))
    # when s < 1 the matched frame is blended back toward the original
    matched = np.clip(out, 0, 255).astype(np.uint8)
    _require_pil()
    if s >= 1.0:
        return Image.fromarray(matched)
    orig = np.asarray(src.convert("RGB"), dtype=np.float64)
    return Image.fromarray(
        np.clip(orig * (1.0 - s) + matched.astype(np.float64) * s,
                0, 255).astype(np.uint8))


def _grade_head(clip: str, ref_frame: Image.Image, *,
                head_frames: int, out_path: str) -> None:
    """Re-encode ``clip`` with its first ``head_frames`` frames graded
    toward ``ref_frame`` (decaying Reinhard strength). Video-only —
    chained clips carry no audio at this stage."""
    from ...media_edit.videos import video_probe
    ff = _ffmpeg_bin()
    wd = Path(tempfile.mkdtemp(prefix="chead_"))
    info = video_probe(clip)
    fps = float(info.get("fps") or 24.0)
    # full clip → PNG frames
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(clip), str(wd / "f_%04d.png")],
        check=True, capture_output=True, timeout=300)
    frames = sorted(wd.glob("f_*.png"))
    if not frames:
        raise RuntimeError(f"no frames decoded from {clip}")
    n_grade = min(head_frames, len(frames))
    _require_pil()
    for j in range(n_grade):
        strength = 1.0 - j / max(1, n_grade)
        img = Image.open(frames[j]).convert("RGB")
        reinhard_match(img, ref_frame, strength=strength).save(frames[j])
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y",
         "-framerate", f"{fps:.4f}", "-i", str(wd / "f_%04d.png"),
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
         str(out_path)],
        check=True, capture_output=True, timeout=600)


def consistency_pass(clips: list[str], *,
                     head_frames: int = CONSISTENCY_HEAD_FRAMES,
                     out_dir: str | os.PathLike | None = None
                     ) -> tuple[list[str], dict]:
    """Grade each clip's opening toward the previous clip's close.

    Returns (adjusted_paths, report). ``report["boundaries"]`` holds
    per-cut ``{"a", "b", "score_before", "score_after"}``. A clip that
    fails to grade keeps its original path and is flagged in the
    report — the pass never kills a chain.
    """
    if len(clips) < 2:
        return list(clips), {"boundaries": [], "skipped": "need ≥2 clips"}
    wd = Path(out_dir) if out_dir else \
        Path(tempfile.mkdtemp(prefix="consistency_"))
    wd.mkdir(parents=True, exist_ok=True)
    adjusted = [clips[0]]
    boundaries: list[dict] = []
    _require_pil()
    for i in range(1, len(clips)):
        entry: dict = {"a": i - 1, "b": i}
        try:
            a_last, b_first = boundary_frames(adjusted[-1], clips[i])
            before = boundary_metric(a_last, b_first)["score"]
            entry["score_before"] = round(before, 3)
            # full-res reference for grading
            from ...media_edit.videos import video_probe
            from ..motion_studio._core import probe_duration
            tmp = Path(tempfile.mkdtemp(prefix="cref_"))
            dur = max(0.1, probe_duration(adjusted[-1]))
            try:
                fps_ref = float(video_probe(adjusted[-1]).get("fps") or 24.0)
            except Exception:  # noqa: BLE001
                fps_ref = 24.0
            ref_path = _grab_frame(
                adjusted[-1], max(0.0, dur - 2.0 / max(1.0, fps_ref)),
                width=640, out=str(tmp / "ref.png"))
            ref = Image.open(ref_path).convert("RGB")
            out_p = wd / f"clip{i:02d}-consistent.mp4"
            _grade_head(clips[i], ref, head_frames=head_frames,
                        out_path=str(out_p))
            adjusted.append(str(out_p))
            # after-score on the graded clip's new first frame
            _, b_first_new = boundary_frames(adjusted[-2], adjusted[-1])
            entry["score_after"] = round(
                boundary_metric(a_last, b_first_new)["score"], 3)
        except Exception as exc:  # noqa: BLE001 - never kill the chain
            _log.warning("consistency: boundary %d→%d skipped: %s",
                         i - 1, i, exc)
            entry["error"] = str(exc)[:120]
            adjusted.append(clips[i])
        boundaries.append(entry)
    return adjusted, {"boundaries": boundaries,
                      "head_frames": head_frames}


# ---------------------------------------------------------------------------
# Project-wide identity + look lock (EDIT_GEN_UPGRADE 2.5)
# ---------------------------------------------------------------------------
# Identity lock: face + wardrobe + palette per character, carried across
# every shot and every backend. Look lock: grade + palette + grain +
# lens as a NAMED preset (data, not code — the look lives in the
# document, the engine just applies it). continuity_check() flags
# mismatches; it never ships a mismatched shot (see
# directed.validate.validate_identity_continuity).

__all__ += [
    "IdentityLock",
    "LookLock",
    "palette_distance",
    "embedding_similarity",
    "check_identity",
    "check_look",
    "continuity_check",
]

#: identity similarity below this → mismatch (flag, not ship).
IDENTITY_MIN_SIM = 0.75
#: look palette distance above this → mismatch.
LOOK_MAX_PALETTE_DIST = 60.0


def palette_distance(a: list, b: list) -> float:
    """Mean per-colour Euclidean distance between two palettes.

    Palettes are ``[[r, g, b], ...]``; compared up to the shorter
    length. ``0`` = identical. Pure — no image I/O.
    """
    import math as _math
    if not a or not b:
        return float("inf")
    n = min(len(a), len(b))
    tot = 0.0
    for i in range(n):
        tot += _math.sqrt(sum((float(x) - float(y)) ** 2
                              for x, y in zip(a[i], b[i])))
    return tot / n


def embedding_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embeddings (0..1, 1 = identical)."""
    import math as _math
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(float(x) * float(y) for x, y in zip(a[:n], b[:n]))
    na = _math.sqrt(sum(float(x) ** 2 for x in a[:n])) or 1e-9
    nb = _math.sqrt(sum(float(y) ** 2 for y in b[:n])) or 1e-9
    return max(0.0, min(1.0, dot / (na * nb)))


from dataclasses import dataclass as _dataclass, field as _field  # noqa: E402


@_dataclass
class IdentityLock:
    """One character's identity lock: face + wardrobe + palette.

    ``face_embedding``: face-recognition vector for the character
    (mean over reference frames). ``wardrobe``: free-text wardrobe
    description (e.g. "red jacket, black jeans"). ``palette``: the
    character's reference palette ``[[r,g,b], ...]``.
    """
    character_id: str
    face_embedding: list[float] = _field(default_factory=list)
    wardrobe: str = ""
    palette: list = _field(default_factory=list)

    def to_dict(self) -> dict:
        return {"character_id": self.character_id,
                "face_embedding": list(self.face_embedding),
                "wardrobe": self.wardrobe,
                "palette": [list(map(float, px)) for px in self.palette]}

    @classmethod
    def from_dict(cls, data: dict) -> "IdentityLock":
        return cls(character_id=str(data.get("character_id", "")),
                   face_embedding=[float(v) for v in
                                   data.get("face_embedding", [])],
                   wardrobe=str(data.get("wardrobe", "")),
                   palette=[list(map(float, px))
                            for px in data.get("palette", [])])


@_dataclass
class LookLock:
    """One named look: grade + palette + grain + lens (data, not code)."""
    name: str
    grade: str = ""
    palette: list = _field(default_factory=list)
    grain: float = 0.0
    lens: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "grade": self.grade,
                "palette": [list(map(float, px)) for px in self.palette],
                "grain": float(self.grain), "lens": self.lens}

    @classmethod
    def from_dict(cls, data: dict) -> "LookLock":
        return cls(name=str(data.get("name", "")),
                   grade=str(data.get("grade", "")),
                   palette=[list(map(float, px))
                            for px in data.get("palette", [])],
                   grain=float(data.get("grain", 0.0)),
                   lens=str(data.get("lens", "")))


def check_identity(lock: IdentityLock, shot: dict) -> dict:
    """Shot vs one identity lock → ``{similarity, palette_dist,
    wardrobe_match, ok, issues}``.

    ``shot``: ``{"face_embedding", "palette", "wardrobe"}`` — any
    missing field is reported as an unmeasured issue, never assumed
    matching.
    """
    issues: list[str] = []
    sim: float | None = None
    if shot.get("face_embedding") is not None and lock.face_embedding:
        sim = embedding_similarity(lock.face_embedding,
                                   shot["face_embedding"])
        if sim < IDENTITY_MIN_SIM:
            issues.append(
                f"face similarity {sim:.2f} < {IDENTITY_MIN_SIM} "
                f"for {lock.character_id!r}")
    elif lock.face_embedding:
        issues.append(f"no face embedding measured for "
                      f"{lock.character_id!r}")
    pdist: float | None = None
    if shot.get("palette") and lock.palette:
        pdist = palette_distance(lock.palette, shot["palette"])
        if pdist > LOOK_MAX_PALETTE_DIST:
            issues.append(f"palette distance {pdist:.1f} > "
                          f"{LOOK_MAX_PALETTE_DIST}")
    wardrobe_match: bool | None = None
    if lock.wardrobe and shot.get("wardrobe") is not None:
        wardrobe_match = (str(shot["wardrobe"]).strip().lower()
                          == lock.wardrobe.strip().lower())
        if not wardrobe_match:
            issues.append(f"wardrobe {shot['wardrobe']!r} != locked "
                          f"{lock.wardrobe!r}")
    return {"character_id": lock.character_id, "similarity": sim,
            "palette_dist": pdist, "wardrobe_match": wardrobe_match,
            "ok": not issues, "issues": issues}


def check_look(lock: LookLock, shot: dict) -> dict:
    """Shot vs the look lock → ``{palette_dist, grade_match, ok,
    issues}``. ``shot``: ``{"grade", "palette", "grain", "lens"}``."""
    issues: list[str] = []
    pdist: float | None = None
    if shot.get("palette") and lock.palette:
        pdist = palette_distance(lock.palette, shot["palette"])
        if pdist > LOOK_MAX_PALETTE_DIST:
            issues.append(f"look palette distance {pdist:.1f} > "
                          f"{LOOK_MAX_PALETTE_DIST}")
    grade_match: bool | None = None
    if lock.grade and shot.get("grade") is not None:
        grade_match = str(shot["grade"]).strip().lower() == \
            lock.grade.strip().lower()
        if not grade_match:
            issues.append(f"grade {shot['grade']!r} != locked "
                          f"{lock.grade!r} ({lock.name!r})")
    return {"look": lock.name, "palette_dist": pdist,
            "grade_match": grade_match, "ok": not issues,
            "issues": issues}


def continuity_check(shots: list[dict], *,
                     identity_locks: list[IdentityLock | dict] | None = None,
                     look_lock: LookLock | dict | None = None) -> dict:
    """Project-wide continuity: every shot vs every lock.

    ``shots``: ``[{"id", "character_id", "face_embedding", "palette",
    "wardrobe", "grade", ...}]``. Returns ``{"ok", "mismatches",
    "checked"}`` — mismatches flag shots that must NOT ship until
    re-locked or re-generated. Pure: no image I/O.
    """
    locks = [lk if isinstance(lk, IdentityLock)
             else IdentityLock.from_dict(dict(lk))
             for lk in (identity_locks or [])]
    look = (look_lock if isinstance(look_lock, LookLock)
            else LookLock.from_dict(dict(look_lock))) \
        if look_lock else None
    mismatches: list[dict] = []
    checked = 0
    for shot in shots:
        checked += 1
        sid = shot.get("id", checked)
        for lk in locks:
            if shot.get("character_id") and shot["character_id"] != \
                    lk.character_id:
                continue
            res = check_identity(lk, shot)
            if not res["ok"]:
                mismatches.append({"shot": sid, "kind": "identity",
                                   **res})
        if look is not None:
            res = check_look(look, shot)
            if not res["ok"]:
                mismatches.append({"shot": sid, "kind": "look", **res})
    return {"ok": not mismatches, "mismatches": mismatches,
            "checked": checked}
