"""Image optimisation: multi-codec trials with measured outcomes (Squoosh).

Google's Squoosh re-encodes locally with real codecs and makes the
trade-off visible — a before/after comparison with live size per codec
and quality setting. Devon's version runs the codec ladder available
on the box (Pillow codecs first; ffmpeg / heif-enc where installed)
over the Studio image-document flow, measures every trial (bytes,
dimensions held, PSNR where actually computed), and recommends the
best trial under the target's constraint — with the full table shown,
never a single default save.

Honesty rules (the ladder's contract):
- a codec rung runs only where the box provides it; absent rungs are
  named in the report and in ``optimize_status()``, never silently
  skipped or faked;
- PSNR is computed from the decoded trial pixels against the source
  pixels (numpy MSE, in-memory); SSIM is NOT claimed — scikit-image
  is not installed, so no SSIM number is ever reported;
- a lossless trial that decodes to identical pixels reports
  ``psnr_db=None`` with the note "identical pixels", never ``inf``;
- target budgets are stated with their basis: the Telegram numbers
  are the Bot API's published ``sendPhoto`` limits (10 MB, width +
  height <= 10000, ratio <= 20); the WhatsApp numbers are the
  working target from the November 2026 write-ups (WhatsApp
  re-encodes every photo sent as media, longest edge ~1600 px
  standard — a ~1 MB / ~1920 px JPEG arrives sharpest), NOT a
  platform cap (hard cap 16 MB as media, 2 GB as a document);
  the ``web`` budget is Devon's own working budget, labelled as
  such;
- alpha is flattened over white for JPEG/AVIF trials (those codecs
  carry no alpha here) and the flattening is noted on the trial;
- a trial that fails (timeout, missing encoder, bad pixels) is
  recorded as failed with its reason — it never raises, and it
  never counts toward the recommendation.
"""

from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from . import ImgGenError

_log = get_logger(__name__)

__all__ = [
    "TARGETS",
    "CODEC_EXT",
    "OptimizeTrial",
    "OptimizeReport",
    "optimize_status",
    "optimize_image",
    "optimize_document",
    "save_optimized",
    "encode_pixels",
    "psnr_db",
]

#: Preferred-codec margin: for a target with a preferred codec, the
#: preferred codec wins when its best trial is within this many dB of
#: the overall best trial under the budget (WhatsApp re-encodes to
#: JPEG on receipt, so a JPEG within 1 dB of the best WebP is the
#: better answer — one conversion, not two).
PREFERRED_CODEC_MARGIN_DB = 1.0

#: PSNR floor for the ``smallest`` target: the smallest file whose
#: measured PSNR stays at or above this is the recommendation.
SMALLEST_PSNR_FLOOR_DB = 30.0

#: ffmpeg wall-clock budget per trial (libaom AVIF is slow on big
#: frames; a trial that exceeds it is recorded as failed, never hung).
FFMPEG_TIMEOUT_S = 90.0

CODEC_EXT = {"jpeg": ".jpg", "webp": ".webp", "png": ".png",
             "avif": ".avif"}
CODEC_FORMAT = {"jpeg": "JPEG", "webp": "WEBP", "png": "PNG",
                "avif": "AVIF"}
CODECS = ("jpeg", "webp", "png", "avif")

#: Named targets: byte budget (0 = no cap), longest-edge cap (0 =
#: none), dimension-sum cap (0 = none), preferred codec, and the
#: basis for the numbers — stated on every answer, never implied.
TARGETS: dict[str, dict[str, Any]] = {
    "whatsapp": {
        "max_bytes": 1_000_000,
        "max_long_edge": 1920,
        "max_dim_sum": 0,
        "preferred_codec": "jpeg",
        "basis": ("WhatsApp re-encodes every photo sent as media "
                  "(longest edge ~1600 px standard, JPEG); a ~1 MB / "
                  "~1920 px JPEG arrives sharpest — a working target "
                  "from the November 2026 write-ups, not a platform "
                  "cap (hard cap 16 MB as media, 2 GB as a document)"),
    },
    "telegram": {
        "max_bytes": 10_000_000,
        "max_long_edge": 0,
        "max_dim_sum": 10000,
        "preferred_codec": "jpeg",
        "basis": ("Telegram Bot API sendPhoto: photo at most 10 MB, "
                  "width + height <= 10000, ratio <= 20"),
    },
    "web": {
        "max_bytes": 500_000,
        "max_long_edge": 0,
        "max_dim_sum": 0,
        "preferred_codec": "webp",
        "basis": ("Devon's working budget for a web-ready file — no "
                  "platform states this number"),
    },
    "archive": {
        "max_bytes": 0,
        "max_long_edge": 0,
        "max_dim_sum": 0,
        "preferred_codec": None,
        "basis": ("archive keeps every pixel — the smallest lossless "
                  "trial wins"),
    },
    "smallest": {
        "max_bytes": 0,
        "max_long_edge": 0,
        "max_dim_sum": 0,
        "preferred_codec": None,
        "basis": ("smallest file whose measured PSNR stays >= "
                  f"{SMALLEST_PSNR_FLOOR_DB:.0f} dB"),
    },
}


@dataclass
class OptimizeTrial:
    """One codec trial: measured, never asserted."""

    codec: str
    quality: int | None  # None = lossless
    size_bytes: int = 0
    width: int = 0
    height: int = 0
    psnr_db: float | None = None  # None = identical pixels (lossless)
    rung: str = ""
    status: str = "ok"  # ok | failed | unavailable
    reason: str = ""
    path: str = ""  # filled when the trial bytes are written to disk
    _data: bytes | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = {"codec": self.codec, "quality": self.quality,
             "bytes": self.size_bytes, "width": self.width,
             "height": self.height, "psnr_db": self.psnr_db,
             "rung": self.rung, "status": self.status,
             "reason": self.reason, "path": self.path}
        return d


@dataclass
class OptimizeReport:
    """The trial table plus the recommendation."""

    source: str
    source_bytes: int
    source_format: str
    width: int
    height: int
    target: str
    target_note: str
    trials: list = field(default_factory=list)
    recommended: int | None = None  # index into trials
    closest: int | None = None  # smallest ok trial, when none fits
    rungs_available: list = field(default_factory=list)
    rungs_unavailable: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_bytes": self.source_bytes,
            "source_format": self.source_format,
            "width": self.width,
            "height": self.height,
            "target": self.target,
            "target_note": self.target_note,
            "trials": [t.to_dict() for t in self.trials],
            "recommended": (self.trials[self.recommended].to_dict()
                            if self.recommended is not None else None),
            "closest": (self.trials[self.closest].to_dict()
                        if self.closest is not None else None),
            "rungs_available": list(self.rungs_available),
            "rungs_unavailable": list(self.rungs_unavailable),
            "notes": list(self.notes),
        }

    def format_table(self) -> str:
        """The trial table as plain text — sizes are measured bytes."""
        lines = [f"optimize {self.source} ({self.source_bytes} bytes, "
                 f"{self.width}x{self.height} {self.source_format}) "
                 f"for {self.target} — {self.target_note}"]
        for i, t in enumerate(self.trials):
            mark = "★ " if i == self.recommended else "  "
            if t.status != "ok":
                lines.append(f"{mark}{t.codec} "
                             f"{'q' + str(t.quality) if t.quality is not None else 'lossless'}: "
                             f"{t.status} ({t.reason})")
                continue
            psnr = ("identical" if t.psnr_db is None
                    else f"{t.psnr_db:.1f} dB")
            lines.append(f"{mark}{t.codec} "
                         f"{'q' + str(t.quality) if t.quality is not None else 'lossless'}: "
                         f"{t.size_bytes} bytes, {t.width}x{t.height}, "
                         f"psnr {psnr} [{t.rung}]"
                         + (f" — {t.reason}" if t.reason else ""))
        if self.recommended is not None:
            best = self.trials[self.recommended]
            lines.append(f"recommended: {best.codec} "
                         f"{'q' + str(best.quality) if best.quality is not None else 'lossless'} "
                         f"({best.size_bytes} bytes)")
        elif self.closest is not None:
            near = self.trials[self.closest]
            lines.append("no trial fits the target — closest: "
                         f"{near.codec} "
                         f"{'q' + str(near.quality) if near.quality is not None else 'lossless'} "
                         f"({near.size_bytes} bytes)")
        else:
            lines.append("no trial produced a file")
        return "\n".join(lines)


# ── quality measurement ──────────────────────────────────────────────

def psnr_db(reference: Any, decoded: Any) -> float | None:
    """PSNR in dB between two PIL images (same size).

    Returns None when the pixels are identical (a lossless trial) —
    never ``inf``. Both images are flattened over white first, so
    an RGBA trial and its RGB reference compare on the same pixels.
    """
    import numpy as np

    ref = np.asarray(_flatten_white(reference), dtype=np.float64)
    got = np.asarray(_flatten_white(decoded), dtype=np.float64)
    if ref.shape != got.shape:
        raise ImgGenError(
            f"PSNR shape mismatch: reference {ref.shape} vs "
            f"decoded {got.shape}")
    mse = float(np.mean((ref - got) ** 2))
    if mse == 0.0:
        return None
    return float(10.0 * math.log10(255.0 * 255.0 / mse))


def _decode(data: bytes, codec: str = "",
            timeout_s: float = FFMPEG_TIMEOUT_S,
            workdir: str | None = None) -> Any:
    """Decode trial bytes back to pixels. AVIF goes through ffmpeg:
    this Pillow build cannot decode AVIF at all (no plugin), so a
    PIL-only decode would mark every AVIF trial failed."""
    from PIL import Image

    if codec == "avif":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or workdir is None:
            raise ImgGenError(
                "cannot decode an AVIF trial: ffmpeg is not "
                "installed")
        src = os.path.join(workdir, "squoosh-decode.avif")
        out = os.path.join(workdir, "squoosh-decode.png")
        with open(src, "wb") as f:
            f.write(data)
        try:
            proc = subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-i", src, out],
                capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise ImgGenError(
                f"ffmpeg AVIF decode timed out after {timeout_s:g}s"
            ) from exc
        if proc.returncode != 0 or not os.path.exists(out):
            raise ImgGenError(
                "ffmpeg AVIF decode failed: "
                f"{(proc.stderr or '').strip()[:200]}")
        try:
            img = Image.open(out)
            img.load()
            return img
        finally:
            for p in (src, out):
                with suppress(OSError):
                    os.unlink(p)
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


# ── codec probes ─────────────────────────────────────────────────────

def _flatten_white(img: Any) -> Any:
    """RGB pixels with alpha composited over white (JPEG/AVIF carry
    no alpha here — Pillow's plain ``convert("RGB")`` composites
    over black, which would darken every transparent pixel)."""
    from PIL import Image

    if img.mode in ("RGBA", "LA", "PA") or (
            img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        base = Image.new("RGB", img.size, (255, 255, 255))
        base.paste(rgba, mask=rgba.split()[-1])
        return base
    return img.convert("RGB")


def _pillow_encode(img: Any, codec: str,
                   quality: int | None) -> bytes:
    """Encode with Pillow. Raises on an unsupported codec."""
    from PIL import Image

    buf = io.BytesIO()
    if codec == "jpeg":
        work = _flatten_white(img)
        # No optimize=True here on purpose: the trial bytes must be
        # the bytes the Studio save path writes (save_image does not
        # optimise), or the ladder would report a size the saved
        # file never has.
        work.save(buf, "JPEG", quality=quality or 90)
    elif codec == "webp":
        if quality is None:
            img.save(buf, "WEBP", lossless=True)
        else:
            img.save(buf, "WEBP", quality=quality)
    elif codec == "png":
        img.save(buf, "PNG")
    elif codec == "avif":
        if "AVIF" not in Image.registered_extensions().values():
            raise ImgGenError(
                "AVIF output is not supported by this Pillow build "
                "(pillow-avif-plugin not installed)")
        work = _flatten_white(img)
        work.save(buf, "AVIF", quality=quality or 60)
    else:
        raise ImgGenError(f"unknown codec {codec!r}")
    return buf.getvalue()


def _ffmpeg_encoders() -> set[str]:
    """Codecs ffmpeg on this box can encode (empty when absent)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return set()
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    found = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in (
                "libaom-av1", "libwebp", "libwebp_anim", "png", "mjpeg"):
            found.add(parts[1])
    return found


def _ffmpeg_encode(img: Any, codec: str, quality: int | None,
                   timeout_s: float, workdir: str) -> bytes:
    """Encode via ffmpeg (AVIF through libaom-av1, WebP fallback).

    Temp files live beside the source image (disk-backed), never on
    the /tmp tmpfs. Raises ImgGenError naming the failure.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ImgGenError("ffmpeg is not installed on this box")
    src_png = os.path.join(workdir, "squoosh-in.png")
    out_name = os.path.join(workdir, f"squoosh-out.{codec}")
    work = _flatten_white(img) if codec == "avif" else img
    work.save(src_png, "PNG")
    if codec == "avif":
        crf = int(round((100 - (quality or 60)) * 63 / 100))
        cmd = [ffmpeg, "-y", "-v", "error", "-i", src_png,
               "-c:v", "libaom-av1", "-cpu-used", "6",
               "-crf", str(crf), out_name]
    elif codec == "webp":
        cmd = [ffmpeg, "-y", "-v", "error", "-i", src_png,
               "-c:v", "libwebp", "-quality", str(quality or 80),
               out_name]
    else:
        raise ImgGenError(f"ffmpeg has no {codec} rung")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise ImgGenError(
            f"ffmpeg {codec} encode timed out after {timeout_s:g}s"
        ) from exc
    if proc.returncode != 0 or not os.path.exists(out_name):
        raise ImgGenError(
            f"ffmpeg {codec} encode failed: "
            f"{(proc.stderr or '').strip()[:200]}")
    try:
        with open(out_name, "rb") as f:
            return f.read()
    finally:
        for p in (src_png, out_name):
            with suppress(OSError):
                os.unlink(p)


def _heif_encode(img: Any, codec: str, quality: int | None,
                 timeout_s: float, workdir: str) -> bytes:
    """Encode AVIF via heif-enc (the standby AVIF rung)."""
    heif_enc = shutil.which("heif-enc")
    if not heif_enc:
        raise ImgGenError("heif-enc is not installed on this box")
    if codec != "avif":
        raise ImgGenError(f"heif-enc has no {codec} rung")
    src_png = os.path.join(workdir, "squoosh-in.png")
    out_name = os.path.join(workdir, "squoosh-out.avif")
    _flatten_white(img).save(src_png, "PNG")
    cmd = [heif_enc, "-A", "-q", str(quality or 60),
           src_png, "-o", out_name]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise ImgGenError(
            f"heif-enc AVIF encode timed out after {timeout_s:g}s"
        ) from exc
    if proc.returncode != 0 or not os.path.exists(out_name):
        raise ImgGenError(
            f"heif-enc AVIF encode failed: "
            f"{(proc.stderr or '').strip()[:200]}")
    try:
        with open(out_name, "rb") as f:
            return f.read()
    finally:
        for p in (src_png, out_name):
            with suppress(OSError):
                os.unlink(p)


def encode_pixels(img: Any, codec: str, quality: int | None,
                  *, timeout_s: float = FFMPEG_TIMEOUT_S,
                  workdir: str | None = None) -> tuple[bytes, str]:
    """Encode ``img`` with ``codec``; returns (bytes, rung label).

    Pillow first (jpeg/webp/png, and avif where the Pillow build
    has it); ffmpeg AVIF where libaom-av1 is present; heif-enc AVIF
    as the standby when ffmpeg lacks it; ffmpeg WebP only when
    Pillow cannot write WebP. Raises ImgGenError naming the rung
    that could not run.
    """
    if codec not in CODECS:
        raise ImgGenError(
            f"unknown codec {codec!r}; want one of {list(CODECS)}")
    if codec in ("jpeg", "png"):
        return _pillow_encode(img, codec, quality), f"pillow-{codec}"
    if codec == "webp":
        try:
            return _pillow_encode(img, codec, quality), "pillow-webp"
        except ImgGenError:
            if workdir is None:
                raise
            return (_ffmpeg_encode(img, codec, quality, timeout_s,
                                   workdir), "ffmpeg-webp")
    # avif
    try:
        return _pillow_encode(img, codec, quality), "pillow-avif"
    except ImgGenError as exc:
        pillow_reason = str(exc)
    encoders = _ffmpeg_encoders()
    if "libaom-av1" in encoders:
        if workdir is None:
            raise ImgGenError(
                "ffmpeg AVIF needs a workdir for its temp files")
        return (_ffmpeg_encode(img, codec, quality, timeout_s,
                               workdir), "ffmpeg-avif")
    if shutil.which("heif-enc"):
        if workdir is None:
            raise ImgGenError(
                "heif-enc AVIF needs a workdir for its temp files")
        return (_heif_encode(img, codec, quality, timeout_s,
                             workdir), "heifenc-avif")
    raise ImgGenError(
        "no AVIF encoder on this box: Pillow lacks AVIF "
        f"({pillow_reason}), ffmpeg "
        "lacks libaom-av1, heif-enc is absent")


def optimize_status() -> dict[str, Any]:
    """Which codec rungs this box can actually run — probed by a
    real tiny encode, never assumed from a version string."""
    from PIL import Image

    probe = Image.new("RGB", (16, 16), (200, 40, 90))
    rungs: dict[str, dict[str, Any]] = {}

    def _pillow_ok(codec: str) -> tuple[bool, str]:
        try:
            _pillow_encode(probe, codec, 80 if codec != "png" else None)
            return True, "pillow"
        except Exception as exc:  # noqa: BLE001 - probe, reason kept
            return False, str(exc)[:160]

    for codec in ("jpeg", "webp", "png"):
        ok, why = _pillow_ok(codec)
        rungs[f"pillow-{codec}"] = {"available": ok,
                                    "reason": "" if ok else why}
    ok, why = _pillow_ok("avif")
    rungs["pillow-avif"] = {"available": ok, "reason": "" if ok else why}
    encoders = _ffmpeg_encoders()
    rungs["ffmpeg-avif"] = {
        "available": "libaom-av1" in encoders,
        "reason": ("" if "libaom-av1" in encoders
                   else "ffmpeg lacks libaom-av1"),
    }
    rungs["ffmpeg-webp"] = {
        "available": "libwebp" in encoders,
        "reason": ("" if "libwebp" in encoders
                   else "ffmpeg lacks libwebp"),
    }
    rungs["heifenc-avif"] = {
        "available": shutil.which("heif-enc") is not None,
        "reason": ("" if shutil.which("heif-enc")
                   else "heif-enc is not installed"),
    }
    available = sorted(k for k, v in rungs.items() if v["available"])
    unavailable = sorted(k for k, v in rungs.items()
                         if not v["available"])
    return {"ok": True, "rungs": rungs,
            "available": available, "unavailable": unavailable,
            "ssim": ("unavailable — scikit-image is not installed; "
                     "quality is PSNR only, computed from decoded "
                     "pixels")}


def _sniff_format(path: Path) -> str:
    """The source's container format from its magic bytes —
    ``load_image`` loses ``img.format`` (exif_transpose returns a
    new image), so the label comes from the file itself."""
    try:
        head = path.read_bytes()[:16]
    except OSError:
        return "unknown"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if head[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    if head[4:8] == b"ftyp":
        return "AVIF" if b"avif" in head else "HEIF"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if head[:2] == b"BM":
        return "BMP"
    return "unknown"


# ── the ladder ───────────────────────────────────────────────────────

def _target_spec(target: str) -> dict[str, Any]:
    key = (target or "whatsapp").strip().lower()
    if key not in TARGETS:
        raise ImgGenError(
            f"unknown optimize target {target!r}; "
            f"want one of {sorted(TARGETS)}")
    return dict(TARGETS[key])


def _trial_plan(codecs: list[str] | None) -> list[tuple[str, int | None]]:
    """The bounded trial ladder: (codec, quality) pairs."""
    plan: list[tuple[str, int | None]] = []
    wanted = codecs or list(CODECS)
    for codec in wanted:
        if codec == "jpeg":
            plan += [("jpeg", 90), ("jpeg", 80), ("jpeg", 70)]
        elif codec == "webp":
            plan += [("webp", 90), ("webp", 80), ("webp", 70),
                     ("webp", None)]
        elif codec == "png":
            plan += [("png", None)]
        elif codec == "avif":
            plan += [("avif", 60)]
    return plan


def _fits_target(trial: OptimizeTrial, spec: dict[str, Any],
                 budget: int) -> tuple[bool, str]:
    """Does an ok trial satisfy the target's constraint?"""
    if budget and trial.size_bytes > budget:
        return False, f"{trial.size_bytes} bytes > {budget} budget"
    max_edge = spec["max_long_edge"]
    if max_edge and max(trial.width, trial.height) > max_edge:
        return False, (f"long edge {max(trial.width, trial.height)} "
                       f"> {max_edge}")
    dim_sum = spec["max_dim_sum"]
    if dim_sum and trial.width + trial.height > dim_sum:
        return False, (f"width+height {trial.width + trial.height} "
                       f"> {dim_sum}")
    if dim_sum and trial.width and trial.height:
        ratio = max(trial.width, trial.height) / min(trial.width,
                                                     trial.height)
        if ratio > 20:
            return False, f"aspect ratio {ratio:.1f} > 20"
    return True, ""


def _pick(trials: list[OptimizeTrial], spec: dict[str, Any],
          budget: int, target: str,
          psnr_floor_db: float = SMALLEST_PSNR_FLOOR_DB,
          ) -> tuple[int | None, int | None]:
    """(recommended index, closest index) over ok trials."""
    ok = [i for i, t in enumerate(trials) if t.status == "ok"]
    if not ok:
        return None, None
    fitting = [i for i in ok
               if _fits_target(trials[i], spec, budget)[0]]
    closest = min(ok, key=lambda i: trials[i].size_bytes)
    if not fitting:
        return None, closest
    if target == "archive":
        lossless = [i for i in fitting
                    if trials[i].psnr_db is None]
        if lossless:
            return min(lossless,
                       key=lambda i: trials[i].size_bytes), closest
        return max(fitting,
                   key=lambda i: (trials[i].psnr_db
                                  if trials[i].psnr_db is not None
                                  else -1.0)), closest
    if target == "smallest":
        # Lossless trials pass any floor (identical pixels are
        # perfect quality); among the trials that pass, the
        # smallest file wins.
        good = [i for i in fitting
                if trials[i].psnr_db is None
                or trials[i].psnr_db >= psnr_floor_db]
        if good:
            return min(good, key=lambda i: trials[i].size_bytes), closest
        return None, closest
    best_psnr = max(
        fitting,
        key=lambda i: (float("inf") if trials[i].psnr_db is None
                       else trials[i].psnr_db))
    best_val = (float("inf") if trials[best_psnr].psnr_db is None
                else trials[best_psnr].psnr_db)
    if best_val == float("inf"):
        # A lossless trial fits the budget: identical pixels beat
        # every lossy trial, whatever the preferred codec.
        lossless = [i for i in fitting
                    if trials[i].psnr_db is None]
        return min(lossless,
                   key=lambda i: trials[i].size_bytes), closest
    preferred = spec["preferred_codec"]
    if preferred:
        near = [i for i in fitting
                if trials[i].codec == preferred
                and trials[i].psnr_db is not None
                and best_val - trials[i].psnr_db
                <= PREFERRED_CODEC_MARGIN_DB]
        if near:
            return max(near, key=lambda i: (
                float("inf") if trials[i].psnr_db is None
                else trials[i].psnr_db)), closest
    return best_psnr, closest


def optimize_image(src: str | os.PathLike[str], *,
                   target: str = "whatsapp",
                   max_bytes: int = 0,
                   codecs: list[str] | None = None,
                   out_dir: str | os.PathLike[str] | None = None,
                   timeout_s: float = FFMPEG_TIMEOUT_S,
                   psnr_floor_db: float = SMALLEST_PSNR_FLOOR_DB,
                   ) -> OptimizeReport:
    """Run the codec ladder over ``src`` and recommend the best
    trial under the target's constraint.

    ``target`` names the intent (whatsapp | telegram | web |
    archive | smallest); ``max_bytes`` overrides the target's byte
    budget (0 = the target's own budget). ``codecs`` restricts the
    ladder. ``out_dir`` writes the recommended trial to disk and
    re-stats it — the reported size is the file's real size. Every
    trial is measured on disk-equivalent bytes; a trial that fails
    is recorded, never raised.
    """
    from PIL import Image

    from ...media_edit.images import load_image

    spec = _target_spec(target)
    if max_bytes and max_bytes < 0:
        raise ImgGenError("max_bytes must be >= 0")
    budget = int(max_bytes) if max_bytes else int(spec["max_bytes"])
    src_path = Path(src)
    if not src_path.exists():
        raise ImgGenError(f"no such image: {src}")
    try:
        img = load_image(src_path)
    except Exception as exc:
        raise ImgGenError(f"could not load image {src}: {exc}") from exc
    notes: list[str] = []
    if getattr(img, "n_frames", 1) > 1:
        notes.append("animated source — trials use the first frame only")
    width, height = img.size
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        notes.append("alpha flattened over white for jpeg/avif trials")

    # Resize for the target before trialling: the trials measure the
    # image that would actually be sent, not the oversized original.
    work = img
    if spec["max_long_edge"] and max(width, height) > spec["max_long_edge"]:
        edge = int(spec["max_long_edge"])
        scale = edge / max(width, height)
        work = img.resize((max(1, int(width * scale)),
                           max(1, int(height * scale))), Image.LANCZOS)
        notes.append(f"resized to {work.size[0]}x{work.size[1]} "
                     f"(long edge {edge}) for {target}")
        width, height = work.size
    if spec["max_dim_sum"] and width + height > spec["max_dim_sum"]:
        total = int(spec["max_dim_sum"])
        scale = total / (width + height)
        work = work.resize((max(1, int(width * scale)),
                            max(1, int(height * scale))), Image.LANCZOS)
        notes.append(f"resized to {work.size[0]}x{work.size[1]} "
                     f"(width+height {total}) for {target}")
        width, height = work.size
    reference = _flatten_white(work)

    plan = _trial_plan(codecs)
    if not plan:
        raise ImgGenError("no codecs requested — nothing to trial")
    trials: list[OptimizeTrial] = []
    rungs_available: list[str] = []
    rungs_unavailable: list[str] = []
    try:
        tmp = tempfile.TemporaryDirectory(prefix=".squoosh-",
                                          dir=str(src_path.parent))
    except OSError as exc:
        raise ImgGenError(
            f"cannot create a workdir beside {src}: {exc}") from exc
    try:
        for codec, quality in plan:
            label = ("lossless" if quality is None
                     else f"q{quality}")
            try:
                data, rung = encode_pixels(
                    work, codec, quality, timeout_s=timeout_s,
                    workdir=tmp.name)
            except ImgGenError as exc:
                trials.append(OptimizeTrial(
                    codec=codec, quality=quality, rung="",
                    status="unavailable", reason=str(exc)[:200]))
                rungs_unavailable.append(f"{codec}-{label}")
                continue
            except Exception as exc:  # noqa: BLE001 - trial failure
                trials.append(OptimizeTrial(
                    codec=codec, quality=quality, rung="",
                    status="failed", reason=str(exc)[:200]))
                continue
            if rung not in rungs_available:
                rungs_available.append(rung)
            trial = OptimizeTrial(
                codec=codec, quality=quality, size_bytes=len(data),
                width=width, height=height, rung=rung, _data=data)
            if codec in ("jpeg", "avif") and has_alpha:
                trial.reason = "alpha flattened over white"
            try:
                decoded = _decode(data, codec,
                                  timeout_s=timeout_s,
                                  workdir=tmp.name)
                trial.psnr_db = psnr_db(reference, decoded)
                if trial.psnr_db is None:
                    trial.reason = (
                        (trial.reason + "; " if trial.reason else "")
                        + "identical pixels")
            except Exception as exc:  # noqa: BLE001 - bad decode
                trial.status = "failed"
                trial.reason = f"trial bytes do not decode: {exc}"[:200]
                trial.psnr_db = None
            trials.append(trial)
    finally:
        tmp.cleanup()

    recommended, closest = _pick(trials, spec, budget, target,
                                 psnr_floor_db)
    report = OptimizeReport(
        source=str(src_path), source_bytes=src_path.stat().st_size,
        source_format=_sniff_format(src_path),
        width=img.size[0], height=img.size[1],
        target=(target or "whatsapp").strip().lower(),
        target_note=spec["basis"], trials=trials,
        recommended=recommended, closest=closest,
        rungs_available=sorted(rungs_available),
        rungs_unavailable=sorted(set(rungs_unavailable)),
        notes=notes)
    if out_dir is not None and recommended is not None:
        best = trials[recommended]
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"optimized-{src_path.stem}{CODEC_EXT[best.codec]}"
        n = 1
        while dest.exists():
            dest = out / (f"optimized-{src_path.stem}-{n}"
                          f"{CODEC_EXT[best.codec]}")
            n += 1
        dest.write_bytes(best._data or b"")
        actual = dest.stat().st_size
        if actual != best.size_bytes:
            raise ImgGenError(
                f"recommended trial wrote {actual} bytes but the "
                f"trial measured {best.size_bytes} — refusing to "
                "report a size that is not on disk")
        best.path = str(dest)
    return report


def document_from_report(report: OptimizeReport, *,
                         max_bytes: int = 0) -> Any:
    """The Studio document for an already-run ladder report: a
    ``resize`` op when the target resized, then the winning
    encoding as an ``optimize`` op, with the full trial table in
    the document notes. Rendering it reproduces the winning
    trial's pixels and encoding — the document is what Studio
    work continues from after the optimisation. Raises when no
    trial fits the target, naming the constraint and the closest
    trial instead of recording a decision that cannot be kept.
    """
    from .document import ImageDocument

    if report.recommended is None:
        near = (report.trials[report.closest]
                if report.closest is not None else None)
        detail = ""
        if near is not None:
            detail = (f" — closest is {near.codec} "
                      f"{'q' + str(near.quality) if near.quality is not None else 'lossless'} "
                      f"at {near.size_bytes} bytes")
        raise ImgGenError(
            f"no trial fits {report.target} "
            f"(budget {report.target_note}){detail}")
    best = report.trials[report.recommended]
    doc = ImageDocument(report.source)
    if (best.width, best.height) != (report.width, report.height):
        # The ladder resized for the target (WhatsApp's 1920 edge,
        # Telegram's dimension sum): the resize is its own op in
        # the document, so replay reproduces the exact trial pixels
        # instead of saving the full-size source in the new codec.
        doc.resize(best.width, best.height, mode="exact")
    doc.optimize(codec=best.codec, quality=best.quality,
                 target=report.target, max_bytes=int(max_bytes or 0),
                 size_bytes=best.size_bytes, psnr_db=best.psnr_db,
                 trials=len([t for t in report.trials
                             if t.status == "ok"]))
    doc.notes.append(report.format_table())
    return doc


def optimize_document(image_path: str | os.PathLike[str], *,
                      target: str = "whatsapp",
                      max_bytes: int = 0,
                      codecs: list[str] | None = None,
                      timeout_s: float = FFMPEG_TIMEOUT_S,
                      ) -> Any:
    """An unrendered :class:`ImageDocument` holding one ``optimize``
    op — the ladder runs once at build time, the winning encoding
    (codec + quality + target + trial summary) is recorded as the
    op, and the Studio renders it later with that encoding.
    Raises when no trial fits the target, naming the constraint and
    the closest trial instead of recording a decision that cannot
    be kept.
    """
    report = optimize_image(image_path, target=target,
                            max_bytes=max_bytes, codecs=codecs,
                            timeout_s=timeout_s)
    return document_from_report(report, max_bytes=max_bytes)


def save_optimized(img: Any, path: str | os.PathLike[str], *,
                   codec: str, quality: int | None = None,
                   timeout_s: float = FFMPEG_TIMEOUT_S) -> Path:
    """Write ``img`` to ``path`` in ``codec`` — the save path the
    Studio document flow uses when an optimize op chose the
    encoding. Pillow codecs go through ``save_image`` (metadata
    stripped, JPEG flattening); AVIF goes through the ffmpeg /
    heif-enc rung (Pillow cannot write AVIF on this box)."""
    from ...media_edit.images import save_image

    out = Path(path)
    if codec == "avif":
        tmp = tempfile.TemporaryDirectory(prefix=".squoosh-",
                                          dir=str(out.parent))
        try:
            data, _rung = encode_pixels(
                img, codec, quality, timeout_s=timeout_s,
                workdir=tmp.name)
        finally:
            tmp.cleanup()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out
    save_image(img, out, fmt=CODEC_FORMAT[codec],
               quality=quality if quality is not None else 90)
    return out
