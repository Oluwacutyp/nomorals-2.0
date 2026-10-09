"""Dataset tooling: folder of images → captioned training set.

``build_dataset_manifest(image_dir)`` walks a directory, pairs every
image with a caption, and writes ``manifest.jsonl`` (``{"image": path,
"caption": text}``). Caption sources, in order:

1. Sidecar ``<name>.txt`` next to the image (highest priority — the
   user's own words win).
2. Devon's own vision stack (``nomorals.agents.seer`` / vision tools)
   when importable — real captions, offline-capable providers first.
3. Filename-derived caption (``lagos_market_01.jpg`` →
   ``"lagos market 01"``) — honest fallback, never empty.

Also: :class:`ImageFolderDataset` (torch) reading the manifest with
resize/center-crop/normalize to [-1, 1], and
:func:`default_training_captions` with Nigeria-relevant prompt starters
so locally-trained models actually know what the user cares about.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from . import ImgGenError, TORCH_AVAILABLE

__all__ = [
    "IMAGE_EXTS",
    "caption_from_filename",
    "caption_with_devon_vision",
    "build_dataset_manifest",
    "load_manifest",
    "default_training_captions",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def caption_from_filename(path: str) -> str:
    """'lagos_market_01.jpg' → 'lagos market 01'. Never empty."""
    stem = Path(path).stem
    text = re.sub(r"[_\-]+", " ", stem).strip()
    text = re.sub(r"\s+", " ", text)
    return text or "untrained image"


def caption_with_devon_vision(path: str) -> str | None:
    """Caption via Devon's own vision stack. None when unavailable.

    Tries the seer/vision tool path used by the rest of the codebase;
    any failure returns None (the caller falls back to filename).
    """
    try:
        from ...agents.seer import see  # type: ignore
    except Exception:
        return None
    try:
        result = see(path, prompt=(
            "Describe this image in one detailed sentence for training "
            "an image generation model. Focus on subject, setting, "
            "colors, lighting, and style."))
    except Exception:
        return None
    if isinstance(result, dict):
        text = str(result.get("description") or result.get("text") or "")
    else:
        text = str(result or "")
    text = text.strip()
    return text or None


def build_dataset_manifest(image_dir: str, out_path: str | None = None,
                           use_vision: bool = True,
                           progress: Any = None) -> dict:
    """Walk ``image_dir`` → ``manifest.jsonl``. Returns a summary dict.

    Never raises on a bad image — it is skipped and counted.
    Raises ImgGenError only when the directory itself is unusable.
    """
    root = Path(image_dir)
    if not root.is_dir():
        raise ImgGenError(f"image directory not found: {image_dir}")
    out = Path(out_path) if out_path else root / "manifest.jsonl"

    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in IMAGE_EXTS
                   and p.name != out.name)
    if not files:
        raise ImgGenError(f"no images found under {image_dir}")

    rows: list[dict[str, str]] = []
    skipped = 0
    vision_used = 0
    for i, img in enumerate(files):
        sidecar = img.with_suffix(".txt")
        caption: str | None = None
        if sidecar.exists():
            try:
                caption = sidecar.read_text(
                    encoding="utf-8").strip() or None
            except OSError:
                caption = None
        if caption is None and use_vision:
            caption = caption_with_devon_vision(str(img))
            if caption:
                vision_used += 1
        if caption is None:
            caption = caption_from_filename(img.name)
        rows.append({"image": str(img), "caption": caption})
        if progress and i % 25 == 0:
            progress(f"captioned {i + 1}/{len(files)}")

    # Validate images cheaply (PIL when available; else trust suffix).
    valid: list[dict[str, str]] = []
    try:
        from PIL import Image

        for row in rows:
            try:
                with Image.open(row["image"]) as im:
                    im.verify()
                valid.append(row)
            except Exception:
                skipped += 1
    except ImportError:
        valid = rows

    with open(out, "w", encoding="utf-8") as f:
        for row in valid:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"manifest": str(out), "images": len(valid),
            "skipped": skipped, "vision_captions": vision_used,
            "filename_captions": len(valid) - vision_used}


def load_manifest(path: str) -> list[dict[str, str]]:
    """Read a manifest.jsonl back. Raises ImgGenError on problems."""
    if not os.path.exists(path):
        raise ImgGenError(f"manifest not found: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ImgGenError(
                    f"manifest line {ln} is not valid JSON: {exc}") from exc
            if "image" not in row or "caption" not in row:
                raise ImgGenError(
                    f"manifest line {ln} needs 'image' and 'caption'")
            rows.append(row)
    if not rows:
        raise ImgGenError(f"manifest is empty: {path}")
    return rows


def default_training_captions() -> list[str]:
    """Nigeria-relevant prompt starters for locally-trained models.

    Used as default captions when building a dataset from uncaptioned
    folders, and as style anchors in the pipeline's prompt builder.
    """
    return [
        "lagos street market at golden hour, vibrant stalls",
        "yoruba bride in aso-oke, portrait photography",
        "danfo bus in lagos traffic, cinematic",
        "afrobeats concert crowd, stage lights, night",
        "nigerian jollof rice close-up, food photography",
        "lekki beach at sunset, palm trees",
        "igbo masquerade festival, colorful costume",
        "abuja city skyline at dusk",
        "hausa horseman at durbar festival",
        "nigerian woman in ankara print, studio portrait",
    ]


if TORCH_AVAILABLE:  # pragma: no cover - needs torch + PIL
    import torch
    from torch.utils.data import Dataset

    class ImageFolderDataset(Dataset):
        """Manifest-backed dataset → (image [-1,1], caption) pairs.

        Images are resized (short side) then center-cropped to
        ``image_size``. Captions pass through as strings; the pipeline
        encodes them at sample time.
        """

        def __init__(self, manifest: str | list[dict[str, str]],
                     image_size: int = 64) -> None:
            from PIL import Image

            self._Image = Image
            self.rows = (load_manifest(manifest)
                         if isinstance(manifest, str) else list(manifest))
            self.image_size = image_size

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int):
            from PIL import ImageOps

            row = self.rows[idx]
            with self._Image.open(row["image"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                # Resize short side, then center crop.
                scale = max(self.image_size / im.width,
                            self.image_size / im.height)
                im = im.resize((round(im.width * scale),
                                round(im.height * scale)),
                                self._Image.BILINEAR)
                left = (im.width - self.image_size) // 2
                top = (im.height - self.image_size) // 2
                im = im.crop((left, top, left + self.image_size,
                              top + self.image_size))
                arr = torch.from_numpy(
                    __import__("numpy").asarray(im)).float()
                arr = arr.permute(2, 0, 1) / 127.5 - 1.0  # [-1, 1]
            return arr, row["caption"]
else:
    ImageFolderDataset = None  # type: ignore[assignment]
