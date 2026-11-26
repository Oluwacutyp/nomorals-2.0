"""Export package — ported from the source's
``trendclip/export_package.py``.

A rendered clip ships as a package, not a bare file: SRT captions, a
thumbnail, per-platform metadata, provenance (source URL, rights label,
job id, both rungs' picks) and a checksummed manifest. The source's
optional Codex metadata step is NOT ported — Devon writes the platform
copy through her own brain (see ``brain.brain_platform_copy``), never a
hardcoded metadata writer.

Manual publish only: the package is ready for the owner to upload; nothing
in this module posts, schedules or uploads anything.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "PLATFORM_METADATA",
    "export_package",
    "write_manifest",
    "write_srt",
]

#: Per-platform upload metadata, from the source. The copy itself (title
#: / description / hashtags) is filled by the caller — deterministically
#: from the topic, or by the brain rung when one is available.
PLATFORM_METADATA: dict[str, dict[str, Any]] = {
    "youtube": {"aspect": "9:16", "max_duration_s": 60,
                "caption": "YouTube Shorts"},
    "facebook": {"aspect": "9:16", "max_duration_s": 90,
                 "caption": "Facebook Reels"},
    "tiktok": {"aspect": "9:16", "max_duration_s": 180,
               "caption": "TikTok"},
}


def write_srt(words: list[dict[str, Any]], path: str | Path,
              *, start: float = 0.0) -> Path:
    """Write word-timed SRT captions, grouped into short lines."""
    def fmt(t: float) -> str:
        t = max(0.0, t - start)
        total_ms = int(round(t * 1000))
        h, rem = divmod(total_ms, 3600000)
        m, rem = divmod(rem, 60000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    chunk: list[dict[str, Any]] = []
    index = 1
    for word in words:
        chunk.append(word)
        text = " ".join(str(w.get("w", "")) for w in chunk)
        last_end = float(chunk[-1].get("e", chunk[-1].get("end", 0.0)))
        first_start = float(chunk[0].get("s", chunk[0].get("start", 0.0)))
        if len(text) >= 42 or last_end - first_start >= 3.0:
            lines.append(str(index))
            lines.append(f"{fmt(first_start)} --> {fmt(last_end)}")
            lines.append(text.strip())
            lines.append("")
            index += 1
            chunk = []
    if chunk:
        lines.append(str(index))
        lines.append(
            f"{fmt(float(chunk[0].get('s', chunk[0].get('start', 0.0))))} "
            f"--> {fmt(float(chunk[-1].get('e', chunk[-1].get('end', 0.0))))}")
        lines.append(" ".join(str(w.get("w", "")) for w in chunk).strip())
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(files: dict[str, Path], path: str | Path) -> Path:
    """Write the checksummed manifest for a package directory."""
    out = Path(path)
    entries = {}
    for name, file_path in files.items():
        if file_path.is_file():
            entries[name] = {
                "path": file_path.name,
                "sha256": _sha256(file_path),
                "bytes": file_path.stat().st_size,
            }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "created_at": datetime.now(UTC).isoformat(),
        "files": entries,
    }, indent=2), encoding="utf-8")
    return out


def export_package(*, clip_path: str | Path,
                   words: list[dict[str, Any]],
                   job_id: str, topic: str, source_url: str,
                   rights_label: str, moment: dict[str, Any],
                   destinations: list[str] | None = None,
                   copy: dict[str, Any] | None = None,
                   thumbnail_path: str | Path | None = None,
                   out_dir: str | Path | None = None) -> dict[str, Any]:
    """Build the export package for one rendered clip.

    ``copy`` is the per-platform title/description/hashtags (deterministic
    by default, brain-written when the caller supplies it). Returns the
    package dict with every file path and the manifest.
    """
    clip = Path(clip_path)
    out = Path(out_dir) if out_dir else clip.parent / (clip.stem + "_package")
    out.mkdir(parents=True, exist_ok=True)
    moment_start = float(moment.get("start", 0.0))
    srt_path = write_srt(words, out / "captions.srt", start=moment_start)
    destinations = list(destinations or ["youtube"])
    platform_meta: dict[str, Any] = {}
    for dest in destinations:
        meta = dict(PLATFORM_METADATA.get(dest, {}))
        platform_copy = (copy or {}).get(dest, {}) if copy else {}
        meta.update({
            "title": str(platform_copy.get("title")
                         or f"{topic} — clip"),
            "description": str(platform_copy.get("description") or ""),
            "hashtags": list(platform_copy.get("hashtags") or []),
        })
        platform_meta[dest] = meta
    files: dict[str, Path] = {"clip": clip, "captions": srt_path}
    if thumbnail_path and Path(thumbnail_path).is_file():
        files["thumbnail"] = Path(thumbnail_path)
    manifest_path = write_manifest(files, out / "manifest.json")
    provenance = {
        "job_id": job_id,
        "topic": topic,
        "source_url": source_url,
        "rights": rights_label,
        "moment": {
            "n": moment.get("n"),
            "start": moment.get("start"),
            "end": moment.get("end"),
            "quality_score": moment.get("quality_score"),
            "strategy": moment.get("strategy"),
            "deterministic": True,
            "brain_pick": bool(moment.get("brain_pick")),
        },
        "destinations": destinations,
        "platforms": platform_meta,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    return {
        "dir": str(out),
        "clip": str(clip),
        "captions": str(srt_path),
        "thumbnail": str(files["thumbnail"]) if "thumbnail" in files else "",
        "manifest": str(manifest_path),
        "provenance": str(out / "provenance.json"),
        "platforms": platform_meta,
    }
