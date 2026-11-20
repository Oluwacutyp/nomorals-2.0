"""The clip-info loader — a manifest ties a CSV to its mapping.

The bevy-facial-mocap pattern (storytold/bevy-mocap, pattern
only): a ``.morph_anim.json`` file points at the CSV, names the
animation root, optionally overrides the fps, and optionally
points at a mapping file. The loader resolves the CSV and
mapping paths relative to the manifest's own directory (the
clone resolves them through Bevy's asset paths; Devon's files
live on disk, so the manifest's directory is the base), parses
the CSV, applies the mapping, and hands back a face take.

A CSV can also load directly, without a manifest — the root
defaults to the file's stem and the fps is the deduced one.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..rig.rig import ARKIT_BLENDSHAPES
from .csv import CSVAnimation
from .errors import FaceMocapError
from .mapping import FaceMapping, MorphMapping, default_arkit_mapping
from .take import FaceFrame, FaceTake

__all__ = [
    "load_csv",
    "load_manifest",
    "load_take_json",
]


def _resolve(base: Path, raw: str, *, what: str) -> Path:
    """A manifest-relative path: absolute stays absolute,
    relative resolves against the manifest's directory."""
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    if not candidate.exists():
        raise FaceMocapError(
            f"no {what} at {candidate}", kind="missing_file")
    return candidate


def _build(name: str, animation: CSVAnimation, *,
           root_name: str, fps: float | None,
           mapping: FaceMapping | None,
           source: str) -> FaceTake:
    """Parse + map + take. ``mapping=None`` means identity:
    CSV columns drive same-named rig blendshapes, and bone
    entries are impossible without a mapping file."""
    use_fps = float(fps) if fps is not None else animation.fps
    if use_fps <= 0:
        raise FaceMocapError(
            f"fps must be positive, got {use_fps}",
            kind="bad_manifest")
    if mapping is not None:
        use_mapping = mapping
    else:
        # Identity, as documented: every CSV column the rig
        # has drives the same-named rig blendshape. A column
        # the rig does not have (a bone-driving column, say)
        # is unmapped, not refused — bones need a mapping file
        # by definition, and the take reports the unmapped
        # columns. (An empty mapping would load a take with no
        # face at all — the first version of this loader did
        # exactly that.)
        use_mapping = FaceMapping(entries=[
            (MorphMapping(name), name)
            for name in animation.morph_names
            if name in ARKIT_BLENDSHAPES])
    applied = use_mapping.apply(
        animation, rig_blendshapes=ARKIT_BLENDSHAPES)
    frames = [
        FaceFrame(time_s=index / use_fps,
                  blendshapes=dict(blend),
                  bones={bone: tuple(values)
                         for bone, values in bones.items()})
        for index, (blend, bones) in enumerate(zip(
            applied["blendshapes"], applied["bones"],
            strict=True))]
    return FaceTake(name=name, root_name=root_name, fps=use_fps,
                    morph_names=list(animation.morph_names),
                    frames=frames,
                    unmapped=tuple(applied["unmapped"]),
                    source=source)


def load_csv(path: str | Path, *, name: str | None = None,
             root_name: str | None = None,
             fps: float | None = None,
             mapping: FaceMapping | str | None = None) -> FaceTake:
    """Load a LiveLink CSV directly. ``mapping`` is a mapping
    object, a path to a mapping JSON file, the literal
    ``"arkit"`` (the bundled 52-name rename table), or None
    (identity — CSV columns drive same-named blendshapes)."""
    target = Path(path).expanduser()
    if not target.exists():
        raise FaceMocapError(
            f"no CSV file at {target}", kind="missing_csv")
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise FaceMocapError(
            f"CSV file at {target} is unreadable: {exc}",
            kind="bad_csv") from exc
    animation = CSVAnimation.parse(text)
    use_mapping = _coerce_mapping(mapping)
    return _build(name or target.stem, animation,
                  root_name=root_name or target.stem, fps=fps,
                  mapping=use_mapping, source=str(target))


def load_manifest(path: str | Path, *,
                  name: str | None = None) -> FaceTake:
    """Load a ``.morph_anim.json`` manifest: ``{csv,
    root_name, fps?, mapping?}``. A missing file, a malformed
    manifest, a missing key, or an unreadable CSV/mapping is
    refused by name."""
    target = Path(path).expanduser()
    if not target.exists():
        raise FaceMocapError(
            f"no manifest at {target}", kind="missing_manifest")
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise FaceMocapError(
            f"manifest at {target} is unreadable: {exc}",
            kind="bad_manifest") from exc
    if not isinstance(blob, dict):
        raise FaceMocapError(
            f"manifest at {target} must be a JSON object",
            kind="bad_manifest")
    csv_raw = blob.get("csv")
    root_name = blob.get("root_name")
    if not isinstance(csv_raw, str) or not csv_raw.strip():
        raise FaceMocapError(
            f"manifest at {target} needs a 'csv' path",
            kind="bad_manifest")
    if not isinstance(root_name, str) or not root_name.strip():
        raise FaceMocapError(
            f"manifest at {target} needs a 'root_name'",
            kind="bad_manifest")
    base = target.parent
    csv_path = _resolve(base, csv_raw, what="CSV file")
    fps = blob.get("fps")
    if fps is not None and not isinstance(fps, (int, float)):
        raise FaceMocapError(
            f"manifest at {target}: 'fps' must be a number",
            kind="bad_manifest")
    mapping: FaceMapping | None = None
    mapping_raw = blob.get("mapping")
    if mapping_raw is not None:
        if not isinstance(mapping_raw, str) \
                or not mapping_raw.strip():
            raise FaceMocapError(
                f"manifest at {target}: 'mapping' must be a "
                f"path string",
                kind="bad_manifest")
        if mapping_raw.strip() == "arkit":
            mapping = default_arkit_mapping()
        else:
            mapping = FaceMapping.from_file(
                _resolve(base, mapping_raw, what="mapping file"))
    text = csv_path.read_text(encoding="utf-8")
    animation = CSVAnimation.parse(text)
    return _build(name or target.stem, animation,
                  root_name=root_name,
                  fps=(float(fps) if fps is not None else None),
                  mapping=mapping, source=str(target))


def _coerce_mapping(
        mapping: FaceMapping | str | None) -> FaceMapping | None:
    """A mapping argument into a mapping object (or None for
    identity)."""
    if mapping is None or isinstance(mapping, FaceMapping):
        return mapping
    if mapping == "arkit":
        return default_arkit_mapping()
    return FaceMapping.from_file(mapping)


def load_take_json(path: str | Path, *,
                   name: str | None = None) -> FaceTake:
    """Load a face take from its own JSON shape (what
    :meth:`FaceTake.to_json` writes). A missing or malformed
    file is refused by name."""
    target = Path(path).expanduser()
    if not target.exists():
        raise FaceMocapError(
            f"no face take at {target}", kind="missing_take")
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise FaceMocapError(
            f"face take at {target} is unreadable: {exc}",
            kind="bad_take") from exc
    take = FaceTake.from_json(blob, where=f"face take {target}")
    if name is not None:
        take.name = name
    return take
