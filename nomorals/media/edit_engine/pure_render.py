"""Pure render: (sources, EditSpec, pack) → output, with a cache.

Rendering is a pure function of three inputs:

* ``sources`` — the source files (fingerprinted by path + size + mtime,
  or by an explicit ``sha256`` when the caller has content hashes);
* ``spec`` — the :class:`EditSpec` as a dict (profile + output path are
  machine hints, NOT part of the identity — see
  :func:`canonical_spec_dict`);
* ``pack`` — the look pack: ``{"style": "phonk", "spec_overrides": {...}}``.
  Swapping the pack swaps the look and changes the cache key, so A/B
  looks never collide in the cache.

Consequences, all deliberate:

* **Render cache.** ``render_cache_key`` is the SHA-256 of the
  canonical triple. :func:`render_pure` returns the cached file without
  re-rendering when the triple was rendered before.
* **Proxy workflow.** :func:`proxy_spec` derives the 480p proxy spec
  from the full spec (same cuts, same timing, scaled frame). Render the
  proxy first for a fast preview, then the full-res render —
  :func:`render_proxy_then_full` does both, each cached separately.
* **Profile-independent output.** The cache key and the stored spec
  never contain ``profile`` or ``out``: the same triple renders the
  same output on workstation, laptop, and termux. (Whether a given
  machine can *run* the render is a capability question, not an
  output-identity question.)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...media_edit.videos import MediaEditError
from .spec import EditSpec

__all__ = [
    "canonical_spec_dict",
    "source_fingerprint",
    "render_cache_key",
    "proxy_spec",
    "apply_pack",
    "render_pure",
    "render_proxy_then_full",
    "PROXY_HEIGHT",
]

#: proxy renders are this tall; width scales to preserve aspect ratio.
PROXY_HEIGHT = 480

#: spec keys a look pack may override. Everything else in the spec is
#: the edit itself — a pack swaps the look, never the cut.
_PACK_OVERRIDE_KEYS = frozenset({
    "transition", "crf", "preset", "fps", "text_layers", "seed",
})


def _canonical(value: Any) -> Any:
    """JSON-canonical form: sorted keys, no tuples, no surprises."""
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # dataclasses / objects with a dict form
    if hasattr(value, "to_dict"):
        return _canonical(value.to_dict())
    raise MediaEditError(
        f"not JSON-serializable for the render cache: {type(value).__name__}")


def canonical_spec_dict(spec: EditSpec | dict[str, Any]) -> dict[str, Any]:
    """The spec as cache identity: profile + output path stripped.

    ``profile`` is a per-machine performance hint (which ffmpeg preset
    the box prefers) and ``out`` is where *this* call wants the file —
    neither changes what the render *is*, so neither enters the key.
    Missing output params fall back to the engine defaults, so the
    canonical form is always complete.
    """
    raw = spec.to_dict() if isinstance(spec, EditSpec) else dict(spec)
    if not raw.get("clips"):
        raise MediaEditError("render needs at least one clip in the spec")
    out = _canonical(raw)
    out.pop("profile", None)
    out.pop("out", None)
    out.setdefault("fps", 30)
    out.setdefault("width", 1080)
    out.setdefault("height", 1920)
    out.setdefault("seed", 0)
    return out


def source_fingerprint(source: Any) -> dict[str, Any]:
    """Fingerprint one source: path + size + mtime, or explicit sha256."""
    if isinstance(source, dict):
        if source.get("sha256"):
            return {"sha256": str(source["sha256"]),
                    "path": str(source.get("path", ""))}
        if source.get("path"):
            return source_fingerprint(source["path"])
        raise MediaEditError(
            f"source dict needs 'path' or 'sha256', got {source!r}")
    p = Path(str(source))
    if not p.exists():
        raise MediaEditError(f"render source not found: {source}")
    st = p.stat()
    return {"path": str(p.resolve()), "size": st.st_size,
            "mtime_ns": st.st_mtime_ns}


def render_cache_key(sources: Any, spec: EditSpec | dict[str, Any],
                     pack: dict[str, Any] | None = None) -> str:
    """SHA-256 of the canonical (sources, spec, pack) triple."""
    if isinstance(sources, (str, os.PathLike, dict)):
        sources = [sources]
    payload = {
        "sources": [source_fingerprint(s) for s in sources],
        "spec": canonical_spec_dict(spec),
        "pack": _canonical(pack or {}),
    }
    blob = json.dumps(payload, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def proxy_spec(spec: EditSpec | dict[str, Any], *,
               height: int = PROXY_HEIGHT) -> dict[str, Any]:
    """Pure: full spec → 480p proxy spec (same cuts, scaled frame).

    Width scales to preserve aspect ratio and is rounded down to an
    even number (ffmpeg's yuv420p requirement). Everything else —
    segments, transitions, text, audio — is identical.
    """
    base = canonical_spec_dict(spec)
    w, h = int(base["width"]), int(base["height"])
    if h <= 0 or w <= 0:
        raise MediaEditError("proxy_spec needs positive width/height")
    new_h = int(height)
    new_w = int(w * new_h / h)
    new_w -= new_w % 2
    new_h -= new_h % 2
    if new_w <= 0 or new_h <= 0:
        raise MediaEditError("proxy dimensions collapsed to zero")
    out = dict(base)
    out["width"] = new_w
    out["height"] = new_h
    return out


def apply_pack(spec: EditSpec | dict[str, Any],
               pack: dict[str, Any] | None) -> dict[str, Any]:
    """Pure: spec + look pack → new spec dict (input untouched).

    ``pack`` = ``{"style": "<preset>", "spec_overrides": {...}}``. The
    style name is validated against the registered presets (a pack
    naming a style that does not exist is a bug, not a fallback), and
    overrides are limited to the look keys in ``_PACK_OVERRIDE_KEYS``.
    A pack may instead name a format registry entry (``{"format":
    "<name>"}``, §1.5b): its grade pack is the base, and the pack's
    own style/overrides win over it.
    """
    base = canonical_spec_dict(spec)
    pack = pack or {}
    if not isinstance(pack, dict):
        raise MediaEditError("pack must be a dict")
    if pack.get("format") is not None:
        # §1.5b: a pack may name a registry format; the format's grade
        # pack is the base, and the pack's own style/spec_overrides
        # win over it. Canvas/fps are NOT pack business — a format
        # steers those at spec level via formats.apply_format.
        from .formats import pack_for_format
        merged = pack_for_format(str(pack["format"]))
        merged.update({k: v for k, v in pack.items() if k != "format"})
        merged_overrides = dict(merged.get("spec_overrides") or {})
        merged_overrides.update(pack.get("spec_overrides") or {})
        if merged_overrides:
            merged["spec_overrides"] = merged_overrides
        pack = merged
    style = pack.get("style")
    if style is not None:
        from ..contentops.styles import list_styles  # lazy: styles import the engine
        if str(style).strip().lower() not in list_styles():
            raise MediaEditError(
                f"unknown pack style {style!r}; use: {list_styles()}")
    overrides = pack.get("spec_overrides") or {}
    if not isinstance(overrides, dict):
        raise MediaEditError("pack['spec_overrides'] must be a dict")
    unknown = sorted(set(overrides) - _PACK_OVERRIDE_KEYS)
    if unknown:
        raise MediaEditError(
            f"pack may not override {unknown} — a pack swaps the look, "
            f"never the cut (allowed: {sorted(_PACK_OVERRIDE_KEYS)})")
    out = dict(base)
    out.update(_canonical(overrides))
    return out


def _cache_paths(cache_dir: str | Path, key: str) -> tuple[Path, Path]:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.mp4", d / f"{key}.json"


def render_pure(sources: Any, spec: EditSpec | dict[str, Any],
                pack: dict[str, Any] | None = None, *,
                cache_dir: str | Path,
                renderer: Any = None) -> dict[str, Any]:
    """Render (sources, spec, pack) → cached mp4.

    Cache hit → the stored file, no ffmpeg run. Cache miss → the
    ``renderer`` (default: :func:`edit_engine.spec.render`) renders
    into a temp file that is atomically moved into the cache, plus a
    ``<key>.json`` manifest recording the triple that produced it.
    Returns ``{"output", "cache_key", "cached", "manifest"}`` plus
    the 1.1b(b) honesty fields: on a miss, ``worked_out`` records
    when and by which renderer the file was computed and
    ``followed`` is null; on a hit, ``followed`` reads the stored
    manifest's worked_out back and ``worked_out`` is null.
    """
    if isinstance(sources, (str, os.PathLike, dict)):
        sources = [sources]
    sources = list(sources)
    packed = apply_pack(spec, pack)
    key = render_cache_key(sources, packed, pack)
    out_path, manifest_path = _cache_paths(cache_dir, key)
    if out_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text("utf-8"))
        # 1.1b(b) honesty, hit side: report what this hit *followed*
        # — the stored manifest's worked_out record. A manifest from
        # before the honesty fields existed says so instead of
        # inventing a provenance.
        stored_worked = manifest.get("worked_out") or {}
        followed = {
            "cache_key": key,
            "worked_out_at": stored_worked.get("at"),
            "renderer": stored_worked.get("renderer")
            or "unknown (pre-honesty manifest)",
        }
        return {"output": str(out_path), "cache_key": key,
                "cached": True, "manifest": manifest,
                "worked_out": None, "followed": followed}
    if renderer is None:
        from .spec import render as _render_spec  # lazy: avoid a cycle
        renderer = _render_spec
        renderer_name = "edit_engine.spec.render"
    else:
        renderer_name = (
            f"{getattr(renderer, '__module__', '?')}."
            f"{getattr(renderer, '__qualname__', type(renderer).__name__)}")
    tmp = Path(tempfile.mkdtemp(prefix="pure-render-"))
    try:
        tmp_out = tmp / "render.mp4"
        produced = renderer(EditSpec.from_dict({**packed, "out": str(tmp_out)}))
        produced_path = Path(produced)
        if not produced_path.exists():
            raise MediaEditError(
                f"renderer returned {produced!r} but no file exists")
        shutil.move(str(produced_path), str(out_path))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 1.1b(b) honesty, miss side: the manifest records that this
    # render was actually worked out, when, and by which renderer;
    # `followed` is null — nothing was followed, it was computed.
    worked_out = {"at": datetime.now(timezone.utc).isoformat(),
                  "renderer": renderer_name, "via": "render_pure"}
    manifest = {"cache_key": key,
                "sources": [source_fingerprint(s) for s in sources],
                "spec": packed, "pack": _canonical(pack or {}),
                "worked_out": worked_out, "followed": None}
    manifest_path.write_text(json.dumps(manifest, indent=2), "utf-8")
    return {"output": str(out_path), "cache_key": key,
            "cached": False, "manifest": manifest,
            "worked_out": worked_out, "followed": None}


def render_proxy_then_full(sources: Any, spec: EditSpec | dict[str, Any],
                           pack: dict[str, Any] | None = None, *,
                           cache_dir: str | Path,
                           renderer: Any = None) -> dict[str, Any]:
    """Proxy workflow: 480p preview first, then the full-res render.

    Both stages go through :func:`render_pure`, so a re-run with an
    unchanged triple is two cache hits and zero ffmpeg. Returns
    ``{"proxy": <render_pure result>, "full": <render_pure result>}``.
    """
    proxy = render_pure(sources, proxy_spec(spec), pack,
                        cache_dir=cache_dir, renderer=renderer)
    full = render_pure(sources, spec, pack,
                       cache_dir=cache_dir, renderer=renderer)
    return {"proxy": proxy, "full": full}
