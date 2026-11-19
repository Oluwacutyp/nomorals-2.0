"""Craft-Fonts: the craft-fonts catalog as a brain tool (Craft-Fonts port
of storytold/craft-fonts — reusable OFL fonts for the Crafting Apps).

A thin surface over :mod:`nomorals.media.typography.craftfonts` —
the organ owns the manifest format, the three catalog tables,
checkout loading, install and the embed bundle; the tool only
routes arguments and shapes the reply. Devon already had a
typography organ (catalogue parsed from host fonts, stocked
library, selection, service); this tool adds what it lacked: the
craft-fonts catalog for the scripts the stocked core does not
cover, checkout loading from ``CRAFT_FONTS_DIR``, install into
the font store, and embed bundles. Pure local work, no upload.
Installs and bundles write files, so FS_WRITE.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.typography.craftfonts import (
    LoadError,
    build_embed,
    check_checkout,
    default_catalog,
    dir_from_env,
    install_checkout,
    load,
    run_parity,
    verify_attribution,
)

__all__ = ["register"]

_ACTIONS = ("catalog", "for_script", "check", "load", "install",
            "bundle", "render", "preview", "health", "parity",
            "attribution")


def _home(params: dict[str, Any]) -> Path:
    override = params.get("home") or os.environ.get("NM_HOME", "")
    if override:
        return Path(str(override)).expanduser()
    return Path.home() / ".nomorals"


def _store(params: dict[str, Any]) -> Path:
    return _home(params) / "fonts" / "craftfonts"


def _checkout(params: dict[str, Any],
              sources: list[str] | None) -> Path:
    if sources:
        return Path(sources[0]).expanduser()
    override = params.get("checkout") or ""
    if override:
        return Path(str(override)).expanduser()
    found = dir_from_env()
    if found is not None:
        return found
    raise ValueError("craftfonts needs a checkout path "
                     "(sources=[<dir>] or CRAFT_FONTS_DIR)")


def _installed_face(store: Path, family: str) -> dict[str, Any] | None:
    record_path = store / "install-record.json"
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    want = " ".join(str(family).split()).casefold()
    for entry in record.get("installed", []):
        if " ".join(str(entry.get("family", "")).split()).casefold() == want:
            return dict(entry)
    return None


def _preview_service(params: dict[str, Any]):
    """The font-preview service for this call.

    A ``home`` override (tests, alternate homes) gets a fresh
    service rooted at that home; otherwise the process-wide one
    — its catalogue build is the expensive part and is cached
    per instance.
    """
    from ..media.typography.preview import (
        FontPreviewService,
        get_preview_service,
    )
    if params.get("home"):
        from ..media.typography.service import TypographyService
        return FontPreviewService(
            TypographyService(store_dir=_home(params) / "typography"))
    return get_preview_service()


def register(registry: Any) -> None:
    """Attach the craftfonts family to a registry."""
    @registry.register(
        "craftfonts",
        description=(
            "Reusable OFL fonts for the Crafting Apps — the craft-fonts "
            "catalog (default 13 faces for Japanese / Chinese / Korean / "
            "Arabic / Thai / Georgian / Hebrew / symbols, opt-in Latin "
            "51 faces, opt-in Simplified Chinese 1 face), checkout "
            "loading from CRAFT_FONTS_DIR, install into the font store, "
            "and embed bundles. Craft-Fonts actions: catalog=<the three "
            "tables and their families>; for_script=<faces for an ISO "
            "15924 script, value=<script>>; check=<verify a checkout's "
            "files against its manifest, sources=[<dir>]>; load=<load a "
            "checkout, sources=[<dir>]>; install=<verify then install a "
            "checkout into the font store, sources=[<dir>], params=<"
            "{home: <dir>}>>; bundle=<build an embed payload zip, "
            "sources=[<dir>], params=<{allowlist: [[family, style], ...], "
            "output: <path>}>>; render=<render text with an installed "
            "face, value=<text>, params=<{family, size, home, output: "
            "<path>}>>; preview=<render a text sample in a named font "
            "at a size/style into a PNG preview through the "
            "font-preview service (any catalogue family, not just "
            "installed craft-fonts faces), value=<text>, params=<"
            "{family, size, weight, style: regular|bold|italic|"
            "bold-italic, format: png, max_width, home, output: "
            "<path>}>>; health=<preview-service health: status, fonts "
            "indexed, cache stats, shaping engine>; parity=<run the "
            "machine-checked parity TOML>; attribution=<verify the "
            "SHA-256 attribution DB>. action=<catalog|for_script|check|"
            "load|install|bundle|render|preview|health|parity|"
            "attribution>; sources=<list of checkout paths>; output=<"
            "path to write>; value=<script code or text>; params=<dict>."
        ),
        capability=Capability.FS_WRITE,
    )
    def craftfonts_tool(action: str = "catalog",
                        sources: list[str] | None = None,
                        output: str = "", kind: str = "",
                        node: str = "", nodes: str | list[str] | None = None,
                        value: str = "", x: float = 0.0, y: float = 0.0,
                        command: str = "",
                        params: dict | None = None) -> dict[str, Any]:
        opts = dict(params or {})
        if action == "catalog":
            catalog = default_catalog()
            return {"ok": True,
                    "default_faces": len(catalog.default_faces),
                    "latin_faces": len(catalog.latin_faces),
                    "chinese_faces": len(catalog.chinese_faces),
                    "default_families": len(
                        {f.family for f in catalog.default_faces}),
                    "families": catalog.families()}
        if action == "for_script":
            catalog = default_catalog()
            faces = catalog.for_script(value or str(opts.get("script", "")))
            return {"ok": True, "script": value,
                    "faces": [entry.to_dict() for entry in faces]}
        if action == "parity":
            return run_parity()
        if action == "attribution":
            return verify_attribution()
        if action == "check":
            return check_checkout(_checkout(opts, sources))
        if action == "load":
            try:
                loaded = load(_checkout(opts, sources))
            except LoadError as exc:
                return {"ok": False, "error": str(exc),
                        "error_kind": type(exc).__name__}
            return {"ok": True, "faces": len(loaded),
                    "families": sorted({e.family for e, _ in loaded})}
        if action == "install":
            try:
                installed = install_checkout(_checkout(opts, sources),
                                             _store(opts))
            except LoadError as exc:
                return {"ok": False, "error": str(exc),
                        "error_kind": type(exc).__name__}
            return {"ok": True, "installed": len(installed),
                    "store": str(_store(opts)),
                    "families": sorted({e.family for e in installed})}
        if action == "bundle":
            checkout = _checkout(opts, sources)
            try:
                loaded = load(checkout)
                allowlist = opts.get("allowlist")
                payload = build_embed(
                    loaded, root=checkout,
                    allowlist=[tuple(item) for item in allowlist]
                    if allowlist else None)
            except (LoadError, ValueError) as exc:
                return {"ok": False, "error": str(exc),
                        "error_kind": type(exc).__name__}
            digest = hashlib.sha256(payload).hexdigest()
            target = output or str(opts.get("output", ""))
            if target:
                Path(target).expanduser().write_bytes(payload)
            return {"ok": True, "bytes": len(payload),
                    "sha256": digest,
                    "output": target or None}
        if action == "render":
            family = str(opts.get("family", "") or "Noto Sans Hebrew")
            size = max(8, int(opts.get("size", 32)))
            store = _store(opts)
            entry = _installed_face(store, family)
            if entry is None:
                return {"ok": False,
                        "error": f"craftfonts: family {family!r} is not "
                        f"installed in {store} — install a checkout "
                        "first (action=install)",
                        "error_kind": "FontNotInstalled"}
            font_path = store / str(entry["file"])
            if not font_path.is_file():
                return {"ok": False,
                        "error": f"craftfonts: installed face file "
                        f"{font_path} is missing",
                        "error_kind": "FontNotInstalled"}
            from PIL import Image, ImageDraw, ImageFont

            font = ImageFont.truetype(str(font_path), size)
            text = value or "Hello"
            bbox = font.getbbox(text)
            width = max(1, bbox[2] - bbox[0] + 16)
            height = max(1, bbox[3] - bbox[1] + 16)
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)
            draw.text((8 - bbox[0], 8 - bbox[1]), text, font=font,
                      fill="black")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            png = buffer.getvalue()
            target = output or str(opts.get("output", ""))
            if target:
                Path(target).expanduser().write_bytes(png)
            return {"ok": True, "family": entry["family"],
                    "style": entry["style"],
                    "licence": entry["licence"],
                    "png_bytes": len(png),
                    "png_sha256": hashlib.sha256(png).hexdigest(),
                    "output": target or None}
        if action == "health":
            return {"ok": True, **_preview_service(opts).health()}
        if action == "preview":
            from ..media.typography.fontlib import FontNotStocked
            from ..media.typography.preview import (
                PreviewBusy,
                PreviewError,
                PreviewRequest,
            )

            def _as_int(raw: Any, name: str) -> int | None:
                if raw is None:
                    return None
                try:
                    return int(raw)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    raise PreviewError(
                        "craftfonts preview: "
                        f"{name} {raw!r} is not a number") from None

            try:
                request = PreviewRequest(
                    family=str(opts.get("family", "") or ""),
                    text=value or str(opts.get("text", "")),
                    size=_as_int(opts.get("size", 48), "size"),
                    weight=_as_int(opts.get("weight"), "weight"),
                    style=str(opts.get("style", "regular") or "regular"),
                    format=str(opts.get("format", "png") or "png"),
                    max_width=_as_int(opts.get("max_width"), "max_width"))
                result = _preview_service(opts).preview(request)
            except (FontNotStocked, PreviewError, PreviewBusy,
                    RuntimeError, TypeError, ValueError) as exc:
                return {"ok": False, "error": str(exc),
                        "error_kind": type(exc).__name__}
            payload = result.to_dict()
            target = output or str(opts.get("output", ""))
            if target:
                Path(target).expanduser().write_bytes(
                    Path(result.path).read_bytes())
            payload["output"] = target or None
            return {"ok": True, **payload}
        raise ValueError(f"unknown craftfonts action: {action}")
