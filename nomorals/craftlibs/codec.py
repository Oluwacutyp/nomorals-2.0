"""The shared ZIP package codec (Craft-Libs port, census 3A.20).

One home for the deterministic package shape the craft organs share:
``document.json`` + ``meta.json`` (+ ``assets/*``), canonical JSON,
fixed zip timestamps, deterministic bytes, so an unedited document
writes back byte-identical. The cad organ (``.cadcraft``) and the
layout organ (``.designcraft``) both used this shape with their own
copies; this module is the single implementation now, and both
organs delegate to it, translating only the error type.

Ported from the cad + layout organs (Devon lineage); the shape
itself follows the layout organ's zip shape (lane 104).
"""

from __future__ import annotations

import io
import json
import zipfile

__all__ = [
    "CodecError",
    "MAX_JSON_DEPTH",
    "MAX_JSON_ELEMENTS",
    "MAX_PACKAGE_BYTES",
    "MAX_PART_BYTES",
    "bounded_json_loads",
    "canonical_json",
    "read_document_package",
    "read_package",
    "write_package",
    "write_package_ordered",
]

MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_PART_BYTES = 32 * 1024 * 1024
MAX_JSON_DEPTH = 100
MAX_JSON_ELEMENTS = 500_000

_DATE = (1980, 1, 1, 0, 0, 0)


class CodecError(Exception):
    """A package that cannot be read or written. Organs translate
    this into their own error type (``CadError`` / ``LayoutError``)
    with their own kind; the message is kept verbatim."""


def canonical_json(obj) -> bytes:
    """Deterministic JSON: sorted keys, tight separators, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def bounded_json_loads(data: bytes):
    """Parse JSON with a depth/element budget so a hostile file
    cannot blow up the parser (same habit as the word organ's
    bounded XML). Raises :class:`CodecError`."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CodecError(f"document.json is not UTF-8: {exc}") from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CodecError(f"document.json is not valid JSON: {exc}") from exc
    count = [0]

    def walk(node, depth: int) -> None:
        if depth > MAX_JSON_DEPTH:
            raise CodecError("document.json nests too deep")
        count[0] += 1
        if count[0] > MAX_JSON_ELEMENTS:
            raise CodecError("document.json has too many elements")
        if isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                walk(value, depth + 1)

    walk(obj, 1)
    return obj


def read_package(data: bytes) -> dict[str, bytes]:
    """Read a package zip into {part name: bytes}."""
    if len(data) > MAX_PACKAGE_BYTES:
        raise CodecError(
            f"package is {len(data)} bytes (max {MAX_PACKAGE_BYTES})")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise CodecError(f"not a package: {exc}") from exc
    parts: dict[str, bytes] = {}
    try:
        for info in archive.infolist():
            if info.is_dir():
                continue
            if info.file_size > MAX_PART_BYTES:
                raise CodecError(f"part {info.filename} is too large")
            parts[info.filename] = archive.read(info.filename)
    except CodecError:
        raise
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        raise CodecError(f"cannot read package: {exc}") from exc
    return parts


def write_package(parts: dict[str, bytes]) -> bytes:
    """Write parts deterministically (fixed timestamps, sorted)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(parts):
            info = zipfile.ZipInfo(name, date_time=_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, parts[name])
    return buf.getvalue()


def write_package_ordered(entries) -> bytes:
    """Write parts deterministically in the given order (the layout
    organ's ``document.json``, ``meta.json``, ``assets/*`` order,
    which differs from sorted order)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, data)
    return buf.getvalue()


def read_document_package(data: bytes) -> tuple[dict, dict[str, bytes], dict]:
    """Read a ``document.json`` + ``meta.json`` + ``assets/*``
    package into (document dict, assets, meta dict)."""
    parts = read_package(data)
    raw = parts.get("document.json")
    if raw is None:
        raise CodecError("package has no document.json")
    doc = bounded_json_loads(raw)
    meta: dict = {}
    raw_meta = parts.get("meta.json")
    if raw_meta is not None:
        meta = bounded_json_loads(raw_meta)
    assets = {name: blob for name, blob in parts.items()
              if name.startswith("assets/")}
    return doc, assets, meta
