"""Artifact packs — files in, files out, through the real system.

A pack is a bundle (the Archivist's bundle seam: manifest + SHA-256
per member, atomic publish, refuse-whole on unsafe members) with a
file-class manifest: every member carries its File card (id, title,
kind, shelf, provenance, tags) alongside the bundle's own name/size/
hash. Packing resolves human refs — paths, ``museum:<id>``,
``library:<slug>``, ``archive:<path>::<member>`` — so a museum
exhibit, a library book, and a download pack into one artifact the
same way. Unpacking verifies first (a bundle whose manifest does not
match its bytes is refused whole, naming the offender), extracts
under the Archivist's safety semantics, then verifies the landed
files on disk and reports good/bad/missing honestly — a partial
landing is named, never silent.

Packs operate on home paths (museum cards, downloads, the file
store), so they use the bundle seam's own file functions, not the
Archivist — the Archivist is workspace-confined and these files live
under the home (the Museum lane's note, same justification).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .file import File, FileError, resolve
from .store import FileStore

__all__ = ["pack_files", "unpack_pack", "pack_info"]

_PACK_KIND = "files-artifact"


def _member_name(f: File, seen: set[str]) -> str:
    """Pack layout preserves the shelf: ``<shelf>/<name>``. A second
    file with the same name lands as ``<shelf>/<name> (2)`` — never a
    silent overwrite, never a refused pack."""
    base = f"{f.shelf}/{f.name}"
    if base not in seen:
        seen.add(base)
        return base
    stem, suffix = os.path.splitext(f.name)
    n = 2
    while True:
        candidate = f"{f.shelf}/{stem} ({n}){suffix}"
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        n += 1


def pack_files(context: Any, refs: list[str], dest: str = "",
               *, title: str = "", container: str = "zip",
               store: FileStore | None = None) -> dict[str, Any]:
    """Pack resolved files into one artifact archive. ``dest`` defaults
    to ``<store>/packs/<title>.zip``. Returns the pack manifest (with
    the file cards under ``files``) plus the pack path."""
    from ..archives import create_bundle

    if not refs:
        raise FileError("pack needs at least one file ref")
    files = [resolve(context, r) for r in refs]
    title = (title or "").strip() or f"pack-{len(files)}-files"
    own = store or FileStore(context)
    if dest:
        dest_p = Path(os.path.expanduser(dest))
        if not dest_p.is_absolute():
            from ..tools.filesystem import safe_path

            dest_p = safe_path(context, dest)
    else:
        safe_title = "".join(
            c if c.isalnum() or c in "-_" else "-" for c in title
        ).strip("-") or "pack"
        ext = "zip" if (container or "zip").lower() in (
            "zip",) else "tar.gz"
        dest_p = own.packs_dir / f"{safe_title}.{ext}"
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    members: list[tuple[str, Any]] = []
    cards: list[dict[str, Any]] = []
    for f in files:
        name = _member_name(f, seen)
        source: Any = f.read_bytes() if f.is_archive_member else f.path
        members.append((name, source))
        card = f.to_card()
        card["member"] = name
        cards.append(card)
    # The file cards ride as a member, not a create_bundle kwarg —
    # the bundle seam's manifest is the per-file name/size/hash
    # truth; the cards (kind/shelf/provenance/tags) are the
    # file-class manifest alongside it, verified by the same hashes.
    files_json = json.dumps(
        {"pack": {"title": title, "kind": _PACK_KIND},
         "files": cards},
        indent=2, ensure_ascii=False).encode("utf-8")
    members.append((".files.json", files_json))
    manifest = create_bundle(
        dest_p, members, created_by="files", kind=_PACK_KIND,
        container=container)
    manifest["path"] = str(dest_p)
    manifest["pack"] = {"title": title, "kind": _PACK_KIND}
    manifest["files"] = cards
    return manifest


def pack_info(archive: str | Path) -> dict[str, Any]:
    """Read a pack's manifest without extracting it. Legacy bundles
    (no manifest) report themselves as legacy, honestly. A file
    pack's cards ride as the ``.files.json`` member and surface as
    ``pack["files"]`` / ``pack["pack"]``."""
    from ..archives import open_bundle, read_bundle_members

    p = Path(archive)
    if not p.is_file():
        raise FileError(f"no such pack: {archive}")
    info = open_bundle(p)
    if not info.get("legacy"):
        try:
            raw = read_bundle_members(p).get(".files.json")
            if raw:
                parsed = json.loads(raw.decode("utf-8"))
                info["pack"] = parsed.get("pack", {})
                info["files"] = parsed.get("files", [])
        except Exception:  # noqa: BLE001 — a pack without cards is a pack
            pass
    return info


def unpack_pack(context: Any, archive: str, dest: str = "",
                *, shelf: str = "",
                store: FileStore | None = None) -> dict[str, Any]:
    """Unpack an artifact archive onto a shelf (or into ``dest``).
    Verifies the bundle against its manifest first — a mismatch
    refuses the whole pack, naming the offender — then extracts
    under the bundle seam's safety semantics and verifies the landed
    files' hashes on disk. Returns the extraction report plus a
    per-file verification list; a partial landing is named, never
    silent."""
    from ..archives import extract_bundle, sha256_file, verify_bundle

    own = store or FileStore(context)
    apath = resolve(context, archive).path
    if not apath.is_file():
        raise FileError(f"no such pack: {archive}")
    verdict = verify_bundle(apath)
    if not verdict.get("ok"):
        problems = verdict.get("problems") or ["unverifiable bundle"]
        raise FileError(
            f"pack {apath.name} refused: " + "; ".join(problems[:5]))
    if dest:
        dest_p = Path(os.path.expanduser(dest))
        if not dest_p.is_absolute():
            from ..tools.filesystem import safe_path

            dest_p = safe_path(context, dest)
    elif shelf:
        dest_p = own.shelf_dir(shelf.strip())
    else:
        dest_p = own.root / "unpacked" / apath.stem
    out = extract_bundle(apath, dest_p)
    manifest = out.get("manifest") or {}
    recorded = {str(m["name"]): m
                for m in manifest.get("members", [])}
    verified: list[dict[str, Any]] = []
    problems: list[str] = []
    for name in out.get("files") or []:
        landed = Path(out["dest"]) / name
        entry = recorded.get(name, {})
        ok = True
        reason = ""
        if not landed.is_file():
            ok, reason = False, "missing after extraction"
        elif entry.get("sha256"):
            try:
                if sha256_file(landed) != entry["sha256"]:
                    ok, reason = False, "sha256 mismatch on disk"
            except OSError as exc:
                ok, reason = False, f"unreadable: {exc}"
        verified.append({"member": name, "ok": ok,
                         "reason": reason})
        if not ok:
            problems.append(f"{name}: {reason}")
    return {"path": str(apath), "dest": out["dest"],
            "format": out.get("format"),
            "extracted": out.get("extracted", 0),
            "bytes": out.get("bytes", 0),
            "files": out.get("files") or [],
            "manifest": manifest,
            "legacy": out.get("legacy", True),
            "verified": verified,
            "problems": problems}
