"""ApkForge editor — apktool decode + journaled, reversible edits.

Every edit follows the same contract (design §3.3/§3.4):

1. the decoded project must exist (decode is its own journaled op);
2. each file the edit touches is snapshotted first
   (``snapshots/edit-<id>/``) with before/after sha256;
3. the edit lands as structured manipulation (ElementTree for the
   manifest and strings — never regex);
4. one row is appended to the project journal carrying the edit id,
   so ``undo`` can restore the exact pre-edit bytes — undo is itself
   a journaled op, and an edit stays visibly undone, never erased.

Resource edits are confined to ``res/``, ``assets/`` and ``lib/``
inside the decoded project; anything outside those trees or outside
the project refuses by name (path guard, same idiom as intake's
zip-slip guard).
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .project import Project, sha256_file
from .toolchain import ToolchainError, run_tool

__all__ = [
    "EditError",
    "decoded_dir",
    "ensure_decoded",
    "manifest_path",
    "edit_manifest",
    "edit_strings",
    "resource_replace",
    "resource_add",
    "resource_remove",
    "apply_edit",
    "undo_edit",
    "list_edits",
    "ANDROID_NS",
]

ANDROID_NS = "http://schemas.android.com/apk/res/android"
_A = "{%s}" % ANDROID_NS
ET.register_namespace("android", ANDROID_NS)

_RESOURCE_TREES = ("res", "assets", "lib")


class EditError(RuntimeError):
    """A designed edit failure — the message names the exact cause."""


def decoded_dir(proj: Project) -> Path:
    return proj.dir / "decoded"


def manifest_path(proj: Project) -> Path:
    return decoded_dir(proj) / "AndroidManifest.xml"


# ── decode ───────────────────────────────────────────────────────────

def ensure_decoded(root: Path, proj: Project, *,
                   force: bool = False) -> Path:
    """Decode the project's original APK with apktool (idempotent:
    a decode of the same source sha is reused)."""
    ddir = decoded_dir(proj)
    fresh = (ddir / "apktool.yml").exists() and \
        proj.record.get("decoded_from_sha") == proj.sha256
    if fresh and not force:
        return ddir
    target = proj.original_path
    if proj.kind != "apk":
        raise EditError(
            f"project kind is {proj.kind!r} — decode works on a "
            "single APK; for a bundle, extract or merge a split "
            "first (splits ops)")
    if not target.exists():
        raise EditError(f"original APK missing at {target}")
    shutil.rmtree(ddir, ignore_errors=True)
    try:
        res = run_tool("", "apktool",
                       ["d", "-f", "-o", str(ddir), str(target)],
                       timeout=900, root=root)
    except ToolchainError as exc:
        raise EditError(f"decode step failed: {exc}") from exc
    if res["rc"] != 0 or not manifest_path(proj).exists():
        tail = (res["stdout"] + res["stderr"]).strip().splitlines()
        raise EditError(
            "apktool decode failed: "
            + (tail[-1][:300] if tail else f"rc={res['rc']}"))
    proj.record["decoded_from_sha"] = proj.sha256
    proj.save()
    proj.journal("decode",
                 f"apktool d {target.name} — source sha "
                 f"{proj.sha256[:16]}…")
    return ddir


# ── snapshot / journal plumbing ──────────────────────────────────────

def _next_edit_id(proj: Project) -> int:
    seq = int(proj.record.get("edit_seq") or 0) + 1
    proj.record["edit_seq"] = seq
    proj.save()
    return seq


def _snapshot(proj: Project, edit_id: int,
              rel_paths: list[str]) -> dict[str, Any]:
    """Copy the pre-edit bytes of each touched file into
    snapshots/edit-<id>/ and describe the pre-state."""
    snap_dir = proj.dir / "snapshots" / f"edit-{edit_id}"
    files: list[dict[str, Any]] = []
    for rel in rel_paths:
        src = decoded_dir(proj) / rel
        entry = {"path": rel, "existed": src.exists(),
                 "before_sha256": sha256_file(src) if src.exists()
                 else ""}
        if src.exists():
            dst = snap_dir / "files" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        files.append(entry)
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "meta.json").write_text(
        json.dumps({"edit_id": edit_id, "files": files}, indent=1),
        encoding="utf-8")
    return {"snapshot": str(snap_dir), "files": files}


def apply_edit(proj: Project, op: str, detail: str,
               rel_paths: list[str], mutate: Any,
               **extra: Any) -> dict[str, Any]:
    """The one edit path: snapshot → mutate → hash → journal.

    ``mutate()`` performs the change inside the decoded project.
    If it raises, the snapshot is restored before the error
    propagates — a failed edit never leaves a half-written tree."""
    edit_id = _next_edit_id(proj)
    snap = _snapshot(proj, edit_id, rel_paths)
    try:
        mutate()
    except Exception:
        _restore(proj, edit_id, snap)
        raise
    files = []
    for entry in snap["files"]:
        p = decoded_dir(proj) / entry["path"]
        entry = dict(entry)
        entry["after_sha256"] = sha256_file(p) if p.exists() else ""
        files.append(entry)
    row = {"edit_id": edit_id, "files": files,
           "snapshot": snap["snapshot"]}
    row.update(extra)
    proj.journal(op, detail, **row)
    return {"edit_id": edit_id, "op": op, "detail": detail,
            "files": files, "snapshot": snap["snapshot"]}


def _restore(proj: Project, edit_id: int,
             snap: dict[str, Any] | None = None) -> None:
    if snap is None:
        meta_path = (proj.dir / "snapshots" / f"edit-{edit_id}"
                     / "meta.json")
        snap = json.loads(meta_path.read_text(encoding="utf-8"))
    for entry in snap["files"]:
        target = decoded_dir(proj) / entry["path"]
        if entry["existed"]:
            src = (proj.dir / "snapshots" / f"edit-{edit_id}"
                   / "files" / entry["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
        else:
            target.unlink(missing_ok=True)


def undone_edit_ids(proj: Project) -> set[int]:
    out: set[int] = set()
    for row in proj.journal_rows(limit=100000):
        if row.get("op") == "undo" and row.get("edit_id") is not None:
            out.add(int(row["edit_id"]))
    return out


def list_edits(proj: Project) -> list[dict[str, Any]]:
    undone = undone_edit_ids(proj)
    edits = []
    for row in proj.journal_rows(limit=100000):
        eid = row.get("edit_id")
        if eid is None or row.get("op") in ("undo", "decode",
                                            "intake", "dissect",
                                            "teach", "diff"):
            continue
        entry = dict(row)
        entry["undone"] = int(eid) in undone
        edits.append(entry)
    return edits


def undo_edit(proj: Project, edit_id: int | None = None) -> dict[str, Any]:
    """Restore the pre-edit snapshot of one edit (default: the most
    recent edit not already undone).  Journaled as op=undo."""
    edits = [e for e in list_edits(proj) if not e["undone"]]
    if edit_id is None:
        if not edits:
            raise EditError("nothing to undo — no live edits")
        target = edits[-1]
    else:
        match = [e for e in list_edits(proj)
                 if int(e["edit_id"]) == int(edit_id)]
        if not match:
            raise EditError(f"no edit #{edit_id} in this project")
        if match[0]["undone"]:
            raise EditError(f"edit #{edit_id} is already undone")
        target = match[0]
    eid = int(target["edit_id"])
    snap_dir = proj.dir / "snapshots" / f"edit-{eid}"
    if not (snap_dir / "meta.json").exists():
        raise EditError(
            f"snapshot for edit #{eid} is missing at {snap_dir} — "
            "cannot undo exactly, refusing to guess")
    _restore(proj, eid)
    proj.journal("undo", f"reverted edit #{eid} ({target.get('op')})",
                 edit_id=eid, reverts=eid)
    return {"undone_edit_id": eid, "op": target.get("op"),
            "detail": target.get("detail", "")}


# ── manifest edits (structured XML, never regex) ─────────────────────

def _load_manifest(proj: Project) -> ET.ElementTree:
    path = manifest_path(proj)
    if not path.exists():
        raise EditError("project is not decoded — run decode first")
    return ET.parse(path)


def _save_manifest(proj: Project, tree: ET.ElementTree) -> None:
    tree.write(manifest_path(proj), encoding="utf-8",
               xml_declaration=True)


def edit_manifest(proj: Project, **changes: Any) -> dict[str, Any]:
    """Structured manifest edits.  Supported changes: label,
    version_name, version_code, min_sdk, target_sdk, debuggable,
    allow_backup, add_permission, remove_permission,
    add_component (JSON string of {tag, name, exported?}),
    remove_component (component class name)."""
    allowed = {"label", "version_name", "version_code", "min_sdk",
               "target_sdk", "debuggable", "allow_backup",
               "add_permission", "remove_permission", "add_component",
               "remove_component"}
    unknown = set(changes) - allowed
    if unknown:
        raise EditError(f"unknown manifest change(s): "
                        f"{sorted(unknown)} — supported: "
                        f"{sorted(allowed)}")
    if not changes:
        raise EditError("edit manifest needs at least one change")
    rel = "AndroidManifest.xml"
    tree = _load_manifest(proj)
    root = tree.getroot()
    applied: list[str] = []

    def mutate() -> None:
        if "label" in changes:
            app = root.find("application")
            if app is None:
                raise EditError("manifest has no <application>")
            app.set(_A + "label", str(changes["label"]))
            applied.append(f"label={changes['label']!r}")
        if "version_name" in changes:
            root.set(_A + "versionName", str(changes["version_name"]))
            applied.append(f"versionName={changes['version_name']}")
        if "version_code" in changes:
            root.set(_A + "versionCode", str(int(changes["version_code"])))
            applied.append(f"versionCode={changes['version_code']}")
        if "min_sdk" in changes or "target_sdk" in changes:
            sdk = root.find("uses-sdk")
            if sdk is None:
                sdk = ET.SubElement(root, "uses-sdk")
            if "min_sdk" in changes:
                sdk.set(_A + "minSdkVersion", str(int(changes["min_sdk"])))
                applied.append(f"minSdk={changes['min_sdk']}")
            if "target_sdk" in changes:
                sdk.set(_A + "targetSdkVersion",
                        str(int(changes["target_sdk"])))
                applied.append(f"targetSdk={changes['target_sdk']}")
        for flag in ("debuggable", "allow_backup"):
            if flag in changes:
                app = root.find("application")
                if app is None:
                    raise EditError("manifest has no <application>")
                attr = "allowBackup" if flag == "allow_backup" \
                    else "debuggable"
                val = "true" if changes[flag] else "false"
                app.set(_A + attr, val)
                applied.append(f"{attr}={val}")
        if changes.get("add_permission"):
            perm = str(changes["add_permission"])
            existing = {el.get(_A + "name")
                        for el in root.findall("uses-permission")}
            if perm in existing:
                raise EditError(f"permission {perm} already declared")
            el = ET.Element("uses-permission")
            el.set(_A + "name", perm)
            idx = 0
            for i, child in enumerate(root):
                if child.tag in ("uses-permission", "uses-sdk"):
                    idx = i + 1
            root.insert(idx, el)
            applied.append(f"+permission {perm}")
        if changes.get("remove_permission"):
            perm = str(changes["remove_permission"])
            hit = None
            for el in root.findall("uses-permission"):
                if el.get(_A + "name") == perm:
                    hit = el
                    break
            if hit is None:
                raise EditError(f"permission {perm} is not declared")
            root.remove(hit)
            applied.append(f"-permission {perm}")
        if changes.get("add_component"):
            spec = changes["add_component"]
            if isinstance(spec, str):
                spec = json.loads(spec)
            tag = str(spec.get("tag") or "activity")
            if tag not in ("activity", "service", "receiver",
                           "provider"):
                raise EditError(f"component tag {tag!r} not supported "
                                "(activity|service|receiver|provider)")
            app = root.find("application")
            if app is None:
                raise EditError("manifest has no <application>")
            el = ET.SubElement(app, tag)
            el.set(_A + "name", str(spec["name"]))
            if "exported" in spec:
                el.set(_A + "exported",
                       "true" if spec["exported"] else "false")
            if tag == "provider":
                el.set(_A + "authorities",
                       str(spec.get("authorities")
                           or (proj.package + ".added")))
            applied.append(f"+component {tag} {spec['name']}")
        if changes.get("remove_component"):
            name = str(changes["remove_component"])
            app = root.find("application")
            hit = None
            if app is not None:
                for el in app:
                    if el.get(_A + "name") == name:
                        hit = (app, el)
                        break
            if hit is None:
                raise EditError(f"component {name} not in manifest")
            hit[0].remove(hit[1])
            applied.append(f"-component {name}")
        _save_manifest(proj, tree)

    return apply_edit(proj, "edit_manifest",
                      "; ".join(applied) if applied else "manifest",
                      [rel], mutate)


# ── strings edits ────────────────────────────────────────────────────

def _strings_path(proj: Project, locale: str = "") -> Path:
    folder = "values" + (f"-{locale}" if locale else "")
    return decoded_dir(proj) / "res" / folder / "strings.xml"


def edit_strings(proj: Project, name: str, value: str, *,
                 locale: str = "") -> dict[str, Any]:
    if not name:
        raise EditError("strings edit needs a string name")
    path = _strings_path(proj, locale)
    rel = str(path.relative_to(decoded_dir(proj)))

    def mutate() -> None:
        if path.exists():
            tree = ET.parse(path)
            root = tree.getroot()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            root = ET.Element("resources")
            tree = ET.ElementTree(root)
        hit = None
        for el in root.findall("string"):
            if el.get("name") == name:
                hit = el
                break
        if hit is None:
            hit = ET.SubElement(root, "string")
            hit.set("name", name)
        hit.text = value
        tree.write(path, encoding="utf-8", xml_declaration=True)

    return apply_edit(proj, "edit_strings",
                      f"strings[{locale or 'default'}].{name} set",
                      [rel], mutate)


def read_string(proj: Project, name: str, *,
                locale: str = "") -> str | None:
    path = _strings_path(proj, locale)
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    for el in root.findall("string"):
        if el.get("name") == name:
            return el.text or ""
    return None


# ── resource file ops ────────────────────────────────────────────────

def _guard_rel(rel_path: str) -> str:
    rel = rel_path.replace("\\", "/").lstrip("/")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise EditError(f"resource path {rel_path!r} climbs out of "
                        "the decoded project — refused")
    if not parts or parts[0] not in _RESOURCE_TREES:
        raise EditError(
            f"resource path must live under "
            f"{'/, '.join(_RESOURCE_TREES)}/ — got {rel_path!r}")
    return "/".join(parts)


def resource_replace(proj: Project, rel_path: str,
                     source_file: str) -> dict[str, Any]:
    rel = _guard_rel(rel_path)
    src = Path(source_file).expanduser()
    if not src.exists():
        raise EditError(f"replacement file does not exist: {src}")
    target = decoded_dir(proj) / rel
    if not target.exists():
        raise EditError(f"{rel} does not exist in the decoded "
                        "project — use resource add instead")

    def mutate() -> None:
        shutil.copyfile(src, target)

    return apply_edit(proj, "edit_resource",
                      f"replace {rel} from {src.name}", [rel], mutate)


def resource_add(proj: Project, rel_path: str,
                 source_file: str) -> dict[str, Any]:
    rel = _guard_rel(rel_path)
    src = Path(source_file).expanduser()
    if not src.exists():
        raise EditError(f"file to add does not exist: {src}")
    target = decoded_dir(proj) / rel
    if target.exists():
        raise EditError(f"{rel} already exists — use replace")

    def mutate() -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)

    return apply_edit(proj, "edit_resource", f"add {rel}", [rel],
                      mutate)


def resource_remove(proj: Project, rel_path: str) -> dict[str, Any]:
    rel = _guard_rel(rel_path)
    target = decoded_dir(proj) / rel
    if not target.exists():
        raise EditError(f"{rel} does not exist in the decoded project")

    def mutate() -> None:
        target.unlink()

    return apply_edit(proj, "edit_resource", f"remove {rel}", [rel],
                      mutate)
