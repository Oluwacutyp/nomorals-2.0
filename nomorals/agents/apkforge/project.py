"""ApkForge project workspace — the unit of APK work.

One project per source artifact (APK or bundle container).  Layout
(design §3.3)::

    <base>/apkforge/projects/<project_id>/
        project.json     id, identity, provenance, kind, journal head
        original.apk     pristine source — NEVER mutated (or original.<ext>
                         for bundle containers)
        bundle/          extracted bundle parts (bundles only)
        edits.jsonl      append-only op journal
        dossier.json     latest dissection (full)
        dossier.md       latest dissection (owner-readable)
        dossiers/        archived dossiers, one per source sha256

The original is immutable and every later pass (A2 edits/builds) works
on copies, so any output can always be traced back to the exact bytes
that arrived, and how they arrived (``provenance``).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "apkforge_dir",
    "Project",
    "create_project",
    "load_project",
    "list_projects",
    "find_project",
    "sha256_file",
    "museum_card",
    "techforge_feed",
]

JOURNAL = "edits.jsonl"
PROJECT_JSON = "project.json"


def apkforge_dir(context: Any = None, base: str = "") -> Path:
    """The class workspace root.

    Resolution order mirrors the bet desk: explicit ``base`` →
    ``NOMORALS_APKFORGE_DIR`` → the context's settings workspace →
    ``~/.config/nomorals/apkforge``.
    """
    if base:
        root = Path(base)
    elif os.environ.get("NOMORALS_APKFORGE_DIR", "").strip():
        root = Path(os.environ["NOMORALS_APKFORGE_DIR"].strip())
    else:
        ws = getattr(getattr(context, "settings", None), "workspace_dir", "")
        root = (Path(str(ws)) / "apkforge") if ws else (
            Path(os.path.expanduser("~")) / ".config" / "nomorals" / "apkforge")
    root.mkdir(parents=True, exist_ok=True)
    (root / "projects").mkdir(parents=True, exist_ok=True)
    return root


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> float:
    return time.time()


@dataclass
class Project:
    """A landed APK/bundle and everything known about it."""

    id: str
    dir: Path
    record: dict[str, Any] = field(default_factory=dict)

    # ── record conveniences ──────────────────────────────────────────
    @property
    def package(self) -> str:
        return str(self.record.get("package") or "")

    @property
    def title(self) -> str:
        return str(self.record.get("title") or self.record.get("package")
                   or self.id)

    @property
    def kind(self) -> str:
        return str(self.record.get("kind") or "apk")

    @property
    def sha256(self) -> str:
        return str((self.record.get("provenance") or {}).get("sha256") or "")

    @property
    def original_path(self) -> Path:
        name = str(self.record.get("original_name") or "original.apk")
        return self.dir / name

    # ── persistence ──────────────────────────────────────────────────
    def save(self) -> None:
        self.record["updated_at"] = _now()
        tmp = self.dir / (PROJECT_JSON + ".tmp")
        tmp.write_text(json.dumps(self.record, indent=1, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(self.dir / PROJECT_JSON)

    def journal(self, op: str, detail: str = "", **extra: Any) -> None:
        """Append one entry to the append-only op journal."""
        row = {"ts": _now(), "op": op, "detail": detail}
        row.update(extra)
        with open(self.dir / JOURNAL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def journal_rows(self, limit: int = 50) -> list[dict[str, Any]]:
        path = self.dir / JOURNAL
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows[-limit:]

    # ── dossier IO ───────────────────────────────────────────────────
    def write_dossier(self, dossier: dict[str, Any], md_text: str) -> None:
        (self.dir / "dossier.json").write_text(
            json.dumps(dossier, indent=1, sort_keys=True), encoding="utf-8")
        (self.dir / "dossier.md").write_text(md_text, encoding="utf-8")
        archive = self.dir / "dossiers"
        archive.mkdir(exist_ok=True)
        sha = str(dossier.get("source_sha256") or "unknown")[:16]
        (archive / f"{sha}.json").write_text(
            json.dumps(dossier, indent=1, sort_keys=True), encoding="utf-8")
        self.record["dossier_sha256"] = dossier.get("source_sha256")
        self.record["dossier_at"] = dossier.get("dissected_at")
        self.save()

    def read_dossier(self) -> dict[str, Any] | None:
        path = self.dir / "dossier.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def create_project(root: Path, *, source_sha256: str, kind: str,
                   original_name: str,
                   provenance: dict[str, Any]) -> Project:
    """Create the project dir for a freshly landed artifact."""
    pid = f"apk_{source_sha256[:12]}"
    pdir = root / "projects" / pid
    pdir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "id": pid,
        "kind": kind,
        "original_name": original_name,
        "provenance": provenance,
        "created_at": _now(),
        "updated_at": _now(),
    }
    proj = Project(id=pid, dir=pdir, record=record)
    proj.save()
    return proj


def load_project(root: Path, project_id: str) -> Project | None:
    pdir = root / "projects" / project_id
    pj = pdir / PROJECT_JSON
    if not pj.exists():
        return None
    try:
        record = json.loads(pj.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return Project(id=project_id, dir=pdir, record=record)


def list_projects(root: Path) -> list[Project]:
    out: list[Project] = []
    pdir = root / "projects"
    if not pdir.exists():
        return out
    for child in sorted(pdir.iterdir()):
        if child.is_dir():
            proj = load_project(root, child.name)
            if proj is not None:
                out.append(proj)
    out.sort(key=lambda p: float(p.record.get("created_at") or 0),
             reverse=True)
    return out


def find_project(root: Path, ref: str) -> Project | None:
    """Resolve a project by id, id prefix, package name, exact title,
    or a distinctive word of the title."""
    ref = (ref or "").strip()
    if not ref:
        return None
    proj = load_project(root, ref)
    if proj is not None:
        return proj
    cands = list_projects(root)
    for cand in cands:
        if cand.id.startswith(ref) or cand.package == ref:
            return cand
    for cand in cands:
        if (cand.record.get("title") or "").lower() == ref.lower():
            return cand
    titled = [c for c in cands
              if ref.lower() in (c.record.get("title") or "").lower()]
    return titled[0] if titled else None


def museum_card(project: Project,
                dossier: dict[str, Any] | None = None) -> dict[str, Any]:
    """The Museum artifact-card shape (banked registry design): every
    ApkForge output carries it from day one so the registry ingests
    these projects unchanged when it lands."""
    prov = project.record.get("provenance") or {}
    ident = (dossier or {}).get("identity") or {}
    tags = ["apk", project.kind]
    if project.package:
        tags.append(project.package)
    if ident.get("version_name"):
        tags.append(f"v{ident['version_name']}")
    return {
        "id": project.id,
        "title": project.title,
        "kind": "apk-project",
        "shelf": "apk-projects",
        "path": str(project.dir),
        "hash": project.sha256,
        "size": int(prov.get("size") or 0),
        "provenance": {
            "type": prov.get("type", ""),
            "ref": prov.get("ref", ""),
            "fetched_at": prov.get("fetched_at"),
            "tool": "apkforge",
        },
        "tags": tags,
    }


def techforge_feed(dossier: dict[str, Any]) -> dict[str, Any]:
    """The TechForge feed shape: what this dissection teaches Devon
    about how this Android app was made (identity, build verdict,
    SDK census) — the forward interface, emitted per dossier."""
    ident = dossier.get("identity") or {}
    apkid = dossier.get("apkid") or {}
    return {
        "kind": "apk-dossier",
        "package": ident.get("package", ""),
        "version_name": ident.get("version_name", ""),
        "version_code": ident.get("version_code"),
        "min_sdk": ident.get("min_sdk"),
        "target_sdk": ident.get("target_sdk"),
        "build_verdict": apkid.get("verdicts") or {},
        "sdk_census": [
            {"name": s.get("name", ""), "kind": s.get("kind", "")}
            for s in (dossier.get("sdk_signatures") or [])
        ],
        "native_abis": sorted({
            n.get("abi", "") for n in (dossier.get("native") or [])
            if n.get("abi")
        }),
        "source_sha256": dossier.get("source_sha256", ""),
    }
