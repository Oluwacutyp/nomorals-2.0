"""ApkForge facade — one class, every A1 op, structured dicts out.

Trigger → logic → result:
  add      → intake.land_file / land_url → project + light identity
  dissect  → dissect.dissect_apk → dossier.json/.md on the project
  teach    → teach.teach_lesson over the stored (or fresh) dossier
  diff     → dissect both (if needed) → diff.diff_dossiers, journaled
             to BOTH projects
  status   → toolchain.probe_capabilities + project census
  list     → project cards

The agent tool, /apk chat command and `nm apk` are thin shells over
these methods (betdesk idiom).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import intake as intake_mod
from .build import BuildError, build_project, render_build_card
from .diff import diff_dossiers, render_diff_card, render_diff_md
from .edit import (EditError, edit_manifest, edit_strings,
                   ensure_decoded, list_edits, read_string,
                   resource_add, resource_remove, resource_replace,
                   undo_edit)
from .patch import PatchError, apply_patch
from .signing import SigningError, ensure_keystore, list_keystores
from .splits import (SplitsError, extract_split, generate_splits,
                     installset, list_splits, merge_bundle)
from .toolchain import ToolchainError, probe_capabilities, provision
from .dissect import (dissect_apk, light_identity, render_dossier_card,
                      render_dossier_md)
from .project import (Project, apkforge_dir, find_project, list_projects,
                      museum_card, techforge_feed)
from .teach import teach_lesson

__all__ = ["ApkForge", "render_status", "render_list", "render_project",
           "render_build_card", "render_toolchain"]


class ApkForge:
    """One ApkForge instance.  ``context`` is the runtime context
    when bound (brain for teach); CLI/tests run context-free."""

    def __init__(self, context: Any = None, base_dir: str = ""):
        self.context = context
        self.root = apkforge_dir(context, base_dir)

    # ── intake ───────────────────────────────────────────────────────
    def add(self, source: str, *, is_url: bool = False) -> dict[str, Any]:
        source = (source or "").strip()
        if not source:
            return {"error": "add needs a file path or URL"}
        try:
            if is_url or source.startswith(("http://", "https://")):
                proj, created = intake_mod.land_url(self.root, source)
            else:
                proj, created = intake_mod.land_file(self.root, source)
        except intake_mod.IntakeError as exc:
            return {"error": str(exc)}
        return {"project": self._card(proj), "created": created,
                "card_text": render_project(self._card(proj))}

    # ── census / status ──────────────────────────────────────────────
    def _card(self, proj: Project) -> dict[str, Any]:
        card = museum_card(proj, proj.read_dossier())
        card["package"] = proj.package
        card["version_name"] = proj.record.get("version_name", "")
        card["version_code"] = proj.record.get("version_code")
        card["kind_detail"] = proj.kind
        card["has_dossier"] = (proj.dir / "dossier.json").exists()
        return card

    def list_projects(self) -> dict[str, Any]:
        return {"projects": [self._card(p)
                             for p in list_projects(self.root)]}

    def status(self) -> dict[str, Any]:
        probe = probe_capabilities(root=self.root)
        projects = list_projects(self.root)
        return {"capabilities": probe["capabilities"],
                "engines": probe["python_engines"],
                "binaries": probe["binaries"],
                "host": probe["host"],
                "notes": probe["notes"],
                "projects": len(projects),
                "workspace": str(self.root)}

    def resolve(self, ref: str) -> Project | None:
        return find_project(self.root, ref)

    # ── dissect ──────────────────────────────────────────────────────
    def _dissect_target(self, proj: Project) -> tuple[Path, str]:
        """The file a dissection actually parses: the APK itself, or
        for a bundle project its base.apk part (else the largest APK
        part) — the container is inventoried, the part is dissected,
        and the dossier says which."""
        if proj.kind == "apk":
            return proj.original_path, proj.sha256
        from .project import sha256_file

        parts = ((proj.record.get("inventory") or {}).get("parts") or [])
        chosen = None
        for part in parts:
            if part.get("name", "").endswith("base.apk"):
                chosen = part
                break
        if chosen is None and parts:
            chosen = max(parts, key=lambda r: int(r.get("size") or 0))
        if chosen is not None:
            target = proj.dir / "bundle" / chosen["name"]
            if target.exists():
                return target, chosen.get("sha256") or sha256_file(target)
        return proj.original_path, proj.sha256

    def dissect(self, ref: str, *, include_apkid: bool = True) -> dict[str, Any]:
        proj = self.resolve(ref)
        if proj is None:
            return {"error": f"no project matches {ref!r} — "
                             "apk list shows what has landed"}
        target, target_sha = self._dissect_target(proj)
        try:
            dossier = dissect_apk(target,
                                  include_apkid=include_apkid,
                                  source_sha256=target_sha)
        except RuntimeError as exc:
            return {"error": str(exc)}
        if proj.kind != "apk":
            dossier["bundle_part"] = {
                "container": proj.original_path.name,
                "part": target.name,
                "note": "dossier describes this part of the bundle; "
                        "the container itself is inventoried on the "
                        "project card"}
        md = render_dossier_md(dossier)
        proj.write_dossier(dossier, md)
        ident = dossier["identity"]
        if ident.get("package"):
            proj.record["package"] = ident["package"]
            proj.record["title"] = (ident.get("app_name")
                                    or ident["package"])
            proj.record["version_name"] = ident.get("version_name", "")
            proj.record["version_code"] = ident.get("version_code")
            proj.save()
        proj.journal("dissect",
                     f"{ident.get('package')} "
                     f"v{ident.get('version_name')} — "
                     f"{dossier['dex']['totals']['classes']} classes, "
                     f"{len(dossier['permissions'])} permissions")
        return {"project": self._card(proj), "dossier": dossier,
                "card_text": render_dossier_card(dossier),
                "md": md,
                "dossier_path": str(proj.dir / "dossier.json"),
                "dossier_md_path": str(proj.dir / "dossier.md"),
                "techforge_feed": techforge_feed(dossier)}

    def _ensure_dossier(self, proj: Project) -> dict[str, Any] | None:
        dossier = proj.read_dossier()
        _target, target_sha = self._dissect_target(proj)
        if dossier is not None and \
                dossier.get("source_sha256") == target_sha:
            return dossier
        res = self.dissect(proj.id)
        if res.get("error"):
            return None
        return res["dossier"]

    # ── teach ────────────────────────────────────────────────────────
    def teach(self, ref: str, topic: str = "") -> dict[str, Any]:
        proj = self.resolve(ref)
        if proj is None:
            return {"error": f"no project matches {ref!r}"}
        dossier = self._ensure_dossier(proj)
        if dossier is None:
            return {"error": "could not dissect this project — "
                             "run dissect for the exact failure"}
        lesson = teach_lesson(self.context, dossier, topic)
        proj.journal("teach", f"topic={lesson['topic']} "
                              f"source={lesson['source']}")
        return {"project": self._card(proj), **lesson}

    # ── diff ─────────────────────────────────────────────────────────
    def diff(self, ref_a: str, ref_b: str) -> dict[str, Any]:
        pa, pb = self.resolve(ref_a), self.resolve(ref_b)
        if pa is None:
            return {"error": f"no project matches {ref_a!r}"}
        if pb is None:
            return {"error": f"no project matches {ref_b!r}"}
        da, db = self._ensure_dossier(pa), self._ensure_dossier(pb)
        if da is None or db is None:
            return {"error": "could not dissect one of the projects — "
                             "run dissect for the exact failure"}
        delta = diff_dossiers(da, db)
        md = render_diff_md(delta)
        out_path = pa.dir / "diffs" / f"{pb.id}.md"
        for proj, other in ((pa, pb), (pb, pa)):
            report = proj.dir / "diffs" / f"{other.id}.md"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(md, encoding="utf-8")
            proj.journal("diff", f"vs {other.id}: "
                                 f"{render_diff_card(delta)}")
        return {"diff": delta, "md": md,
                "card_text": render_diff_card(delta),
                "report_path": str(out_path),
                "a": self._card(pa), "b": self._card(pb)}


    # ── toolchain (A2) ───────────────────────────────────────────
    def toolchain_status(self) -> dict[str, Any]:
        return probe_capabilities(root=self.root)

    def toolchain_provision(self, *, force: bool = False) -> dict[str, Any]:
        try:
            report = provision(root=self.root, force=force)
        except ToolchainError as exc:
            return {"error": str(exc)}
        return {"provisioned_now": report["installed"],
                "already_present": report["present"],
                "smoke_apktool_build": report["smoke"],
                "capabilities":
                    probe_capabilities(root=self.root)["capabilities"]}

    # ── decode / edit / patch / undo (A2) ────────────────────────
    def _need_project(self, ref: str) -> Project | dict[str, Any]:
        proj = self.resolve(ref)
        if proj is None:
            return {"error": f"no project matches {ref!r} — "
                             "apk list shows what has landed"}
        return proj

    def decode(self, ref: str, *, force: bool = False) -> dict[str, Any]:
        proj = self._need_project(ref)
        if isinstance(proj, dict):
            return proj
        try:
            ddir = ensure_decoded(self.root, proj, force=force)
        except EditError as exc:
            return {"error": str(exc)}
        return {"project": self._card(proj), "decoded": str(ddir),
                "manifest": str(ddir / "AndroidManifest.xml")}

    def edit(self, ref: str, op: str, **kw: Any) -> dict[str, Any]:
        proj = self._need_project(ref)
        if isinstance(proj, dict):
            return proj
        try:
            ensure_decoded(self.root, proj)
            op = (op or "").strip().lower()
            if op == "manifest":
                changes = {k: v for k, v in kw.items()
                           if v not in ("", None)}
                res = edit_manifest(proj, **changes)
            elif op == "strings":
                res = edit_strings(proj, str(kw.get("name") or ""),
                                   str(kw.get("value") or ""),
                                   locale=str(kw.get("locale") or ""))
            elif op == "replace":
                res = resource_replace(proj, str(kw.get("target")
                                                 or ""),
                                       str(kw.get("source") or ""))
            elif op == "add":
                res = resource_add(proj, str(kw.get("target") or ""),
                                   str(kw.get("source") or ""))
            elif op == "remove":
                res = resource_remove(proj, str(kw.get("target") or ""))
            else:
                return {"error": f"unknown edit op {op!r} — manifest|"
                                 "strings|replace|add|remove"}
        except (EditError, ToolchainError) as exc:
            return {"error": str(exc)}
        return {"project": self._card(proj), "edit": res}

    def get_string(self, ref: str, name: str, *,
                   locale: str = "") -> dict[str, Any]:
        proj = self._need_project(ref)
        if isinstance(proj, dict):
            return proj
        try:
            ensure_decoded(self.root, proj)
        except EditError as exc:
            return {"error": str(exc)}
        return {"name": name, "locale": locale,
                "value": read_string(proj, name, locale=locale)}

    def patch(self, ref: str, patch_doc: Any) -> dict[str, Any]:
        proj = self._need_project(ref)
        if isinstance(proj, dict):
            return proj
        try:
            ensure_decoded(self.root, proj)
            res = apply_patch(proj, patch_doc)
        except (PatchError, EditError, ToolchainError) as exc:
            return {"error": str(exc)}
        return {"project": self._card(proj), "patch_result": res}

    def edits(self, ref: str) -> dict[str, Any]:
        proj = self._need_project(ref)
        if isinstance(proj, dict):
            return proj
        return {"project": self._card(proj),
                "edits": list_edits(proj)}

    def undo(self, ref: str, edit_id: int | None = None) -> dict[str, Any]:
        proj = self._need_project(ref)
        if isinstance(proj, dict):
            return proj
        try:
            res = undo_edit(proj, edit_id)
        except EditError as exc:
            return {"error": str(exc)}
        return {"project": self._card(proj), "undo": res}

    # ── build / sign (A2) ────────────────────────────────────────
    def build(self, ref: str, *, alias: str = "") -> dict[str, Any]:
        proj = self._need_project(ref)
        if isinstance(proj, dict):
            return proj
        try:
            card = build_project(self.context, self.root, proj,
                                 alias=alias)
        except (BuildError, SigningError, ToolchainError) as exc:
            return {"error": str(exc)}
        return {"project": self._card(proj), "build": card,
                "card_text": render_build_card(card)}

    def keys(self) -> dict[str, Any]:
        try:
            keys = list_keystores(self.context)
        except SigningError as exc:
            return {"error": str(exc)}
        return {"keystores": keys}

    def ensure_key(self, alias: str = "") -> dict[str, Any]:
        try:
            rec = ensure_keystore(self.context, self.root, alias)
        except SigningError as exc:
            return {"error": str(exc)}
        return {"keystore": rec}

    # ── splits (A2) ──────────────────────────────────────────────
    def splits(self, ref: str, op: str = "list", **kw: Any) -> dict[str, Any]:
        proj = self._need_project(ref)
        if isinstance(proj, dict):
            return proj
        op = (op or "list").strip().lower()
        try:
            if op == "list":
                return {"project": self._card(proj),
                        **list_splits(proj)}
            if op == "extract":
                res = extract_split(proj, str(kw.get("part") or ""))
                return {"project": self._card(proj), "extract": res}
            if op == "merge":
                res = merge_bundle(self.context, self.root, proj,
                                   alias=str(kw.get("alias") or ""))
                return {"project": self._card(proj), "merge": res}
            if op == "installset":
                spec = kw.get("spec") or {}
                if isinstance(spec, str):
                    import json as _json

                    spec = _json.loads(spec)
                res = installset(proj, spec, root=self.root)
                return {"project": self._card(proj), "installset": res}
            if op == "generate":
                res = generate_splits(self.context, self.root, proj,
                                      alias=str(kw.get("alias")
                                                or ""))
                return {"project": self._card(proj), "generate": res}
        except (SplitsError, SigningError, ToolchainError,
                ValueError) as exc:
            return {"error": str(exc)}
        return {"error": f"unknown splits op {op!r} — list|extract|"
                         "merge|installset|generate"}

    # ── jadx reading surface (A2) ────────────────────────────────
    def jadx(self, ref: str, *, force: bool = False) -> dict[str, Any]:
        """Decompile the project's APK to readable Java under
        jadx/ — a reading surface only (jadx output cannot be
        recompiled; edits happen in the apktool project)."""
        from .toolchain import run_tool as _run

        proj = self._need_project(ref)
        if isinstance(proj, dict):
            return proj
        target, _sha = self._dissect_target(proj)
        out = proj.dir / "jadx"
        marker = out / ".source-sha"
        if out.exists() and marker.exists() and not force and                 marker.read_text(encoding="utf-8").strip() == _sha:
            return {"project": self._card(proj), "jadx": str(out),
                    "cached": True}
        import shutil as _shutil

        _shutil.rmtree(out, ignore_errors=True)
        try:
            res = _run("", "jadx", ["-d", str(out), "--no-res",
                                    str(target)],
                       timeout=1800, root=self.root)
        except ToolchainError as exc:
            return {"error": f"jadx step failed: {exc}"}
        java_files = list(out.rglob("*.java")) if out.exists() else []
        if not java_files:
            tail = (res["stdout"] + res["stderr"]).strip().splitlines()
            return {"error": "jadx produced no Java sources: "
                             + (tail[-1][:300] if tail
                                else f"rc={res['rc']}")}
        marker.write_text(_sha, encoding="utf-8")
        proj.journal("jadx", f"decompiled {target.name} → "
                             f"{len(java_files)} java files "
                             "(reading surface only)")
        return {"project": self._card(proj), "jadx": str(out),
                "java_files": len(java_files), "cached": False}


# ── text renderers for the chat / CLI surfaces ───────────────────────

def render_toolchain(st: dict[str, Any]) -> str:
    caps = st.get("capabilities") or {}
    tc = st.get("toolchain") or {}
    lines = ["🪓 **ApkForge toolchain**"]
    arts = tc.get("artifacts") or {}
    if arts:
        for name, art in sorted(arts.items()):
            lines.append(f"  {name}: v{art.get('version')} "
                         f"({art.get('source')}) "
                         f"sha {str(art.get('sha256'))[:12]}…")
    else:
        lines.append("  nothing provisioned yet — "
                     "apkforge toolchain provision")
    smoke = tc.get("smoke_apktool_build") or {}
    if smoke:
        lines.append(f"  apktool smoke build: "
                     f"{'OK' if smoke.get('ok') else 'FAILED — ' + str(smoke.get('detail'))}")
    lines.append(f"  routing: {tc.get('routing', '')}")
    for name, state in caps.items():
        lines.append(f"  {name}: {state}")
    return "\n".join(lines)


def render_status(st: dict[str, Any]) -> str:
    caps = st.get("capabilities") or {}
    lines = ["🪓 **ApkForge status**",
             f"workspace: {st.get('workspace')}",
             f"projects landed: {st.get('projects', 0)}",
             "capabilities:"]
    for name, state in caps.items():
        lines.append(f"  {name}: {state}")
    eng = st.get("engines") or {}
    lines.append(f"engines: androguard {eng.get('androguard') or 'MISSING'}"
                 f" · apkid {eng.get('apkid') or 'missing'}")
    return "\n".join(lines)


def render_project(card: dict[str, Any]) -> str:
    return (f"🪓 {card.get('title') or card.get('id')} "
            f"[{card.get('id')}] — {card.get('kind_detail')}\n"
            f"package: {card.get('package') or '—'} · "
            f"v{card.get('version_name') or '?'} "
            f"({card.get('version_code') if card.get('version_code') is not None else '?'})\n"
            f"sha256: {str(card.get('hash') or '')[:16]}… · "
            f"{card.get('size', 0):,} bytes · "
            f"dossier: {'yes' if card.get('has_dossier') else 'not yet'}")


def render_list(res: dict[str, Any]) -> str:
    projects = res.get("projects") or []
    if not projects:
        return ("no APK projects yet — land one with "
                "add <path|url>")
    lines = [f"🪓 {len(projects)} APK project(s):"]
    for card in projects:
        lines.append("  " + render_project(card).replace("\n", "\n  "))
    return "\n".join(lines)
