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

import shutil
from pathlib import Path
from typing import Any

from . import intake as intake_mod
from . import signatures as sig_mod
from . import srcbuild as srcbuild_mod
from .build import BuildError, build_project, render_build_card
from .diff import diff_dossiers, render_diff_card, render_diff_md
from .edit import (EditError, edit_manifest, edit_strings,
                   ensure_decoded, list_edits, read_string,
                   read_text_file, resource_add, resource_remove,
                   resource_replace, undo_edit, write_text_file)
from .patch import PatchError, apply_patch
from .signing import SigningError, ensure_keystore, list_keystores
from .splits import (SplitsError, extract_split, generate_splits,
                     installset, list_splits, merge_bundle)
from .srcbuild import SrcBuildError, render_srcbuild_card
from .toolchain import (ToolchainError, probe_capabilities, provision,
                        provision_srcbuild)
from .dissect import (dissect_apk, light_identity, render_dossier_card,
                      render_dossier_md)
from .project import (Project, apkforge_dir, find_project, list_projects,
                      museum_card, sha256_file, techforge_feed)
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
            dossier = dissect_apk(
                target,
                include_apkid=include_apkid,
                source_sha256=target_sha,
                extra_signatures=sig_mod.active_signatures(self.root))
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
        # A3 signature discovery: this dissection is evidence for
        # the growing tracker table (candidate census + grown
        # sightings).  A census failure is reported, never hidden,
        # and never fails the dissection itself.
        try:
            observed = sig_mod.observe_dossier(self.root, proj.id,
                                               dossier)
        except Exception as exc:  # noqa: BLE001 — reported in-band
            observed = {"error": f"signature observation failed: "
                                 f"{exc}"}
        return {"project": self._card(proj), "dossier": dossier,
                "card_text": render_dossier_card(dossier),
                "md": md,
                "dossier_path": str(proj.dir / "dossier.json"),
                "dossier_md_path": str(proj.dir / "dossier.md"),
                "signature_observation": observed,
                "techforge_feed": techforge_feed(dossier)}

    def _ensure_dossier(self, proj: Project) -> dict[str, Any] | None:
        dossier, _err = self._ensure_dossier_report(proj)
        return dossier

    def _ensure_dossier_report(
            self, proj: Project) -> tuple[dict[str, Any] | None, str]:
        """_ensure_dossier with the failure kept: (dossier, "")
        or (None, the dissection's own error text) — the bisect
        judge records that text as a skip's reason."""
        dossier = proj.read_dossier()
        _target, target_sha = self._dissect_target(proj)
        if dossier is not None and \
                dossier.get("source_sha256") == target_sha:
            return dossier, ""
        res = self.dissect(proj.id)
        if res.get("error"):
            return None, str(res["error"])
        return res["dossier"], ""

    # ── teach ────────────────────────────────────────────────────────
    def teach(self, ref: str, topic: str = "", *,
              target_class: str = "", method: str = "",
              mapping: str = "") -> dict[str, Any]:
        proj = self.resolve(ref)
        if proj is None:
            return {"error": f"no project matches {ref!r}"}
        if target_class:
            return self._teach_code(proj, target_class,
                                    method=method, mapping=mapping)
        dossier = self._ensure_dossier(proj)
        if dossier is None:
            return {"error": "could not dissect this project — "
                             "run dissect for the exact failure"}
        lesson = teach_lesson(self.context, dossier, topic)
        proj.journal("teach", f"topic={lesson['topic']} "
                              f"source={lesson['source']}")
        return {"project": self._card(proj), **lesson}

    def _teach_code(self, proj: Project, target_class: str, *,
                    method: str = "",
                    mapping: str = "") -> dict[str, Any]:
        """Teach v2 (A5): one class's method walked in smali AND
        jadx Java — the smali from the decoded tree, the Java from
        the jadx surface (mapping-restored when a mapping is given
        or the project carries one)."""
        from . import teach as teach_mod
        from .edit import EditError as _EditError
        from .edit import ensure_decoded as _ensure_decoded

        dotted = teach_mod.normalize_class_name(target_class)
        try:
            decoded = _ensure_decoded(self.root, proj)
        except _EditError as exc:
            return {"error": str(exc)}
        smali_path = teach_mod.find_smali_path(decoded, dotted)
        if smali_path is None:
            return {"error": f"no smali for class {dotted} in the "
                             "decoded tree — the class is not in "
                             "this APK, or decode produced no smali "
                             "for its dex"}
        smali_text = smali_path.read_text(encoding="utf-8",
                                          errors="replace")
        # mapping: explicit param (project-relative, jadx idiom),
        # else a mapping.txt the project itself carries.
        mpath: Path | None = None
        if mapping:
            cand = Path(mapping).expanduser()
            if not cand.is_absolute():
                cand = proj.dir / cand
            if not cand.exists():
                return {"error": f"mapping file does not exist: "
                                 f"{cand}"}
            mpath = cand
        elif (proj.dir / "mapping.txt").exists():
            mpath = proj.dir / "mapping.txt"
        original = dotted
        if mpath is not None:
            original = teach_mod.parse_mapping(
                mpath.read_text(encoding="utf-8",
                                errors="replace")).get(dotted,
                                                        dotted)
        jres = self.jadx(proj.id,
                         mapping=str(mpath) if mpath else "")
        if jres.get("error"):
            return {"error": f"jadx side of the lesson failed: "
                             f"{jres['error']}"}
        jadx_dir = Path(jres["jadx"])
        simple = (original.rsplit(".", 1)[-1]) + ".java"
        hits = sorted(jadx_dir.rglob(simple))
        pkg_tail = Path(*original.split(".")[:-1])
        java_path: Path | None = None
        for hit in hits:
            if hit.parent == jadx_dir / "sources" / pkg_tail or \
                    hit.parent.name == pkg_tail.name:
                java_path = hit
                break
        if java_path is None and hits:
            java_path = hits[0]
        java_text = java_path.read_text(encoding="utf-8",
                                        errors="replace") \
            if java_path is not None else ""
        lesson = teach_mod.teach_code_lesson(
            self.context, class_name=dotted, original_name=original,
            method=method, smali_text=smali_text,
            java_text=java_text, mapping_used=mpath is not None)
        if lesson.get("error"):
            return lesson
        proj.journal("teach",
                     f"v2 class={dotted} "
                     f"method={lesson['method']} "
                     f"source={lesson['source']}"
                     + (f" mapping={mpath.name}" if mpath else ""))
        return {"project": self._card(proj), **lesson,
                "smali_path": str(smali_path),
                "java_path": str(java_path) if java_path else ""}

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

    def toolchain_provision_srcbuild(
            self, *, force: bool = False) -> dict[str, Any]:
        """A4: provision the build-from-source set (platform,
        cmdline-tools, R8, Gradle) + SDK layout + licenses."""
        try:
            report = provision_srcbuild(root=self.root, force=force)
        except ToolchainError as exc:
            return {"error": str(exc)}
        return {"provisioned_now": report["installed"],
                "already_present": report["present"],
                "smoke_apktool_build": report["smoke"],
                "sdk_root": report.get("sdk_root", ""),
                "sdk_licenses": report.get("sdk_licenses", {}),
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
            elif op == "write":
                content = kw.get("value")
                if content in ("", None) and kw.get("source"):
                    from pathlib import Path as _P

                    spath = _P(str(kw["source"])).expanduser()
                    if not spath.exists():
                        return {"error": f"content source file "
                                         f"does not exist: {spath}"}
                    content = spath.read_text(encoding="utf-8")
                res = write_text_file(proj, str(kw.get("target")
                                                or ""),
                                      str(content or ""))
            else:
                return {"error": f"unknown edit op {op!r} — manifest|"
                                 "strings|replace|add|remove|write"}
        except (EditError, ToolchainError) as exc:
            return {"error": str(exc)}
        return {"project": self._card(proj), "edit": res}

    def read_text(self, ref: str, target: str) -> dict[str, Any]:
        """Read one text file (smali above all) from the decoded
        tree — the editor's read surface (A4)."""
        proj = self._need_project(ref)
        if isinstance(proj, dict):
            return proj
        try:
            ensure_decoded(self.root, proj)
            res = read_text_file(proj, target)
        except (EditError, ToolchainError) as exc:
            return {"error": str(exc)}
        return {"project": self._card(proj), "file": res}

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

    # ── jadx reading surface (A2; mapping restoration A4) ────────
    def jadx(self, ref: str, *, force: bool = False,
             mapping: str = "") -> dict[str, Any]:
        """Decompile the project's APK to readable Java under
        jadx/ — a reading surface only (jadx output cannot be
        recompiled; edits happen in the apktool project).

        With ``mapping`` (a ProGuard/R8 mapping.txt, as produced by
        an R8 build), the mapping is converted to the Tiny v2 form
        jadx's --mappings-path surface consumes and the decompile
        comes back with the original class names restored."""
        from .toolchain import run_tool as _run

        proj = self._need_project(ref)
        if isinstance(proj, dict):
            return proj
        target, _sha = self._dissect_target(proj)
        out = proj.dir / "jadx"
        marker = out / ".source-sha"
        map_sha = ""
        tiny_path: Path | None = None
        if mapping:
            mpath = Path(mapping).expanduser()
            if not mpath.is_absolute():
                mpath = proj.dir / mpath
            if not mpath.exists():
                return {"error": f"mapping file does not exist: "
                                 f"{mpath}"}
            from .project import sha256_file as _sha256

            map_sha = _sha256(mpath)
            tiny_path = proj.dir / "jadx-mapping.tiny"
            tiny_path.write_text(
                srcbuild_mod.proguard_to_tiny(
                    mpath.read_text(encoding="utf-8",
                                    errors="replace")),
                encoding="utf-8")
        cache_key = _sha + (f"|{map_sha}" if map_sha else "")
        if out.exists() and marker.exists() and not force and \
                marker.read_text(encoding="utf-8").strip() == cache_key:
            return {"project": self._card(proj), "jadx": str(out),
                    "cached": True,
                    "mapping": str(tiny_path) if tiny_path else ""}
        import shutil as _shutil

        _shutil.rmtree(out, ignore_errors=True)
        args = ["-d", str(out), "--no-res"]
        if tiny_path is not None:
            args += ["--mappings-path", str(tiny_path)]
        args.append(str(target))
        try:
            res = _run("", "jadx", args, timeout=1800, root=self.root)
        except ToolchainError as exc:
            return {"error": f"jadx step failed: {exc}"}
        java_files = list(out.rglob("*.java")) if out.exists() else []
        if not java_files:
            tail = (res["stdout"] + res["stderr"]).strip().splitlines()
            return {"error": "jadx produced no Java sources: "
                             + (tail[-1][:300] if tail
                                else f"rc={res['rc']}")}
        marker.write_text(cache_key, encoding="utf-8")
        proj.journal("jadx", f"decompiled {target.name} → "
                             f"{len(java_files)} java files "
                             "(reading surface only)"
                             + (" with R8 mapping name restoration"
                                if tiny_path else ""))
        return {"project": self._card(proj), "jadx": str(out),
                "java_files": len(java_files), "cached": False,
                "mapping": str(tiny_path) if tiny_path else ""}

    # ── build from source (A4) ───────────────────────────────────
    @staticmethod
    def _slug(text: str) -> str:
        import re as _re

        slug = _re.sub(r"[^a-z0-9]+", "-",
                       (text or "source").lower()).strip("-")
        return slug or "source"

    def _sources_dir(self) -> Path:
        d = self.root / "sources"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _land_source(self, source: str) -> tuple[Path, dict[str, Any]]:
        """Land a source project into the class workspace:
        a local dir (copied, content-hashed), a local .zip
        (extracted, zip-slip guarded), a git URL (cloned), or a
        .zip URL (downloaded via the shared HTTP organ, then
        extracted).  Returns (project dir, provenance)."""
        import hashlib
        import re as _re
        import subprocess as _sp

        raw = (source or "").strip()
        if not raw:
            raise SrcBuildError("buildsrc needs a source: a local "
                                "project dir, a .zip, or a git URL")
        sdir = self._sources_dir()
        # a previously landed source, referenced by name
        named = sdir / raw
        if named.is_dir() and (named / "src").is_dir():
            return named / "src", {"type": "landed", "ref": raw}
        if raw.startswith(("http://", "https://")):
            if raw.split("?")[0].endswith(".zip"):
                from ...core.http import download

                dl = Path(download(raw, sdir / ".inbox"))
                digest = sha256_file(dl)
                slug = self._slug(raw.rsplit("/", 1)[-1])
                dest = sdir / f"{slug}-{digest[:12]}"
                self._extract_source_zip(dl, dest)
                return dest / "src", {"type": "url-zip", "ref": raw,
                                      "sha256": digest}
            slug = self._slug(raw.rsplit("/", 1)[-1].removesuffix(".git"))
            digest = hashlib.sha256(raw.encode()).hexdigest()
            dest = sdir / f"{slug}-{digest[:12]}"
            if not (dest / "src").exists():
                git = shutil.which("git")
                if not git:
                    raise SrcBuildError(
                        "git clone step: no git binary on this host "
                        f"to clone {raw}")
                (dest / "src").mkdir(parents=True, exist_ok=True)
                proc = _sp.run([git, "clone", "--depth", "1", raw,
                                str(dest / "src")],
                               capture_output=True, text=True,
                               timeout=900)
                if proc.returncode != 0:
                    raise SrcBuildError(
                        f"git clone step failed for {raw}: "
                        f"{(proc.stderr or proc.stdout)[-400:]}")
            return dest / "src", {"type": "git", "ref": raw}
        path = Path(raw).expanduser()
        if path.is_file() and path.suffix.lower() == ".zip":
            digest = sha256_file(path)
            dest = sdir / f"{self._slug(path.stem)}-{digest[:12]}"
            self._extract_source_zip(path, dest)
            return dest / "src", {"type": "zip", "ref": str(path),
                                  "sha256": digest}
        if path.is_dir():
            if sdir in path.parents or path.parent == sdir:
                inner = path / "src" if (path / "src").is_dir() \
                    else path
                return inner, {"type": "landed", "ref": str(path)}
            digest = srcbuild_mod.tree_hash(path)
            dest = sdir / f"{self._slug(path.name)}-{digest[:12]}"
            if not (dest / "src").exists():
                ignore = shutil.ignore_patterns(
                    ".git", ".gradle", "build", "__pycache__")
                shutil.copytree(path, dest / "src", ignore=ignore)
            return dest / "src", {"type": "dir", "ref": str(path),
                                  "sha256": digest}
        raise SrcBuildError(
            f"source {raw!r} is not a directory, a .zip, or a URL — "
            "buildsrc lands source projects, not single files")

    def _extract_source_zip(self, zpath: Path, dest: Path) -> None:
        if (dest / "src").exists():
            return
        import zipfile as _zf

        from .toolchain import _safe_extract_zip

        stage = dest / ".extract"
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True, exist_ok=True)
        try:
            with _zf.ZipFile(zpath) as zf:
                _safe_extract_zip(zf, stage)
        except ToolchainError as exc:
            raise SrcBuildError(f"source zip step: {exc}") from exc
        children = [c for c in stage.iterdir()]
        inner = children[0] if len(children) == 1 \
            and children[0].is_dir() else stage
        shutil.move(str(inner), str(dest / "src"))
        shutil.rmtree(stage, ignore_errors=True)

    def buildsrc(self, source: str, *, route: str = "",
                 variant: str = "debug", bundle: bool = False,
                 use_r8: bool = False,
                 alias: str = "") -> dict[str, Any]:
        """Build an app from a source project (dir | zip | git URL)
        → signed, verified APK (manual pipeline or gradle route,
        auto-detected) with the build card naming the route."""
        try:
            src_dir, provenance = self._land_source(source)
            card = srcbuild_mod.build_source(
                self.context, self.root, src_dir, alias=alias,
                route=route, variant=variant, bundle=bundle,
                use_r8=use_r8)
        except (SrcBuildError, SigningError, ToolchainError) as exc:
            return {"error": str(exc)}
        card["provenance"] = provenance
        return {"source": {"dir": str(src_dir),
                           "provenance": provenance},
                "build": card,
                "card_text": render_srcbuild_card(card)}

    def new_project(self, name: str, *,
                    package: str = "") -> dict[str, Any]:
        """Scaffold a real, buildable app project into the class
        sources area (A4): manifest + behavior-bearing
        MainActivity + string resources, ready for buildsrc."""
        try:
            info = srcbuild_mod.scaffold_project(
                self._sources_dir(), name, package=package)
        except SrcBuildError as exc:
            return {"error": str(exc)}
        return {"project": info,
                "card_text": (
                    f"🪓 **new source project** — {info['name']} "
                    f"({info['package']})\n"
                    f"dir: {info['dir']}\n"
                    f"files: {', '.join(info['files'])}\n"
                    f"build it with buildsrc {info['name']}")}

    # ── bisect (A3) ──────────────────────────────────────────────
    def bisect(self, series: Any, check: str) -> dict[str, Any]:
        """Version-series bisect: order the landed versions of one
        app by versionCode and halve on a dossier predicate (the
        git-bisect good/bad/skip contract) to the exact version
        where the property flipped, with the pair's diff as
        evidence.  Journaled + resumable per series+predicate."""
        from . import bisect as bisect_mod

        refs = series if isinstance(series, list) else \
            [r.strip() for r in str(series or "").split(",")
             if r.strip()]
        if len(refs) < 2:
            return {"error": "bisect needs at least two versions "
                             "of the same app"}
        try:
            pred = bisect_mod.parse_predicate(check)
        except bisect_mod.BisectError as exc:
            return {"error": str(exc)}
        projects: list[Project] = []
        for ref in refs:
            proj = self.resolve(ref)
            if proj is None:
                return {"error": f"no project matches {ref!r} — "
                                 "apk list shows what has landed"}
            projects.append(proj)
        dossiers: dict[str, dict[str, Any] | None] = {}
        dossier_errors: dict[str, str] = {}
        meta: dict[str, dict[str, Any]] = {}
        packages: set[str] = set()
        for proj in projects:
            dossier, derr = self._ensure_dossier_report(proj)
            dossiers[proj.id] = dossier
            if derr:
                dossier_errors[proj.id] = derr
            ident = (dossier or {}).get("identity") or {}
            code = ident.get("version_code")
            if code is None:
                code = proj.record.get("version_code")
            if code is None:
                return {"error": f"project {proj.id} has no "
                                 "readable versionCode — a version "
                                 "series cannot be ordered without "
                                 "it; dissect it first or land a "
                                 "parseable build"}
            if ident.get("package"):
                packages.add(str(ident["package"]))
            meta[proj.id] = {
                "version_name": ident.get("version_name")
                                or proj.record.get("version_name", ""),
                "version_code": int(code),
                "created_at": float(proj.record.get("created_at")
                                    or 0)}
        if len(packages) > 1:
            return {"error": "the series spans more than one app: "
                             + ", ".join(sorted(packages))
                             + " — a version series is one package"}
        projects.sort(key=lambda p: (meta[p.id]["version_code"],
                                     meta[p.id]["created_at"]))
        ids = [p.id for p in projects]
        state = bisect_mod.load_session(self.root, ids, check)
        resumed = bool(state.get("resumed"))

        def skip_detail(pid: str, dossier: Any) -> str:
            return (dossier_errors.get(pid)
                    or bisect_mod.dossier_damage(dossier)
                    or "this version's dossier cannot judge the "
                       "predicate (field missing)")

        if pred["kind"] == "exported":
            # three-valued component states: the session records
            # states, and the state driver names every boundary
            # crossing (A5).  Known verdicts in any other shape
            # (a pre-A5 session) are not states — drop them.
            known = {k: v for k, v in state["verdicts"].items()
                     if v in bisect_mod.COMPONENT_STATES
                     or v == bisect_mod.SKIP}

            def state_judge(pid: str) -> str:
                dossier = dossiers.get(pid)
                v = bisect_mod.component_state(
                    dossier, pred["value"])
                if v is None:
                    v = bisect_mod.SKIP
                bisect_mod.record_verdict(
                    self.root, state, pid, v,
                    detail=skip_detail(pid, dossier)
                    if v == bisect_mod.SKIP else "")
                return v

            try:
                result = bisect_mod.halve_states(ids, state_judge,
                                                 known=known)
            except bisect_mod.BisectError as exc:
                return {"error": str(exc)}
        else:
            def judge(pid: str) -> str:
                dossier = dossiers.get(pid)
                v = bisect_mod.evaluate_predicate(dossier, pred)
                bisect_mod.record_verdict(
                    self.root, state, pid, v,
                    detail=skip_detail(pid, dossier)
                    if v == bisect_mod.SKIP else "")
                return v

            try:
                result = bisect_mod.halve(ids, judge,
                                          known=state["verdicts"])
            except bisect_mod.BisectError as exc:
                return {"error": str(exc)}
        details = dict(state.get("skip_details") or {})
        result["skip_reasons"] = {
            pid: details[pid] for pid in result.get("skipped") or []
            if details.get(pid)}
        bisect_mod.finish_session(self.root, state, result)
        out: dict[str, Any] = {
            **result, "predicate": pred, "versions": meta,
            "resumed": resumed, "session": state["key"],
            "series_projects": [self._card(p) for p in projects]}
        summary = bisect_mod.render_bisect_card(out)
        for proj in projects:
            proj.journal("bisect",
                         f"check {check!r}: {result['status']}"
                         + (f" — transition {result['transition']}"
                            if result.get("transition") else ""))
        if result.get("transition"):
            older = self.resolve(result["transition"]["older"])
            newer = self.resolve(result["transition"]["newer"])
            da, db = dossiers[older.id], dossiers[newer.id]
            if da is not None and db is not None:
                delta = diff_dossiers(da, db)
                out["diff"] = delta
                out["diff_card"] = render_diff_card(delta)
                report = (self.root / "bisects" / state["key"]
                          / "transition-diff.md")
                report.write_text(render_diff_md(delta),
                                  encoding="utf-8")
                out["diff_path"] = str(report)
                summary = bisect_mod.render_bisect_card(out)
        out["card_text"] = summary
        return out

    # ── signature growth (A3) ────────────────────────────────────
    def signatures(self, op: str = "list", *, prefix: str = "",
                   label: str = "", kind: str = "",
                   reason: str = "") -> dict[str, Any]:
        op = (op or "list").strip().lower()
        try:
            if op in ("list", "status"):
                st = sig_mod.signatures_status(self.root)
                return {"signatures": st,
                        "card_text": sig_mod.render_signatures(st)}
            if op == "proposals":
                return {"proposals": sig_mod.proposals(self.root),
                        "threshold": sig_mod.recurrence_threshold()}
            if op == "promote":
                entry = sig_mod.promote(self.root, prefix,
                                        label=label, kind=kind)
                return {"promoted": entry,
                        "signatures":
                            sig_mod.signatures_status(self.root)}
            if op == "demote":
                gone = sig_mod.demote(self.root, prefix,
                                      reason=reason)
                return {"demoted": gone,
                        "signatures":
                            sig_mod.signatures_status(self.root)}
        except sig_mod.SignatureError as exc:
            return {"error": str(exc)}
        return {"error": f"unknown signatures op {op!r} — "
                         "list|proposals|promote|demote"}

    # ── native hex patching (A3) ────────────────────────────────
    def hexpatch(self, ref: str, patch_doc: Any) -> dict[str, Any]:
        from .hexpatch import HexPatchError, apply_hexpatch

        proj = self._need_project(ref)
        if isinstance(proj, dict):
            return proj
        try:
            ensure_decoded(self.root, proj)
            res = apply_hexpatch(proj, patch_doc)
        except (HexPatchError, EditError, ToolchainError) as exc:
            return {"error": str(exc)}
        return {"project": self._card(proj), "hexpatch_result": res}

    # ── Museum artifacts (A3) ────────────────────────────────────
    def artifacts(self) -> dict[str, Any]:
        from .museum import list_artifacts, render_artifacts

        res = list_artifacts(self.root)
        return {**res, "card_text": render_artifacts(res)}

    def send_artifact(self, ref: str, platform: str,
                      chat_id: str) -> dict[str, Any]:
        """Deliver one artifact (project / dossier / build) through
        the repo's real file-send surface — the card's path is the
        file, sent as-is via the live gateway."""
        from .museum import find_artifact

        card = find_artifact(self.root, ref)
        if card is None:
            return {"error": f"no artifact matches {ref!r} — "
                             "artifacts lists what exists"}
        if not platform or not chat_id:
            return {"error": "send needs a platform and chat_id"}
        from ...tools.filesend import send_file

        try:
            sent = send_file(self.context, platform, chat_id,
                             str(card["path"]),
                             caption=f"🪓 {card['title']}")
        except Exception as exc:  # noqa: BLE001 — ToolError + kin
            return {"error": f"send failed: {exc}"}
        return {"artifact": card, "sent": sent}


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
