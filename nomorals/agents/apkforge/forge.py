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
from .diff import diff_dossiers, render_diff_card, render_diff_md
from .dissect import (dissect_apk, light_identity, render_dossier_card,
                      render_dossier_md)
from .project import (Project, apkforge_dir, find_project, list_projects,
                      museum_card, techforge_feed)
from .teach import teach_lesson
from .toolchain import probe_capabilities

__all__ = ["ApkForge", "render_status", "render_list", "render_project"]


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
        probe = probe_capabilities()
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


# ── text renderers for the chat / CLI surfaces ───────────────────────

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
