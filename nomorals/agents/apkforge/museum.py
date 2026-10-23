"""ApkForge Museum wiring — every class output as one listable,
sendable artifact (design §3.9).

The Museum registry itself is a banked program, not yet built; the
surface the repo uses *today* for finished files is the gateway
file-send (``tools/filesend.send_file``) fed by Museum-shaped cards
(``id/title/kind/shelf/path/hash/size/provenance/tags``).  A1/A2
already emitted that shape for projects and builds.  A3 completes
the wiring:

* dossiers get their own cards (shelf ``apk-dossiers``), next to
  project cards (``apk-projects``) and build cards (``apk-builds``);
* ``list_artifacts`` aggregates all three shelves across the whole
  class workspace — one census the ``artifacts`` surface renders
  and the send path resolves against;
* ``ApkForge.send`` resolves an artifact id from that census and
  delivers the real file at its card path through the same
  ``send_file`` every other organ uses — no parallel registry, no
  copied files: the card path IS the artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .project import Project, list_projects, museum_card

__all__ = ["dossier_card", "build_cards", "list_artifacts",
           "find_artifact", "render_artifacts"]


def dossier_card(project: Project,
                 dossier: dict[str, Any] | None = None
                 ) -> dict[str, Any] | None:
    """The Museum card for a project's latest dossier, or None when
    the project has never been dissected."""
    dossier = dossier if dossier is not None else project.read_dossier()
    if dossier is None:
        return None
    path = project.dir / "dossier.json"
    ident = dossier.get("identity") or {}
    return {
        "id": f"{project.id}-dossier",
        "title": f"{project.title} dossier "
                 f"(v{ident.get('version_name') or '?'})",
        "kind": "apk-dossier",
        "shelf": "apk-dossiers",
        "path": str(path),
        "hash": str(dossier.get("source_sha256") or ""),
        "size": path.stat().st_size if path.exists() else 0,
        "provenance": {"type": "dissect", "ref": project.id,
                       "tool": "apkforge"},
        "tags": ["apk", "dossier", project.package],
    }


def build_cards(project: Project) -> list[dict[str, Any]]:
    """Museum cards for every build the project recorded (the build
    card JSON already embeds its card; the file must still exist)."""
    out: list[dict[str, Any]] = []
    builds = project.dir / "builds"
    if not builds.exists():
        return out
    for card_path in sorted(builds.glob("*.json")):
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mc = card.get("museum_card")
        if isinstance(mc, dict) and Path(str(mc.get("path") or "")) \
                .exists():
            out.append(mc)
    return out


def source_build_cards(root: Path) -> list[dict[str, Any]]:
    """Museum cards for A4 build-from-source outputs: each landed
    source keeps its srcbuild card under build/outputs/, and the
    card already embeds its museum shape."""
    out: list[dict[str, Any]] = []
    sources = root / "sources"
    if not sources.exists():
        return out
    for card_path in sorted(sources.glob(
            "*/src/build/outputs/srcbuild-card.json")):
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mc = card.get("museum_card")
        if isinstance(mc, dict) and Path(str(mc.get("path") or "")) \
                .exists():
            out.append(mc)
    return out


def list_artifacts(root: Path) -> dict[str, Any]:
    """The whole class census: projects, dossiers, builds."""
    artifacts: list[dict[str, Any]] = []
    for proj in list_projects(root):
        artifacts.append(museum_card(proj, proj.read_dossier()))
        dc = dossier_card(proj)
        if dc is not None:
            artifacts.append(dc)
        artifacts.extend(build_cards(proj))
    artifacts.extend(source_build_cards(root))
    by_shelf: dict[str, int] = {}
    for card in artifacts:
        by_shelf[card["shelf"]] = by_shelf.get(card["shelf"], 0) + 1
    return {"artifacts": artifacts, "by_shelf": by_shelf,
            "count": len(artifacts)}


def find_artifact(root: Path, ref: str) -> dict[str, Any] | None:
    """Resolve an artifact by exact id or unique id prefix."""
    ref = (ref or "").strip()
    if not ref:
        return None
    arts = list_artifacts(root)["artifacts"]
    for card in arts:
        if card["id"] == ref:
            return card
    prefixed = [c for c in arts if c["id"].startswith(ref)]
    return prefixed[0] if len(prefixed) == 1 else None


def render_artifacts(res: dict[str, Any]) -> str:
    arts = res.get("artifacts") or []
    if not arts:
        return ("no ApkForge artifacts yet — land an APK, dissect "
                "it, or build one and its cards land here")
    by_shelf = ", ".join(
        f"{k}: {v}" for k, v in sorted((res.get("by_shelf") or {})
                                        .items()))
    lines = [f"🪓 **ApkForge artifacts** — {len(arts)} "
             f"({by_shelf})"]
    for card in arts:
        lines.append(
            f"  [{card['id']}] {card['title']} — {card['kind']} · "
            f"{card.get('size', 0):,} bytes · {card['path']}")
    return "\n".join(lines)
