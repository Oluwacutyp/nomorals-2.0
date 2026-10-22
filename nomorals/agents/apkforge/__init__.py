"""ApkForge — Devon's APK class (A1: intake + dissect + teach + diff).

Lands APKs and bundle containers from the owner's uploads or by URL,
dissects them with androguard + APKiD into a Dossier whose every
claim traces to a real parse, teaches Android anatomy from the
specimen itself, and diffs versions component-wise.  Pure Python —
full capability on every profile, Termux included.  Editing,
building and signing arrive in A2 behind the Java toolchain and are
reported as not-yet-built until they exist; installing/testing on a
device is out of scope by design (no device lab organ).
"""

from __future__ import annotations

from typing import Any

from .diff import render_diff_card, render_diff_md
from .dissect import render_dossier_card, render_dossier_md
from .forge import ApkForge, render_list, render_project, render_status
from .project import apkforge_dir, museum_card, techforge_feed

__all__ = [
    "ApkForge",
    "apkforge_dir",
    "museum_card",
    "techforge_feed",
    "render_list",
    "render_project",
    "render_status",
    "render_dossier_card",
    "render_dossier_md",
    "render_diff_card",
    "render_diff_md",
    "register",
]


def register(registry: Any) -> None:
    """Register the ``apkforge`` agent tool (add / list / status /
    dissect / teach / diff)."""
    from ...core.policy import Capability

    context = registry.context

    @registry.register(
        "apkforge",
        description=(
            "ApkForge — the APK class (A1: read-only dissection). "
            "action=add (land an APK/bundle from a file path or URL, "
            "provenance + hash recorded, original immutable) | list "
            "(landed projects) | status (host capability matrix — "
            "edit/build honestly marked A2) | dissect (androguard + "
            "APKiD anatomy: manifest, permissions with AOSP weight, "
            "components + exported flags, dex census, certificate, "
            "packer verdict → dossier card + full MD) | teach "
            "(Android anatomy lessons from this APK's own dossier; "
            "topic=overview|manifest|permissions|components|"
            "certificate|code|build) | diff (two projects: "
            "permissions/components/cert/dex deltas). Static "
            "analysis only — never executes the app."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — add|list|status|dissect|teach|diff",
            "source": "str — file path or URL for action=add",
            "project": "str — project id/prefix/package for "
                       "dissect|teach, or the first project for diff",
            "other": "str — second project for action=diff",
            "topic": "str — teach topic (see description)",
        },
    )
    def apkforge(*, action: str = "status", source: str = "",
                 project: str = "", other: str = "",
                 topic: str = "") -> dict[str, Any]:
        act = (action or "status").strip().lower()
        forge = ApkForge(context)
        if act == "status":
            return forge.status()
        if act == "list":
            return forge.list_projects()
        if act == "add":
            return forge.add(source)
        if act == "dissect":
            return forge.dissect(project)
        if act == "teach":
            return forge.teach(project, topic)
        if act == "diff":
            return forge.diff(project, other)
        return {"error": f"unknown apkforge action {act!r} — "
                         "add|list|status|dissect|teach|diff"}
