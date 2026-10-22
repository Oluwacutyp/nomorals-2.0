"""ApkForge — Devon's APK class (A1: intake + dissect + teach +
diff; A2: editor + builder + signing + splits).

Lands APKs and bundle containers from the owner's uploads or by
URL, dissects them with androguard + APKiD into a Dossier whose
every claim traces to a real parse, teaches Android anatomy from
the specimen itself, and diffs versions component-wise (A1).
A2 adds the workbench: a self-provisioned pinned Java toolchain
(JDK + apktool + jadx + bundletool + APKEditor + build-tools,
sha256-verified, probe-executed), apktool decode, journaled +
snapshot-reversible edits (manifest / strings / resources), a
fail-closed fingerprint patch engine, rebuild → zipalign →
vault-keystore sign → verify with honest build cards, and bundle
splits (list / extract / merge / installset / generate).
Installing/testing on a device stays out of scope by design
(no device lab organ).
"""

from __future__ import annotations

from typing import Any

from .build import render_build_card
from .diff import render_diff_card, render_diff_md
from .dissect import render_dossier_card, render_dossier_md
from .forge import (ApkForge, render_list, render_project,
                    render_status, render_toolchain)
from .project import apkforge_dir, museum_card, techforge_feed

__all__ = [
    "ApkForge",
    "apkforge_dir",
    "museum_card",
    "techforge_feed",
    "render_list",
    "render_project",
    "render_status",
    "render_toolchain",
    "render_build_card",
    "render_dossier_card",
    "render_dossier_md",
    "render_diff_card",
    "render_diff_md",
    "register",
]


def register(registry: Any) -> None:
    """Register the ``apkforge`` agent tools: the read tool (A1
    actions, unchanged tier) and ``apkforge_build`` (A2's write /
    exec tier: toolchain, decode, edit, patch, undo, build, keys,
    splits, jadx)."""
    from ...core.policy import Capability

    context = registry.context

    @registry.register(
        "apkforge",
        description=(
            "ApkForge — the APK class (A1: read-only dissection). "
            "action=add (land an APK/bundle from a file path or URL, "
            "provenance + hash recorded, original immutable) | list "
            "(landed projects) | status (host capability matrix) | "
            "dissect (androguard + APKiD anatomy: manifest, "
            "permissions with AOSP weight, components + exported "
            "flags, dex census, certificate, packer verdict → "
            "dossier card + full MD) | teach (Android anatomy "
            "lessons from this APK's own dossier; "
            "topic=overview|manifest|permissions|components|"
            "certificate|code|build) | diff (two projects: "
            "permissions/components/cert/dex deltas). Static "
            "analysis only — never executes the app. Editing and "
            "building live in the apkforge_build tool."
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

    @registry.register(
        "apkforge_build",
        description=(
            "ApkForge workbench (A2: editor + builder + signing + "
            "splits) over projects landed by the apkforge tool. "
            "action=toolchain (op=status|provision — pinned, "
            "sha256-verified JDK/apktool/jadx/bundletool/APKEditor/"
            "build-tools, probe-executed capability report) | "
            "decode (apktool decode the original into the project) "
            "| edit (op=manifest|strings|replace|add|remove; "
            "manifest changes via fields: label, version_name, "
            "version_code, min_sdk, target_sdk, debuggable, "
            "allow_backup, add_permission, remove_permission, "
            "add_component, remove_component; strings via "
            "name/value/locale; files via target/source — every "
            "edit journaled + snapshotted) | patch (a fail-closed "
            "fingerprint patch document as JSON in 'patch') | "
            "edits | undo (edit_id optional) | build (apktool "
            "rebuild → zipalign → sign with the vault-held class "
            "keystore → verify; card prints the re-sign facts) | "
            "keys (op=list|ensure, alias) | splits (op=list|"
            "extract|merge|installset|generate; part / spec JSON "
            "for a device) | jadx (decompile to readable Java — "
            "reading surface only). Re-signed APKs do NOT install "
            "over the original and fail Play Integrity — stated "
            "on every build card."
        ),
        capability=Capability.EXEC_SHELL,
        parameters={
            "action": "str — toolchain|decode|edit|patch|edits|"
                      "undo|build|keys|splits|jadx",
            "project": "str — project id/prefix/package",
            "op": "str — sub-operation for edit/splits/keys/"
                  "toolchain",
            "patch": "str — patch document JSON (action=patch)",
            "target": "str — decoded-project path for edit "
                      "replace/add/remove",
            "source": "str — source file for edit replace/add",
            "name": "str — string name (edit strings)",
            "value": "str — string value (edit strings)",
            "locale": "str — values-<locale> for edit strings",
            "label": "str — manifest label",
            "version_name": "str — manifest versionName",
            "version_code": "str — manifest versionCode",
            "min_sdk": "str — manifest minSdkVersion",
            "target_sdk": "str — manifest targetSdkVersion",
            "debuggable": "str — true|false manifest debuggable",
            "allow_backup": "str — true|false manifest allowBackup",
            "add_permission": "str — permission to add",
            "remove_permission": "str — permission to remove",
            "add_component": "str — JSON {tag,name,exported?}",
            "remove_component": "str — component class to remove",
            "edit_id": "str — edit id for undo (default latest)",
            "alias": "str — keystore alias (default 'devon'; "
                     "'debug' = labeled debug key)",
            "part": "str — split part name (splits extract)",
            "spec": "str — device spec JSON (splits installset)",
            "force": "str — true to force decode/jadx/provision",
        },
    )
    def apkforge_build(*, action: str = "toolchain", project: str = "",
                       op: str = "", patch: str = "", target: str = "",
                       source: str = "", name: str = "", value: str = "",
                       locale: str = "", label: str = "",
                       version_name: str = "", version_code: str = "",
                       min_sdk: str = "", target_sdk: str = "",
                       debuggable: str = "", allow_backup: str = "",
                       add_permission: str = "",
                       remove_permission: str = "",
                       add_component: str = "",
                       remove_component: str = "", edit_id: str = "",
                       alias: str = "", part: str = "", spec: str = "",
                       force: str = "") -> dict[str, Any]:
        act = (action or "toolchain").strip().lower()
        forge = ApkForge(context)
        forced = str(force).lower() in ("1", "true", "yes")

        def _bool(s: str) -> Any:
            if s == "":
                return ""
            return str(s).lower() in ("1", "true", "yes")

        if act == "toolchain":
            if (op or "status") == "provision":
                return forge.toolchain_provision(force=forced)
            return forge.toolchain_status()
        if act == "decode":
            return forge.decode(project, force=forced)
        if act == "edit":
            return forge.edit(
                project, op, label=label, version_name=version_name,
                version_code=version_code, min_sdk=min_sdk,
                target_sdk=target_sdk, debuggable=_bool(debuggable),
                allow_backup=_bool(allow_backup),
                add_permission=add_permission,
                remove_permission=remove_permission,
                add_component=add_component,
                remove_component=remove_component,
                name=name, value=value, locale=locale,
                target=target, source=source)
        if act == "patch":
            return forge.patch(project, patch)
        if act == "edits":
            return forge.edits(project)
        if act == "undo":
            return forge.undo(project,
                              int(edit_id) if edit_id else None)
        if act == "build":
            return forge.build(project, alias=alias)
        if act == "keys":
            if (op or "list") == "ensure":
                return forge.ensure_key(alias)
            return forge.keys()
        if act == "splits":
            return forge.splits(project, op or "list", part=part,
                                spec=spec, alias=alias)
        if act == "jadx":
            return forge.jadx(project, force=forced)
        return {"error": f"unknown apkforge_build action {act!r} — "
                         "toolchain|decode|edit|patch|edits|undo|"
                         "build|keys|splits|jadx"}
