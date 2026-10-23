"""ApkForge — Devon's APK class (A1: intake + dissect + teach +
diff; A2: editor + builder + signing + splits; A3: version bisect,
tracker-signature growth, native hex patching, Museum wiring;
A4: build-from-source — the manual aapt2/javac/d8 pipeline, the
R8 variant with mapping-based jadx name restoration, the Gradle
route with a real AAB output, project scaffolding, and general
text-file editing of the decoded tree).

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
A3 completes the class: version-series bisect on the git-bisect
good/bad/skip contract with journaled, resumable sessions and the
transition pair's diff as evidence; the seed tracker/SDK table
grows from dissection evidence (discovery → proposal → journaled
promotion/demotion, grown hits marked in every dossier); native
.so byte-pattern patching under the same fail-closed, snapshot-
reversible discipline, rebuilt + re-signed through the A2 path;
and Museum wiring — projects, dossiers and builds as one listable
artifact census, sendable through the repo's file-send surface.
Installing/testing on a device stays out of scope by design
(no device lab organ).
A5 burns down the class's named weaknesses: signature discovery
reads the referenced-type census (not just defined classes), its
evidence split defined vs referenced, under a config recurrence
threshold; the bisect's exported: predicate is three-valued
(absent / present-not-exported / present-exported) with every
state crossing reported by name, and damaged specimens (a dex
androguard cannot parse) skip with the real parse error
journaled; the native census parses real ELF images — sections,
dynamic symbols, relocations — instead of listing zip entries;
teach v2 walks one method in its smali and its jadx Java side
by side, divergences named.
"""

from __future__ import annotations

from typing import Any

from .bisect import render_bisect_card
from .build import render_build_card
from .diff import render_diff_card, render_diff_md
from .dissect import render_dossier_card, render_dossier_md
from .forge import (ApkForge, render_list, render_project,
                    render_status, render_toolchain)
from .museum import render_artifacts
from .project import apkforge_dir, museum_card, techforge_feed
from .signatures import render_signatures
from .srcbuild import render_srcbuild_card

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
    "render_srcbuild_card",
    "render_dossier_card",
    "render_dossier_md",
    "render_diff_card",
    "render_diff_md",
    "render_bisect_card",
    "render_signatures",
    "render_artifacts",
    "register",
]


def register(registry: Any) -> None:
    """Register the ``apkforge`` agent tools: the read tool (A1
    actions + A3's bisect / signature census / artifact census)
    and ``apkforge_build`` (A2's write / exec tier + A3's hexpatch,
    signature promotion/demotion and artifact send)."""
    from ...core.policy import Capability

    context = registry.context

    @registry.register(
        "apkforge",
        description=(
            "ApkForge — the APK class (A1: read-only dissection; "
            "A3: bisect + signature growth + Museum census). "
            "action=add (land an APK/bundle from a file path or URL, "
            "provenance + hash recorded, original immutable) | list "
            "(landed projects) | status (host capability matrix) | "
            "dissect (androguard + APKiD anatomy: manifest, "
            "permissions with AOSP weight, components + exported "
            "flags, dex census, certificate, packer verdict → "
            "dossier card + full MD) | teach (Android anatomy "
            "lessons from this APK's own dossier; "
            "topic=overview|manifest|permissions|components|"
            "certificate|code|build) | teach v2 (target_class = a "
            "class in the project, method optional, mapping = an "
            "R8 mapping.txt: one method walked in its smali AND "
            "its jadx Java, divergences named) | diff (two "
            "projects: permissions/components/cert/dex deltas) | "
            "bisect "
            "(series = comma-separated project refs of ONE app; "
            "check = a dossier predicate like 'permission: "
            "android.permission.CAMERA', 'class: com.x.Y', "
            "'tracker: <name>', 'exported: <component>', "
            "'endpoint: <substr>', 'cert: <sha-prefix>' — halves "
            "the version series to the exact flip, journaled + "
            "resumable, transition diff attached) | signatures "
            "(op=list|proposals — the tracker/SDK table: seed + "
            "grown entries, candidates, proposals; promotion lives "
            "in apkforge_build) | artifacts (the Museum census: "
            "projects, dossiers, builds as sendable cards). Static "
            "analysis only — never executes the app. Editing and "
            "building live in the apkforge_build tool."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — add|list|status|dissect|teach|diff|"
                      "bisect|signatures|artifacts",
            "source": "str — file path or URL for action=add",
            "project": "str — project id/prefix/package for "
                       "dissect|teach, or the first project for diff",
            "other": "str — second project for action=diff",
            "topic": "str — teach topic (see description)",
            "target_class": "str — teach v2: the class to walk, "
                            "dotted or descriptor form",
            "method": "str — teach v2: the method to walk "
                      "(default: first real method in both forms)",
            "mapping": "str — teach v2: an R8/ProGuard mapping.txt "
                       "(project-relative or absolute) for name "
                       "restoration",
            "series": "str — comma-separated project refs "
                      "(action=bisect)",
            "check": "str — dossier predicate (action=bisect)",
            "op": "str — signatures op: list|proposals",
        },
    )
    def apkforge(*, action: str = "status", source: str = "",
                 project: str = "", other: str = "",
                 topic: str = "", series: str = "", check: str = "",
                 op: str = "", target_class: str = "",
                 method: str = "", mapping: str = ""
                 ) -> dict[str, Any]:
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
            return forge.teach(project, topic,
                               target_class=target_class,
                               method=method, mapping=mapping)
        if act == "diff":
            return forge.diff(project, other)
        if act == "bisect":
            return forge.bisect(series or project, check)
        if act == "signatures":
            return forge.signatures(op or "list")
        if act == "artifacts":
            return forge.artifacts()
        return {"error": f"unknown apkforge action {act!r} — "
                         "add|list|status|dissect|teach|diff|bisect|"
                         "signatures|artifacts"}

    @registry.register(
        "apkforge_build",
        description=(
            "ApkForge workbench (A2: editor + builder + signing + "
            "splits; A3: native hex patching, signature promotion, "
            "artifact send; A4: build-from-source, scaffolding, "
            "general text editing) over projects landed by the "
            "apkforge tool. action=toolchain (op=status|provision|"
            "provision-srcbuild — pinned, sha256-verified "
            "JDK/apktool/jadx/bundletool/APKEditor/build-tools, "
            "plus the A4 srcbuild set: SDK platform 35, "
            "cmdline-tools, R8, Gradle; probe-executed capability "
            "report) | "
            "decode (apktool decode the original into the project) "
            "| edit (op=manifest|strings|replace|add|remove|write; "
            "manifest changes via fields: label, version_name, "
            "version_code, min_sdk, target_sdk, debuggable, "
            "allow_backup, add_permission, remove_permission, "
            "add_component, remove_component; strings via "
            "name/value/locale; resource files via target/source; "
            "write = a general TEXT file write into the decoded "
            "tree (smali above all — target = decoded path, "
            "content in value or from the source file; binaries "
            "refuse by name) — every edit journaled + snapshotted) "
            "| read (one text file from the decoded tree: project "
            "+ target) | patch (a fail-closed "
            "fingerprint patch document as JSON in 'patch') | "
            "hexpatch (A3: a native byte-pattern patch document as "
            "JSON in 'patch' — target lib/<abi>/*.so in the decoded "
            "project, hex pattern with ?? wildcards, replacement "
            "of identical length, occurrence count must match "
            "expect_matches exactly or nothing changes; lands as a "
            "journaled, undoable edit, rebuild via action=build) | "
            "edits | undo (edit_id optional) | build (apktool "
            "rebuild → zipalign → sign with the vault-held class "
            "keystore → verify; card prints the re-sign facts) | "
            "buildsrc (A4 — build an app from a SOURCE project: "
            "source = local dir | .zip | git URL | .zip URL; route "
            "auto-detected — gradle wrapper/build files → the "
            "Gradle route (variant=debug|release|all; bundle=true "
            "also runs bundleRelease for a real .aab), simple "
            "manifest+res+java layout → the manual aapt2 compile/"
            "link → javac → d8 pipeline (r8=true for the R8 "
            "shrinking variant with mapping.txt); output signed + "
            "verified, card names the route) | new (A4 — scaffold "
            "a real buildable app project: name, package via "
            "'value'; lands in the sources area, ready for "
            "buildsrc) | "
            "keys (op=list|ensure, alias) | splits (op=list|"
            "extract|merge|installset|generate; part / spec JSON "
            "for a device) | jadx (decompile to readable Java — "
            "reading surface only; mapping = a ProGuard/R8 "
            "mapping.txt path restores original class names) | "
            "signatures (op=promote|demote "
            "— A3 tracker-table growth: promote a proposed prefix "
            "(prefix; label via 'label', kind via 'value') into "
            "the active table, or demote a grown entry (prefix, "
            "reason); both journaled + reversible; seeds cannot "
            "be demoted) | "
            "send (A3 Museum wiring: deliver one artifact — "
            "artifact = id/prefix from the artifacts census — to "
            "platform + chat_id via the gateway file-send). "
            "Re-signed APKs do NOT install over the original and "
            "fail Play Integrity — stated on every build card."
        ),
        capability=Capability.EXEC_SHELL,
        parameters={
            "action": "str — toolchain|decode|edit|read|patch|"
                      "hexpatch|edits|undo|build|buildsrc|new|"
                      "keys|splits|jadx|signatures|send",
            "project": "str — project id/prefix/package",
            "op": "str — sub-operation for edit/splits/keys/"
                  "toolchain/signatures",
            "patch": "str — patch document JSON (action=patch or "
                   "action=hexpatch)",
            "prefix": "str — signature package prefix "
                      "(signatures promote|demote)",
            "reason": "str — demotion reason (signatures demote)",
            "artifact": "str — artifact id/prefix (action=send)",
            "platform": "str — chat platform (action=send)",
            "chat_id": "str — destination chat (action=send)",
            "target": "str — decoded-project path for edit "
                      "replace/add/remove/write and action=read",
            "source": "str — source file for edit replace/add; "
                      "the source project for action=buildsrc "
                      "(dir | .zip | git URL | .zip URL)",
            "name": "str — string name (edit strings); project "
                    "name (action=new)",
            "value": "str — string value (edit strings); text "
                     "content (edit write); package (action=new)",
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
            "route": "str — buildsrc route override: manual|"
                     "gradle (default auto-detect)",
            "variant": "str — gradle buildsrc variant: "
                       "debug|release|all (default debug)",
            "bundle": "str — true: gradle buildsrc also runs "
                      "bundleRelease (a real .aab)",
            "r8": "str — true: manual buildsrc uses R8 (mapping "
                  ".txt produced) instead of d8",
            "mapping": "str — jadx: ProGuard/R8 mapping.txt path "
                       "for original-name restoration",
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
                       force: str = "", prefix: str = "",
                       reason: str = "", artifact: str = "",
                       platform: str = "", chat_id: str = "",
                       route: str = "", variant: str = "",
                       bundle: str = "", r8: str = "",
                       mapping: str = ""
                       ) -> dict[str, Any]:
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
            if (op or "status") == "provision-srcbuild":
                return forge.toolchain_provision_srcbuild(
                    force=forced)
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
        if act == "read":
            return forge.read_text(project, target)
        if act == "buildsrc":
            return forge.buildsrc(
                source or project, route=route,
                variant=variant or "debug",
                bundle=str(bundle).lower() in ("1", "true", "yes"),
                use_r8=str(r8).lower() in ("1", "true", "yes"),
                alias=alias)
        if act == "new":
            return forge.new_project(name or project, package=value)
        if act == "patch":
            return forge.patch(project, patch)
        if act == "hexpatch":
            return forge.hexpatch(project, patch)
        if act == "signatures":
            return forge.signatures(op or "list", prefix=prefix,
                                    label=label, kind=value,
                                    reason=reason)
        if act == "send":
            return forge.send_artifact(artifact or project,
                                       platform, chat_id)
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
            return forge.jadx(project, force=forced, mapping=mapping)
        return {"error": f"unknown apkforge_build action {act!r} — "
                         "toolchain|decode|edit|read|patch|hexpatch|"
                         "edits|undo|build|buildsrc|new|keys|splits|"
                         "jadx|signatures|send"}
