"""``nm apk`` — ApkForge: intake, dissect, teach, diff (A1) +
toolchain, decode, edit, patch, undo, build, keys, splits, jadx
(A2 workbench) + bisect, signatures, hexpatch, artifacts, send
(A3) + buildsrc, new, read (A4 build-from-source)."""

from __future__ import annotations

from typing import Any


def _cmd_apk(args: Any, context: Any) -> int:
    """Route `nm apk` subcommands."""
    from ...agents.apkforge import (ApkForge, render_artifacts,
                                    render_bisect_card,
                                    render_build_card,
                                    render_diff_card,
                                    render_dossier_card, render_list,
                                    render_project,
                                    render_signatures,
                                    render_srcbuild_card,
                                    render_status, render_toolchain)

    forge = ApkForge(context)
    action = args.apk_action

    def _fail(res: dict) -> int:
        print(res["error"])
        return 2

    if action == "status":
        print(render_status(forge.status()))
        return 0

    if action == "list":
        print(render_list(forge.list_projects()))
        return 0

    if action == "add":
        res = forge.add(args.source)
        if res.get("error"):
            return _fail(res)
        print(render_project(res["project"]))
        return 0

    if action == "show":
        proj = forge.resolve(args.project)
        if proj is None:
            print(f"no project matches {args.project!r}")
            return 2
        print(render_project(proj.card()))
        return 0

    if action == "dissect":
        res = forge.dissect(args.project,
                            include_apkid=not args.no_apkid)
        if res.get("error"):
            return _fail(res)
        print(render_dossier_card(res["dossier"]))
        print(f"\nfull dossier: {res['dossier_path']}")
        return 0

    if action == "teach":
        res = forge.teach(args.project, args.topic or "overview",
                          target_class=getattr(args, "target_class",
                                               "") or "",
                          method=getattr(args, "method", "") or "",
                          mapping=getattr(args, "mapping", "") or "")
        if res.get("error"):
            return _fail(res)
        print(res["text"])
        return 0

    if action == "diff":
        res = forge.diff(args.project, args.other)
        if res.get("error"):
            return _fail(res)
        print(render_diff_card(res["diff"]))
        print(f"\nfull report: {res['report_path']}")
        return 0

    if action == "toolchain":
        if args.op == "provision":
            res = forge.toolchain_provision()
            if res.get("error"):
                return _fail(res)
            print(f"installed now: {res['provisioned_now']}")
            print(f"already present: {res['already_present']}")
            print(f"smoke build: {res['smoke_apktool_build']}")
            return 0
        if args.op == "provision-srcbuild":
            res = forge.toolchain_provision_srcbuild()
            if res.get("error"):
                return _fail(res)
            print(f"installed now: {res['provisioned_now']}")
            print(f"already present: {res['already_present']}")
            print(f"sdk root: {res['sdk_root']}")
            print(f"sdk licenses: {res['sdk_licenses']}")
            return 0
        print(render_toolchain(forge.toolchain_status()))
        return 0

    if action == "buildsrc":
        res = forge.buildsrc(
            args.source, route=args.route, variant=args.variant,
            bundle=args.bundle, use_r8=args.r8, alias=args.alias)
        if res.get("error"):
            return _fail(res)
        print(render_srcbuild_card(res["build"]))
        return 0

    if action == "new":
        res = forge.new_project(args.name, package=args.package)
        if res.get("error"):
            return _fail(res)
        print(res["card_text"])
        return 0

    if action == "read":
        res = forge.read_text(args.project, args.target)
        if res.get("error"):
            return _fail(res)
        print(res["file"]["text"])
        return 0

    if action == "decode":
        res = forge.decode(args.project)
        if res.get("error"):
            return _fail(res)
        print(f"decoded → {res['decoded']}")
        return 0

    if action == "edit":
        def _b(s: str):
            return "" if s == "" else s.lower() in ("1", "true", "yes")

        res = forge.edit(
            args.project, args.op,
            label=args.label, version_name=args.version_name,
            version_code=args.version_code, min_sdk=args.min_sdk,
            target_sdk=args.target_sdk, debuggable=_b(args.debuggable),
            allow_backup=_b(args.allow_backup),
            add_permission=args.add_permission,
            remove_permission=args.remove_permission,
            name=args.name, value=args.value, locale=args.locale,
            target=args.target, source=args.source)
        if res.get("error"):
            return _fail(res)
        ed = res["edit"]
        print(f"edit #{ed['edit_id']}: {ed['detail']}")
        print(f"snapshot: {ed['snapshot']}")
        return 0

    if action == "patch":
        import sys

        raw = (sys.stdin.read() if args.patch_file == "-"
               else open(args.patch_file, encoding="utf-8").read())
        res = forge.patch(args.project, raw)
        if res.get("error"):
            return _fail(res)
        pr = res["patch_result"]
        print(f"patch {pr['patch']!r} applied as edit "
              f"#{pr['edit_id']}: {pr['detail']}")
        return 0

    if action == "edits":
        res = forge.edits(args.project)
        if res.get("error"):
            return _fail(res)
        if not res["edits"]:
            print("no edits yet")
            return 0
        for e in res["edits"]:
            print(f"  #{e['edit_id']} {e['op']}: {e.get('detail', '')}"
                  + (" (undone)" if e["undone"] else ""))
        return 0

    if action == "undo":
        res = forge.undo(args.project,
                         int(args.edit_id) if args.edit_id else None)
        if res.get("error"):
            return _fail(res)
        print(f"reverted edit #{res['undo']['undone_edit_id']}")
        return 0

    if action == "build":
        res = forge.build(args.project, alias=args.alias)
        if res.get("error"):
            return _fail(res)
        print(render_build_card(res["build"]))
        return 0

    if action == "keys":
        if args.op == "ensure":
            res = forge.ensure_key(args.alias)
            if res.get("error"):
                return _fail(res)
            k = res["keystore"]
            print(f"{k['alias']}: cert {k['cert_sha256'][:24]}… "
                  f"({'created + vaulted' if k['created'] else 'in vault'})")
            return 0
        res = forge.keys()
        if res.get("error"):
            return _fail(res)
        if not res["keystores"]:
            print("no ApkForge keystores in the vault yet")
            return 0
        for k in res["keystores"]:
            print(f"  {k['alias']}: cert {k['cert_sha256'][:24]}…"
                  + (" (DEBUG)" if k["debug"] else ""))
        return 0

    if action == "splits":
        res = forge.splits(args.project, args.op, part=args.part,
                           spec=args.spec, alias=args.alias)
        if res.get("error"):
            return _fail(res)
        if args.op == "list":
            for p in res["parts"]:
                print(f"  {p['name']} — {p.get('size', 0):,} bytes · "
                      f"{p.get('package') or '—'}")
            return 0
        print({k: v for k, v in res.items() if k != "project"})
        return 0

    if action == "jadx":
        res = forge.jadx(args.project, mapping=args.mapping)
        if res.get("error"):
            return _fail(res)
        print(f"jadx → {res['jadx']} "
              f"({res.get('java_files', 'cached')} java files)")
        return 0

    if action == "bisect":
        res = forge.bisect(args.series, args.check)
        if res.get("error"):
            return _fail(res)
        print(render_bisect_card(res))
        if res.get("diff_path"):
            print(f"\ntransition diff: {res['diff_path']}")
        return 0

    if action == "signatures":
        res = forge.signatures(args.op, prefix=args.prefix,
                               label=args.label, kind=args.kind,
                               reason=args.reason)
        if res.get("error"):
            return _fail(res)
        if "card_text" in res:
            print(res["card_text"])
        elif "promoted" in res:
            e = res["promoted"]
            print(f"promoted {e['prefix']} — {e['label']} "
                  f"({e['kind']}), evidence: "
                  f"{e['provenance']['occurrences']} dossier(s)")
        elif "demoted" in res:
            print(f"demoted {res['demoted']['prefix']}")
        else:
            print(res.get("proposals"))
        return 0

    if action == "hexpatch":
        import sys

        raw = (sys.stdin.read() if args.patch_file == "-"
               else open(args.patch_file, encoding="utf-8").read())
        res = forge.hexpatch(args.project, raw)
        if res.get("error"):
            return _fail(res)
        hr = res["hexpatch_result"]
        print(f"hex patch {hr['patch']!r} applied as edit "
              f"#{hr['edit_id']}: {hr['detail']}")
        return 0

    if action == "artifacts":
        res = forge.artifacts()
        print(render_artifacts(res))
        return 0

    if action == "send":
        res = forge.send_artifact(args.artifact, args.platform,
                                  args.chat_id)
        if res.get("error"):
            return _fail(res)
        print(f"sent {res['artifact']['id']} → "
              f"{res['sent']['platform']}:{res['sent']['chat_id']}")
        return 0

    print(f"unknown apk action: {action}")
    return 2
