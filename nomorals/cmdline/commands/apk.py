"""``nm apk`` — ApkForge: intake, dissect, teach, diff APKs."""

from __future__ import annotations

from typing import Any


def _cmd_apk(args: Any, context: Any) -> int:
    """Route `nm apk` subcommands.  A1: intake + dissect + diff."""
    from ...agents.apkforge import (ApkForge, render_diff_card,
                                    render_dossier_card, render_list,
                                    render_project, render_status)

    forge = ApkForge(context)
    action = args.apk_action

    if action == "status":
        print(render_status(forge.status()))
        return 0

    if action == "list":
        print(render_list(forge.list_projects()))
        return 0

    if action == "add":
        res = forge.add(args.source)
        if res.get("error"):
            print(res["error"])
            return 2
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
            print(res["error"])
            return 2
        print(render_dossier_card(res["dossier"]))
        print(f"\nfull dossier: {res['dossier_path']}")
        return 0

    if action == "teach":
        res = forge.teach(args.project, args.topic or "overview")
        if res.get("error"):
            print(res["error"])
            return 2
        print(res["text"])
        return 0

    if action == "diff":
        res = forge.diff(args.project, args.other)
        if res.get("error"):
            print(res["error"])
            return 2
        print(render_diff_card(res["diff"]))
        print(f"\nfull report: {res['report_path']}")
        return 0

    print(f"unknown apk action: {action}")
    return 2
