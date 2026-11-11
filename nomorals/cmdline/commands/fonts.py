"""``nm fonts`` — the FontLibrary provisioning + inspection
surface: stock the pack, verify it, and read the registry.

Stocking is the one network act in the whole font programme
(one-time, whole-library, from the pinned google/fonts
commit); everything after it renders offline from the stock.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def _cmd_fonts(args: argparse.Namespace, context: Any) -> int:
    """Route `nm fonts` subcommands."""
    from ...media.typography import fontlib, fontpack

    action = args.fonts_action

    if action == "stock":
        try:
            if args.build:
                manifest = fontpack.build_pack(
                    args.pack_dir,
                    commit=args.commit or
                    fontpack.DEFAULT_SOURCE_COMMIT,
                    log=print)
                print(f"built: {manifest['family_count']} "
                      f"families, {manifest['file_count']} files, "
                      f"{manifest['total_bytes']} bytes")
            report = fontpack.stock_pack(args.pack_dir, log=print)
        except fontpack.FontPackError as exc:
            print(f"fonts: {exc}")
            return 1
        print(json.dumps(report, indent=2))
        return 0

    if action == "verify":
        report = fontpack.verify_pack(
            args.pack_dir, full=not args.sample,
            sample=25 if args.sample else 0)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    if action == "build":
        try:
            manifest = fontpack.build_pack(
                args.pack_dir,
                commit=args.commit or fontpack.DEFAULT_SOURCE_COMMIT,
                log=print)
        except fontpack.FontPackError as exc:
            print(f"fonts: {exc}")
            return 1
        print(f"built: {manifest['family_count']} families, "
              f"{manifest['file_count']} files, "
              f"{manifest['total_bytes']} bytes "
              f"@ {manifest['source']['commit'][:12]}")
        return 0

    # list / search / info — reads of the live registry.
    library = fontlib.FontLibrary.discover()
    if action == "list":
        stats = library.stats()
        print(json.dumps(stats, indent=2))
        return 0
    if action == "search":
        hits = library.search(args.query)
        for fam in hits:
            print(f"{fam.name} · {fam.category or '—'} · "
                  f"{len(fam.faces)} faces · {fam.tier}")
        print(f"{len(hits)} families")
        return 0
    if action == "info":
        fam = library.family(args.family)
        if fam is None:
            try:
                library.resolve(args.family)
            except fontlib.FontNotStocked as exc:
                print(str(exc))
            return 1
        print(f"{fam.name} — {fam.category or '—'} · "
              f"licence {fam.license or '—'} · tier {fam.tier}")
        for face in fam.faces:
            print(f"  {face.file} (weight {face.weight}"
                  f"{', variable' if face.variable else ''}"
                  f"{', italic' if face.italic else ''})")
        print(f"resolves to: {library.resolve(fam.name)}")
        return 0
    return 1
