"""``nm maps`` — the Devon Atlas basemap pipeline (docs 19+21 M2).

The operator surface over :mod:`nomorals.maps.pipeline`:

* ``nm maps build <pack>`` runs the whole pipeline for a
  registered pack definition (acquire → build → package →
  verify → install) and prints the run's truth — stages reached,
  tile count, digest, signature state, receipt ids stay internal
  (operator text carries facts, never internal ids; ``--json``
  is the machine surface).
* ``nm maps packs`` lists the registered definitions and what
  this host actually holds (the release registry).
* ``nm maps verify <pack|dir>`` re-verifies an installed pack
  against its manifest through the shipped M1 operation.
* ``nm maps sources`` lists the registered sources with their
  hash discipline and attribution.
* ``nm maps index [pack]`` builds or resumes the gazetteer
  index over the installed packs (the universality pass): the
  one build door for the local geocoder's index — chunked and
  digest-cached, so an interrupted build resumes and a current
  pack is a no-op.  Query paths never build it inline; they
  name this command when the index is absent, stale or
  mid-build.  ``--status`` reports the state and builds
  nothing; ``--force`` rebuilds even digest-current packs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..emit import _emit


def _home_of(context: Any) -> str:
    """The Devon home: the context settings' home, else ~/.nomorals."""
    settings = getattr(context, "settings", None)
    if settings is not None:
        return str(getattr(settings, "home", "~/.nomorals"))
    return "~/.nomorals"


def _packs_dir(args: argparse.Namespace, context: Any) -> Path:
    named = str(getattr(args, "packs_dir", "") or "")
    if named:
        return Path(named).expanduser()
    from ...maps.settings import map_settings

    configured = map_settings().packs_dir
    if configured:
        return Path(configured).expanduser()
    return Path(_home_of(context)).expanduser() / "maps" / "packs"


def _build_root(args: argparse.Namespace, context: Any) -> Path:
    named = str(getattr(args, "build_dir", "") or "")
    if named:
        return Path(named).expanduser()
    return Path(_home_of(context)).expanduser() / "maps" / "build"


def _build(args: argparse.Namespace, context: Any) -> int:
    from ...maps.pipeline import run_build
    from ...maps.receipts import MapReceiptStore

    home = str(Path(_home_of(context)).expanduser())
    store = MapReceiptStore(home=home)
    report = run_build(
        str(args.pack), build_root=_build_root(args, context),
        packs_dir=_packs_dir(args, context), store=store,
        xmx=str(getattr(args, "xmx", "2g") or "2g"),
        threads=int(getattr(args, "threads", 2) or 2),
        build_timeout_s=float(getattr(args, "timeout", 7200.0)
                              or 7200.0))
    lines = ["maps build " + report.pack_id + " — "
             + ("OK" if report.ok else "FAILED")]
    lines.append("stages: " + " → ".join(report.stages or ["(none)"]))
    if report.tiles:
        lines.append("tiles: " + str(report.tiles))
    if report.pack_dir:
        lines.append("installed: " + report.pack_dir)
    if report.digest:
        lines.append("digest: " + report.digest)
    if report.signature_state:
        lines.append("signature: " + report.signature_state)
    if report.error:
        lines.append("error: " + report.error)
    _emit(args, report.to_dict(), "\n".join(lines))
    return 0 if report.ok else 1


def _packs(args: argparse.Namespace, context: Any) -> int:
    from ...maps.definitions import PACKS
    from ...maps.registry import ReleaseRegistry

    entries = {e.pack_id: e
               for e in ReleaseRegistry(_packs_dir(args, context)).all()}
    rows: list[dict[str, Any]] = []
    lines = ["map packs — registered definitions + installed releases"]
    for pack in PACKS:
        entry = entries.get(pack.pack_id)
        state = "not installed"
        if entry is not None:
            state = ("installed · digest " + entry.digest[:16]
                     + "… · signature " + entry.signature_state)
        lines.append(
            "  " + pack.pack_id + " — " + pack.name
            + " · z" + str(pack.zoom_min) + "–" + str(pack.zoom_max)
            + (" (stretch z" + str(pack.stretch_zoom_max) + ")"
               if pack.stretch_zoom_max else "")
            + " · " + pack.tile_format + " · " + state)
        rows.append({"pack_id": pack.pack_id, "name": pack.name,
                     "zoom_min": pack.zoom_min,
                     "zoom_max": pack.zoom_max,
                     "stretch_zoom_max": pack.stretch_zoom_max,
                     "tile_format": pack.tile_format,
                     "installed": entry.to_dict() if entry else None})
    _emit(args, {"packs": rows}, "\n".join(lines))
    return 0


def _verify(args: argparse.Namespace, context: Any) -> int:
    from ...maps.packs import verify_pack_dir
    from ...maps.receipts import MapReceiptStore

    target = str(args.pack)
    candidate = Path(target).expanduser()
    pack_dir = candidate if candidate.is_dir() \
        else _packs_dir(args, context) / target
    home = str(Path(_home_of(context)).expanduser())
    store = MapReceiptStore(home=home)
    try:
        verification, _receipt = verify_pack_dir(pack_dir, store=store)
    except Exception as exc:  # noqa: BLE001 - CLI last line of defence
        print("maps verify: " + type(exc).__name__ + ": " + str(exc),
              file=sys.stderr)
        return 1
    _emit(args, verification.to_dict(), verification.summary()
          + " · attribution: " + verification.attribution)
    return 0 if verification.ok else 1


def _sources(args: argparse.Namespace, context: Any) -> int:
    from ...maps.definitions import SOURCES

    rows: list[dict[str, Any]] = []
    lines = ["map sources — registered inputs + hash discipline"]
    for source in SOURCES:
        # A row with notes is a live service, not a downloadable
        # input (census B9 geocoder rungs): there is no artifact
        # to pin, so the hash discipline does not apply — the
        # notes carry its honest limits instead.
        if source.notes:
            discipline = "live service — no artifact"
        elif source.sha256:
            discipline = "sha256 pinned"
        elif source.md5_url:
            discipline = "publisher md5 + pin-after-fetch"
        else:
            discipline = "pin-after-fetch"
        line = ("  " + source.source_id + " — " + source.name
                + " · " + discipline + " · " + source.license
                + " · credit: " + source.attribution)
        if source.notes:
            line += " · " + source.notes
        lines.append(line)
        rows.append({"source_id": source.source_id,
                     "name": source.name, "url": source.url,
                     "sha256": source.sha256,
                     "md5_url": source.md5_url,
                     "attribution": source.attribution,
                     "license": source.license,
                     "notes": source.notes})
    _emit(args, {"sources": rows}, "\n".join(lines))
    return 0


def _installed_shelf(packs_dir: Path):
    """The installed packs as ``(dir, manifest)`` pairs plus a
    list of named problems — the agent's discipline: a directory
    whose manifest will not load is carried as a named corrupt
    entry, never silently skipped."""
    from ...maps.manifest import load_manifest

    shelf: list = []
    problems: list[str] = []
    if packs_dir.is_dir():
        for entry in sorted(packs_dir.iterdir()):
            if not entry.is_dir():
                continue
            try:
                shelf.append((entry, load_manifest(entry)))
            except Exception as exc:  # noqa: BLE001 - named, not fatal
                problems.append(entry.name + ": " + str(exc))
    return shelf, problems


def _index(args: argparse.Namespace, context: Any) -> int:
    from ...maps.gazetteer import Gazetteer

    packs_dir = _packs_dir(args, context)
    shelf, problems = _installed_shelf(packs_dir)
    rows: list[dict[str, Any]] = [
        {"problem": p} for p in problems]
    lines = ["maps index — " + str(packs_dir)]
    for problem in problems:
        lines.append("  defective pack directory: " + problem)
    if not shelf:
        lines.append("no map packs installed — nothing to index "
                     "(nm maps packs shows the shelf)")
        _emit(args, {"packs": rows}, "\n".join(lines))
        return 1
    wanted = str(getattr(args, "pack", "") or "")
    if wanted:
        shelf = [(d, m) for d, m in shelf if m.pack_id == wanted]
        if not shelf:
            lines.append("no installed pack named " + repr(wanted))
            _emit(args, {"packs": rows}, "\n".join(lines))
            return 1
    from ...maps.receipts import MapReceiptStore

    home = str(Path(_home_of(context)).expanduser())
    gazetteer = Gazetteer(packs_dir, store=MapReceiptStore(home=home))
    if getattr(args, "status", False):
        for status in gazetteer.index_status(shelf):
            line = ("  " + status.pack_id + " — " + status.state
                    + " · " + str(status.entries) + " places")
            if status.state == "building":
                line += (" · " + str(status.members_done) + " of "
                         + str(status.members_total)
                         + " tiles scanned")
            lines.append(line)
            rows.append({"pack_id": status.pack_id,
                         "state": status.state,
                         "entries": status.entries,
                         "members_done": status.members_done,
                         "members_total": status.members_total})
        _emit(args, {"packs": rows}, "\n".join(lines))
        return 0
    last_pct: dict[str, int] = {}

    def progress(pack_id: str, done: int, total: int) -> None:
        pct = int(done * 100 / total) if total else 100
        if pct >= last_pct.get(pack_id, -1) + 5 or done >= total:
            last_pct[pack_id] = pct
            print("  " + pack_id + ": " + str(done) + " of "
                  + str(total) + " tiles (" + str(pct) + "%)",
                  file=sys.stderr)

    result = gazetteer.build(
        shelf, force=bool(getattr(args, "force", False)),
        progress=progress)
    for pack_result in result.packs:
        line = ("  " + pack_result.pack_id + " — "
                + str(pack_result.entries) + " places from "
                + str(pack_result.tiles_scanned) + " tiles")
        if pack_result.skipped:
            line += " · skipped (digest unchanged)"
        if pack_result.corrupt_tiles:
            line += (" · corrupt: "
                     + str(len(pack_result.corrupt_tiles)))
        if pack_result.note and not pack_result.skipped:
            line += " · " + pack_result.note
        lines.append(line)
        rows.append({"pack_id": pack_result.pack_id,
                     "entries": pack_result.entries,
                     "tiles_scanned": pack_result.tiles_scanned,
                     "corrupt_tiles": len(pack_result.corrupt_tiles),
                     "skipped": pack_result.skipped,
                     "note": pack_result.note})
    lines.append("total: " + str(result.total_entries)
                 + " named places indexed")
    _emit(args, {"packs": rows,
                 "total_entries": result.total_entries},
          "\n".join(lines))
    return 0


def _cmd_maps(args: argparse.Namespace, context: Any) -> int:
    """Route ``nm maps`` — build / packs / verify / sources / index."""
    action = getattr(args, "maps_action", "")
    try:
        if action == "build":
            return _build(args, context)
        if action == "packs":
            return _packs(args, context)
        if action == "verify":
            return _verify(args, context)
        if action == "sources":
            return _sources(args, context)
        if action == "index":
            return _index(args, context)
    except Exception as exc:  # noqa: BLE001 - CLI last line of defence
        print("maps: " + type(exc).__name__ + ": " + str(exc),
              file=sys.stderr)
        return 1
    print("unknown maps action: " + str(action), file=sys.stderr)
    return 2
