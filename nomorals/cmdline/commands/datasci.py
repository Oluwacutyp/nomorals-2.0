"""``nm datasci`` — data-science workspace: load, describe, query, plot.

Datasets persist as parquet in the workspace dir, so separate invocations
share them: ``nm datasci load data.csv --name sales`` then
``nm datasci describe sales``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _workspace(context: Any) -> Any:
    from ...datasci import DataWorkspace
    settings = getattr(context, "settings", None)
    root = getattr(settings, "workspace_dir", None) if settings else None
    base = Path(root) if root else Path.cwd() / "workspace"
    return DataWorkspace(base / "datasci")


def _artifacts(context: Any) -> Any:
    from ...storage.artifacts import ArtifactStore
    from ...storage.blob import BlobStore
    settings = getattr(context, "settings", None)
    root = getattr(settings, "workspace_dir", None) if settings else None
    base = Path(root) if root else Path.cwd() / "workspace"
    db = context.db
    return ArtifactStore(db, BlobStore(db, str(base / "blobs")))


def _cmd_datasci(args: Any, context: Any) -> int:
    """Route ``nm datasci <verb>``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm datasci load <file> [--name N] [--overwrite]\n"
              "       nm datasci list [--json]\n"
              "       nm datasci describe <name> [--json]\n"
              "       nm datasci head <name> [--rows N] [--json]\n"
              "       nm datasci query <name> \"<pandas expr>\" [--as NEW] [--json]\n"
              "       nm datasci plot <name> --kind line|bar|scatter|hist\n"
              "                      --x COL [--y COL] [--title T]\n"
              "       nm datasci drop <name>",
              file=sys.stderr)
        return 2
    verb, rest = words[0], words[1:]
    if verb == "load":
        return _ds_load(args, context, rest)
    if verb == "list":
        return _ds_list(args, context)
    if verb == "describe":
        return _ds_describe(args, context, rest)
    if verb == "head":
        return _ds_head(args, context, rest)
    if verb == "query":
        return _ds_query(args, context, rest)
    if verb == "plot":
        return _ds_plot(args, context, rest)
    if verb == "drop":
        return _ds_drop(args, context, rest)
    print(f"unknown datasci verb: {verb}", file=sys.stderr)
    return 2


def _as_json(args: Any) -> bool:
    return bool(getattr(args, "json", False))


def _out(obj: Any, args: Any) -> int:
    if _as_json(args):
        print(json.dumps(obj, indent=2, default=str))
    else:
        print(json.dumps(obj, indent=2, default=str))
    return 0


def _ds_load(args: Any, context: Any, rest: list[str]) -> int:
    from ...datasci import DataSciError
    if not rest:
        print("usage: nm datasci load <file> [--name N]", file=sys.stderr)
        return 2
    path = rest[0]
    name = getattr(args, "name", "") or Path(path).stem
    try:
        ds = _workspace(context).load(
            name, path, overwrite=bool(getattr(args, "overwrite", False)))
    except DataSciError as exc:
        print(f"load failed: {exc}", file=sys.stderr)
        return 1
    print(f"loaded {ds.name}: {ds.rows} rows x {len(ds.columns)} cols")
    print(f"  columns: {', '.join(ds.columns[:12])}"
          + (" ..." if len(ds.columns) > 12 else ""))
    print(f"  source: {ds.source}  sha256: {ds.sha256[:12]}")
    return 0


def _ds_list(args: Any, context: Any) -> int:
    sets = _workspace(context).list_datasets()
    if _as_json(args):
        return _out(sets, args)
    if not sets:
        print("no datasets loaded. `nm datasci load <file> [--name N]`")
        return 0
    for s in sets:
        print(f"  {s['name']:20s} {s['rows']:>8d} rows  "
              f"{len(s['columns'])} cols  {s['source']}")
    return 0


def _ds_describe(args: Any, context: Any, rest: list[str]) -> int:
    from ...datasci import DataSciError
    if not rest:
        print("usage: nm datasci describe <name>", file=sys.stderr)
        return 2
    try:
        info = _workspace(context).describe(rest[0])
    except DataSciError as exc:
        print(f"describe failed: {exc}", file=sys.stderr)
        return 1
    if _as_json(args):
        return _out(info, args)
    print(f"{info['name']}: {info['rows']} rows, {len(info['columns'])} cols")
    print("dtypes:")
    for c, t in info["dtypes"].items():
        nulls = info["nulls"].get(c, 0)
        print(f"  {c:24s} {t:12s} nulls={nulls}")
    print("summary (describe):")
    for col, stats in info["summary"].items():
        line = "  " + col + ": " + ", ".join(
            f"{k}={v}" for k, v in stats.items() if v is not None)
        print(line[:120])
    return 0


def _ds_head(args: Any, context: Any, rest: list[str]) -> int:
    from ...datasci import DataSciError
    if not rest:
        print("usage: nm datasci head <name> [--rows N]", file=sys.stderr)
        return 2
    n = int(getattr(args, "rows", 0) or 5)
    try:
        rows = _workspace(context).head(rest[0], n)
    except DataSciError as exc:
        print(f"head failed: {exc}", file=sys.stderr)
        return 1
    return _out(rows, args)


def _ds_query(args: Any, context: Any, rest: list[str]) -> int:
    from ...datasci import DataSciError
    if len(rest) < 2:
        print("usage: nm datasci query <name> \"<pandas expr>\" [--as NEW]",
              file=sys.stderr)
        return 2
    name, expr = rest[0], rest[1]
    as_name = getattr(args, "as_name", "") or None
    try:
        out = _workspace(context).query(name, expr, as_name=as_name)
    except DataSciError as exc:
        print(f"query failed: {exc}", file=sys.stderr)
        return 1
    if as_name:
        print(f"saved {out.rows} rows as dataset {as_name!r}")
        return 0
    return _out(out.head(20).to_dict(orient="records"), args)


def _ds_plot(args: Any, context: Any, rest: list[str]) -> int:
    from ...datasci import DataSciError
    if not rest:
        print("usage: nm datasci plot <name> --kind KIND --x COL [--y COL]",
              file=sys.stderr)
        return 2
    name = rest[0]
    kind = getattr(args, "kind", "") or "line"
    x = getattr(args, "x", "") or ""
    y = getattr(args, "y", "") or ""
    title = getattr(args, "title", "") or f"{name}: {kind}"
    try:
        from ...datasci import render_plot
        ws = _workspace(context)
        ds = ws.get(name)
        png = render_plot(ds.frame, kind, x=x, y=y, title=title)
        art = _artifacts(context).put(
            png, type="chart", mime="image/png", creator="datasci",
            metadata={"dataset": name, "kind": kind, "x": x, "y": y},
            provenance={"dataset": ds.name, "source": ds.source,
                        "sha256": ds.sha256, "kind": kind, "x": x, "y": y})
    except DataSciError as exc:
        print(f"plot failed: {exc}", file=sys.stderr)
        return 1
    print(f"chart artifact: {art.id} ({len(png)} bytes)")
    return 0


def _ds_drop(args: Any, context: Any, rest: list[str]) -> int:
    from ...datasci import DataSciError
    if not rest:
        print("usage: nm datasci drop <name>", file=sys.stderr)
        return 2
    try:
        _workspace(context).drop(rest[0])
    except DataSciError as exc:
        print(f"drop failed: {exc}", file=sys.stderr)
        return 1
    print(f"dropped {rest[0]}")
    return 0
