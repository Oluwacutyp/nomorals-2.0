"""``nm browse`` — the Wave K browser service: sessions, tabs, downloads, screenshots."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _service(args: Any, context: Any):
    """Build a BrowserService, restore persisted sessions, attach artifacts."""
    from ...browser import BrowserService

    svc = BrowserService()
    try:
        svc.restore()
    except Exception:  # noqa: BLE001 - first run has nothing to restore
        pass
    db = getattr(context, "db", None)
    if db is not None:
        try:
            from ...storage.artifacts import ArtifactStore
            from ...storage.blob import BlobStore

            db_path = getattr(db, "path", None)
            blob_dir = Path(db_path).parent / "blobs" if db_path else Path("data/blobs")
            svc.attach_store(ArtifactStore(db, BlobStore(db, blob_dir)))
        except Exception:  # noqa: BLE001 - artifacts are a bonus, not required
            pass
    return svc


def _session(svc: Any, args: Any):
    name = getattr(args, "session", "") or "cli"
    try:
        return svc.get_session(name)
    except Exception:  # noqa: BLE001 - unknown session -> open a fresh one
        return svc.open_session(name)


def _active_tab(handle: Any):
    try:
        return handle.active_tab
    except Exception as exc:  # noqa: BLE001
        from ...browser import BrowserError

        raise BrowserError("no tabs open — `nm browse open <url>` first") from exc


def _cmd_browse(args: Any, context: Any) -> int:
    """Route ``nm browse <open|tabs|text|md|links|shot|download|history|close|sessions>``."""
    from ...browser import BrowserError

    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm browse open <url> [--session S]\n"
              "       nm browse tabs [--session S] | nm browse text|md|links\n"
              "       nm browse shot [--out PATH] | nm browse download <url>\n"
              "       nm browse history | nm browse close | nm browse sessions",
              file=sys.stderr)
        return 2
    verb = words[0]
    try:
        if verb == "open":
            return _browse_open(args, context, words[1:])
        if verb == "tabs":
            return _browse_tabs(args, context)
        if verb == "text":
            return _browse_read(args, context, "text")
        if verb == "md":
            return _browse_read(args, context, "markdown")
        if verb == "links":
            return _browse_links(args, context)
        if verb == "shot":
            return _browse_shot(args, context)
        if verb == "download":
            return _browse_download(args, context, words[1:])
        if verb == "history":
            return _browse_history(args, context)
        if verb == "close":
            return _browse_close(args, context)
        if verb == "sessions":
            return _browse_sessions(args, context)
        print(f"unknown browse verb: {verb}", file=sys.stderr)
        return 2
    except BrowserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _browse_open(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm browse open <url> [--session S]", file=sys.stderr)
        return 2
    svc = _service(args, context)
    handle = _session(svc, args)
    tab = handle.open_tab(rest[0])
    svc.save()
    if getattr(args, "json", False):
        print(json.dumps(tab.to_dict(), indent=2, default=str))
    else:
        print(f"tab {tab.tab_id}: {tab.title or '(loading)'} — {tab.url}")
    return 0


def _browse_tabs(args: Any, context: Any) -> int:
    svc = _service(args, context)
    handle = _session(svc, args)
    tabs = handle.list_tabs()
    if getattr(args, "json", False):
        print(json.dumps(tabs, indent=2))
        return 0
    if not tabs:
        print("no tabs open")
        return 0
    for t in tabs:
        print(f"{t['tab_id']}  {t.get('title') or '(untitled)'}  {t.get('url')}")
    return 0


def _browse_read(args: Any, context: Any, kind: str) -> int:
    svc = _service(args, context)
    tab = _active_tab(_session(svc, args))
    data = tab.text() if kind == "text" else tab.markdown()
    print(data.get("text") or data.get("markdown") or "")
    return 0


def _browse_links(args: Any, context: Any) -> int:
    svc = _service(args, context)
    tab = _active_tab(_session(svc, args))
    links = tab.links().get("links") or []
    if getattr(args, "json", False):
        print(json.dumps(links, indent=2))
        return 0
    for link in links[:100]:
        print(f"{link.get('text', '')[:60]:60}  {link.get('href', '')}")
    return 0


def _browse_shot(args: Any, context: Any) -> int:
    svc = _service(args, context)
    tab = _active_tab(_session(svc, args))
    result = svc.screenshot(tab)
    out = getattr(args, "out", "") or ""
    if out:
        Path(out).expanduser().write_bytes(Path(result.path).read_bytes())
        print(f"screenshot -> {out}")
    else:
        print(f"screenshot -> {result.path}")
    if result.artifact_uri:
        print(f"artifact: {result.artifact_uri}")
    return 0


def _browse_download(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm browse download <url>", file=sys.stderr)
        return 2
    svc = _service(args, context)
    handle = _session(svc, args)
    try:
        tab = handle.active_tab
    except Exception:  # noqa: BLE001 - no tab: download bare, no cookies
        tab = None
    result = svc.download(tab if tab is not None else rest[0],
                          rest[0] if tab is not None else "")
    if getattr(args, "json", False):
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"downloaded {result.size} bytes -> {result.path}")
        if result.artifact_uri:
            print(f"artifact: {result.artifact_uri}")
    return 0


def _browse_history(args: Any, context: Any) -> int:
    svc = _service(args, context)
    history = _session(svc, args).history()
    if getattr(args, "json", False):
        print(json.dumps(history, indent=2, default=str))
        return 0
    for entry in history:
        print(f"{entry.get('ts', 0):.0f}  {entry.get('title') or ''}  {entry.get('url')}")
    if not history:
        print("no history yet")
    return 0


def _browse_close(args: Any, context: Any) -> int:
    svc = _service(args, context)
    name = getattr(args, "session", "") or "cli"
    svc.close_session(name)
    print(f"session {name!r} closed")
    return 0


def _browse_sessions(args: Any, context: Any) -> int:
    svc = _service(args, context)
    names = svc.list_sessions()
    if getattr(args, "json", False):
        print(json.dumps(names, indent=2))
        return 0
    for name in names:
        print(name)
    if not names:
        print("no sessions")
    return 0
