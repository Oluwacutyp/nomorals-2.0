"""``nm browse`` — the Wave K browser service: sessions, tabs, downloads, screenshots.

Two flows, one command surface:

* **default** (daemon not running): the classic restore→act→save per
  invocation — ``_LocalBackend``;
* **daemon mode** (``nm browse daemon start`` was run): every verb is served
  by the persistent daemon over its socket — ``_DaemonBackend`` — so
  sessions/tabs/cookies stay warm with no restore churn. Events the daemon
  emitted are re-published on this process's bus so the timeline records
  them exactly as in the default flow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _service(args: Any, context: Any):
    """Build a BrowserService, restore persisted sessions, attach artifacts."""
    from ...browser import BrowserService
    from ...browser.daemon import default_data_dir

    svc = BrowserService(data_dir=default_data_dir())
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


# ── backends ─────────────────────────────────────────────────────────────────


class _LocalBackend:
    """The default restore→act→save flow (used when no daemon is running)."""

    def __init__(self, args: Any, context: Any) -> None:
        self._svc = _service(args, context)
        self.session_name = getattr(args, "session", "") or "cli"

    def _handle(self):
        try:
            return self._svc.get_session(self.session_name)
        except Exception:  # noqa: BLE001 - unknown session -> open a fresh one
            return self._svc.open_session(self.session_name)

    def open(self, url: str) -> dict[str, Any]:
        tab = self._handle().open_tab(url)
        self._svc.save()
        return tab.to_dict()

    def tabs(self) -> list[dict[str, str]]:
        return self._handle().list_tabs()

    def read(self, kind: str) -> dict[str, Any]:
        tab = _active_tab(self._handle())
        return tab.text() if kind == "text" else tab.markdown()

    def links(self) -> list[dict[str, Any]]:
        return _active_tab(self._handle()).links().get("links") or []

    def shot(self) -> dict[str, Any]:
        return self._svc.screenshot(_active_tab(self._handle())).to_dict()

    def download(self, url: str) -> dict[str, Any]:
        handle = self._handle()
        try:
            tab = handle.active_tab
        except Exception:  # noqa: BLE001 - no tab: download bare, no cookies
            tab = None
        result = self._svc.download(tab if tab is not None else url,
                                    url if tab is not None else "")
        return result.to_dict()

    def history(self) -> list[dict[str, Any]]:
        return self._handle().history()

    def close(self) -> str:
        self._svc.close_session(self.session_name)
        return self.session_name

    def sessions(self) -> list[str]:
        return self._svc.list_sessions()


class _DaemonBackend:
    """Persistent-daemon flow: every op is a socket round trip, no restore."""

    def __init__(self, args: Any, client: Any) -> None:
        from ...browser.daemon import republish_events

        self._client = client
        self._republish = republish_events
        self.session_name = getattr(args, "session", "") or "cli"

    def _call(self, op: str, **params: Any) -> Any:
        result, events = self._client.call(
            op, {"session": self.session_name, **params})
        self._republish(events)
        return result

    def open(self, url: str) -> dict[str, Any]:
        return self._call("open", url=url)["tab"]

    def tabs(self) -> list[dict[str, str]]:
        return self._call("tabs")["tabs"]

    def read(self, kind: str) -> dict[str, Any]:
        return self._call("read", kind=kind)["data"]

    def links(self) -> list[dict[str, Any]]:
        return self._call("links")["links"]

    def shot(self) -> dict[str, Any]:
        return self._call("shot")["result"]

    def download(self, url: str) -> dict[str, Any]:
        return self._call("download", url=url)["result"]

    def history(self) -> list[dict[str, Any]]:
        return self._call("history")["history"]

    def close(self) -> str:
        return self._call("close")["closed"]

    def sessions(self) -> list[str]:
        result, events = self._client.call("sessions", {})
        self._republish(events)
        return result["sessions"]


def _backend(args: Any, context: Any):
    """Daemon when it's up (opt-in via ``daemon start``), else the default."""
    from ...browser.daemon import DaemonControl

    ctl = DaemonControl()
    if ctl.running():
        return _DaemonBackend(args, ctl.client())
    return _LocalBackend(args, context)


# ── verbs ────────────────────────────────────────────────────────────────────


def _cmd_browse(args: Any, context: Any) -> int:
    """Route ``nm browse <open|tabs|text|md|links|shot|download|history|close|sessions|daemon>``."""
    from ...browser import BrowserError, DaemonError

    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm browse open <url> [--session S]\n"
              "       nm browse tabs [--session S] | nm browse text|md|links\n"
              "       nm browse shot [--out PATH] | nm browse download <url>\n"
              "       nm browse history | nm browse close | nm browse sessions\n"
              "       nm browse daemon start|stop|status",
              file=sys.stderr)
        return 2
    verb = words[0]
    try:
        if verb == "daemon":
            return _browse_daemon(args, words[1:])
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
    except (BrowserError, DaemonError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _browse_daemon(args: Any, rest: list[str]) -> int:
    from ...browser.daemon import DaemonControl, DaemonError

    ctl = DaemonControl()
    sub = (rest[0] if rest else "").strip().lower()
    as_json = getattr(args, "json", False)
    if sub == "start":
        try:
            info = ctl.start()
        except DaemonError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if as_json:
            print(json.dumps(info, indent=2, default=str))
        elif info.get("already"):
            print(f"browser daemon already running (pid {info.get('pid')})")
        else:
            print(f"browser daemon started (pid {info.get('pid')})")
        return 0
    if sub == "stop":
        try:
            info = ctl.stop()
        except DaemonError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if as_json:
            print(json.dumps(info, indent=2, default=str))
        elif info.get("stopped"):
            extra = " (forced)" if info.get("forced") else ""
            print(f"browser daemon stopped (pid {info.get('pid')}){extra}")
        else:
            print("browser daemon is not running")
        return 0
    if sub == "status":
        info = ctl.status()
        if as_json:
            print(json.dumps(info, indent=2, default=str))
            return 0
        if not info.get("running"):
            stale = info.get("stale_pid")
            print(f"browser daemon not running"
                  + (f" (stale pid file: {stale})" if stale else ""))
        elif not info.get("responsive", True):
            print(f"browser daemon pid {info.get('pid')}: NOT RESPONDING "
                  f"(`nm browse daemon stop` to reset)")
        else:
            uptime = info.get("uptime_s", 0)
            sessions = info.get("sessions") or []
            print(f"browser daemon running (pid {info.get('pid')}, "
                  f"up {uptime:.0f}s, {len(sessions)} session(s), "
                  f"{info.get('tabs', 0)} tab(s))")
            for name in sessions:
                print(f"  session: {name}")
        return 0
    print("usage: nm browse daemon start|stop|status [--json]", file=sys.stderr)
    return 2


def _browse_open(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm browse open <url> [--session S]", file=sys.stderr)
        return 2
    backend = _backend(args, context)
    tab = backend.open(rest[0])
    if getattr(args, "json", False):
        print(json.dumps(tab, indent=2, default=str))
    else:
        print(f"tab {tab['tab_id']}: {tab.get('title') or '(loading)'} — {tab.get('url')}")
    return 0


def _browse_tabs(args: Any, context: Any) -> int:
    backend = _backend(args, context)
    tabs = backend.tabs()
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
    backend = _backend(args, context)
    data = backend.read(kind)
    print(data.get("text") or data.get("markdown") or "")
    return 0


def _browse_links(args: Any, context: Any) -> int:
    backend = _backend(args, context)
    links = backend.links()
    if getattr(args, "json", False):
        print(json.dumps(links, indent=2))
        return 0
    for link in links[:100]:
        print(f"{link.get('text', '')[:60]:60}  {link.get('href', '')}")
    return 0


def _browse_shot(args: Any, context: Any) -> int:
    backend = _backend(args, context)
    result = backend.shot()
    out = getattr(args, "out", "") or ""
    if out:
        Path(out).expanduser().write_bytes(Path(result["path"]).read_bytes())
        print(f"screenshot -> {out}")
    else:
        print(f"screenshot -> {result['path']}")
    if result.get("artifact_uri"):
        print(f"artifact: {result['artifact_uri']}")
    return 0


def _browse_download(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm browse download <url>", file=sys.stderr)
        return 2
    backend = _backend(args, context)
    result = backend.download(rest[0])
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
    else:
        print(f"downloaded {result['size']} bytes -> {result['path']}")
        if result.get("artifact_uri"):
            print(f"artifact: {result['artifact_uri']}")
    return 0


def _browse_history(args: Any, context: Any) -> int:
    backend = _backend(args, context)
    history = backend.history()
    if getattr(args, "json", False):
        print(json.dumps(history, indent=2, default=str))
        return 0
    for entry in history:
        print(f"{entry.get('ts', 0):.0f}  {entry.get('title') or ''}  {entry.get('url')}")
    if not history:
        print("no history yet")
    return 0


def _browse_close(args: Any, context: Any) -> int:
    backend = _backend(args, context)
    name = backend.close()
    print(f"session {name!r} closed")
    return 0


def _browse_sessions(args: Any, context: Any) -> int:
    backend = _backend(args, context)
    names = backend.sessions()
    if getattr(args, "json", False):
        print(json.dumps(names, indent=2))
        return 0
    for name in names:
        print(name)
    if not names:
        print("no sessions")
    return 0
