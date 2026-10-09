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

    def download(self, url: str, organize: bool = False) -> dict[str, Any]:
        handle = self._handle()
        try:
            tab = handle.active_tab
        except Exception:  # noqa: BLE001 - no tab: download bare, no cookies
            tab = None
        result = self._svc.download(tab if tab is not None else url,
                                    url if tab is not None else "",
                                    organize=organize)
        return result.to_dict()

    def fill(self, name: str, value: str) -> dict[str, Any]:
        return _active_tab(self._handle()).fill(name, value)

    def select(self, name: str, value: str) -> dict[str, Any]:
        return _active_tab(self._handle()).select(name, value)

    def check(self, name: str, checked: bool = True) -> dict[str, Any]:
        return _active_tab(self._handle()).check(name, checked)

    def captcha_check(self) -> dict[str, Any]:
        return _active_tab(self._handle()).check_captcha()

    def click(self, target: str) -> dict[str, Any]:
        tab = _active_tab(self._handle())
        result = tab.click(target)
        self._svc.save()
        return result

    def submit(self, target: str,
               uploads: dict[str, str] | None = None) -> dict[str, Any]:
        tab = _active_tab(self._handle())
        result = tab.submit(target, uploads=uploads)
        self._svc.save()
        return result

    def extract(self, target: str, kind: str = "") -> dict[str, Any]:
        return _active_tab(self._handle()).extract(target, kind=kind)

    def task(self, steps: Any) -> dict[str, Any]:
        return _active_tab(self._handle()).task(steps)

    def cookies_list(self) -> dict[str, Any]:
        return {"cookies": _active_tab(self._handle()).cookies()}

    def cookies_export(self, path: str, format: str) -> dict[str, Any]:
        return self._svc.export_cookies(self.session_name, path, format=format)

    def cookies_import(self, path: str, format: str) -> dict[str, Any]:
        return self._svc.import_cookies(self.session_name, path, format=format)

    def proxy_rotate(self) -> dict[str, Any]:
        return self._svc.rotate_proxy(self.session_name)

    def proxy_set(self, proxy_url: str) -> dict[str, Any]:
        return self._svc.set_session_proxy(self.session_name, proxy_url)

    def proxy_clear(self) -> dict[str, Any]:
        return self._svc.set_session_proxy(self.session_name, "")

    def proxy_status(self) -> dict[str, Any]:
        from ...browser.service import _mask_proxy

        proxy = self._svc.session_proxy(self.session_name)
        return {"attached": self._svc.proxy_pool_attached(),
                "proxy": _mask_proxy(proxy) if proxy else "direct"}

    def downloads(self, category: str = "", limit: int = 100) -> list[dict[str, Any]]:
        return self._svc.list_downloads(self.session_name, category=category,
                                        limit=limit)

    # -- rendered tabs -------------------------------------------------------
    def _rtab(self, tab_id: str):
        from ...browser import BrowserError

        tab = self._svc.find_rendered_tab(tab_id)
        if tab is None:
            raise BrowserError(f"unknown rendered tab {tab_id!r}")
        return tab

    def r_open(self, url: str, proxy: str = "") -> dict[str, Any]:
        tab = self._svc.open_rendered_tab(self.session_name, url, proxy=proxy)
        return {"tab": tab.to_dict(), "tab_id": tab.tab_id}

    def r_tabs(self) -> list[dict[str, str]]:
        return self._svc.list_rendered_tabs()

    def r_close(self, tab_id: str) -> str:
        self._svc.close_rendered_tab(tab_id)
        return tab_id

    def r_shot(self, tab_id: str) -> dict[str, Any]:
        return self._svc.screenshot_rendered(tab_id).to_dict()

    def r_fill(self, tab_id: str, name: str, value: str) -> dict[str, Any]:
        return self._rtab(tab_id).fill(name, value)

    def r_fill_form(self, tab_id: str,
                    fields: dict[str, str]) -> dict[str, Any]:
        return self._rtab(tab_id).fill_form(fields)

    def r_set_date(self, tab_id: str, name: str,
                   value: str) -> dict[str, Any]:
        return self._rtab(tab_id).set_date(name, value)

    def r_fields(self, tab_id: str) -> dict[str, Any]:
        return self._rtab(tab_id).describe_fields()

    def r_evaluate(self, tab_id: str, js: str,
                   arg: Any = None) -> Any:
        if arg is None:
            return self._rtab(tab_id).evaluate(js)
        return self._rtab(tab_id).evaluate(js, arg)

    def r_storage(self, tab_id: str, action: str = "get", key: str = "",
                  value: Any = None) -> dict[str, Any]:
        return self._rtab(tab_id).local_storage(action, key, value)

    def r_persist(self, tab_id: str) -> dict[str, Any]:
        return self._rtab(tab_id).persist()

    def r_click(self, tab_id: str, target: str) -> dict[str, Any]:
        return self._rtab(tab_id).click(target)

    def r_submit(self, tab_id: str, target: str = "") -> dict[str, Any]:
        return self._rtab(tab_id).submit(target)

    def r_wait(self, tab_id: str, selector: str) -> dict[str, Any]:
        return self._rtab(tab_id).wait_for(selector)

    def r_wait_state(self, tab_id: str, state: str = "load",
                     timeout: int = 30_000) -> dict[str, Any]:
        return self._rtab(tab_id).wait_for_load_state(state,
                                                      timeout=timeout)

    def r_wait_idle(self, tab_id: str,
                    timeout: int = 15_000) -> dict[str, Any]:
        return self._rtab(tab_id).wait_for_network_idle(timeout=timeout)

    def r_wait_url(self, tab_id: str, pattern: str) -> dict[str, Any]:
        return self._rtab(tab_id).wait_for_url(pattern)

    def r_wait_text(self, tab_id: str, text: str) -> dict[str, Any]:
        return self._rtab(tab_id).wait_for_text(text)

    def r_select(self, tab_id: str, name: str, value: str) -> dict[str, Any]:
        return self._rtab(tab_id).select(name, value)

    def r_check(self, tab_id: str, name: str,
                checked: bool = True) -> dict[str, Any]:
        return self._rtab(tab_id).check(name, checked)

    def r_captcha(self, tab_id: str) -> dict[str, Any]:
        return self._rtab(tab_id).check_captcha()

    def r_extract(self, tab_id: str, target: str = "",
                  kind: str = "") -> dict[str, Any]:
        return self._rtab(tab_id).extract(target, kind=kind)

    def r_download(self, tab_id: str, target: str) -> dict[str, Any]:
        return self._svc.rendered_tab_download(tab_id, target)

    def r_upload(self, tab_id: str, selector: str,
                 path: str) -> dict[str, Any]:
        return self._rtab(tab_id).upload(selector, path)

    def r_wait_field(self, tab_id: str, name: str,
                     timeout: int = 10_000) -> dict[str, Any]:
        return self._rtab(tab_id).wait_for_field(name, timeout=timeout)

    def r_switch(self, session: str, tab_id: str) -> dict[str, Any]:
        return self._svc.switch_rendered_tab(session, tab_id)

    def r_active(self, session: str) -> dict[str, Any] | None:
        return self._svc.active_rendered_tab(session)

    def pacing(self, action: str = "get",
               **kwargs: Any) -> dict[str, Any]:
        if action == "get":
            return self._svc.pacing.describe()
        preset = str(kwargs.pop("preset", "") or "").strip().lower()
        if preset:
            from ...browser.pacing import Pacing
            if preset == "human":
                return self._svc.set_pacing(Pacing.human())
            if preset == "off":
                return self._svc.set_pacing(Pacing.disabled())
            if preset == "profile":
                return self._svc.set_pacing(Pacing.from_profile())
            raise ValueError(
                f"unknown pacing preset {preset!r} (want human|off|profile)")
        return self._svc.set_pacing(**kwargs)

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

    def download(self, url: str, organize: bool = False) -> dict[str, Any]:
        return self._call("download", url=url, organize=organize)["result"]

    def fill(self, name: str, value: str) -> dict[str, Any]:
        return self._call("fill", name=name, value=value)["result"]

    def select(self, name: str, value: str) -> dict[str, Any]:
        return self._call("select", name=name, value=value)["result"]

    def check(self, name: str, checked: bool = True) -> dict[str, Any]:
        return self._call("check", name=name, checked=checked)["result"]

    def captcha_check(self) -> dict[str, Any]:
        return self._call("captcha_check")["result"]

    def click(self, target: str) -> dict[str, Any]:
        return self._call("click", target=target)["result"]

    def submit(self, target: str,
               uploads: dict[str, str] | None = None) -> dict[str, Any]:
        return self._call("submit", target=target,
                           uploads=uploads)["result"]

    def extract(self, target: str, kind: str = "") -> dict[str, Any]:
        return self._call("extract", target=target, kind=kind)["result"]

    def task(self, steps: Any) -> dict[str, Any]:
        return self._call("task", steps=steps)["result"]

    def cookies_list(self) -> dict[str, Any]:
        return self._call("cookies")

    def cookies_export(self, path: str, format: str) -> dict[str, Any]:
        return self._call("cookies_export", path=path, format=format)["result"]

    def cookies_import(self, path: str, format: str) -> dict[str, Any]:
        return self._call("cookies_import", path=path, format=format)["result"]

    def proxy_rotate(self) -> dict[str, Any]:
        return self._call("proxy", action="rotate")["result"]

    def proxy_set(self, proxy_url: str) -> dict[str, Any]:
        return self._call("proxy", action="set", proxy_url=proxy_url)["result"]

    def proxy_clear(self) -> dict[str, Any]:
        return self._call("proxy", action="clear")["result"]

    def proxy_status(self) -> dict[str, Any]:
        return self._call("proxy", action="status")

    def downloads(self, category: str = "",
                  limit: int = 100) -> list[dict[str, Any]]:
        return self._call("downloads", category=category,
                           limit=limit)["downloads"]

    # -- rendered tabs -------------------------------------------------------
    def r_open(self, url: str, proxy: str = "") -> dict[str, Any]:
        return self._call("r_open", url=url, proxy=proxy)

    def r_tabs(self) -> list[dict[str, str]]:
        return self._call("r_tabs")["tabs"]

    def r_close(self, tab_id: str) -> str:
        return self._call("r_close", tab_id=tab_id)["closed"]

    def r_shot(self, tab_id: str) -> dict[str, Any]:
        return self._call("r_shot", tab_id=tab_id)["result"]

    def r_fill(self, tab_id: str, name: str, value: str) -> dict[str, Any]:
        return self._call("r_fill", tab_id=tab_id, name=name,
                           value=value)["result"]

    def r_fill_form(self, tab_id: str,
                    fields: dict[str, str]) -> dict[str, Any]:
        return self._call("r_fill_form", tab_id=tab_id,
                           fields=fields)["result"]

    def r_set_date(self, tab_id: str, name: str,
                   value: str) -> dict[str, Any]:
        return self._call("r_set_date", tab_id=tab_id, name=name,
                           value=value)["result"]

    def r_fields(self, tab_id: str) -> dict[str, Any]:
        return self._call("r_fields", tab_id=tab_id)["result"]

    def r_evaluate(self, tab_id: str, js: str,
                   arg: Any = None) -> Any:
        params: dict[str, Any] = {"tab_id": tab_id, "js": js}
        if arg is not None:
            params["arg"] = arg
        return self._call("r_evaluate", **params)["result"]

    def r_storage(self, tab_id: str, action: str = "get", key: str = "",
                  value: Any = None) -> dict[str, Any]:
        return self._call("r_storage", tab_id=tab_id, action=action,
                           key=key, value=value)["result"]

    def r_persist(self, tab_id: str) -> dict[str, Any]:
        return self._call("r_persist", tab_id=tab_id)["result"]

    def r_click(self, tab_id: str, target: str) -> dict[str, Any]:
        return self._call("r_click", tab_id=tab_id, target=target)["result"]

    def r_submit(self, tab_id: str, target: str = "") -> dict[str, Any]:
        return self._call("r_submit", tab_id=tab_id,
                           target=target)["result"]

    def r_wait(self, tab_id: str, selector: str) -> dict[str, Any]:
        return self._call("r_wait", tab_id=tab_id,
                           selector=selector)["result"]

    def r_wait_state(self, tab_id: str, state: str = "load",
                     timeout: int = 30_000) -> dict[str, Any]:
        return self._call("r_wait_state", tab_id=tab_id, state=state,
                           timeout=timeout)["result"]

    def r_wait_idle(self, tab_id: str,
                    timeout: int = 15_000) -> dict[str, Any]:
        return self._call("r_wait_idle", tab_id=tab_id,
                           timeout=timeout)["result"]

    def r_wait_url(self, tab_id: str, pattern: str) -> dict[str, Any]:
        return self._call("r_wait_url", tab_id=tab_id,
                           pattern=pattern)["result"]

    def r_wait_text(self, tab_id: str, text: str) -> dict[str, Any]:
        return self._call("r_wait_text", tab_id=tab_id,
                           text=text)["result"]

    def r_select(self, tab_id: str, name: str, value: str) -> dict[str, Any]:
        return self._call("r_select", tab_id=tab_id, name=name,
                           value=value)["result"]

    def r_check(self, tab_id: str, name: str,
                checked: bool = True) -> dict[str, Any]:
        return self._call("r_check", tab_id=tab_id, name=name,
                           checked=checked)["result"]

    def r_captcha(self, tab_id: str) -> dict[str, Any]:
        return self._call("r_captcha", tab_id=tab_id)["result"]

    def r_extract(self, tab_id: str, target: str = "",
                  kind: str = "") -> dict[str, Any]:
        return self._call("r_extract", tab_id=tab_id, target=target,
                           kind=kind)["result"]

    def r_download(self, tab_id: str, target: str) -> dict[str, Any]:
        return self._call("r_download", tab_id=tab_id,
                           target=target)["result"]

    def r_upload(self, tab_id: str, selector: str,
                 path: str) -> dict[str, Any]:
        return self._call("r_upload", tab_id=tab_id, selector=selector,
                           path=path)["result"]

    def r_wait_field(self, tab_id: str, name: str,
                     timeout: int = 10_000) -> dict[str, Any]:
        return self._call("r_wait_field", tab_id=tab_id, name=name,
                           timeout=timeout)["result"]

    def r_switch(self, session: str, tab_id: str) -> dict[str, Any]:
        return self._call("r_switch", session=session,
                           tab_id=tab_id)["result"]

    def r_active(self, session: str) -> dict[str, Any] | None:
        return self._call("r_active", session=session)["result"]

    def pacing(self, action: str = "get",
               **kwargs: Any) -> dict[str, Any]:
        params: dict[str, Any] = {"action": action}
        params.update(kwargs)
        return self._call("pacing", **params)["result"]

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
              "       nm browse fill <name> <value> | nm browse click <target>\n"
              "       nm browse select <name> <value> | nm browse check <name> [off]\n"
              "       nm browse captcha-check\n"
              "       nm browse submit [target] [--upload field=path ...]\n"
              "       nm browse extract [target] [--kind K]\n"
              "       nm browse task --steps-file PATH | --steps-json '[...]'\n"
              "       nm browse cookies [export|import <path> [--format F]]\n"
              "       nm browse proxy status|rotate|set <url>|clear\n"
              "       nm browse downloads [--category C] [--limit N]\n"
              "       nm browse rtab open <url> | rtab tabs|shot|fill|fill-form|set-date|fields|eval|storage|persist|click|submit|wait|wait-url|wait-text|wait-state|wait-idle|select|check|captcha|extract|download|upload|wait-field|switch|active\n"
              "       nm browse pacing [get] | nm browse pacing set preset=human|off|profile | enabled=true delay-ms=400 jitter-ms=300\n"
              "       nm browse shot [--out PATH] | nm browse download <url> [--organize]\n"
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
        if verb == "fill":
            return _browse_fill(args, context, words[1:])
        if verb == "select":
            return _browse_select(args, context, words[1:])
        if verb == "check":
            return _browse_check(args, context, words[1:])
        if verb == "captcha-check":
            return _browse_captcha_check(args, context)
        if verb == "click":
            return _browse_click(args, context, words[1:])
        if verb == "submit":
            return _browse_submit(args, context, words[1:])
        if verb == "extract":
            return _browse_extract(args, context, words[1:])
        if verb == "task":
            return _browse_task(args, context)
        if verb == "cookies":
            return _browse_cookies(args, context, words[1:])
        if verb == "proxy":
            return _browse_proxy(args, context, words[1:])
        if verb == "downloads":
            return _browse_downloads(args, context)
        if verb == "rtab":
            return _browse_rtab(args, context, words[1:])
        if verb == "pacing":
            return _browse_pacing(args, context, words[1:])
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
    result = backend.download(rest[0], organize=getattr(args, "organize", False))
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


# ── new verbs: forms, uploads, cookies, proxies, downloads, rendered tabs ────


def _browse_fill(args: Any, context: Any, rest: list[str]) -> int:
    if len(rest) < 2:
        print("usage: nm browse fill <name> <value> [--session S]",
              file=sys.stderr)
        return 2
    backend = _backend(args, context)
    result = backend.fill(rest[0], " ".join(rest[1:]))
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"filled {rest[0]!r}")
    return 0


def _browse_click(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm browse click <target> [--session S]", file=sys.stderr)
        return 2
    backend = _backend(args, context)
    result = backend.click(" ".join(rest))
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"{result.get('title') or '(untitled)'} — {result.get('url')}")
    return 0


def _browse_select(args: Any, context: Any, rest: list[str]) -> int:
    if len(rest) < 2:
        print("usage: nm browse select <name> <value> [--session S]",
              file=sys.stderr)
        return 2
    backend = _backend(args, context)
    result = backend.select(rest[0], " ".join(rest[1:]))
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"selected {result.get('picked')!r} in {rest[0]!r}")
    return 0


def _browse_check(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm browse check <name> [off] [--session S]",
              file=sys.stderr)
        return 2
    # trailing "off" token unchecks; anything else checks.
    checked = not (len(rest) > 1 and rest[-1].lower() == "off")
    name = rest[0]
    backend = _backend(args, context)
    result = backend.check(name, checked)
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"{'checked' if result.get('checked') else 'unchecked'} {name!r}")
    return 0


def _browse_captcha_check(args: Any, context: Any) -> int:
    backend = _backend(args, context)
    result = backend.captcha_check()
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))
    else:
        challenges = result.get("challenges") or []
        if not challenges:
            print(f"no captcha on {result.get('url') or 'this page'}")
        else:
            print(f"{len(challenges)} captcha(s) on {result.get('url')}:")
            for ch in challenges:
                print(f"  - {ch.get('kind')} "
                      f"(sitekey: {ch.get('sitekey') or 'n/a'}, "
                      f"domain: {ch.get('domain') or 'n/a'})")
    return 0


def _parse_uploads(args: Any) -> tuple[dict[str, str] | None, int]:
    """Parse repeatable --upload field=path flags. Returns (uploads, rc)."""
    uploads: dict[str, str] = {}
    for item in getattr(args, "upload", None) or []:
        if "=" not in item:
            print(f"error: --upload needs field=path, got {item!r}",
                  file=sys.stderr)
            return None, 2
        field, _, path = item.partition("=")
        field, path = field.strip(), path.strip()
        if not field or not path:
            print(f"error: --upload needs field=path, got {item!r}",
                  file=sys.stderr)
            return None, 2
        uploads[field] = path
    return uploads or None, 0


def _browse_submit(args: Any, context: Any, rest: list[str]) -> int:
    uploads, rc = _parse_uploads(args)
    if rc:
        return rc
    backend = _backend(args, context)
    result = backend.submit(rest[0] if rest else "", uploads)
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))
    else:
        up = result.get("uploaded") or []
        extra = f" (uploaded: {', '.join(up)})" if up else ""
        print(f"{result.get('title') or '(untitled)'} — {result.get('url')}{extra}")
    return 0


def _browse_extract(args: Any, context: Any, rest: list[str]) -> int:
    backend = _backend(args, context)
    kind = getattr(args, "kind", "") or ""
    result = backend.extract(" ".join(rest), kind)
    print(json.dumps(result, indent=2, default=str))
    return 0


def _browse_task(args: Any, context: Any) -> int:
    steps_file = getattr(args, "steps_file", "") or ""
    steps_json = getattr(args, "steps_json", "") or ""
    if steps_file:
        try:
            steps = json.loads(Path(steps_file).expanduser().read_text(
                encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"error: cannot read steps file: {exc}", file=sys.stderr)
            return 1
    elif steps_json:
        try:
            steps = json.loads(steps_json)
        except ValueError as exc:
            print(f"error: --steps-json is not valid JSON: {exc}",
                  file=sys.stderr)
            return 2
    else:
        print("usage: nm browse task --steps-file PATH | --steps-json '[...]'",
              file=sys.stderr)
        return 2
    backend = _backend(args, context)
    result = backend.task(steps)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


def _browse_cookies(args: Any, context: Any, rest: list[str]) -> int:
    sub = (rest[0] if rest else "list").strip().lower()
    fmt = getattr(args, "format", "") or "netscape"
    backend = _backend(args, context)
    if sub == "export":
        if len(rest) < 2:
            print("usage: nm browse cookies export <path> [--format netscape|json]",
                  file=sys.stderr)
            return 2
        result = backend.cookies_export(rest[1], fmt)
    elif sub == "import":
        if len(rest) < 2:
            print("usage: nm browse cookies import <path> [--format netscape|json]",
                  file=sys.stderr)
            return 2
        result = backend.cookies_import(rest[1], fmt)
    elif sub == "list":
        result = backend.cookies_list()
    else:
        print(f"unknown cookies verb: {sub} (list|export|import)",
              file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


def _browse_proxy(args: Any, context: Any, rest: list[str]) -> int:
    sub = (rest[0] if rest else "status").strip().lower()
    backend = _backend(args, context)
    if sub == "rotate":
        result = backend.proxy_rotate()
    elif sub == "set":
        if len(rest) < 2:
            print("usage: nm browse proxy set <proxy-url>", file=sys.stderr)
            return 2
        result = backend.proxy_set(rest[1])
    elif sub == "clear":
        result = backend.proxy_clear()
    elif sub == "status":
        result = backend.proxy_status()
    else:
        print(f"unknown proxy verb: {sub} (status|rotate|set|clear)",
              file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


def _browse_pacing(args: Any, context: Any, rest: list[str]) -> int:
    """``nm browse pacing`` — inspect/configure pacing between actions.

    ``nm browse pacing`` → show the active config.
    ``nm browse pacing set preset=human|off|profile``
    ``nm browse pacing set enabled=true delay-ms=400 jitter-ms=300``

    Pacing is an explicit owner setting (politeness/rate-limit control),
    not a default — it starts disabled.
    """
    sub = (rest[0] if rest else "get").strip().lower()
    backend = _backend(args, context)
    if sub in {"get", "show", "status"}:
        result = backend.pacing("get")
        print(json.dumps(result, indent=2, default=str))
        return 0
    if sub != "set":
        print("usage: nm browse pacing [get]\n"
              "       nm browse pacing set preset=human|off|profile\n"
              "       nm browse pacing set enabled=true delay-ms=400 "
              "jitter-ms=300", file=sys.stderr)
        return 2
    kwargs: dict[str, Any] = {}
    for token in rest[1:]:
        if "=" not in token:
            print(f"pacing: {token!r} is not key=value", file=sys.stderr)
            return 2
        key, val = token.split("=", 1)
        key = key.strip().lower().replace("-", "_")
        val = val.strip()
        if key == "enabled":
            if val.lower() not in {"true", "false", "1", "0", "on", "off"}:
                print(f"pacing: enabled needs true|false, got {val!r}",
                      file=sys.stderr)
                return 2
            kwargs["enabled"] = val.lower() in {"true", "1", "on"}
        elif key in {"delay_ms", "jitter_ms"}:
            if not val.isdigit():
                print(f"pacing: {key} needs milliseconds, got {val!r}",
                      file=sys.stderr)
                return 2
            kwargs[key] = int(val)
        elif key == "preset":
            kwargs["preset"] = val
        else:
            print(f"pacing: unknown key {key!r} "
                  "(enabled|delay-ms|jitter-ms|preset)", file=sys.stderr)
            return 2
    try:
        result = backend.pacing("set", **kwargs)
    except ValueError as exc:
        print(f"pacing: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


def _browse_downloads(args: Any, context: Any) -> int:
    backend = _backend(args, context)
    category = getattr(args, "category", "") or ""
    limit = getattr(args, "limit", 100) or 100
    recs = backend.downloads(category, limit)
    if getattr(args, "json", False):
        print(json.dumps(recs, indent=2, default=str))
        return 0
    for rec in recs:
        where = rec.get("path") or rec.get("url") or ""
        print(f"{str(rec.get('id'))[:8]}  {rec.get('status', ''):10}  "
              f"{rec.get('size', 0):>10}  {where}")
    if not recs:
        print("no downloads")
    return 0


def _browse_rtab(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm browse rtab open <url> [--proxy P] | rtab tabs\n"
              "       nm browse rtab close|shot <tab-id>\n"
              "       nm browse rtab fill <tab-id> <name> <value>\n"
              "       nm browse rtab fill-form <tab-id> name=value ...\n"
              "       nm browse rtab set-date <tab-id> <name> <date>\n"
              "       nm browse rtab fields <tab-id>\n"
              "       nm browse rtab eval <tab-id> <js>\n"
              "       nm browse rtab storage <tab-id> get|set|remove|clear <key> [value]\n"
              "       nm browse rtab persist <tab-id>\n"
              "       nm browse rtab select <tab-id> <name> <value>\n"
              "       nm browse rtab check <tab-id> <name> [off]\n"
              "       nm browse rtab captcha <tab-id>\n"
              "       nm browse rtab click|submit <tab-id> <target>\n"
              "       nm browse rtab wait <tab-id> <selector>\n"
              "       nm browse rtab wait-url <tab-id> <pattern>\n"
              "       nm browse rtab wait-text <tab-id> <text>\n"
              "       nm browse rtab wait-state <tab-id> [state]\n"
              "       nm browse rtab wait-idle <tab-id>\n"
              "       nm browse rtab extract <tab-id> [target] [--kind K]\n"
              "       nm browse rtab download <tab-id> <target>",
              file=sys.stderr)
        return 2
    sub = rest[0].strip().lower()
    backend = _backend(args, context)
    as_json = getattr(args, "json", False)
    if sub == "open":
        if len(rest) < 2:
            print("usage: nm browse rtab open <url> [--proxy P]",
                  file=sys.stderr)
            return 2
        result = backend.r_open(rest[1], getattr(args, "proxy", "") or "")
        tab = result["tab"]
        if as_json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"rendered tab {result['tab_id']}: "
                  f"{tab.get('title') or '(loading)'} — {tab.get('url')}")
        return 0
    if sub == "tabs":
        tabs = backend.r_tabs()
        if as_json:
            print(json.dumps(tabs, indent=2))
            return 0
        for t in tabs:
            print(f"{t['tab_id']}  {t.get('title') or '(untitled)'}  {t.get('url')}")
        if not tabs:
            print("no rendered tabs")
        return 0
    if sub in {"close", "shot"}:
        if len(rest) < 2:
            print(f"usage: nm browse rtab {sub} <tab-id>", file=sys.stderr)
            return 2
        result = backend.r_close(rest[1]) if sub == "close" \
            else backend.r_shot(rest[1])
        if as_json:
            print(json.dumps(result, indent=2, default=str))
        elif sub == "close":
            print(f"rendered tab {rest[1]} closed")
        else:
            print(f"screenshot -> {result['path']}")
        return 0
    if sub == "fill":
        if len(rest) < 4:
            print("usage: nm browse rtab fill <tab-id> <name> <value>",
                  file=sys.stderr)
            return 2
        result = backend.r_fill(rest[1], rest[2], " ".join(rest[3:]))
    elif sub == "fill-form":
        if len(rest) < 3:
            print("usage: nm browse rtab fill-form <tab-id> name=value ...",
                  file=sys.stderr)
            return 2
        fields: dict[str, str] = {}
        for pair in rest[2:]:
            if "=" not in pair:
                print(f"fill-form: {pair!r} is not name=value",
                      file=sys.stderr)
                return 2
            name, value = pair.split("=", 1)
            fields[name] = value
        result = backend.r_fill_form(rest[1], fields)
    elif sub == "set-date":
        if len(rest) < 4:
            print("usage: nm browse rtab set-date <tab-id> <name> <date>\n"
                  "       (e.g. 2026-10-04, 04/10/2026, '4 Oct 2026')",
                  file=sys.stderr)
            return 2
        result = backend.r_set_date(rest[1], rest[2], " ".join(rest[3:]))
    elif sub == "fields":
        if len(rest) < 2:
            print("usage: nm browse rtab fields <tab-id>", file=sys.stderr)
            return 2
        result = backend.r_fields(rest[1])
    elif sub == "eval":
        if len(rest) < 3:
            print("usage: nm browse rtab eval <tab-id> <js>",
                  file=sys.stderr)
            return 2
        result = backend.r_evaluate(rest[1], " ".join(rest[2:]))
    elif sub == "storage":
        if len(rest) < 4:
            print("usage: nm browse rtab storage <tab-id> "
                  "get|set|remove|clear <key> [value]", file=sys.stderr)
            return 2
        action = rest[2].lower()
        if action not in {"get", "set", "remove", "clear"}:
            print(f"unknown storage action {action!r}", file=sys.stderr)
            return 2
        result = backend.r_storage(rest[1], action, rest[3],
                                   " ".join(rest[4:]) if len(rest) > 4 else None)
    elif sub == "persist":
        if len(rest) < 2:
            print("usage: nm browse rtab persist <tab-id>", file=sys.stderr)
            return 2
        result = backend.r_persist(rest[1])
    elif sub == "click":
        if len(rest) < 3:
            print("usage: nm browse rtab click <tab-id> <target>",
                  file=sys.stderr)
            return 2
        result = backend.r_click(rest[1], " ".join(rest[2:]))
    elif sub == "submit":
        if len(rest) < 2:
            print("usage: nm browse rtab submit <tab-id> [target]",
                  file=sys.stderr)
            return 2
        result = backend.r_submit(rest[1], " ".join(rest[2:]))
    elif sub == "wait":
        if len(rest) < 3:
            print("usage: nm browse rtab wait <tab-id> <selector>",
                  file=sys.stderr)
            return 2
        result = backend.r_wait(rest[1], rest[2])
    elif sub == "wait-url":
        if len(rest) < 3:
            print("usage: nm browse rtab wait-url <tab-id> <pattern>\n"
                  "       (substring, or re:regex for SPAs)",
                  file=sys.stderr)
            return 2
        result = backend.r_wait_url(rest[1], " ".join(rest[2:]))
    elif sub == "wait-text":
        if len(rest) < 3:
            print("usage: nm browse rtab wait-text <tab-id> <text>",
                  file=sys.stderr)
            return 2
        result = backend.r_wait_text(rest[1], " ".join(rest[2:]))
    elif sub == "wait-state":
        if len(rest) < 2:
            print("usage: nm browse rtab wait-state <tab-id> "
                  "[load|domcontentloaded|networkidle]",
                  file=sys.stderr)
            return 2
        result = backend.r_wait_state(rest[1], rest[2] if len(rest) > 2 else "load")
    elif sub == "wait-idle":
        if len(rest) < 2:
            print("usage: nm browse rtab wait-idle <tab-id>",
                  file=sys.stderr)
            return 2
        result = backend.r_wait_idle(rest[1])
    elif sub == "select":
        if len(rest) < 4:
            print("usage: nm browse rtab select <tab-id> <name> <value>",
                  file=sys.stderr)
            return 2
        result = backend.r_select(rest[1], rest[2], " ".join(rest[3:]))
    elif sub == "check":
        if len(rest) < 3:
            print("usage: nm browse rtab check <tab-id> <name> [off]",
                  file=sys.stderr)
            return 2
        checked = not (len(rest) > 3 and rest[-1].lower() == "off")
        result = backend.r_check(rest[1], rest[2], checked)
    elif sub == "captcha":
        if len(rest) < 2:
            print("usage: nm browse rtab captcha <tab-id>",
                  file=sys.stderr)
            return 2
        result = backend.r_captcha(rest[1])
    elif sub == "extract":
        if len(rest) < 2:
            print("usage: nm browse rtab extract <tab-id> [target] [--kind K]",
                  file=sys.stderr)
            return 2
        result = backend.r_extract(rest[1], " ".join(rest[2:]),
                                   getattr(args, "kind", "") or "")
    elif sub == "download":
        if len(rest) < 3:
            print("usage: nm browse rtab download <tab-id> <target>",
                  file=sys.stderr)
            return 2
        result = backend.r_download(rest[1], " ".join(rest[2:]))
    elif sub == "upload":
        if len(rest) < 4:
            print("usage: nm browse rtab upload <tab-id> <selector> <path>",
                  file=sys.stderr)
            return 2
        result = backend.r_upload(rest[1], rest[2], " ".join(rest[3:]))
    elif sub == "wait-field":
        if len(rest) < 3:
            print("usage: nm browse rtab wait-field <tab-id> <name> "
                  "[timeout-ms]", file=sys.stderr)
            return 2
        timeout = 10_000
        name_parts = list(rest[2:])
        if name_parts and name_parts[-1].isdigit() and len(name_parts) > 1:
            timeout = int(name_parts.pop())
        result = backend.r_wait_field(rest[1], " ".join(name_parts),
                                      timeout=timeout)
    elif sub == "switch":
        if len(rest) < 3:
            print("usage: nm browse rtab switch <session> <tab-id>",
                  file=sys.stderr)
            return 2
        result = backend.r_switch(rest[1], rest[2])
    elif sub == "active":
        session = rest[1] if len(rest) > 1 else (
            getattr(args, "session", "") or "cli")
        result = backend.r_active(session)
        if result is None:
            print(f"no active rendered tab in session {session!r}")
            return 0
    else:
        print(f"unknown rtab verb: {sub}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0
