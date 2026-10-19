"""Human-in-the-loop browser takeover: ONE session, owner takes the mouse.

When Devon hits a wall she cannot clear herself, the takeover manager
opens a live view of the EXACT browser session she is driving — not a
copy. The owner opens the link (Telegram webview, or "open in Chrome"),
sees Devon's real browser, taps/types into it, and those inputs go
straight into her session: same cookies, same login, same challenge.

Safety, by construction:
- the link carries a one-time, expiring token; the stream never serves
  without a valid token;
- a new takeover for a session supersedes (closes) any prior grant;
- the grant dies on hand-back, expiry, or supersede — a leaked link
  stops working;
- the audit log records every grant / hand-back / resume with only a
  token-hash prefix, never the token;
- while a takeover is active the agent's own actions are paused
  (``HandoffSession`` HANDED state) — Devon never fights the owner
  for the mouse.

Deployment-agnostic: bind address, port, public base URL, TTLs and
timeouts all come from :class:`TakeoverConfig` (env ``NM_TAKEOVER_*``
or settings) — nothing here knows about EC2, Windows, hostnames, or
IPs. When no public base URL is configured and the bind is loopback,
the grant is flagged ``needs_public_base``: the link works on the host
network only until the operator configures a tunnel / port-forward.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

from ..core.logging_setup import get_logger
from .handoff import HandoffSession

_log = get_logger(__name__)

__all__ = [
    "Resolution",
    "TakeoverConfig",
    "TakeoverGrant",
    "TakeoverManager",
    "TakeoverTabAdapter",
]

#: Frame refresh floor per profile — screenshots are cheap, but a
#: phone-streaming owner does not need 60fps, and Termux hosts are
#: slow. Config ``frame_interval_s`` overrides when > 0.
_FRAME_INTERVAL_S = {"termux": 3.0, "laptop": 1.0, "workstation": 0.75}
_DEFAULT_FRAME_INTERVAL_S = 1.0


class TakeoverTabAdapter(Protocol):
    """Duck-typed live tab. Anything with these members works."""

    @property
    def url(self) -> str: ...
    @property
    def title(self) -> str: ...
    def html(self) -> str: ...
    def reload(self) -> None: ...
    def screenshot_png(self) -> bytes: ...
    def click_norm(self, x: float, y: float) -> None: ...
    def type_text(self, text: str) -> None: ...
    def press_key(self, key: str) -> None: ...
    def scroll_norm(self, dx: float, dy: float) -> None: ...


@dataclass
class TakeoverConfig:
    """All takeover knobs. No hostnames, IPs, or ports are hardcoded."""

    enabled: bool = True
    #: Public URL prefix for takeover links, e.g. a tunnel URL or
    #: ``http://host:PORT``. Empty = fall back to the bind address
    #: (works on the host network only).
    base_url: str = ""
    bind_host: str = "127.0.0.1"
    port: int = 0  # 0 = ephemeral; the OS picks a free port
    link_ttl_s: int = 900
    wait_timeout_s: int = 900
    poll_s: float = 2.0
    #: Min seconds between served frames; 0 = profile default.
    frame_interval_s: float = 0.0
    profile: str | None = None

    @classmethod
    def from_env(cls) -> "TakeoverConfig":
        """Read ``NM_TAKEOVER_*`` env vars. Missing = defaults."""
        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, "") or default)
            except ValueError:
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, "") or default)
            except ValueError:
                return default

        return cls(
            enabled=os.environ.get("NM_TAKEOVER_ENABLED", "1") != "0",
            base_url=os.environ.get("NM_TAKEOVER_BASE_URL", "").strip(),
            bind_host=os.environ.get("NM_TAKEOVER_BIND_HOST",
                                     "127.0.0.1").strip() or "127.0.0.1",
            port=_int("NM_TAKEOVER_PORT", 0),
            link_ttl_s=_int("NM_TAKEOVER_LINK_TTL_S", 900),
            wait_timeout_s=_int("NM_TAKEOVER_WAIT_TIMEOUT_S", 900),
            poll_s=_float("NM_TAKEOVER_POLL_S", 2.0),
        )

    @classmethod
    def from_settings(cls, settings: Any) -> "TakeoverConfig":
        """Read a ``settings.takeover`` section when present."""
        section = getattr(settings, "takeover", None)
        if section is None:
            return cls.from_env()
        return cls(
            enabled=bool(getattr(section, "enabled", True)),
            base_url=str(getattr(section, "base_url", "") or "").strip(),
            bind_host=str(getattr(section, "bind_host", "127.0.0.1")
                          or "127.0.0.1"),
            port=int(getattr(section, "port", 0) or 0),
            link_ttl_s=int(getattr(section, "link_ttl_s", 900) or 900),
            wait_timeout_s=int(getattr(section, "wait_timeout_s", 900)
                               or 900),
            poll_s=float(getattr(section, "poll_s", 2.0) or 2.0),
        )

    def frame_interval(self) -> float:
        if self.frame_interval_s and self.frame_interval_s > 0:
            return self.frame_interval_s
        try:
            from ..core.profiles import get_profile_kind
            kind = (self.profile or get_profile_kind()).strip().lower()
        except Exception:  # noqa: BLE001 - profile lookup never blocks
            kind = ""
        return _FRAME_INTERVAL_S.get(kind, _DEFAULT_FRAME_INTERVAL_S)


@dataclass
class TakeoverGrant:
    """One takeover grant: one token, one session, one expiry."""

    token: str
    token_hash: str
    url: str
    session_id: str
    owner_chat: str
    created_at: float
    expires_at: float
    resume_state: dict[str, Any] = field(default_factory=dict)
    task_label: str = ""
    signal_kind: str = ""
    done: bool = False
    done_note: str = ""
    closed: bool = False
    #: True when no public base URL is configured and the bind is
    #: loopback — the link is host-network-only until the operator
    #: configures a tunnel / port-forward.
    needs_public_base: bool = False
    tab: Any = None
    handoff: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "session_id": self.session_id,
                "owner_chat": self.owner_chat,
                "created_at": self.created_at,
                "expires_at": self.expires_at,
                "task_label": self.task_label,
                "signal_kind": self.signal_kind, "done": self.done,
                "closed": self.closed,
                "needs_public_base": self.needs_public_base,
                "token_hash_prefix": self.token_hash[:8]}


@dataclass
class Resolution:
    cleared: bool
    via: str  # owner_done | auto_detected | still_blocked
    resume_state: dict[str, Any] | None = None
    waited_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"cleared": self.cleared, "via": self.via,
                "resume_state": self.resume_state,
                "waited_s": self.waited_s}


_TAKEOVER_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Devon browser takeover</title>
<style>
body{font-family:system-ui,sans-serif;margin:12px;background:#111;color:#eee}
img{width:100%;border:1px solid #444;background:#000;touch-action:none}
button{font-size:16px;padding:10px 16px;margin:6px 4px}
input{font-size:16px;padding:8px;width:70%}
.note{color:#aaa;font-size:13px}
</style></head><body>
<h3>🧑‍💻 You are controlling Devon's live browser</h3>
<p class="note">Same session, same login. Tap the screenshot to click.
Type below to send text to the focused field.</p>
<img id="frame" alt="live browser frame">
<div>
<input id="text" placeholder="type text…">
<button onclick="sendType()">Send text</button>
<button onclick="sendKey('Enter')">Enter</button>
<button onclick="sendKey('Tab')">Tab</button>
</div>
<div><button onclick="done()" style="background:#2a7;color:#fff">
✅ Done, take over</button></div>
<p class="note">If this view feels cramped, copy this page's link and
open it in Chrome — it is the same live session. The link expires and
stops working once you tap Done.</p>
<script>
const base = location.pathname.replace(/\\/$/, "");
const img = document.getElementById("frame");
function refresh(){ img.src = base + "/frame?t=" + Date.now(); }
setInterval(refresh, 1000); refresh();
img.addEventListener("click", (e) => {
  const r = img.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width;
  const y = (e.clientY - r.top) / r.height;
  fetch(base + "/click", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({x, y})});
});
function sendType(){
  const v = document.getElementById("text").value;
  if(!v) return;
  fetch(base + "/type", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text: v})});
  document.getElementById("text").value = "";
}
function sendKey(k){
  fetch(base + "/key", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({key: k})});
}
function done(){
  fetch(base + "/done", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({note: "owner tapped Done"})})
    .then(() => { document.body.innerHTML =
      "<h3>✅ Handed back — Devon is checking the wall and resuming.</h3>"; });
}
</script></body></html>
"""


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _make_handler(manager: "TakeoverManager"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # noqa: ANN401
            pass  # never log takeover URLs (they carry tokens)

        # -- plumbing ------------------------------------------------------
        def _send(self, code: int, body: bytes,
                  ctype: str = "application/json") -> None:
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"))

        def _grant(self) -> TakeoverGrant | None:
            parts = urlparse(self.path).path.strip("/").split("/")
            # /takeover/<token>[/action]
            if len(parts) < 2 or parts[0] != "takeover":
                return None
            return manager.validate_token(parts[1])

        def _body_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return {}
            if length <= 0 or length > 1_000_000:
                return {}
            try:
                raw = self.rfile.read(length)
                parsed = json.loads(raw.decode("utf-8"))
                return parsed if isinstance(parsed, dict) else {}
            except (ValueError, UnicodeDecodeError):
                return {}

        # -- routes ----------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            grant = self._grant()
            if grant is None:
                self._json(403, {"ok": False, "error": "invalid/expired "
                              "takeover link"})
                return
            parts = urlparse(self.path).path.strip("/").split("/")
            if len(parts) == 3 and parts[2] == "frame":
                png = manager.frame_for(grant)
                if png is None:
                    self._json(502, {"ok": False,
                                     "error": "no frame available"})
                    return
                self._send(200, png, ctype="image/png")
                return
            if len(parts) == 2:
                self._send(200, _TAKEOVER_PAGE.encode("utf-8"),
                           ctype="text/html; charset=utf-8")
                return
            self._json(404, {"ok": False, "error": "unknown route"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            grant = self._grant()
            if grant is None:
                self._json(403, {"ok": False, "error": "invalid/expired "
                              "takeover link"})
                return
            parts = urlparse(self.path).path.strip("/").split("/")
            if len(parts) != 3 or parts[0] != "takeover":
                self._json(404, {"ok": False, "error": "unknown route"})
                return
            action = parts[2]
            body = self._body_json()
            tab = grant.tab
            try:
                if action == "click" and tab is not None:
                    x = max(0.0, min(1.0, float(body.get("x", 0.0))))
                    y = max(0.0, min(1.0, float(body.get("y", 0.0))))
                    tab.click_norm(x, y)
                elif action == "type" and tab is not None:
                    tab.type_text(str(body.get("text", "")))
                elif action == "key" and tab is not None:
                    tab.press_key(str(body.get("key", "")))
                elif action == "scroll" and tab is not None:
                    tab.scroll_norm(float(body.get("dx", 0.0)),
                                    float(body.get("dy", 0.0)))
                elif action == "done":
                    manager.mark_done(
                        grant, note=str(body.get("note", "") or ""))
                else:
                    self._json(400, {"ok": False,
                                     "error": f"unknown action {action!r}"})
                    return
            except Exception as exc:  # noqa: BLE001 - input errors are data
                self._json(502, {"ok": False, "error": str(exc)})
                return
            self._json(200, {"ok": True, "action": action})

    return Handler


class TakeoverManager:
    """Issues takeover grants, serves the live view, waits for hand-back."""

    def __init__(self, config: TakeoverConfig | None = None, *,
                 data_dir: str | Path | None = None,
                 notifier: Callable[[str, str], Any] | None = None,
                 session_store: Any | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self._config = config or TakeoverConfig()
        self._data_dir = Path(data_dir).expanduser() if data_dir else None
        if self._data_dir is not None:
            self._data_dir.mkdir(parents=True, exist_ok=True)
        self._notifier = notifier
        self._session_store = session_store
        self._clock = clock or time.time
        self._grants: dict[str, TakeoverGrant] = {}
        self._by_session: dict[str, TakeoverGrant] = {}
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._frame_cache: dict[str, tuple[bytes, float]] = {}
        self._frame_lock = threading.Lock()

    # -- config ---------------------------------------------------------------
    @property
    def wait_timeout_s(self) -> int:
        return self._config.wait_timeout_s

    # -- grants -----------------------------------------------------------------
    def request_takeover(self, *, session_id: str, url: str,
                         task_label: str, signal_kind: str,
                         attempts: list[Any], resume_state: dict[str, Any],
                         tab: Any, owner_chat: str = "") -> TakeoverGrant:
        """Pause the agent, issue a one-time expiring takeover link."""
        now = self._clock()
        with self._lock:
            prior = self._by_session.get(session_id)
            if prior is not None and not prior.closed:
                prior.closed = True
                self._audit("grant_superseded", prior)
            token = secrets.token_urlsafe(32)
            grant = TakeoverGrant(
                token=token, token_hash=_hash_token(token), url="",
                session_id=session_id, owner_chat=owner_chat,
                created_at=now,
                expires_at=now + max(1, self._config.link_ttl_s),
                resume_state=dict(resume_state or {}),
                task_label=task_label, signal_kind=signal_kind,
                needs_public_base=(
                    not self._config.base_url
                    and self._config.bind_host in
                    ("127.0.0.1", "localhost", "::1")),
                tab=tab)
            handoff = self._new_handoff(session_id, task_label)
            grant.handoff = handoff
            grant.url = f"{self._public_base()}/takeover/{token}"
            self._grants[grant.token_hash] = grant
            self._by_session[session_id] = grant
        try:
            handoff.start_live()
            handoff.pause()
            handoff.hand_mouse()
        except Exception:  # noqa: BLE001 - handoff state is bookkeeping
            _log.debug("takeover handoff transition failed", exc_info=True)
        self._audit("grant_issued", grant,
                    {"url": url, "attempts": len(attempts)})
        return grant

    def validate_token(self, token: str) -> TakeoverGrant | None:
        """Token -> grant, or None. Never reveals why a token failed."""
        grant = self._grants.get(_hash_token(token or ""))
        if grant is None or grant.closed or grant.done:
            return None
        if self._clock() >= grant.expires_at:
            grant.closed = True
            return None
        return grant

    def record_session_saved(self, grant: TakeoverGrant, *,
                             site: str, cookie_count: int,
                             via: str) -> None:
        """Audit a solved-session save. Site + count + via only — never
        cookie values. Called by the solver on a verified clear."""
        from .challenge_memory import _site_of
        self._audit("session_saved", grant,
                    {"site": _site_of(site),
                     "cookies": max(0, int(cookie_count)),
                     "via": via})

    def mark_done(self, grant: TakeoverGrant, *, note: str = "") -> None:
        """Owner tapped Done. The token dies; resolution verifies the wall."""
        with self._lock:
            if grant.done or grant.closed:
                return
            grant.done = True
            grant.done_note = (note or "").strip()
            handoff = grant.handoff
        if handoff is not None:
            try:
                handoff.owner_note(grant.done_note or "owner done")
            except Exception:  # noqa: BLE001 - notes are bookkeeping
                pass
        self._audit("owner_done", grant, {"note": grant.done_note})

    # -- resolution ---------------------------------------------------------------
    def wait_for_resolution(self, grant: TakeoverGrant, *,
                            cleared_check: Callable[[], bool],
                            timeout_s: float | None = None,
                            poll_s: float | None = None) -> Resolution:
        """Wait for hand-back: explicit Done, or auto-detected clear.

        Done alone is NOT a clear — the wall is verified before resume.
        If the wall is still up, the result is honestly ``still_blocked``
        and the run does NOT blindly continue.
        """
        started = self._clock()
        timeout = (self._config.wait_timeout_s if timeout_s is None
                   else timeout_s)
        interval = (self._config.poll_s if poll_s is None else poll_s)
        deadline = started + max(0.0, float(timeout))
        while True:
            if grant.done:
                if self._safe_check(cleared_check):
                    self._resume(grant, "owner_done")
                    return Resolution(cleared=True, via="owner_done",
                                      resume_state=dict(grant.resume_state),
                                      waited_s=self._clock() - started)
                self._audit("still_blocked_after_done", grant)
                return Resolution(cleared=False, via="still_blocked",
                                  resume_state=None,
                                  waited_s=self._clock() - started)
            if self._safe_check(cleared_check):
                self._resume(grant, "auto_detected")
                return Resolution(cleared=True, via="auto_detected",
                                  resume_state=dict(grant.resume_state),
                                  waited_s=self._clock() - started)
            if self._clock() >= deadline:
                return Resolution(cleared=False, via="still_blocked",
                                  resume_state=None,
                                  waited_s=self._clock() - started)
            time.sleep(max(0.01, float(interval)))

    def _safe_check(self, cleared_check: Callable[[], bool]) -> bool:
        try:
            return bool(cleared_check())
        except Exception:  # noqa: BLE001 - check errors mean "not cleared"
            return False

    def _resume(self, grant: TakeoverGrant, via: str) -> None:
        with self._lock:
            grant.closed = True
        handoff = grant.handoff
        if handoff is not None:
            try:
                handoff.resume(note=f"takeover resolved via {via}")
            except Exception:  # noqa: BLE001 - handoff is bookkeeping
                _log.debug("takeover handoff resume failed", exc_info=True)
        self._audit("resumed", grant, {"via": via})

    # -- live view server -----------------------------------------------------------
    def ensure_server(self) -> ThreadingHTTPServer:
        """Start the stream server (once). Binds the configured address."""
        if self._server is not None:
            return self._server
        handler = _make_handler(self)
        server = ThreadingHTTPServer(
            (self._config.bind_host, self._config.port), handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True,
                                  name="takeover-stream")
        thread.start()
        self._server = server
        self._server_thread = thread
        return server

    @property
    def server_url(self) -> str:
        server = self.ensure_server()
        host, port = server.server_address[:2]
        return f"http://{host}:{port}"

    def _public_base(self) -> str:
        if self._config.base_url:
            return self._config.base_url.rstrip("/")
        return self.server_url

    def frame_for(self, grant: TakeoverGrant) -> bytes | None:
        """Latest screenshot for a grant, throttled by profile interval."""
        tab = grant.tab
        if tab is None:
            return None
        interval = self._config.frame_interval()
        now = self._clock()
        with self._frame_lock:
            cached = self._frame_cache.get(grant.session_id)
            if cached is not None and now - cached[1] < interval:
                return cached[0]
        try:
            png = tab.screenshot_png()
        except Exception:  # noqa: BLE001 - frame errors are 502s
            return None
        if not isinstance(png, (bytes, bytearray)) or not png:
            return None
        raw = bytes(png)
        with self._frame_lock:
            self._frame_cache[grant.session_id] = (raw, now)
        return raw

    def shutdown(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass

    def __enter__(self) -> "TakeoverManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # -- internals --------------------------------------------------------------------
    def _new_handoff(self, session_id: str,
                     task_label: str) -> HandoffSession:
        state_path = None
        if self._data_dir is not None:
            state_path = (self._data_dir / "handoffs"
                          / f"{session_id}.json")
        return HandoffSession(session_id=session_id,
                              task_label=task_label,
                              state_path=state_path)

    def _audit(self, event: str, grant: TakeoverGrant,
               extra: dict[str, Any] | None = None) -> None:
        if self._data_dir is None:
            return
        entry = {"ts": self._clock(), "event": event,
                 "token_hash_prefix": grant.token_hash[:8],
                 "session_id": grant.session_id,
                 "owner_chat": grant.owner_chat,
                 "task_label": grant.task_label,
                 "signal_kind": grant.signal_kind}
        if extra:
            entry.update(extra)
        try:
            path = self._data_dir / "takeover-audit.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError:
            _log.debug("takeover audit write failed", exc_info=True)
