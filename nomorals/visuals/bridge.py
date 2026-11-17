"""The authenticated localhost bridge control plane.

clickyX's bridge (``src-tauri/src/bridge.rs``, ``docs/BRIDGE_API.md``)
is the reference: an actix-web server on 127.0.0.1:32123 with a
high-entropy token (Bearer / x-openclicky-token / X-Bridge-Token,
constant-time compare, hot-reloadable), ``/health`` the only
unauthenticated route, a "dangerous tier" that requires the token
even when auth is disabled for read-only routes, Host-header
validation against 127.0.0.1/localhost (DNS-rebinding defense), and
600 req/60s per-IP rate limiting.

Devon's bridge is stdlib-only (``http.server``) and stricter in the
ways that matter:

- the token is ALWAYS required for every route except ``/health``
  — there is no "auth disabled" mode, so there is no tier to
  misconfigure;
- the farzaa Worker's failure is not repeated: no route is
  authenticated by URL obscurity alone;
- the bridge drives the scene organ only — point, annotate, flight,
  render, clear. It is a control plane for Devon's own surface, not
  a general computer-use API: there is no click, no keystroke, no
  screenshot route. Annotation is never actuation.

The token lives at ``<home>/visuals/bridge_token`` (0600),
generated on first run with :mod:`secrets` and read from disk on
every request, so rotating the file takes effect immediately.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..core.failure_store import home_dir as _devon_home_dir
from .annotations import get_board
from .flight import flight_frames
from .interactive import write_interactive_page
from .tags import parse_tags, tags_to_annotations

__all__ = [
    "DEFAULT_BRIDGE_PORT",
    "SceneBridgeServer",
    "load_or_create_token",
    "start_bridge",
    "stop_bridge",
    "token_path",
]

#: Devon's bridge port (clickyX uses 32123; Devon takes her own).
DEFAULT_BRIDGE_PORT = 18723

TOKEN_FILENAME = "bridge_token"
SERVICE_NAME = "devon-scene-bridge"

#: Max request body: 1 MiB (a scene spec is kilobytes).
_MAX_BODY_BYTES = 1024 * 1024
#: Per-IP rate limit: 600 requests per 60 s (clickyX's numbers).
_RATE_LIMIT_REQUESTS = 600
_RATE_LIMIT_WINDOW_S = 60.0

#: Hosts the bridge answers for (DNS-rebinding defense).
_ALLOWED_HOSTS = ("127.0.0.1", "localhost")


def token_path(home: Path | str | None = None) -> Path:
    """``<home>/visuals/bridge_token``."""
    base = Path(home) if home is not None else Path(_devon_home_dir())
    return base / "visuals" / TOKEN_FILENAME


def load_or_create_token(home: Path | str | None = None) -> str:
    """The bridge token: read it, or generate + store it (0600)."""
    path = token_path(home)
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    if existing:
        return existing
    token = secrets.token_hex(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token + "\n", encoding="utf-8")
    except OSError:
        # A token that cannot be stored still works for this process;
        # the next start generates a fresh one.
        pass
    with contextlib.suppress(OSError):
        path.chmod(0o600)
    return token


def _read_token(home: Path | str | None) -> str:
    try:
        return token_path(home).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def verify_token(provided: str | None, expected: str) -> bool:
    """Constant-time token comparison. An empty expected token
    never verifies."""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)


class SceneBridgeServer(ThreadingHTTPServer):
    """The bridge: a board, a cursor, and a rate-limit table."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int],
                 home: Path | str | None = None) -> None:
        self.home = home
        self.board = get_board(home)
        self.cursor: tuple[float, float, int] = self.board.cursor
        self.rate_limit = _RATE_LIMIT_REQUESTS
        self.rate_window_s = _RATE_LIMIT_WINDOW_S
        self._hits: dict[str, list[float]] = {}
        self._hits_lock = threading.Lock()
        super().__init__(address, _SceneBridgeHandler)

    def rate_limited(self, ip: str) -> bool:
        now = time.time()
        with self._hits_lock:
            hits = [t for t in self._hits.get(ip, [])
                    if now - t < self.rate_window_s]
            if len(hits) >= self.rate_limit:
                self._hits[ip] = hits
                return True
            hits.append(now)
            self._hits[ip] = hits
            return False


class _SceneBridgeHandler(BaseHTTPRequestHandler):
    server: SceneBridgeServer  # type: ignore[assignment]

    # ── plumbing ─────────────────────────────────────────────────

    def log_message(self, *args: Any) -> None:  # noqa: D102 - silent
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > _MAX_BODY_BYTES:
            return {} if length == 0 else None
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        return host in _ALLOWED_HOSTS

    def _authed(self) -> bool:
        auth = self.headers.get("Authorization") or ""
        provided = None
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        elif self.headers.get("X-Bridge-Token"):
            provided = self.headers.get("X-Bridge-Token", "").strip()
        expected = _read_token(self.server.home)
        return verify_token(provided, expected)

    def _guard(self) -> bool:
        """Host + rate + auth checks. True when the request may
        proceed; otherwise the refusal is already sent."""
        if not self._host_ok():
            self._json(403, {"ok": False,
                             "error": "host not allowed"})
            return False
        ip = self.client_address[0] if self.client_address else "?"
        if self.server.rate_limited(ip):
            self._json(429, {"ok": False,
                             "error": "rate limited"})
            return False
        if self.path.rstrip("/") != "/health" and not self._authed():
            self._json(401, {"ok": False,
                             "error": "missing or bad bridge token"})
            return False
        return True

    # ── routes ───────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if not self._guard():
            return
        if self.path.rstrip("/") == "/health":
            self._json(200, {"ok": True, "service": SERVICE_NAME,
                             "version": 1})
            return
        if self.path.rstrip("/") == "/annotations":
            self._json(200, {
                "ok": True,
                "annotations": [a.to_dict()
                                for a in self.server.board.list()],
            })
            return
        if self.path.rstrip("/") == "/state":
            board = self.server.board
            self._json(200, {
                "ok": True,
                "cursor": {"x": board.cursor[0],
                           "y": board.cursor[1],
                           "screen": board.cursor[2]},
                "annotations": [a.to_dict() for a in board.list()],
                "last_flight": board.last_flight,
            })
            return
        self._json(404, {"ok": False, "error": "unknown route"})

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        if not self._guard():
            return
        body = self._read_body()
        if body is None:
            self._json(400, {"ok": False,
                             "error": "body must be a JSON object"})
            return
        board = self.server.board
        path = self.path.rstrip("/")
        if path == "/cursor":
            self._cursor(body)
        elif path == "/annotations":
            self._create_annotation(body)
        elif path == "/point":
            self._point(body)
        elif path == "/scene":
            self._scene(body)
        elif path == "/clear":
            cleared = board.clear()
            board.set_last_flight([])
            self._json(200, {"ok": True, "cleared": cleared})
        else:
            self._json(404, {"ok": False, "error": "unknown route"})

    def do_PATCH(self) -> None:  # noqa: N802 - http.server naming
        if not self._guard():
            return
        body = self._read_body()
        if body is None:
            self._json(400, {"ok": False,
                             "error": "body must be a JSON object"})
            return
        parts = [p for p in self.path.split("/") if p]
        if len(parts) == 2 and parts[0] == "annotations":
            try:
                ann = self.server.board.update(parts[1], **body)
            except KeyError:
                self._json(404, {"ok": False,
                                 "error": "no such annotation"})
                return
            except ValueError as exc:
                self._json(400, {"ok": False, "error": str(exc)})
                return
            self._json(200, {"ok": True,
                             "annotation": ann.to_dict()})
            return
        self._json(404, {"ok": False, "error": "unknown route"})

    def do_DELETE(self) -> None:  # noqa: N802 - http.server naming
        if not self._guard():
            return
        parts = [p for p in self.path.split("/") if p]
        if len(parts) == 2 and parts[0] == "annotations":
            removed = self.server.board.remove(parts[1])
            if not removed:
                self._json(404, {"ok": False,
                                 "error": "no such annotation"})
                return
            self._json(200, {"ok": True, "removed": parts[1]})
            return
        self._json(404, {"ok": False, "error": "unknown route"})

    # ── handlers ─────────────────────────────────────────────────

    def _cursor(self, body: dict[str, Any]) -> None:
        try:
            x = float(body["x"])
            y = float(body["y"])
        except (KeyError, TypeError, ValueError):
            self._json(400, {"ok": False,
                             "error": "cursor needs numeric x and y"})
            return
        screen = 1
        with contextlib.suppress(TypeError, ValueError):
            screen = int(body.get("screen", 1))
        label = str(body.get("label", ""))
        board = self.server.board
        start = (board.cursor[0], board.cursor[1])
        frames = flight_frames(start, (x, y))
        board.point(x, y, label=label, screen=screen)
        board.set_last_flight(frames)
        self._json(200, {"ok": True,
                         "from": {"x": start[0], "y": start[1]},
                         "to": {"x": x, "y": y, "screen": screen},
                         "label": label,
                         "flight": frames})

    def _create_annotation(self, body: dict[str, Any]) -> None:
        try:
            x = float(body["x"])
            y = float(body["y"])
        except (KeyError, TypeError, ValueError):
            self._json(400, {"ok": False,
                             "error": "annotation needs numeric x and y"})
            return
        points = body.get("points") or ()
        try:
            ann = self.server.board.create(
                kind=str(body.get("kind", "cursor")),
                x=x, y=y,
                w=float(body.get("w", 0.0) or 0.0),
                h=float(body.get("h", 0.0) or 0.0),
                points=points,
                shape=str(body.get("shape", "")),
                text=str(body.get("text", "")),
                screen=int(body.get("screen", 1) or 1),
                timeout_ms=int(body.get("timeout_ms", 5000) or 5000),
            )
        except (TypeError, ValueError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        self._json(200, {"ok": True, "annotation": ann.to_dict()})

    def _point(self, body: dict[str, Any]) -> None:
        text = body.get("text", "")
        if not isinstance(text, str):
            self._json(400, {"ok": False,
                             "error": "point needs a text string"})
            return
        width = body.get("width")
        height = body.get("height")
        parsed = parse_tags(
            text,
            width=float(width) if width else None,
            height=float(height) if height else None,
            screen_count=int(body.get("screen_count", 1) or 1))
        board = self.server.board
        placed: list[str] = []
        for ann in tags_to_annotations(parsed.tags):
            created = board.create(
                kind=ann["kind"], x=ann["x"], y=ann["y"],
                w=ann.get("w", 0.0), h=ann.get("h", 0.0),
                points=ann.get("points", ()),
                shape=ann.get("shape", ""), text=ann.get("text", ""),
                screen=ann.get("screen", 1))
            placed.append(created.id)
        self._json(200, {"ok": True, "text": parsed.text,
                         "placed": placed, "none": parsed.none})

    def _scene(self, body: dict[str, Any]) -> None:
        spec = body.get("spec")
        if not isinstance(spec, dict):
            self._json(400, {"ok": False,
                             "error": "scene needs a spec dict"})
            return
        try:
            flight = body.get("flight") or []
            path = write_interactive_page(
                spec,
                body.get("path") or _default_page_path(self.server.home),
                flight=flight,
                title=str(body.get("title", "")))
        except Exception as exc:  # noqa: BLE001 - refusal, not crash
            self._json(400, {"ok": False, "error": str(exc)})
            return
        self._json(200, {"ok": True, "path": path})


def _default_page_path(home: Path | str | None) -> str:
    from ..core.ids import new_id

    base = Path(home) if home is not None else Path(_devon_home_dir())
    return str(base / "visuals" / f"scene-{new_id()[:8]}.html")


def start_bridge(home: Path | str | None = None, *,
                 port: int = DEFAULT_BRIDGE_PORT) -> SceneBridgeServer:
    """Start the bridge on 127.0.0.1:port (daemon thread).

    The token is created on first run. The server object is
    returned; :func:`stop_bridge` shuts it down.
    """
    load_or_create_token(home)
    server = SceneBridgeServer(("127.0.0.1", port), home=home)
    thread = threading.Thread(target=server.serve_forever,
                              name="devon-scene-bridge", daemon=True)
    thread.start()
    return server


def stop_bridge(server: SceneBridgeServer) -> None:
    """Stop a running bridge."""
    try:
        server.shutdown()
    finally:
        server.server_close()


def bridge_fingerprint(home: Path | str | None = None) -> str:
    """A non-secret fingerprint of the token (for status lines —
    the token itself is never printed)."""
    token = _read_token(home)
    if not token:
        return ""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
