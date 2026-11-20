"""The cameras wall as a hosted page — the viewer's view, in Python.

The clone IS a page: a multiview grid of camera images that refresh
from each camera's own snapshot URL, click for singleview,
right-click to pause. This module exports that page over Devon's
own wall state, on the B.28 show/web.py shape:

* ``index.html`` — the wall page, the wall's title.
* ``main.js`` — the player: one interval (800 ms, the clone's loop)
  refreshes every unpaused camera's image from its snapshot URL
  with the cache-busting timestamp; click a tile for singleview
  (the rest pause, like the clone), right-click toggles pause;
  the titlebar shows location, last loaded time, queue % and fps.
  The manifest is inlined as a JS literal, so the page works from
  ``file://`` as well as any static host — no fetch, no server, no
  CDN, no dependencies.
* ``styles.css`` — the wall.
* ``data/wall.json`` — the manifest: wall, rows, cameras (with
  their snapshot URLs), stats, and which frames were exported.
* ``data/frame-<id>.jpg`` — the wall's latest fetched frame per
  camera, when it has one: the tile's first image, and its
  fallback when the live URL cannot be reached from the viewer's
  browser (a LAN camera, for instance).

Serving — :func:`serve_wall` runs one export dir on a loopback
HTTP server (stdlib only, daemon thread): one directory, one
port, no listing, path traversal refused. ``host="0.0.0.0"`` is
an explicit opt-in that exposes the wall on the network and is
named as such in the receipt. Per-export serving, not a hub.
"""

from __future__ import annotations

import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .errors import CamWallError

__all__ = [
    "WallServer",
    "export_web",
    "list_served",
    "serve_wall",
    "stop_wall",
]

_MANIFEST_NAME = "wall.json"

#: Content types the wall server answers with. Anything else is
#: refused — the server serves a wall, not a file browser.
_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json; charset=utf-8",
}

_TEMPLATES = Path(__file__).resolve().parent / "web_templates"


def export_web(wall: Any, out_dir: str | Path, *,
               title: str | None = None,
               overwrite: bool = False) -> dict[str, Any]:
    """Export a wall as a hosted page — a static site directory.

    Exports the wall's CURRENT state: cameras, their snapshot
    URLs, their stats, and the latest fetched frame per camera
    when the wall has one. The page then refreshes live from the
    camera URLs; a camera the viewer's browser cannot reach keeps
    showing its last exported frame, marked stale.
    """
    target = Path(out_dir).expanduser()
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise CamWallError(
            f"camwall web export refuses non-empty {target} — pick "
            f"an empty directory or pass overwrite=True",
            kind="out_not_empty")
    status = wall.status()
    if not status.get("camera_list"):
        raise CamWallError(
            f"wall {wall.name!r} has no cameras to export",
            kind="no_cameras")
    data_dir = target / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frames: list[str] = []
    cameras: list[dict[str, Any]] = []
    for cam in status["camera_list"]:
        entry = {
            "id": cam["id"],
            "name": cam["name"],
            "location": cam["location"],
            "vendor": cam["vendor"],
            "model": cam["model"],
            "transport": cam["transport"],
            "paused": cam["paused"],
            "url": cam["url"],
            "has_frame": cam["has_frame"],
            "frame_file": None,
            "frame_at": cam["frame_at"],
            "fps": cam["fps"],
            "counts": cam["counts"],
            "times": cam["times"],
        }
        if cam["has_frame"]:
            frame_bytes = wall.snapshot(cam["id"])
            name = f"frame-{cam['id']}.jpg"
            (data_dir / name).write_bytes(frame_bytes)
            entry["frame_file"] = f"data/{name}"
            frames.append(name)
        cameras.append(entry)
    manifest = {
        "wall": wall.name,
        "title": title or wall.name,
        "rows": status["rows"],
        "server": status["server"],
        "running": status["running"],
        "totals": status["totals"],
        "cameras": cameras,
        "frames": frames,
    }
    (data_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    title_html = html.escape(str(title or wall.name))
    index_html = (_TEMPLATES / "index.html").read_text(
        encoding="utf-8").replace("{{TITLE}}", title_html)
    (target / "index.html").write_text(index_html, encoding="utf-8")
    manifest_js = json.dumps(manifest).replace("<", "\\u003c")
    main_js = (_TEMPLATES / "main.js").read_text(
        encoding="utf-8").replace("{{MANIFEST}}", manifest_js)
    (target / "main.js").write_text(main_js, encoding="utf-8")
    styles = (_TEMPLATES / "styles.css").read_text(encoding="utf-8")
    (target / "styles.css").write_text(styles, encoding="utf-8")
    return {
        "wall": wall.name,
        "dir": str(target),
        "index": str(target / "index.html"),
        "cameras": len(cameras),
        "frames": len(frames),
        "files": ["index.html", "main.js", "styles.css",
                  f"data/{_MANIFEST_NAME}"]
                 + [f"data/{name}" for name in frames],
    }


# -- serving ---------------------------------------------------------

class WallServer:
    """One export directory, served on loopback.

    Stdlib only, daemon thread — the server dies with the
    process when nobody stops it first. One server per
    directory; serving the same directory twice returns the
    server that already serves it.
    """

    def __init__(self, root: Path, *,
                 host: str = "127.0.0.1", port: int = 0) -> None:
        self.root = root
        self.host = host
        self._httpd = ThreadingHTTPServer((host, port), _WallHandler)
        self._httpd.wall_root = root  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True,
            name=f"camwall-serve:{root.name}")
        self._thread.start()

    @property
    def port(self) -> int:
        """The bound port (the ephemeral one, when 0 was asked)."""
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        """The wall's URL — loopback unless host says otherwise."""
        return f"http://{self.host}:{self.port}/"

    def stop(self) -> None:
        """Stop serving. Safe to call twice."""
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:  # noqa: BLE001 - stop is best effort
            pass

    def status(self) -> dict[str, Any]:
        """The server, as data."""
        return {"dir": str(self.root), "host": self.host,
                "port": self.port, "url": self.url,
                "network": self.host == "0.0.0.0"}


class _WallHandler(BaseHTTPRequestHandler):
    """Serve one export directory: index + assets, nothing else.

    No directory listing, no path traversal, no content type
    outside the wall's own files — an unknown path is a 404,
    identically, so there is no oracle for guessing names.
    """

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        from urllib.parse import unquote, urlparse
        raw = unquote(urlparse(self.path).path)
        rel = raw.lstrip("/")
        if not rel:
            rel = "index.html"
        try:
            resolved = (self.server.wall_root / rel).resolve()
        except (ValueError, RuntimeError):
            self._answer(404, b"not found", "text/plain")
            return
        try:
            resolved.relative_to(self.server.wall_root.resolve())
        except ValueError:
            self._answer(404, b"not found", "text/plain")
            return
        mime = _MIME.get(resolved.suffix.lower())
        if mime is None or not resolved.is_file():
            self._answer(404, b"not found", "text/plain")
            return
        self._answer(200, resolved.read_bytes(), mime)

    def _answer(self, code: int, body: bytes, mime: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_SERVERS: dict[str, WallServer] = {}


def serve_wall(out_dir: str | Path, *,
               host: str = "127.0.0.1", port: int = 0) -> dict[str, Any]:
    """Serve an exported wall directory on loopback.

    Returns the server's status (dir, host, port, url). Serving
    the same directory twice returns the server that already
    serves it. ``host="0.0.0.0"`` is an explicit opt-in that
    exposes the wall on the network — the receipt says so.
    """
    root = Path(out_dir).expanduser().resolve()
    if not (root / "index.html").is_file():
        raise CamWallError(
            f"serve needs an exported wall directory — "
            f"{root} has no index.html",
            kind="bad_demo")
    if not isinstance(port, int) or port < 0 or port > 65535:
        raise CamWallError(
            f"wall port must be 0..65535, got {port}",
            kind="bad_port")
    key = str(root)
    server = _SERVERS.get(key)
    if server is None:
        server = WallServer(root, host=host, port=port)
        _SERVERS[key] = server
    return server.status()


def stop_wall(out_dir: str | Path) -> dict[str, Any]:
    """Stop serving an exported wall directory."""
    key = str(Path(out_dir).expanduser().resolve())
    server = _SERVERS.pop(key, None)
    if server is None:
        return {"dir": key, "serving": False}
    server.stop()
    return {"dir": key, "serving": False, "stopped": True}


def list_served() -> list[dict[str, Any]]:
    """The wall servers running now, one per directory."""
    return [server.status() for server in _SERVERS.values()]
