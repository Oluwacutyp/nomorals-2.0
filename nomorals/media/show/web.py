"""The hosted demo — a show as a deployable static site.

The pattern (storytold/hanashi-graphics-demo, pattern only, no
upstream code, no licence): a compiled game-engine demo committed
as a static site — ``index.html`` (a Start button and a canvas
host page), ``main.js``, ``styles.css``, ``pkg/`` (the compiled
engine) and ``data/`` (assets). The artefact *is* the evaluation:
the demo runs in the browser, no server, no build step.

Devon has no compiled engine to ship, and she does not need one:
her show already renders real frames (the rig's 2D puppet, RGB24
per tick). The hosted demo is the same shape over her own
renderer — :func:`export_web` runs a show (or takes the poses it
already drove) and writes the static site:

* ``index.html`` — Start button + canvas, the show's title.
* ``main.js`` — the player: preloads the frames as images,
  plays them on the canvas at the show's fps, draws the act's
  say-lines as captions and applies the camera shots as crops.
  The manifest is inlined as a JS literal, so the page works
  from ``file://`` as well as any static host — no fetch, no
  server, no CDN, no dependencies. ``?autoplay=1`` starts
  without a click (embeds and headless proofs); otherwise the
  Start button is the gesture, like the demo this comes from.
* ``styles.css`` — the stage.
* ``data/frame-NNNN.png`` — the rendered poses, one PNG per
  exported frame.
* ``data/show.json`` — the manifest: show, fps, size, frames,
  captions and shots in act-seconds, and what the export had
  to drop.

Captions and shots come from the show's own journal — the
``vtuber_beat_*`` events the act player records, in act-seconds
(``detail.at``), read off each beat's own ``say``/``camera``
fields, since the journal kind is only the beat's first
action. A line shows
from its beat until the next line or the end of the clip; a
shot holds until the next camera beat or the end — the same
hold rule the player itself uses for shots. Audience cues are
not journaled by the player, so they do not reach the export;
the manifest says so when the journal dropped events.

Serving — :func:`serve_demo` runs one export dir on a loopback
HTTP server (stdlib only, daemon thread): one directory, one
port, no listing, path traversal refused. ``host="0.0.0.0"``
is an explicit opt-in that exposes the demo on the network and
is named as such in the receipt. This is per-export serving,
not a hub: each server owns exactly one directory, and
:func:`stop_demo` takes it down.
"""

from __future__ import annotations

import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .errors import ShowError

__all__ = [
    "DemoServer",
    "MAX_EXPORT_FRAMES",
    "MAX_EXPORT_SIZE",
    "export_web",
    "list_demos",
    "serve_demo",
    "stop_demo",
]

#: Frames per export, at most — 30 seconds at 15 fps by default.
#: The show buffers 900 posed frames (60 seconds); the export
#: takes the last MAX_EXPORT_FRAMES and says how many it dropped.
MAX_EXPORT_FRAMES = 450

#: Largest frame edge the export renders, in pixels.
MAX_EXPORT_SIZE = 1920

#: Smallest frame edge, matching the rig's own floor.
MIN_EXPORT_SIZE = 64

#: File the player loads per frame, and the manifest.
_MANIFEST_NAME = "show.json"

#: Content types the demo server answers with. Anything else is
#: refused — the server serves a demo, not a file browser.
_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".json": "application/json; charset=utf-8",
}

_TEMPLATES = Path(__file__).resolve().parent / "web_templates"


def _require_pil() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ShowError(
            "exporting a show as a web demo needs Pillow — it is "
            "not installed on this machine (pip install Pillow)",
            kind="missing_pil") from exc
    return Image


def _overlays(show: Any, duration_s: float) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Captions + shots from the show's journal, in act-seconds.

    The act player journals one event per beat
    (``vtuber_beat_say`` / ``vtuber_beat_camera``) with
    ``detail.at`` the beat's time in seconds. A caption runs
    from its beat until the next caption or the end of the
    clip; a shot holds until the next camera beat or the end.
    Returns (captions, shots, partial) — partial when the
    journal dropped events, so the overlays may miss early
    beats.
    """
    events = show.events()
    partial = show.events_dropped > 0
    say_starts: list[tuple[float, str]] = []
    shot_starts: list[tuple[float, str]] = []
    for event in events:
        kind = event.get("kind", "")
        if not kind.startswith("vtuber_beat"):
            continue
        detail = event.get("detail") or {}
        beat = detail.get("beat") or {}
        at = detail.get("at")
        if not isinstance(at, (int, float)):
            continue
        # The journal kind is the beat's FIRST action only — a
        # camera or say can ride a beat whose primary action is
        # a gesture or an expression, so read the beat's own
        # fields, not the kind.
        if isinstance(beat.get("say"), str):
            say_starts.append((float(at), str(beat["say"])))
        if isinstance(beat.get("camera"), str):
            shot_starts.append((float(at), str(beat["camera"])))
    say_starts.sort()
    shot_starts.sort()
    captions = []
    for i, (at, text) in enumerate(say_starts):
        next_at = (say_starts[i + 1][0] if i + 1 < len(say_starts)
                   else duration_s)
        captions.append({"from_s": round(at, 3),
                         "to_s": round(next_at, 3), "text": text})
    shots = []
    for i, (at, shot) in enumerate(shot_starts):
        next_at = (shot_starts[i + 1][0] if i + 1 < len(shot_starts)
                   else duration_s)
        shots.append({"from_s": round(at, 3),
                      "to_s": round(next_at, 3), "shot": shot})
    return captions, shots, partial


def export_web(show: Any, out_dir: str | Path, *,
               ticks: int = 30, max_frames: int = MAX_EXPORT_FRAMES,
               width: int | None = None, height: int | None = None,
               title: str | None = None,
               overwrite: bool = False) -> dict[str, Any]:
    """Export a show as a hosted demo — a static site directory.

    Runs the show for ``ticks`` when it has driven no poses yet
    (a show that has poses exports those instead — the export
    never re-runs a show that already played). The last
    ``max_frames`` posed frames render to PNGs in ``data/``;
    the manifest, captions and shots land in ``data/show.json``;
    the player page lands at the root.
    """
    if not isinstance(ticks, int) or ticks < 1 or ticks > 3600:
        raise ShowError(
            f"show web export needs ticks 1..3600, got {ticks}",
            kind="bad_run")
    if not isinstance(max_frames, int) or max_frames < 1 \
            or max_frames > 900:
        raise ShowError(
            f"show web export needs max_frames 1..900, got "
            f"{max_frames}", kind="bad_run")
    target = Path(out_dir).expanduser()
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise ShowError(
            f"show web export refuses non-empty {target} — pick "
            f"an empty directory or pass overwrite=True",
            kind="out_not_empty")
    try:
        rig = show.resource("rig")
    except ShowError as exc:
        raise ShowError(
            f"show {show.name!r} has no rig — attach one before "
            f"exporting it as a web demo",
            kind="no_rig") from exc
    out_w = int(width) if width is not None else int(rig.width)
    out_h = int(height) if height is not None else int(rig.height)
    if (out_w < MIN_EXPORT_SIZE or out_w > MAX_EXPORT_SIZE
            or out_h < MIN_EXPORT_SIZE
            or out_h > MAX_EXPORT_SIZE):
        raise ShowError(
            f"show web frame size must be {MIN_EXPORT_SIZE}.."
            f"{MAX_EXPORT_SIZE} per edge, got {out_w}x{out_h}",
            kind="bad_size")
    if not show._posed:
        show.run(ticks)
    if not show._posed:
        raise ShowError(
            f"show {show.name!r} has driven no poses yet — "
            f"start it and tick first",
            kind="no_poses")
    Image = _require_pil()
    from ..rig.render import render_pose

    posed = list(show._posed[-max_frames:])
    dropped = len(show._posed) - len(posed)
    fps = float(show.fps)
    duration_s = len(posed) / fps
    captions, shots, partial = _overlays(show, duration_s)
    data_dir = target / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for index, pose in enumerate(posed):
        rgb = render_pose(rig.rig, pose, width=out_w, height=out_h)
        name = f"frame-{index:04d}.png"
        Image.frombytes("RGB", (out_w, out_h), rgb).save(
            data_dir / name, format="PNG")
        files.append(name)
    manifest = {
        "show": show.name,
        "title": (title or show.name),
        "fps": fps,
        "width": out_w,
        "height": out_h,
        "frames": files,
        "frame_count": len(files),
        "frames_dropped": dropped,
        "duration_s": round(duration_s, 3),
        "captions": captions,
        "shots": shots,
        "overlays_partial": partial,
        "events_dropped": show.events_dropped,
        "caption_rule": "a line shows until the next line or "
                        "the end of the clip",
        "shot_rule": "a shot holds until the next camera beat "
                     "or the end of the clip",
    }
    (data_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    title_html = html.escape(str(title or show.name))
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
        "show": show.name,
        "dir": str(target),
        "index": str(target / "index.html"),
        "frames": len(files),
        "frames_dropped": dropped,
        "duration_s": round(duration_s, 3),
        "fps": fps,
        "width": out_w,
        "height": out_h,
        "captions": len(captions),
        "shots": len(shots),
        "overlays_partial": partial,
        "events_dropped": show.events_dropped,
        "files": ["index.html", "main.js", "styles.css",
                  f"data/{_MANIFEST_NAME}"]
                 + [f"data/{name}" for name in files],
    }


# -- serving ---------------------------------------------------------

class DemoServer:
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
        self._httpd = ThreadingHTTPServer((host, port), _DemoHandler)
        self._httpd.demo_root = root  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True,
            name=f"show-demo:{root.name}")
        self._thread.start()

    @property
    def port(self) -> int:
        """The bound port (the ephemeral one, when 0 was asked)."""
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        """The demo's URL — loopback unless host says otherwise."""
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


class _DemoHandler(BaseHTTPRequestHandler):
    """Serve one export directory: index + assets, nothing else.

    No directory listing, no path traversal, no content type
    outside the demo's own files — an unknown path is a 404,
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
            resolved = (self.server.demo_root / rel).resolve()
        except (ValueError, RuntimeError):
            self._answer(404, b"not found", "text/plain")
            return
        try:
            resolved.relative_to(self.server.demo_root.resolve())
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


_SERVERS: dict[str, DemoServer] = {}


def serve_demo(out_dir: str | Path, *,
               host: str = "127.0.0.1", port: int = 0) -> dict[str, Any]:
    """Serve an exported demo directory on loopback.

    Returns the server's status (dir, host, port, url). Serving
    the same directory twice returns the server that already
    serves it. ``host="0.0.0.0"`` is an explicit opt-in that
    exposes the demo on the network — the receipt says so.
    """
    root = Path(out_dir).expanduser().resolve()
    if not (root / "index.html").is_file():
        raise ShowError(
            f"serve needs an exported demo directory — "
            f"{root} has no index.html",
            kind="bad_demo")
    if not isinstance(port, int) or port < 0 or port > 65535:
        raise ShowError(
            f"demo port must be 0..65535, got {port}",
            kind="bad_port")
    key = str(root)
    server = _SERVERS.get(key)
    if server is None:
        server = DemoServer(root, host=host, port=port)
        _SERVERS[key] = server
    return server.status()


def stop_demo(out_dir: str | Path) -> dict[str, Any]:
    """Stop serving an exported demo directory."""
    key = str(Path(out_dir).expanduser().resolve())
    server = _SERVERS.pop(key, None)
    if server is None:
        return {"dir": key, "serving": False}
    server.stop()
    return {"dir": key, "serving": False, "stopped": True}


def list_demos() -> list[dict[str, Any]]:
    """The demo servers running now, one per directory."""
    return [server.status() for server in _SERVERS.values()]
