"""Show web export — the hosted demo (census §3B.28).

The pattern (storytold/hanashi-graphics-demo, pattern only):
a compiled game-engine demo committed as a deployable static
site — ``index.html`` (a Start button and a canvas host page),
``main.js``, ``styles.css``, ``pkg/`` (the compiled engine),
``data/`` (assets). The artefact *is* the evaluation.

Devon has no compiled engine to ship, and needs none: her
show already renders real frames. The hosted demo is the same
shape over her own renderer — :func:`export_web` runs a show
and writes the static site (index + player + styles + PNG
frames + manifest), the act's lines as captions and its
camera cuts as shots, and :func:`serve_demo` serves one
export directory on loopback. The page must actually PLAY
the show in a headless browser — never a screenshot of a
placeholder.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from pathlib import Path

import pytest

from nomorals.media.mocap import MocapStateStore
from nomorals.media.rig import default_remap, default_rig
from nomorals.media.rig.session import RigSession
from nomorals.media.show import (
    Show,
    export_web,
    list_demos,
    serve_demo,
    stop_demo,
)
from nomorals.media.show.errors import ShowError
from nomorals.media.vtuber.act import Act
from nomorals.media.vtuber.character import Character
from nomorals.media.vtuber.plugin import attach_performance
from nomorals.tools.registry import ToolRegistry

# -- fixtures ---------------------------------------------------------

RIG_W, RIG_H = 160, 90


def _rig() -> RigSession:
    return RigSession(default_rig(), default_remap("bvh"),
                      width=RIG_W, height=RIG_H, fps=15.0)


def _show(name: str = "demo") -> Show:
    return Show(name, MocapStateStore(), rig=_rig())


def _character() -> Character:
    # The exact doc from tests/test_gapfix_vtuber.py — blendshape
    # and bone names are validated against the rig.
    return Character.from_dict({
        "name": "devon",
        "rig": "default",
        "expressions": {
            "smile": {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8,
                      "cheekSquintLeft": 0.4, "cheekSquintRight": 0.4},
            "laugh": {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0,
                      "jawOpen": 0.6, "eyeSquintLeft": 0.5,
                      "eyeSquintRight": 0.5},
        },
        "gestures": {
            "wave": {"shoulder_R": (0.0, 0.0, -120.0),
                     "upper_arm_R": (0.0, 0.0, -30.0),
                     "forearm_R": (0.0, 0.0, -20.0)},
            "bow": {"spine": (25.0, 0.0, 0.0),
                    "spine1": (15.0, 0.0, 0.0),
                    "neck": (-10.0, 0.0, 0.0)},
        },
    }, default_rig())


def _act() -> Act:
    return Act.from_dict({
        "name": "opener",
        "character": "devon",
        "fps": 15.0,
        "beats": [
            {"at": 0.0, "say": "Hello everyone",
             "expression": "smile"},
            {"at": 1.0, "gesture": "wave", "camera": "close"},
            {"at": 2.0, "expression": "laugh", "camera": "face"},
        ],
    })


def _perform(tmp_path: Path, ticks: int = 52) -> Show:
    show = _show()
    attach_performance(show, _act(), _character())
    show.start()
    show.run(ticks)
    return show


def _manifest(out: Path) -> dict:
    return json.loads((out / "data" / "show.json").read_text(
        encoding="utf-8"))


def _headless_shell() -> str:
    """The headless Chromium on this box, or skip the play test.

    Playwright is not installed here; the Chromium binaries
    it left behind ARE — the headless shell runs JS.
    """
    root = Path.home() / ".cache" / "ms-playwright"
    for glob in ("chromium_headless_shell-*/chrome-headless-shell-*"
                 "/chrome-headless-shell",
                 "chromium-*/chrome-linux64/chrome"):
        for path in sorted(root.glob(glob)):
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
    pytest.skip("no headless Chromium on this box")


def _chrome_dump(binary: str, url: str, *, virtual_time_ms: int) -> str:
    result = subprocess.run(
        [binary, "--headless", "--disable-gpu", "--no-sandbox",
         f"--virtual-time-budget={virtual_time_ms}",
         "--dump-dom", url],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


# -- the export -------------------------------------------------------

def test_export_writes_the_site(tmp_path: Path) -> None:
    show = _perform(tmp_path)
    out = tmp_path / "demo"
    result = export_web(show, out)
    assert result["frames"] == 53
    assert result["duration_s"] == pytest.approx(53 / 15.0, abs=0.001)
    assert result["frames_dropped"] == 0
    assert result["captions"] == 1
    assert result["shots"] == 2
    assert result["overlays_partial"] is False
    assert (out / "index.html").is_file()
    assert (out / "main.js").is_file()
    assert (out / "styles.css").is_file()
    assert (out / "data" / "show.json").is_file()
    pngs = sorted((out / "data").glob("frame-*.png"))
    assert len(pngs) == 53
    assert pngs[0].name == "frame-0000.png"
    assert pngs[-1].name == "frame-0052.png"
    assert all(p.stat().st_size > 0 for p in pngs)
    assert "Hello everyone" in (out / "main.js").read_text(
        encoding="utf-8")


def test_export_manifest_overlays(tmp_path: Path) -> None:
    show = _perform(tmp_path)
    out = tmp_path / "demo"
    export_web(show, out)
    manifest = _manifest(out)
    assert manifest["show"] == "demo"
    assert manifest["fps"] == 15.0
    assert manifest["width"] == RIG_W
    assert manifest["height"] == RIG_H
    assert manifest["frame_count"] == 53
    assert manifest["frames"] == [f"frame-{n:04d}.png"
                                  for n in range(53)]
    # One say-line, from its beat until the end of the clip.
    assert manifest["captions"] == [
        {"from_s": 0.0, "to_s": pytest.approx(53 / 15.0, abs=0.001),
         "text": "Hello everyone"}]
    # The camera is the beat's SECONDARY action — its journal
    # kind is gesture/expression, so the export must read the
    # beat's own camera field, not the kind.
    assert manifest["shots"] == [
        {"from_s": 1.0, "to_s": 2.0, "shot": "close"},
        {"from_s": 2.0, "to_s": pytest.approx(53 / 15.0, abs=0.001),
         "shot": "face"}]
    assert manifest["caption_rule"].startswith("a line shows")
    assert manifest["shot_rule"].startswith("a shot holds")
    assert manifest["events_dropped"] == 0


def test_export_frames_are_real_renders(tmp_path: Path) -> None:
    """The frames are rendered poses, not copies of one image.

    The act waves at 1s and laughs at 2s — the frame at the
    wave and the frame at the laugh must differ from each other
    and from the opener, as bytes.
    """
    show = _perform(tmp_path)
    out = tmp_path / "demo"
    export_web(show, out)
    data = out / "data"
    opener = (data / "frame-0000.png").read_bytes()
    wave = (data / "frame-0016.png").read_bytes()
    laugh = (data / "frame-0031.png").read_bytes()
    assert wave != opener
    assert laugh != opener
    assert laugh != wave


def test_export_runs_a_show_with_no_poses(tmp_path: Path) -> None:
    show = _show()
    out = tmp_path / "demo"
    result = export_web(show, out, ticks=20)
    assert result["frames"] == 21
    from nomorals.media.show.states import ShowState
    assert show.state is ShowState.LIVE


def test_export_max_frames_caps_and_reports(tmp_path: Path) -> None:
    show = _perform(tmp_path, ticks=200)
    assert len(show._posed) == 201
    out = tmp_path / "demo"
    result = export_web(show, out, max_frames=50)
    assert result["frames"] == 50
    assert result["frames_dropped"] == 151
    assert _manifest(out)["frames_dropped"] == 151


def test_export_refusals(tmp_path: Path) -> None:
    show = _show()
    with pytest.raises(ShowError) as excinfo:
        export_web(show, tmp_path / "demo", ticks=0)
    assert excinfo.value.kind == "bad_run"
    with pytest.raises(ShowError) as excinfo:
        export_web(show, tmp_path / "demo", ticks=20,
                   max_frames=0)
    assert excinfo.value.kind == "bad_run"
    with pytest.raises(ShowError) as excinfo:
        export_web(show, tmp_path / "demo", width=32)
    assert excinfo.value.kind == "bad_size"
    with pytest.raises(ShowError) as excinfo:
        export_web(show, tmp_path / "demo", width=4096)
    assert excinfo.value.kind == "bad_size"
    show = Show("norigname", MocapStateStore())
    with pytest.raises(ShowError) as excinfo:
        export_web(show, tmp_path / "demo")
    assert excinfo.value.kind == "no_rig"
    # A non-empty directory is refused unless overwrite is given.
    show = _perform(tmp_path)
    out = tmp_path / "demo"
    out.mkdir()
    (out / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(ShowError) as excinfo:
        export_web(show, out)
    assert excinfo.value.kind == "out_not_empty"
    result = export_web(show, out, overwrite=True)
    assert result["frames"] == 53
    assert (out / "index.html").is_file()


# -- the page actually plays ------------------------------------------

def test_page_plays_in_headless_browser(tmp_path: Path) -> None:
    """The artefact is the evaluation: the page must play.

    The headless shell runs the player with autoplay on virtual
    time: the early dump must be mid-show (the act's line on
    screen, the frame counter running) and the late dump must
    be the finished show (last frame, face shot, done).
    """
    binary = _headless_shell()
    show = _perform(tmp_path)
    out = tmp_path / "demo"
    export_web(show, out)
    early = _chrome_dump(binary, f"file://{out}/index.html"
                                 "?autoplay=1",
                         virtual_time_ms=1500)
    late = _chrome_dump(binary, f"file://{out}/index.html"
                                "?autoplay=1",
                        virtual_time_ms=20000)
    assert "Show failed to start" not in early
    assert "Hello everyone" in early
    assert "Frame " in early
    assert "Frame 53 / 53" in late
    assert "face" in late
    assert "done" in late
    assert early != late, "the page did not advance between dumps"


def test_page_plays_from_the_server(tmp_path: Path) -> None:
    """The same page, served — the server is the second path."""
    binary = _headless_shell()
    show = _perform(tmp_path)
    out = tmp_path / "demo"
    export_web(show, out)
    status = serve_demo(out)
    try:
        dom = _chrome_dump(binary, f"{status['url']}index.html"
                                   "?autoplay=1",
                           virtual_time_ms=1500)
    finally:
        stop_demo(out)
    assert "Hello everyone" in dom
    assert "Frame " in dom


# -- serving -----------------------------------------------------------

def test_serve_http(tmp_path: Path) -> None:
    show = _perform(tmp_path)
    out = tmp_path / "demo"
    export_web(show, out)
    status = serve_demo(out)
    try:
        assert status["url"].startswith("http://127.0.0.1:")
        assert status["port"] > 0
        assert status["network"] is False
        assert list_demos() == [status]
        with urllib.request.urlopen(status["url"]) as response:
            assert response.status == 200
            assert "text/html" in response.headers["Content-Type"]
            assert b"Start" in response.read()
        with urllib.request.urlopen(
                status["url"] + "main.js") as response:
            assert response.status == 200
            assert "javascript" in response.headers["Content-Type"]
        with urllib.request.urlopen(
                status["url"] + "data/show.json") as response:
            assert response.status == 200
            assert json.loads(response.read())["show"] == "demo"
        with urllib.request.urlopen(
                status["url"] + "data/frame-0000.png") as response:
            assert response.status == 200
            assert "image/png" in response.headers["Content-Type"]
        # Serving the same directory twice returns the same server.
        assert serve_demo(out)["port"] == status["port"]
    finally:
        stop_demo(out)
    assert list_demos() == []
    assert stop_demo(out)["serving"] is False


def test_serve_refuses_traversal_and_unknown(tmp_path: Path) -> None:
    from urllib.error import HTTPError
    show = _perform(tmp_path)
    out = tmp_path / "demo"
    export_web(show, out)
    status = serve_demo(out)
    try:
        for path in ("../web.py", "..%2Fweb.py", "nope.html",
                     "data/nope.png", "web.py"):
            with pytest.raises(HTTPError) as excinfo:
                urllib.request.urlopen(status["url"] + path)
            assert excinfo.value.code == 404
    finally:
        stop_demo(out)


def test_serve_refusals(tmp_path: Path) -> None:
    with pytest.raises(ShowError) as excinfo:
        serve_demo(tmp_path / "nope")
    assert excinfo.value.kind == "bad_demo"
    out = tmp_path / "demo"
    out.mkdir()
    (out / "index.html").write_text("<h1>x</h1>", encoding="utf-8")
    with pytest.raises(ShowError) as excinfo:
        serve_demo(out, port=70000)
    assert excinfo.value.kind == "bad_port"


# -- wiring: tool ------------------------------------------------------

def _tool_registry(tmp_path: Path):
    from types import SimpleNamespace
    registry = ToolRegistry(SimpleNamespace(home=str(tmp_path)))
    registry.register_builtins()
    return registry


def _tool(registry, **kwargs):
    outcome = registry.call("show", **kwargs)
    if getattr(outcome, "ok", False):
        return outcome.value
    raise AssertionError(
        f"show call failed: {getattr(outcome, 'error', outcome)}")


def test_tool_web_action(tmp_path: Path) -> None:
    registry = _tool_registry(tmp_path)
    _tool(registry, action="create", node="demo")
    result = _tool(registry, action="web", node="demo",
                   output=str(tmp_path / "demo"),
                   params={"ticks": 20})
    assert result["frames"] == 21
    assert result["auto_attached"] == ["rig:default", "cloudcam:idle"]
    assert (tmp_path / "demo" / "index.html").is_file()


def test_tool_serve_unserve(tmp_path: Path) -> None:
    registry = _tool_registry(tmp_path)
    _tool(registry, action="create", node="demo")
    _tool(registry, action="web", node="demo",
          output=str(tmp_path / "demo"), params={"ticks": 10})
    status = _tool(registry, action="serve",
                   output=str(tmp_path / "demo"))
    try:
        assert status["url"].startswith("http://127.0.0.1:")
        with urllib.request.urlopen(status["url"]) as response:
            assert response.status == 200
    finally:
        _tool(registry, action="unserve",
              output=str(tmp_path / "demo"))
    assert list_demos() == []


def test_tool_web_needs_output(tmp_path: Path) -> None:
    registry = _tool_registry(tmp_path)
    outcome = registry.call("show", action="web", node="demo")
    assert not outcome.ok
    assert "web needs output" in str(
        getattr(outcome.error, "message", outcome.error))


# -- wiring: CLI -------------------------------------------------------

def _cmd_show(args: list[str], tmp_path: Path) -> int:
    # The vtuber CLI test pattern: a Namespace with every
    # attribute _cmd_show reads, not the real parser.
    import argparse

    from nomorals.cmdline.commands.show import _cmd_show

    def build(action: str, **kwargs):
        base = {"show_action": action, "session": "demo",
                "json": False, "quiet": False, "home": "",
                "out": "", "ticks": 30, "max_frames": 450,
                "width": 0, "height": 0, "title": "",
                "overwrite": False, "host": "127.0.0.1", "port": 0,
                "limit": 0, "interval": 0.0, "path": "", "value": "",
                "sources": [], "rig": "", "take": "", "render_dir": "",
                "cloudcam": "", "cloudcam_source": "", "endpoint": "",
                "export": "", "export_path": "", "fps": 15.0}
        base.update(kwargs)
        return argparse.Namespace(**base)

    action, rest = args[0], args[1:]
    kwargs: dict = {}
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--session":
            kwargs["session"] = rest[index + 1]
            index += 2
        elif token == "--out":
            kwargs["out"] = rest[index + 1]
            index += 2
        elif token == "--ticks":
            kwargs["ticks"] = int(rest[index + 1])
            index += 2
        elif token == "--max-frames":
            kwargs["max_frames"] = int(rest[index + 1])
            index += 2
        elif token == "--overwrite":
            kwargs["overwrite"] = True
            index += 1
        else:
            index += 1
    return _cmd_show(build(action, **kwargs), None)


def test_cli_web(tmp_path: Path) -> None:
    assert _cmd_show(["create", "demo"], tmp_path) == 0
    assert _cmd_show(["web", "--session", "demo", "--out",
                      str(tmp_path / "demo"), "--ticks", "20"],
                     tmp_path) == 0
    assert (tmp_path / "demo" / "index.html").is_file()
    assert _manifest(tmp_path / "demo")["frame_count"] == 21


def test_cli_web_needs_out(tmp_path: Path, capsys) -> None:
    assert _cmd_show(["web", "--session", "demo"], tmp_path) == 2
    assert "needs --out" in capsys.readouterr().err


def test_cli_unserve_not_serving(tmp_path: Path) -> None:
    assert _cmd_show(["unserve", "--out",
                      str(tmp_path / "demo")], tmp_path) == 0


# -- wiring: chat ------------------------------------------------------

def test_chat_web(tmp_path: Path) -> None:
    from nomorals.agents.partner.runtime_mocap import (
        RuntimeMocapMixin,
    )
    mixin = RuntimeMocapMixin()
    assert "created" in mixin._control_live("create demo", "test")
    assert "live" in mixin._control_live("start demo", "test")
    assert "tick" in mixin._control_live("tick demo", "test")
    reply = mixin._control_live(
        f"web demo {tmp_path / 'demo'}", "test")
    assert "hosted demo" in reply
    assert str(tmp_path / "demo" / "index.html") in reply
    assert (tmp_path / "demo" / "index.html").is_file()
    assert mixin._control_live("web demo", "test") == (
        "/live web [name] <dir> [ticks]")
    assert "web" in mixin._control_live("", "test")


# -- registry count ----------------------------------------------------

def test_registry_tool_count(tmp_path: Path) -> None:
    registry = _tool_registry(tmp_path)
    assert len(registry._tools) == 804
