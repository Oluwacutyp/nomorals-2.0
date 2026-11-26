"""§3C.4 cameras wall (ip-security-camera-viewer port, MIT).

The wall is proved against a REAL loopback HTTP server serving
real JPEGs (PIL-made), plus a slow camera that sleeps 8 s to
prove the rest of the wall never stalls for it. No fake frames,
no fake success: an unreachable camera refuses by name.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from nomorals.media.camwall import (
    Camera,
    CameraUrl,
    CamWallError,
    RequestQueue,
    drop_wall,
    dump_config,
    get_wall,
    list_walls,
    load_config,
    load_wall,
)
from nomorals.media.camwall.config import camera_from_dict
from nomorals.media.camwall.fetch import (
    MAX_FRAME_BYTES,
    MJPEGReader,
    fetch_mjpeg_frame,
    fetch_snapshot,
    is_jpeg,
)

# -- real JPEGs ---------------------------------------------------------

def _jpeg(color: str, size: tuple[int, int] = (64, 48)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


RED = _jpeg("red")
BLUE = _jpeg("blue", (80, 60))
assert is_jpeg(RED) and is_jpeg(BLUE) and RED != BLUE

_BOUNDARY = "camwall-test-boundary"


class _CamHandler(BaseHTTPRequestHandler):
    """One handler, many routes — the wall's cameras, live."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102 - quiet
        pass

    def _send(self, body: bytes, mime: str, code: int = 200,
              extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - stdlib name
        if self.path.startswith("/front.jpg"):
            self._send(RED, "image/jpeg")
        elif self.path.startswith("/back.jpg"):
            self._send(BLUE, "image/jpeg")
        elif self.path.startswith("/slow.jpg"):
            time.sleep(8)
            self._send(RED, "image/jpeg")
        elif self.path.startswith("/text.txt"):
            self._send(b"this is not a picture", "text/plain")
        elif self.path.startswith("/redir.jpg"):
            self._send(b"", "image/jpeg", code=302,
                       extra={"Location": "/front.jpg"})
        elif self.path.startswith("/big.jpg"):
            self._send(b"\xff\xd8\xff\xe0" + b"\x00" * MAX_FRAME_BYTES,
                       "image/jpeg")
        elif self.path.startswith("/auth.jpg"):
            want = "Basic " + base64.b64encode(
                b"cam:secret").decode()
            if self.headers.get("Authorization") != want:
                self._send(b"no", "text/plain", code=401)
            else:
                self._send(RED, "image/jpeg")
        elif self.path.startswith("/mjpeg"):
            body = b""
            for frame in (RED, BLUE):
                body += (
                    f"--{_BOUNDARY}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n"
                ).encode() + frame + b"\r\n"
            body += f"--{_BOUNDARY}--\r\n".encode()
            self._send(body,
                       f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        else:
            self._send(b"not found", "text/plain", code=404)


@pytest.fixture(scope="module")
def server():
    """The wall's cameras, on loopback, for the whole module."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _CamHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True,
                              name="camwall-test-server")
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def wall_name():
    name = f"test-{time.time_ns()}"
    yield name
    with contextlib.suppress(CamWallError):
        get_wall(name).stop()
    with contextlib.suppress(CamWallError):
        drop_wall(name)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# -- config --------------------------------------------------------------

class TestConfig:
    def test_example_json_shape(self):
        # The clone's example.json shape: remote/local hosts + image.
        cams = load_config({"cameras": [
            {"name": "Front Door", "location": "Porch",
             "vendor": "Axis", "type": "IP Camera",
             "remote-host": "cam.example.com",
             "local-host": "192.168.1.10",
             "image": "/jpg/image.jpg"},
            {"name": "Back Yard", "location": "Yard",
             "full-url": "http://192.168.1.11/snap.jpg"},
        ]})
        assert len(cams) == 2
        assert cams[0].snapshot_url() == \
            "http://cam.example.com/jpg/image.jpg"
        assert cams[0].snapshot_url(local=True) == \
            "http://192.168.1.10/jpg/image.jpg"
        assert cams[1].snapshot_url() == \
            "http://192.168.1.11/snap.jpg"

    def test_full_url_and_host_path_forms(self):
        cams = load_config({"cameras": [
            {"name": "A", "full-url": "https://cam.local:8443/a.jpg"},
            {"name": "B", "remote-host": "cam.local:8080",
             "image": "b.jpg"},
        ]})
        assert cams[0].full_url == "https://cam.local:8443/a.jpg"
        assert cams[1].snapshot_url() == "http://cam.local:8080/b.jpg"

    def test_auth_fields(self):
        cams = load_config({"cameras": [
            {"name": "A", "full-url": "http://cam.local/a.jpg",
             "username": "cam", "password": "secret",
             "timeout_s": 3.5, "refresh_s": 2.0},
        ]})
        assert cams[0].username == "cam"
        assert cams[0].password == "secret"
        assert cams[0].timeout_s == 3.5
        assert cams[0].refresh_s == 2.0

    def test_load_from_file_and_json_string(self, tmp_path):
        doc = {"cameras": [
            {"name": "A", "full-url": "http://cam.local/a.jpg"}]}
        path = tmp_path / "cameras.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert len(load_config(path)) == 1
        assert len(load_config(str(path))) == 1
        assert len(load_config(json.dumps(doc))) == 1

    def test_dump_config_round_trips(self, tmp_path):
        cams = load_config({"cameras": [
            {"name": "A", "location": "Gate",
             "remote-host": "cam.local", "local-host": "10.0.0.2",
             "image": "/a.jpg", "username": "cam",
             "password": "secret"},
        ]})
        path = tmp_path / "cameras.json"
        written = dump_config(cams, path)
        assert written == str(path)
        again = load_config(path)
        assert again[0].name == "A"
        assert again[0].snapshot_url(local=True) == \
            "http://10.0.0.2/a.jpg"
        assert again[0].username == "cam"

    def test_bad_configs_are_named(self):
        with pytest.raises(CamWallError) as exc:
            load_config({"cameras": [
                {"name": "A", "full-url": "http://a.local/a.jpg",
                 "remote-host": "b.local", "image": "/b.jpg"}]})
        assert exc.value.kind == "bad_url"  # one URL form, not both
        with pytest.raises(CamWallError) as exc:
            load_config({"cameras": [{"name": "A"}]})
        assert exc.value.kind == "bad_url"  # a camera with no URL form
        with pytest.raises(CamWallError) as exc:
            load_config({"cameras": [
                {"name": "A", "full-url": "ftp://a.local/a.jpg"}]})
        assert exc.value.kind == "bad_url"
        with pytest.raises(CamWallError) as exc:
            load_config({"cameras": []})
        assert exc.value.kind == "no_cameras"
        with pytest.raises(CamWallError) as exc:
            load_config({"nope": True})
        assert exc.value.kind == "bad_config"
        with pytest.raises(CamWallError) as exc:
            load_config("/definitely/not/here.json")
        assert exc.value.kind == "bad_config"


# -- CameraUrl ------------------------------------------------------------

class TestCameraUrl:
    def test_cache_busting_timestamp(self):
        url = CameraUrl(camera_from_dict(
            {"name": "A", "full-url": "http://cam.local/a.jpg"}))
        assert url.get_image_url(now_ms=123) == \
            "http://cam.local/a.jpg?t=123"
        url2 = CameraUrl(camera_from_dict(
            {"name": "A", "full-url": "http://cam.local/a.jpg?x=1"}))
        assert url2.get_image_url(now_ms=7) == \
            "http://cam.local/a.jpg?x=1&t=7"

    def test_local_remote_switch(self):
        url = CameraUrl(camera_from_dict(
            {"name": "A", "remote-host": "cam.example.com",
             "local-host": "192.168.1.10",
             "image": "/jpg/image.jpg"}))
        assert url.base_url() == "http://cam.example.com/jpg/image.jpg"
        url.set_local()
        assert url.base_url() == "http://192.168.1.10/jpg/image.jpg"
        url.set_remote()
        assert url.base_url() == "http://cam.example.com/jpg/image.jpg"

    def test_full_url_ignores_local_remote_switch(self):
        # A full-url camera has one URL; the local/remote switch
        # leaves it alone.
        url = CameraUrl(camera_from_dict(
            {"name": "A", "full-url": "http://cam.local/a.jpg"}))
        url.set_local()
        assert url.base_url() == "http://cam.local/a.jpg"
        url.set_remote()
        assert url.base_url() == "http://cam.local/a.jpg"


# -- queue -----------------------------------------------------------------

class TestQueue:
    def test_max_length_drops_oldest(self):
        queue = RequestQueue(max_length=4)
        for n in range(5):
            queue.request(f"http://x/{n}.jpg", now=1000.0 + n)
        assert queue.percent_full() == 1.0
        assert queue.oldest() == 1001.0  # the first was dropped
        assert queue.newest() == 1004.0
        assert len(queue.queue) == 4

    def test_purge_old_and_stale(self):
        queue = RequestQueue()
        queue.request("http://x/1.jpg", now=1000.0)
        t2 = queue.request("http://x/2.jpg", now=1001.0)
        queue.resolve(t2, "loaded")
        assert queue.is_stale(1001.0) is True
        assert queue.is_stale(1002.0) is False
        # Purge drops every entry at/before the last loaded one,
        # the loaded entry included (the port says so).
        assert queue.purge_old() == 2
        assert queue.newest() is None

    def test_camera_ignores_stale_frames(self):
        camera = Camera(camera_from_dict(
            {"name": "A", "full-url": "http://x/a.jpg"}))
        camera.note_request("http://x/a.jpg?t=1", now=1000.0)
        camera.note_request("http://x/a.jpg?t=2", now=1001.0)
        assert camera.note_loaded(RED, 1001.0, now=1002.0) is True
        assert camera.counts["load"] == 1
        # An older frame arriving late is refused, never shown.
        assert camera.note_loaded(BLUE, 1000.0, now=1003.0) is False
        assert camera.counts["load"] == 1
        assert camera.cur_frame == RED

    def test_camera_fps(self):
        camera = Camera(camera_from_dict(
            {"name": "A", "full-url": "http://x/a.jpg"}))
        camera.note_request("http://x/a.jpg", now=1000.0)
        camera.note_loaded(RED, 1000.0, now=1001.0)
        camera.note_request("http://x/a.jpg", now=1002.0)
        camera.note_loaded(RED, 1002.0, now=1003.0)
        assert camera.fps(now=1004.0) == pytest.approx(2 / 4, abs=0.01)


# -- fetch -----------------------------------------------------------------

class TestFetch:
    def test_snapshot_from_real_server(self, server):
        frame = fetch_snapshot(f"{server}/front.jpg", timeout_s=5)
        assert frame == RED

    def test_snapshot_rejects_non_jpeg(self, server):
        with pytest.raises(CamWallError) as exc:
            fetch_snapshot(f"{server}/text.txt", timeout_s=5)
        assert exc.value.kind == "bad_image"

    def test_snapshot_refuses_redirect(self, server):
        with pytest.raises(CamWallError) as exc:
            fetch_snapshot(f"{server}/redir.jpg", timeout_s=5)
        assert exc.value.kind == "bad_status"

    def test_snapshot_refuses_oversize_frame(self, server):
        with pytest.raises(CamWallError) as exc:
            fetch_snapshot(f"{server}/big.jpg", timeout_s=10)
        assert exc.value.kind == "frame_too_large"

    def test_snapshot_unreachable_is_named(self):
        with pytest.raises(CamWallError) as exc:
            fetch_snapshot(f"http://127.0.0.1:{_free_port()}/a.jpg",
                           timeout_s=2)
        assert exc.value.kind == "camera_unreachable"

    def test_auth_snapshot(self, server):
        frame = fetch_snapshot(f"{server}/auth.jpg", timeout_s=5,
                               username="cam", password="secret")
        assert frame == RED
        with pytest.raises(CamWallError) as exc:
            fetch_snapshot(f"{server}/auth.jpg", timeout_s=5)
        assert exc.value.kind == "bad_status"

    def test_mjpeg_reader_from_recorded_fixture(self):
        body = b""
        for frame in (RED, BLUE):
            body += (
                f"--{_BOUNDARY}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(frame)}\r\n\r\n"
            ).encode() + frame + b"\r\n"
        body += f"--{_BOUNDARY}--\r\n".encode()
        reader = MJPEGReader()
        got = reader.feed(body)
        assert got[0] == RED
        assert got[1] == BLUE

    def test_mjpeg_frame_from_real_server(self, server):
        frame = fetch_mjpeg_frame(f"{server}/mjpeg", timeout_s=5)
        assert frame == RED


# -- wall -------------------------------------------------------------------

class TestWall:
    def _config(self, server, slow: bool = False) -> dict:
        cams = [
            {"name": "Front", "location": "Gate",
             "full-url": f"{server}/front.jpg"},
            {"name": "Back", "location": "Yard",
             "full-url": f"{server}/back.jpg"},
        ]
        if slow:
            cams.append({"name": "Slow", "location": "Drive",
                         "full-url": f"{server}/slow.jpg",
                         "timeout_s": 15})
        return {"cameras": cams}

    def test_wall_ticks_against_real_server(self, server, wall_name):
        wall = load_wall(wall_name, self._config(server))
        assert wall.status()["cameras"] == 2
        wall.tick(wait=True)
        status = wall.status()
        for cam in status["camera_list"]:
            assert cam["has_frame"] is True
            assert cam["counts"]["load"] >= 1
        assert wall.snapshot("Front") == RED
        assert wall.snapshot("Back") == BLUE

    def test_slow_camera_never_stalls_the_rest(self, server, wall_name):
        wall = load_wall(wall_name, self._config(server, slow=True))
        wall.tick(wait=True, timeout=2)  # the slow fetch takes 8 s
        status = wall.status()
        by_name = {c["name"]: c for c in status["camera_list"]}
        assert by_name["Front"]["has_frame"] is True
        assert by_name["Back"]["has_frame"] is True
        assert by_name["Slow"]["has_frame"] is False
        deadline = time.time() + 12
        while time.time() < deadline:
            if get_wall(wall_name).status()["camera_list"][2][
                    "has_frame"]:
                break
            time.sleep(0.25)
        slow = get_wall(wall_name).status()["camera_list"][2]
        assert slow["has_frame"] is True
        assert wall.snapshot("Slow") == RED
        # The fast cameras loaded long before the slow one finished.
        assert by_name["Front"]["times"]["last_loaded"] < \
            slow["times"]["last_loaded"]

    def test_wall_loop_runs_and_stops(self, server, wall_name):
        wall = load_wall(wall_name, self._config(server))
        started = wall.start()
        assert started["running"] is True
        deadline = time.time() + 10
        while time.time() < deadline:
            cams = wall.status()["camera_list"]
            if all(c["has_frame"] for c in cams):
                break
            time.sleep(0.25)
        assert all(c["has_frame"]
                   for c in wall.status()["camera_list"])
        stopped = wall.stop()
        assert stopped["running"] is False

    def test_pause_resume_focus_wall(self, server, wall_name):
        wall = load_wall(wall_name, self._config(server))
        wall.tick(wait=True)
        wall.pause("Front")
        assert wall.status()["camera_list"][0]["paused"] is True
        wall.focus("Back")
        status = wall.status()
        assert status["camera_list"][0]["paused"] is True
        assert status["camera_list"][1]["paused"] is False
        wall.multiview()
        assert all(not c["paused"]
                   for c in wall.status()["camera_list"])
        wall.resume("Front")
        assert wall.status()["camera_list"][0]["paused"] is False

    def test_server_and_rows(self, server, wall_name):
        wall = load_wall(wall_name, self._config(server), rows=2)
        wall.set_server("local")
        assert wall.status()["server"] == "local"
        wall.set_rows(4)
        assert wall.status()["rows"] == 4
        with pytest.raises(CamWallError) as exc:
            wall.set_rows(9)
        assert exc.value.kind == "bad_run"
        with pytest.raises(CamWallError) as exc:
            wall.set_server("sideways")
        assert exc.value.kind == "bad_config"

    def test_add_camera_and_unknown_camera(self, server, wall_name):
        wall = load_wall(wall_name, self._config(server))
        added = wall.add_camera(
            {"name": "Side", "location": "Alley",
             "full-url": f"{server}/front.jpg"})
        assert added.id == 2
        assert wall.status()["cameras"] == 3
        with pytest.raises(CamWallError) as exc:
            wall.pause("No Such Camera")
        assert exc.value.kind == "unknown_camera"
        with pytest.raises(CamWallError) as exc:
            wall.snapshot("No Such Camera")
        assert exc.value.kind == "unknown_camera"

    def test_snapshot_without_frame_is_named(self, server, wall_name):
        wall = load_wall(wall_name, self._config(server))
        with pytest.raises(CamWallError) as exc:
            wall.snapshot("Front")
        assert exc.value.kind == "no_frame"

    def test_unreachable_camera_refuses_by_name(self, wall_name):
        wall = load_wall(wall_name, {"cameras": [
            {"name": "Dead", "location": "Nowhere",
             "full-url": f"http://127.0.0.1:{_free_port()}/a.jpg",
             "timeout_s": 2},
        ]})
        wall.tick(wait=True)
        cam = wall.status()["camera_list"][0]
        assert cam["has_frame"] is False
        assert cam["counts"]["fail"] == 1
        assert cam["last_error"] == "camera_unreachable"

    def test_wall_registry(self, server, wall_name):
        load_wall(wall_name, self._config(server))
        names = [w["wall"] for w in list_walls()]
        assert wall_name in names
        with pytest.raises(CamWallError) as exc:
            get_wall("no-such-wall")
        assert exc.value.kind == "no_wall"
        assert drop_wall(wall_name)["dropped"] is True
        assert wall_name not in [w["wall"] for w in list_walls()]


# -- web export + serving ----------------------------------------------------

class TestWeb:
    def test_export_and_serve(self, server, wall_name, tmp_path):
        wall = load_wall(wall_name, {
            "cameras": [
                {"name": "Front", "location": "Gate",
                 "full-url": f"{server}/front.jpg"},
                {"name": "Back", "location": "Yard",
                 "full-url": f"{server}/back.jpg"},
            ]})
        wall.tick(wait=True)
        out = tmp_path / "wall"
        import urllib.error
        import urllib.request

        from nomorals.media.camwall import export_web, serve_wall, stop_wall

        result = export_web(wall, out, title="Home wall")
        assert result["cameras"] == 2
        assert result["frames"] == 2
        assert (out / "index.html").is_file()
        assert (out / "main.js").is_file()
        assert (out / "styles.css").is_file()
        manifest = json.loads((out / "data" / "wall.json").read_text())
        assert manifest["wall"] == wall_name
        assert manifest["cameras"][0]["url"] == f"{server}/front.jpg"
        assert (out / "data" / "frame-0.jpg").read_bytes() == RED
        page_js = (out / "main.js").read_text()
        assert f"{server}/front.jpg" in page_js  # live refresh URLs
        assert "800" in page_js  # the clone's 800 ms loop
        assert "singleview" in page_js
        assert "Home wall" in (out / "index.html").read_text()

        served = serve_wall(out)
        try:
            with urllib.request.urlopen(served["url"], timeout=5) as resp:
                assert resp.status == 200
                assert "Home wall" in resp.read().decode()
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(
                    served["url"] + "../secret", timeout=5)
        finally:
            stop_wall(out)

    def test_export_refuses_non_empty(self, server, wall_name, tmp_path):
        from nomorals.media.camwall import export_web

        wall = load_wall(wall_name, {
            "cameras": [{"name": "A",
                         "full-url": f"{server}/front.jpg"}]})
        out = tmp_path / "wall"
        out.mkdir()
        (out / "keep.txt").write_text("keep", encoding="utf-8")
        with pytest.raises(CamWallError) as exc:
            export_web(wall, out)
        assert exc.value.kind == "out_not_empty"
        assert export_web(wall, out, overwrite=True)["cameras"] == 1


# -- tool ----------------------------------------------------------------------

class TestTool:
    def test_tool_load_status_tick_snapshot(self, server, wall_name,
                                           tmp_path):
        from nomorals.tools.camwall import camwall_sync

        result = camwall_sync(
            action="load", node=wall_name,
            params={"config": {"cameras": [
                {"name": "Front", "location": "Gate",
                 "full-url": f"{server}/front.jpg"}]}})
        assert result["ok"] is True
        assert result["cameras"] == 1
        result = camwall_sync(action="tick", node=wall_name)
        assert result["camera_list"][0]["has_frame"] is True
        shot = tmp_path / "front.jpg"
        result = camwall_sync(action="snapshot", node=wall_name,
                              value="Front", output=str(shot))
        assert result["ok"] is True
        assert shot.read_bytes() == RED
        result = camwall_sync(action="status", node=wall_name)
        assert result["camera_list"][0]["counts"]["load"] >= 1

    def test_tool_unknown_action_is_named(self):
        from nomorals.tools.camwall import camwall_sync

        with pytest.raises(ValueError, match="unknown camwall action"):
            camwall_sync(action="explode")

    def test_tool_bad_config_is_named(self, wall_name):
        from nomorals.tools.camwall import camwall_sync

        result = camwall_sync(
            action="load", node=wall_name,
            params={"config": {"cameras": [
                {"name": "A", "full-url": "ftp://x/a.jpg"}]}})
        assert result["ok"] is False
        assert result["kind"] == "bad_url"


# -- CLI + chat ------------------------------------------------------------------

class TestCliAndChat:
    def test_cli_parser(self):
        from nomorals.cmdline.parser import _parser

        ns = _parser().parse_args(["camwall", "load", "cfg.json"])
        assert ns.camwall_action == "load"
        assert ns.config == "cfg.json"
        ns = _parser().parse_args(
            ["camwall", "pause", "--camera", "Front"])
        assert ns.camwall_action == "pause"
        assert ns.camera == "Front"
        ns = _parser().parse_args(["cw", "status", "--wall", "home"])
        assert ns.camwall_action == "status"
        assert ns.wall == "home"

    def test_cli_command_runs(self, server, wall_name, capsys, tmp_path):
        from nomorals.cmdline.commands.camwall import _cmd_camwall
        from nomorals.cmdline.parser import _parser

        config = tmp_path / "cameras.json"
        config.write_text(json.dumps({"cameras": [
            {"name": "Front", "location": "Gate",
             "full-url": f"{server}/front.jpg"}]}), encoding="utf-8")
        args = _parser().parse_args(
            ["camwall", "load", str(config), "--wall", wall_name])
        assert _cmd_camwall(args, None) == 0
        out = capsys.readouterr().out
        assert "Front" in out
        args = _parser().parse_args(
            ["camwall", "status", "--wall", wall_name])
        assert _cmd_camwall(args, None) == 0
        assert "Front" in capsys.readouterr().out

    def test_chat_catalog(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_ONELINERS,
        )

        assert "camwall" in CONTROL_COMMANDS
        assert CONTROL_COMMANDS["camwall"][0] == 0  # owner only
        assert "camwall" in COMMAND_DETAILS
        assert "camwall" in LIST_ONELINERS

    def test_chat_handler(self, server, wall_name, tmp_path):
        from nomorals.agents.partner.runtime_camwall import (
            RuntimeCamwallMixin,
        )

        config = tmp_path / "cameras.json"
        config.write_text(json.dumps({"cameras": [
            {"name": "Front", "location": "Gate",
             "full-url": f"{server}/front.jpg"}]}), encoding="utf-8")
        mixin = RuntimeCamwallMixin()
        found = mixin._control_camwall(f"load {config}", "test-key")
        assert "Front" in found
        found = mixin._control_camwall("list", "test-key")
        assert "default" in found
        found = mixin._control_camwall("", "test-key")
        assert "/camwall load" in found


# -- tool count -------------------------------------------------------------------

class TestToolCount:
    def test_tool_count_is_786_after_c4_added_camwall(self):
        # 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet. 808 after this lane (F3) added control. 809 after this lane (moneyprinter) added make_short.
        from nomorals.tools.registry import ToolRegistry

        count = len(ToolRegistry(SimpleNamespace()).register_builtins()
                    ._tools)
        assert count == 809
