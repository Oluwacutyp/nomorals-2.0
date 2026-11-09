"""The driver protocol against the REAL desktop bridge + Xvfb.

No mocks anywhere in this file: a real Xvfb virtual display (a
real X server), the committed Go bridge binary speaking JSON lines
over stdio, and assertions on what comes back — a probe that
reports the display's real size, a screenshot that decodes to a
PNG of exactly that size, pointer motion verified by asking the
server where the pointer is afterwards.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys

import pytest

from nomorals.computer import provision
from nomorals.computer.protocol import DriverClient, DriverDied, DriverError

pytestmark = pytest.mark.skipif(
    shutil.which("Xvfb") is None, reason="Xvfb not installed on this host")


@pytest.fixture(scope="module")
def bridge_path(tmp_path_factory):
    path = provision.bridge_binary_path()
    if not path.is_file():
        path = provision.build_bridge()
    return path


@pytest.fixture()
def display(tmp_path):
    number = provision.find_free_display()
    auth = tmp_path / "xauth"
    provision.write_xauthority(auth, number)
    server = provision.start_xvfb(number, 1280, 1024, auth)
    yield f":{number}", auth
    provision.stop_process(server.pid)


@pytest.fixture()
def client(bridge_path, display):
    disp, auth = display
    env = dict(os.environ, DISPLAY=disp, XAUTHORITY=str(auth))
    drv = DriverClient([str(bridge_path)], env=env).start()
    yield drv
    drv.close()


class TestProbeAndPerception:
    def test_probe_reports_the_real_display(self, client):
        info = client.probe()
        assert info["available"] is True, info
        assert info["protocol"] == 1
        assert info["width"] == 1280
        assert info["height"] == 1024
        assert "screenshot" in info["features"]

    def test_ping(self, client):
        assert client.ping() is True

    def test_screenshot_is_a_png_of_the_display(self, client):
        import base64

        shot = client.screenshot()
        assert shot["width"] == 1280 and shot["height"] == 1024
        png = base64.b64decode(shot["data"])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", png[16:24])
        assert (width, height) == (1280, 1024)

    def test_windows_starts_empty(self, client):
        wins = client.windows()
        assert wins["windows"] == []

    def test_pointer_move_is_real(self, client):
        moved = client.call("mouse_move", {"x": 321, "y": 234})
        assert (moved["x"], moved["y"]) == (321, 234)
        pointer = client.call("pointer")
        assert (pointer["x"], pointer["y"]) == (321, 234)


class TestClientHonesty:
    def test_unknown_method_refused_client_side(self, client):
        with pytest.raises(DriverError):
            client.call("definitely_not_a_method")

    def test_dead_bridge_fails_the_call(self, tmp_path, display):
        disp, auth = display
        env = dict(os.environ, DISPLAY=disp, XAUTHORITY=str(auth))
        # A "bridge" that exits at once: the call must fail as
        # DriverDied, never hang and never fake an answer.
        drv = DriverClient([sys.executable, "-c", "pass"], env=env).start()
        try:
            with pytest.raises(DriverDied):
                drv.call("ping", timeout=10.0)
        finally:
            drv.close()

    def test_bridge_killed_midflight(self, bridge_path, display):
        disp, auth = display
        env = dict(os.environ, DISPLAY=disp, XAUTHORITY=str(auth))
        drv = DriverClient([str(bridge_path)], env=env).start()
        assert drv.ping() is True
        proc = drv._proc
        assert proc is not None
        proc.kill()
        proc.wait(timeout=5)
        with pytest.raises(DriverError):
            drv.call("ping", timeout=10.0)
        drv.close()


def test_built_bridge_answers_probe(bridge_path, display):
    """The committed binary, spawned fresh, probes its display."""
    disp, auth = display
    env = dict(os.environ, DISPLAY=disp, XAUTHORITY=str(auth))
    drv = DriverClient([str(bridge_path)], env=env).start()
    try:
        info = drv.probe()
        assert info["available"] is True
    finally:
        drv.close()
    # And the binary exits cleanly on stdin EOF (no zombie servers).
    proc = subprocess.run([str(bridge_path)], input=b"",
                          env=env, timeout=10)
    assert proc.returncode == 0
