"""Self-provisioning: bridge discovery/build, Xauthority, Xvfb.

The pairfix contract, tested: what's present is found; what's
missing is built or installed by Devon herself; what cannot be
had is a plain-words ProvisionError — never instructions to the
owner, never a silent stub.
"""

from __future__ import annotations

import os
import shutil
import socket
import struct
import subprocess

import pytest

from nomorals.computer import provision


class TestBridge:
    def test_ensure_bridge_finds_or_builds(self):
        result = provision.ensure_bridge()
        assert result.how in ("found", "built")
        assert result.path.is_file()
        assert os.access(result.path, os.X_OK)

    def test_env_override_missing_is_an_error(self, monkeypatch):
        monkeypatch.setenv("NM_COMPUTER_BRIDGE", "/nonexistent/bridge")
        with pytest.raises(provision.ProvisionError):
            provision.ensure_bridge()

    def test_build_bridge_into_a_directory(self, tmp_path):
        target = tmp_path / ("desktop-bridge.exe"
                             if provision.bridge_binary_path()
                             .name.endswith(".exe")
                             else "desktop-bridge-linux")
        built = provision.build_bridge(into=target)
        assert built.is_file()
        # The built binary runs and exits cleanly on EOF.
        proc = subprocess.run([str(built)], input=b"", timeout=15)
        assert proc.returncode == 0

    def test_find_go(self):
        go = provision.find_go()
        assert go is not None  # the workspace toolchain or PATH
        proc = subprocess.run([go, "version"], capture_output=True,
                              text=True, timeout=30)
        assert proc.returncode == 0
        assert "go version" in proc.stdout


class TestXauthority:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "xauth"
        cookie_hex = provision.write_xauthority(path, 123)
        assert len(bytes.fromhex(cookie_hex)) == 16
        raw = path.read_bytes()

        def read_field(buf, off):
            (ln,) = struct.unpack(">H", buf[off:off + 2])
            return buf[off + 2:off + 2 + ln], off + 2 + ln

        (family,) = struct.unpack(">H", raw[:2])
        assert family == 256  # FamilyLocal
        host, off = read_field(raw, 2)
        assert host.decode() == socket.gethostname()
        number, off = read_field(raw, off)
        assert number == b"123"
        name, off = read_field(raw, off)
        assert name == b"MIT-MAGIC-COOKIE-1"
        data, off = read_field(raw, off)
        assert data.hex() == cookie_hex

    def test_permissions_are_private(self, tmp_path):
        path = tmp_path / "xauth"
        provision.write_xauthority(path, 99)
        assert (path.stat().st_mode & 0o077) == 0


@pytest.mark.skipif(shutil.which("Xvfb") is None,
                    reason="Xvfb not installed on this host")
class TestXvfb:
    def test_ensure_xvfb_binary(self):
        assert provision.ensure_xvfb_binary()

    def test_free_display_skips_taken_numbers(self):
        first = provision.find_free_display()
        socket_path = provision.display_socket_path(first)
        lock = socket_path.parent.parent / f".X{first}-lock"
        made_socket = made_lock = False
        try:
            if not socket_path.exists():
                socket_path.parent.mkdir(parents=True, exist_ok=True)
                socket_path.touch()
                made_socket = True
            if not lock.exists():
                lock.touch()
                made_lock = True
            assert provision.find_free_display() != first
        finally:
            if made_socket:
                socket_path.unlink(missing_ok=True)
            if made_lock:
                lock.unlink(missing_ok=True)

    def test_start_and_stop_a_virtual_display(self, tmp_path):
        number = provision.find_free_display()
        auth = tmp_path / "xauth"
        provision.write_xauthority(auth, number)
        server = provision.start_xvfb(number, 800, 600, auth)
        try:
            assert provision.process_alive(server.pid)
            assert provision._display_accepts(number)
        finally:
            gone = provision.stop_process(server.pid)
        assert gone
        assert not provision.process_alive(server.pid)

    def test_process_alive_is_honest(self):
        assert provision.process_alive(os.getpid()) is True
        assert provision.process_alive(2 ** 30) is False
        assert provision.process_alive(0) is False
