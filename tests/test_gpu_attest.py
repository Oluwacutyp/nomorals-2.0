"""Forwarded-GPU VRAM attestation (register §11(2) follow-up 2).

A forwarded GPU (``NM_GPU_URL``) is declared but never verified:
``detect_gpu()`` reports it with ``vram_gb`` 0.0 and the detail
"not probed — no verification protocol", so every VRAM floor
(LatentSync 18 GB, TRELLIS 24 GB, UniRig 8 GB) refuses it — correctly,
since the floor cannot be verified. This lane adds the attestation
protocol: the endpoint proves its VRAM by measuring its own GPU and
allocating the challenged bytes on CUDA, returning a receipt derived
from the allocated memory; ``detect_gpu()`` reports the attested
number and the floors consult it. A bare ``NM_GPU_URL`` string, a
self-report with no allocation, and a timed-out endpoint all stay
unattested and keep refusing.

The stub endpoint below implements the wire format independently
(pattern = nonce bytes repeated, 8 samples of 8 KiB at eighths of the
buffer, sha256 over the concatenated samples), so these pins prove the
contract, not the module's own helper agreeing with itself. New names
are imported inside the test functions, so this file collects on the
pristine base and fails behaviourally there.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from nomorals.core import gpu as gpu_mod
from nomorals.core.gpu import GpuInfo, detect_gpu

SAMPLE_BYTES = 8192
NSAMPLES = 8


# ── the wire format, implemented independently ───────────────────────

def _wire_offsets(n_bytes: int) -> list[int]:
    return [i * n_bytes // NSAMPLES for i in range(NSAMPLES)]


def _wire_receipt(challenge_hex: str, n_bytes: int) -> str:
    raw = bytes.fromhex(challenge_hex)
    samples = []
    for off in _wire_offsets(n_bytes):
        samples.append(
            bytes(raw[(off + i) % len(raw)] for i in range(SAMPLE_BYTES)))
    return hashlib.sha256(b"".join(samples)).hexdigest()


class _StubState:
    def __init__(self) -> None:
        self.mode = "ok"          # ok | bad-receipt | no-alloc | refused | sleep
        self.vram_gb = 24.0
        self.name = "NVIDIA RTX 4090"
        self.total_bytes = 25769803776
        self.free_before_bytes = 20_000_000_000
        self.requests = 0


def _stub_handler(state: _StubState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            parsed = urlparse(self.path)
            if parsed.path != "/gpu/attest":
                self.send_response(404)
                self.end_headers()
                return
            state.requests += 1
            query = parse_qs(parsed.query)
            challenge = (query.get("challenge") or [""])[0]
            try:
                n_bytes = int((query.get("bytes") or ["0"])[0])
            except ValueError:
                n_bytes = 0
            if state.mode == "sleep":
                time.sleep(2.0)
            if state.mode == "refused":
                payload: dict[str, Any] = {
                    "ok": False, "reason": "no CUDA device on this host"}
            else:
                receipt = _wire_receipt(challenge, n_bytes)
                if state.mode == "bad-receipt":
                    receipt = "00" * 32
                free_after = state.free_before_bytes - n_bytes
                if state.mode == "no-alloc":
                    free_after = state.free_before_bytes
                payload = {
                    "ok": True,
                    "name": state.name,
                    "vram_gb": state.vram_gb,
                    "total_bytes": state.total_bytes,
                    "total_gb": round(state.total_bytes / 1e9, 2),
                    "free_before_bytes": state.free_before_bytes,
                    "free_after_bytes": free_after,
                    "free_before_gb": round(
                        state.free_before_bytes / 1e9, 2),
                    "free_after_gb": round(free_after / 1e9, 2),
                    "allocated_bytes": n_bytes,
                    "receipt": receipt,
                }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            # The timeout test disconnects mid-response on purpose.
            with contextlib.suppress(BrokenPipeError,
                                      ConnectionResetError):
                self.wfile.write(body)

    return Handler


@pytest.fixture()
def stub(monkeypatch, _clean_gpu):
    # _clean_gpu is a declared dependency so its env cleanup runs
    # before this fixture sets NM_GPU_URL (fixture order is not
    # otherwise guaranteed between an autouse fixture and this one).
    state = _StubState()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), _stub_handler(state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "NM_GPU_URL", f"http://127.0.0.1:{server.server_address[1]}")
    yield state
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def _clean_gpu(monkeypatch):
    monkeypatch.delenv("NM_GPU", raising=False)
    monkeypatch.delenv("NM_GPU_URL", raising=False)
    yield
    monkeypatch.undo()
    # Undo the test's probe/env patches FIRST, then restore the cache
    # to the real box — otherwise the restore re-caches the faked GPU.
    if hasattr(gpu_mod, "_GPU_CACHE"):
        gpu_mod._GPU_CACHE = None
    if hasattr(gpu_mod, "_GPU_CACHE_AT"):
        gpu_mod._GPU_CACHE_AT = 0.0
    if hasattr(gpu_mod, "_ATTEST_ATTEMPT_AT"):
        gpu_mod._ATTEST_ATTEMPT_AT = 0.0
    detect_gpu(refresh=True)


def _gpu_url() -> str:
    import os
    return os.environ["NM_GPU_URL"]


# ── the detector ────────────────────────────────────────────────────

def test_forwarded_unattested_when_endpoint_dead(monkeypatch):
    monkeypatch.setenv("NM_GPU_URL", "http://127.0.0.1:1")
    got = detect_gpu(refresh=True)
    assert got.path == "forwarded"
    assert got.has_gpu is True
    assert got.vram_gb == 0.0
    assert getattr(got, "attested", False) is False


def test_attested_endpoint_reports_vram(stub):
    got = detect_gpu(refresh=True)
    assert got.attested is True
    assert got.path == "forwarded"
    assert got.has_gpu is True
    assert got.name == "NVIDIA RTX 4090"
    assert got.vram_gb == 24.0
    assert got.source == "attested"
    assert got.forward_url == _gpu_url()
    assert "attested" in got.detail


def test_attestation_refused_stays_unattested(stub):
    stub.mode = "refused"
    got = detect_gpu(refresh=True)
    assert got.path == "forwarded"
    assert got.vram_gb == 0.0
    assert getattr(got, "attested", False) is False
    assert "not probed" in got.detail


def test_bad_receipt_fails_attestation(stub):
    stub.mode = "bad-receipt"
    got = detect_gpu(refresh=True)
    assert got.vram_gb == 0.0
    assert getattr(got, "attested", False) is False


def test_no_allocation_fails_attestation(stub):
    stub.mode = "no-alloc"
    got = detect_gpu(refresh=True)
    assert got.vram_gb == 0.0
    assert getattr(got, "attested", False) is False


def test_attest_forwarded_timeout(stub):
    from nomorals.core.gpu import attest_forwarded

    stub.mode = "sleep"
    assert attest_forwarded(_gpu_url(), timeout=0.2) is None


def test_attestation_cached_within_ttl(stub, monkeypatch):
    got = detect_gpu(refresh=True)
    assert got.attested is True
    assert stub.requests == 1
    # A second refresh inside the TTL reuses the attestation — the
    # endpoint is not challenged again.
    assert detect_gpu(refresh=True) is got
    assert stub.requests == 1
    # With the TTL expired, a refresh re-attests.
    monkeypatch.setattr(gpu_mod, "ATTEST_TTL_S", 0)
    detect_gpu(refresh=True)
    assert stub.requests == 2


def test_receipt_wire_format() -> None:
    from nomorals.core.gpu import _attest_expected_receipt, _attest_sample_offsets

    challenge = "ab" * 16
    n_bytes = 1 << 30
    assert _attest_sample_offsets(n_bytes) == _wire_offsets(n_bytes)
    assert _attest_expected_receipt(challenge, n_bytes) == (
        _wire_receipt(challenge, n_bytes))


def test_server_probe_reports_no_cuda_honestly() -> None:
    from nomorals.core.gpu import probe_attestation

    # This box has no torch and no nvidia-smi: the probe must say so,
    # never pretend.
    got = probe_attestation("ab" * 16, 1 << 20)
    assert got["ok"] is False
    assert got["reason"]


# ── the floors consult the attested number ──────────────────────────

def test_trellis_gate_passes_attested_24(stub):
    from nomorals.media import trellis

    ok, reason = trellis.gpu_gate(
        {"platform": "linux", "cpus": 4, "total_ram_gb": 16})
    assert ok is True
    assert "forwarded" in reason
    assert "24" in reason


def test_trellis_gate_refuses_attested_8(stub):
    from nomorals.media import trellis

    stub.vram_gb = 8.0
    ok, reason = trellis.gpu_gate(
        {"platform": "linux", "cpus": 4, "total_ram_gb": 16})
    assert ok is False
    assert "8.0" in reason


def test_latentsync_status_reports_attested_vram(stub):
    from nomorals.media.directed import lipsync as ls

    detect_gpu(refresh=True)
    st = ls.latentsync_status()
    assert st["vram_gb"] == 24.0
    # The floor (vram >= 18) reads the attested number — but the
    # driver renders locally, so availability stays a local question:
    # no local CUDA here, so not available.
    assert st["cuda"] is False
    assert st["available"] is False


def test_avatar_latentsync_gate_passes_attested_18(stub):
    from nomorals.media_edit import avatar

    ok, reason = avatar.latentsync_gate(
        {"platform": "linux", "cpus": 4, "total_ram_gb": 16})
    assert ok is True
    assert "forwarded" in reason


def test_unirig_gate_passes_attested_24(stub):
    from nomorals.media import unirig

    ok, reason = unirig.gpu_gate(
        {"platform": "linux", "cpus": 4, "total_ram_gb": 16})
    assert ok is True
    assert "forwarded" in reason


def test_profile_line_marks_attested_forwarded(monkeypatch):
    from nomorals.core import profile as profile_mod
    from nomorals.core.profile_visible import profile_summary

    canned = GpuInfo(path="forwarded", has_gpu=True,
                     name="NVIDIA RTX 4090", vram_gb=24.0,
                     source="attested", attested=True,
                     forward_url="https://gpu.example:8787",
                     detail="forwarded GPU attested at "
                            "https://gpu.example:8787")
    monkeypatch.setattr(profile_mod, "detect_gpu", lambda **kw: canned)
    out = profile_summary(shed="none")
    assert "RTX 4090" in out["line"]
    assert "fwd" in out["line"]


# ── the server-side allocation, with a fake torch ───────────────────

class _FakeTensor:
    """Minimal torch-Tensor surface over a bytearray."""

    def __init__(self, data: bytes | bytearray):
        self._data = bytearray(data)

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self._data))
            vals = (value._data if isinstance(value, _FakeTensor)
                    else bytes(value))
            count = len(range(start, stop, step))
            tiled = bytes(vals[i % len(vals)] for i in range(count))
            self._data[start:stop:step] = tiled
        else:
            self._data[key] = value

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            return _FakeTensor(self._data[key])
        return self._data[key]

    def tolist(self) -> list[int]:
        return list(self._data)

    def cpu(self) -> _FakeTensor:
        return self

    def detach(self) -> _FakeTensor:
        return self

    def contiguous(self) -> _FakeTensor:
        return self


def _fake_torch(total_bytes: int) -> Any:
    """A torch that really allocates (in RAM) and really fills."""
    state = {"allocated": 0}

    def _mem_get_info() -> tuple[int, int]:
        return total_bytes - state["allocated"], total_bytes

    def _empty(n: int, dtype: Any = None, device: Any = None) -> _FakeTensor:
        state["allocated"] = n
        return _FakeTensor(bytearray(n))

    def _tensor(data: Any, dtype: Any = None) -> _FakeTensor:
        return _FakeTensor(bytes(data))

    torch = SimpleNamespace(
        uint8="uint8",
        empty=_empty,
        tensor=_tensor,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=_mem_get_info,
            empty_cache=lambda: None,
        ),
    )
    return torch


def test_allocate_in_process_with_fake_torch(monkeypatch) -> None:
    from nomorals.core.gpu import _attest_allocate_in_process, _attest_expected_receipt

    total = 25769803776
    n_bytes = 1 << 20
    challenge = "cd" * 16
    monkeypatch.setitem(__import__("sys").modules, "torch",
                        _fake_torch(total))
    got = _attest_allocate_in_process(challenge, n_bytes)
    assert got["ok"] is True, got
    assert got["allocated_bytes"] == n_bytes
    assert got["receipt"] == _attest_expected_receipt(challenge, n_bytes)
    assert got["free_before_bytes"] == total
    assert got["free_after_bytes"] == total - n_bytes


def test_allocate_in_process_without_torch(monkeypatch) -> None:
    from nomorals.core.gpu import _attest_allocate_in_process

    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    got = _attest_allocate_in_process("ab" * 16, 1 << 20)
    assert got["ok"] is False
    assert "torch" in got["reason"].lower()


# ── the route, over live HTTP ───────────────────────────────────────

def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _context(db: Any, tmp: str) -> Any:
    from nomorals.tools.registry import ToolRegistry

    settings = SimpleNamespace(
        api=SimpleNamespace(token="", max_body_mb=0, cors_origins=None),
        backup_dir=tmp,
        workspace_dir=tmp,
    )
    return SimpleNamespace(
        db=db, settings=settings, tools=ToolRegistry(), bus=None,
        memory=None)


def _wait_up(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/live")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"server did not come up on port {port}")


def _get(port: int, path: str) -> tuple[int, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        data = resp.read()
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            parsed = {"_unparsed": data.decode("utf-8", "replace")}
        return resp.status, parsed
    finally:
        conn.close()


def _canned_probe(challenge: str, n_bytes: int) -> dict[str, Any]:
    from nomorals.core.gpu import _attest_expected_receipt

    total = 25769803776
    return {
        "ok": True,
        "name": "NVIDIA RTX 4090",
        "vram_gb": 24.0,
        "total_bytes": total,
        "total_gb": round(total / 1e9, 2),
        "free_before_bytes": 20_000_000_000,
        "free_after_bytes": 20_000_000_000 - n_bytes,
        "free_before_gb": 20.0,
        "free_after_gb": round((20_000_000_000 - n_bytes) / 1e9, 2),
        "allocated_bytes": n_bytes,
        "receipt": _attest_expected_receipt(challenge, n_bytes),
    }


class LiveAttestRouteTests(unittest.TestCase):
    """The /gpu/attest route over live HTTP, allocation faked.

    Built in setUp (N4 pattern): unittest.TestCase methods do not
    receive pytest fixtures in this pytest version.
    """

    def setUp(self) -> None:
        from unittest import mock

        from nomorals.api.server import serve
        from nomorals.storage.db import Database

        self.tmp = tempfile.mkdtemp(prefix="attest-")
        self.db = Database(f"{self.tmp}/t.db")
        self.port = _free_port()
        serve(_context(self.db, self.tmp), host="127.0.0.1",
              port=self.port, background=True, protocols=False, token="")
        _wait_up(self.port)
        self._patcher = mock.patch.object(
            gpu_mod, "probe_attestation", _canned_probe)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()

    def test_route_serves_attestation(self):
        status, payload = _get(
            self.port, "/gpu/attest?challenge=" + "ab" * 16
            + "&bytes=1048576")
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["name"], "NVIDIA RTX 4090")

    def test_route_rejects_missing_challenge(self):
        status, payload = _get(self.port, "/gpu/attest?bytes=1048576")
        self.assertEqual(status, 400, payload)

    def test_route_rejects_oversize_proof(self):
        status, payload = _get(
            self.port, "/gpu/attest?challenge=" + "ab" * 16
            + "&bytes=3221225472")
        self.assertEqual(status, 400, payload)

    def test_route_throttles_repeat_attestation(self):
        status, _ = _get(self.port, "/gpu/attest?challenge=" + "ab" * 16
                         + "&bytes=1048576")
        self.assertEqual(status, 200)
        status, payload = _get(
            self.port, "/gpu/attest?challenge=" + "cd" * 16
            + "&bytes=1048576")
        self.assertEqual(status, 429, payload)

    def test_round_trip_client_verifies_server(self):
        from nomorals.core.gpu import attest_forwarded

        got = attest_forwarded(f"http://127.0.0.1:{self.port}")
        self.assertIsNotNone(got)
        assert got is not None
        self.assertTrue(got.attested)
        self.assertEqual(got.vram_gb, 24.0)
        self.assertEqual(got.name, "NVIDIA RTX 4090")
