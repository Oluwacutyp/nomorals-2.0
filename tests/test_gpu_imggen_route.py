"""Remote render routing — text-to-image to the attested forwarded GPU.

When this box cannot render locally (no torch, or no loadable
checkpoint) and an attested forwarded GPU clears the generating floor,
``Studio.generate`` routes the job to ``POST /gpu/imggen`` on the GPU
host and saves the returned PNG through the same ``_finalize`` as a
local render. The server side runs the exact production path —
``Studio.generate(allow_remote=False)`` — never a second render
implementation.

Live-loopback pattern from ``test_triggers_header.py``: a real
``serve()`` on 127.0.0.1 with a real DB, real HTTP posts, and the GPU
seam patched (``detect_gpu`` → attested-forwarded at the loopback
URL, ``TORCH_AVAILABLE`` → False, ``serve_render`` → a fake 1x1 red
PNG). There is no GPU on this box, so every routing test fakes the
endpoint seam — a real render on a real forwarded endpoint is the
live proof owed after pull, not something this suite can show.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import os
import socket
import tempfile
import threading
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from PIL import Image

from nomorals.api.server import serve
from nomorals.core.errors import NoMoralsError
from nomorals.core.gpu import GpuInfo
from nomorals.media.imggen import ImgGenError
from nomorals.media.imggen import remote as remote_mod
from nomorals.media.imggen import studio as studio_mod
from nomorals.media.imggen.remote import (
    MAX_IMAGE_BYTES,
    remote_available,
    remote_generate,
    serve_render,
    validate_render_params,
)
from nomorals.media.imggen.studio import Studio
from nomorals.storage.db import Database

ENDPOINT_URL = "http://127.0.0.1:9999"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _context(db: Database, tmp: str) -> SimpleNamespace:
    settings = SimpleNamespace(
        api=SimpleNamespace(token="", max_body_mb=0, cors_origins=None),
        backup_dir=tmp, workspace_dir=tmp)
    return SimpleNamespace(
        settings=settings, db=db, tools={}, bus=None,
        memory=SimpleNamespace(), context=None)


def _wait_up(port: int, timeout: float = 10.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(
                "127.0.0.1", port, timeout=2)
            conn.request("GET", "/live")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError("server did not come up")


def _post(port: int, path: str, body: dict[str, Any],
          timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", path, body=json.dumps(body),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            parsed = {"_unparsed": data.decode("utf-8", "replace")}
        return resp.status, parsed
    finally:
        conn.close()


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    img = Image.new("RGB", (1, 1), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _attested_info(url: str = ENDPOINT_URL,
                   vram: float = 24.0) -> GpuInfo:
    return GpuInfo(
        path="forwarded", has_gpu=True,
        name="RTX 4090 (forwarded)", vram_gb=vram, source="attested",
        forward_url=url, attested=True)


def _cpu_info() -> GpuInfo:
    return GpuInfo(path="cpu")


class RemoteRenderTests(unittest.TestCase):
    """Live-loopback pins: the client routes, the endpoint answers."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="gpuroute-")
        self.db = Database(f"{self.tmp}/t.db")
        self.port = _free_port()
        serve(_context(self.db, self.tmp), host="127.0.0.1",
              port=self.port, background=True, protocols=False,
              token="")
        _wait_up(self.port)

    def test_end_to_end_remote_render(self) -> None:
        """Local unavailable + attested endpoint → PNG saved, sidecar
        written, rung names the forwarded GPU."""
        png = _png_bytes()
        calls: list[dict[str, Any]] = []

        def fake_serve(prompt: str, **params: Any) -> dict[str, Any]:
            calls.append({"prompt": prompt, "params": params})
            return {
                "ok": True,
                "image_b64": base64.b64encode(png).decode("ascii"),
                "format": "png", "bytes": len(png),
                "model": "sd1.5", "engine": "imggen",
            }

        url = f"http://127.0.0.1:{self.port}"
        with patch.object(remote_mod, "detect_gpu",
                          return_value=_attested_info(url)), \
             patch.object(studio_mod, "TORCH_AVAILABLE", False), \
             patch.object(remote_mod, "serve_render",
                          side_effect=fake_serve):
            studio = Studio()
            out = os.path.join(self.tmp, "remote.png")
            paths = studio.generate("a red dot", save_to=out)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["prompt"], "a red dot")
        self.assertEqual(len(paths), 1)
        with open(paths[0], "rb") as handle:
            self.assertEqual(handle.read(), png)
        sidecar = paths[0] + ".txt"
        with open(sidecar, encoding="utf-8") as handle:
            sidecar_text = handle.read()
        self.assertIn("a red dot", sidecar_text)
        self.assertIn("forwarded GPU", sidecar_text)
        self.assertTrue(studio.last_route.startswith("forwarded "))
        self.assertIn(url, studio.last_route)

    def test_generate_document_names_remote_rung(self) -> None:
        """generate_document's provenance names the forwarded route."""
        png = _png_bytes()

        def fake_serve(prompt: str, **params: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "image_b64": base64.b64encode(png).decode("ascii"),
                "format": "png", "bytes": len(png),
                "model": "", "engine": "imggen",
            }

        url = f"http://127.0.0.1:{self.port}"
        with patch.object(remote_mod, "detect_gpu",
                          return_value=_attested_info(url)), \
             patch.object(studio_mod, "TORCH_AVAILABLE", False), \
             patch.object(remote_mod, "serve_render",
                          side_effect=fake_serve):
            studio = Studio()
            doc = studio.generate_document(
                "a red dot", save_to=os.path.join(self.tmp, "doc.png"))
        self.assertIn("forwarded", doc.source_rung)
        self.assertIn(url, doc.source_rung)

    def test_empty_prompt_is_400(self) -> None:
        status, body = _post(self.port, "/gpu/imggen", {"prompt": "  "})
        self.assertEqual(status, 400)
        self.assertIn("prompt", body.get("error", ""))

    def test_unknown_param_is_400(self) -> None:
        status, body = _post(
            self.port, "/gpu/imggen",
            {"prompt": "a red dot", "nope": 1})
        self.assertEqual(status, 400)
        self.assertIn("nope", body.get("error", ""))

    def test_busy_second_render_is_429(self) -> None:
        """A second render while one is in flight gets 429; the first
        still completes 200 — the GPU is one-at-a-time."""
        started = threading.Event()
        release = threading.Event()

        def blocking_serve(prompt: str, **params: Any) -> dict[str, Any]:
            started.set()
            release.wait(timeout=30)
            png = _png_bytes()
            return {
                "ok": True,
                "image_b64": base64.b64encode(png).decode("ascii"),
                "format": "png", "bytes": len(png),
                "model": "", "engine": "imggen",
            }

        results: list[tuple[int, dict[str, Any]]] = []

        def first() -> None:
            results.append(_post(
                self.port, "/gpu/imggen", {"prompt": "first"},
                timeout=30))

        with patch.object(remote_mod, "serve_render",
                          side_effect=blocking_serve):
            thread = threading.Thread(target=first, daemon=True)
            thread.start()
            self.assertTrue(started.wait(timeout=10))
            status, body = _post(
                self.port, "/gpu/imggen", {"prompt": "second"})
            self.assertEqual(status, 429)
            self.assertIn("already running", body.get("error", ""))
            release.set()
            thread.join(timeout=30)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 200)
        self.assertTrue(results[0][1].get("ok"))

    def test_unattested_never_routes(self) -> None:
        """A forwarded GPU that never attested is never routed to —
        the call fails with today's local error, not a remote one."""
        calls: list[Any] = []

        def recording_generate(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            raise AssertionError("must not route to an unattested GPU")

        unattested = GpuInfo(
            path="forwarded", has_gpu=False, vram_gb=0.0,
            forward_url=ENDPOINT_URL, attested=False)
        with patch.object(remote_mod, "detect_gpu",
                          return_value=unattested), \
             patch.object(studio_mod, "TORCH_AVAILABLE", False), \
             patch.object(remote_mod, "remote_generate",
                          side_effect=recording_generate), \
             self.assertRaises(ImgGenError) as caught:
            Studio().generate("a red dot")
        self.assertEqual(calls, [])
        # The failure is the LOCAL unavailability (no torch), not a
        # remote error — routing was never attempted.
        self.assertIn("torch", str(caught.exception).lower())

    def test_remote_failure_names_both_reasons(self) -> None:
        """Remote attempted + failed → the error names the remote
        failure AND the local unavailability that caused routing."""
        url = f"http://127.0.0.1:{self.port}"

        def failing_generate(prompt: str, **kwargs: Any) -> Any:
            raise ImgGenError("endpoint down (connection refused)")

        with patch.object(remote_mod, "detect_gpu",
                          return_value=_attested_info(url)), \
             patch.object(studio_mod, "TORCH_AVAILABLE", False), \
             patch.object(remote_mod, "remote_generate",
                          side_effect=failing_generate), \
             self.assertRaises(ImgGenError) as caught:
            Studio().generate("a red dot")
        message = str(caught.exception)
        self.assertIn("endpoint down", message)
        self.assertIn("local unavailable", message)
        self.assertIn("torch", message.lower())

    def test_local_path_unchanged_when_available(self) -> None:
        """Local renderable box never routes: the fake pipeline runs
        and remote_generate is never called."""
        calls: list[Any] = []

        def recording_generate(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            raise AssertionError("must not route when local is ready")

        class FakePipe:
            last_warnings: list[str] = []

            def generate(self, prompt: str, cfg: Any) -> list[Any]:
                return [Image.new("RGB", (2, 2), (0, 255, 0))]

        with patch.object(Studio, "_local_ready",
                          return_value=(True, "")), \
             patch.object(Studio, "pipeline",
                          return_value=FakePipe()), \
             patch.object(remote_mod, "remote_generate",
                          side_effect=recording_generate):
            studio = Studio()
            paths = studio.generate(
                "a green square",
                save_to=os.path.join(self.tmp, "local.png"))
        self.assertEqual(calls, [])
        self.assertEqual(studio.last_route, "local")
        self.assertEqual(len(paths), 1)
        with open(paths[0], "rb") as handle:
            self.assertTrue(handle.read().startswith(b"\x89PNG"))

    def test_batch_never_routes(self) -> None:
        """A multi-image request never routes — the wire carries one
        image. The local path runs (and fails honestly here)."""
        calls: list[Any] = []

        def recording_generate(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            raise AssertionError("must not route a batch")

        url = f"http://127.0.0.1:{self.port}"
        with patch.object(remote_mod, "detect_gpu",
                          return_value=_attested_info(url)), \
             patch.object(studio_mod, "TORCH_AVAILABLE", False), \
             patch.object(remote_mod, "remote_generate",
                          side_effect=recording_generate), \
             self.assertRaises(ImgGenError):
            Studio().generate("a red dot", batch_size=2)
        self.assertEqual(calls, [])


class RemoteAvailableTests(unittest.TestCase):
    """The routing gate: attested + floor-clear, nothing else."""

    def test_no_forwarded_gpu(self) -> None:
        with patch.object(remote_mod, "detect_gpu",
                          return_value=_cpu_info()):
            ok, reason = remote_available()
        self.assertFalse(ok)
        self.assertIn("no forwarded GPU", reason)

    def test_unattested_forwarded_gpu(self) -> None:
        info = GpuInfo(
            path="forwarded", has_gpu=False, vram_gb=0.0,
            forward_url=ENDPOINT_URL, attested=False)
        with patch.object(remote_mod, "detect_gpu", return_value=info):
            ok, reason = remote_available()
        self.assertFalse(ok)
        self.assertIn("never attested", reason)

    def test_attested_too_small_gpu(self) -> None:
        with patch.object(remote_mod, "detect_gpu",
                          return_value=_attested_info(vram=8.0)):
            ok, reason = remote_available()
        self.assertFalse(ok)
        self.assertIn("16", reason)

    def test_attested_big_gpu_clears(self) -> None:
        with patch.object(remote_mod, "detect_gpu",
                          return_value=_attested_info(vram=24.0)):
            ok, reason = remote_available()
        self.assertTrue(ok)
        self.assertEqual(reason, "")


class RemoteGenerateValidationTests(unittest.TestCase):
    """The client validates what comes back — never a faked image."""

    def _patched(self, info: GpuInfo | None = None):
        info = info or _attested_info()
        return (patch.object(remote_mod, "remote_available",
                             return_value=(True, "")),
                patch.object(remote_mod, "detect_gpu",
                             return_value=info))

    def _resp(self, ok: bool = True, status: int = 200,
              text: str = "", payload: dict[str, Any] | None = None):
        def _json() -> dict[str, Any]:
            if payload is None:
                raise ValueError("not json")
            return payload

        return SimpleNamespace(
            ok=ok, status=status, text=text, json=_json)

    def test_happy_path(self) -> None:
        png = _png_bytes()
        payload = {
            "ok": True,
            "image_b64": base64.b64encode(png).decode("ascii"),
            "format": "png", "bytes": len(png), "model": "sd1.5",
            "engine": "imggen",
        }
        (avail, detect) = self._patched()
        with avail, detect:
            raw, info = remote_generate(
                "a red dot",
                http_post=lambda url, body: self._resp(payload=payload))
        self.assertEqual(raw, png)
        self.assertEqual(info["format"], "png")
        self.assertEqual(info["bytes"], len(png))
        self.assertEqual(info["model"], "sd1.5")
        self.assertEqual(info["endpoint"], ENDPOINT_URL)

    def test_error_status(self) -> None:
        (avail, detect) = self._patched()
        with avail, detect, \
             self.assertRaises(ImgGenError) as caught:
            remote_generate(
                "a red dot",
                http_post=lambda url, body: self._resp(
                    ok=False, status=500, text="boom"))
        self.assertIn("500", str(caught.exception))

    def test_bad_base64_rejected(self) -> None:
        (avail, detect) = self._patched()
        with avail, detect, \
             self.assertRaises(ImgGenError) as caught:
            remote_generate(
                "a red dot",
                http_post=lambda url, body: self._resp(payload={
                    "ok": True, "image_b64": "!!!not-base64!!!"}))
        self.assertIn("invalid image bytes", str(caught.exception))

    def test_non_png_rejected(self) -> None:
        (avail, detect) = self._patched()
        with avail, detect, \
             self.assertRaises(ImgGenError) as caught:
            remote_generate(
                "a red dot",
                http_post=lambda url, body: self._resp(payload={
                    "ok": True,
                    "image_b64": base64.b64encode(
                        b"this is not a png").decode("ascii")}))
        self.assertIn("non-PNG", str(caught.exception))

    def test_ok_false_rejected(self) -> None:
        (avail, detect) = self._patched()
        with avail, detect, \
             self.assertRaises(ImgGenError) as caught:
            remote_generate(
                "a red dot",
                http_post=lambda url, body: self._resp(payload={
                    "ok": False, "error": "no GPU here"}))
        self.assertIn("no GPU here", str(caught.exception))

    def test_oversized_rejected(self) -> None:
        (avail, detect) = self._patched()
        big = b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_IMAGE_BYTES
        with avail, detect, \
             self.assertRaises(ImgGenError) as caught:
            remote_generate(
                "a red dot",
                http_post=lambda url, body: self._resp(payload={
                    "ok": True,
                    "image_b64": base64.b64encode(big).decode(
                        "ascii")}))
        self.assertIn("over the", str(caught.exception))

    def test_unavailable_never_posts(self) -> None:
        calls: list[Any] = []
        with patch.object(remote_mod, "remote_available",
                          return_value=(False, "never attested")), \
             self.assertRaises(ImgGenError) as caught:
            remote_generate(
                "a red dot",
                http_post=lambda url, body: calls.append(url))
        self.assertEqual(calls, [])
        self.assertIn("never attested", str(caught.exception))


class ValidateRenderParamsTests(unittest.TestCase):
    """The endpoint validates before it renders."""

    def test_valid(self) -> None:
        params = validate_render_params({
            "prompt": "  a red dot  ", "steps": 30,
            "guidance_scale": 7.5, "width": 512, "height": 512,
            "negative_prompt": "blurry", "sampler": "ddim",
            "seed": 42, "checkpoint": "sd1.5"})
        self.assertEqual(params["prompt"], "a red dot")
        self.assertEqual(params["steps"], 30)
        self.assertEqual(params["seed"], 42)
        self.assertEqual(params["sampler"], "ddim")

    def test_missing_prompt(self) -> None:
        with self.assertRaises(NoMoralsError):
            validate_render_params({})

    def test_empty_prompt(self) -> None:
        with self.assertRaises(NoMoralsError):
            validate_render_params({"prompt": "   "})

    def test_prompt_too_long(self) -> None:
        with self.assertRaises(NoMoralsError):
            validate_render_params({"prompt": "x" * 4097})

    def test_steps_out_of_range(self) -> None:
        for bad in (0, 151, "30", True, 7.5):
            with self.assertRaises(NoMoralsError, msg=str(bad)):
                validate_render_params(
                    {"prompt": "x", "steps": bad})

    def test_bad_sampler(self) -> None:
        with self.assertRaises(NoMoralsError):
            validate_render_params({"prompt": "x", "sampler": "euler"})

    def test_unknown_key(self) -> None:
        with self.assertRaises(NoMoralsError) as caught:
            validate_render_params({"prompt": "x", "surprise": 1})
        self.assertIn("surprise", str(caught.exception))

    def test_non_object_body(self) -> None:
        with self.assertRaises(NoMoralsError):
            validate_render_params(["not", "a", "dict"])  # type: ignore[arg-type]


class ServeRenderTests(unittest.TestCase):
    """The endpoint side calls Studio.generate — never a re-write."""

    def test_serve_render_calls_studio_locally(self) -> None:
        png = _png_bytes()
        path = os.path.join(tempfile.mkdtemp(prefix="gpuroute-"),
                            "srv.png")
        with open(path, "wb") as handle:
            handle.write(png)
        calls: list[dict[str, Any]] = []

        def fake_generate(self, prompt: str, **kwargs: Any) -> list[str]:
            calls.append({"prompt": prompt, "kwargs": kwargs})
            return [path]

        with patch.object(Studio, "generate", fake_generate):
            result = serve_render("hi", seed=7, checkpoint="sd1.5")
        self.assertEqual(len(calls), 1)
        # A forwarded request is never forwarded on.
        self.assertEqual(calls[0]["kwargs"].get("allow_remote"), False)
        self.assertEqual(calls[0]["kwargs"].get("seed"), 7)
        self.assertTrue(result["ok"])
        self.assertEqual(result["format"], "png")
        self.assertEqual(result["bytes"], len(png))
        self.assertEqual(base64.b64decode(result["image_b64"]), png)
        self.assertEqual(result["model"], "sd1.5")

    def test_serve_render_no_image_is_error(self) -> None:
        with patch.object(Studio, "generate", lambda self, p, **k: []), \
             self.assertRaises(ImgGenError):
            serve_render("hi")


if __name__ == "__main__":
    unittest.main()
