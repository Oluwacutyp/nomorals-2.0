"""Offline tests for the ComfyUI backend (mocked HTTP — no server needed)."""

import io
import json
import threading
import time
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from nomorals.media_edit.comfy import (
    ComfyUIBackend,
    GenerativeEditError,
    _img2img_workflow,
    _inpaint_workflow,
    _load_workflow,
    _txt2img_workflow,
    _upscale_workflow,
    comfy_available,
)
from nomorals.media_edit.generate import get_backend

WORKFLOWS = Path(__file__).resolve().parent.parent / "nomorals" / "media_edit" / "workflows"


def _png_bytes(size=(64, 64), color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status=200, body=b"", ctype="application/json"):
        self.status = status
        self._body = body
        self.headers = _FakeHeaders(ctype)

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeHeaders:
    def __init__(self, ctype):
        self._ctype = ctype

    def get_content_type(self):
        return self._ctype


class FakeComfy:
    """Stands in for urllib.request.urlopen. Records requests, serves canned
    ComfyUI API responses."""

    def __init__(self, *, fail_prompt=False, error_node=False,
                 prompt_delay=0.0, n_images=1):
        self.calls = []  # (method, url, body, content_type)
        self.fail_prompt = fail_prompt
        self.error_node = error_node
        self.prompt_delay = prompt_delay
        self.n_images = n_images
        self.prompt_spans = []  # (start, end) per /prompt call
        self._lock = threading.Lock()

    def __call__(self, req, timeout=None):
        method = req.get_method()
        url = req.full_url
        body = req.data or b""
        ctype = req.get_header("Content-type") or ""
        with self._lock:
            self.calls.append((method, url, body, ctype))
        path = url.split("8188", 1)[-1]
        if path.startswith("/system_stats"):
            return FakeResponse(body=json.dumps({"system": {}}).encode())
        if path.startswith("/upload/image"):
            assert b"image/png" in body or b"filename=" in body
            return FakeResponse(body=json.dumps(
                {"name": "uploaded.png", "subfolder": "", "type": "input"}
            ).encode())
        if path.startswith("/prompt"):
            if self.fail_prompt:
                raise urllib.error.URLError("connection refused (test)")
            start = time.time()
            if self.prompt_delay:
                time.sleep(self.prompt_delay)
            with self._lock:
                self.prompt_spans.append((start, time.time()))
            payload = json.loads(body.decode())
            assert "prompt" in payload and "client_id" in payload
            return FakeResponse(body=json.dumps(
                {"prompt_id": "pid-1", "number": 1, "node_errors": {}}).encode())
        if path.startswith("/history/"):
            if self.error_node:
                entry = {"status": {"status_str": "error", "completed": True,
                                    "messages": [
                                        ["execution_error",
                                         {"node_id": "3",
                                          "exception_message": "OOM boom",
                                          "exception_type": "RuntimeError"}]]},
                         "outputs": {}}
            else:
                entry = {"status": {"status_str": "success", "completed": True,
                                    "messages": []},
                         "outputs": {"9": {"images": [
                             {"filename": f"out_{i}.png", "subfolder": "",
                              "type": "output"}
                             for i in range(self.n_images)]}}}
            return FakeResponse(body=json.dumps({"pid-1": entry}).encode())
        if path.startswith("/view"):
            return FakeResponse(body=_png_bytes(), ctype="image/png")
        raise AssertionError(f"unexpected path {path}")

    def posted_workflow(self):
        for method, url, body, _ in self.calls:
            if method == "POST" and url.endswith("/prompt"):
                return json.loads(body.decode())["prompt"]
        raise AssertionError("no /prompt call recorded")


# ── templates ─────────────────────────────────────────────────────────────

def test_templates_are_valid_node_dicts():
    for name in ("txt2img", "img2img", "inpaint", "upscale"):
        wf = _load_workflow(name)
        assert isinstance(wf, dict) and wf
        for node_id, node in wf.items():
            assert "class_type" in node and "inputs" in node, (name, node_id)


def test_txt2img_parameterization():
    wf = _txt2img_workflow(prompt="a cat", negative_prompt="blurry",
                           seed=42, steps=30, guidance_scale=7.5,
                           width=768, height=512, n=2, checkpoint="sd.safetensors")
    texts = [n["inputs"]["text"] for n in wf.values()
             if n.get("class_type") == "CLIPTextEncode"]
    assert texts == ["a cat", "blurry"]
    latent = next(n for n in wf.values()
                  if n.get("class_type") == "EmptyLatentImage")["inputs"]
    assert (latent["width"], latent["height"], latent["batch_size"]) == (768, 512, 2)
    sampler = next(n for n in wf.values()
                   if n.get("class_type") == "KSampler")["inputs"]
    assert (sampler["seed"], sampler["steps"], sampler["cfg"]) == (42, 30, 7.5)
    ckpt = next(n for n in wf.values()
                if n.get("class_type") == "CheckpointLoaderSimple")["inputs"]
    assert ckpt["ckpt_name"] == "sd.safetensors"


def test_img2img_sets_denoise_and_image():
    wf = _img2img_workflow(image_name="up.png", prompt="p", negative_prompt=None,
                           strength=0.6, seed=None, steps=None,
                           guidance_scale=None, checkpoint=None)
    sampler = next(n for n in wf.values()
                   if n.get("class_type") == "KSampler")["inputs"]
    assert sampler["denoise"] == 0.6
    loader = next(n for n in wf.values()
                  if n.get("class_type") == "LoadImage")["inputs"]
    assert loader["image"] == "up.png"


def test_img2img_rejects_bad_strength():
    with pytest.raises(GenerativeEditError):
        _img2img_workflow(image_name="x", prompt="p", negative_prompt=None,
                          strength=0.0, seed=None, steps=None,
                          guidance_scale=None, checkpoint=None)


def test_inpaint_wires_image_and_mask():
    wf = _inpaint_workflow(image_name="i.png", mask_name="m.png", prompt="p",
                           negative_prompt=None, seed=1, steps=None,
                           guidance_scale=None, checkpoint=None)
    loaders = [n for n in wf.values() if n.get("class_type") == "LoadImage"]
    assert [n["inputs"]["image"] for n in loaders] == ["i.png", "m.png"]
    assert any(n.get("class_type") == "VAEEncodeForInpaint" for n in wf.values())


def test_upscale_sets_scale():
    wf = _upscale_workflow(image_name="i.png", scale=3.0)
    node = next(n for n in wf.values()
                if n.get("class_type") == "UpscaleImageBy")["inputs"]
    assert node["scale_by"] == 3.0
    assert node["image"] == ["10", 0]


# ── backend behavior (mocked HTTP) ────────────────────────────────────────

def _backend(**kw):
    kw.setdefault("timeout_s", 30)
    return ComfyUIBackend(**kw)


def test_generate_returns_pil_images():
    fake = FakeComfy()
    with patch("urllib.request.urlopen", fake):
        imgs = _backend().generate("a cat", seed=7, n=1)
    assert len(imgs) == 1 and isinstance(imgs[0], Image.Image)
    wf = fake.posted_workflow()
    texts = [n["inputs"]["text"] for n in wf.values()
             if n.get("class_type") == "CLIPTextEncode"]
    assert texts[0] == "a cat"


def test_generate_batch_n():
    fake = FakeComfy(n_images=3)
    with patch("urllib.request.urlopen", fake):
        imgs = _backend().generate("x", n=3)
    assert len(imgs) == 3


def test_generate_rejects_empty_prompt():
    with pytest.raises(GenerativeEditError):
        _backend().generate("  ")


def test_img2img_uploads_input_image():
    fake = FakeComfy()
    img = Image.new("RGB", (32, 32), (200, 0, 0))
    with patch("urllib.request.urlopen", fake):
        out = _backend().img2img(img, "make it blue", strength=0.5)
    assert isinstance(out, Image.Image)
    uploads = [c for c in fake.calls if c[1].endswith("/upload/image")]
    assert len(uploads) == 1
    # PNG magic in the multipart body
    assert b"\x89PNG" in uploads[0][2]
    wf = fake.posted_workflow()
    assert wf["3"]["inputs"]["denoise"] == 0.5


def test_inpaint_uploads_image_and_mask():
    fake = FakeComfy()
    img = Image.new("RGB", (32, 32))
    with patch("urllib.request.urlopen", fake):
        out = _backend().inpaint(img, (0, 0, 16, 16), "fill red")
    assert isinstance(out, Image.Image)
    uploads = [c for c in fake.calls if c[1].endswith("/upload/image")]
    assert len(uploads) == 2  # image + mask


def test_error_node_raises_with_message():
    fake = FakeComfy(error_node=True)
    with patch("urllib.request.urlopen", fake):
        with pytest.raises(GenerativeEditError, match="OOM boom"):
            _backend().generate("x")


def test_unreachable_server_raises_clearly():
    fake = FakeComfy(fail_prompt=True)
    with patch("urllib.request.urlopen", fake):
        with pytest.raises(GenerativeEditError, match="unreachable"):
            _backend().generate("x")


def test_available_false_on_refused_no_raise():
    def _boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")
    with patch("urllib.request.urlopen", _boom):
        assert _backend().available() is False


def test_available_true_when_server_answers():
    fake = FakeComfy()
    with patch("urllib.request.urlopen", fake):
        assert _backend().available() is True


def test_init_never_connects():
    with patch("urllib.request.urlopen",
               side_effect=AssertionError("must not connect")):
        be = ComfyUIBackend()
        assert be.describe().startswith("generative backend 'comfy'")


def test_gpu_lock_serializes_concurrent_generations():
    fake = FakeComfy(prompt_delay=0.4)
    be = _backend()
    with patch("urllib.request.urlopen", fake):
        threads = [threading.Thread(target=lambda: be.generate("x"))
                   for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    assert len(fake.prompt_spans) == 2
    (s1, e1), (s2, e2) = fake.prompt_spans
    assert e1 <= s2 or e2 <= s1, "GPU lock did not serialize"


def test_gpu_lock_queue_timeout():
    be = _backend(queue_timeout_s=0.1)
    be._sem.acquire()  # hold the lock externally
    try:
        with pytest.raises(GenerativeEditError, match="GPU lock"):
            be.generate("x")
    finally:
        be._sem.release()


# ── registration / gating ─────────────────────────────────────────────────

def test_get_backend_comfy():
    be = get_backend("comfy")
    assert isinstance(be, ComfyUIBackend) and be.name == "comfy"


def test_backend_status_has_comfy_keys():
    from nomorals.media_edit.generate import backend_status
    st = backend_status()
    assert "comfy_available" in st and "comfy_reason" in st
    assert "comfy_host" in st and "comfy_port" in st


def test_comfy_available_termux_says_no(monkeypatch):
    monkeypatch.setenv("NM_PROFILE", "termux")
    ok, reason = comfy_available()
    assert ok is False and "HF backend" in reason


def test_comfy_available_unreachable():
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("nope")), \
         patch("nomorals.media_edit.comfy.get_profile_kind",
               return_value="workstation"):
        ok, reason = comfy_available()
    assert ok is False and "not reachable" in reason


def test_comfy_available_reachable():
    fake = FakeComfy()
    with patch("urllib.request.urlopen", fake):
        with patch("nomorals.media_edit.comfy.get_profile_kind",
                   return_value="workstation"):
            ok, reason = comfy_available()
    assert (ok, reason) == (True, "")


# ── jobs routing ──────────────────────────────────────────────────────────

def test_submit_gen_routes_to_comfy_when_available(tmp_path, monkeypatch):
    from nomorals.media_edit.jobs import JobManager
    fake = FakeComfy()
    monkeypatch.chdir(tmp_path)
    with patch("urllib.request.urlopen", fake), \
         patch("nomorals.media_edit.comfy.get_profile_kind",
               return_value="workstation"):
        mgr = JobManager()
        jid = mgr.submit_gen("t", "txt2img", {"prompt": "a cat", "n": 1})
        info = mgr.wait(jid, timeout=30)
    assert info["status"] == "done", info
    assert info["result"]["backend"] == "comfy"
    assert Path(info["result"]["output"]).exists()
    assert info["backend"] == "auto"  # the routing hint is recorded


def test_submit_gen_falls_back_when_comfy_down(tmp_path):
    from nomorals.media_edit.jobs import JobManager, get_manager  # noqa
    from nomorals.media_edit import generate as gen
    import os
    os.chdir(tmp_path)
    sentinel = Image.new("RGB", (8, 8))
    with patch("nomorals.media_edit.comfy.comfy_available",
               return_value=(False, "down")), \
         patch.object(gen, "get_backend") as gb:
        gb.return_value = _FakeGenBackend(sentinel)
        mgr = JobManager()
        jid = mgr.submit_gen("t", "txt2img", {"prompt": "x"})
        info = mgr.wait(jid, timeout=30)
    assert info["status"] == "done", info
    assert info["result"]["backend"] == "default"


def test_submit_gen_rejects_unknown_op():
    from nomorals.media_edit.jobs import JobManager
    with pytest.raises(Exception, match="unknown gen op"):
        JobManager().submit_gen("t", "nope", {})


class _FakeGenBackend:
    name = "fake"

    def __init__(self, img):
        self._img = img

    def generate(self, prompt, **kw):
        return [self._img]


def test_media_job_backend_field_defaults():
    from nomorals.media_edit.jobs import MediaJob
    j = MediaJob(id="x", kind="image", label="l")
    assert j.backend == "" and j.as_dict()["backend"] == ""
