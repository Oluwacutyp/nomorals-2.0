"""Gapfix168 tests: the provision-or-fallback seam (imggen as proof).

The seam (``nomorals/core/provision.py``): a resource that is
missing is fetched for her (idempotent, resumable, validated,
with real progress); a rung that cannot produce sheds and the
next rung runs; the honest error comes only when nothing can.

The proof (imggen): a bare ``/imggen <prompt>`` with an empty
checkpoint dir provisions the stock SD1.5 checkpoint and renders
— no training, no file drop. The live network is never hit here:
the download, the diffusion load and the HF call are all faked
at their seams; the wiring, the classification, the resume, the
rung accounting and the honest errors are exercised for real.

Failing-first: on pristine at 1686b544 this file does not even
collect (``ModuleNotFoundError`` for the new ``core/provision``
and ``imggen/provision`` modules) — the correct shape for a
new-seam lane.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.core.provision import (
    ProvisionError,
    ResourceSpec,
    attempt_chain,
    ensure_package,
    ensure_resource,
)
from nomorals.media.imggen import ImgGenError

SD15_KEYS = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.out.0.weight",
]


def _write_safetensors(path, keys):
    """A structurally valid safetensors file with the given keys.

    Header JSON + zero payload: the registry only reads the header.
    """
    header = {}
    for i, k in enumerate(keys):
        header[k] = {"dtype": "F32", "shape": [1],
                     "data_offsets": [i * 4, i * 4 + 4]}
    hj = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(b"\0" * (4 * len(keys)))


def _fake_downloader(payload: bytes, calls: list):
    """A downloader that writes ``payload`` to dest and records calls."""

    def dl(url, dest, *, progress=None, user_agent=None,
           timeout=None):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        if progress is not None:
            progress(len(payload), len(payload))
        return {"path": str(dest), "bytes": len(payload),
                "sha256": "deadbeef", "resumed": False,
                "downloaded_bytes": len(payload), "url": url}

    return dl


def _sd15_payload(tmp_path: Path) -> bytes:
    p = tmp_path / "stock.safetensors"
    _write_safetensors(p, SD15_KEYS)
    return p.read_bytes()


class _FakeResp:
    """A urllib response stand-in with canned bytes + headers."""

    def __init__(self, body, status=206, length=None):
        self._body = body
        self.status = status
        self.headers = {"Content-Length": str(
            length if length is not None else len(body))}
        self._pos = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        if self._pos >= len(self._body):
            return b""
        chunk = self._body[self._pos:self._pos + (n if n > 0
                                                 else len(self._body))]
        self._pos += len(chunk)
        return chunk


# ---------------------------------------------------------------------------
# The shared seam: core/provision
# ---------------------------------------------------------------------------

def test_ensure_resource_downloads_when_missing(tmp_path):
    dest = tmp_path / "model.bin"
    seen: list = []
    calls: list = []
    spec = ResourceSpec(
        name="test weights", urls=("https://example.invalid/m.bin",),
        dest=dest, validate=lambda p: p.read_bytes() == b"weights"
        or "wrong bytes")
    facts = ensure_resource(
        spec, progress=lambda d, t: seen.append((d, t)),
        downloader=_fake_downloader(b"weights", calls))
    assert facts["already"] is False
    assert dest.read_bytes() == b"weights"
    assert calls == ["https://example.invalid/m.bin"]
    assert seen == [(7, 7)]


def test_ensure_resource_skips_when_valid_present(tmp_path):
    dest = tmp_path / "model.bin"
    dest.write_bytes(b"weights")
    calls: list = []
    spec = ResourceSpec(
        name="test weights", urls=("https://example.invalid/m.bin",),
        dest=dest, validate=lambda p: p.read_bytes() == b"weights"
        or "wrong bytes")
    facts = ensure_resource(
        spec, downloader=_fake_downloader(b"weights", calls))
    assert facts["already"] is True
    assert calls == []


def test_ensure_resource_resumes_partial_download(tmp_path, monkeypatch):
    from nomorals.core import provision as prov

    dest = tmp_path / "model.bin"
    part = dest.with_suffix(".bin.part")
    part.write_bytes(b"0123456789")  # 10 bytes already landed
    rest = b"ABCDEFGHIJ"  # 10 more to go
    seen_headers: list = []

    def fake_urlopen(req, timeout=None):
        seen_headers.append(dict(req.headers))
        return _FakeResp(rest, status=206, length=len(rest))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    facts = prov._download_url(
        "https://example.invalid/m.bin", dest)
    assert facts["resumed"] is True
    assert dest.read_bytes() == b"0123456789ABCDEFGHIJ"
    assert seen_headers[0].get("Range") == "bytes=10-"


def test_ensure_resource_rejects_invalid_landing(tmp_path):
    dest = tmp_path / "model.bin"
    spec = ResourceSpec(
        name="test weights", urls=("https://example.invalid/m.bin",),
        dest=dest, validate=lambda p: "not a model")
    with pytest.raises(ProvisionError) as ei:
        ensure_resource(
            spec, downloader=_fake_downloader(b"garbage", []))
    assert "not a model" in str(ei.value)
    assert not dest.exists()
    assert ei.value.attempts


def test_ensure_package_installs_when_import_missing(monkeypatch):
    import importlib

    real_find_spec = importlib.util.find_spec
    seen = {"n": 0}

    def fake_find_spec(name, *a, **k):
        if name == "huggingface_hub":
            seen["n"] += 1
            return None if seen["n"] == 1 else object()
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    calls: list = []

    def fake_runner(argv, timeout):
        calls.append(argv)
        return (0, "Successfully installed huggingface_hub")

    facts = ensure_package("huggingface_hub", runner=fake_runner)
    assert facts["already"] is False
    assert calls and calls[0][:3] == [sys.executable, "-m", "pip"]
    assert "huggingface_hub" in calls[0]


def test_attempt_chain_first_working_wins():
    calls: list = []

    def first():
        calls.append("first")
        raise RuntimeError("nope")

    def second():
        calls.append("second")
        return "image-bytes"

    name, result, attempts = attempt_chain(
        "an image", [("first", first), ("second", second)])
    assert (name, result) == ("second", "image-bytes")
    assert attempts == ["first: nope"]
    assert calls == ["first", "second"]


def test_attempt_chain_all_shed_honest_error():
    with pytest.raises(ProvisionError) as ei:
        attempt_chain("an image", [
            ("local", lambda: (_ for _ in ()).throw(
                RuntimeError("no checkpoint"))),
            ("remote", lambda: (_ for _ in ()).throw(
                RuntimeError("no network"))),
        ])
    assert "nothing in the system could produce an image" in str(ei.value)
    assert "local: no checkpoint" in str(ei.value)
    assert "remote: no network" in str(ei.value)


# ---------------------------------------------------------------------------
# The proof: imggen rides the seam
# ---------------------------------------------------------------------------

def test_stock_checkpoint_provisions_on_empty_registry(
        tmp_path, monkeypatch):
    from nomorals.media.imggen.provision import ensure_stock_checkpoint

    # The payload source lives OUTSIDE the checkpoint dir: the
    # registry scans the dir, so a source file inside it would
    # count as an installed model and skip the download.
    monkeypatch.setenv("DEVON_IMGGEN_DIR", str(tmp_path / "ckpt"))
    payload = _sd15_payload(tmp_path / "src")
    monkeypatch.setattr(
        "nomorals.media.imggen.provision.STOCK_BYTES", len(payload))
    calls: list = []
    entry = ensure_stock_checkpoint(
        downloader=_fake_downloader(payload, calls))
    assert entry["model_class"] == "sd15"
    assert entry["loadable"] is True
    assert calls == [
        "https://huggingface.co/runwayml/stable-diffusion-v1-5/"
        "resolve/main/v1-5-pruned-emaonly.safetensors"]
    # Second call: already there — no download.
    calls.clear()
    again = ensure_stock_checkpoint(
        downloader=_fake_downloader(payload, calls))
    assert again["name"] == entry["name"]
    assert calls == []


def test_stock_checkpoint_skipped_when_owner_checkpoint_present(
        tmp_path, monkeypatch):
    from nomorals.media.imggen.provision import ensure_stock_checkpoint

    monkeypatch.setenv("DEVON_IMGGEN_DIR", str(tmp_path))
    _write_safetensors(tmp_path / "mine.safetensors", SD15_KEYS)
    calls: list = []
    entry = ensure_stock_checkpoint(
        downloader=_fake_downloader(b"whatever", calls))
    assert entry["name"] == "mine"
    assert entry["loadable"] is True
    assert calls == []


class _FakePipe:
    """An SD1.5 pipeline stand-in: renders a solid PIL image."""

    warnings: list = []
    device = "cpu"
    tokenizer_mode = "hash"
    dtype = "fp32"
    quant = "none"

    @classmethod
    def from_safetensors(cls, path, device="", dtype="fp32",
                         quant="none"):
        return cls()

    def generate(self, text, *, steps, guidance_scale, seed, width,
                 height, negative_prompt, eta):
        return [Image.new("RGB", (width or 64, height or 64),
                          (10, 20, 30))]


class _FakeAdapter:
    """The Studio adapter shape around _FakePipe.

    The real sdcompat module refuses to import without torch, so
    the test stages a fake module (below) carrying this adapter
    instead of the real one; it maps PipelineConfig onto the
    pipe exactly the way the real adapter does.
    """

    model_class = "sd15"

    def __init__(self, pipeline):
        self._pipe = pipeline
        self.last_warnings = list(pipeline.warnings)

    @property
    def device(self):
        return self._pipe.device

    def generate(self, prompt, cfg=None):
        from nomorals.media.imggen.pipeline import PipelineConfig

        cfg = cfg or PipelineConfig()
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        out: list = []
        for text in prompts:
            for i in range(max(1, cfg.batch_size)):
                seed = None if cfg.seed is None else cfg.seed + i
                out.extend(self._pipe.generate(
                    text, steps=cfg.steps,
                    guidance_scale=cfg.guidance_scale, seed=seed,
                    width=cfg.width, height=cfg.height,
                    negative_prompt=cfg.negative_prompt, eta=cfg.eta))
        self.last_warnings = list(self._pipe.warnings)
        return out


def test_studio_generate_provisions_and_renders(tmp_path, monkeypatch):
    from nomorals.media.imggen.studio import Studio

    ckpt_dir = tmp_path / "ckpt"
    out = tmp_path / "out.png"
    monkeypatch.setenv("DEVON_IMGGEN_DIR", str(ckpt_dir))
    payload = _sd15_payload(tmp_path / "src")
    monkeypatch.setattr(
        "nomorals.media.imggen.provision.STOCK_BYTES", len(payload))
    calls: list = []

    def dl(url, dest, *, progress=None, user_agent=None,
           timeout=None):
        return _fake_downloader(payload, calls)(
            url, dest, progress=progress, user_agent=user_agent,
            timeout=timeout)

    # The provision call inside Studio resolves
    # ensure_stock_checkpoint from its module at call time, so
    # the fake downloader rides in through this wrapper.
    import nomorals.media.imggen.provision as prov_mod

    real_ensure = prov_mod.ensure_stock_checkpoint

    def ensure_with_fake(**kw):
        kw["downloader"] = dl
        return real_ensure(**kw)

    monkeypatch.setattr(prov_mod, "ensure_stock_checkpoint",
                        ensure_with_fake)
    monkeypatch.setattr(
        "nomorals.media.imggen.studio.TORCH_AVAILABLE", True)
    # sdcompat refuses to import without torch, so stage a fake
    # module carrying the fake pipe + adapter; Studio.pipeline()
    # imports it at call time and takes the fakes.
    fake_mod = SimpleNamespace(
        SD15Pipeline=_FakePipe, SD15StudioAdapter=_FakeAdapter)
    monkeypatch.setitem(
        sys.modules, "nomorals.media.imggen.sdcompat", fake_mod)
    studio = Studio()
    paths = studio.generate("cyberpunk city", save_to=str(out),
                            width=64, height=64)
    assert paths and Path(paths[0]).exists()
    assert studio.last_provision is not None
    assert calls  # the stock file was actually fetched
    assert (ckpt_dir / "v1-5-pruned-emaonly.safetensors").exists()


def test_studio_no_provision_without_torch(tmp_path, monkeypatch):
    from nomorals.media.imggen.studio import Studio

    monkeypatch.setenv("DEVON_IMGGEN_DIR", str(tmp_path))
    monkeypatch.setattr(
        "nomorals.media.imggen.studio.TORCH_AVAILABLE", False)
    called: list = []

    def _boom(**kw):
        called.append(True)
        raise AssertionError("provision must not run without torch")

    monkeypatch.setattr(
        "nomorals.media.imggen.provision.ensure_stock_checkpoint",
        _boom)
    with pytest.raises(ImgGenError) as ei:
        Studio().generate("cyberpunk city")
    assert "torch" in str(ei.value)
    assert called == []


def test_generate_image_falls_back_to_hf(tmp_path, monkeypatch):
    from nomorals.media_edit import generate as gen

    monkeypatch.setattr("nomorals.media.imggen.TORCH_AVAILABLE",
                        False)
    monkeypatch.setattr(gen.DiffusersBackend, "available",
                        staticmethod(lambda: False))
    # Block the real huggingface_hub (it IS installed in this
    # venv) so the auto rung genuinely sheds: its hf candidate
    # probes the import, and only the final direct-HF rung may
    # serve. Without this the test's premise depends on the
    # environment — the auto rung would return the fake HF
    # backend directly and the "auto:" note would never land.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    class _FakeHF:
        name = "hf"

        def generate(self, prompt, **kw):
            return [Image.new("RGB", (64, 64), (1, 2, 3))]

    monkeypatch.setattr(gen, "HFInferenceBackend", _FakeHF)
    images, rung, notes = gen.generate_image("cyberpunk city")
    assert rung == "hf"
    assert len(images) == 1
    assert notes[0].startswith("native:")
    assert notes[1].startswith("auto:")


def test_generate_image_honest_when_nothing_can_produce(monkeypatch):
    from nomorals.media_edit import generate as gen
    from nomorals.media_edit.generate import GenerativeEditError

    monkeypatch.setattr("nomorals.media.imggen.TORCH_AVAILABLE",
                        False)
    monkeypatch.setattr(gen.DiffusersBackend, "available",
                        staticmethod(lambda: False))
    # Same as the fallback test: block the real huggingface_hub
    # so the auto rung sheds at resolve, not at generate.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    class _FailingHF:
        name = "hf"

        def generate(self, prompt, **kw):
            raise RuntimeError("HF unreachable")

    monkeypatch.setattr(gen, "HFInferenceBackend", _FailingHF)
    with pytest.raises(GenerativeEditError) as ei:
        gen.generate_image("cyberpunk city")
    assert "nothing in the system could produce an image" in str(ei.value)
    assert "native:" in str(ei.value)
    assert "auto:" in str(ei.value)
    assert "hf: HF unreachable" in str(ei.value)


class _FakeSettings:
    def __init__(self, root):
        self._root = Path(root)

    def resolve(self, relative):
        return self._root / relative


def test_control_imggen_prompt_uses_chain(tmp_path, monkeypatch):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    class M(RuntimeMediaMixin):
        def __init__(self):
            self.context = SimpleNamespace(
                settings=_FakeSettings(tmp_path))

    def fake_generate(prompt, **kw):
        return ([Image.new("RGB", (64, 64), (9, 9, 9))], "hf",
                ["native: torch not installed on this box"])

    monkeypatch.setattr(
        "nomorals.media_edit.generate.generate_image", fake_generate)
    m = M()
    out = m._control_imggen("cyberpunk city")
    assert "🎨 generated: cyberpunk city" in out
    assert "saved:" in out
    assert "via hf" in out
    assert "local unavailable: torch not installed" in out
    saved = out.split("saved:")[1].strip().splitlines()[0]
    assert Path(saved).exists()
