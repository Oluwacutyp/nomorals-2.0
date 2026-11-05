"""Conformance suite for the §1.1e generative-backend contract.

Devon's analogue of craft-ai's shared ``conformance`` suite + mock
provider (INTEGRATION_PLAN §1.1e, REPORT §1.11): the SAME checks run
against every backend in the ``media_edit/generate.py`` ladder —
models() / estimate() / submit() / poll() / cancel() with honest
semantics:

- estimate() numbers are real or named unknowns — never fabricated;
- cancel() actually cancels, or the returned status says it cannot;
- failures carry a machine-readable ``kind`` separate from the human
  message (poll's ``error_kind``, the exception's ``kind``/``code``).

Network-bound backends run against fakes (stub HF client, stub
torch/diffusers modules, a stateful fake ComfyUI server, fake paid
connectors) — no real provider is ever contacted. The local fake
backend mirrors ``tests/test_studio.py``'s FakeBackend, the suite's
starting point per the plan.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
from PIL import Image

from nomorals.core.errors import MissingCredentialError
from nomorals.media_edit import generate as G
from nomorals.media_edit.comfy import ComfyUIBackend
from nomorals.media_edit.generate import (
    CONTRACT_OPS,
    Estimate,
    GenerativeBackend,
    GenerativeEditError,
    JobRef,
    JobStatus,
    ModelInfo,
    contract_error_kind,
)

def _img(size=(16, 16), color=(10, 20, 30)):
    return Image.new("RGB", size, color)


def _png_bytes(size=(16, 16), color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# shared contract checks
# ---------------------------------------------------------------------------

def assert_estimate_honest(est: Estimate) -> None:
    """Every number has a named basis; every unknown is named."""
    assert isinstance(est, Estimate)
    for field_name in ("money", "credits", "seconds"):
        value = getattr(est, field_name)
        if value is None:
            assert any(u.startswith(field_name + ":")
                       for u in est.unknowns), (field_name, est)
        else:
            assert value >= 0, (field_name, est)
            assert est.basis, ("a number without a basis", est)


def assert_models_ok(models: list[ModelInfo]) -> None:
    assert isinstance(models, list)
    for m in models:
        assert isinstance(m, ModelInfo)
        assert m.id and isinstance(m.id, str)
        assert set(m.ops) <= set(CONTRACT_OPS), m
        json.dumps(asdict(m))  # comparable data must serialize


def check_sync_backend(backend: GenerativeBackend,
                       samples: dict[str, dict],
                       bad_sample: tuple[str, dict]) -> None:
    """The shared conformance checks for synchronous backends.

    ``samples`` maps each supported op to kwargs that succeed against
    the backend's fake; ``bad_sample`` is one (op, kwargs) that fails
    validation, for the failed-job error-kind check.
    """
    assert_models_ok(backend.models())
    for op in backend.supported_ops:
        assert_estimate_honest(backend.estimate(op))
    # submit → poll round-trip for every supported op
    for op, kwargs in samples.items():
        assert op in backend.supported_ops
        ref = backend.submit(op, **kwargs)
        assert isinstance(ref, JobRef)
        assert ref.backend == backend.name and ref.op == op and ref.job_id
        status = backend.poll(ref)
        assert isinstance(status, JobStatus)
        assert status.state == "succeeded", (op, status)
        assert status.result is not None, op
        # cancel of a finished job: state unchanged, inability stated
        cancelled = backend.cancel(ref)
        assert cancelled.state == "succeeded"
        assert "synchronous" in cancelled.detail
    # a failing op becomes a failed JOB with a machine-readable kind
    op, kwargs = bad_sample
    status = backend.poll(backend.submit(op, **kwargs))
    assert status.state == "failed"
    assert status.error_kind == "invalid_request", status
    assert status.error and status.error_kind != status.error
    # malformed requests raise typed errors instead of returning refs
    with pytest.raises(GenerativeEditError) as ei:
        backend.submit("frobnicate", prompt="x")
    assert ei.value.kind == "invalid_request"
    with pytest.raises(GenerativeEditError) as ei:
        backend.poll("no-such-job")
    assert ei.value.kind == "unknown_job"
    with pytest.raises(GenerativeEditError) as ei:
        backend.estimate("frobnicate")
    assert ei.value.kind == "invalid_request"


# ---------------------------------------------------------------------------
# the fake backend (test_studio.py lineage)
# ---------------------------------------------------------------------------

class FakeBackend(GenerativeBackend):
    """Deterministic stand-in implementing all four ops: records the
    call, marks the image. Same shape as test_studio.py's FakeBackend,
    extended to the full op set for the contract round-trips."""

    name = "fake"

    def __init__(self):
        self.calls = []

    def _mark(self, image):
        out = image.convert("RGB").copy()
        from PIL import ImageDraw
        ImageDraw.Draw(out).rectangle([0, 0, 3, 3], fill=(255, 0, 0))
        return out

    def edit(self, image, instruction, *, mask=None, strength=0.75,
             seed=None, **kwargs):
        if not instruction or not instruction.strip():
            raise GenerativeEditError("generative edit needs an instruction",
                                      kind="invalid_request")
        self.calls.append({"op": "edit", "instruction": instruction})
        return self._mark(image)

    def generate(self, prompt, *, seed=None, n=1, **kwargs):
        if not prompt or not prompt.strip():
            raise GenerativeEditError("text-to-image needs a prompt",
                                      kind="invalid_request")
        self.calls.append({"op": "generate", "prompt": prompt})
        return [_img() for _ in range(n)]

    def img2img(self, image, prompt, *, strength=0.6, seed=None, **kwargs):
        if not prompt or not prompt.strip():
            raise GenerativeEditError("img2img needs a prompt",
                                      kind="invalid_request")
        self.calls.append({"op": "img2img", "prompt": prompt})
        return self._mark(image)

    def inpaint(self, image, mask, prompt, *, seed=None, **kwargs):
        if not prompt or not prompt.strip():
            raise GenerativeEditError("inpaint needs a prompt",
                                      kind="invalid_request")
        self.calls.append({"op": "inpaint", "prompt": prompt})
        return self._mark(image)


def test_fake_backend_conforms():
    img = _img()
    check_sync_backend(
        FakeBackend(),
        samples={
            "edit": {"image": img, "instruction": "make it sunset"},
            "generate": {"prompt": "a cat"},
            "img2img": {"image": img, "prompt": "restyle"},
            "inpaint": {"image": img, "mask": [0, 0, 4, 4],
                        "prompt": "fill"},
        },
        bad_sample=("generate", {"prompt": "  "}),
    )


def test_partial_backend_declares_its_ops():
    """A backend implementing one op declares it; the rest fail typed —
    the base class never leaks NotImplementedError."""

    class EditOnly(FakeBackend):
        name = "edit-only"
        supported_ops = ("edit",)

    be = EditOnly()
    status = be.poll(be.submit("generate", prompt="x"))
    assert status.state == "failed"
    assert status.error_kind == "unsupported_op"
    # and a backend that genuinely lacks the method raises typed on a
    # direct call (the bare base class stands in for all such backends)
    with pytest.raises(GenerativeEditError) as ei:
        GenerativeBackend().generate("x")
    assert ei.value.kind == "unsupported_op"
    assert not isinstance(ei.value, NotImplementedError)


def test_estimate_unknown_names_everything():
    est = Estimate.unknown("no prices", "no credits", "no timing")
    assert est.money is None and est.credits is None and est.seconds is None
    assert_estimate_honest(est)


def test_error_taxonomy_is_machine_readable():
    exc = GenerativeEditError("human text", kind="provider_error")
    assert exc.kind == "provider_error"
    assert exc.code == "media.generative"
    assert exc.details["kind"] == "provider_error"
    assert str(exc) == "human text"
    assert exc.to_dict()["code"] == "media.generative"
    # still the media-edit error consumers catch
    from nomorals.media_edit.images import MediaEditError
    assert isinstance(exc, MediaEditError)
    assert contract_error_kind(exc) == "provider_error"
    missing = MissingCredentialError("no key", service="huggingface",
                                     env_key="HF_TOKEN",
                                     feature="media_edit.hf")
    assert contract_error_kind(missing) == "missing_credential"
    assert contract_error_kind(RuntimeError("x")) == "backend_error"


# ---------------------------------------------------------------------------
# HF backend (stub client — no network)
# ---------------------------------------------------------------------------

class _StubHFClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def text_to_image(self, prompt, model=None, **kwargs):
        return _img(color=(1, 2, 3))

    def image_to_image(self, image, prompt=None, model=None, **kwargs):
        return _img(image.size, (4, 5, 6))


def _hf_backend(**kw):
    kw.setdefault("model", "m/x")
    kw.setdefault("token", "hf_test")
    stub = SimpleNamespace(InferenceClient=_StubHFClient)
    patcher = mock.patch.dict(sys.modules, {"huggingface_hub": stub})
    patcher.start()
    return G.HFInferenceBackend(**kw), patcher


def test_hf_backend_conforms():
    be, patcher = _hf_backend()
    try:
        img = _img()
        check_sync_backend(
            be,
            samples={
                "edit": {"image": img, "instruction": "make it sunset"},
                "generate": {"prompt": "a cat"},
                "img2img": {"image": img, "prompt": "restyle"},
                "inpaint": {"image": img, "mask": [0, 0, 4, 4],
                            "prompt": "fill"},
            },
            bad_sample=("edit", {"image": img, "instruction": " "}),
        )
        models = be.models()
        assert models[0].id == "m/x" and models[0].locality == "remote"
        est = be.estimate("generate")
        assert est.money is None and est.seconds is None
    finally:
        patcher.stop()


def test_hf_missing_package_is_unavailable_kind():
    with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
        be = G.HFInferenceBackend(model="m/x", token="t")
        status = be.poll(be.submit("generate", prompt="a cat"))
    assert status.state == "failed"
    assert status.error_kind == "unavailable"


def test_hf_auth_failure_maps_to_missing_credential():
    class AuthClient:
        def __init__(self, **kwargs):
            pass

        def image_to_image(self, image, prompt=None, model=None, **kw):
            raise RuntimeError("401 Unauthorized")

    stub = SimpleNamespace(InferenceClient=AuthClient)
    with mock.patch.dict(sys.modules, {"huggingface_hub": stub}):
        be = G.HFInferenceBackend(model="m/x", token=None)
        status = be.poll(be.submit("edit", image=_img(),
                                   instruction="make it sunset"))
    assert status.state == "failed"
    assert status.error_kind == "missing_credential"


# ---------------------------------------------------------------------------
# diffusers backend (stub torch/diffusers — no downloads, no GPU)
# ---------------------------------------------------------------------------

def _diffusers_stubs():
    torch = ModuleType("torch")

    class _Generator:
        def manual_seed(self, seed):
            return self

    torch.Generator = _Generator
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.float16 = "fp16"
    torch.float32 = "fp32"

    diffusers = ModuleType("diffusers")

    class _Pipe:
        def __call__(self, prompt, image=None, mask_image=None,
                     generator=None, **kwargs):
            return SimpleNamespace(images=[_img(color=(7, 7, 7))])

        def to(self, device):
            return self

    def _auto():
        class _Auto:
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                return _Pipe()
        return _Auto

    diffusers.AutoPipelineForImage2Image = _auto()
    diffusers.AutoPipelineForText2Image = _auto()
    diffusers.AutoPipelineForInpainting = _auto()
    return {"torch": torch, "diffusers": diffusers}


def test_diffusers_backend_conforms():
    with mock.patch.dict(sys.modules, _diffusers_stubs()):
        be = G.DiffusersBackend(model="m/local")
        img = _img()
        check_sync_backend(
            be,
            samples={
                "edit": {"image": img, "instruction": "make it sunset"},
                "generate": {"prompt": "a cat"},
                "img2img": {"image": img, "prompt": "restyle"},
                "inpaint": {"image": img, "mask": [0, 0, 4, 4],
                            "prompt": "fill"},
            },
            bad_sample=("generate", {"prompt": ""}),
        )
        est = be.estimate("generate")
        assert est.money == 0.0 and est.credits == 0.0
        assert est.seconds is None  # hardware-dependent: a named unknown
        assert be.models()[0].locality == "local"


# ---------------------------------------------------------------------------
# native backend (stubbed registry + studio — no torch, no checkpoint)
# ---------------------------------------------------------------------------

def _native_patches(tmp: Path):
    import nomorals.media.imggen as imggen_pkg
    import nomorals.media.imggen.registry as registry

    entries = [{"loadable": True, "name": "tiny-1",
                "model_class": "sdxl-mini", "loadable_reason": "",
                "path": "/ckpt/tiny-1.safetensors"}]
    studio_mod = ModuleType("nomorals.media.imggen.studio")

    class _FakeStudio:
        def __init__(self, cfg):
            pass

        def generate(self, prompt, **kwargs):
            path = tmp / "native_gen.png"
            _img(color=(3, 3, 3)).save(path)
            return [str(path)]

    studio_mod.Studio = _FakeStudio
    studio_mod.StudioConfig = object
    pipeline_mod = ModuleType("nomorals.media.imggen.pipeline")
    pipeline_mod.PipelineConfig = object
    return [
        mock.patch.object(imggen_pkg, "TORCH_AVAILABLE", True),
        mock.patch.object(registry, "list_models", lambda: entries),
        mock.patch.dict(sys.modules, {
            "nomorals.media.imggen.studio": studio_mod,
            "nomorals.media.imggen.pipeline": pipeline_mod,
        }),
    ]


def test_native_backend_conforms(tmp_path):
    patches = _native_patches(tmp_path)
    for p in patches:
        p.start()
    try:
        be = G.NativeBackend()
        assert be.supported_ops == ("generate",)
        models = be.models()
        assert [m.id for m in models] == ["tiny-1"]
        assert models[0].locality == "local"
        assert_estimate_honest(be.estimate("generate"))
        status = be.poll(be.submit("generate", prompt="a cat"))
        assert status.state == "succeeded"
        assert len(status.result) == 1
        # the ops it never had fail typed now — no NotImplementedError
        status = be.poll(be.submit("edit", image=_img(),
                                   instruction="make it sunset"))
        assert status.state == "failed"
        assert status.error_kind == "unsupported_op"
        with pytest.raises(GenerativeEditError) as ei:
            be.edit(_img(), "make it sunset")
        assert ei.value.kind == "unsupported_op"
    finally:
        for p in patches:
            p.stop()


def test_native_models_with_broken_registry_is_unavailable():
    import nomorals.media.imggen.registry as registry

    def boom():
        raise RuntimeError("registry unreadable")

    with mock.patch.object(registry, "list_models", boom):
        with pytest.raises(GenerativeEditError) as ei:
            G.NativeBackend().models()
    assert ei.value.kind == "unavailable"


# ---------------------------------------------------------------------------
# paid backends (fake connectors — no network, no vault)
# ---------------------------------------------------------------------------

class _FakeLeonardoConnector:
    def generate_image(self, prompt, **kwargs):
        return [{"id": "g1", "url": "https://example.invalid/out.png"}]

    def edit_image(self, image_bytes, prompt, **kwargs):
        return [{"id": "g2", "url": "https://example.invalid/edit.png"}]


class _FakeStabilityConnector:
    def generate_image(self, prompt, **kwargs):
        return {"image_bytes": _png_bytes()}

    def edit_image(self, image_bytes, prompt, **kwargs):
        return {"image_bytes": _png_bytes()}


class _FakeNanoConnector:
    def generate_image(self, prompt, **kwargs):
        return [{"image_bytes": _png_bytes()}]

    def edit_image(self, image_bytes, prompt, **kwargs):
        return [{"image_bytes": _png_bytes()}]


def _paid_backend(cls, fake_conn):
    return cls(vault=object(),
               connector_factory=lambda name, vault: fake_conn)


def test_paid_backends_conform():
    img = _img()
    cases = [
        (G.LeonardoBackend, _FakeLeonardoConnector(), "leonardo-default"),
        (G.StabilityAIBackend, _FakeStabilityConnector(), "sd3.5-large"),
        (G.NanoBananaBackend, _FakeNanoConnector(), "gemini-2.5-flash-image"),
    ]
    with mock.patch.object(G.LeonardoBackend, "_download",
                           staticmethod(lambda url, timeout=60.0:
                                        _png_bytes())):
        for cls, conn, model_id in cases:
            be = _paid_backend(cls, conn)
            check_sync_backend(
                be,
                samples={
                    "edit": {"image": img, "instruction": "make it sunset"},
                    "generate": {"prompt": "a cat"},
                    "img2img": {"image": img, "prompt": "restyle"},
                    "inpaint": {"image": img, "mask": [0, 0, 4, 4],
                                "prompt": "fill"},
                },
                bad_sample=("generate", {"prompt": " "}),
            )
            models = be.models()
            assert models[0].id == model_id, (cls, models)
            assert models[0].paid is True
            # the honest core of the paid estimate: no invented price
            est = be.estimate("generate")
            assert est.money is None and est.credits is None
            assert_estimate_honest(est)
            # the declared gap is in code, not discovered by callers
            assert be.contract_gaps


def test_paid_backend_provider_failure_kind():
    class Exploding:
        def generate_image(self, prompt, **kwargs):
            raise RuntimeError("provider exploded")

    be = _paid_backend(G.StabilityAIBackend, Exploding())
    status = be.poll(be.submit("generate", prompt="a cat"))
    assert status.state == "failed"
    assert status.error_kind == "provider_error"
    assert "provider exploded" in status.error


def test_paid_backend_without_factory_is_unavailable():
    with mock.patch.object(G, "_CONNECTOR_FACTORY", None):
        be = G.StabilityAIBackend(vault=object())
        status = be.poll(be.submit("generate", prompt="a cat"))
    assert status.state == "failed"
    assert status.error_kind == "unavailable"


# ---------------------------------------------------------------------------
# comfy backend (stateful fake server — the real job lifecycle)
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, body=b"", ctype="application/json", status=200):
        self._body = body
        self.status = status
        self.headers = SimpleNamespace(
            get_content_type=lambda: ctype)

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeComfyServer:
    """A one-job-at-a-time ComfyUI: the first job runs, the rest queue;
    a running job completes after ``complete_after`` history polls.
    Records interrupts and queue deletes so tests can prove cancel()
    really cancelled — on the server, not just in our bookkeeping."""

    def __init__(self, *, complete_after=2, error=False, down=False):
        self.jobs: dict[str, dict] = {}
        self.order: list[str] = []
        self.next_id = 0
        self.complete_after = complete_after
        self.error = error
        self.down = down
        self.interrupts: list[str] = []
        self.deleted: list[str] = []

    # -- state helpers ----------------------------------------------------
    def _running_id(self):
        for pid in self.order:
            if self.jobs[pid]["state"] == "running":
                return pid
        return None

    def _promote(self):
        if self._running_id() is None:
            for pid in self.order:
                if self.jobs[pid]["state"] == "pending":
                    self.jobs[pid]["state"] = "running"
                    break

    # -- HTTP surface -------------------------------------------------------
    def __call__(self, req, timeout=None):
        if self.down:
            raise urllib.error.URLError("connection refused (test)")
        method = req.get_method()
        path = req.full_url.split("8188", 1)[-1]
        body = req.data or b""
        if path.startswith("/system_stats"):
            return _FakeResp(json.dumps({"system": {}}).encode())
        if path.startswith("/object_info"):
            info = {"CheckpointLoaderSimple": {"input": {"required": {
                "ckpt_name": [["sd15.safetensors", "flux.safetensors"]]}}}}
            return _FakeResp(json.dumps(info).encode())
        if path.startswith("/upload/image"):
            return _FakeResp(json.dumps(
                {"name": "uploaded.png", "subfolder": "",
                 "type": "input"}).encode())
        if path == "/prompt" and method == "POST":
            self.next_id += 1
            pid = f"pid-{self.next_id}"
            self.jobs[pid] = {"state": "pending", "polls": 0}
            self.order.append(pid)
            self._promote()
            return _FakeResp(json.dumps(
                {"prompt_id": pid, "number": self.next_id,
                 "node_errors": {}}).encode())
        if path == "/interrupt" and method == "POST":
            running = self._running_id()
            if running is not None:
                self.jobs[running]["state"] = "interrupted"
                self.interrupts.append(running)
            return _FakeResp(b"{}")
        if path == "/queue" and method == "POST":
            for pid in json.loads(body.decode()).get("delete", []):
                if pid in self.jobs:
                    self.jobs[pid]["state"] = "deleted"
                    self.deleted.append(pid)
            self._promote()
            return _FakeResp(b"{}")
        if path == "/queue":
            self._promote()
            running = [[0, pid] for pid in self.order
                       if self.jobs[pid]["state"] == "running"]
            pending = [[0, pid] for pid in self.order
                       if self.jobs[pid]["state"] == "pending"]
            return _FakeResp(json.dumps(
                {"queue_running": running,
                 "queue_pending": pending}).encode())
        if path.startswith("/history/"):
            pid = path.rsplit("/", 1)[-1]
            job = self.jobs.get(pid)
            if job is None or job["state"] in ("pending", "deleted"):
                return _FakeResp(b"{}")
            if job["state"] == "interrupted":
                entry = {"status": {
                    "status_str": "error", "completed": True,
                    "messages": [["execution_interrupted",
                                  {"node_id": "3"}]]},
                    "outputs": {}}
                return _FakeResp(json.dumps({pid: entry}).encode())
            if job["state"] == "running":
                job["polls"] += 1
                if job["polls"] < self.complete_after:
                    return _FakeResp(b"{}")
                job["state"] = "done"
            if self.error:
                entry = {"status": {
                    "status_str": "error", "completed": True,
                    "messages": [["execution_error",
                                  {"node_id": "3",
                                   "exception_message": "OOM boom",
                                   "exception_type": "RuntimeError"}]]},
                    "outputs": {}}
            else:
                entry = {"status": {
                    "status_str": "success", "completed": True,
                    "messages": []},
                    "outputs": {"9": {"images": [
                        {"filename": "out_0.png", "subfolder": "",
                         "type": "output"}]}}}
            return _FakeResp(json.dumps({pid: entry}).encode())
        if path.startswith("/view"):
            return _FakeResp(_png_bytes(), ctype="image/png")
        raise AssertionError(f"unexpected path {path}")


def _comfy(server: FakeComfyServer, **kw) -> ComfyUIBackend:
    kw.setdefault("timeout_s", 30)
    kw.setdefault("queue_timeout_s", 2)
    be = ComfyUIBackend(**kw)
    patcher = mock.patch("urllib.request.urlopen", server)
    patcher.start()
    return be, patcher


def test_comfy_models_and_estimate_are_live_data():
    server = FakeComfyServer()
    be, patcher = _comfy(server)
    try:
        models = be.models()
        assert [m.id for m in models] == ["sd15.safetensors",
                                          "flux.safetensors"]
        assert all(m.locality == "server" for m in models)
        est = be.estimate("generate")
        assert est.money == 0.0 and est.credits == 0.0
        assert "live queue" in est.basis
        assert_estimate_honest(est)
        assert be.cancellable is True and be.contract_gaps
    finally:
        patcher.stop()


def test_comfy_submit_poll_succeeds_with_real_prompt_id():
    server = FakeComfyServer()
    be, patcher = _comfy(server)
    try:
        ref = be.submit("generate", prompt="a cat")
        assert ref.job_id == "pid-1"  # the server's own id
        first = be.poll(ref)
        assert first.state == "running", first
        done = be.poll(ref)
        assert done.state == "succeeded"
        assert isinstance(done.result, list) and len(done.result) == 1
        # single-image ops unwrap to one image, like the sync methods
        ref2 = be.submit("img2img", image=_img(), prompt="restyle")
        be.poll(ref2)
        done2 = be.poll(ref2)
        assert done2.state == "succeeded"
        assert isinstance(done2.result, Image.Image)
        # terminal cancel: too late, and it says so
        late = be.cancel(ref2)
        assert late.state == "succeeded" and "too late" in late.detail
    finally:
        patcher.stop()


def test_comfy_cancel_running_job_really_interrupts():
    server = FakeComfyServer(complete_after=99)
    be, patcher = _comfy(server)
    try:
        ref = be.submit("generate", prompt="a long render")
        assert be.poll(ref).state == "running"
        status = be.cancel(ref)
        assert status.state == "cancelled", status
        assert server.interrupts == ["pid-1"]
        # and the server agrees afterwards
        assert be.poll(ref).state == "cancelled"
    finally:
        patcher.stop()


def test_comfy_cancel_queued_job_really_deletes():
    server = FakeComfyServer(complete_after=99)
    be, patcher = _comfy(server)
    try:
        first = be.submit("generate", prompt="occupies the GPU")
        second = be.submit("generate", prompt="waits in queue")
        assert be.poll(second).state == "queued"
        status = be.cancel(second)
        assert status.state == "cancelled", status
        assert server.deleted == ["pid-2"]
        assert server.interrupts == []  # the running job was NOT touched
        assert be.poll(first).state == "running"
        be.cancel(first)  # leave nothing holding the GPU slot
    finally:
        patcher.stop()


def test_comfy_failed_job_reports_provider_error():
    server = FakeComfyServer(error=True)
    be, patcher = _comfy(server)
    try:
        ref = be.submit("generate", prompt="boom")
        be.poll(ref)
        status = be.poll(ref)
        assert status.state == "failed"
        assert status.error_kind == "provider_error"
        assert "OOM boom" in status.error
    finally:
        patcher.stop()


def test_comfy_down_server_submit_fails_typed_and_poll_unknown():
    server = FakeComfyServer(down=True)
    be, patcher = _comfy(server)
    try:
        ref = be.submit("generate", prompt="a cat")
        status = be.poll(ref)
        assert status.state == "failed"
        assert status.error_kind == "unavailable"
        # a job id the server never saw: honestly unknown, not a crash
        assert be.poll("pid-999").state == "unknown"
        with pytest.raises(GenerativeEditError) as ei:
            be.models()
        assert ei.value.kind == "unavailable"
    finally:
        patcher.stop()


def test_comfy_sync_path_still_works():
    """The contract is additive: generate() keeps its fused behavior."""
    server = FakeComfyServer(complete_after=1)
    be, patcher = _comfy(server)
    try:
        out = be.generate("a cat")
        assert isinstance(out, list) and len(out) == 1
        one = be.img2img(_img(), "restyle")
        assert isinstance(one, Image.Image)
    finally:
        patcher.stop()


# ---------------------------------------------------------------------------
# backend_status(): the contract as comparable data
# ---------------------------------------------------------------------------

def test_backend_status_contract_section():
    contract = G.backend_status()["contract"]
    assert set(contract) == {"native", "hf", "diffusers", "comfy",
                             "leonardo", "stability_ai", "nano_banana"}
    for name, entry in contract.items():
        assert entry["conformant"] is True, name
        assert set(entry["ops"]) <= set(CONTRACT_OPS), name
        assert isinstance(entry["gaps"], list), name
    assert contract["comfy"]["cancellable"] is True
    assert contract["native"]["ops"] == ["generate"]
    assert contract["hf"]["cancellable"] is False
    json.dumps(contract)  # the whole section must serialize
