"""Model services — the storyteller-ml pattern (census §3B.11).

Train-in-Python -> TorchScript export -> register -> serve, behind the
load/execute/unload seam, with the job system as the worker-pod shape.
Torch tests run only where torch is installed (torchvenv); the registry,
validation, and refusal paths are stdlib and run everywhere.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from nomorals.voice.services import (
    ContainerSpec,
    ModelArtifact,
    ModelServer,
    ServiceDeclaration,
    ServiceRegistry,
    TensorSpec,
    control_service,
    default_services_dir,
    export_torchscript_service,
    export_vits_probe_service,
    health,
    service_executor,
)

try:
    from nomorals.voice.vits._torch import torch_available
    _HAVE_TORCH = torch_available()
except ImportError:
    _HAVE_TORCH = False


def _fake_artifact(tmp, name="tiny", version="0.1.0", task="probe",
                   content=b"fake-torchscript-bytes"):
    """A fake artifact: a VALID zip (passes the format sniff) whose
    bytes torch could never load — so registry tests gate torch off
    (see RegistryTests.setUp) and the load check is covered by the
    real VITS probe test instead."""
    import hashlib
    import zipfile
    path = os.path.join(tmp, f"{name}-{version}.pt")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("fake.txt", content)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return ModelArtifact(
        name=name, version=version, task=task, format="torchscript",
        path=path, sha256=digest,
        inputs=[TensorSpec("x", "float32", [None, 4])],
        outputs=[TensorSpec("output_0", "float32", [None, 2])],
        entry="forward", traced=False, source="test")


def _fake_decl(tmp, **kw):
    probe = kw.pop("health_probe", {})
    art = _fake_artifact(tmp, **kw)
    return ServiceDeclaration(
        slug=kw.get("name", "tiny"), version=kw.get("version", "0.1.0"),
        task=kw.get("task", "probe"), artifact=art,
        container=ContainerSpec(owner="Oluwacutyp",
                                build_context="voice/services"),
        health_probe=probe)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name
        # Fake artifacts are valid zips torch could never load; gate
        # torch off so registration tests the registry, not the loader
        # (the loader is covered by the real VITS probe test).
        self._torch_off = mock.patch(
            "nomorals.voice.vits._torch._TORCH_AVAILABLE", False)
        self._torch_off.start()

    def tearDown(self):
        self._torch_off.stop()
        self._td.cleanup()

    def test_register_list_latest_remove(self):
        reg = ServiceRegistry(self.tmp)
        self.assertEqual(reg.list(), [])
        reg.register(_fake_decl(self.tmp, version="0.1.0"))
        reg.register(_fake_decl(self.tmp, version="0.2.0"))
        self.assertEqual([d.version for d in reg.list()],
                         ["0.1.0", "0.2.0"])
        self.assertEqual(reg.resolve("tiny").version, "0.2.0")
        self.assertEqual(reg.resolve("tiny@0.1.0").version, "0.1.0")
        self.assertEqual(reg.get("tiny", "0.1.0").version, "0.1.0")
        removed = reg.remove("tiny", "0.1.0")
        self.assertEqual(removed.version, "0.1.0")
        self.assertEqual([d.version for d in reg.list()], ["0.2.0"])
        # The artifact file stays — removal drops the registration only.
        self.assertTrue(os.path.isfile(
            os.path.join(self.tmp, "tiny-0.1.0.pt")))

    def test_duplicate_version_refused_by_name(self):
        reg = ServiceRegistry(self.tmp)
        reg.register(_fake_decl(self.tmp))
        with self.assertRaises(ValueError) as ctx:
            reg.register(_fake_decl(self.tmp))
        self.assertIn("tiny@0.1.0", str(ctx.exception))

    def test_bad_slug_and_version_refused(self):
        reg = ServiceRegistry(self.tmp)
        bad_slug = _fake_decl(self.tmp)
        bad_slug.slug = "Tiny_Service"
        with self.assertRaises(ValueError):
            reg.register(bad_slug)
        bad_ver = _fake_decl(self.tmp)
        bad_ver.version = "1.0"
        bad_ver.artifact.version = "1.0"
        with self.assertRaises(ValueError):
            reg.register(bad_ver)

    def test_unknown_service_is_key_error(self):
        reg = ServiceRegistry(self.tmp)
        with self.assertRaises(KeyError):
            reg.resolve("nope")
        with self.assertRaises(KeyError):
            reg.remove("nope", "0.1.0")

    def test_missing_artifact_refused(self):
        reg = ServiceRegistry(self.tmp)
        decl = _fake_decl(self.tmp)
        decl.artifact.path = os.path.join(self.tmp, "gone.pt")
        with self.assertRaises(FileNotFoundError):
            reg.register(decl)

    def test_sha_mismatch_refused(self):
        reg = ServiceRegistry(self.tmp)
        decl = _fake_decl(self.tmp)
        with open(decl.artifact.path, "ab") as fh:
            fh.write(b"tampered")
        with self.assertRaises(ValueError) as ctx:
            reg.register(decl)
        self.assertIn("sha256 mismatch", str(ctx.exception))

    def test_non_torchscript_file_refused(self):
        import hashlib
        reg = ServiceRegistry(self.tmp)
        path = os.path.join(self.tmp, "plain-0.1.0.pt")
        content = b"not a zip at all"
        with open(path, "wb") as fh:
            fh.write(content)
        art = ModelArtifact(
            name="plain", version="0.1.0", task="probe",
            format="torchscript", path=path,
            sha256=hashlib.sha256(content).hexdigest())
        decl = ServiceDeclaration(slug="plain", version="0.1.0",
                                  task="probe", artifact=art)
        with self.assertRaises(ValueError) as ctx:
            reg.register(decl)
        self.assertIn("not a TorchScript zip", str(ctx.exception))

    def test_mismatched_artifact_and_service_refused(self):
        reg = ServiceRegistry(self.tmp)
        decl = _fake_decl(self.tmp)
        decl.artifact.name = "other"
        with self.assertRaises(ValueError):
            reg.register(decl)

    def test_corrupt_store_raises_not_silently_drops(self):
        with open(os.path.join(self.tmp, "services.json"), "w") as fh:
            fh.write("{not json")
        with self.assertRaises(ValueError) as ctx:
            ServiceRegistry(self.tmp)
        self.assertIn("corrupt", str(ctx.exception))

    def test_persists_across_instances(self):
        reg = ServiceRegistry(self.tmp)
        reg.register(_fake_decl(self.tmp))
        reg2 = ServiceRegistry(self.tmp)
        self.assertEqual([d.ref for d in reg2.list()], ["tiny@0.1.0"])

    def test_list_by_task(self):
        reg = ServiceRegistry(self.tmp)
        reg.register(_fake_decl(self.tmp, name="a", task="tts-mel"))
        reg.register(_fake_decl(self.tmp, name="b", task="probe"))
        self.assertEqual([d.slug for d in reg.list(task="tts-mel")], ["a"])


class ContainerSpecTests(unittest.TestCase):
    def test_image_refs_mirror_the_workflow(self):
        spec = ContainerSpec(owner="Oluwacutyp",
                             base_image="pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime")
        self.assertEqual(spec.image_refs("gpt-sovits", "1.2.3", "abc123def456"),
                         ["ghcr.io/oluwacutyp/gpt-sovits:1.2.3",
                          "ghcr.io/oluwacutyp/gpt-sovits:abc123def456"])

    def test_no_owner_is_a_local_image(self):
        spec = ContainerSpec()
        self.assertEqual(spec.image_name("tiny"), "local/tiny")
        self.assertEqual(spec.image_refs("tiny", "0.1.0", "deadbeef"),
                         ["local/tiny:0.1.0", "local/tiny:deadbeef"])


class TensorSpecTests(unittest.TestCase):
    def test_wildcard_dims(self):
        spec = TensorSpec("x", "float32", [None, 4])
        self.assertIsNone(spec.matches("float32", [7, 4]))
        self.assertIsNotNone(spec.matches("float32", [7, 5]))
        self.assertIsNotNone(spec.matches("int64", [7, 4]))
        self.assertIsNotNone(spec.matches("float32", [7]))

    def test_exact_shapes(self):
        spec = TensorSpec("x", "int64", [1, 12])
        self.assertIsNone(spec.matches("int64", [1, 12]))
        self.assertIsNotNone(spec.matches("int64", [1, 13]))


class NoTorchTests(unittest.TestCase):
    """Refusals name torch when torch is absent — simulated by flag."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name
        self._torch_off = mock.patch(
            "nomorals.voice.vits._torch._TORCH_AVAILABLE", False)
        self._torch_off.start()

    def tearDown(self):
        self._torch_off.stop()
        self._td.cleanup()

    def _no_torch(self):
        return mock.patch(
            "nomorals.voice.vits._torch._TORCH_AVAILABLE", False)

    def test_export_refuses_without_torch(self):
        with self._no_torch(), self.assertRaises(ImportError) as ctx:
            export_torchscript_service(object(), {"x": None},
                                       "/tmp/x.pt", name="t",
                                       task="probe")
        self.assertIn("torch", str(ctx.exception))

    def test_serve_refuses_without_torch(self):
        reg = ServiceRegistry(self.tmp)
        reg.register(_fake_decl(self.tmp))
        server = ModelServer(reg)
        with self._no_torch(), self.assertRaises(ImportError) as ctx:
            server.execute("tiny", {"x": {"dtype": "float32",
                                          "shape": [1, 4],
                                          "values": [[1, 2, 3, 4]]}})
        self.assertIn("torch", str(ctx.exception))

    def test_health_reports_torch_missing(self):
        reg = ServiceRegistry(self.tmp)
        reg.register(_fake_decl(
            self.tmp,
            health_probe={"inputs": {"x": {"dtype": "float32",
                                           "shape": [1, 4],
                                           "values": [[1, 2, 3, 4]]}}}))
        with self._no_torch():
            result = health("tiny", registry=reg)
        self.assertFalse(result["ok"])
        self.assertIn("torch", result["error"])


class ControlServiceTests(unittest.TestCase):
    def setUp(self):
        self._torch_off = mock.patch(
            "nomorals.voice.vits._torch._TORCH_AVAILABLE", False)
        self._torch_off.start()

    def tearDown(self):
        self._torch_off.stop()

    def test_list_empty_health_unknown_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("no model services", control_service("list",
                                                               services_dir=tmp))
            self.assertIn("usage", control_service("health",
                                                   services_dir=tmp))
            result = control_service("health nope", services_dir=tmp)
            self.assertIn("unhealthy", result)
            status = control_service("status", services_dir=tmp)
            self.assertIn("0 service(s) registered", status)
            self.assertIn("usage", control_service("bogus",
                                                   services_dir=tmp))

    def test_list_shows_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = ServiceRegistry(tmp)
            reg.register(_fake_decl(tmp))
            out = control_service("list", services_dir=tmp)
            self.assertIn("tiny@0.1.0", out)
            self.assertIn("probe", out)


class CliParseTests(unittest.TestCase):
    def test_voice_service_parses(self):
        from nomorals.cmdline.parser import _parser
        args = _parser().parse_args(["voice", "service", "list"])
        self.assertEqual(args.voice_action, "service")
        self.assertEqual(args.voice_service_action, "list")
        args = _parser().parse_args(
            ["voice", "service", "serve", "tiny@0.1.0",
             "--inputs", "{}"])
        self.assertEqual(args.voice_service_action, "serve")
        self.assertEqual(args.ref, "tiny@0.1.0")
        args = _parser().parse_args(["voice", "service", "export-vits-probe"])
        self.assertEqual(args.voice_service_action, "export-vits-probe")


@unittest.skipUnless(_HAVE_TORCH, "torch not installed")
class TorchExportServeTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def _tiny_module(self):
        import torch
        from torch import nn

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 2, bias=False)

            def forward(self, x):
                return self.lin(x) * 2.0

        torch.manual_seed(0)
        return Tiny()

    def _register_tiny(self, method="script", version="0.1.0"):
        import torch
        module = self._tiny_module()
        example = {"x": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}
        out = os.path.join(self.tmp, f"tiny-{version}.pt")
        artifact = export_torchscript_service(
            module, example, out, method=method, task="probe",
            name="tiny", version=version, source="test")
        decl = ServiceDeclaration(
            slug="tiny", version=version, task="probe", artifact=artifact,
            container=ContainerSpec(owner="Oluwacutyp"))
        ServiceRegistry(self.tmp).register(decl)
        return artifact

    def test_script_export_serve_round_trip(self):
        import torch
        self._register_tiny()
        reg = ServiceRegistry(self.tmp)
        server = ModelServer(reg)
        payload = {"x": {"dtype": "float32", "shape": [1, 4],
                         "values": [[1.0, 2.0, 3.0, 4.0]]}}
        result = server.execute("tiny", payload)
        self.assertEqual(result["service"], "tiny")
        self.assertEqual(result["version"], "0.1.0")
        self.assertEqual(result["output_shapes"]["output_0"], [1, 2])
        # The served values equal the module run directly.
        module = self._tiny_module()
        with torch.no_grad():
            want = module(torch.tensor([[1.0, 2.0, 3.0, 4.0]])).tolist()
        self.assertEqual(result["outputs"]["output_0"]["values"], want)
        # A cache hit serves the same handle without reloading.
        self.assertIs(server.load("tiny"), server.load("tiny"))

    def test_script_is_shape_polymorphic(self):
        self._register_tiny()
        server = ModelServer(ServiceRegistry(self.tmp))
        payload = {"x": {"dtype": "float32", "shape": [3, 4],
                         "values": [[1.0, 2.0, 3.0, 4.0]] * 3}}
        result = server.execute("tiny", payload)
        self.assertEqual(result["output_shapes"]["output_0"], [3, 2])

    def test_traced_is_shape_exact(self):
        self._register_tiny(method="trace")
        server = ModelServer(ServiceRegistry(self.tmp))
        bad = {"x": {"dtype": "float32", "shape": [2, 4],
                     "values": [[1.0, 2.0, 3.0, 4.0]] * 2}}
        with self.assertRaises(ValueError) as ctx:
            server.execute("tiny", bad)
        self.assertIn("shape", str(ctx.exception))

    def test_wrong_dtype_and_unknown_input_refused(self):
        self._register_tiny()
        server = ModelServer(ServiceRegistry(self.tmp))
        with self.assertRaises(ValueError):
            server.execute("tiny", {"x": {"dtype": "int64",
                                          "shape": [1, 4],
                                          "values": [[1, 2, 3, 4]]}})
        with self.assertRaises(ValueError):
            server.execute("tiny", {"y": {"dtype": "float32",
                                          "shape": [1, 4],
                                          "values": [[1, 2, 3, 4]]}})
        with self.assertRaises(ValueError):
            server.execute("tiny", {})

    def test_lru_eviction_and_unload(self):
        from nomorals.voice.lifecycle import LifecyclePolicy
        self._register_tiny(version="0.1.0")
        self._register_tiny(version="0.2.0")
        reg = ServiceRegistry(self.tmp)
        server = ModelServer(reg, policy=LifecyclePolicy(max_loaded=1))
        server.load("tiny@0.1.0")
        self.assertEqual(server._manager.loaded_keys(), ["tiny@0.1.0"])
        server.load("tiny@0.2.0")
        self.assertEqual(server._manager.loaded_keys(), ["tiny@0.2.0"])
        self.assertEqual(server._manager.state_of("tiny@0.1.0"), "evicted")
        self.assertTrue(server.unload("tiny@0.2.0"))
        self.assertEqual(server._manager.state_of("tiny@0.2.0"), "evicted")
        self.assertFalse(server.unload("tiny@0.2.0"))

    def test_health_ok(self):
        import torch
        module = self._tiny_module()
        example = {"x": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}
        out = os.path.join(self.tmp, "tiny-0.1.0.pt")
        artifact = export_torchscript_service(
            module, example, out, method="script", task="probe",
            name="tiny", version="0.1.0", source="test")
        decl = ServiceDeclaration(
            slug="tiny", version="0.1.0", task="probe", artifact=artifact,
            health_probe={"inputs": {
                "x": {"dtype": "float32", "shape": [1, 4],
                      "values": [[1.0, 2.0, 3.0, 4.0]]}}})
        ServiceRegistry(self.tmp).register(decl)
        result = health("tiny", registry=ServiceRegistry(self.tmp))
        self.assertTrue(result["ok"])
        self.assertEqual(result["service"], "tiny")
        self.assertEqual(result["output_shapes"]["output_0"], [1, 2])

    def test_job_executor_drives_a_real_job(self):
        import uuid

        from nomorals.core.inference_jobs import InferenceJobStore
        self._register_tiny()
        reg = ServiceRegistry(self.tmp)
        store = InferenceJobStore(path=None)
        job = store.submit(
            "audio_generation", "tiny@0.1.0", "",
            params={"inputs": {"x": {"dtype": "float32", "shape": [1, 4],
                                     "values": [[1.0, 2.0, 3.0, 4.0]]}}},
            idempotency_token=uuid.uuid4().hex)
        done = store.run_local(job.token, service_executor(reg))
        self.assertEqual(done.status, "complete_success")
        self.assertEqual(done.result["service"], "tiny")
        self.assertEqual(done.result["output_shapes"]["output_0"], [1, 2])

    def test_job_executor_bad_job_is_executor_error(self):
        import uuid

        from nomorals.core.inference_jobs import InferenceJobStore
        self._register_tiny()
        reg = ServiceRegistry(self.tmp)
        store = InferenceJobStore(path=None)
        job = store.submit("audio_generation", "tiny@0.1.0", "",
                           params={},
                           idempotency_token=uuid.uuid4().hex)
        done = store.run_local(job.token, service_executor(reg))
        self.assertEqual(done.status, "attempt_failed")
        self.assertEqual(done.failure_category, "executor_error")

    def test_default_services_dir(self):
        self.assertTrue(default_services_dir().endswith(
            os.path.join("voice", "services")))


@unittest.skipUnless(_HAVE_TORCH, "torch not installed")
class VitsProbeServiceTests(unittest.TestCase):
    """The one real end-to-end service: real VITS architecture, random
    weights, traced, registered, served. Slow (~240MB artifact) — runs
    with TMPDIR on disk, never /tmp."""

    def test_probe_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            decl = export_vits_probe_service(tmp, seed=0)
            self.assertEqual(decl.ref, "vits-probe@0.1.0")
            self.assertEqual(decl.task, "tts-spectral")
            self.assertTrue(decl.artifact.traced)
            self.assertEqual(decl.artifact.entry, "infer_ts_noistft")
            self.assertEqual([t.name for t in decl.artifact.outputs],
                             ["spec", "phase", "x1", "attn"])
            self.assertTrue(os.path.isfile(decl.artifact.path))
            self.assertGreater(os.path.getsize(decl.artifact.path),
                               100_000_000)
            reg = ServiceRegistry(tmp)
            self.assertEqual(reg.resolve("vits-probe").ref,
                             "vits-probe@0.1.0")
            # Re-registering the same service refuses (duplicate).
            with self.assertRaises(ValueError):
                reg.register(decl)
            # Health runs the declared probe through the server.
            result = health("vits-probe", registry=reg)
            self.assertTrue(result["ok"], result)
            shapes = result["output_shapes"]
            # spec is the iSTFT magnitude: [batch, n_fft//2+1, frames]
            # — NOT an 80-bin mel spectrogram (that was my first,
            # wrong assumption; the generator's post_n_fft is 32).
            self.assertEqual(len(shapes["spec"]), 3)
            self.assertEqual(shapes["spec"][0], 1)
            self.assertEqual(shapes["spec"][1], 17)
            self.assertEqual(shapes["phase"], shapes["spec"])
            # Serve returns the same STRUCTURE (the traced module draws
            # duration noise, so frame counts vary run to run — never
            # assert two separate runs have identical shapes).
            server = ModelServer(reg)
            served = server.execute("vits-probe",
                                    decl.health_probe["inputs"])
            served_shapes = served["output_shapes"]
            self.assertEqual(len(served_shapes["spec"]), 3)
            self.assertEqual(served_shapes["spec"][0], 1)
            self.assertEqual(served_shapes["spec"][1], 17)
            self.assertEqual(served_shapes["phase"],
                             served_shapes["spec"])
            spec_values = served["outputs"]["spec"]["values"]
            self.assertEqual(len(spec_values), 1)
            self.assertEqual(len(spec_values[0]), 17)
            self.assertTrue(all(
                v == v for row in spec_values[0] for v in row),
                "spec carries NaNs")
            # Chat surface sees it too.
            out = control_service("list", services_dir=tmp)
            self.assertIn("vits-probe@0.1.0", out)
            out = control_service("health vits-probe", services_dir=tmp)
            self.assertIn("healthy", out)


if __name__ == "__main__":
    unittest.main()
