"""Tests for the model lifecycle (nomorals.llm.lifecycle).

All offline: a FakeProvisioner stands in for llama-server, and a real temp
file stands in for a GGUF.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from nomorals.core.errors import NotFound, ValidationError
from nomorals.llm.lifecycle import (
    LifecycleError,
    ManagedModel,
    ModelLifecycle,
)


class FakeProvisioner:
    """In-memory stand-in for LocalGGUFProvisioner."""

    def __init__(self, *, warm_ok: bool = True) -> None:
        self.warm_ok = warm_ok
        self.loaded: dict[str, ManagedModel] = {}
        self.calls: list[str] = []

    def load(self, model: ManagedModel):
        self.calls.append(f"load:{model.id}")
        handle = object()
        self.loaded[model.id] = model
        return handle

    def warm(self, model: ManagedModel, handle) -> bool:
        self.calls.append(f"warm:{model.id}")
        return self.warm_ok and model.id in self.loaded

    def unload(self, model: ManagedModel, handle) -> None:
        self.calls.append(f"unload:{model.id}")
        self.loaded.pop(model.id, None)


class LifecycleTransitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-lifecycle-")
        self.addCleanup(self.tmp.cleanup)
        self.gguf = Path(self.tmp.name) / "tiny.gguf"
        self.gguf.write_bytes(b"GGUF" + os.urandom(4096))
        self.expected_sha = hashlib.sha256(self.gguf.read_bytes()).hexdigest()
        self.lc = ModelLifecycle(
            Path(self.tmp.name) / "lc.db",
            provisioner=FakeProvisioner(),
        )

    def test_full_walk_registered_to_warm(self):
        m = self.lc.add_gguf(self.gguf, model_id="tiny", quant="Q4_K_M",
                             context_len=8192)
        self.assertEqual(m.status, "registered")
        self.assertEqual(m.provider, "llama_cpp")
        self.assertEqual(m.capabilities, ["chat", "code"])

        m = self.lc.download("tiny")
        self.assertEqual(m.status, "downloaded")
        self.assertGreater(m.size_bytes, 0)

        m = self.lc.verify("tiny")
        self.assertEqual(m.status, "verified")
        self.assertEqual(m.sha256, self.expected_sha)

        m = self.lc.load("tiny")
        self.assertEqual(m.status, "loaded")

        m = self.lc.warm("tiny")
        self.assertEqual(m.status, "warm")

        m = self.lc.unload("tiny")
        self.assertEqual(m.status, "verified")  # bytes kept, server stopped

        events = self.lc.history("tiny")
        stages = [(e["from_status"], e["to_status"]) for e in events]
        self.assertIn(("registered", "downloaded"), stages)
        self.assertIn(("downloaded", "verified"), stages)
        self.assertIn(("verified", "loaded"), stages)
        self.assertIn(("loaded", "warm"), stages)
        self.assertIn(("warm", "verified"), stages)

    def test_illegal_transitions_raise(self):
        self.lc.add_gguf(self.gguf, model_id="tiny")
        with self.assertRaises(LifecycleError):
            self.lc.load("tiny")  # registered → loaded skips steps
        with self.assertRaises(LifecycleError):
            self.lc.warm("tiny")  # registered → warm skips steps
        with self.assertRaises(LifecycleError):
            self.lc.verify("tiny")  # registered → verified skips download
        self.assertEqual(self.lc.get("tiny").status, "registered")

    def test_verify_rejects_wrong_expected_sha(self):
        self.lc.add_gguf(self.gguf, model_id="tiny")
        self.lc.download("tiny")
        with self.assertRaises(LifecycleError):
            self.lc.verify("tiny", expected_sha256="0" * 64)
        self.assertEqual(self.lc.get("tiny").status, "failed")
        # retry re-walks from registered
        self.lc.retry("tiny")
        self.assertEqual(self.lc.get("tiny").status, "registered")

    def test_verify_detects_changed_artifact(self):
        self.lc.add_gguf(self.gguf, model_id="tiny")
        self.lc.download("tiny")
        self.lc.verify("tiny")
        self.gguf.write_bytes(b"GGUF-tampered")
        with self.assertRaises(LifecycleError):
            self.lc.verify("tiny")

    def test_download_missing_local_file_fails(self):
        self.lc.add_gguf("/nonexistent/ghost.gguf", model_id="ghost")
        with self.assertRaises(LifecycleError):
            self.lc.download("ghost")
        self.assertEqual(self.lc.get("ghost").status, "failed")

    def test_add_hf_id_registers_without_path(self):
        m = self.lc.add("someone/some-model-9b", capabilities=["chat"])
        self.assertEqual(m.status, "registered")
        self.assertEqual(m.provider, "hf_serverless")
        self.assertEqual(m.path, "")

    def test_add_requires_source(self):
        with self.assertRaises(ValidationError):
            self.lc.add("")

    def test_unknown_model_raises_not_found(self):
        with self.assertRaises(NotFound):
            self.lc.get("nope")

    def test_warm_failure_marks_failed(self):
        lc = ModelLifecycle(provisioner=FakeProvisioner(warm_ok=False))
        gguf = Path(self.tmp.name) / "cold.gguf"
        gguf.write_bytes(b"GGUF" + os.urandom(64))
        lc.add_gguf(gguf, model_id="cold")
        lc.download("cold")
        lc.verify("cold")
        lc.load("cold")
        with self.assertRaises(LifecycleError):
            lc.warm("cold")
        self.assertEqual(lc.get("cold").status, "failed")


class PromoteRollbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-lifecycle-")
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "lc.db"
        self.lc = ModelLifecycle(self.db_path, provisioner=FakeProvisioner())
        for name in ("alpha", "beta"):
            gguf = Path(self.tmp.name) / f"{name}.gguf"
            gguf.write_bytes(b"GGUF" + os.urandom(128))
            self.lc.add_gguf(gguf, model_id=name)
            self.lc.download(name)
            self.lc.verify(name)

    def test_promote_and_rollback_restores_previous(self):
        self.lc.promote("alpha")
        self.assertEqual(self.lc.primary, "alpha")
        self.lc.promote("beta")
        self.assertEqual(self.lc.primary, "beta")
        restored = self.lc.rollback()
        self.assertEqual(restored.id, "alpha")
        self.assertEqual(self.lc.primary, "alpha")

    def test_rollback_without_history_raises(self):
        with self.assertRaises(LifecycleError):
            self.lc.rollback()

    def test_promote_requires_verified(self):
        gguf = Path(self.tmp.name) / "raw.gguf"
        gguf.write_bytes(b"GGUF")
        self.lc.add_gguf(gguf, model_id="raw")
        with self.assertRaises(LifecycleError):
            self.lc.promote("raw")

    def test_primary_and_history_survive_restart(self):
        self.lc.promote("alpha")
        self.lc.promote("beta")
        fresh = ModelLifecycle(self.db_path, provisioner=FakeProvisioner())
        self.assertEqual(fresh.primary, "beta")
        restored = fresh.rollback()
        self.assertEqual(restored.id, "alpha")

    def test_remove_primary_refused(self):
        self.lc.promote("alpha")
        with self.assertRaises(LifecycleError):
            self.lc.remove("alpha")

    def test_remove_unloads_and_forgets(self):
        prov = self.lc.provisioner
        self.lc.load("alpha")
        self.lc.warm("alpha")
        self.assertTrue(self.lc.remove("alpha"))
        with self.assertRaises(NotFound):
            self.lc.get("alpha")
        self.assertNotIn("alpha", prov.loaded)


if __name__ == "__main__":
    unittest.main()
