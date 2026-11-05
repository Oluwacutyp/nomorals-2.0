"""Tests for the voice model lifecycle manager (Wave 3a).

storyteller-ml's policy ported as a governed seam: load → LRU evict
→ reload, failure as a recorded state with retry recovery, policy
and state as data. Fake loaders stand in for real backends; the
Piper wiring is exercised with the piper test-suite's own stub
shape (a fake ``piper`` module + protobuf-shaped fake voice files).
"""

from __future__ import annotations

import os
import shutil
import struct
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.voice.lifecycle import (
    LifecyclePolicy,
    ModelLifecycleManager,
)


class _Clock:
    """Deterministic monotonic clock for LRU ordering."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def _manager(max_loaded: int = 2, max_total_bytes: int = 0):
    return ModelLifecycleManager(
        LifecyclePolicy(max_loaded=max_loaded,
                        max_total_bytes=max_total_bytes),
        clock=_Clock())


class TestLifecycleCore(unittest.TestCase):
    def test_load_then_cache_hit_loads_once(self):
        mgr = _manager()
        calls = []
        loader = lambda: calls.append(1) or {"model": "a"}  # noqa: E731
        first = mgr.acquire("a", loader)
        second = mgr.acquire("a", loader)
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(mgr.state_of("a"), "loaded")

    def test_lru_evict_then_reload(self):
        mgr = _manager(max_loaded=2)
        released = []
        mgr.acquire("a", lambda: "A",
                    release=lambda h: released.append(h))
        mgr.acquire("b", lambda: "B")
        mgr.acquire("a", lambda: "A2")          # touch a → b is LRU
        mgr.acquire("c", lambda: "C")            # over policy → evict b
        self.assertEqual(mgr.state_of("b"), "evicted")
        self.assertEqual(mgr.state_of("a"), "loaded")
        self.assertEqual(mgr.state_of("c"), "loaded")
        self.assertEqual(mgr.loaded_keys(), ["c", "a"])
        # reload of an evicted entry runs its loader again
        self.assertEqual(mgr.acquire("b", lambda: "B2"), "B2")
        self.assertEqual(mgr.state_of("b"), "loaded")

    def test_eviction_calls_release_hook(self):
        mgr = _manager(max_loaded=1)
        released = []
        mgr.acquire("a", lambda: "A",
                    release=lambda h: released.append(h))
        mgr.acquire("b", lambda: "B")
        self.assertEqual(released, ["A"])

    def test_byte_budget_evicts_lru(self):
        mgr = _manager(max_loaded=10, max_total_bytes=100)
        mgr.acquire("a", lambda: "A", size_bytes=60)
        mgr.acquire("b", lambda: "B", size_bytes=60)   # 120 > 100
        self.assertEqual(mgr.state_of("a"), "evicted")
        self.assertEqual(mgr.state_of("b"), "loaded")
        self.assertEqual(mgr.status()["total_bytes"], 60)

    def test_failed_load_is_a_state_and_recovers(self):
        mgr = _manager()

        def boom():
            raise RuntimeError("weights corrupt")

        with self.assertRaises(RuntimeError):
            mgr.acquire("x", boom)
        self.assertEqual(mgr.state_of("x"), "failed")
        entry = mgr.status()["entries"][0]
        self.assertEqual(entry["failures"], 1)
        self.assertEqual(entry["attempts"], 1)
        self.assertIn("weights corrupt", entry["last_error"])
        # recovery: the next acquire retries the loader
        self.assertEqual(mgr.acquire("x", lambda: "ok"), "ok")
        self.assertEqual(mgr.state_of("x"), "loaded")
        entry = mgr.status()["entries"][0]
        self.assertEqual(entry["attempts"], 2)
        self.assertEqual(entry["loads"], 1)
        self.assertEqual(entry["last_error"], "")

    def test_peek_never_loads_and_manual_evict(self):
        mgr = _manager()
        self.assertIsNone(mgr.peek("ghost"))
        self.assertEqual(mgr.state_of("ghost"), "absent")
        mgr.acquire("a", lambda: "A")
        self.assertEqual(mgr.peek("a"), "A")
        self.assertTrue(mgr.evict("a"))
        self.assertIsNone(mgr.peek("a"))
        self.assertFalse(mgr.evict("a"))
        self.assertEqual(mgr.evict_all(), 0)

    def test_release_hook_failure_still_evicts(self):
        mgr = _manager(max_loaded=1)

        def bad_release(handle):
            raise RuntimeError("session close exploded")

        mgr.acquire("a", lambda: "A", release=bad_release)
        mgr.acquire("b", lambda: "B")   # forces a's eviction
        self.assertEqual(mgr.state_of("a"), "evicted")

    def test_status_is_data_without_handles(self):
        mgr = _manager()
        mgr.acquire("a", lambda: object(), size_bytes=42)
        status = mgr.status()
        self.assertEqual(status["policy"],
                         {"max_loaded": 2, "max_total_bytes": 0})
        self.assertEqual(status["loaded"], 1)
        self.assertEqual(status["total_bytes"], 42)
        entry = status["entries"][0]
        self.assertEqual(entry["key"], "a")
        self.assertNotIn("handle", entry)
        import json
        json.dumps(status)              # must be plain data


# -- Piper wiring (the governed seam in a real backend) ---------------

class _FakeConfig:
    sample_rate = 22050


class _FakePiperVoice:
    config = _FakeConfig()
    loads: list[str] = []

    def __init__(self, path: str) -> None:
        self.path = path

    def synthesize_stream_raw(self, text):
        yield struct.pack("<8h", *([1000] * 8))


def _stub_piper():
    _FakePiperVoice.loads = []
    mod = types.ModuleType("piper")

    def _load(path):
        _FakePiperVoice.loads.append(path)
        return _FakePiperVoice(path)

    mod.PiperVoice = _FakePiperVoice
    mod.PiperVoice.load = staticmethod(_load)
    sys.modules["piper"] = mod


def _write_voice(directory: str, stem: str) -> str:
    path = os.path.join(directory, stem + ".onnx")
    with open(path, "wb") as fh:
        fh.write(b"\x08\x07fake-onnx")
    return path


class TestPiperLifecycleWiring(unittest.TestCase):
    def setUp(self):
        from nomorals.voice.tts import PiperBackend  # noqa: F401

        self.tmp = tempfile.mkdtemp()
        _stub_piper()
        os.environ["PIPER_VOICES_DIR"] = self.tmp
        os.environ.pop("PIPER_VOICE", None)

    def tearDown(self):
        sys.modules.pop("piper", None)
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("PIPER_VOICES_DIR", None)

    def test_sessions_governed_and_visible(self):
        from nomorals.voice.tts import PiperBackend, VoiceProfile

        _write_voice(self.tmp, "en_US-test-medium")
        backend = PiperBackend()
        status = backend.lifecycle_status()
        self.assertEqual(status["loaded"], 1)
        self.assertEqual(status["entries"][0]["key"], "default")
        self.assertEqual(status["policy"]["max_loaded"], 4)
        # a second voice loads through the same seam
        _write_voice(self.tmp, "en_GB-alan-medium")
        voice = VoiceProfile(name="alan",
                             preset_id="en_GB-alan-medium")
        backend.synthesize("hello", voice)
        keys = {e["key"] for e in backend.lifecycle_status()["entries"]
                if e["state"] == "loaded"}
        self.assertEqual(keys, {"default", "en_GB-alan-medium"})
        # cache hit: resolving again does not reload
        before = len(_FakePiperVoice.loads)
        backend.synthesize("hello again", voice)
        self.assertEqual(len(_FakePiperVoice.loads), before)

    def test_lru_eviction_across_voices(self):
        from nomorals.voice.tts import PiperBackend, VoiceProfile

        _write_voice(self.tmp, "en_US-test-medium")
        backend = PiperBackend()
        stems = [f"v{i}" for i in range(4)]
        for stem in stems:
            _write_voice(self.tmp, stem)
            backend.synthesize(
                "hi", VoiceProfile(name=stem, preset_id=stem))
        # policy is 4 loaded: default + 3 of the 4 named voices
        status = backend.lifecycle_status()
        self.assertEqual(status["loaded"], 4)
        states = {e["key"]: e["state"] for e in status["entries"]}
        self.assertEqual(states["default"], "evicted")

    def test_engine_reports_lifecycle(self):
        from nomorals.voice.tts import UniversalTTS

        _write_voice(self.tmp, "en_US-test-medium")
        engine = UniversalTTS(backend="piper")
        self.assertEqual(engine.lifecycle_status(),
                         {"backend": None, "loaded": False})
        engine.speak("hello lifecycle")
        status = engine.lifecycle_status()
        self.assertEqual(status["backend"], "piper")
        self.assertEqual(status["loaded"], 1)
        info = engine.backend_info("piper")
        self.assertIn("lifecycle", info)
        self.assertEqual(info["lifecycle"]["loaded"], 1)


if __name__ == "__main__":
    unittest.main()
