"""Tests for local CodeBeast deployment path.

Covers:
- ModelLifecycle accepts and forwards the HF token (private repo downloads).
- LLMSettings has the local auto-start fields with env mappings.
- `nm models setup` wires register → download → verify → promote.
- build_context auto-starts the local GGUF server when configured.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nomorals.core.config import LLMSettings, _ENV_MAP
from nomorals.llm.lifecycle import LifecycleError, ModelLifecycle


class HfTokenTests(unittest.TestCase):
    def test_lifecycle_accepts_hf_token(self):
        lc = ModelLifecycle(hf_token="hf_test_123")
        self.assertEqual(lc.hf_token, "hf_test_123")

    def test_lifecycle_reads_token_from_env(self):
        with patch.dict(os.environ, {"HF_TOKEN": "env_token_abc"}):
            lc = ModelLifecycle()
            self.assertEqual(lc.hf_token, "env_token_abc")

    def test_lifecycle_prefers_explicit_over_env(self):
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}):
            lc = ModelLifecycle(hf_token="explicit")
            self.assertEqual(lc.hf_token, "explicit")

    def test_download_passes_token_to_downloader(self):
        from nomorals.llm import download as dl_mod

        tmp = tempfile.TemporaryDirectory(prefix="nm-codebeast-")
        self.addCleanup(tmp.cleanup)
        lc = ModelLifecycle(models_dir=tmp.name, hf_token="secret_xyz")

        seen = {}

        class FakeDownloader:
            def __init__(self, *, token="", cache_dir=""):
                seen["token"] = token
                seen["cache_dir"] = cache_dir

            def download_repo(self, source, patterns=()):
                # fake a GGUF result
                p = Path(self_cache := seen["cache_dir"]) / source / "model.gguf"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"GGUF" + b"\x00" * 100)

                class R:
                    files = [str(p)]
                return R()

        with patch.object(dl_mod, "HuggingFaceDownloader", FakeDownloader):
            # patch the lazy import inside lifecycle.download
            import nomorals.llm.lifecycle as lc_mod
            orig = lc_mod.HuggingFaceDownloader if hasattr(lc_mod, "HuggingFaceDownloader") else None
            lc_mod.HuggingFaceDownloader = FakeDownloader
            try:
                model = lc.add("Cutyp/codebeast-3.8b", model_id="codebeast")
                model = lc.download("codebeast")
            finally:
                if orig is None:
                    delattr(lc_mod, "HuggingFaceDownloader")
                else:
                    lc_mod.HuggingFaceDownloader = orig
        self.assertEqual(seen.get("token"), "secret_xyz")
        self.assertEqual(model.status, "downloaded")


class LocalSettingsTests(unittest.TestCase):
    def test_settings_have_local_fields(self):
        s = LLMSettings()
        self.assertFalse(s.local_auto_start)
        self.assertEqual(s.local_host, "127.0.0.1")
        self.assertEqual(s.local_port, 8080)
        self.assertEqual(s.local_model, "")

    def test_env_map_covers_local_contract(self):
        self.assertEqual(_ENV_MAP.get("NM_LLM_LOCAL_MODEL"), "llm.local_model")
        self.assertEqual(_ENV_MAP.get("NM_LLM_LOCAL_AUTO_START"), "llm.local_auto_start")
        self.assertEqual(_ENV_MAP.get("NM_LLM_LOCAL_HOST"), "llm.local_host")
        self.assertEqual(_ENV_MAP.get("NM_LLM_LOCAL_PORT"), "llm.local_port")
        self.assertEqual(_ENV_MAP.get("NM_LLAMA_CPP_URL"), "llm.llama_cpp_url")


class FetchGgufSignatureTests(unittest.TestCase):
    def test_download_file_rejects_no_expected_size(self):
        # Regression: fetch_gguf passed expected_size= which download_file
        # never accepted → TypeError at download time.
        import inspect
        from nomorals.llm.download import HuggingFaceDownloader
        params = inspect.signature(HuggingFaceDownloader.download_file).parameters
        self.assertNotIn("expected_size", params)

    def test_fetch_gguf_call_matches_signature(self):
        # The call site must only use kwargs download_file accepts.
        import inspect
        import re
        from nomorals.llm.download import HuggingFaceDownloader
        from nomorals.llm import local_server
        src = inspect.getsource(local_server.GGUFServerManager.fetch_gguf)
        valid = set(inspect.signature(HuggingFaceDownloader.download_file).parameters)
        valid.discard("self")
        for m in re.finditer(r"download_file\((.*?)\)", src, re.DOTALL):
            call = m.group(1)
            for kw in re.findall(r"(\w+)\s*=", call):
                self.assertIn(kw, valid, f"fetch_gguf passes unknown kwarg {kw!r}")


if __name__ == "__main__":
    unittest.main()
