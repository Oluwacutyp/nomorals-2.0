"""Pass #49 — setup wizard + first-run surface.

Pins the defects found by driving the real wizard the way a fresh user
would (scripted stdin, temp NM_HOME + temp cwd), then booting from
exactly what the wizard wrote:

* an out-of-range provider selection fell through every branch and
  printed "✅ Configuration written" over an empty write (rc 0);
* an empty API key / HF token was accepted, written, and blessed —
  the install then booted to the mock (or a keyless provider);
* the OpenAI branch never wrote NM_OPENAI_BASE_URL, so the booted
  "openai" provider pointed at the localhost Ollama-shim default
  (http://localhost:11434/v1) carrying the owner's OpenAI key;
* ``nm models --promote-local`` promoted GGUF paths that do not
  exist ("the next boot runs entirely on it") and claimed the write
  landed in "the home .env" even when it landed in the cwd .env;
* EOF/Ctrl-C at a wizard prompt died with a raw EOFError traceback;
* the shipped .env.example taught the retired HF endpoint
  (api-inference.huggingface.co, 410 Gone) as its active default.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock


def _wizard_context(home: str):
    return SimpleNamespace(
        router=SimpleNamespace(active_model="mock"),
        settings=SimpleNamespace(home=home),
        db=None, bus=None)


def _run_wizard(cwd: str, home: str, answers: list):
    """Drive _cmd_setup with scripted stdin in a clean env."""
    from nomorals.cmdline.commands.doctor import _cmd_setup
    old_cwd = os.getcwd()
    os.chdir(cwd)
    try:
        with mock.patch.dict(os.environ, {"NM_HOME": home}, clear=True), \
                mock.patch("builtins.input", side_effect=list(answers)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_setup(SimpleNamespace(), _wizard_context(home))
        return rc, buf.getvalue()
    finally:
        os.chdir(old_cwd)


def _env_text(cwd: str, home: str) -> str:
    for path in (os.path.join(cwd, ".env"), os.path.join(home, ".env")):
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                return fh.read()
    return ""


class TestWizardRefusals(unittest.TestCase):
    def test_invalid_provider_choice_writes_nothing(self):
        with tempfile.TemporaryDirectory() as cwd, \
                tempfile.TemporaryDirectory() as home:
            rc, out = _run_wizard(cwd, home, ["9", "n"])
            self.assertEqual(rc, 1)
            self.assertNotIn("✅ Configuration written", out)
            self.assertEqual(_env_text(cwd, home), "")

    def test_empty_openrouter_key_refused(self):
        with tempfile.TemporaryDirectory() as cwd, \
                tempfile.TemporaryDirectory() as home:
            rc, out = _run_wizard(cwd, home, ["1", "", "1", "n"])
            self.assertEqual(rc, 1)
            self.assertNotIn("✅ Configuration written", out)
            self.assertNotIn("NM_LLM_PROVIDER", _env_text(cwd, home))

    def test_empty_hf_token_refused(self):
        with tempfile.TemporaryDirectory() as cwd, \
                tempfile.TemporaryDirectory() as home:
            rc, out = _run_wizard(cwd, home, ["2", "", "1", "n"])
            self.assertEqual(rc, 1)
            self.assertNotIn("✅ Configuration written", out)
            self.assertNotIn("NM_LLM_PROVIDER", _env_text(cwd, home))

    def test_empty_openai_key_refused(self):
        with tempfile.TemporaryDirectory() as cwd, \
                tempfile.TemporaryDirectory() as home:
            rc, out = _run_wizard(cwd, home, ["4", "", "n"])
            self.assertEqual(rc, 1)
            self.assertNotIn("✅ Configuration written", out)
            self.assertNotIn("NM_LLM_PROVIDER", _env_text(cwd, home))

    def test_eof_at_prompt_aborts_cleanly(self):
        with tempfile.TemporaryDirectory() as cwd, \
                tempfile.TemporaryDirectory() as home:
            rc, out = _run_wizard(
                cwd, home, ["1", EOFError("stdin closed")])
            self.assertEqual(rc, 1)
            self.assertIn("abort", out.lower())
            self.assertEqual(_env_text(cwd, home), "")


class TestOpenAiBranchBootsRealEndpoint(unittest.TestCase):
    def test_openai_wizard_output_boots_to_api_openai(self):
        with tempfile.TemporaryDirectory() as cwd, \
                tempfile.TemporaryDirectory() as home:
            rc, _out = _run_wizard(cwd, home, ["4", "sk-testkey", "n"])
            self.assertEqual(rc, 0)
            content = _env_text(cwd, home)
            self.assertIn(
                "NM_OPENAI_BASE_URL=https://api.openai.com/v1", content)
            # Boot from exactly what the wizard wrote.
            from nomorals.core.config import load_settings
            from nomorals.core.events import EventBus
            from nomorals.agents.context import build_router
            old_cwd = os.getcwd()
            os.chdir(cwd)
            try:
                with mock.patch.dict(os.environ, {"NM_HOME": home},
                                     clear=True):
                    settings = load_settings()
                    router = build_router(settings, EventBus())
            finally:
                os.chdir(old_cwd)
            self.assertEqual(settings.llm.provider, "openai")
            provider = router.get("openai")
            self.assertIsNotNone(provider)
            self.assertEqual(provider.base_url,
                             "https://api.openai.com/v1")
            self.assertEqual(provider.model, "gpt-4o-mini")


class TestPromoteLocal(unittest.TestCase):
    def _args(self, path: str):
        return SimpleNamespace(
            promote_local=path, lora="", activate="", catalog=False,
            search="", kind="", max_params=0, json=False)

    def _context(self, home: str):
        from nomorals.storage.db import Database
        db = Database(os.path.join(home, "data", "nomorals.db"))
        return SimpleNamespace(
            db=db, settings=SimpleNamespace(home=home))

    def test_missing_gguf_is_refused_and_writes_nothing(self):
        from nomorals.cmdline.commands.doctor import _cmd_models
        with tempfile.TemporaryDirectory() as cwd, \
                tempfile.TemporaryDirectory() as home:
            old_cwd = os.getcwd()
            os.chdir(cwd)
            try:
                with mock.patch.dict(os.environ, {"NM_HOME": home},
                                     clear=True):
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        rc = _cmd_models(
                            self._args("/nonexistent/deep/model.gguf"),
                            self._context(home))
            finally:
                os.chdir(old_cwd)
            self.assertEqual(rc, 1)
            self.assertNotIn("promoted", buf.getvalue())
            self.assertEqual(_env_text(cwd, home), "")

    def test_success_message_names_the_actual_file_written(self):
        from nomorals.cmdline.commands.doctor import _cmd_models
        with tempfile.TemporaryDirectory() as cwd, \
                tempfile.TemporaryDirectory() as home:
            gguf = os.path.join(cwd, "model.gguf")
            with open(gguf, "wb") as fh:
                fh.write(b"GGUF")
            # A cwd .env exists, so the boot file is the cwd one —
            # the message must name THAT file, not "the home .env".
            with open(os.path.join(cwd, ".env"), "w",
                      encoding="utf-8") as fh:
                fh.write("NM_EXISTING=1\n")
            old_cwd = os.getcwd()
            os.chdir(cwd)
            try:
                with mock.patch.dict(os.environ, {"NM_HOME": home},
                                     clear=True):
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        rc = _cmd_models(self._args(gguf),
                                         self._context(home))
            finally:
                os.chdir(old_cwd)
            self.assertEqual(rc, 0)
            self.assertIn(os.path.join(cwd, ".env"), buf.getvalue())
            with open(os.path.join(cwd, ".env"), encoding="utf-8") as fh:
                self.assertIn(f"NM_LLM_LOCAL_MODEL={gguf}", fh.read())


class TestShippedExample(unittest.TestCase):
    def test_env_example_hf_base_url_is_the_live_endpoint(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, ".env.example"),
                  encoding="utf-8") as fh:
            text = fh.read()
        self.assertNotIn("api-inference.huggingface.co", text)
        self.assertIn(
            "NM_HF_BASE_URL=https://router.huggingface.co/hf-inference",
            text)


if __name__ == "__main__":
    unittest.main()
