"""Provider registry — construction parity with the legacy chain
(Echelon Wave 4).

The legacy ``build_provider()`` was a hand-written if-chain over
eleven kinds plus their aliases.  These pins encode that chain's
exact observable behavior — kind/alias → class, defaults, error
strings — and prove the declarative registry reproduces it, that
adding a provider is a data-only change, and that both callers of
the old chain (``llm/defaults.py`` chain builder,
``agents/context.py`` settings chain) still behave identically.
"""

from __future__ import annotations

import json
import os
import unittest
from contextlib import contextmanager
from unittest import mock

from nomorals.llm.providers import (
    build_provider,
    canonical_kind,
    expected_kinds,
    get_provider_spec,
    list_provider_specs,
    register_provider_spec,
    render_status_markdown,
    unregister_provider_spec,
)
from nomorals.llm.providers.anthropic import AnthropicProvider
from nomorals.llm.providers.deepseek import DeepSeekProvider
from nomorals.llm.providers.gemini import GeminiProvider
from nomorals.llm.providers.groq import GroqProvider
from nomorals.llm.providers.hf_serverless import HFServerlessProvider
from nomorals.llm.providers.llama_cpp import LlamaCppProvider
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.providers.ocr import OCRProvider
from nomorals.llm.providers.ollama import OllamaProvider
from nomorals.llm.providers.openai_compat import OpenAICompatProvider
from nomorals.llm.providers.openrouter import OpenRouterProvider
from nomorals.llm.providers.registry import ProviderSpec

# The legacy chain's full kind/alias → class matrix, transcribed from
# the pre-Wave-4 if-chain.  This table IS the parity contract.
LEGACY_MATRIX = {
    "mock": MockProvider, "offline": MockProvider, "test": MockProvider,
    "ollama": OllamaProvider, "ollama_native": OllamaProvider,
    "groq": GroqProvider,
    "anthropic": AnthropicProvider, "claude": AnthropicProvider,
    "deepseek": DeepSeekProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
    "openai": OpenAICompatProvider, "openai_compat": OpenAICompatProvider,
    "vllm": OpenAICompatProvider, "lmstudio": OpenAICompatProvider,
    "ollama_compat": OpenAICompatProvider,
    "hf": HFServerlessProvider, "hf_serverless": HFServerlessProvider,
    "huggingface": HFServerlessProvider, "hf_endpoint": HFServerlessProvider,
    "llama_cpp": LlamaCppProvider, "llamacpp": LlamaCppProvider,
    "gguf": LlamaCppProvider,
    "ocr": OCRProvider, "tesseract": OCRProvider,
}

EXPECTED_KINDS = ["mock", "ollama", "groq", "anthropic", "deepseek",
                  "gemini", "openrouter", "openai_compat", "hf_serverless",
                  "llama_cpp", "ocr"]

_ALL_KEY_ENV = (
    "GROQ_API_KEY", "NM_GROQ_API_KEY", "OPENROUTER_API_KEY",
    "NM_OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "NM_ANTHROPIC_API_KEY",
    "GEMINI_API_KEY", "NM_GEMINI_API_KEY", "DEEPSEEK_API_KEY",
    "NM_DEEPSEEK_API_KEY", "HF_TOKEN", "NM_HF_TOKEN",
    "GROQ_MODELS", "NM_GROQ_MODELS", "OPENROUTER_MODELS",
    "NM_OPENROUTER_MODELS",
)


@contextmanager
def controlled_env(**set_vars: str):
    saved = {k: os.environ.get(k) for k in _ALL_KEY_ENV}
    for key in _ALL_KEY_ENV:
        os.environ.pop(key, None)
    os.environ.update(set_vars)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class LegacyChainParity(unittest.TestCase):
    def test_full_kind_alias_matrix(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-fake"}):
            for kind, cls in LEGACY_MATRIX.items():
                with self.subTest(kind=kind):
                    provider = build_provider(kind)
                    self.assertIsInstance(provider, cls)
                    self.assertEqual(provider.name,
                                     get_provider_spec(kind).id)

    def test_kind_normalization(self):
        self.assertIsInstance(build_provider("  Groq "), GroqProvider)
        self.assertIsInstance(build_provider("MOCK"), MockProvider)

    def test_error_messages_are_the_legacy_strings(self):
        cases = [
            (None, "provider kind must be a non-empty string, got None"),
            ("", "provider kind must be a non-empty string, got ''"),
            ("   ", "provider kind must be a non-empty string, got '   '"),
            (123, "provider kind must be a non-empty string, got 123"),
        ]
        for bad, expected in cases:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as ctx:
                    build_provider(bad)
                self.assertEqual(str(ctx.exception), expected)
        with self.assertRaises(ValueError) as ctx:
            build_provider("NoSuchProvider")
        self.assertEqual(
            str(ctx.exception),
            "unknown provider 'nosuchprovider'; expected "
            + " | ".join(EXPECTED_KINDS))

    def test_expected_kinds_order(self):
        self.assertEqual(expected_kinds(), EXPECTED_KINDS)


class LegacyDefaultParity(unittest.TestCase):
    def test_defaults(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-fake"}):
            self.assertEqual(build_provider("mock").model, "mock-7b")
            self.assertEqual(build_provider("groq").model,
                             "openai/gpt-oss-120b")
            self.assertEqual(build_provider("groq").base_url,
                             "https://api.groq.com/openai/v1")
            self.assertEqual(build_provider("anthropic").model,
                             "claude-sonnet-4-6")
            self.assertEqual(build_provider("deepseek").model, "deepseek-chat")
            self.assertEqual(build_provider("deepseek").base_url,
                             "https://api.deepseek.com/v1")
            self.assertEqual(build_provider("gemini").model,
                             "gemini-2.5-flash")
            self.assertEqual(build_provider("openrouter").model,
                             "openrouter/free")
            self.assertEqual(build_provider("hf_serverless").model,
                             "meta-llama/Llama-3.1-8B-Instruct")
            llama = build_provider("llama_cpp")
            self.assertEqual(llama.model, "local")
            self.assertEqual(llama.base_url, "http://localhost:8080/v1")
            compat = build_provider("openai_compat")
            self.assertEqual(compat.base_url, "http://localhost:11434/v1")
            self.assertEqual(build_provider("ocr").model, "tesseract")
            ollama = build_provider("ollama")
            self.assertEqual(ollama.base_url, "http://localhost:11434")


class RegistryData(unittest.TestCase):
    def test_canonical_kind(self):
        self.assertEqual(canonical_kind(" CLAUDE "), "anthropic")
        self.assertEqual(canonical_kind("gguf"), "llama_cpp")
        self.assertIsNone(canonical_kind("nope"))
        self.assertIsNone(canonical_kind(None))
        self.assertIsNone(canonical_kind(""))

    def test_every_spec_class_loads(self):
        for spec in list_provider_specs():
            with self.subTest(provider=spec.id):
                self.assertIs(spec.load_class(),
                              type(build_provider(
                                  spec.id, **_safe_kwargs(spec))))

    def test_specs_are_json_data_without_secrets(self):
        with controlled_env(GROQ_API_KEY="sk-never-in-a-spec"):
            blob = json.dumps([s.to_dict() for s in list_provider_specs()])
        self.assertNotIn("sk-never-in-a-spec", blob)
        self.assertEqual(len(json.loads(blob)), len(EXPECTED_KINDS))

    def test_registering_is_a_data_only_change(self):
        spec = ProviderSpec(
            id="echo_test", module="nomorals.llm.providers.mock",
            class_name="MockProvider", aliases=("echo-alias",),
            transport="in-process", capabilities=("chat",),
            default_model="mock-7b",
        )
        register_provider_spec(spec)
        try:
            self.assertIsInstance(build_provider("echo-alias"), MockProvider)
            self.assertEqual(canonical_kind("echo_test"), "echo_test")
            self.assertIn("echo_test", expected_kinds())
        finally:
            unregister_provider_spec("echo_test")
        with self.assertRaises(ValueError):
            build_provider("echo_test")

    def test_registration_collisions_refuse(self):
        with self.assertRaises(ValueError):
            register_provider_spec(ProviderSpec(
                id="mock", module="nomorals.llm.providers.mock",
                class_name="MockProvider", transport="in-process"))
        with self.assertRaises(ValueError):
            register_provider_spec(ProviderSpec(
                id="fresh_id", module="nomorals.llm.providers.mock",
                class_name="MockProvider", aliases=("groq",),
                transport="in-process"))


def _safe_kwargs(spec: ProviderSpec) -> dict:
    return {spec.auth.kwarg: "fake"} if spec.auth.kwarg else {}


class StatusFile(unittest.TestCase):
    def test_status_md_matches_the_registry(self):
        from pathlib import Path

        path = (Path(__file__).resolve().parent.parent
                / "nomorals" / "llm" / "STATUS.md")
        self.assertTrue(path.exists(), "nomorals/llm/STATUS.md missing")
        self.assertEqual(path.read_text(encoding="utf-8"),
                         render_status_markdown())


class DefaultsChainParity(unittest.TestCase):
    """llm/defaults.py builds the default router through build_provider;
    the registry path must yield the same router, names and classes."""

    def test_chain_from_env(self):
        from nomorals.llm.defaults import build_chain, specs_from_env

        with controlled_env(HF_TOKEN="hf-fake", GROQ_API_KEY="groq-fake"):
            specs = specs_from_env()
            kinds = [s.kind for s in specs]
            router = build_chain()
        self.assertEqual(kinds, ["ollama", "llama_cpp", "hf_serverless",
                                 "hf_serverless", "groq"])
        self.assertEqual(
            router.providers(),
            ["ollama", "llama_cpp", "codebeast", "hf_serverless", "groq"])
        for spec in specs:
            self.assertIsInstance(router.get(spec.name or spec.kind),
                                  build_provider(spec.kind).__class__)

    def test_env_model_overrides_reach_the_provider(self):
        from nomorals.llm.defaults import build_chain

        # The defaults chain reads the BARE env names (llm/defaults.py
        # _models_kwarg); the NM_* names feed the settings channel
        # (core/config.py env map → agents/context.py), pinned in
        # ContextChainParity below.
        with controlled_env(
                GROQ_API_KEY="groq-fake",
                GROQ_MODELS="custom/one, custom/two",
                OPENROUTER_API_KEY="or-fake",
                OPENROUTER_MODELS="custom/three"):
            router = build_chain()
        groq = router.get("groq")
        # The explicit env list replaces the roster in the walk chain
        # (GroqProvider._fallback_models() itself stays the roster;
        # _model_chain() is where the operator's word wins).
        self.assertEqual(groq._model_chain(),  # noqa: SLF001
                         ["openai/gpt-oss-120b", "custom/one", "custom/two"])
        self.assertEqual(groq.model, "openai/gpt-oss-120b")
        openrouter = router.get("openrouter")
        self.assertEqual(openrouter._model_chain(),  # noqa: SLF001
                         ["openrouter/free", "custom/three"])


class ContextChainParity(unittest.TestCase):
    """agents/context.py's settings chain canonicalizes through the
    registry: aliases land on the right classes/kwargs branches."""

    def test_aliases_canonicalize_through_build_router(self):
        from nomorals.agents.context import build_router
        from nomorals.core.config import Settings
        from nomorals.core.events import EventBus

        import tempfile
        from pathlib import Path

        home = Path(tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR")
                                      or None))
        settings = Settings(home=home)
        settings.llm.provider = "claude"
        settings.llm.fallback_chain = ["hf", "gguf"]
        settings.llm.hf_token = "hf-fake"
        with mock.patch.dict(os.environ,
                             {"ANTHROPIC_API_KEY": "sk-ant-fake"}):
            router = build_router(settings, EventBus())
        self.assertIsInstance(router.get("claude"), AnthropicProvider)
        hf = router.get("hf")
        self.assertIsInstance(hf, HFServerlessProvider)
        self.assertEqual(hf.token, "hf-fake")
        gguf = router.get("gguf")
        self.assertIsInstance(gguf, LlamaCppProvider)
        self.assertEqual(gguf.base_url, "http://localhost:8080/v1")

    def test_env_keyed_provider_without_key_is_skipped(self):
        from nomorals.agents.context import build_router
        from nomorals.core.config import Settings
        from nomorals.core.events import EventBus

        import tempfile
        from pathlib import Path

        home = Path(tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR")
                                      or None))
        settings = Settings(home=home)
        settings.llm.provider = "mock"
        settings.llm.fallback_chain = ["anthropic"]
        with controlled_env():
            router = build_router(settings, EventBus())
        self.assertNotIn("anthropic", router.providers())
        self.assertIn("mock", router.providers())

    def test_groq_models_setting_walks_through(self):
        from nomorals.agents.context import build_router
        from nomorals.core.config import Settings
        from nomorals.core.events import EventBus

        import tempfile
        from pathlib import Path

        home = Path(tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR")
                                      or None))
        settings = Settings(home=home)
        settings.llm.provider = "groq"
        settings.llm.groq_api_key = "groq-fake"
        settings.llm.groq_models = "openai/gpt-oss-20b, qwen/qwen3.6-27b"
        router = build_router(settings, EventBus())
        groq = router.get("groq")
        self.assertIsInstance(groq, GroqProvider)
        self.assertEqual(groq._model_chain(),  # noqa: SLF001
                         ["openai/gpt-oss-120b", "openai/gpt-oss-20b",
                          "qwen/qwen3.6-27b"])


if __name__ == "__main__":
    unittest.main()
