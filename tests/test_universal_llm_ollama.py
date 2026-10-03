"""Universal wave — native Ollama provider.

Offline tests against a stubbed HTTP client: native /api request/response
shapes, model management (tags/show/pull/delete/ps), option mapping, and
error paths.
"""

from __future__ import annotations

import unittest

from nomorals.core.errors import ModelError, ProviderError
from nomorals.llm.base import Message, SamplingParams
from nomorals.llm.providers.ollama import OLLAMA_DEFAULT_HOST, OllamaProvider


class FakeResponse:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self._ok = ok

    @property
    def ok(self):
        return self._ok

    def json(self):
        return self._payload


class FakeHttp:
    """Stub for HttpClient. Routes by (method, path suffix)."""

    def __init__(self, routes=None, fail_with=None):
        # routes: {(method, suffix): payload-or-FakeResponse}
        self.routes = routes or {}
        self.fail_with = fail_with
        self.calls = []

    def _handle(self, method, url, payload=None):
        self.calls.append((method, url, payload))
        if self.fail_with is not None:
            raise self.fail_with
        for (rmethod, suffix), result in self.routes.items():
            if method == rmethod and url.endswith(suffix):
                if isinstance(result, FakeResponse):
                    return result
                return FakeResponse(result)
        raise AssertionError(f"unexpected {method} {url}")

    def get(self, url, **kw):
        return self._handle("GET", url)

    def post_json(self, url, payload, **kw):
        return self._handle("POST", url, payload)

    def request(self, method, url, **kw):
        return self._handle(method, url, kw.get("json"))


def chat_payload():
    return {
        "model": "llama3.1",
        "created_at": "2026-10-03T00:00:00Z",
        "message": {"role": "assistant", "content": "hello there"},
        "done": True,
        "prompt_eval_count": 12,
        "eval_count": 3,
        "total_duration": 1000000,
    }


class OllamaConstructionTests(unittest.TestCase):
    def test_defaults(self):
        provider = OllamaProvider(model="llama3.1")
        self.assertEqual(provider.base_url, OLLAMA_DEFAULT_HOST)
        self.assertEqual(provider.model_id, "llama3.1")
        self.assertEqual(provider.name, "ollama")

    def test_v1_suffix_is_stripped_to_native_api(self):
        provider = OllamaProvider(base_url="http://localhost:11434/v1", model="x")
        self.assertEqual(provider.base_url, "http://localhost:11434")

    def test_model_from_env_fallback(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"OLLAMA_MODEL": "env-model"}):
            provider = OllamaProvider()
        self.assertEqual(provider.model, "env-model")

    def test_capabilities(self):
        provider = OllamaProvider(model="x")
        self.assertEqual(provider.capabilities, {"chat", "complete", "embed", "vision"})

    def test_health_true_and_false(self):
        good = OllamaProvider(model="x", max_retries=1)
        good.http = FakeHttp(routes={("GET", "/api/version"): {"version": "0.9.0"}})
        self.assertTrue(good.health())

        bad = OllamaProvider(model="x", max_retries=1)
        bad.http = FakeHttp(fail_with=ConnectionError("down"))
        self.assertFalse(bad.health())


class OllamaChatTests(unittest.TestCase):
    def make_provider(self, payload):
        provider = OllamaProvider(model="llama3.1", max_retries=1)
        provider.http = FakeHttp(routes={("POST", "/api/chat"): payload})
        return provider

    def test_chat_posts_native_shape(self):
        provider = self.make_provider(chat_payload())
        response = provider.chat(
            [Message.system("be brief"), Message.user("hi")],
            SamplingParams(temperature=0.2, max_tokens=50),
        )
        self.assertTrue(response.ok)
        self.assertEqual(response.text, "hello there")
        self.assertEqual(response.usage.prompt_tokens, 12)
        self.assertEqual(response.usage.completion_tokens, 3)
        self.assertEqual(response.provider, "ollama")

        method, url, payload = provider.http.calls[0]
        self.assertTrue(url.endswith("/api/chat"))
        self.assertEqual(payload["model"], "llama3.1")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["keep_alive"], "5m")
        self.assertEqual(
            [m["role"] for m in payload["messages"]], ["system", "user"]
        )
        # SamplingParams translated to Ollama options, not OpenAI keys.
        self.assertEqual(payload["options"]["num_predict"], 50)
        self.assertEqual(payload["options"]["temperature"], 0.2)
        self.assertNotIn("max_tokens", payload["options"])

    def test_json_mode_sets_format(self):
        provider = self.make_provider(chat_payload())
        provider.chat([Message.user("hi")], SamplingParams(json_mode=True))
        _, _, payload = provider.http.calls[0]
        self.assertEqual(payload["format"], "json")

    def test_chat_server_error_raises_provider_error(self):
        provider = self.make_provider({"error": 'model "nope" not found'})
        with self.assertRaises(ProviderError):
            provider.chat([Message.user("hi")])

    def test_chat_transport_failure_raises_provider_error(self):
        provider = OllamaProvider(model="x", max_retries=1)
        provider.http = FakeHttp(fail_with=ConnectionError("refused"))
        with self.assertRaises(ProviderError):
            provider.chat([Message.user("hi")])

    def test_complete_uses_generate_endpoint(self):
        body = {
            "model": "llama3.1", "response": "done it", "done": True,
            "prompt_eval_count": 5, "eval_count": 2,
        }
        provider = OllamaProvider(model="llama3.1", max_retries=1)
        provider.http = FakeHttp(routes={("POST", "/api/generate"): body})
        response = provider.complete("write code", SamplingParams(max_tokens=10))
        self.assertEqual(response.text, "done it")
        method, url, payload = provider.http.calls[0]
        self.assertTrue(url.endswith("/api/generate"))
        self.assertEqual(payload["prompt"], "write code")
        self.assertEqual(payload["options"]["num_predict"], 10)

    def test_embed_uses_native_embed_endpoint(self):
        provider = OllamaProvider(model="nomic-embed-text", max_retries=1)
        provider.http = FakeHttp(
            routes={("POST", "/api/embed"): {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}}
        )
        vectors = provider.embed(["a", "b"])
        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        _, url, payload = provider.http.calls[0]
        self.assertTrue(url.endswith("/api/embed"))
        self.assertEqual(payload["input"], ["a", "b"])

    def test_embed_count_mismatch_raises(self):
        provider = OllamaProvider(model="x", max_retries=1)
        provider.http = FakeHttp(routes={("POST", "/api/embed"): {"embeddings": [[0.1]]}})
        with self.assertRaises(ModelError):
            provider.embed(["a", "b"])

    def test_describe_image_sends_native_images_field(self):
        provider = self.make_provider(chat_payload())
        response = provider.describe_image(b"\x89PNG fake", prompt="what is this?")
        self.assertTrue(response.ok)
        self.assertEqual(response.text, "hello there")
        _, url, payload = provider.http.calls[0]
        self.assertTrue(url.endswith("/api/chat"))
        message = payload["messages"][0]
        self.assertEqual(message["role"], "user")
        self.assertIn("images", message)
        self.assertEqual(len(message["images"]), 1)
        # raw base64, not a data: URL
        import base64

        self.assertEqual(base64.b64decode(message["images"][0]), b"\x89PNG fake")


class OllamaManagementTests(unittest.TestCase):
    def test_list_and_has_model(self):
        provider = OllamaProvider(model="x", max_retries=1)
        provider.http = FakeHttp(
            routes={
                ("GET", "/api/tags"): {
                    "models": [
                        {"name": "llama3.1:8b", "size": 100},
                        {"name": "llava:7b", "size": 200},
                    ]
                }
            }
        )
        self.assertEqual(provider.model_names(), ["llama3.1:8b", "llava:7b"])
        self.assertTrue(provider.has_model("llama3.1:8b"))
        self.assertTrue(provider.has_model("llama3.1"))  # tag-insensitive
        self.assertFalse(provider.has_model("mistral"))

    def test_show_and_model_capabilities(self):
        provider = OllamaProvider(model="x", max_retries=1)
        provider.http = FakeHttp(
            routes={
                ("POST", "/api/show"): {
                    "modelfile": "# Modelfile",
                    "capabilities": ["vision", "completion"],
                    "details": {"parameter_size": "7.2B"},
                }
            }
        )
        info = provider.show("llava:7b")
        self.assertEqual(info["details"]["parameter_size"], "7.2B")
        self.assertEqual(provider.model_capabilities("llava:7b"),
                         ["vision", "completion"])
        _, _, payload = provider.http.calls[0]
        self.assertEqual(payload["name"], "llava:7b")

    def test_model_capabilities_unknown_model_returns_empty(self):
        provider = OllamaProvider(model="x", max_retries=1)
        provider.http = FakeHttp(routes={("POST", "/api/show"): {"error": "not found"}})
        self.assertEqual(provider.model_capabilities("nope"), [])

    def test_pull_posts_name(self):
        provider = OllamaProvider(model="x", max_retries=1)
        provider.http = FakeHttp(routes={("POST", "/api/pull"): {"status": "success"}})
        result = provider.pull("qwen2.5:7b")
        self.assertEqual(result["status"], "success")
        _, url, payload = provider.http.calls[0]
        self.assertTrue(url.endswith("/api/pull"))
        self.assertEqual(payload["name"], "qwen2.5:7b")
        self.assertFalse(payload["stream"])

    def test_pull_error_raises(self):
        provider = OllamaProvider(model="x", max_retries=1)
        provider.http = FakeHttp(routes={("POST", "/api/pull"): {"error": "disk full"}})
        with self.assertRaises(ProviderError):
            provider.pull("big:70b")

    def test_delete_model(self):
        provider = OllamaProvider(model="x", max_retries=1)
        provider.http = FakeHttp(routes={("DELETE", "/api/delete"): {}})
        self.assertTrue(provider.delete_model("old:7b"))

    def test_running_models(self):
        provider = OllamaProvider(model="x", max_retries=1)
        provider.http = FakeHttp(
            routes={("GET", "/api/ps"): {"models": [{"name": "llama3.1:8b"}]}}
        )
        self.assertEqual(provider.running_models(), [{"name": "llama3.1:8b"}])

    def test_unload_issues_keep_alive_zero_generate(self):
        provider = OllamaProvider(model="llama3.1:8b", max_retries=1)
        provider.http = FakeHttp(routes={("POST", "/api/generate"): {"done": True}})
        provider.unload()
        _, url, payload = provider.http.calls[0]
        self.assertTrue(url.endswith("/api/generate"))
        self.assertEqual(payload["keep_alive"], 0)


if __name__ == "__main__":
    unittest.main()
