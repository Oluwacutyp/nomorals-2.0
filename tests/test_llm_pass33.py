"""Pass #33 — LLM router + providers audit fixes.

Defects pinned here (all reproduced on pristine HEAD fe90cf6e):

1. ``build_router`` could never register Anthropic (``kwargs_for`` passed
   ``base_url``, which ``AnthropicProvider.__init__`` does not accept —
   the TypeError was swallowed and Devon booted model-less), and it
   registered Gemini/DeepSeek pointing at the *OpenAI* base URL with the
   *OpenAI* key and model — the named provider never talked to its own
   endpoint.
2. HF ``chat()`` returned the ``HttpResponse`` object's repr as the
   reply text on every successful call (``_post_checked`` returned the
   response, ``_parse_chat`` was written for a dict; the test suite
   mocked ``_post_checked`` to return dicts, so it never saw this).
3. HF token supplied via the ``HF_TOKEN`` env var was stored on
   ``self.token`` but never sent — the Authorization header was gated on
   the constructor argument, not the resolved token.
4. HF model healing swapped the model in the URL but not in the request
   payload, so the "healed" call still named the dead model in the body.
5. An HTTP-200 body of ``{"error": ...}`` from HF parsed into an empty
   *successful* response (chat and complete) — a fake success.
6. ``LLMRouter.verify(timeout_seconds=...)`` passed ``timeout=`` into
   ``provider.chat(**kw)``; OpenAI-compat/Ollama/Anthropic merged it
   into the wire payload (``"timeout": 10.0`` sent to the API) and the
   probe was not actually bounded by it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomorals.core.http import HttpResponse  # noqa: E402
from nomorals.llm.base import Message, SamplingParams  # noqa: E402


def _http_response(payload: dict, status: int = 200) -> HttpResponse:
    return HttpResponse(
        status=status,
        body=json.dumps(payload).encode("utf-8"),
        headers={},
        url="http://provider.test",
    )


def _chat_ok(text: str = "REAL ANSWER") -> HttpResponse:
    return _http_response({
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "model": "m1",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    })


# ── 1. build_router wiring for anthropic / gemini / deepseek ────────────────

def _router_for(monkeypatch, kind: str, env: dict[str, str]):
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
                "NM_ANTHROPIC_API_KEY", "NM_GEMINI_API_KEY",
                "NM_DEEPSEEK_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from nomorals.agents.context import build_router
    from nomorals.core.config import Settings
    from nomorals.core.events import EventBus

    settings = Settings()
    settings.llm.provider = kind
    settings.llm.fallback_chain = []
    return build_router(settings, EventBus())


def test_build_router_registers_anthropic(monkeypatch):
    router = _router_for(monkeypatch, "anthropic",
                         {"ANTHROPIC_API_KEY": "sk-ant-test"})
    assert "anthropic" in router.providers()
    provider = router.get("anthropic")
    # The provider's own default model — NOT the OpenAI model from settings.
    assert provider.model == "claude-sonnet-4-6"
    assert provider.api_key == "sk-ant-test"


def test_build_router_registers_gemini_on_its_own_endpoint(monkeypatch):
    from nomorals.llm.providers.gemini import GEMINI_BASE_URL

    router = _router_for(monkeypatch, "gemini", {"GEMINI_API_KEY": "gem-key"})
    assert "gemini" in router.providers()
    provider = router.get("gemini")
    # OpenAICompatProvider strips the trailing slash from the base URL.
    assert provider.base_url == GEMINI_BASE_URL.rstrip("/")
    assert provider.model == "gemini-2.5-flash"
    assert provider.api_key == "gem-key"


def test_build_router_registers_deepseek_on_its_own_endpoint(monkeypatch):
    from nomorals.llm.providers.deepseek import DEEPSEEK_BASE_URL

    router = _router_for(monkeypatch, "deepseek",
                         {"DEEPSEEK_API_KEY": "ds-key"})
    assert "deepseek" in router.providers()
    provider = router.get("deepseek")
    assert provider.base_url == DEEPSEEK_BASE_URL
    assert provider.model == "deepseek-chat"
    assert provider.api_key == "ds-key"


def test_build_router_skips_keyless_gemini(monkeypatch):
    # Mirrors the groq/openrouter rule in build_router: a keyed cloud
    # provider with no key configured is skipped, not registered to 401.
    router = _router_for(monkeypatch, "gemini", {})
    assert "gemini" not in router.providers()


# ── 2–5. HF serverless provider ─────────────────────────────────────────────

def test_hf_env_token_is_sent_as_authorization_header(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env_token_xyz")
    monkeypatch.delenv("NM_HF_TOKEN", raising=False)
    from nomorals.llm.providers.hf_serverless import HFServerlessProvider

    provider = HFServerlessProvider()
    assert provider.token == "hf_env_token_xyz"
    assert provider.http.headers.get("Authorization") == \
        "Bearer hf_env_token_xyz"


def test_hf_chat_returns_the_parsed_answer_not_the_response_repr():
    from nomorals.llm.providers.hf_serverless import HFServerlessProvider

    provider = HFServerlessProvider(token="t", model="m1")
    provider.http.post_json = lambda url, payload, **kw: _chat_ok()
    resp = provider.chat([Message.user("hi")])
    assert resp.ok
    assert resp.text == "REAL ANSWER"
    assert resp.usage.prompt_tokens == 3
    assert resp.usage.completion_tokens == 2


def test_hf_heal_updates_the_payload_model(monkeypatch):
    from nomorals.llm.providers.hf_serverless import HFServerlessProvider

    provider = HFServerlessProvider(token="t", model="dead/model-1")
    calls: list[dict] = []

    def fake_post(url, payload, **kw):
        calls.append(dict(payload))
        if len(calls) == 1:
            raise Exception("404 Client Error: model not found")
        return _chat_ok("healed answer")

    provider.http.post_json = fake_post
    provider.fetch_catalog = lambda: [
        {"id": "live/model-2", "providers": [{"status": "live"}]}]
    resp = provider.chat([Message.user("hi")])
    assert resp.ok
    assert resp.text == "healed answer"
    assert provider.model == "live/model-2"
    # The second attempt's BODY must name the healed model, not just the URL.
    assert calls[1]["model"] == "live/model-2"


def test_hf_error_body_is_not_a_fake_success():
    from nomorals.llm.providers.hf_serverless import HFServerlessProvider

    provider = HFServerlessProvider(token="t", model="m1")
    provider.http.post_json = lambda url, payload, **kw: _http_response(
        {"error": "Model m1 is currently loading"})
    provider.fetch_catalog = lambda: []
    resp = provider.chat([Message.user("hi")])
    assert not resp.ok
    assert "currently loading" in resp.error


def test_hf_complete_error_body_is_not_a_fake_success():
    from nomorals.llm.providers.hf_serverless import HFServerlessProvider

    provider = HFServerlessProvider(token="t", model="m1",
                                    use_chat_endpoint=False)
    provider.http.post_json = lambda url, payload, **kw: _http_response(
        {"error": "model not hosted"})
    resp = provider.complete("hello")
    assert not resp.ok
    assert "not hosted" in resp.error


# ── 6. verify() timeout must bound the probe, not leak into the payload ────

def test_openai_compat_chat_timeout_kwarg_bounds_call_not_payload():
    from nomorals.llm.providers.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider(base_url="http://x/v1", api_key="k",
                                    model="m")
    seen: dict = {}

    def fake_post(url, payload, **kw):
        seen["payload"] = dict(payload)
        seen["timeout"] = kw.get("timeout")
        return _chat_ok("pong")

    provider.http.post_json = fake_post
    resp = provider.chat([Message.user("ping")],
                         SamplingParams(max_tokens=4), timeout=7.0)
    assert resp.ok
    assert "timeout" not in seen["payload"]
    assert seen["timeout"] == 7.0


def test_ollama_chat_timeout_kwarg_bounds_call_not_payload():
    from nomorals.llm.providers.ollama import OllamaProvider

    provider = OllamaProvider(model="m")
    seen: dict = {}

    def fake_post(url, payload, **kw):
        seen["payload"] = dict(payload)
        seen["timeout"] = kw.get("timeout")
        return _http_response({
            "message": {"content": "pong"}, "model": "m", "done": True,
            "prompt_eval_count": 1, "eval_count": 1})

    provider.http.post_json = fake_post
    resp = provider.chat([Message.user("ping")], timeout=7.0)
    assert resp.ok
    assert "timeout" not in seen["payload"]
    assert seen["timeout"] == 7.0


def test_anthropic_chat_timeout_kwarg_bounds_call_not_payload():
    from nomorals.llm.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="k")
    seen: dict = {}

    def fake_post(url, payload, **kw):
        seen["payload"] = dict(payload)
        seen["timeout"] = kw.get("timeout")
        return _http_response({
            "content": [{"type": "text", "text": "pong"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn", "model": "claude-sonnet-4-6"})

    provider.http.post_json = fake_post
    resp = provider.chat([Message.user("ping")], timeout=7.0)
    assert resp.ok
    assert "timeout" not in seen["payload"]
    assert seen["timeout"] == 7.0


def test_router_verify_sends_no_timeout_field():
    from nomorals.llm.providers.openai_compat import OpenAICompatProvider
    from nomorals.llm.router import LLMRouter

    provider = OpenAICompatProvider(base_url="http://x/v1", api_key="k",
                                    model="m", timeout=120.0)
    seen: dict = {}
    provider.http.post_json = lambda url, payload, **kw: (
        seen.update(payload=dict(payload), timeout=kw.get("timeout")),
        _chat_ok("pong"))[1]
    router = LLMRouter()
    router.add(provider, primary=True)
    report = router.verify(timeout_seconds=10.0)
    assert report["ok"] == ["openai_compat"]
    assert "timeout" not in seen["payload"]
    assert seen["timeout"] == 10.0


# ── pins: behaviour that was already correct ────────────────────────────────

def test_router_total_failure_reports_the_whole_chain():
    from nomorals.llm.base import LLMProvider, LLMResponse
    from nomorals.llm.router import LLMRouter

    class Boom(LLMProvider):
        def __init__(self, name, err):
            super().__init__()
            self.name = name
            self._err = err

        def chat(self, messages, params=None, **kw):
            return LLMResponse(text="", model="m", error=self._err)

    router = LLMRouter()
    router.add(Boom("first", "429 rate limited"), primary=True)
    router.add(Boom("second", "401 invalid api key"))
    resp = router.chat([Message.user("hi")])
    assert not resp.ok
    assert resp.failed_providers == ["first", "second"]
    assert "429 rate limited" in resp.fallback_note
    assert "401 invalid api key" in resp.fallback_note


def test_heal_model_id_never_rewrites_a_deliberate_model():
    from nomorals.llm.providers.openai_compat import heal_model_id

    known = {"openai/gpt-oss-120b", "qwen/qwen3-8b:free"}
    # A deliberately different id passes through untouched.
    assert heal_model_id("llama-3.3-70b-versatile", known) == \
        "llama-3.3-70b-versatile"
    assert heal_model_id("my/custom-model", known) == "my/custom-model"
    # Only exact near-misses heal.
    assert heal_model_id("openai/gpt-oss-120", known) == "openai/gpt-oss-120b"
    assert heal_model_id("qwen/qwen3-8b", known) == "qwen/qwen3-8b:free"
