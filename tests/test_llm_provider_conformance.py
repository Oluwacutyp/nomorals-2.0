"""Shared provider conformance suite (Echelon Wave 4).

Every provider registered in the declarative registry
(``nomorals.llm.providers.registry``) must pass the SAME checks —
the §1.1e discipline applied to the llm layer:

* construction from spec (offline-safe kwargs, fake credentials);
* capability honesty — the instance advertises exactly the set its
  spec declares;
* the models-roster hook resolves (module table read offline, or the
  named instance discovery method exists);
* a completion round-trip against a mocked transport, in the shape
  of that provider's own wire format;
* error classification — a failed call surfaces text that
  ``llm.failures.classify_failure`` maps to the right FailureClass,
  or (for providers whose contract is to raise) a typed
  ``NoMoralsError`` with a machine-readable ``code``;
* the in-provider model failover walk (ladderfix semantics): a
  model-scoped failure on the configured model walks to the next
  roster id in order;
* no credential leaks: the sentinel key used at construction appears
  in no response, error, stats dump, spec dump, or STATUS.md.

OCR is the deliberate exception that proves the shape: it has no chat
surface, so its "round-trip" is the typed refusal from ``chat()``
plus a real ``describe_image`` response object.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from nomorals.core.errors import NoMoralsError, ProviderError, RateLimited
from nomorals.llm.base import LLMProvider, LLMResponse, Message
from nomorals.llm.failures import FailureClass, classify_failure
from nomorals.llm.providers import build_provider, list_provider_specs
from nomorals.llm.providers.registry import ProviderSpec

SPECS = list_provider_specs()
BY_ID = {s.id: s for s in SPECS}


def _sentinel(spec: ProviderSpec) -> str:
    return f"sk-conformance-{spec.id}-s3cr3t-value"


def _offline_kwargs(spec: ProviderSpec, **extra: Any) -> dict[str, Any]:
    """Offline-safe construction kwargs: fake credential, no network."""
    kwargs: dict[str, Any] = {}
    if spec.auth.kwarg:
        kwargs[spec.auth.kwarg] = _sentinel(spec)
    if spec.id == "openai_compat":
        kwargs["model"] = "conformance-model"
    kwargs.update(extra)
    return kwargs


class _Resp:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _ScriptedHttp:
    """post_json routed to a handler; records the models attempted."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def post_json(self, url, payload, **kw):
        self.urls.append(url)
        self.calls.append(dict(payload))
        return self.handler(url, payload)

    def get(self, url, **kw):  # pragma: no cover - must never be hit
        raise AssertionError(f"conformance transport: unexpected GET {url}")


def _openai_ok(model: str, text: str = "conformance-ok") -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _ok_handler(spec: ProviderSpec):
    if spec.id == "ollama":
        def ollama_handler(url, payload):
            return _Resp({
                "model": payload.get("model") or "conformance",
                "message": {"role": "assistant", "content": "conformance-ok"},
                "done": True, "prompt_eval_count": 3, "eval_count": 2,
            })
        return ollama_handler
    if spec.id == "anthropic":
        def anthropic_handler(url, payload):
            return _Resp({
                "content": [{"type": "text", "text": "conformance-ok"}],
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "stop_reason": "end_turn",
                "model": payload.get("model") or "claude-sonnet-4-6",
            })
        return anthropic_handler

    def openai_handler(url, payload):
        return _Resp(_openai_ok(payload.get("model") or "conformance"))
    return openai_handler


def _raising(exc: BaseException):
    def handler(url, payload):
        raise exc
    return handler


def _build_with_http(spec: ProviderSpec, handler, **extra: Any) -> LLMProvider:
    provider = build_provider(spec.id, **_offline_kwargs(spec, **extra))
    provider.http = _ScriptedHttp(handler)  # type: ignore[attr-defined]
    return provider


# Providers whose chat speaks the OpenAI wire shape through the shared
# OpenAICompatProvider walk (the ladderfix machinery).
_OPENAI_WALK_IDS = ("groq", "deepseek", "gemini", "openrouter",
                    "openai_compat", "llama_cpp")


class ConstructionConformance(unittest.TestCase):
    def test_every_provider_constructs_from_spec(self):
        for spec in SPECS:
            with self.subTest(provider=spec.id):
                provider = build_provider(spec.id, **_offline_kwargs(spec))
                self.assertIsInstance(provider, LLMProvider)
                self.assertIsInstance(provider, spec.load_class())
                self.assertEqual(provider.name, spec.id)

    def test_capabilities_match_spec(self):
        for spec in SPECS:
            with self.subTest(provider=spec.id):
                provider = build_provider(spec.id, **_offline_kwargs(spec))
                self.assertEqual(set(provider.capabilities),
                                 set(spec.capabilities))

    def test_roster_hook_resolves(self):
        for spec in SPECS:
            with self.subTest(provider=spec.id):
                provider = build_provider(spec.id, **_offline_kwargs(spec))
                if spec.models_hook.startswith("module:"):
                    roster = spec.roster()
                    self.assertTrue(roster, f"{spec.id}: empty module roster")
                    self.assertEqual(spec.roster_for(provider), roster)
                elif spec.models_hook.startswith("instance:"):
                    method = spec.models_hook.split(":", 1)[1]
                    self.assertTrue(
                        callable(getattr(provider, method, None)),
                        f"{spec.id}: discovery method {method} missing")
                else:
                    self.assertEqual(spec.roster(), [])

    def test_model_chain_starts_with_configured_model(self):
        for spec_id in _OPENAI_WALK_IDS:
            spec = BY_ID[spec_id]
            with self.subTest(provider=spec_id):
                provider = build_provider(spec_id, **_offline_kwargs(spec))
                chain = provider._model_chain()  # noqa: SLF001
                self.assertEqual(chain[0], provider.model)
                self.assertEqual(len(chain), len(set(chain)))


class RoundTripConformance(unittest.TestCase):
    def test_completion_round_trip(self):
        for spec in SPECS:
            with self.subTest(provider=spec.id):
                if spec.id == "mock":
                    provider = build_provider("mock")
                    resp = provider.chat([Message.user("hello")])
                    self.assertTrue(resp.ok)
                    self.assertTrue(resp.text)
                    self.assertEqual(resp.provider, "mock")
                    continue
                if spec.id == "ocr":
                    provider = build_provider("ocr")
                    with self.assertRaises(ProviderError):
                        provider.chat([Message.user("hello")])
                    resp = provider.describe_image(b"not really an image")
                    self.assertIsInstance(resp, LLMResponse)
                    self.assertEqual(resp.provider, "ocr")
                    continue
                provider = _build_with_http(spec, _ok_handler(spec))
                resp = provider.chat([Message.user("hello")])
                self.assertTrue(resp.ok, f"{spec.id}: {resp.error}")
                self.assertEqual(resp.text, "conformance-ok")
                self.assertEqual(resp.provider, spec.id)
                # Ollama reports prompt/completion counts without a
                # total; the OpenAI-shaped providers report all three.
                total = resp.usage.total_tokens or (
                    resp.usage.prompt_tokens + resp.usage.completion_tokens)
                self.assertEqual(total, 5)


class FailureClassificationConformance(unittest.TestCase):
    def test_rate_limit_classifies(self):
        for spec_id in _OPENAI_WALK_IDS:
            spec = BY_ID[spec_id]
            with self.subTest(provider=spec_id):
                provider = _build_with_http(
                    spec,
                    _raising(ProviderError(
                        "429 rate limit exceeded",
                        details={"http_status": 429})),
                    max_retries=1, fallback_models=[])
                resp = provider.chat([Message.user("hello")])
                self.assertFalse(resp.ok)
                info = classify_failure(resp.error)
                self.assertEqual(info.failure_class, FailureClass.RATE_LIMITED)

    def test_auth_aborts_the_leg_immediately(self):
        for spec_id in _OPENAI_WALK_IDS:
            spec = BY_ID[spec_id]
            with self.subTest(provider=spec_id):
                provider = _build_with_http(
                    spec,
                    _raising(ProviderError(
                        "401 unauthorized: invalid api key",
                        details={"http_status": 401})),
                    max_retries=1)
                resp = provider.chat([Message.user("hello")])
                self.assertFalse(resp.ok)
                # Provider-scoped: no model walk, exactly one attempt.
                self.assertEqual(len(provider.http.calls), 1)
                info = classify_failure(resp.error)
                self.assertEqual(info.failure_class, FailureClass.AUTH)

    def test_anthropic_auth_classifies(self):
        spec = BY_ID["anthropic"]
        provider = _build_with_http(
            spec,
            _raising(ProviderError("401 unauthorized: invalid api key",
                                   details={"http_status": 401})),
            max_retries=1)
        resp = provider.chat([Message.user("hello")])
        self.assertFalse(resp.ok)
        self.assertEqual(classify_failure(resp.error).failure_class,
                         FailureClass.AUTH)

    def test_hf_auth_classifies(self):
        spec = BY_ID["hf_serverless"]
        provider = _build_with_http(
            spec,
            _raising(ProviderError("401 unauthorized: invalid token",
                                   details={"http_status": 401})))
        resp = provider.chat([Message.user("hello")])
        self.assertFalse(resp.ok)
        self.assertEqual(classify_failure(resp.error).failure_class,
                         FailureClass.AUTH)

    def test_ollama_failure_is_typed_and_classifies(self):
        spec = BY_ID["ollama"]
        provider = _build_with_http(
            spec, _raising(ProviderError("429 rate limit exceeded")),
            max_retries=1)
        with self.assertRaises(ProviderError) as ctx:
            provider.chat([Message.user("hello")])
        self.assertIsInstance(ctx.exception, NoMoralsError)
        self.assertTrue(ctx.exception.code)
        info = classify_failure(str(ctx.exception), ctx.exception)
        self.assertEqual(info.failure_class, FailureClass.RATE_LIMITED)

    def test_mock_failure_is_an_error_response(self):
        provider = build_provider("mock", failure_rate=1.0)
        resp = provider.chat([Message.user("hello")])
        self.assertFalse(resp.ok)
        self.assertEqual(classify_failure(resp.error).failure_class,
                         FailureClass.UNKNOWN)

    def test_ocr_chat_refusal_is_typed(self):
        provider = build_provider("ocr")
        with self.assertRaises(NoMoralsError) as ctx:
            provider.chat([Message.user("hello")])
        self.assertEqual(ctx.exception.code, "model.provider")


class FailoverWalkConformance(unittest.TestCase):
    """Ladderfix semantics, per rostered provider: a model-scoped
    failure on the configured model walks to the next roster id."""

    def _walk(self, spec_id: str, first: str, second: str, **build_kw):
        spec = BY_ID[spec_id]
        script = {
            first: ProviderError("404 model_not_found",
                                 details={"http_status": 404}),
            second: None,
        }

        def handler(url, payload):
            outcome = script[payload["model"]]
            if outcome is not None:
                raise outcome
            return _Resp(_openai_ok(payload["model"]))

        provider = _build_with_http(spec, handler, model=first, **build_kw)
        resp = provider.chat([Message.user("hello")])
        attempted = [c["model"] for c in provider.http.calls]
        return provider, resp, attempted

    def test_groq_walk(self):
        roster = BY_ID["groq"].roster()
        provider, resp, attempted = self._walk("groq", roster[0], roster[1])
        self.assertEqual(attempted, [roster[0], roster[1]])
        self.assertTrue(resp.ok)
        self.assertEqual(resp.model, roster[1])
        # A dead configured id heals: the fallback stays active.
        self.assertEqual(provider.model, roster[1])

    def test_deepseek_walk(self):
        roster = BY_ID["deepseek"].roster()
        _provider, resp, attempted = self._walk("deepseek", roster[0], roster[1])
        self.assertEqual(attempted, [roster[0], roster[1]])
        self.assertTrue(resp.ok)

    def test_gemini_walk(self):
        roster = BY_ID["gemini"].roster()
        _provider, resp, attempted = self._walk("gemini", roster[0], roster[1])
        self.assertEqual(attempted, [roster[0], roster[1]])
        self.assertTrue(resp.ok)

    def test_openrouter_walk_to_free_alias(self):
        _provider, resp, attempted = self._walk(
            "openrouter", "dead/model:free", "openrouter/free")
        self.assertEqual(attempted, ["dead/model:free", "openrouter/free"])
        self.assertTrue(resp.ok)

    def test_openai_compat_explicit_fallback_walk(self):
        _provider, resp, attempted = self._walk(
            "openai_compat", "m1", "m2", fallback_models=["m2"])
        self.assertEqual(attempted, ["m1", "m2"])
        self.assertTrue(resp.ok)

    def test_explicit_fallback_models_replace_the_roster(self):
        """The NM_GROQ_MODELS mechanism: an explicit list is the
        operator's word and replaces the built-in roster exactly."""
        spec = BY_ID["groq"]

        def handler(url, payload):
            if payload["model"] == "custom/a":
                raise RateLimited("429 rate limit", retry_after=30.0)
            return _Resp(_openai_ok(payload["model"]))

        provider = _build_with_http(
            spec, handler, model="custom/a",
            fallback_models=["custom/b"])
        self.assertEqual(provider._model_chain(), ["custom/a", "custom/b"])  # noqa: SLF001
        resp = provider.chat([Message.user("hello")])
        self.assertTrue(resp.ok)
        self.assertEqual([c["model"] for c in provider.http.calls],
                         ["custom/a", "custom/b"])
        # A rate-limit walk is transient: the configured model is restored.
        self.assertEqual(provider.model, "custom/a")


class NoCredentialLeakConformance(unittest.TestCase):
    def test_sentinel_key_never_surfaces(self):
        from nomorals.llm.providers import render_status_markdown

        status_text = render_status_markdown()
        for spec in SPECS:
            if not spec.auth.kwarg:
                continue
            sentinel = _sentinel(spec)
            with self.subTest(provider=spec.id):
                self.assertNotIn(sentinel, json.dumps(spec.to_dict()))
                self.assertNotIn(sentinel, status_text)
                # Success surface.
                if spec.id == "ocr":
                    continue
                provider = _build_with_http(spec, _ok_handler(spec))
                resp = provider.chat([Message.user("hello")])
                self.assertNotIn(sentinel, resp.text)
                self.assertNotIn(sentinel, resp.error)
                self.assertNotIn(sentinel, json.dumps(resp.to_dict()))
                self.assertNotIn(
                    sentinel, json.dumps(provider.stats_snapshot()))
                # Failure surface.
                failing = _build_with_http(
                    spec,
                    _raising(ProviderError(
                        "401 unauthorized: invalid api key",
                        details={"http_status": 401})),
                    max_retries=1)
                try:
                    resp2 = failing.chat([Message.user("hello")])
                    surfaced = resp2.error + resp2.text
                except NoMoralsError as exc:
                    surfaced = str(exc)
                self.assertNotIn(sentinel, surfaced)


if __name__ == "__main__":
    unittest.main()
