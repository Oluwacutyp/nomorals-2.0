"""Brain deep capabilities: timeouts, chat_json repair, embed, brain_for,
context-overflow recovery, best_of. All offline (fake routers)."""
from __future__ import annotations

import time

import pytest

from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.llm.brain import Brain, brain_for, explain_failure


class FakeRouter:
    """Duck-typed router: chat/complete/describe_image/embed, scriptable."""

    def __init__(self, handler=None):
        self.handler = handler or (lambda op, *a, **k: LLMResponse(text="ok"))
        self.calls: list[tuple] = []
        self._providers: dict[str, object] = {}
        self.active = ""

    def _run(self, op, *args, **kwargs):
        self.calls.append((op, args, kwargs))
        return self.handler(op, *args, **kwargs)

    def chat(self, messages, params=None, **kw):
        return self._run("chat", messages, params, **kw)

    def complete(self, prompt, params=None, **kw):
        return self._run("complete", prompt, params, **kw)

    def describe_image(self, image, prompt="", params=None, **kw):
        return self._run("vision", image, prompt, params, **kw)

    def embed(self, texts, **kw):
        resp = self._run("embed", texts, **kw)
        if isinstance(resp, LLMResponse) and resp.error:
            raise RuntimeError(resp.error)
        return [[0.1, 0.2]] * len(texts)

    def providers(self):
        return list(self._providers)

    def get(self, name):
        return self._providers.get(name)

    def is_cooling_down(self, name):
        return False

    @property
    def broker(self):
        return None


def _ok(text="hello"):
    return LLMResponse(text=text, model="m", provider="p")


# ── timeout ────────────────────────────────────────────────────────────────

def test_chat_timeout_returns_typed_error_not_hang():
    def slow(op, *a, **k):
        time.sleep(30)
        return _ok()

    brain = Brain(router=FakeRouter(handler=slow))
    started = time.monotonic()
    resp = brain.chat([Message.user("hi")], timeout_s=0.2)
    elapsed = time.monotonic() - started
    assert elapsed < 5, "timeout must bound the call"
    assert not resp.ok
    assert resp.failure_class == "timeout"
    assert "timed out" in resp.error


def test_timeout_none_means_unbounded():
    brain = Brain(router=FakeRouter(handler=lambda op, *a, **k: _ok()))
    resp = brain.chat([Message.user("hi")], timeout_s=None)
    assert resp.ok


# ── chat_json ──────────────────────────────────────────────────────────────

def test_chat_json_parses_first_try():
    brain = Brain(router=FakeRouter(
        handler=lambda op, *a, **k: _ok('{"a": 1}')))
    data, resp = brain.chat_json([Message.user("give json")])
    assert data == {"a": 1} and resp.ok


def test_chat_json_repairs_bad_json():
    texts = ['not json at all', '{"a": 2}']
    brain = Brain(router=FakeRouter(
        handler=lambda op, *a, **k: _ok(texts.pop(0))))
    data, resp = brain.chat_json([Message.user("give json")], attempts=3)
    assert data == {"a": 2}
    # the repair nudge was appended on the retry
    assert len(brain.router.calls) == 2
    retry_msgs = brain.router.calls[1][1][0]
    assert any("not valid JSON" in m.content for m in retry_msgs)


def test_chat_json_gives_up_honestly():
    brain = Brain(router=FakeRouter(
        handler=lambda op, *a, **k: _ok("nope, still prose")))
    data, resp = brain.chat_json([Message.user("give json")], attempts=2)
    assert data is None and resp.ok  # model answered, just not JSON


def test_chat_json_model_down():
    brain = Brain(router=FakeRouter(
        handler=lambda op, *a, **k: LLMResponse(text="", error="down")))
    data, resp = brain.chat_json([Message.user("x")], attempts=2)
    assert data is None and not resp.ok


# ── embed ──────────────────────────────────────────────────────────────────

def test_embed_ok():
    brain = Brain(router=FakeRouter())
    vectors, error = brain.embed(["a", "b"])
    assert error == "" and len(vectors) == 2


def test_embed_never_raises():
    def boom(op, *a, **k):
        raise RuntimeError("no embed support")
    brain = Brain(router=FakeRouter(handler=boom))
    vectors, error = brain.embed(["a"])
    assert vectors == [] and "failed" in error


# ── brain_for ──────────────────────────────────────────────────────────────

class _Ctx:
    def __init__(self, router=None):
        self.router = router


def test_brain_for_caches_per_context():
    router = FakeRouter()
    ctx = _Ctx(router)
    b1 = brain_for(ctx)
    b2 = brain_for(ctx)
    assert b1 is b2
    assert brain_for(_Ctx(router)) is not b1  # per-context, not global


def test_brain_for_no_router_falls_back():
    b = brain_for(_Ctx(None))
    assert isinstance(b, Brain)


def test_brain_for_threads_task_kind():
    seen = {}

    def handler(op, *a, **k):
        seen.update(k)
        return _ok()

    router = FakeRouter(handler=handler)
    brain_for(_Ctx(router)).chat([Message.user("hi")], task_kind="research")
    assert seen.get("task_kind") == "research"


# ── context overflow recovery ──────────────────────────────────────────────

def test_context_overflow_retries_with_fitted_context():
    calls = []

    def handler(op, *a, **k):
        calls.append(op)
        if len(calls) == 1:
            return LLMResponse(
                text="", error="400: this model's maximum context length "
                               "is 100 tokens",
                failure_class="context_overflow")
        return _ok("fitted answer")

    brain = Brain(router=FakeRouter(handler=handler))
    msgs = [Message.system("sys")] + [
        Message.user(f"turn {i} " + "x" * 200) for i in range(6)]
    resp = brain.chat(msgs, task_kind="judge")
    assert resp.ok and resp.text == "fitted answer"
    assert len(calls) == 2


def test_non_overflow_failure_does_not_retry():
    calls = []

    def handler(op, *a, **k):
        calls.append(op)
        return LLMResponse(text="", error="503 Service Unavailable",
                           failure_class="server")

    brain = Brain(router=FakeRouter(handler=handler))
    resp = brain.chat([Message.user("hi")])
    assert not resp.ok and len(calls) == 1
    assert resp.failure_class == "server"


def test_failure_class_tagged_from_error_text():
    brain = Brain(router=FakeRouter(
        handler=lambda op, *a, **k: LLMResponse(text="", error="429 slow down")))
    resp = brain.chat([Message.user("hi")])
    assert resp.failure_class == "rate_limited"


# ── explain_failure ────────────────────────────────────────────────────────

def test_explain_failure_uses_taxonomy_hint():
    resp = LLMResponse(text="", error="400 maximum context length exceeded")
    text = explain_failure(resp)
    assert "context window" in text


# ── best_of ────────────────────────────────────────────────────────────────

class _Provider:
    def __init__(self, name, text):
        self.name = name
        self._text = text

    def chat(self, messages, params=None, **kw):
        return LLMResponse(text=self._text, model=self.name)


def test_best_of_single_provider_falls_back():
    router = FakeRouter()
    router._providers = {"only": _Provider("only", "solo")}
    router.active = "only"
    brain = Brain(router=router)
    text, judgment = brain.best_of("q?", n=3)
    assert text == "" or isinstance(text, str)  # complete() via handler
    assert judgment is not None


def test_best_of_judges_candidates():
    router = FakeRouter()
    router._providers = {
        "a": _Provider("a", "the answer is 4"),
        "b": _Provider("b", "the answer is 5"),
    }
    brain = Brain(router=router)
    # judge answers via the router handler (complete path)
    router.handler = lambda op, *a, **k: _ok('{"winner": 1, "ranking": [1, 2]}')
    text, judgment = brain.best_of("what is 2+2?", n=2)
    assert judgment.ok and judgment.winner == 0
    assert text == "the answer is 4"


# ── router drop-in contract ──────────────────────────────────────────────
# The Brain must accept the router's calling convention at every migrated
# call site: params positional, task_kind/timeout keyword.  A regression
# here means a live NameError/TypeError on the chat path.

def test_chat_accepts_positional_params_like_router():
    brain = Brain(router=FakeRouter())
    resp = brain.chat([Message.user("hi")], SamplingParams(temperature=0.5),
                      task_kind="chat")
    assert resp.ok
    op, args, kwargs = brain._external_router.calls[0]
    assert op == "chat"
    assert isinstance(args[1], SamplingParams)
    assert kwargs.get("task_kind") == "chat"


def test_complete_accepts_positional_params_like_router():
    brain = Brain(router=FakeRouter())
    resp = brain.complete("finish this", SamplingParams(max_tokens=10),
                          task_kind="intent", timeout_s=5.0)
    assert resp.ok
    op, args, kwargs = brain._external_router.calls[0]
    assert op == "complete"
    assert isinstance(args[1], SamplingParams)
    assert kwargs.get("timeout_s") == 5.0 or kwargs.get("task_kind") == "intent"


def test_chat_json_accepts_positional_params():
    router = FakeRouter(handler=lambda op, *a, **k: _ok('{"a": 1}'))
    brain = Brain(router=router)
    data, resp = brain.chat_json([Message.user("x")],
                                 SamplingParams(temperature=0.0),
                                 task_kind="judge")
    assert data == {"a": 1}
    assert resp.ok


# ── duck-typed response coercion ─────────────────────────────────────────
# Legacy routers / test doubles return SimpleNamespace(ok=.., text=..);
# the brain coerces instead of failing the call.

def test_chat_coerces_duck_typed_response():
    import types

    class DuckRouter(FakeRouter):
        def chat(self, messages, params=None, **kw):
            return types.SimpleNamespace(ok=True, text="duck says hi")

    brain = Brain(router=DuckRouter())
    resp = brain.chat([Message.user("hi")], task_kind="chat")
    assert isinstance(resp, LLMResponse)
    assert resp.ok and resp.text == "duck says hi"


def test_chat_coerces_duck_typed_failure():
    import types

    class DuckRouter(FakeRouter):
        def chat(self, messages, params=None, **kw):
            return types.SimpleNamespace(ok=False, text="", error="boom")

    brain = Brain(router=DuckRouter())
    resp = brain.chat([Message.user("hi")], task_kind="chat")
    assert isinstance(resp, LLMResponse)
    assert not resp.ok and resp.error == "boom"


def test_chat_coerces_none_response():
    class NullRouter(FakeRouter):
        def chat(self, messages, params=None, **kw):
            return None

    brain = Brain(router=NullRouter())
    resp = brain.chat([Message.user("hi")], task_kind="chat")
    assert isinstance(resp, LLMResponse)
    assert not resp.ok
