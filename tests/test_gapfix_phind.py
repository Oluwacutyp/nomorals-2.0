"""Phind lane (52-websites #40): dev source pack + dev-answer mode.

The developer answer engine: a question (or a pasted stack trace) is
understood as a DevQuery, answered from the dev source pack (Devon's
own code index first, then the live web backends), synthesised with
real citations, and every self-contained code block is EXECUTED
before it is presented — a block that did not run green is labelled
unverified, never silently presented as working.

Failing-first contract: every test here imports the new names lazily
(inside the test), so on the pristine base each fails with an
ImportError/AttributeError for the right reason, and after the build
each passes on real behaviour. No test touches the network: retrieval
is always a fake ``search_fn`` returning canned federated hits, and
execution is always a fake ``run_fn``.
"""

from __future__ import annotations

PY_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "app.py", line 12, in <module>\n'
    "    foo()\n"
    "ValueError: invalid literal for int() with base 10: 'x'\n"
)

GO_PANIC = (
    "panic: runtime error: index out of range [3] with length 2\n"
    "\n"
    "goroutine 1 [running]:\n"
    "main.main()\n"
    "\t/app/main.go:14 +0x2a\n"
)

JS_ERROR = (
    "TypeError: Cannot read properties of undefined (reading 'map')\n"
    "    at App (App.js:10:15)\n"
    "    at renderWithHooks (react-dom.js:123:9)\n"
)

QUESTION = "how do I reverse a list in Python?"

SO_URL = "https://stackoverflow.com/questions/12345/reverse-a-list"
DOCS_URL = "https://docs.python.org/3/library/stdtypes.html#list.reverse"


def _hits():
    """Two canned federated hits: a Stack Overflow answer and the docs."""
    from nomorals.search.model import SearchResponse, SearchResult

    so = SearchResult(
        query=QUESTION,
        title="How to reverse a list in Python?",
        snippet=(
            "Use list.reverse() to reverse a list in place. "
            "It returns None and mutates the list. "
            "```python\n"
            "xs = [1, 2, 3]\n"
            "xs.reverse()\n"
            "print(xs)  # [3, 2, 1]\n"
            "```"
        ),
        source="web_ddgs",
        type="web",
        raw_score=9.0,
        provenance={"url": SO_URL, "provider": "web_ddgs"},
        source_id="web_ddgs:abc123",
    )
    docs = SearchResult(
        query=QUESTION,
        title="list.reverse — Built-in Types",
        snippet=(
            "list.reverse() reverses the elements of the list in place. "
            "The method returns None."
        ),
        source="web_searxng",
        type="web",
        raw_score=8.0,
        provenance={"url": DOCS_URL, "provider": "web_searxng"},
        source_id="web_searxng:def456",
    )
    return SearchResponse(
        query=QUESTION,
        hits=[so, docs],
        sources_searched=["web_ddgs", "web_searxng"],
        sources_skipped={},
    )


def _fake_search_factory(response=None):
    def _search(query, *, sources=None, fusion="rrf",
                source_weights=None, limit=10):
        return response if response is not None else _hits()

    return _search


def _ok_run(code, lang):
    return {"ok": True, "exit_code": 0, "stdout": "[3, 2, 1]\n",
            "stderr": "", "timed_out": False, "seconds": 0.12,
            "language": lang or "python"}


def _failing_run(code, lang):
    return {"ok": False, "exit_code": 1, "stdout": "",
            "stderr": "NameError: name 'xs' is not defined",
            "timed_out": False, "seconds": 0.05,
            "language": lang or "python"}


class TestParseDevQuery:
    def test_python_traceback(self):
        from nomorals.research.dev_answer import parse_dev_query

        q = parse_dev_query(PY_TRACEBACK)
        assert q.is_traceback is True
        assert q.language == "python"
        assert q.error_kind == "ValueError"
        assert "invalid literal" in q.error_text

    def test_go_panic(self):
        from nomorals.research.dev_answer import parse_dev_query

        q = parse_dev_query(GO_PANIC)
        assert q.is_traceback is True
        assert q.language == "go"
        assert "index out of range" in q.error_text

    def test_js_type_error(self):
        from nomorals.research.dev_answer import parse_dev_query

        q = parse_dev_query(JS_ERROR)
        assert q.is_traceback is True
        assert q.language == "javascript"
        assert q.error_kind == "TypeError"

    def test_plain_python_question(self):
        from nomorals.research.dev_answer import parse_dev_query

        q = parse_dev_query(QUESTION)
        assert q.is_traceback is False
        assert q.language == "python"
        assert q.question == QUESTION

    def test_framework_and_versions(self):
        from nomorals.research.dev_answer import parse_dev_query

        q = parse_dev_query(
            "why does django 4.2 raise ImproperlyConfigured on go 1.22?")
        assert q.framework == "django"
        assert q.versions.get("django") == "4.2"
        assert q.versions.get("go") == "1.22"

    def test_fenced_code_language(self):
        from nomorals.research.dev_answer import parse_dev_query

        q = parse_dev_query(
            "why is this slow?\n```go\nfor i := range xs {\n}\n```")
        assert q.language == "go"

    def test_rust_panic(self):
        from nomorals.research.dev_answer import parse_dev_query

        q = parse_dev_query(
            "thread 'main' panicked at src/main.rs:4:5:\n"
            "index out of bounds: the len is 2 but the index is 3")
        assert q.is_traceback is True
        assert q.language == "rust"


class TestDevPack:
    def test_pack_is_code_first_then_web(self):
        from nomorals.search.sources import DEV_PACK_SOURCES, DEV_PACK_WEIGHTS, valid_source_names

        assert DEV_PACK_SOURCES[0] == "code"
        for name in ("web_searxng", "web_ddgs", "web_tavily",
                     "web_serper", "web_exa", "web_brave"):
            assert name in DEV_PACK_SOURCES
        valid = valid_source_names()
        assert all(n in valid for n in DEV_PACK_SOURCES)
        assert all(w > 0 for w in DEV_PACK_WEIGHTS.values())
        assert DEV_PACK_WEIGHTS["code"] > DEV_PACK_WEIGHTS["web_brave"]
        assert set(DEV_PACK_WEIGHTS) == set(DEV_PACK_SOURCES)

    def test_pack_helper_matches_constants(self):
        from nomorals.search.sources import DEV_PACK_SOURCES, DEV_PACK_WEIGHTS, dev_pack

        sources, weights = dev_pack()
        assert sources == list(DEV_PACK_SOURCES)
        assert weights == dict(DEV_PACK_WEIGHTS)


class TestDomainTiers:
    def test_dev_domains_boosted_unknown_neutral(self):
        from nomorals.research.dev_answer import dev_boost

        assert dev_boost(SO_URL) > 1.0
        assert dev_boost("https://github.com/foo/bar/issues/1") > 1.0
        assert dev_boost(DOCS_URL) > 1.0
        assert dev_boost("https://example.com/some-blog-post") == 1.0
        assert dev_boost("https://random-dev-blog.example.org/x") == 1.0

    def test_stackoverflow_outranks_generic(self):
        from nomorals.research.dev_answer import dev_boost

        assert dev_boost(SO_URL) >= dev_boost(
            "https://www.geeksforgeeks.org/python-list-reverse/")


class TestAnswerDev:
    def test_ask_cites_real_sources_and_verifies_code(
            self, tmp_path):
        from nomorals.research.dev_answer import answer_dev_question, open_dev_answer_store

        store = open_dev_answer_store(str(tmp_path))
        ans = answer_dev_question(
            str(tmp_path), QUESTION, store=store,
            search_fn=_fake_search_factory(), run_fn=_ok_run)
        assert ans["ok"] is True
        assert "reverse" in ans["answer"].lower()
        assert len(ans["citations"]) == 2
        assert ans["citations"][0]["url"] in (SO_URL, DOCS_URL)
        assert all(c["url"].startswith("http") for c in ans["citations"])
        assert len(ans["code_blocks"]) == 1
        block = ans["code_blocks"][0]
        assert block["verified_by_execution"] is True
        assert block["verification"]["exit_code"] == 0
        assert "✅ verified" in ans["reply"]
        assert "stackoverflow.com" in ans["reply"]

    def test_failing_code_is_labelled_not_verified(self, tmp_path):
        from nomorals.research.dev_answer import answer_dev_question, open_dev_answer_store

        store = open_dev_answer_store(str(tmp_path))
        ans = answer_dev_question(
            str(tmp_path), QUESTION, store=store,
            search_fn=_fake_search_factory(), run_fn=_failing_run)
        block = ans["code_blocks"][0]
        assert block["verified_by_execution"] is False
        assert "NameError" in block["verification"]["stderr"]
        assert "⚠️ not verified" in ans["reply"]
        assert "✅ verified" not in ans["reply"]

    def test_no_runner_means_unverified_never_presented_as_working(
            self, tmp_path):
        from nomorals.research.dev_answer import answer_dev_question, open_dev_answer_store

        store = open_dev_answer_store(str(tmp_path))
        ans = answer_dev_question(
            str(tmp_path), QUESTION, store=store,
            search_fn=_fake_search_factory(), run_fn=None)
        block = ans["code_blocks"][0]
        assert block["verified_by_execution"] is False
        assert "runner" in block["verification"]["reason"].lower()
        assert "⚠️ not verified" in ans["reply"]

    def test_fragment_block_is_not_executed(self, tmp_path):
        from nomorals.research.dev_answer import answer_dev_question, open_dev_answer_store
        from nomorals.search.model import SearchResponse, SearchResult

        frag = SearchResult(
            query=QUESTION, title="frag", snippet=(
                "Do it like this:\n```python\nxs.reverse()\n"
                "# ... more setup ...\n```"),
            source="web_ddgs", type="web", raw_score=5.0,
            provenance={"url": SO_URL}, source_id="web_ddgs:frag")
        resp = SearchResponse(
            query=QUESTION, hits=[frag], sources_searched=["web_ddgs"],
            sources_skipped={})
        calls = []

        def _recording_run(code, lang):
            calls.append(code)
            return _ok_run(code, lang)

        store = open_dev_answer_store(str(tmp_path))
        ans = answer_dev_question(
            str(tmp_path), QUESTION, store=store,
            search_fn=_fake_search_factory(resp), run_fn=_recording_run)
        assert calls == []
        assert ans["code_blocks"][0]["verified_by_execution"] is False
        assert "fragment" in ans["code_blocks"][0]["verification"]["reason"]

    def test_no_hits_is_an_honest_refusal(self, tmp_path):
        from nomorals.research.dev_answer import answer_dev_question, open_dev_answer_store
        from nomorals.search.model import SearchResponse

        empty = SearchResponse(
            query="xqzvk blorp", hits=[], sources_searched=[],
            sources_skipped={"web_ddgs": "no results",
                             "web_searxng": "no instance configured"})
        store = open_dev_answer_store(str(tmp_path))
        ans = answer_dev_question(
            str(tmp_path), "xqzvk blorp", store=store,
            search_fn=_fake_search_factory(empty), run_fn=_ok_run)
        assert ans["ok"] is False
        assert "no" in ans["error"].lower()
        assert "web_ddgs" in ans["error"] or "sources" in ans["error"].lower()

    def test_follow_up_carries_prior_context(self, tmp_path):
        from nomorals.research.dev_answer import answer_dev_question, open_dev_answer_store

        store = open_dev_answer_store(str(tmp_path))
        first = answer_dev_question(
            str(tmp_path), QUESTION, store=store,
            search_fn=_fake_search_factory(), run_fn=_ok_run)
        assert first["ok"] is True
        second = answer_dev_question(
            str(tmp_path), "what about a copy instead?", store=store,
            search_fn=_fake_search_factory(), run_fn=_ok_run,
            answer_id=first["answer_id"])
        assert second["ok"] is True
        assert second["follow_up_of"] == first["answer_id"]
        assert second["prior_question"] == QUESTION

    def test_store_round_trip(self, tmp_path):
        from nomorals.research.dev_answer import answer_dev_question, open_dev_answer_store

        store = open_dev_answer_store(str(tmp_path))
        ans = answer_dev_question(
            str(tmp_path), QUESTION, store=store,
            search_fn=_fake_search_factory(), run_fn=_ok_run)
        listed = store.list()
        assert len(listed) == 1
        assert listed[0]["answer_id"] == ans["answer_id"]
        assert store.get(ans["answer_id"])["question"] == QUESTION
        assert store.get("no-such-id") is None

    def test_traceback_question_uses_error_text(self, tmp_path):
        from nomorals.research.dev_answer import answer_dev_question, open_dev_answer_store

        store = open_dev_answer_store(str(tmp_path))
        ans = answer_dev_question(
            str(tmp_path), PY_TRACEBACK, store=store,
            search_fn=_fake_search_factory(), run_fn=_ok_run)
        assert ans["ok"] is True
        assert ans["dev_query"]["is_traceback"] is True
        assert ans["dev_query"]["error_kind"] == "ValueError"


class _FakeRegistry:
    """Records the tool specs ``register()`` declares."""

    context = None

    def __init__(self):
        self.tools = {}
        self.calls = []

    def register(self, name, **kwargs):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco

    def call(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name == "run_code":
            from nomorals.core.result import Err
            return Err(Exception("run_code not registered"))
        from nomorals.core.result import Err
        return Err(Exception(f"unknown tool {name}"))


class _FakeOk:
    def __init__(self, value):
        self.ok = True
        self.value = value


class TestTool:
    def test_tool_registered_and_asks(self, tmp_path, monkeypatch):
        from nomorals.tools import dev_answer as dev_tool

        monkeypatch.setattr(
            "nomorals.research.dev_answer._default_search_fn",
            lambda: _fake_search_factory())
        registry = _FakeRegistry()
        registry.context = _FakeCtx(str(tmp_path))
        dev_tool.register(registry)
        assert "dev_answer" in registry.tools
        fn = registry.tools["dev_answer"]
        out = fn(question=QUESTION)
        assert out["ok"] is True
        # run_code is absent on the fake registry: blocks stay
        # unverified, labelled honestly — never presented as working.
        assert out["code_blocks"][0]["verified_by_execution"] is False

    def test_tool_verifies_when_run_code_exists(
            self, tmp_path, monkeypatch):
        from nomorals.tools import dev_answer as dev_tool

        monkeypatch.setattr(
            "nomorals.research.dev_answer._default_search_fn",
            lambda: _fake_search_factory())
        registry = _FakeRegistry()
        registry.context = _FakeCtx(str(tmp_path))

        def _call(name, **kwargs):
            registry.calls.append((name, kwargs))
            return _FakeOk(_ok_run(kwargs.get("code", ""),
                                   kwargs.get("lang", "")))

        registry.call = _call
        dev_tool.register(registry)
        out = registry.tools["dev_answer"](question=QUESTION)
        assert out["ok"] is True
        assert out["code_blocks"][0]["verified_by_execution"] is True
        assert any(c[0] == "run_code" for c in registry.calls)

    def test_tool_lists_and_shows(self, tmp_path, monkeypatch):
        from nomorals.tools import dev_answer as dev_tool

        monkeypatch.setattr(
            "nomorals.research.dev_answer._default_search_fn",
            lambda: _fake_search_factory())
        registry = _FakeRegistry()
        registry.context = _FakeCtx(str(tmp_path))
        dev_tool.register(registry)
        fn = registry.tools["dev_answer"]
        first = fn(question=QUESTION)
        assert fn(action="list")["ok"] is True
        assert len(fn(action="list")["answers"]) == 1
        shown = fn(action="show", answer_id=first["answer_id"])
        assert shown["ok"] is True
        assert shown["answer_id"] == first["answer_id"]


class _FakeCtx:
    def __init__(self, home):
        from types import SimpleNamespace

        self.settings = SimpleNamespace(home_path=home)
        self.router = None


class TestRegistration:
    def test_devask_is_a_public_command(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            LIST_ONELINERS,
            PUBLIC_COMMANDS,
            caller_may_run,
            list_catalog,
        )

        assert "devask" in CONTROL_COMMANDS
        assert CONTROL_COMMANDS["devask"] == (1, None)
        assert "devask" in PUBLIC_COMMANDS
        assert len(PUBLIC_COMMANDS) == 278
        assert "devask" in COMMAND_DETAILS
        assert "/devask" in COMMAND_DETAILS["devask"]["usage"]
        assert "devask" in LIST_ONELINERS
        assert "devask" in [c for _, kinds in LIST_GROUPS for c in kinds]
        assert [c for _, kinds in LIST_GROUPS for c in kinds].count(
            "devask") == 1
        assert "devask" in list_catalog()
        assert caller_may_run("devask", False) is True

    def test_dev_answer_tool_registered(self):
        from nomorals.tools import dev_answer as dev_tool
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        dev_tool.register(reg)
        assert "dev_answer" in reg._tools


class TestHandler:
    def test_devask_handler_answers_and_never_raises(
            self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
        from nomorals.search.model import SearchResponse

        def _search(query, *, sources=None, fusion="rrf",
                    source_weights=None, limit=10):
            if "xqzvk" in query:
                return SearchResponse(
                    query=query, hits=[], sources_searched=[],
                    sources_skipped={"web_ddgs": "no results"})
            return _hits()

        monkeypatch.setattr(
            "nomorals.research.dev_answer._default_search_fn",
            lambda: _search)
        runtime = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        runtime.settings = SimpleNamespace(home_path=str(tmp_path))
        handler = runtime._control_devask
        assert "reverse" in handler(QUESTION).lower()
        assert "usage" in handler("").lower()
        assert "no" in handler("xqzvk blorp").lower()

    def test_devask_handler_lists(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
        from nomorals.research.dev_answer import answer_dev_question, open_dev_answer_store

        monkeypatch.setattr(
            "nomorals.research.dev_answer._default_search_fn",
            lambda: _fake_search_factory())
        store = open_dev_answer_store(str(tmp_path))
        answer_dev_question(str(tmp_path), QUESTION, store=store,
                            search_fn=_fake_search_factory(),
                            run_fn=_ok_run)
        runtime = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        runtime.settings = SimpleNamespace(home_path=str(tmp_path))
        assert "reverse a list" in runtime._control_devask("list").lower()


class TestCli:
    @staticmethod
    def _context(tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
            settings=SimpleNamespace(home_path=str(tmp_path)))

    def test_nm_devask_answers(self, capsys, tmp_path, monkeypatch):
        import argparse

        from nomorals.cmdline.commands.devask import _cmd_devask

        monkeypatch.setattr(
            "nomorals.research.dev_answer._default_search_fn",
            lambda: _fake_search_factory())
        monkeypatch.setattr(
            "nomorals.cmdline.commands.devask._default_run_fn",
            lambda context: _ok_run)
        args = argparse.Namespace(devask_query=[QUESTION], json=False,
                                  limit=5)
        assert _cmd_devask(args, self._context(tmp_path)) == 0
        assert "reverse" in capsys.readouterr().out.lower()

    def test_nm_devask_unknown_exits_1(self, capsys, tmp_path, monkeypatch):
        import argparse

        from nomorals.cmdline.commands.devask import _cmd_devask
        from nomorals.search.model import SearchResponse

        empty = SearchResponse(
            query="xqzvk blorp", hits=[], sources_searched=[],
            sources_skipped={"web_ddgs": "no results"})
        monkeypatch.setattr(
            "nomorals.research.dev_answer._default_search_fn",
            lambda: _fake_search_factory(empty))
        args = argparse.Namespace(devask_query=["xqzvk blorp"], json=False,
                                  limit=5)
        assert _cmd_devask(args, self._context(tmp_path)) == 1

    def test_nm_devask_no_query_exits_2(self, capsys, tmp_path):
        import argparse

        from nomorals.cmdline.commands.devask import _cmd_devask

        args = argparse.Namespace(devask_query=[], json=False, limit=5)
        assert _cmd_devask(args, self._context(tmp_path)) == 2
