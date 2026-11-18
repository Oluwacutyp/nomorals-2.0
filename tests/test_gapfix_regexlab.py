"""Regex101 lane (52-websites #41): the regex workbench.

Paste a pattern and sample text and get matches with full match
information (groups, named groups, positions), a plain-English
explanation of every token from a REAL pattern parser (never an LLM
guess), a step-budget debugger that catches catastrophic
backtracking instead of hanging, substitution testing, unit tests
run as a suite, code generation per language whose snippets EXECUTE
in codews and reproduce the session's matches, and the reverse loop
("write me a regex for X": generate, then PROVE against positive
and negative cases before answering).

Failing-first contract: every test here imports the new names lazily
(inside the test), so on the pristine base each fails with an
ImportError/AttributeError for the right reason, and after the build
each passes on real behaviour. No test touches the network and no
test needs a live engine: foreign-flavour execution always goes
through an injected fake ``run_fn``.
"""

from __future__ import annotations

# ── parser / explainer ────────────────────────────────────────────────

class TestParser:
    def test_word_boundary_escape_is_an_anchor(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern(r"\bword\b")
        assert out["ok"] is True
        kinds = [t["kind"] for t in out["tokens"]]
        assert kinds.count("anchor") == 2
        assert out["tokens"][0]["text"] == r"\b"
        assert out["tokens"][0]["plain"] == "a word boundary"
        assert out["tokens"][-1]["plain"] == "a word boundary"

    def test_token_tree_for_digit_pattern(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern(r"\d{3}-\d{4}")
        assert out["ok"] is True
        kinds = [t["kind"] for t in out["tokens"]]
        assert kinds == ["escape_class", "quantifier", "literal",
                         "escape_class", "quantifier"]
        assert out["tokens"][0]["text"] == r"\d"
        assert out["tokens"][0]["span"] == [0, 2]
        assert out["tokens"][1]["children"][0]["text"] == r"\d"
        assert out["tokens"][1]["plain"].startswith("exactly 3")

    def test_named_group_token(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern(r"(?P<year>\d{4})")
        assert out["ok"] is True
        group = out["tokens"][0]
        assert group["kind"] == "group"
        assert group["name"] == "year"
        assert group["number"] == 1
        assert group["children"][0]["kind"] == "escape_class"

    def test_alternation_and_anchors(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern(r"^cat|dog$")
        assert out["ok"] is True
        kinds = [t["kind"] for t in out["tokens"]]
        assert "anchor" in kinds
        assert "alternation" in kinds

    def test_invalid_pattern_reports_position(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern("(a")
        assert out["ok"] is False
        assert out["error"]
        assert isinstance(out["error_pos"], int)

    def test_backreference_token(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern(r"(a)\1")
        assert out["ok"] is True
        assert out["tokens"][1]["kind"] == "backref"
        assert out["tokens"][1]["number"] == 1

    def test_char_class_with_range_and_negation(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern(r"[^a-z0-9_]")
        assert out["ok"] is True
        cls = out["tokens"][0]
        assert cls["kind"] == "char_class"
        assert cls["negated"] is True
        assert "a-z" in cls["plain"]

    def test_lazy_and_possessive_quantifiers(self):
        from nomorals.tools.regex_lab import parse_pattern

        lazy = parse_pattern(r"a+?")
        assert lazy["ok"] is True
        assert lazy["tokens"][1]["kind"] == "quantifier"
        assert lazy["tokens"][1]["lazy"] is True
        poss = parse_pattern(r"a++")
        assert poss["ok"] is True
        assert poss["tokens"][1]["kind"] == "quantifier"
        assert poss["tokens"][1]["possessive"] is True

    def test_lookaround_tokens(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern(r"(?<=@)\w+(?=\.com)")
        assert out["ok"] is True
        kinds = [t["kind"] for t in out["tokens"]]
        assert kinds.count("lookbehind") == 1
        assert kinds.count("lookahead") == 1


class TestExplain:
    def test_explain_emailish_pattern(self):
        from nomorals.tools.regex_lab import explain_pattern

        out = explain_pattern(r"\b\w+@\w+\.\w+\b")
        assert out["ok"] is True
        joined = " ".join(t["plain"] for t in out["tokens"])
        assert "word character" in joined
        assert "@" in joined
        assert out["reference"]
        assert "quick reference" in out["reference"].lower() or \
            "\\d" in out["reference"]

    def test_explain_invalid_is_honest(self):
        from nomorals.tools.regex_lab import explain_pattern

        out = explain_pattern(r"a{2,1}")
        assert out["ok"] is False
        assert out["error"]


class TestFlavours:
    def test_go_rejects_backreference(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern(r"(a)\1", flavour="go")
        assert out["ok"] is False
        assert any("backreference" in u for u in out["unsupported"])

    def test_go_rejects_lookahead(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern(r"(?=x)a", flavour="go")
        assert out["ok"] is False
        assert any("lookahead" in u for u in out["unsupported"])

    def test_python_accepts_lookbehind(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern(r"(?<=@)\w+", flavour="python")
        assert out["ok"] is True
        assert out["unsupported"] == []

    def test_unknown_flavour_refused(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern(r"a", flavour="cobol")
        assert out["ok"] is False
        assert "unknown flavour" in out["error"]

    def test_javascript_named_group_syntax(self):
        from nomorals.tools.regex_lab import parse_pattern

        out = parse_pattern(r"(?<year>\d{4})", flavour="javascript")
        assert out["ok"] is True
        assert out["tokens"][0]["name"] == "year"


# ── matching ──────────────────────────────────────────────────────────

class TestMatchPython:
    def test_spans_and_groups(self):
        from nomorals.tools.regex_lab import match_pattern

        out = match_pattern(r"(\w+)@(\w+)", ["mail me at a@b please"])
        assert out["ok"] is True
        assert out["verifiable"] is True
        found = out["results"][0]["matches"]
        assert len(found) == 1
        assert found[0]["match"] == "a@b"
        assert found[0]["start"] == 11
        assert found[0]["groups"][0]["value"] == "a"
        assert found[0]["groups"][1]["value"] == "b"

    def test_named_groups_returned(self):
        from nomorals.tools.regex_lab import match_pattern

        out = match_pattern(
            r"(?P<year>\d{4})-(?P<month>\d{2})", ["2026-11-17"])
        found = out["results"][0]["matches"][0]
        assert found["groups"][0]["name"] == "year"
        assert found["groups"][0]["value"] == "2026"
        assert found["groups"][1]["name"] == "month"

    def test_no_match_is_empty_not_error(self):
        from nomorals.tools.regex_lab import match_pattern

        out = match_pattern(r"\d+", ["no digits here"])
        assert out["ok"] is True
        assert out["results"][0]["matches"] == []

    def test_case_insensitive_flag(self):
        from nomorals.tools.regex_lab import match_pattern

        out = match_pattern(r"hello", ["HELLO world"], flags="i")
        assert out["ok"] is True
        assert len(out["results"][0]["matches"]) == 1

    def test_multiline_anchor(self):
        from nomorals.tools.regex_lab import match_pattern

        out = match_pattern(r"^b", ["a\nb"], flags="m")
        assert out["ok"] is True
        assert len(out["results"][0]["matches"]) == 1

    def test_invalid_pattern_is_error(self):
        from nomorals.tools.regex_lab import match_pattern

        out = match_pattern(r"(a", ["aaa"])
        assert out["ok"] is False
        assert out["error"]


def _fake_js_run(expected):
    """A fake run_fn that asserts the JS program shape and returns
    canned node output for ``expected`` match results."""

    import json

    def _run(code, lang):
        assert lang == "javascript"
        assert "RegExp" in code
        assert ("hasIndices" in code or "'gd'" in code
                or '"d"' in code or "'d'" in code)
        return {"ok": True, "exit_code": 0, "stdout": json.dumps(
            {"results": expected}), "stderr": "", "timed_out": False}

    return _run


class TestMatchForeign:
    def test_javascript_via_run_fn(self):
        from nomorals.tools.regex_lab import match_pattern

        expected = [{"text": "abc 123",
                     "matches": [{"start": 4, "end": 7, "match": "123",
                                  "groups": []}]}]
        out = match_pattern(r"\d+", ["abc 123"], flavour="javascript",
                            run_fn=_fake_js_run(expected))
        assert out["ok"] is True
        assert out["verifiable"] is True
        assert out["results"][0]["matches"][0]["match"] == "123"

    def test_javascript_without_runner_is_honest(self):
        from nomorals.tools.regex_lab import match_pattern

        out = match_pattern(r"\d+", ["abc 123"], flavour="javascript")
        assert out["ok"] is True
        assert out["verifiable"] is False
        assert "not verified" in out["reason"] or "runner" in out["reason"]

    def test_go_runner_failure_is_honest(self):
        from nomorals.tools.regex_lab import match_pattern

        def _dead(code, lang):
            raise RuntimeError("go is not installed here")

        out = match_pattern(r"\d+", ["abc 123"], flavour="go",
                            run_fn=_dead)
        assert out["ok"] is True
        assert out["verifiable"] is False
        assert "go" in out["reason"]

    def test_java_has_no_engine_here(self):
        from nomorals.tools.regex_lab import match_pattern

        out = match_pattern(r"\d+", ["abc 123"], flavour="java")
        assert out["ok"] is True
        assert out["verifiable"] is False
        assert "not verifiable" in out["reason"]

    def test_pcre2_has_no_engine_here(self):
        from nomorals.tools.regex_lab import match_pattern

        out = match_pattern(r"\d+", ["abc 123"], flavour="pcre2")
        assert out["ok"] is True
        assert out["verifiable"] is False


# ── substitution + unit tests ─────────────────────────────────────────

class TestSubstituteAndTests:
    def test_substitute_python_backref(self):
        from nomorals.tools.regex_lab import substitute_pattern

        out = substitute_pattern(r"(\w+)@(\w+)", r"\2@\1",
                                 "a@b and c@d")
        assert out["ok"] is True
        assert out["result"] == "b@a and d@c"
        assert out["count"] == 2

    def test_substitute_js_without_runner(self):
        from nomorals.tools.regex_lab import substitute_pattern

        out = substitute_pattern(r"\d+", "#", "a1b2",
                                 flavour="javascript")
        assert out["ok"] is True
        assert out["verifiable"] is False

    def test_unit_tests_call_out_failures(self):
        from nomorals.tools.regex_lab import run_regex_tests

        out = run_regex_tests(r"^\d+$", [
            {"text": "123", "should_match": True},
            {"text": "12a", "should_match": False},
            {"text": "abc", "should_match": True},
        ])
        assert out["ok"] is True
        assert out["passed"] == 2
        assert out["failed"] == 1
        assert out["results"][2]["ok"] is False
        assert out["results"][2]["text"] == "abc"


# ── debugger ──────────────────────────────────────────────────────────

class TestDebugTrace:
    def test_simple_trace_records_steps(self):
        from nomorals.tools.regex_lab import debug_trace

        out = debug_trace(r"a+b", "aaab")
        assert out["ok"] is True
        assert out["matched"] is True
        assert out["end"] == 4
        assert out["steps_taken"] > 0
        assert out["blow_up"] is False
        assert all("action" in s and "pos" in s for s in out["steps"])

    def test_catastrophic_backtracking_caught(self):
        from nomorals.tools.regex_lab import debug_trace

        out = debug_trace(r"(a+)+$", "a" * 25 + "b", budget=5000)
        assert out["ok"] is True
        assert out["blow_up"] is True
        assert out["steps_taken"] == 5000
        assert out["matched"] is False

    def test_trace_agrees_with_re_on_goldens(self):
        import re

        from nomorals.tools.regex_lab import debug_trace

        cases = [
            (r"\d{3}-\d{4}", "call 555-1234 now"),
            (r"^(cat|dog)s?$", "dogs"),
            (r"[^a-z]+", "abc123def"),
            (r"(?P<w>\w+)@\w+", "hi bob@x"),
            (r"a.*?b", "axxbxxb"),
        ]
        for pattern, text in cases:
            out = debug_trace(pattern, text)
            assert out["ok"] is True, pattern
            assert out["blow_up"] is False, pattern
            real = re.search(pattern, text)
            assert out["matched"] == (real is not None), pattern
            if real is not None:
                assert out["end"] == real.end(), pattern

    def test_trace_of_no_match(self):
        from nomorals.tools.regex_lab import debug_trace

        out = debug_trace(r"xyz", "abc")
        assert out["ok"] is True
        assert out["matched"] is False
        assert out["blow_up"] is False


# ── code generation + verification ────────────────────────────────────

class TestCodegen:
    def test_python_snippet_shape(self):
        from nomorals.tools.regex_lab import generate_code

        out = generate_code(r"\d+", "python", "python",
                            texts=["abc 123"])
        assert out["ok"] is True
        assert "compiled.finditer" in out["code"]
        assert "json.dumps" in out["code"]

    def test_javascript_snippet_shape(self):
        from nomorals.tools.regex_lab import generate_code

        out = generate_code(r"\d+", "python", "javascript",
                            texts=["abc 123"])
        assert out["ok"] is True
        assert "RegExp" in out["code"]

    def test_go_snippet_shape(self):
        from nomorals.tools.regex_lab import generate_code

        out = generate_code(r"\d+", "python", "go", texts=["abc 123"])
        assert out["ok"] is True
        assert "regexp" in out["code"]

    def test_verify_matching_output(self):
        import json

        from nomorals.tools.regex_lab import verify_code

        expected = [{"text": "abc 123",
                     "matches": [{"start": 4, "end": 7, "match": "123",
                                  "groups": []}]}]

        def _run(code, lang):
            return {"ok": True, "exit_code": 0,
                    "stdout": json.dumps({"results": expected}),
                    "stderr": "", "timed_out": False}

        out = verify_code("print('x')", "python", expected, run_fn=_run)
        assert out["ok"] is True
        assert out["verified"] is True

    def test_verify_mismatch_is_labelled(self):
        import json

        from nomorals.tools.regex_lab import verify_code

        expected = [{"text": "abc 123",
                     "matches": [{"start": 4, "end": 7, "match": "123",
                                  "groups": []}]}]
        wrong = [{"text": "abc 123", "matches": []}]

        def _run(code, lang):
            return {"ok": True, "exit_code": 0,
                    "stdout": json.dumps({"results": wrong}),
                    "stderr": "", "timed_out": False}

        out = verify_code("print('x')", "python", expected, run_fn=_run)
        assert out["ok"] is True
        assert out["verified"] is False
        assert out["diff"]

    def test_verify_without_runner_is_honest(self):
        from nomorals.tools.regex_lab import verify_code

        out = verify_code("print('x')", "python", [])
        assert out["ok"] is True
        assert out["verified"] is False
        assert "runner" in out["reason"]


# ── reverse loop: generate, then prove ────────────────────────────────

class TestGenerate:
    def test_template_email_proves(self):
        from nomorals.tools.regex_lab import generate_regex

        out = generate_regex(
            "an email address",
            positives=["a@b.com", "first.last+tag@sub.domain.org"],
            negatives=["not an email", "a@b", "@b.com", "a b@c.com"])
        assert out["ok"] is True
        assert out["proved"] is True
        assert out["failures"] == []
        assert out["template"] == "email"

    def test_template_ipv4_proves(self):
        from nomorals.tools.regex_lab import generate_regex

        out = generate_regex(
            "an IPv4 address",
            positives=["192.168.0.1", "8.8.8.8"],
            negatives=["999.1.1.1", "1.2.3", "abc"])
        assert out["ok"] is True
        assert out["proved"] is True

    def test_unknown_description_without_model_refuses(self):
        from nomorals.tools.regex_lab import generate_regex

        out = generate_regex(
            "a frobnicate quux pattern",
            positives=["quux1"], negatives=["nope"])
        assert out["ok"] is False
        assert "no model" in out["error"] or "no template" in out["error"]

    def test_model_pattern_that_fails_is_not_presented(self):
        from nomorals.tools.regex_lab import generate_regex

        def _llm(prompt):
            return "```\n[a-z]+\n```"

        out = generate_regex(
            "a code like abc123",
            positives=["abc123", "xy9"], negatives=["abc", "123"],
            llm_fn=_llm)
        assert out["ok"] is True
        assert out["proved"] is False
        assert out["failures"]
        # the failing case is named, never hidden
        assert any("abc" in f or "123" in f for f in out["failures"])


# ── store, sessions, links, reply ─────────────────────────────────────

class TestStoreAndSession:
    def test_store_round_trip(self, tmp_path):
        from nomorals.tools.regex_lab import open_regex_store, run_regex_session

        store = open_regex_store(str(tmp_path))
        rec = run_regex_session(
            str(tmp_path), pattern=r"\d+", texts=["abc 123"],
            store=store)
        assert rec["ok"] is True
        assert rec["session_id"]
        listed = store.list()
        assert len(listed) == 1
        assert listed[0]["session_id"] == rec["session_id"]
        assert store.get(rec["session_id"])["pattern"] == r"\d+"
        assert store.get(rec["session_id"][:6]) is not None
        assert store.get("no-such-id") is None

    def test_session_records_tests_and_failures(self, tmp_path):
        from nomorals.tools.regex_lab import run_regex_session

        rec = run_regex_session(
            str(tmp_path), pattern=r"^\d+$",
            tests=[{"text": "123", "should_match": True},
                   {"text": "abc", "should_match": True}])
        assert rec["ok"] is True
        assert rec["tests"]["failed"] == 1
        assert rec["tests"]["passed"] == 1

    def test_link_round_trip(self):
        from nomorals.tools.regex_lab import decode_link, encode_link

        session = {"pattern": r"\d+", "flavour": "python", "flags": "i",
                   "tests": [{"text": "a1", "should_match": True}]}
        link = encode_link(session)
        assert link.startswith("regex:")
        back = decode_link(link)
        assert back["pattern"] == r"\d+"
        assert back["tests"][0]["text"] == "a1"
        assert decode_link("regex:!!!not-base64!!!") is None

    def test_format_reply_names_matches_and_failures(self, tmp_path):
        from nomorals.tools.regex_lab import format_regex_reply, run_regex_session

        rec = run_regex_session(
            str(tmp_path), pattern=r"^\d+$", texts=["123", "abc"],
            tests=[{"text": "abc", "should_match": True}])
        reply = format_regex_reply(rec)
        assert "123" in reply
        assert "abc" in reply
        assert "fail" in reply.lower()


# ── brain tool ────────────────────────────────────────────────────────

class _FakeCtx:
    def __init__(self, home):
        class _Settings:
            home_path = home

        self.settings = _Settings()
        self.router = None
        self.tools = None


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
        from nomorals.core.result import Err
        return Err(Exception(f"unknown tool {name}"))


class TestTool:
    def test_tool_registered_explain_and_match(self, tmp_path):
        from nomorals.tools import regex_lab as lab

        registry = _FakeRegistry()
        registry.context = _FakeCtx(str(tmp_path))
        lab.register(registry)
        assert "regex_lab" in registry.tools
        fn = registry.tools["regex_lab"]
        out = fn(action="match", pattern=r"\d+", texts=["abc 123"])
        assert out["ok"] is True
        assert out["matches"]["results"][0]["matches"][0][
            "match"] == "123"
        explained = fn(action="explain", pattern=r"\d+")
        assert explained["ok"] is True
        assert explained["explanation"]["tokens"]

    def test_tool_gen_without_model_or_template_refuses(self, tmp_path):
        from nomorals.tools import regex_lab as lab

        registry = _FakeRegistry()
        registry.context = _FakeCtx(str(tmp_path))
        lab.register(registry)
        out = registry.tools["regex_lab"](
            action="gen", description="a frobnicate quux pattern",
            positives=["quux1"], negatives=["nope"])
        assert out["ok"] is False

    def test_tool_list_and_show(self, tmp_path):
        from nomorals.tools import regex_lab as lab

        registry = _FakeRegistry()
        registry.context = _FakeCtx(str(tmp_path))
        lab.register(registry)
        fn = registry.tools["regex_lab"]
        made = fn(action="match", pattern=r"\d+", texts=["abc 123"])
        assert made["ok"] is True
        listed = fn(action="list")
        assert listed["ok"] is True
        assert len(listed["sessions"]) == 1
        shown = fn(action="show",
                   session_id=made["session_id"])
        assert shown["ok"] is True
        assert shown["pattern"] == r"\d+"


# ── chat surface + CLI ────────────────────────────────────────────────

class TestChatAndCli:
    @staticmethod
    def _runtime(tmp_path):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        rt.settings = SimpleNamespace(home_path=str(tmp_path))
        return rt

    def test_control_regex_explain(self, tmp_path):
        rt = self._runtime(tmp_path)
        out = rt._control_regex(r"\d+")
        assert "digit" in out.lower()

    def test_control_regex_match_and_store(self, tmp_path):
        rt = self._runtime(tmp_path)
        out = rt._control_regex(r"\d+ :: abc 123")
        assert "123" in out
        listed = rt._control_regex("list")
        assert "session" in listed.lower() or "\\d+" in listed

    def test_control_regex_bad_pattern_is_words(self, tmp_path):
        rt = self._runtime(tmp_path)
        out = rt._control_regex("(a")
        assert "invalid" in out.lower() or "unbalanced" in out.lower()

    def test_control_regex_empty_is_usage(self, tmp_path):
        rt = self._runtime(tmp_path)
        assert rt._control_regex("").startswith("usage:")

    @staticmethod
    def _context(tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
            settings=SimpleNamespace(home_path=str(tmp_path)))

    def test_cli_match_exits_zero(self, tmp_path, capsys):
        import argparse

        from nomorals.cmdline.commands.regex import _cmd_regex

        args = argparse.Namespace(
            regex_query=[r"\d+"], text=["abc 123"], flavour="python",
            flags="", test=[], sub=None, debug=False, code=None,
            gen=None, positives=[], negatives=[], json=False,
            limit=5)
        assert _cmd_regex(args, self._context(tmp_path)) == 0
        assert "123" in capsys.readouterr().out

    def test_cli_bad_pattern_exits_one(self, tmp_path, capsys):
        import argparse

        from nomorals.cmdline.commands.regex import _cmd_regex

        args = argparse.Namespace(
            regex_query=["(a"], text=[], flavour="python",
            flags="", test=[], sub=None, debug=False, code=None,
            gen=None, positives=[], negatives=[], json=False,
            limit=5)
        assert _cmd_regex(args, self._context(tmp_path)) == 1

    def test_public_command_registered(self):
        from nomorals.social.chat.control import COMMAND_DETAILS, CONTROL_COMMANDS, PUBLIC_COMMANDS

        assert "regex" in PUBLIC_COMMANDS
        assert CONTROL_COMMANDS["regex"] == (1, None)
        assert "regex" in COMMAND_DETAILS
        assert len(PUBLIC_COMMANDS) == 279
