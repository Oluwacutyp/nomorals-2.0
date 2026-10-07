"""Tests for code-first tool calls (no network, no LLM)."""

import unittest

from nomorals.agents.orchestration.code_exec import SafeCodeRunner, CodeSafetyError


def _tools():
    calls = []
    def web_search(query=""):
        calls.append(("web_search", query))
        return f"results for {query}"
    def add(a=0, b=0):
        calls.append(("add", a, b))
        return str(int(a) + int(b))
    return {"web_search": web_search, "add": add}, calls


class SafeRunnerTests(unittest.TestCase):
    def test_basic_tool_call(self):
        tools, calls = _tools()
        ok, obs = SafeCodeRunner(tools).run(
            'result = web_search(query="gigs")')
        self.assertTrue(ok)
        self.assertIn("results for gigs", obs)
        self.assertEqual(calls[0][0], "web_search")

    def test_loop_in_one_block(self):
        tools, calls = _tools()
        ok, obs = SafeCodeRunner(tools).run(
            'total = 0\n'
            'for i in range(3):\n'
            '    total += int(add(a=i, b=1))\n'
            'result = f"total={total}"')
        self.assertTrue(ok)
        self.assertEqual("total=6", obs)

    def test_import_blocked(self):
        tools, _ = _tools()
        ok, obs = SafeCodeRunner(tools).run('import os\nresult = "x"')
        self.assertFalse(ok)
        self.assertIn("blocked", obs)

    def test_dunder_blocked(self):
        tools, _ = _tools()
        ok, obs = SafeCodeRunner(tools).run('result = (1).__class__')
        self.assertFalse(ok)

    def test_unknown_name_blocked(self):
        tools, _ = _tools()
        ok, obs = SafeCodeRunner(tools).run('result = mystery_tool()')
        self.assertFalse(ok)

    def test_eval_blocked(self):
        tools, _ = _tools()
        ok, obs = SafeCodeRunner(tools).run('result = eval("1+1")')
        self.assertFalse(ok)

    def test_method_call_blocked(self):
        tools, _ = _tools()
        ok, obs = SafeCodeRunner(tools).run('result = "x".upper()')
        self.assertFalse(ok)

    def test_no_result(self):
        tools, _ = _tools()
        ok, obs = SafeCodeRunner(tools).run('x = 1')
        self.assertTrue(ok)
        self.assertIn("no result set", obs)

    def test_exception_in_tool(self):
        def boom(**kw):
            raise RuntimeError("kaput")
        ok, obs = SafeCodeRunner({"boom": boom}).run('result = boom()')
        self.assertFalse(ok)
        self.assertIn("RuntimeError", obs)

    def test_empty_and_long(self):
        tools, _ = _tools()
        ok, _ = SafeCodeRunner(tools).run("")
        self.assertFalse(ok)
        ok, obs = SafeCodeRunner(tools).run("x = 1\n" * 2000)
        self.assertFalse(ok)
        self.assertIn("too long", obs)


class CodeAdapterTests(unittest.TestCase):
    def test_namespace_and_describe(self):
        from unittest.mock import MagicMock
        from nomorals.agents.orchestration.tools import CodeAdapter
        adapter = MagicMock()
        adapter._schemas.return_value = [
            {"name": "web_search",
             "parameters": {"query": {}},
             "description": "search the web"},
            {"name": "not-a-fn", "parameters": {}, "description": "skip me"},
        ]
        adapter.call.return_value = (True, "obs text")
        ca = CodeAdapter(adapter)
        ns = ca.namespace()
        self.assertIn("web_search", ns)
        self.assertNotIn("not-a-fn", ns)
        # the generated fn routes through ToolAdapter.call
        self.assertEqual("obs text", ns["web_search"](query="x"))
        adapter.call.assert_called_with("web_search", {"query": "x"})
        desc = ca.describe_code()
        self.assertIn("def web_search", desc)


if __name__ == "__main__":
    unittest.main()
