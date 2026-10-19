"""Pass #16 — commerce/money guards: the money_apply tool must not submit.

The ``money_apply`` tool is registered with ``capability=NET_OUT``
(non-confirmable) and took ``submit: bool`` — prose said "pass
submit=true with an explicit user instruction", but nothing enforced
it. An LLM setting ``submit=True`` on its own initiative submitted a
gig application in the owner's name with no confirmation gate, no
trial-pattern go-ahead, bypassing the entire authority design that
``GigsAdapter.submit`` (explicit per-gig go-ahead) and the
``/money apply <gig_id> submit`` chat path (owner-only, user typed the
word) enforce.

The tool is now draft/show only. Submission stays on the two gated
paths.
"""

import unittest


class _FakeRegistry:
    def __init__(self):
        self.calls = {}

    def register(self, name, **kw):
        self.calls[name] = kw

        def deco(fn):
            self.calls[name]["fn"] = fn
            return fn

        return deco


def _registered():
    from nomorals.agents import opportunities as O
    reg = _FakeRegistry()
    O.register(reg)
    return reg.calls


class MoneyApplyToolGateTests(unittest.TestCase):
    def test_no_submit_parameter_exposed(self):
        """The tool schema must not offer a submit knob to the model."""
        calls = _registered()
        self.assertIn("money_apply", calls)
        params = calls["money_apply"].get("parameters") or {}
        self.assertNotIn(
            "submit", params,
            "money_apply must not expose submit — an LLM-set bool is not "
            "an explicit user instruction")

    def test_tool_fn_has_no_submit_kwarg(self):
        """The underlying function must not accept a submit flag either."""
        import inspect
        calls = _registered()
        fn = calls["money_apply"]["fn"]
        self.assertNotIn(
            "submit", inspect.signature(fn).parameters,
            "money_apply() must not take submit — draft/show only")

    def test_tool_never_submits(self):
        """Even a crafted call cannot flip the tool into submitting."""
        calls = _registered()
        fn = calls["money_apply"]["fn"]
        # No registered board submitters in this env → but the point is
        # the tool must not even attempt a submit path.
        res = fn(gig_id="nonexistent-gig")
        self.assertIsInstance(res, dict)
        self.assertNotEqual(res.get("status"), "submitted")
        self.assertFalse(res.get("ok") and res.get("status") == "submitted")

    def test_description_points_to_explicit_path(self):
        """The tool description must direct explicit submits to the
        owner-typed command, not invite submit=true."""
        calls = _registered()
        desc = (calls["money_apply"].get("description") or "").lower()
        self.assertNotIn("submit=true", desc)
        self.assertNotIn("submit immediately", desc)


class ExplicitChatPathPinTests(unittest.TestCase):
    def test_chat_submit_still_owner_typed(self):
        """Pin: the /money apply <id> submit chat path still treats a
        typed 'submit' as the explicit instruction (it is owner-gated
        by caller_may_run, not by this tool)."""
        from nomorals.agents.gig_applier import is_explicit_submit
        self.assertTrue(is_explicit_submit("submit"))
        self.assertTrue(is_explicit_submit("yes, submit it now"))
        self.assertFalse(is_explicit_submit(""))
        self.assertFalse(is_explicit_submit("maybe later"))

    def test_money_not_public(self):
        """/money stays off the public command list (owner-only gate)."""
        from nomorals.social.chat.control import (
            PUBLIC_COMMANDS, caller_may_run)
        self.assertNotIn("money", PUBLIC_COMMANDS)
        self.assertTrue(caller_may_run("money", True))
        self.assertFalse(caller_may_run("money", False))


if __name__ == "__main__":
    unittest.main()
