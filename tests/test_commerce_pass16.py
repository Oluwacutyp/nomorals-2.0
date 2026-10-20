"""Pass #16 (owner-corrected 2026-10-19) — commerce/money: money_apply
follows the owner's design.

The owner's design: ``money_apply`` drafts AND submits. ``submit=true``
is passed with an explicit user instruction and the tool submits
immediately via ``GigApplier.submit(explicit=True)``. The default
(``submit`` omitted/False) is draft/show only — never a surprise
submission. An unknown gig with submit=true gets an honest error, not
a submission attempt.

(An earlier pass removed the submit parameter on the auditor's own
call; the owner corrected it — "That was my design" — and it was
restored. These tests pin the design as the owner specified it.)
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


class MoneyApplyDesignTests(unittest.TestCase):
    def test_submit_parameter_exposed(self):
        """Owner design: the tool schema offers submit for use with an
        explicit user instruction."""
        calls = _registered()
        self.assertIn("money_apply", calls)
        params = calls["money_apply"].get("parameters") or {}
        self.assertIn(
            "submit", params,
            "money_apply must expose submit (owner design: submit=true "
            "with an explicit user instruction submits immediately)")

    def test_tool_fn_accepts_submit(self):
        import inspect
        calls = _registered()
        fn = calls["money_apply"]["fn"]
        sig = inspect.signature(fn)
        self.assertIn("submit", sig.parameters)
        self.assertEqual(sig.parameters["submit"].default, False)

    def test_default_is_draft_only(self):
        """Omitting submit must never submit."""
        calls = _registered()
        fn = calls["money_apply"]["fn"]
        res = fn(gig_id="nonexistent-gig")
        self.assertIsInstance(res, dict)
        self.assertNotEqual(res.get("status"), "submitted")

    def test_submit_unknown_gig_honest_error(self):
        """submit=true on an unknown gig: honest 'no gig' error, and no
        submission is attempted."""
        calls = _registered()
        fn = calls["money_apply"]["fn"]
        res = fn(gig_id="nonexistent-gig", submit=True)
        self.assertIsInstance(res, dict)
        self.assertFalse(res.get("ok"))
        self.assertIn("no gig", (res.get("error") or ""))
        self.assertNotEqual(res.get("status"), "submitted")

    def test_description_names_explicit_instruction(self):
        """The description must tie submit=true to an explicit user
        instruction (the owner's design contract)."""
        calls = _registered()
        desc = (calls["money_apply"].get("description") or "").lower()
        self.assertIn("submit=true", desc)
        self.assertIn("explicit user instruction", desc)


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
