"""Gap fix: /plan must be listed in a /list group.

``plan`` is in CONTROL_COMMANDS but was in no LIST_GROUPS group, so
``test_every_command_is_listed`` failed with missing == ['plan'].
The fix is one line: "plan" joins the "building for real — code &
missions" group next to its siblings skills / delegate / workflow.
"""

import unittest

from nomorals.social.chat.control import CONTROL_COMMANDS, LIST_GROUPS


class TestPlanListed(unittest.TestCase):
    def test_plan_is_in_a_list_group(self):
        listed = {c for _, v in LIST_GROUPS for c in v}
        self.assertIn("plan", listed, "'plan' is in no /list group")

    def test_plan_is_registered_and_owner_only(self):
        self.assertIn("plan", CONTROL_COMMANDS)
        from nomorals.social.chat.control import PUBLIC_COMMANDS

        self.assertNotIn("plan", PUBLIC_COMMANDS)

    def test_every_command_is_listed_missing_excludes_plan(self):
        listed = {c for _, v in LIST_GROUPS for c in v}
        missing = sorted(set(CONTROL_COMMANDS) - listed)
        self.assertNotIn("plan", missing,
                         f"'plan' still missing from /list: {missing}")


if __name__ == "__main__":
    unittest.main()
