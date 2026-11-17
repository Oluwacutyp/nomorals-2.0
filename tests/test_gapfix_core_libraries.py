"""Guard pins for audit R3-62: four dormant core libraries are deleted.

`nomorals/core/output_style.py`, `nomorals/core/result_contract.py`,
`nomorals/core/clock.py` and `nomorals/core/selfheal_facade.py` each had
zero production importers (verified in GAPFIX62_SURVEY.md): their live
replacements already own the behavior — the presentation layer
(`social/chat/present.py` + `core/tool_render`), the result contract
(`core/result.py` `Outcome`/`Ok`/`Err`), the local clock defaults
(`policy.py::_DefaultClock`, `time.monotonic` callables), and the direct
`selfheal`/`self_heal` imports. Deletion is the verdict; these tests pin
that the modules stay gone and nothing in production still references
them.
"""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path

import nomorals.core as core_pkg

DELETED_MODULES = (
    "nomorals.core.output_style",
    "nomorals.core.result_contract",
    "nomorals.core.clock",
    "nomorals.core.selfheal_facade",
)

# Dotted / import spellings that would prove a live reference survived.
# ("clock" alone is far too common a word — only the module spellings
# count; the plain word appears in entry_clock, clock= params, etc.)
DELETED_SPELLINGS = (
    "output_style",
    "result_contract",
    "selfheal_facade",
    "core.clock",
    "from .clock",
    "from ..core.clock",
    "from nomorals.core.clock",
)


class TestDeletedCoreLibrariesGone(unittest.TestCase):
    def test_deleted_modules_unimportable(self):
        for name in DELETED_MODULES:
            with self.assertRaises(ModuleNotFoundError, msg=name):
                importlib.import_module(name)

    def test_no_production_source_references(self):
        root = Path(__file__).resolve().parents[1] / "nomorals"
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            for lineno, line in enumerate(
                    path.read_text(encoding="utf-8", errors="replace").splitlines(),
                    start=1):
                # Historical notes that record the deletion itself are
                # allowed (they carry the "R3-62" marker); anything else
                # naming a deleted module is a live reference that
                # survived.
                if "R3-62" in line:
                    continue
                for spelling in DELETED_SPELLINGS:
                    if spelling in line:
                        offenders.append(
                            f"{path.relative_to(root)}:{lineno}: {spelling!r}")
        self.assertEqual(offenders, [])

    def test_core_package_exports_no_clock(self):
        for name in ("Clock", "MonotonicClock", "SystemClock"):
            self.assertFalse(hasattr(core_pkg, name), name)
            self.assertNotIn(name, core_pkg.__all__)

    def test_live_replacements_intact(self):
        # The presentation layer owns output styling now.
        from nomorals.core.tool_render import render_tool_value
        from nomorals.social.chat.present import canonical_text

        self.assertIn("a: 1", render_tool_value({"a": 1}))
        self.assertTrue(callable(canonical_text))
        # The live result contract is core.result, not result_contract.
        from nomorals.core.result import Err, Ok

        self.assertTrue(Ok(1).ok)
        self.assertFalse(Err("x").ok)
        self.assertEqual(Ok(1).value, 1)
        # The self-healing halves are imported directly, never via facade.
        from nomorals.core import self_heal, selfheal

        self.assertIsNotNone(selfheal.SelfHealingExecutor)
        self.assertIsNotNone(self_heal.heal_one)
        # Policy still works on its own default clock (no core.clock).
        from nomorals.core.policy import Policy

        policy = Policy()
        self.assertIsNotNone(policy)


if __name__ == "__main__":
    unittest.main()
