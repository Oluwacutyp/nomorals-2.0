"""Update health gate × layering baseline — the 2026-11-08 live bug.

The EC2 /update of fd6c871 passed disk, quiesce, snapshot, pull,
verify, and migrate, then ``health:_layering`` FAILED on exactly the
15 grandfathered layering violations and rolled the whole update
back.  The health check ran ``test_no_upward_imports`` raw — and that
test's ``KNOWN_VIOLATIONS`` had been emptied by T1.6 (2026-10-23)
while later waves re-introduced upward imports in agency/, anime/,
identity/, museum, music_sources/ and os/ that were never repaired
or re-grandfathered.  The owner's standing ruling (2026-10-25):
leave the grandfathered set; do not fix the violations.

The repair pinned here:

* the accepted baseline lives in exactly ONE place —
  ``tests/test_layering.py``'s ``KNOWN_VIOLATIONS`` — read by both
  the test suite and the update health gate (no duplicated lists);
* the health gate applies SUBSET semantics over that baseline:
  found ⊆ baseline passes (a repaired grandfathered violation must
  not fail an update), any violation NOT in the baseline fails and
  is named in the health detail.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = str(REPO_ROOT / "tests")
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import test_layering  # noqa: E402

#: The accepted grandfathered pairs at the fix's base (bf9afba):
#: 13 (importer, target) pairs producing 15 violation lines.
EXPECTED_BASELINE = frozenset({
    ("nomorals.agency.failures", "nomorals.core"),
    ("nomorals.anime.present", "nomorals.social.chat.tgbot_buttons"),
    ("nomorals.anime.present", "nomorals.social.chat.typography"),
    ("nomorals.anime.transport", "nomorals.core.http"),
    ("nomorals.community.group_gangs", "nomorals.games.join_codes"),
    ("nomorals.identity.chat", "nomorals.core.logging_setup"),
    ("nomorals.identity.chat", "nomorals.core.tool_render"),
    ("nomorals.museum", "nomorals.agents.apkforge.museum"),
    ("nomorals.museum", "nomorals.characters.store"),
    ("nomorals.museum_failures", "nomorals.core"),
    ("nomorals.music_sources.registry_bridge",
     "nomorals.media.source_registry"),
    ("nomorals.music_sources.registry_bridge", "nomorals.storage.kv"),
    ("nomorals.os.update", "nomorals.os.verify_install"),
})


def _health_layering():
    from nomorals.os.update import default_health_checks

    checks = default_health_checks(repo_dir=REPO_ROOT)
    return next(c for c in checks if c.__name__ == "_layering")


class TestBaselineIsOneSharedSet(unittest.TestCase):
    def test_baseline_pairs_are_the_accepted_set(self) -> None:
        self.assertEqual(test_layering.KNOWN_VIOLATIONS,
                         EXPECTED_BASELINE)

    def test_layering_test_method_green_on_real_tree(self) -> None:
        # The test suite reads the same baseline: with exactly the
        # grandfathered set present (and every pair still occurring),
        # test_no_upward_imports passes.  On the pristine base this
        # raises AssertionError with the 15 violations — the bug.
        case = test_layering.TestLayering("test_no_upward_imports")
        case.test_no_upward_imports()


class TestHealthLayeringSemantics(unittest.TestCase):
    def test_health_layering_green_on_real_tree(self) -> None:
        # THE reproduction pin: the health check over the tree as it
        # actually ships.  Pristine: False ("layering failed: Lists
        # differ: [the 15] != []") — the live failure.  Fixed: True.
        ok, detail = _health_layering()()
        self.assertTrue(ok, detail)
        self.assertIn("15 known grandfathered, 0 new", detail)

    def test_health_layering_names_a_new_violation(self) -> None:
        new = ["synthetic.py:9 (L0) imports nomorals.core (L1)"]
        canned = (new, [], {("nomorals.synthetic", "nomorals.core")},
                  {"nomorals.synthetic"})
        with mock.patch.object(test_layering, "scan_layering",
                               return_value=canned):
            ok, detail = _health_layering()()
        self.assertFalse(ok)
        self.assertIn("synthetic.py:9", detail)

    def test_health_layering_green_with_one_grandfathered_fixed(
            self) -> None:
        # Subset semantics: a baseline pair disappearing (a repair
        # landed) must NOT fail the update — the raw test's stale
        # ratchet is a suite concern, not a health verdict.
        known = [f"known.py:{i} (L0) imports nomorals.core (L1)"
                 for i in range(14)]
        seen = set(EXPECTED_BASELINE)
        seen.discard(("nomorals.museum", "nomorals.characters.store"))
        canned = ([], known, seen,
                  {pair[0] for pair in EXPECTED_BASELINE})
        with mock.patch.object(test_layering, "scan_layering",
                               return_value=canned):
            ok, detail = _health_layering()()
        self.assertTrue(ok, detail)
        self.assertIn("14 known grandfathered, 0 new", detail)


if __name__ == "__main__":
    unittest.main()
