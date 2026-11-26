"""Update gate x webgames layering — the 2026-11-25 live bug.

The owner's /update from 62a66a75 to 1686b544 passed disk, quiesce,
snapshot, pull (57 files), verify_install, verify_tree and migrate
(schema v90), then ``health:_layering`` FAILED on exactly 9
violations and rolled the whole update back to 62a66a75 — the
goal-class ship never landed on his box:

  webgames/chat.py:7 (L0) imports nomorals.core.logging_setup (L1)
  webgames/driver.py:25 (L0) imports nomorals.core.errors (L1)
  webgames/driver.py:26 (L0) imports nomorals.core.logging_setup (L1)
  webgames/driver.py:84 (L0) imports nomorals.core.shell (L1)
  webgames/lagos.py:20 (L0) imports nomorals.core.errors (L1)
  webgames/lagos.py:21 (L0) imports nomorals.core.http (L1)
  webgames/lagos.py:22 (L0) imports nomorals.core.logging_setup (L1)
  webgames/runner.py:21 (L0) imports nomorals.core.logging_setup (L1)
  webgames/runner.py:59 (L0) imports nomorals.tools.proxy_rotate (L4)

Root cause (traced, not guessed): ``nomorals/webgames/`` — the
owner's browser-automation game driver (his 2026-11-24 pushes) — is
an UNMAPPED top-level package, so ``module_layer`` falls through to
SHARED_LAYER (0) and every one of its ``nomorals.*`` imports reads
as an upward violation.  The gate and the test suite AGREE here —
both read the same scan over the same baseline, and both failed on
the same 9 pairs; the pairs are genuinely ungrandfathered, not a
gate/test disagreement.  The fix is the missing layer assignment:
webgames is an L4 service organ (it drives Playwright + the proxy
pool, beside browser/; it imports core L1 downward and
tools.proxy_rotate L4 same-layer; nothing in the repo imports it),
so L4 is the unique lowest layer satisfying the pin rule
(max_imported 4 <= L <= min_importer, vacuous).

What this fix is NOT: no KNOWN_VIOLATIONS entry was added or
removed (audit decision D2 is untouched — the set stays exactly as
the owner left it; the 9 pairs are fixed by the layer, never hidden
by grandfathering), and the update gate itself is unchanged (it
already reads the one shared baseline with subset semantics since
the 2026-11-08 repair).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = str(REPO_ROOT / "tests")
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import test_layering  # noqa: E402

#: The 9 (importer, target) pairs the owner's 2026-11-25 /update
#: failed on.  None of them is grandfathered — that is the bug.
WEBGAMES_PAIRS = frozenset({
    ("nomorals.webgames.chat", "nomorals.core.logging_setup"),
    ("nomorals.webgames.driver", "nomorals.core.errors"),
    ("nomorals.webgames.driver", "nomorals.core.logging_setup"),
    ("nomorals.webgames.driver", "nomorals.core.shell"),
    ("nomorals.webgames.lagos", "nomorals.core.errors"),
    ("nomorals.webgames.lagos", "nomorals.core.http"),
    ("nomorals.webgames.lagos", "nomorals.core.logging_setup"),
    ("nomorals.webgames.runner", "nomorals.core.logging_setup"),
    ("nomorals.webgames.runner", "nomorals.tools.proxy_rotate"),
})

#: KNOWN_VIOLATIONS exactly as it stands at 1686b544: the 13 pairs
#: pinned by tests/test_update_health_layering.EXPECTED_BASELINE
#: (bf9afba) plus the 3 music_sources.live pairs grandfathered since.
#: D2 guard — this lane adds and removes NOTHING here.  When the
#: owner rules on D2 and L16 removes entries, update this pin in the
#: same change (and EXPECTED_BASELINE with it).
KNOWN_AT_1686B544 = frozenset({
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
    ("nomorals.music_sources.live", "nomorals.media.download_orchestrator"),
    ("nomorals.music_sources.live", "nomorals.media.playback"),
    ("nomorals.music_sources.live", "nomorals.media.resolver"),
    ("nomorals.music_sources.registry_bridge",
     "nomorals.media.source_registry"),
    ("nomorals.music_sources.registry_bridge", "nomorals.storage.kv"),
    ("nomorals.os.update", "nomorals.os.verify_install"),
})


def _health_layering():
    from nomorals.os.update import default_health_checks

    checks = default_health_checks(repo_dir=REPO_ROOT)
    return next(c for c in checks if c.__name__ == "_layering")


class TestWebgamesLayer(unittest.TestCase):
    def test_webgames_modules_are_L4(self) -> None:
        for name in ("chat.py", "driver.py", "lagos.py", "runner.py"):
            with self.subTest(module=name):
                self.assertEqual(
                    test_layering.module_layer(Path("webgames") / name),
                    4,
                    "webgames is an L4 service organ (see module docstring)")

    def test_webgames_layer_satisfies_the_pin_rule(self) -> None:
        # max_imported (tools.proxy_rotate, L4) <= 4, and nothing in
        # the repo imports webgames, so the min-importer bound is
        # vacuous — L4 is the lowest layer that fits, and the honest
        # one (a service organ, not the L5 composition layer).
        self.assertEqual(test_layering.LAYERS.get("webgames"), 4)


class TestUpdateGatePasses(unittest.TestCase):
    def test_scan_finds_no_new_violations(self) -> None:
        new, _known, _seen, _scanned = test_layering.scan_layering()
        self.assertEqual(new, [], "layer violations:\n  " + "\n  ".join(new))

    def test_health_layering_green_on_head_tree(self) -> None:
        # THE reproduction pin: the exact check /update runs, over
        # the tree as it ships.  Before the fix it returned False
        # ("layering failed: 9 new violation(s) beyond the 16
        # grandfathered pairs") — the live rollback.
        ok, detail = _health_layering()()
        self.assertTrue(ok, detail)
        self.assertIn("0 new", detail)

    def test_layering_test_method_green_on_head_tree(self) -> None:
        case = test_layering.TestLayering("test_no_upward_imports")
        case.test_no_upward_imports()


class TestNoCarveOut(unittest.TestCase):
    def test_webgames_pairs_are_not_grandfathered(self) -> None:
        # The fix is the layer assignment, never grandfathering:
        # none of the 9 pairs may appear in KNOWN_VIOLATIONS.
        self.assertFalse(
            WEBGAMES_PAIRS & test_layering.KNOWN_VIOLATIONS,
            "webgames pairs must be fixed by the layer, not hidden "
            f"in the grandfathered set: {WEBGAMES_PAIRS & test_layering.KNOWN_VIOLATIONS!r}")

    def test_known_violations_untouched_by_this_lane(self) -> None:
        # D2: no entry added, none removed — the set is exactly the
        # one the owner left standing at 1686b544.
        self.assertEqual(test_layering.KNOWN_VIOLATIONS, KNOWN_AT_1686B544)

    def test_webgames_pairs_no_longer_violate(self) -> None:
        _new, _known, seen, _scanned = test_layering.scan_layering()
        self.assertFalse(
            WEBGAMES_PAIRS & seen,
            f"webgames pairs still violate after the fix: {WEBGAMES_PAIRS & seen!r}")


if __name__ == "__main__":
    unittest.main()
