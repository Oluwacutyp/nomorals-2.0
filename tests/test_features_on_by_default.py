"""ON-by-default: every feature flag defaults ON, and an explicit OFF
still disables it. (Owner rule: max capability, opt-out not opt-in.)"""

from __future__ import annotations

import unittest

from nomorals.agents.features import FEATURES, FeatureRegistry, feature_enabled
from nomorals.storage.db import Database


class _Ctx:
    def __init__(self, db):
        self.db = db


class TestFeaturesOnByDefault(unittest.TestCase):
    def test_every_feature_defaults_on(self) -> None:
        self.assertTrue(FEATURES, "FEATURES must not be empty")
        for name, (default_on, _desc) in FEATURES.items():
            self.assertTrue(
                default_on, f"feature {name!r} must default ON")

    def test_explicit_off_still_disables(self) -> None:
        db = Database(":memory:")
        db.migrate()
        try:
            reg = FeatureRegistry(db)
            # Default: ON.
            self.assertTrue(reg.get("arena"))
            self.assertTrue(reg.get("research"))
            self.assertTrue(reg.get("news"))
            # Explicit OFF wins.
            self.assertTrue(reg.set("arena", False))
            self.assertFalse(reg.get("arena"))
            # And ON again wins back.
            self.assertTrue(reg.set("arena", True))
            self.assertTrue(reg.get("arena"))
        finally:
            db.close()

    def test_feature_enabled_without_db_uses_default(self) -> None:
        # A context without a database is treated as "default" — which is
        # now ON for every feature.
        self.assertTrue(feature_enabled(_Ctx(None), "arena"))
        self.assertTrue(feature_enabled(_Ctx(None), "research"))
        self.assertTrue(feature_enabled(_Ctx(None), "news"))
        self.assertFalse(feature_enabled(_Ctx(None), "no_such_feature"))


if __name__ == "__main__":
    unittest.main()
