"""Power mode default-on posture: unlimited/unrestricted by default."""

import unittest
from types import SimpleNamespace

from nomorals.agents.power import PowerMode, power_mode_for


class FakeDB:
    def __init__(self):
        self.kv = {}
        self.audit = []

    def execute(self, sql, params=()):
        if "audit_log" in sql:
            self.audit.append(params)
        elif "kv_store" in sql:
            # KVStore's upsert binds (key, value, kind, updated_at,
            # expires_at) and its reads select that whole row back —
            # the fake must store the full row shape or every read
            # misses (the storage sweep, 75418251, widened the row
            # beyond the {"value"}-only shape this fake once modeled).
            self.kv[params[0]] = {
                "value": params[1], "kind": params[2],
                "updated_at": params[3], "expires_at": params[4],
            }

    def query_one(self, sql, params=()):
        if "kv_store" in sql:
            row = self.kv.get(params[0])
            return dict(row) if row is not None else None
        return None

    def audit_actions(self):
        return [p[3] for p in self.audit]


def make_context(power_default_on=True, owner_key=""):
    partner = SimpleNamespace(
        owner_key=owner_key,
        power_default_on=power_default_on,
        autonomy_mode="suggest",
        max_parallel_chats=5,
        max_proactive_dm_per_day=10,
        max_group_posts_per_day=5,
        history_window=50,
        typing_seconds=1.0,
    )
    settings = SimpleNamespace(
        partner=partner,
        autonomy=None,
        improvement=None,
        llm=SimpleNamespace(local_model=""),
    )
    return SimpleNamespace(settings=settings, extras={}, db=FakeDB(), router=None)


class TestPowerDefaultOn(unittest.TestCase):
    def test_default_on_auto_activates(self):
        ctx = make_context()
        power = power_mode_for(ctx)
        self.assertTrue(power.active)
        self.assertEqual(power.unlocked_by, "default")
        self.assertIn("power.auto_activate", ctx.db.audit_actions())
        # unlimited: caps removed
        self.assertEqual(ctx.settings.partner.max_proactive_dm_per_day, 0)
        self.assertEqual(ctx.settings.partner.autonomy_mode, "auto")

    def test_explicit_lock_beats_default(self):
        ctx = make_context()
        power = power_mode_for(ctx)
        self.assertTrue(power.active)
        power.lock(actor="test")
        self.assertFalse(power.active)
        # fresh process, same db: stays locked despite default-on
        ctx2 = make_context()
        ctx2.db = ctx.db
        power2 = power_mode_for(ctx2)
        self.assertFalse(power2.active)
        status = power2.status()
        self.assertTrue(status["explicitly_locked"])
        self.assertIn("max_parallel_chats", status["limited_profile"])

    def test_unlock_after_lock_reactivates(self):
        ctx = make_context(power_default_on=False, owner_key="s3cret")
        power = power_mode_for(ctx)
        self.assertFalse(power.active)
        # lock explicitly, then unlock with the key
        power.lock(actor="test")
        out = power.unlock("s3cret", actor="test")
        self.assertTrue(out["ok"])
        self.assertTrue(power.active)

    def test_default_off_stays_locked_without_key(self):
        ctx = make_context(power_default_on=False)
        power = power_mode_for(ctx)
        self.assertFalse(power.active)
        out = power.unlock("nope", actor="test")
        self.assertFalse(out["ok"])
        self.assertFalse(power.active)

    def test_limited_profile_reports_concrete_caps(self):
        ctx = make_context(power_default_on=False)
        power = power_mode_for(ctx)
        profile = power.limited_profile()
        self.assertEqual(profile["max_parallel_chats"], 5)
        self.assertEqual(profile["max_proactive_dm_per_day"], 10)
        self.assertEqual(profile["autonomy_mode"], "suggest")

    def test_status_shows_default_on_flag(self):
        ctx = make_context()
        power = power_mode_for(ctx)
        self.assertTrue(power.status()["default_on"])
        ctx2 = make_context(power_default_on=False)
        power2 = power_mode_for(ctx2)
        self.assertFalse(power2.status()["default_on"])


if __name__ == "__main__":
    unittest.main()
