"""WS2 P1 — the goal-class axis.

Covers: the class registry (descriptors as data), validation/refusal
of unknown class ids, the ``goal_class`` column on GoalTracker goals
(default ``general``, legacy rows read as ``general``), template →
class filing (tags kept), the stats by-class rollup, and the progress
-provider seam (measured readings pass through labelled ``measured``;
the manual fallback is labelled ``self-reported``, never dressed up).
"""
from __future__ import annotations

import time
import unittest

from nomorals.goals import (
    DEFAULT_CLASS_ID,
    GOAL_CLASSES,
    GOAL_TEMPLATES,
    ClassProgress,
    GoalTracker,
    class_for_template,
)
from nomorals.goals import classes as goal_classes
from nomorals.goals.classes import (
    GoalClassError,
    get_class,
    goal_progress,
    list_classes,
    register_progress_provider,
    require_class,
    validate_class_id,
)
from nomorals.storage.db import Database

UID = "ws2-user"

EXPECTED_IDS = [
    "careers", "productivity", "finance", "interests", "health",
    "learning", "build", "relationships", "trading", "general",
]


def _db() -> Database:
    return Database(":memory:")


class RegistryTests(unittest.TestCase):
    def test_exactly_the_ten_classes(self):
        self.assertEqual([c.id for c in list_classes()], EXPECTED_IDS)
        self.assertEqual(sorted(GOAL_CLASSES), sorted(EXPECTED_IDS))
        self.assertEqual(DEFAULT_CLASS_ID, "general")

    def test_descriptors_complete_and_consistent(self):
        for cid, descriptor in GOAL_CLASSES.items():
            self.assertEqual(descriptor.id, cid)
            self.assertTrue(descriptor.title.strip())
            self.assertTrue(descriptor.description.strip())
            self.assertTrue(descriptor.progress_model.strip())
            self.assertTrue(descriptor.review_cadence.strip())
            self.assertIsInstance(descriptor.template_names, tuple)
            self.assertIsInstance(descriptor.tool_hints, tuple)
            as_dict = descriptor.to_dict()
            self.assertEqual(as_dict["id"], cid)
            self.assertEqual(as_dict["template_names"],
                             list(descriptor.template_names))

    def test_template_ownership_is_real_and_exclusive(self):
        owned: dict[str, str] = {}
        for descriptor in GOAL_CLASSES.values():
            for name in descriptor.template_names:
                self.assertIn(name, GOAL_TEMPLATES)
                self.assertNotIn(name, owned)  # one owner per template
                owned[name] = descriptor.id
        self.assertEqual(owned, {
            "save-money": "finance",
            "fitness": "health",
            "learn-language": "learning",
            "read-books": "learning",
            "ship-project": "build",
        })

    def test_get_class_defaults_and_normalises(self):
        self.assertEqual(get_class(None).id, "general")
        self.assertEqual(get_class("").id, "general")
        self.assertEqual(get_class(" Finance ").id, "finance")
        self.assertEqual(validate_class_id(None), "general")
        self.assertEqual(validate_class_id("TRADING"), "trading")

    def test_unknown_class_refused_in_words(self):
        with self.assertRaises(GoalClassError) as ctx:
            get_class("astrology")
        message = str(ctx.exception)
        self.assertIn("astrology", message)
        self.assertIn("finance", message)  # known classes named
        # It is a ValueError, so the store's error handling catches it.
        self.assertIsInstance(ctx.exception, ValueError)
        with self.assertRaises(GoalClassError):
            validate_class_id("astrology")
        with self.assertRaises(GoalClassError):
            require_class("")

    def test_class_for_template(self):
        self.assertEqual(class_for_template("learn-language"), "learning")
        self.assertEqual(class_for_template("read-books"), "learning")
        self.assertEqual(class_for_template("fitness"), "health")
        self.assertEqual(class_for_template("ship-project"), "build")
        self.assertEqual(class_for_template("save-money"), "finance")
        # declutter is a built-in owned by no class -> general.
        self.assertEqual(class_for_template("declutter"), "general")
        # Not a built-in at all -> no mapping.
        self.assertIsNone(class_for_template("my-custom-template"))
        self.assertIsNone(class_for_template(""))
        self.assertIsNone(class_for_template(None))


class StoreAxisTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_defaults_to_general(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Water the plants")
        self.assertEqual(goal.goal_class, "general")
        self.assertEqual(goal.to_dict()["goal_class"], "general")
        fetched = await gt.get(goal.goal_id)
        self.assertEqual(fetched.goal_class, "general")

    async def test_create_with_explicit_class(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Save for rent", goal_class="finance")
        self.assertEqual(goal.goal_class, "finance")

    async def test_create_with_unknown_class_refused_and_stores_nothing(self):
        gt = GoalTracker(_db())
        with self.assertRaises(GoalClassError):
            await gt.create(UID, "Read the stars", goal_class="astrology")
        self.assertEqual(await gt.list_goals(UID), [])

    async def test_update_refiles_and_validates(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Morning pages")
        moved = await gt.update(goal.goal_id, goal_class="learning")
        self.assertEqual(moved.goal_class, "learning")
        # An unrelated edit leaves the class alone.
        edited = await gt.update(goal.goal_id, priority=5)
        self.assertEqual(edited.goal_class, "learning")
        with self.assertRaises(GoalClassError):
            await gt.update(goal.goal_id, goal_class="astrology")
        self.assertEqual((await gt.get(goal.goal_id)).goal_class, "learning")

    async def test_legacy_database_without_the_column(self):
        db = _db()
        # A pre-WS2 goals table: no goal_class (and no kind) column.
        db.execute("""
            CREATE TABLE goals (
                goal_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                progress REAL NOT NULL DEFAULT 0.0,
                priority INTEGER NOT NULL DEFAULT 0,
                target_date REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL,
                tags TEXT NOT NULL DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                workspace TEXT NOT NULL DEFAULT 'default'
            )
        """)
        db.execute(
            "INSERT INTO goals (goal_id, user_id, title, created_at, updated_at)"
            " VALUES ('legacy-1', ?, 'Old goal', 1.0, 1.0)", (UID,))
        gt = GoalTracker(db)  # _ensure_schema adds the column
        legacy = await gt.get("legacy-1")
        self.assertIsNotNone(legacy)
        self.assertEqual(legacy.goal_class, "general")
        # And it can be filed under a class afterwards.
        moved = await gt.update("legacy-1", goal_class="health")
        self.assertEqual(moved.goal_class, "health")

    async def test_list_filter_by_class(self):
        gt = GoalTracker(_db())
        await gt.create(UID, "Budget", goal_class="finance")
        await gt.create(UID, "Run 5k", goal_class="health")
        await gt.create(UID, "Misc")
        finance = await gt.list_goals(UID, goal_class="finance")
        self.assertEqual([g.title for g in finance], ["Budget"])
        general = await gt.list_goals(UID, goal_class="general")
        self.assertEqual([g.title for g in general], ["Misc"])
        with self.assertRaises(GoalClassError):
            await gt.list_goals(UID, goal_class="astrology")

    async def test_stats_by_class(self):
        gt = GoalTracker(_db())
        await gt.create(UID, "Budget", goal_class="finance")
        overdue = await gt.create(
            UID, "File taxes", goal_class="finance",
            target_date=time.time() - 86400)
        await gt.create(UID, "Run 5k", goal_class="health")
        await gt.create(UID, "Misc")
        stats = await gt.stats(UID)
        by_class = stats["by_class"]
        self.assertEqual(by_class["finance"]["total"], 2)
        self.assertEqual(by_class["finance"]["overdue"], 1)
        self.assertEqual(by_class["health"]["total"], 1)
        self.assertEqual(by_class["general"]["total"], 1)
        self.assertNotIn("careers", by_class)
        self.assertEqual(overdue.goal_class, "finance")


class TemplateClassTests(unittest.IsolatedAsyncioTestCase):
    async def test_builtin_templates_file_under_their_class(self):
        gt = GoalTracker(_db())
        cases = {
            "learn-language": "learning",
            "read-books": "learning",
            "fitness": "health",
            "ship-project": "build",
            "save-money": "finance",
            "declutter": "general",
        }
        for name, expected in cases.items():
            goal = await gt.create_from_template(UID, name, subject="X", count="3")
            self.assertEqual(goal.goal_class, expected, name)
            # Tags are kept for compatibility.
            self.assertEqual(goal.tags, list(GOAL_TEMPLATES[name]["tags"]))

    async def test_explicit_class_overrides_template_default(self):
        gt = GoalTracker(_db())
        goal = await gt.create_from_template(
            UID, "save-money", subject="a car", goal_class="build")
        self.assertEqual(goal.goal_class, "build")
        self.assertEqual(goal.tags, ["money", "finance"])

    async def test_custom_template_stays_general(self):
        gt = GoalTracker(_db())
        await gt.save_template(UID, "my-rhythm", "My rhythm",
                               tags=["home"], subgoals=["step one"])
        goal = await gt.create_from_template(UID, "my-rhythm")
        self.assertEqual(goal.goal_class, "general")


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        goal_classes._PROGRESS_PROVIDERS.pop("finance", None)
        goal_classes._PROGRESS_PROVIDERS.pop("general", None)

    async def test_manual_fallback_is_labelled_self_reported(self):
        db = _db()
        gt = GoalTracker(db)
        goal = await gt.create(UID, "Save", goal_class="finance")
        await gt.update_progress(goal.goal_id, 40.0)
        goal = await gt.get(goal.goal_id)
        reading = goal_progress(db, goal)
        self.assertEqual(reading.percent, 40.0)
        self.assertEqual(reading.basis, "self-reported")
        self.assertEqual(reading.source, "manual")
        self.assertIsNone(reading.measured_at)
        # Same through the tracker seam.
        via_tracker = await gt.measured_progress(goal.goal_id)
        self.assertEqual(via_tracker.basis, "self-reported")

    async def test_registered_provider_reading_passes_through_as_measured(self):
        db = _db()
        gt = GoalTracker(db)
        goal = await gt.create(UID, "Save", goal_class="finance")
        seen = {}

        def stub(db_arg, goal_arg, *, now=None):
            seen["db"] = db_arg
            seen["goal_id"] = goal_arg.goal_id
            # Claims self-reported + leaves fields blank: the seam
            # must still surface it as measured, with the provider's
            # own source named.
            return ClassProgress(
                percent=250.0, basis="self-reported", source="ledger",
                headline="funded from ledger",
                evidence={"category": "rent"})

        register_progress_provider("finance", stub)
        reading = goal_progress(db, goal, now=123.0)
        self.assertIs(seen["db"], db)
        self.assertEqual(seen["goal_id"], goal.goal_id)
        self.assertEqual(reading.basis, "measured")
        self.assertEqual(reading.percent, 100.0)  # clamped to 0-100
        self.assertEqual(reading.source, "ledger")
        self.assertEqual(reading.measured_at, 123.0)
        self.assertEqual(reading.headline, "funded from ledger")
        self.assertEqual(reading.evidence, {"category": "rent"})
        self.assertEqual(reading.to_dict()["basis"], "measured")

    async def test_provider_returning_none_falls_back(self):
        db = _db()
        gt = GoalTracker(db)
        goal = await gt.create(UID, "Save", goal_class="finance")
        register_progress_provider("finance", lambda db_, g, *, now=None: None)
        reading = goal_progress(db, goal)
        self.assertEqual(reading.basis, "self-reported")
        self.assertEqual(reading.percent, 0.0)

    async def test_provider_only_applies_to_its_class(self):
        db = _db()
        gt = GoalTracker(db)
        register_progress_provider(
            "finance",
            lambda db_, g, *, now=None: ClassProgress(
                percent=80.0, basis="measured", source="ledger"))
        # The probe class is `general`: since WS2 P6 every other
        # class carries a real provider, and general is manual by
        # design — a finance provider must never leak into it.
        other_goal = await gt.create(UID, "Run", goal_class="general")
        reading = goal_progress(db, other_goal)
        self.assertEqual(reading.basis, "self-reported")

    def test_register_provider_validates(self):
        with self.assertRaises(GoalClassError):
            register_progress_provider(
                "astrology", lambda db_, g, *, now=None: None)
        with self.assertRaises(GoalClassError):
            register_progress_provider("finance", "not-callable")

    async def test_measured_progress_missing_goal(self):
        gt = GoalTracker(_db())
        with self.assertRaises(KeyError):
            await gt.measured_progress("goal-nope")


if __name__ == "__main__":
    unittest.main()
