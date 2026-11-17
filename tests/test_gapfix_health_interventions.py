"""Gapfix R3-57: the health interventions organ gets an owner surface.

Failing-first: ``control_interventions`` does not exist yet and
``/health`` has no ``exercise`` sub-verb, so every test here fails on
the unmodified tree. The verb wires the same organ the CBT pilot cell
proved (record -> re-validate -> render -> scan -> blocked on guard
hits), over the chat surface the owner already uses.
"""

import types
import unittest

from nomorals.health.interventions import (
    INTERVENTION_KINDS,
    InterventionStore,
)

_THOUGHT_FIELDS = (
    "situation=Missed the bus; "
    "automatic_thought=Everyone thinks I am unreliable; "
    "emotion=shame; evidence_for=I was late once; "
    "evidence_against=I am usually on time; "
    "balanced_thought=One late arrival does not define me"
)


def _control():
    from nomorals.health.interventions import control_interventions

    return control_interventions


def _store(tmp_path):
    return InterventionStore(tmp_path / "interventions.db")


class CreateTests(unittest.TestCase):
    def test_create_thought_record_stores_kind_and_fields(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path

            store = _store(Path(d))
            out = _control()(
                f"thought_record {_THOUGHT_FIELDS}", store=store)
            rows = store.list()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["kind"], "thought_record")
            self.assertEqual(rows[0]["fields"]["situation"],
                             "Missed the bus")
            self.assertEqual(rows[0]["fields"]["balanced_thought"],
                             "One late arrival does not define me")
            self.assertEqual(rows[0]["status"], "recorded")
            self.assertIn(rows[0]["id"], out)

    def test_missing_required_field_refuses_and_stores_nothing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path

            store = _store(Path(d))
            out = _control()(
                "thought_record situation=Missed the bus; "
                "emotion=shame", store=store)
            self.assertIn("automatic_thought", out)
            self.assertIn("evidence_for", out)
            self.assertEqual(store.list(), [])

    def test_unknown_kind_refuses_and_names_the_kinds(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path

            store = _store(Path(d))
            out = _control()("dream_journal situation=x", store=store)
            self.assertIn("unknown", out.lower())
            self.assertIn("thought_record", out)
            self.assertEqual(store.list(), [])

    def test_unknown_field_refuses_no_invented_defaults(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path

            store = _store(Path(d))
            out = _control()(
                f"thought_record {_THOUGHT_FIELDS}; mood=low",
                store=store)
            self.assertIn("mood", out)
            self.assertEqual(store.list(), [])

    def test_score_out_of_range_refuses(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path

            store = _store(Path(d))
            out = _control()(
                "thought_record situation=x; automatic_thought=y; "
                "emotion=z; evidence_for=a; evidence_against=b; "
                "balanced_thought=c; intensity_before=11",
                store=store)
            self.assertIn("intensity_before", out)
            self.assertIn("0 to 10", out)
            self.assertEqual(store.list(), [])

    def test_list_field_splits_on_commas(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path

            store = _store(Path(d))
            out = _control()(
                "sleep_routine bedtime=22:30; wake_time=06:30; "
                "wind_down=tea, book, lights low",
                store=store)
            rows = store.list()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["fields"]["wind_down"],
                             ["tea", "book", "lights low"])
            self.assertIn(rows[0]["id"], out)


class GuardTests(unittest.TestCase):
    def test_guard_language_blocks_but_keeps_the_record(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path

            store = _store(Path(d))
            out = _control()(
                "thought_record situation=This disease keeps me awake; "
                "automatic_thought=I will never sleep again; "
                "emotion=fear; evidence_for=I was awake; "
                "evidence_against=I slept last week; "
                "balanced_thought=Sleep comes and goes",
                store=store)
            rows = store.list()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "blocked")
            self.assertIn("blocked", out.lower())
            self.assertIn("disease", out)

    def test_clean_record_is_not_blocked(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path

            store = _store(Path(d))
            _control()(f"thought_record {_THOUGHT_FIELDS}", store=store)
            self.assertEqual(store.list()[0]["status"], "recorded")


class ReviewTests(unittest.TestCase):
    def _seeded(self, tmp):
        from pathlib import Path

        store = _store(Path(tmp))
        _control()(f"thought_record {_THOUGHT_FIELDS}", store=store)
        _control()(
            "behavioral_activation activity=Walk to the park; "
            "scheduled_for=Saturday morning",
            store=store)
        return store

    def test_list_shows_records_newest_first(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            store = self._seeded(d)
            out = _control()("list", store=store)
            self.assertIn("thought_record", out)
            self.assertIn("behavioral_activation", out)

    def test_list_filters_by_kind(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            store = self._seeded(d)
            out = _control()("list thought_record", store=store)
            self.assertIn("thought_record", out)
            self.assertNotIn("behavioral_activation", out)

    def test_show_renders_one_record(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            store = self._seeded(d)
            rid = store.list(kind="thought_record")[0]["id"]
            out = _control()(f"show {rid}", store=store)
            self.assertIn("Missed the bus", out)
            self.assertIn("balanced thought", out)

    def test_show_unknown_id_is_honest(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path

            store = _store(Path(d))
            out = _control()("show iv-doesnotexist", store=store)
            self.assertIn("no intervention record", out.lower())

    def test_complete_marks_completed_and_show_reflects_it(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            store = self._seeded(d)
            rid = store.list(kind="thought_record")[0]["id"]
            out = _control()(f"complete {rid}", store=store)
            self.assertIn("completed", out.lower())
            self.assertEqual(store.get(rid)["status"], "completed")
            self.assertIn("completed",
                          _control()(f"show {rid}", store=store).lower())

    def test_kinds_lists_all_seven_with_required_fields(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path

            store = _store(Path(d))
            out = _control()("kinds", store=store)
            for kind in INTERVENTION_KINDS:
                self.assertIn(kind, out)
            self.assertIn("situation", out)
            self.assertIn("activity", out)


class DispatchTests(unittest.TestCase):
    def test_health_exercise_routes_to_the_control_function(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        from nomorals.agents.partner.runtime import PartnerRuntime

        class _TimelineStub:
            def close(self):
                return None

        with tempfile.TemporaryDirectory() as d:
            store = _store(Path(d))
            ns = types.SimpleNamespace(_intervention_store=store)
            with mock.patch(
                    "nomorals.health.timeline.HealthTimeline",
                    _TimelineStub):
                out = PartnerRuntime._control_health(
                    ns, "exercise kinds")
            for kind in INTERVENTION_KINDS:
                self.assertIn(kind, out)

    def test_health_exercise_create_through_the_verb(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        from nomorals.agents.partner.runtime import PartnerRuntime

        class _TimelineStub:
            def close(self):
                return None

        with tempfile.TemporaryDirectory() as d:
            store = _store(Path(d))
            ns = types.SimpleNamespace(_intervention_store=store)
            with mock.patch(
                    "nomorals.health.timeline.HealthTimeline",
                    _TimelineStub):
                out = PartnerRuntime._control_health(
                    ns, f"exercise thought_record {_THOUGHT_FIELDS}")
            rows = store.list()
            self.assertEqual(len(rows), 1)
            self.assertIn(rows[0]["id"], out)

    def test_existing_health_verbs_still_work(self):
        from unittest import mock

        from nomorals.agents.partner.runtime import PartnerRuntime

        class _TimelineStub:
            def timeline(self, limit=50):
                return []

            def close(self):
                return None

        ns = types.SimpleNamespace()
        with mock.patch(
                "nomorals.health.timeline.HealthTimeline",
                _TimelineStub):
            out = PartnerRuntime._control_health(ns, "timeline")
        self.assertIn("empty", out.lower())


if __name__ == "__main__":
    unittest.main()
