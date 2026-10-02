"""``_attach_timeline``: the CLI persists bus events to the H2 timeline.

Without this wiring, ``nm timeline`` and ``nm mission replay`` only ever saw
what tests persisted — the production serving path never called
``Timeline.attach()``. These tests pin the wiring: attach persists to the
same database file ``nm timeline`` reads, and a broken attach never breaks
a command.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.cmdline.dispatch import _attach_timeline
from nomorals.core.events import global_bus
from nomorals.os.timeline import Timeline


def _ctx(db_path: str | None):
    return SimpleNamespace(
        db=SimpleNamespace(path=db_path),
        extras={},
    )


class AttachTimelineTests(unittest.TestCase):
    def tearDown(self) -> None:
        # Detach anything this test attached so later tests see a clean bus.
        extras = getattr(self, "_extras", None)
        tl = (extras or {}).get("timeline")
        if tl is not None:
            try:
                tl.detach(global_bus)
            finally:
                tl.close()

    def test_attach_persists_mission_events_to_context_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.db")
            ctx = _ctx(db_path)
            _attach_timeline(ctx)
            self._extras = ctx.extras
            self.assertIn("timeline", ctx.extras)

            mission_id = "msn_timeline_attach_probe"
            global_bus.publish(
                "mission.transition",
                {"mission_id": mission_id, "from_state": "CREATED",
                 "to_state": "PLANNED"},
            )

            # Read back exactly the way `nm timeline` does.
            tl = Timeline(db_path)
            try:
                rows = tl.query(mission_id=mission_id, limit=10)
            finally:
                tl.close()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["topic"], "mission.transition")
            self.assertEqual(rows[0]["mission_id"], mission_id)

    def test_attach_without_db_path_is_silent(self) -> None:
        ctx = _ctx(None)
        _attach_timeline(ctx)  # must not raise
        self.assertNotIn("timeline", ctx.extras)

    def test_broken_attach_never_raises(self) -> None:
        # Unwritable location: sqlite connect fails -> helper must swallow it.
        ctx = _ctx("/nonexistent-dir-xyz/state.db")
        _attach_timeline(ctx)  # must not raise
        # Either skipped or attached; never an exception, never a crash.
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
