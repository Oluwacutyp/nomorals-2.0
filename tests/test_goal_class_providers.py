"""WS2 P3 — finance + trading measured-progress providers.

Pins:

* production wiring: importing the goals package registers the
  finance + trading providers (module-end registration in
  ``nomorals/goals/__init__.py``) — proven in-process AND in a fresh
  interpreter, so the providers can never be test-only shelfware;
* finance (``ledger_funded``): funded % comes from the savings
  store's own ``goal_progress`` over the real ledger — exact math,
  the link is an explicit ``finance-goal:<id>`` tag first, else a
  unique normalised name match; unlinked / ambiguous / stale-tag
  goals measure NOTHING (None → self-reported fallback intact);
* trading (``trading_journal``): realized P&L from the desk
  journal's close records at/after the goal's creation, divided by
  the goal's declared ``pnl-target:<amount>`` tag; no valid target
  = no measurement; pre-goal closes never count;
* failure containment: a broken store or a raising path resolver
  yields None + a named warning in the providers log — never an
  exception into a board — and the P2 surfaces flip from
  (self-reported) to (measured · <source>) purely by registration.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.finance.goals import GoalStore, create_goal
from nomorals.finance.ledger import Ledger
from nomorals.goals import GoalTracker, providers, surface
from nomorals.goals.classes import registered_provider
from nomorals.storage.db import Database

UID = "owner"
REPO_ROOT = Path(__file__).resolve().parent.parent

K500K = 50_000_000   # ₦500,000 in kobo
K150K = 15_000_000   # ₦150,000 in kobo


def _goal(**kwargs: Any) -> SimpleNamespace:
    base = dict(goal_id="goal_test", user_id=UID, title="",
                tags=[], created_at=0.0, goal_class="finance",
                progress=0.0, status="active")
    base.update(kwargs)
    return SimpleNamespace(**base)


class _ProviderCase(unittest.TestCase):
    """Tmp finance home + the providers' settings seam pointed at it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.home = Path(self._tmp.name)
        self.fin_dir = self.home / "finance"
        self.fin_dir.mkdir(parents=True)
        self._real_get_settings = providers.get_settings
        providers.get_settings = lambda: SimpleNamespace(home_path=self.home)

    def tearDown(self) -> None:
        providers.get_settings = self._real_get_settings
        self._tmp.cleanup()

    # seeding helpers -------------------------------------------------
    def savings_store(self) -> GoalStore:
        return GoalStore(self.fin_dir / "goals.json")

    def ledger(self) -> Ledger:
        return Ledger(self.fin_dir / "ledger.jsonl")

    def seed_savings(self, name: str = "Emergency Fund",
                     target_kobo: int = K500K, **kwargs: Any):
        return create_goal(name, target_kobo,
                           store=self.savings_store(), **kwargs)

    def contribute(self, amount_kobo: int, *, category: str = "savings",
                   ts: float) -> None:
        self.ledger().log(amount_kobo, category=category,
                          kind="income", ts=ts)


class FinanceProviderTests(_ProviderCase):

    def test_funded_percent_exact_from_real_ledger(self) -> None:
        saved = self.seed_savings()
        self.contribute(K150K, ts=saved.created_at + 60)
        goal = _goal(title="Emergency fund")  # case differs: normalised
        reading = providers.finance_progress(None, goal, now=saved.created_at + 3600)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.percent, 30.0)
        self.assertEqual(reading.basis, "measured")
        self.assertEqual(reading.source, "savings ledger")
        self.assertEqual(reading.evidence["contributed_kobo"], K150K)
        self.assertEqual(reading.evidence["target_kobo"], K500K)
        self.assertEqual(reading.evidence["finance_goal_id"], saved.id)
        self.assertEqual(reading.evidence["link"], "name")
        self.assertIn("Emergency Fund", reading.headline)

    def test_contributions_before_savings_goal_creation_do_not_count(self) -> None:
        saved = self.seed_savings()
        self.contribute(K150K, ts=saved.created_at - 86400)
        reading = providers.finance_progress(
            None, _goal(title="Emergency Fund"), now=saved.created_at + 3600)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.percent, 0.0)
        self.assertEqual(reading.evidence["contributed_kobo"], 0)

    def test_explicit_tag_wins_over_name_match(self) -> None:
        named = self.seed_savings("Emergency Fund", K500K)
        tagged = self.seed_savings("Car Fund", 20_000_000, category="car")
        self.contribute(10_000_000, category="car",
                        ts=tagged.created_at + 60)
        self.contribute(K150K, ts=named.created_at + 60)
        goal = _goal(title="Emergency Fund",
                     tags=[f"finance-goal:{tagged.id}"])
        reading = providers.finance_progress(
            None, goal, now=tagged.created_at + 3600)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.evidence["finance_goal_id"], tagged.id)
        self.assertEqual(reading.evidence["link"], "tag")
        self.assertEqual(reading.percent, 50.0)

    def test_unlinked_finance_goal_measures_nothing(self) -> None:
        self.seed_savings()
        reading = providers.finance_progress(
            None, _goal(title="A boat, somehow"), now=time.time())
        self.assertIsNone(reading)

    def test_ambiguous_name_match_measures_nothing(self) -> None:
        self.seed_savings("Fund", 10_000_000)
        self.seed_savings("Fund", 20_000_000)
        reading = providers.finance_progress(
            None, _goal(title="Fund"), now=time.time())
        self.assertIsNone(reading)

    def test_stale_tag_does_not_fall_through_to_name(self) -> None:
        self.seed_savings()
        self.contribute(K150K, ts=time.time())
        goal = _goal(title="Emergency Fund",
                     tags=["finance-goal:goal_nonexistent"])
        reading = providers.finance_progress(None, goal, now=time.time())
        self.assertIsNone(reading)

    def test_done_savings_goal_still_measures(self) -> None:
        saved = self.seed_savings()
        self.contribute(K500K, ts=saved.created_at + 60)
        self.assertTrue(self.savings_store().mark_done(saved.id))
        reading = providers.finance_progress(
            None, _goal(title="Emergency Fund"), now=saved.created_at + 3600)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.percent, 100.0)


class TradingProviderTests(_ProviderCase):

    def write_journal(self, records: list[Any]) -> Path:
        path = self.fin_dir / "trading_journal.jsonl"
        lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_realized_pnl_exact_and_scoped_to_goal_age(self) -> None:
        t0 = 1_700_000_000.0
        self.write_journal([
            {"ts": t0 - 100, "event": "paper_close", "pnl": 500.0},
            {"ts": t0 + 10, "event": "paper_close", "pnl": 250.0},
            {"ts": t0 + 20, "event": "live_close", "pnl": -50.0},
            {"ts": t0 + 30, "event": "paper_open", "pnl": 999.0},
            "{torn line",
            {"ts": t0 + 40, "event": "paper_close", "pnl": "abc"},
        ])
        goal = _goal(title="Grow the desk", goal_class="trading",
                     tags=["pnl-target:1000"], created_at=t0)
        reading = providers.trading_progress(None, goal, now=t0 + 100)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.percent, 20.0)
        self.assertEqual(reading.source, "trading journal")
        self.assertEqual(reading.evidence["realized_pnl"], 200.0)
        self.assertEqual(reading.evidence["closed_trades"], 2)
        self.assertEqual(reading.evidence["unreadable_journal_lines"], 1)
        self.assertEqual(reading.evidence["unreadable_pnl_records"], 1)
        self.assertIn("+200.00", reading.headline)

    def test_no_target_tag_measures_nothing(self) -> None:
        self.write_journal([
            {"ts": 10.0, "event": "paper_close", "pnl": 250.0}])
        goal = _goal(title="Grow the desk", goal_class="trading",
                     created_at=0.0)
        self.assertIsNone(providers.trading_progress(None, goal))

    def test_invalid_targets_measure_nothing(self) -> None:
        self.write_journal([
            {"ts": 10.0, "event": "paper_close", "pnl": 250.0}])
        for tag in ("pnl-target:0", "pnl-target:-5", "pnl-target:abc",
                    "pnl-target:"):
            goal = _goal(title="Grow the desk", goal_class="trading",
                         tags=[tag], created_at=0.0)
            self.assertIsNone(providers.trading_progress(None, goal),
                              f"tag {tag!r} must not measure")

    def test_no_journal_yet_measures_zero_with_honest_headline(self) -> None:
        goal = _goal(title="Grow the desk", goal_class="trading",
                     tags=["pnl-target:1000"], created_at=0.0)
        reading = providers.trading_progress(None, goal)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.percent, 0.0)
        self.assertEqual(reading.evidence["closed_trades"], 0)
        self.assertIn("no closed trades", reading.headline)


class FailureContainmentTests(_ProviderCase):

    def test_raising_path_resolution_is_contained_and_named(self) -> None:
        real = providers._finance_dir

        def boom() -> Path:
            raise RuntimeError("home exploded")

        providers._finance_dir = boom
        try:
            goal = _goal(title="Emergency Fund")
            with self.assertLogs("nomorals.goals.providers",
                                 level="WARNING") as caught:
                self.assertIsNone(providers.finance_progress(None, goal))
                self.assertIsNone(providers.trading_progress(
                    None, _goal(title="Desk", goal_class="trading",
                                tags=["pnl-target:10"])))
            joined = "\n".join(caught.output)
            self.assertIn("goal_test", joined)
            self.assertIn("home exploded", joined)
        finally:
            providers._finance_dir = real

    def test_corrupt_savings_store_falls_back_without_raising(self) -> None:
        (self.fin_dir / "goals.json").write_text(
            "{not json at all", encoding="utf-8")
        reading = providers.finance_progress(
            None, _goal(title="Emergency Fund"), now=time.time())
        self.assertIsNone(reading)

    def test_corrupt_ledger_lines_are_skipped_not_fatal(self) -> None:
        saved = self.seed_savings()
        path = self.fin_dir / "ledger.jsonl"
        path.write_text("{garbage\n", encoding="utf-8")
        self.contribute(K150K, ts=saved.created_at + 60)
        reading = providers.finance_progress(
            None, _goal(title="Emergency Fund"), now=saved.created_at + 3600)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.percent, 30.0)


class WiringTests(unittest.TestCase):

    def test_registry_holds_providers_after_package_import(self) -> None:
        self.assertIs(registered_provider("finance"),
                      providers.finance_progress)
        self.assertIs(registered_provider("trading"),
                      providers.trading_progress)

    def test_register_providers_is_idempotent(self) -> None:
        providers.register_providers()
        providers.register_providers()
        self.assertIs(registered_provider("finance"),
                      providers.finance_progress)
        self.assertIs(registered_provider("trading"),
                      providers.trading_progress)

    def test_fresh_interpreter_wiring(self) -> None:
        """A bare `import nomorals.goals` — how the runtime/CLI start —
        leaves both providers registered. No test-only wiring."""
        code = (
            "import nomorals.goals\n"
            "from nomorals.goals.classes import registered_provider\n"
            "from nomorals.goals import providers\n"
            "assert registered_provider('finance') is providers.finance_progress\n"
            "assert registered_provider('trading') is providers.trading_progress\n"
            "print('wired')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("wired", proc.stdout)


class SurfaceFlipTests(_ProviderCase):
    """The P2 surfaces flip purely by registration (P3's whole point)."""

    def setUp(self) -> None:
        super().setUp()
        self.tracker = GoalTracker(Database(":memory:"))

    def create(self, title: str, **kwargs: Any):
        return surface.run_sync(
            self.tracker.create(UID, title, **kwargs))

    def test_finance_board_and_progress_render_measured(self) -> None:
        # Seeded a day old so the contribution sits in the board's past
        # (the store scopes contributions to [goal creation, now]).
        from nomorals.finance.goals import Goal as SavingsGoal
        saved = SavingsGoal(id="goal_seed0001", name="Emergency Fund",
                            target_kobo=K500K,
                            created_at=time.time() - 86400)
        self.savings_store().add(saved)
        self.contribute(K150K, ts=saved.created_at + 60)
        self.create("Emergency fund", goal_class="finance")
        self.create("Unlinked money goal", goal_class="finance")
        board = surface.render_class_board(self.tracker, UID,
                                           only_class="finance")
        self.assertIn("30% (measured · savings ledger)", board)
        self.assertIn("(self-reported)", board)  # the unlinked goal
        text = surface.render_goal_progress(self.tracker, UID,
                                            "Emergency fund")
        self.assertIn("30% (measured · savings ledger)", text)
        self.assertIn("of", text)  # the funded headline line renders

    def test_trading_progress_renders_measured(self) -> None:
        goal = self.create("Grow the desk", goal_class="trading",
                           tags=["pnl-target:1000"])
        records = [
            {"ts": goal.created_at + 10, "event": "paper_close",
             "pnl": 250.0},
            {"ts": goal.created_at + 20, "event": "paper_close",
             "pnl": 250.0},
        ]
        (self.fin_dir / "trading_journal.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8")
        text = surface.render_goal_progress(self.tracker, UID,
                                            "Grow the desk")
        self.assertIn("50% (measured · trading journal)", text)

    def test_negative_realized_pnl_clamps_but_stays_measured(self) -> None:
        goal = self.create("Grow the desk", goal_class="trading",
                           tags=["pnl-target:1000"])
        (self.fin_dir / "trading_journal.jsonl").write_text(
            json.dumps({"ts": goal.created_at + 10,
                        "event": "live_close", "pnl": -100.0}) + "\n",
            encoding="utf-8")
        reading = self.tracker.class_progress(goal)
        self.assertEqual(reading.basis, "measured")
        self.assertEqual(reading.percent, 0.0)  # wrapper clamps
        self.assertIn("-100.00", reading.headline)  # truth in headline


if __name__ == "__main__":
    unittest.main()
