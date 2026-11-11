"""Navigator A5 — recipe auto-compile: compile, store, replay, gate.

A compiled recipe is a deterministic replay of a run that already
worked. The pins:
  * compile parameterises what the run typed (slots, never literals),
    marks consequential steps with their approval outcomes, and
    drops steps the run did not verify;
  * the durable store saves/lists/gets/deletes and never matches a
    stale recipe;
  * replay on a matching fake page executes the steps with slot
    values substituted; a mismatched page fails LOUDLY naming the
    mismatch and marks the recipe stale; a missing slot fails
    loudly WITHOUT marking the recipe stale (the caller failed,
    not the recipe);
  * a consequential step inside a recipe still goes through the
    approver — denied means the step never executes;
  * a completed job auto-compiles its trace; a blocked one does not;
  * the recipe verb is owner-only.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

from nomorals.agents.partner.runtime_reach import RuntimeReachMixin
from nomorals.browser.agent import Page
from nomorals.browser.approvals import ActingApprover
from nomorals.browser.jobs import BrowserJobManager
from nomorals.browser.recipe_replay import RecipeReplayError, replay_recipe
from nomorals.browser.recipes import (
    DurableRecipeStore,
    compile_recipe,
    recipe_name_for,
)
from nomorals.storage.db import Database

URL = "https://s.example/login"
HOME = "https://s.example/home"
GOAL = "log in and claim"


def _trace() -> list[dict[str, Any]]:
    return [
        {"step": 1, "before_url": URL, "after_url": URL,
         "action": {"kind": "fill", "target": "ref:q", "label": "Search",
                    "payload": {"value": "red shoes"}},
         "result": {"ok": True, "verified": True}, "verified": True},
        {"step": 2, "before_url": URL, "after_url": HOME,
         "action": {"kind": "click", "target": "ref:go", "label": "Go",
                    "payload": {}},
         "result": {"ok": True, "verified": True}, "verified": True},
        {"step": 3, "before_url": HOME, "after_url": HOME,
         "action": {"kind": "submit", "target": "ref:buy",
                    "label": "Buy now", "payload": {}},
         "result": {"ok": True, "verified": True}, "verified": True},
    ]


def _approvals() -> list[dict[str, Any]]:
    return [{"kind": "submit", "url": HOME, "verdict": "acted",
             "answered_by": "devon"}]


def _compiled():
    return compile_recipe(name=recipe_name_for(URL, GOAL), goal=GOAL,
                          trace=_trace(), start_url=URL,
                          approvals=_approvals())


class FakeDriver:
    """A BrowserDriver over canned pages."""

    def __init__(self, pages: dict[str, Page]) -> None:
        self.pages = pages
        self.acts: list[Any] = []

    def fetch(self, url: str) -> Page:
        return self.pages[url]

    def act(self, action: Any, page: Page) -> Page:
        self.acts.append(action)
        if action.kind == "goto":
            return self.fetch(action.target)
        if action.kind == "click" and action.label == "Go":
            return self.pages[HOME]
        return Page(url=page.url, html=page.html, title=page.title,
                    elements=list(page.elements))


def _pages() -> dict[str, Page]:
    return {
        URL: Page(url=URL, html="<html>login</html>", title="Login",
                  elements=[{"id": "ref:q", "label": "Search"},
                            {"id": "ref:go", "label": "Go"}]),
        HOME: Page(url=HOME, html="<html>home</html>", title="Home",
                   elements=[{"id": "ref:buy", "label": "Buy now"}]),
    }


class DenyingApprover:
    """An approver that refuses consequential steps and records it."""

    def __init__(self) -> None:
        self.outcomes: list[dict[str, Any]] = []
        self.asked: list[str] = []

    def __call__(self, decision: Any, obs: Any) -> bool:
        from nomorals.browser.agent import CONSEQUENTIAL_KINDS

        if decision.action.kind not in CONSEQUENTIAL_KINDS:
            return True
        self.asked.append(decision.action.kind)
        self.outcomes.append({"kind": decision.action.kind,
                              "label": decision.action.label,
                              "url": obs.page.url, "verdict": "denied",
                              "answered_by": "owner"})
        return False


class TestCompile(unittest.TestCase):
    def test_parameterizes_typed_values_and_marks_consequential(self):
        recipe = _compiled()
        self.assertEqual(recipe.steps[0].kind, "goto")
        self.assertEqual(recipe.steps[0].target, URL)
        fill = recipe.steps[1]
        self.assertEqual(fill.kind, "fill")
        self.assertEqual(fill.payload["value"], "{{search}}")
        self.assertEqual(fill.slot, "search")
        self.assertEqual(recipe.slots,
                         {"search": {"label": "Search", "kind": "fill"}})
        # The literal the run typed is nowhere in the recipe.
        self.assertNotIn("red shoes", json.dumps(recipe.to_dict()))
        submit = recipe.steps[3]
        self.assertTrue(submit.consequential)
        self.assertEqual(submit.approval.get("verdict"), "acted")
        self.assertEqual(recipe.origin, "s.example")
        self.assertIn("Buy now", recipe.element_labels)
        self.assertEqual(recipe.unmatched_approvals, [])

    def test_skips_failed_steps_and_rejects_empty_proof(self):
        trace = _trace()
        trace.insert(1, {"step": 9, "before_url": URL, "after_url": URL,
                         "action": {"kind": "click", "target": "ref:bad",
                                    "label": "Bad", "payload": {}},
                         "result": {"ok": False, "error": "no such ref"},
                         "verified": False})
        recipe = compile_recipe(name="x", goal=GOAL, trace=trace,
                                start_url=URL)
        kinds = [s.kind for s in recipe.steps]
        self.assertNotIn("ref:bad", [s.target for s in recipe.steps])
        self.assertEqual(kinds, ["goto", "fill", "click", "submit"])
        failed_only = [dict(trace[3], result={"ok": False},
                            verified=False)]
        with self.assertRaises(ValueError):
            compile_recipe(name="x", goal=GOAL, trace=failed_only,
                           start_url=URL)
        with self.assertRaises(ValueError):
            compile_recipe(name="x", goal=GOAL, trace=[])

    def test_unmatched_approvals_survive_on_the_recipe(self):
        recipe = compile_recipe(
            name="x", goal=GOAL, trace=_trace(), start_url=URL,
            approvals=[{"kind": "pay", "url": HOME, "verdict": "acted"}])
        self.assertEqual(len(recipe.unmatched_approvals), 1)
        self.assertEqual(recipe.unmatched_approvals[0]["kind"], "pay")


class TestDurableStore(unittest.TestCase):
    def setUp(self):
        self.store = DurableRecipeStore(Database(":memory:"))

    def test_round_trip_list_get_delete(self):
        recipe = _compiled()
        self.store.save(recipe)
        got = self.store.get(recipe.name)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.version, "1.0.0")
        self.assertEqual(got.slots, recipe.slots)
        self.assertEqual([r.name for r in self.store.list_all()],
                         [recipe.name])
        self.assertTrue(self.store.delete(recipe.name))
        self.assertIsNone(self.store.get(recipe.name))
        self.assertFalse(self.store.delete(recipe.name))

    def test_recompile_bumps_version_and_clears_stale(self):
        recipe = _compiled()
        self.store.save(recipe)
        self.store.mark_stale(recipe.name, "page moved")
        again = _compiled()
        self.store.save(again)
        got = self.store.get(recipe.name)
        assert got is not None
        self.assertEqual(got.version, "1.0.1")
        self.assertFalse(got.stale)
        self.assertEqual(got.consecutive_successes, 0)

    def test_find_for_needs_site_and_goal_and_freshness(self):
        recipe = _compiled()
        self.store.save(recipe)
        found = self.store.find_for(url=URL, labels=["Search"], goal=GOAL)
        self.assertIsNotNone(found)
        self.assertIsNone(self.store.find_for(
            url=URL, labels=["Search"], goal="something else"))
        self.assertIsNone(self.store.find_for(
            url="https://other.example/", labels=["Search"], goal=GOAL))
        self.store.mark_stale(recipe.name, "page moved")
        self.assertIsNone(self.store.find_for(
            url=URL, labels=["Search"], goal=GOAL))

    def test_mark_stale_and_trust(self):
        recipe = _compiled()
        self.store.save(recipe)
        self.assertTrue(self.store.mark_stale(recipe.name, "moved"))
        got = self.store.get(recipe.name)
        assert got is not None
        self.assertTrue(got.stale)
        self.assertEqual(got.last_error, "moved")
        self.assertEqual(got.failures, 1)
        self.assertFalse(got.trusted)
        self.store.note_replay_success(recipe.name)
        got = self.store.get(recipe.name)
        assert got is not None
        self.assertEqual(got.successes, 1)
        self.assertEqual(got.consecutive_successes, 1)
        self.assertFalse(got.trusted)  # one success is not earned trust
        self.store.note_replay_success(recipe.name)
        got = self.store.get(recipe.name)
        assert got is not None
        # Still stale: trust alone never un-stales a failed recipe.
        self.assertFalse(got.trusted)


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.store = DurableRecipeStore(Database(":memory:"))
        self.recipe = _compiled()
        self.store.save(self.recipe)

    def test_replay_executes_matching_recipe_with_slots(self):
        driver = FakeDriver(_pages())
        approver = ActingApprover()
        out = replay_recipe(self.recipe, driver=driver, approve=approver,
                            slots={"search": "blue hat"}, goal=GOAL,
                            store=self.store)
        self.assertTrue(out["replayed"])
        self.assertEqual(out["steps"], 4)
        self.assertEqual(out["final_url"], HOME)
        self.assertEqual(out["stopped_reason"], "")
        fills = [a for a in driver.acts if a.kind == "fill"]
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].payload["value"], "blue hat")
        self.assertEqual(len(out["trace"]), 4)
        self.assertEqual(out["consequential_approvals"][0]["verdict"],
                         "acted")
        got = self.store.get(self.recipe.name)
        assert got is not None
        self.assertEqual(got.consecutive_successes, 1)

    def test_missing_slot_fails_loudly_without_staling(self):
        driver = FakeDriver(_pages())
        with self.assertRaises(RecipeReplayError) as ctx:
            replay_recipe(self.recipe, driver=driver,
                          approve=ActingApprover(), slots={}, goal=GOAL,
                          store=self.store)
        self.assertIn("search", str(ctx.exception))
        self.assertFalse(ctx.exception.stale)
        got = self.store.get(self.recipe.name)
        assert got is not None
        self.assertFalse(got.stale)
        # Nothing was typed: the fill never executed.
        self.assertEqual([a.kind for a in driver.acts], ["goto"])

    def test_url_drift_fails_preflight_and_marks_stale(self):
        moved = Page(url="https://other.example/", html="", title="",
                     elements=[])
        driver = FakeDriver({URL: moved, HOME: _pages()[HOME]})
        with self.assertRaises(RecipeReplayError) as ctx:
            replay_recipe(self.recipe, driver=driver,
                          approve=ActingApprover(),
                          slots={"search": "x"}, goal=GOAL,
                          store=self.store)
        self.assertIn("no longer matches", str(ctx.exception))
        self.assertIn("0.0", str(ctx.exception))
        self.assertTrue(ctx.exception.stale)
        got = self.store.get(self.recipe.name)
        assert got is not None
        self.assertTrue(got.stale)
        self.assertIn("no longer matches", got.last_error)
        self.assertEqual(driver.acts, [])

    def test_element_drift_fails_at_the_step_and_marks_stale(self):
        changed = {URL: Page(url=URL, html="", title="",
                             elements=[{"id": "ref:x", "label": "Other"}]),
                   HOME: _pages()[HOME]}
        driver = FakeDriver(changed)
        with self.assertRaises(RecipeReplayError) as ctx:
            replay_recipe(self.recipe, driver=driver,
                          approve=ActingApprover(),
                          slots={"search": "x"}, goal=GOAL,
                          store=self.store)
        self.assertIn("step 2", str(ctx.exception))
        self.assertIn("Search", str(ctx.exception))
        self.assertIn("Other", str(ctx.exception))
        self.assertEqual(ctx.exception.step_index, 1)
        got = self.store.get(self.recipe.name)
        assert got is not None
        self.assertTrue(got.stale)

    def test_consequential_step_still_demands_approval(self):
        driver = FakeDriver(_pages())
        approver = DenyingApprover()
        out = replay_recipe(self.recipe, driver=driver, approve=approver,
                            slots={"search": "x"}, goal=GOAL,
                            store=self.store)
        self.assertEqual(out["stopped_reason"],
                         "approval denied / no approver")
        # The submit never executed; the benign steps before it did.
        self.assertEqual([a.kind for a in driver.acts],
                         ["goto", "fill", "click"])
        self.assertEqual(approver.asked, ["submit"])
        self.assertEqual(out["consequential_approvals"][0]["verdict"],
                         "denied")
        # A denial is a stop, not a stale recipe.
        got = self.store.get(self.recipe.name)
        assert got is not None
        self.assertFalse(got.stale)


class TestAutoCompile(unittest.TestCase):
    def setUp(self):
        # A FILE database, not :memory:: the manager compiles on the
        # job's worker thread, and an in-memory sqlite database is
        # per-connection — the worker thread would see an empty
        # database (and the compile would fail-open into the ledger).
        # The production home database is file-backed, same as this.
        self._tmp = tempfile.TemporaryDirectory()
        self.store = DurableRecipeStore(
            Database(self._tmp.name + "/recipes.sqlite3"))
        self.mgr = BrowserJobManager(recipe_store=self.store)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, stopped_reason: str):
        def runner(job: Any) -> dict[str, Any]:
            return {"goal": GOAL, "steps": 3,
                    "stopped_reason": stopped_reason,
                    "final_url": HOME, "trace": _trace(),
                    "consequential_approvals": _approvals()}

        job, _dup = self.mgr.start(URL, GOAL, runner)
        self.assertTrue(job.wait(timeout=5))
        return job

    def test_done_job_compiles_its_trace(self):
        job = self._run("")
        name = recipe_name_for(URL, GOAL)
        self.assertEqual(job.recipe_name, name)
        got = self.store.get(name)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.steps[0].kind, "goto")
        self.assertEqual(got.steps[3].approval.get("verdict"), "acted")
        self.assertIn("Compiled recipe", job.status_text())

    def test_blocked_job_compiles_nothing(self):
        job = self._run("blocked: wall (no wall handler)")
        self.assertEqual(job.recipe_name, "")
        self.assertEqual(self.store.list_all(), [])

    def test_denied_job_compiles_nothing(self):
        job = self._run("approval denied / no approver")
        self.assertEqual(job.recipe_name, "")
        self.assertEqual(self.store.list_all(), [])


class TestRecipeVerb(unittest.TestCase):
    def _mixin(self, *, operator: bool):
        mixin = RuntimeReachMixin.__new__(RuntimeReachMixin)
        mixin.context = SimpleNamespace(db=Database(":memory:"))
        mixin._is_operator = lambda _msg: operator  # noqa: E731
        return mixin

    def test_refuses_non_owner(self):
        mixin = self._mixin(operator=False)
        reply = mixin._control_browser_recipe(
            "list", message=SimpleNamespace())
        self.assertIn("owner-only", reply)

    def test_list_show_delete_for_owner(self):
        mixin = self._mixin(operator=True)
        self.assertIn("no browser recipes yet",
                      mixin._control_browser_recipe("list",
                                                    message=SimpleNamespace()))
        store = mixin._browser_recipe_store()
        assert store is not None
        store.save(_compiled())
        listing = mixin._control_browser_recipe(
            "list", message=SimpleNamespace())
        self.assertIn(recipe_name_for(URL, GOAL), listing)
        shown = mixin._control_browser_recipe(
            "show " + recipe_name_for(URL, GOAL),
            message=SimpleNamespace())
        self.assertIn("consequential", shown)
        self.assertIn("parameters: search", shown)
        deleted = mixin._control_browser_recipe(
            "delete " + recipe_name_for(URL, GOAL),
            message=SimpleNamespace())
        self.assertIn("deleted", deleted)
        self.assertIsNone(store.get(recipe_name_for(URL, GOAL)))

    def test_replay_verb_parses_params_and_routes(self):
        mixin = self._mixin(operator=True)
        store = mixin._browser_recipe_store()
        assert store is not None
        store.save(_compiled())
        seen: dict[str, Any] = {}

        def fake_start(url: str, goal: str, **kwargs: Any):
            seen.update({"url": url, "goal": goal, **kwargs})
            return SimpleNamespace(job_id="br-fake"), False

        mixin.start_browser_run = fake_start  # type: ignore[method-assign]
        reply = mixin._control_browser_recipe(
            "replay " + recipe_name_for(URL, GOAL) + " search=shoes",
            message=SimpleNamespace())
        self.assertIn("replaying recipe", reply)
        self.assertEqual(seen["recipe"], recipe_name_for(URL, GOAL))
        self.assertEqual(seen["recipe_params"], {"search": "shoes"})
        self.assertEqual(seen["url"], URL)
        bad = mixin._control_browser_recipe(
            "replay " + recipe_name_for(URL, GOAL) + " search",
            message=SimpleNamespace())
        self.assertIn("key=value", bad)


if __name__ == "__main__":
    unittest.main()
