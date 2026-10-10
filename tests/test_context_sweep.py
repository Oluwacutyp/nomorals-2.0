"""Sweep tests for the context module upgrade.

Covers: budget profiles, utilization reports, salient extraction, section
metadata (volatile/fingerprint/tagged/density), pluggable token counters,
assembly ordering modes (priority/cache/rot), cache plans + prefix audits,
incremental history compaction, rot analysis, dashboards, snapshot diffs,
pruning, and the upgraded waste detectors (near-duplicate, low-entropy,
recoverability).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nomorals.context import (
    BUDGET_PROFILES,
    ORDERING_MODES,
    BuiltContext,
    ContextBudget,
    ContextEngine,
    Section,
    SnapshotStore,
    WasteDetector,
    audit_prefix_stability,
    cache_plan,
    classify_recoverability,
    extractive_summary,
    format_snapshot_diff,
    salient_extract,
    set_token_counter,
    simhash,
    snapshot_diff,
    token_count,
)


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def _section(name: str, words: int, **kwargs) -> Section:
    kwargs.setdefault("priority", 50.0)
    return Section(name=name, content=_words(words), **kwargs)


# ── budget profiles ──────────────────────────────────────────────────────
class BudgetProfileTest(unittest.TestCase):
    def test_known_profiles(self) -> None:
        for name in ("default", "chat", "coding", "research", "minimal"):
            with self.subTest(name=name):
                budget = ContextBudget.profile(name, total=4000)
                self.assertEqual(budget.total, 4000)
                self.assertGreater(budget.allocation_for("mission"), 0)

    def test_unknown_profile_raises(self) -> None:
        with self.assertRaises(ValueError):
            ContextBudget.profile("nope")

    def test_coding_profile_favors_artifacts_over_chat(self) -> None:
        coding = ContextBudget.profile("coding", total=10000)
        chat = ContextBudget.profile("chat", total=10000)
        self.assertGreater(
            coding.allocation_for("artifacts"), chat.allocation_for("artifacts"))
        self.assertGreater(
            chat.allocation_for("history"), coding.allocation_for("history"))

    def test_profile_registry_exposed(self) -> None:
        self.assertIn("coding", BUDGET_PROFILES)


class UtilizationReportTest(unittest.TestCase):
    def test_report_rows_and_table(self) -> None:
        budget = ContextBudget.profile("coding", total=10000)
        sections = [
            _section("system", 300, load_bearing=True),
            _section("history", 2000),  # over its ~1800 allocation, not dropped
        ]
        budget.fit(sections)
        # Report over the full section list: dropped sections stay visible.
        rows = budget.utilization_report(sections)
        by_name = {r["name"]: r for r in rows}
        self.assertIn("system", by_name)
        self.assertIn("history", by_name)
        self.assertGreater(by_name["history"]["ratio"], 1.0)
        self.assertEqual(by_name["history"]["status"], "over")
        self.assertEqual(by_name["system"]["status"], "under")
        table = budget.render_table(sections, style="fancy")
        self.assertIn("system", table)
        self.assertIn("history", table)
        self.assertIn("TOTAL", table)

    def test_dropped_sections_stay_visible(self) -> None:
        budget = ContextBudget(total=500)
        sections = [
            _section("system", 100, load_bearing=True),
            _section("history", 2000),
        ]
        budget.fit(sections)
        rows = budget.utilization_report(sections)
        by_name = {r["name"]: r for r in rows}
        self.assertTrue(by_name["history"]["dropped"])
        self.assertEqual(by_name["history"]["status"], "unused")
        table = budget.render_table(sections)
        self.assertIn("dropped", table)

    def test_render_table_plain_without_sections(self) -> None:
        budget = ContextBudget.default(total=8000)
        table = budget.render_table()
        self.assertIn("8000", table)
        self.assertIn("mission", table)


# ── sections ─────────────────────────────────────────────────────────────
class SectionMetaTest(unittest.TestCase):
    def test_volatile_flag_defaults_false(self) -> None:
        self.assertFalse(Section(name="x").volatile)

    def test_fingerprint_stable(self) -> None:
        a = _section("s", 10)
        b = _section("s", 10)
        self.assertEqual(a.fingerprint(), b.fingerprint())
        b.content += " extra"
        self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_render_tagged(self) -> None:
        s = _section("tools", 5)
        out = s.render_tagged()
        self.assertIn('<section name="tools">', out)
        self.assertIn("</section>", out)

    def test_density(self) -> None:
        dense = Section(name="d", content="deploy() failed with ERR_404 at 10.0.0.1 port 8080")
        sparse = Section(name="d", content="the thing is quite nice and rather pleasant today")
        self.assertGreater(dense.density(), sparse.density())

    def test_summary_line_flags(self) -> None:
        s = _section("m", 5, load_bearing=True, volatile=True)
        line = s.summary_line()
        self.assertIn("pinned", line)
        self.assertIn("volatile", line)


class TokenCounterHookTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_token_counter(None)

    def test_custom_counter_used(self) -> None:
        set_token_counter(lambda text: 42)
        self.assertEqual(token_count("anything at all"), 42)
        self.assertEqual(_section("s", 100).tokens, 42)

    def test_broken_counter_falls_back(self) -> None:
        def _boom(text: str) -> int:
            raise RuntimeError("nope")

        set_token_counter(_boom)
        self.assertGreater(token_count("hello world"), 0)

    def test_reset_restores_estimator(self) -> None:
        set_token_counter(lambda text: 42)
        set_token_counter(None)
        self.assertNotEqual(token_count("hello world foo bar"), 42)


# ── compression ──────────────────────────────────────────────────────────
class SalientExtractTest(unittest.TestCase):
    def test_reduces_to_budget(self) -> None:
        text = "\n\n".join(
            f"Paragraph {i} discusses the deployment pipeline, artifact "
            f"artifact://report-{i}, error ERR_{1000 + i}, and host 10.0.0.{i}."
            for i in range(12)
        )
        out = salient_extract(text, 60)
        self.assertLessEqual(token_count(out), 120)  # estimator slack
        self.assertIn("omitted", out)

    def test_keeps_dense_sentences(self) -> None:
        dense = "Deploy failed: ERR_9042 on host 10.0.0.7 during migration step 12."
        filler = " ".join(["the weather is rather pleasant today indeed"] * 6)
        text = "\n\n".join([filler] * 6 + [dense] + [filler] * 6)
        out = salient_extract(text, 40)
        self.assertIn("ERR_9042", out)

    def test_deterministic(self) -> None:
        text = "\n\n".join(f"Sentence block number {i} with value VAL_{i}." for i in range(20))
        self.assertEqual(salient_extract(text, 50), salient_extract(text, 50))

    def test_short_text_passthrough(self) -> None:
        text = "Just one short paragraph here."
        self.assertEqual(salient_extract(text, 100), text)


# ── engine ordering ──────────────────────────────────────────────────────
class OrderingModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ContextEngine(budget=ContextBudget(total=20000))

    def test_modes_declared(self) -> None:
        self.assertEqual(set(ORDERING_MODES), {"priority", "cache", "rot"})

    def test_priority_is_canonical(self) -> None:
        built = self.engine.build(
            mission={"goal": "g"}, history=[("user", "hi")], ordering="priority")
        names = [s.name for s in built.sections]
        self.assertEqual(names[0], "system")
        self.assertEqual(names[-1], "history")
        self.assertEqual(built.meta["ordering"], "priority")

    def test_cache_puts_volatile_last(self) -> None:
        built = self.engine.build(
            mission={"goal": "g"}, history=[("user", "hi")], ordering="cache")
        names = [s.name for s in built.sections]
        self.assertEqual(names[-1], "history")
        self.assertLess(names.index("system"), names.index("history"))
        self.assertLess(names.index("tools") if "tools" in names else 0,
                        names.index("history"))

    def test_rot_load_bearing_first_with_echo(self) -> None:
        built = self.engine.build(
            mission={"goal": "ship it", "acceptance_criteria": ["tests pass"]},
            history=[("user", "hi")],
            ordering="rot",
        )
        names = [s.name for s in built.sections]
        self.assertLess(names.index("mission"), names.index("history"))
        self.assertIn("Key facts (echo", built.text)
        self.assertIn("tests pass", built.text.split("Key facts (echo")[1])

    def test_tagged_render(self) -> None:
        built = self.engine.build(mission={"goal": "g"}, tagged=True)
        self.assertIn('<section name="system">', built.text)
        self.assertTrue(built.meta["tagged"])

    def test_unknown_ordering_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.build(ordering="sideways")

    def test_legacy_step_prompt_unchanged(self) -> None:
        mission = {"goal": "G", "state": {"outputs": {"a": "1"}}}
        step = {"goal": "S"}
        out = self.engine.build_step_prompt(mission, step)
        self.assertIn("Mission goal: G", out)
        self.assertIn("Current step: S", out)

    def test_history_section_marked_volatile(self) -> None:
        built = self.engine.build(history=[("user", "hi")])
        history = built.section("history")
        self.assertIsNotNone(history)
        assert history is not None
        self.assertTrue(history.volatile)


# ── cache plan ───────────────────────────────────────────────────────────
class CachePlanTest(unittest.TestCase):
    def test_plan_splits_stable_and_volatile(self) -> None:
        engine = ContextEngine(budget=ContextBudget(total=20000))
        built = engine.build(
            mission={"goal": "g"}, history=[("user", "hi")], ordering="cache")
        plan = cache_plan(built.sections)
        self.assertIn("system", plan.stable_prefix)
        self.assertIn("history", plan.volatile_tail)
        self.assertNotIn("history", plan.stable_prefix)
        self.assertLessEqual(len(plan.breakpoints), 4)
        self.assertGreater(plan.cacheable_tokens, 0)
        rendered = plan.render(style="fancy")
        self.assertIn("cacheable", rendered)

    def test_audit_detects_timestamp_in_stable_prefix(self) -> None:
        sections = [
            Section(name="system", content="Policy as of 2026-10-10 08:00:00 UTC."),
            Section(name="history", content="2026-10-10 hello", volatile=True),
        ]
        warnings = audit_prefix_stability(sections)
        self.assertTrue(any(w.section == "system" and w.pattern == "timestamp"
                            for w in warnings))
        # volatile sections are never flagged
        self.assertFalse(any(w.section == "history" for w in warnings))

    def test_audit_detects_uuid_and_session_id(self) -> None:
        sections = [Section(
            name="tools",
            content="session_id: 550e8400-e29b-41d4-a716-446655440000 ready",
        )]
        warnings = audit_prefix_stability(sections)
        patterns = {w.pattern for w in warnings}
        self.assertIn("uuid", patterns)

    def test_engine_records_cache_plan_and_fingerprint(self) -> None:
        engine = ContextEngine(budget=ContextBudget(total=20000))
        built = engine.build(mission={"goal": "g"})
        self.assertIn("cache_plan", built.meta)
        self.assertIn("stable_fingerprint", built.meta)
        self.assertEqual(len(built.meta["stable_fingerprint"]), 16)

    def test_fingerprint_changes_on_prefix_drift(self) -> None:
        engine = ContextEngine(budget=ContextBudget(total=20000))
        a = engine.build(system="v1")
        b = engine.build(system="v2")
        self.assertNotEqual(a.meta["stable_fingerprint"],
                            b.meta["stable_fingerprint"])


# ── history compaction ───────────────────────────────────────────────────
class CompactHistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ContextEngine(budget=ContextBudget(total=20000))

    def _history(self, n: int) -> list[tuple[str, str]]:
        return [("user" if i % 2 == 0 else "assistant",
                 f"message number {i} about topic T{i}")
                for i in range(n)]

    def test_compacts_old_span_keeps_tail(self) -> None:
        compacted, info = self.engine.compact_history(
            self._history(10), keep_last_n=3)
        self.assertTrue(info["compacted"])
        self.assertEqual(info["entries_summarized"], 7)
        # boundary marker + 3 recent entries
        self.assertEqual(len(compacted), 4)
        self.assertIn("compact boundary", compacted[0][1])
        self.assertIn("message number 9", compacted[-1][1])

    def test_short_history_passes_through(self) -> None:
        compacted, info = self.engine.compact_history(
            self._history(4), keep_last_n=6)
        self.assertFalse(info["compacted"])
        self.assertEqual(len(compacted), 4)

    def test_incremental_boundary_never_resummarizes(self) -> None:
        first, info1 = self.engine.compact_history(
            self._history(10), keep_last_n=3)
        self.assertTrue(info1["compacted"])
        # Feed the compacted result back with the boundary: nothing new to do.
        second, info2 = self.engine.compact_history(
            first, keep_last_n=3, boundary=info1)
        self.assertFalse(info2["compacted"])
        self.assertEqual(len(second), len(first))

    def test_incremental_resumes_after_boundary(self) -> None:
        first, info1 = self.engine.compact_history(
            self._history(10), keep_last_n=3)
        # Simulate 5 more turns arriving after the boundary.
        extended = list(first) + self._history(5)
        second, info2 = self.engine.compact_history(
            extended, keep_last_n=3, boundary=info1)
        self.assertTrue(info2["compacted"])
        # The old summary marker survives; only the new span was summarized.
        markers = [e for e in second if "compact boundary" in e[1]]
        self.assertEqual(len(markers), 2)

    def test_focus_recorded(self) -> None:
        _, info = self.engine.compact_history(
            self._history(10), keep_last_n=3, focus="database decisions")
        self.assertEqual(info["focus"], "database decisions")

    def test_compacted_history_feeds_build(self) -> None:
        compacted, _ = self.engine.compact_history(
            self._history(12), keep_last_n=3)
        built = self.engine.build(history=compacted)
        self.assertIsNotNone(built.section("history"))
        self.assertIn("compact boundary", built.section("history").content)


# ── rot analysis + dashboard ─────────────────────────────────────────────
class RotReportTest(unittest.TestCase):
    def test_buried_load_bearing_detected(self) -> None:
        engine = ContextEngine()
        sections = [
            _section("history", 1500),
            _section("mission", 80, load_bearing=True, priority=90.0),
            _section("tools", 1500),
        ]
        report = engine.rot_report_for(sections)
        self.assertIn("mission", report["positions"])
        self.assertIn("mission", report["buried"])

    def test_edge_load_bearing_not_buried(self) -> None:
        engine = ContextEngine()
        sections = [
            _section("mission", 80, load_bearing=True, priority=90.0),
            _section("history", 3000),
        ]
        report = engine.rot_report_for(sections)
        self.assertNotIn("mission", report["buried"])

    def test_build_records_rot_meta(self) -> None:
        engine = ContextEngine(budget=ContextBudget(total=20000))
        built = engine.build(mission={"goal": "g"},
                             history=[("user", "hi")])
        self.assertIn("rot", built.meta)
        self.assertIn("positions", built.meta["rot"])


class DashboardTest(unittest.TestCase):
    def test_render_dashboard(self) -> None:
        engine = ContextEngine(budget=ContextBudget(total=20000))
        built = engine.build(
            mission={"goal": "g", "acceptance_criteria": ["x"]},
            history=[("user", "hi")],
        )
        for style in ("plain", "fancy"):
            out = built.render_dashboard(style=style)
            self.assertIn("Context dashboard", out)
            self.assertIn("system", out)
            self.assertIn("mission", out)
            self.assertIn("tokens", out)

    def test_preview_shows_table_and_cache_plan(self) -> None:
        engine = ContextEngine(budget=ContextBudget(total=20000))
        out = engine.preview(mission={"goal": "g"},
                             history=[("user", "hi")])
        self.assertIn("Context budget", out)
        self.assertIn("Cache plan", out)


# ── snapshots ────────────────────────────────────────────────────────────
class SnapshotDiffTest(unittest.TestCase):
    def _built(self, **overrides) -> BuiltContext:
        sections = [
            Section(name="system", content=overrides.get("system", "sys"),
                    load_bearing=True),
            Section(name="mission", content=overrides.get("mission", "m1"),
                    load_bearing=True),
        ]
        return BuiltContext(text="t", sections=sections, total_tokens=100)

    def test_diff_detects_change_and_add(self) -> None:
        older = self._built(mission="m1")
        newer = self._built(mission="m1 extended with more words here")
        newer.sections.append(Section(name="history", content=_words(50)))
        diff = snapshot_diff(older, newer)
        by_name = {r["name"]: r for r in diff["sections"]}
        self.assertEqual(by_name["mission"]["status"], "changed")
        self.assertGreater(by_name["mission"]["token_delta"], 0)
        self.assertEqual(by_name["history"]["status"], "added")
        self.assertEqual(by_name["system"]["status"], "same")

    def test_diff_detects_removal(self) -> None:
        older = self._built()
        older.sections.append(Section(name="tools", content="t"))
        newer = self._built()
        diff = snapshot_diff(older, newer)
        by_name = {r["name"]: r for r in diff["sections"]}
        self.assertEqual(by_name["tools"]["status"], "removed")

    def test_format_diff(self) -> None:
        older = self._built()
        newer = self._built(mission="m1 extended with more words here yes")
        out = format_snapshot_diff(snapshot_diff(older, newer), style="fancy")
        self.assertIn("Snapshot diff", out)
        self.assertIn("mission", out)

    def test_format_diff_no_changes(self) -> None:
        out = format_snapshot_diff(snapshot_diff(self._built(), self._built()))
        self.assertIn("no section changes", out)


class SnapshotStoreExtrasTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(Path(self.tmp.name))
        self.engine = ContextEngine(budget=ContextBudget(total=20000))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _save(self, name: str) -> None:
        self.store.save(name, self.engine.build(mission={"goal": name}))

    def test_describe(self) -> None:
        self._save("alpha")
        info = self.store.describe("alpha")
        self.assertEqual(info["name"], "alpha")
        self.assertGreater(info["total_tokens"], 0)
        self.assertIn("mission", info["sections"])

    def test_describe_unknown_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.store.describe("ghost")

    def test_prune_keeps_newest(self) -> None:
        import time as _time

        for i in range(4):
            self._save(f"snap{i}")
            _time.sleep(0.02)
        deleted = self.store.prune(keep_n=2)
        self.assertEqual(len(deleted), 2)
        remaining = self.store.list()
        self.assertEqual(len(remaining), 2)
        self.assertIn("snap3", remaining)
        self.assertIn("snap2", remaining)

    def test_roundtrip_still_works(self) -> None:
        built = self.engine.build(mission={"goal": "g"})
        self.store.save("rt", built)
        loaded = self.store.load("rt")
        self.assertEqual(loaded.text, built.text)


# ── waste upgrades ───────────────────────────────────────────────────────
class WasteUpgradeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = WasteDetector()

    def _built(self, sections: list[Section]) -> BuiltContext:
        return BuiltContext(text="t", sections=sections)

    def test_near_duplicate_detection(self) -> None:
        base = (
            "The deployment pipeline failed at the migration step with error "
            "ERR_9042 on host alpha. The on-call engineer restarted the worker "
            "nodes and re-ran the migration. All integration tests passed after "
            "the retry and the release was promoted to staging successfully. "
            "A follow-up review was scheduled for the next morning to confirm "
            "that the database replicas had caught up and that no customer "
            "traffic was affected during the incident window at all."
        )
        variant = base.replace("ERR_9042", "ERR_9043").replace("alpha", "beta")
        section = Section(name="history", content=f"{base}\n\n{variant}")
        report = self.detector.analyze(self._built([section]))
        kinds = [f.kind for f in report.findings]
        self.assertIn("near_duplicate", kinds)

    def test_exact_duplicate_still_found(self) -> None:
        line = ("word " * 50).strip()
        section = Section(name="history", content=f"{line}\n{line}")
        report = self.detector.analyze(self._built([section]))
        self.assertIn("duplicate_output",
                      [f.kind for f in report.findings])

    def test_low_entropy_boilerplate(self) -> None:
        section = Section(
            name="tools",
            content=("listing tool definitions for agent use\n" * 120),
        )
        report = self.detector.analyze(self._built([section]))
        self.assertIn("low_entropy", [f.kind for f in report.findings])

    def test_load_bearing_never_flagged(self) -> None:
        line = ("word " * 50).strip()
        section = Section(name="mission", content=f"{line}\n{line}",
                          load_bearing=True)
        report = self.detector.analyze(self._built([section]))
        self.assertEqual(report.findings, [])

    def test_recoverability_labels(self) -> None:
        self.assertEqual(
            classify_recoverability(Section(name="artifacts")), "reconstructible")
        self.assertEqual(
            classify_recoverability(Section(name="history")), "reproducible")
        self.assertEqual(
            classify_recoverability(
                Section(name="mission", load_bearing=True)), "irreplaceable")

    def test_findings_carry_recoverability(self) -> None:
        line = ("word " * 50).strip()
        section = Section(name="history", content=f"{line}\n{line}")
        report = self.detector.analyze(self._built([section]))
        for finding in report.findings:
            self.assertIn(finding.recoverability,
                          {"reconstructible", "reproducible", "irreplaceable"})

    def test_render_styles(self) -> None:
        line = ("word " * 50).strip()
        section = Section(name="history", content=f"{line}\n{line}")
        report = self.detector.analyze(self._built([section]))
        for style in ("plain", "fancy"):
            out = report.render(style=style)
            self.assertIn("reclaimable", out)

    def test_empty_report_render(self) -> None:
        out = WasteDetector().analyze(self._built([])).render()
        self.assertIn("No context waste", out)


class SimhashTest(unittest.TestCase):
    def test_similar_texts_close(self) -> None:
        a = "the quick brown fox jumps over the lazy dog near the river bank"
        b = "the quick brown fox jumps over the lazy dog near the river shore"
        c = "quantum entanglement violates local realism in bell experiments"
        ha, hb, hc = simhash(a), simhash(b), simhash(c)

        def _hamming(x: int, y: int) -> int:
            return bin(x ^ y).count("1")

        self.assertLess(_hamming(ha, hb), _hamming(ha, hc))


# ── back-compat: old behavior preserved ──────────────────────────────────
class BackCompatTest(unittest.TestCase):
    def test_default_build_order_unchanged(self) -> None:
        engine = ContextEngine(budget=ContextBudget(total=20000))
        built = engine.build(
            mission={"goal": "g"},
            artifacts=[{"id": "a1"}],
            tools=[{"name": "t"}],
            project={"name": "p"},
            user_profile={"name": "u"},
            history=[("user", "hi")],
        )
        names = [s.name for s in built.sections]
        self.assertEqual(
            names,
            ["system", "mission", "artifacts", "tools", "project",
             "user_profile", "history"],
        )

    def test_extractive_summary_unchanged(self) -> None:
        paras = [f"para {i} " + ("word " * 20) for i in range(6)]
        out = extractive_summary("\n\n".join(paras), 40)
        self.assertIn("omitted", out)

    def test_report_shape_extended_not_broken(self) -> None:
        engine = ContextEngine(budget=ContextBudget(total=20000))
        built = engine.build(mission={"goal": "g"})
        report = built.report()
        self.assertIn("total_tokens", report)
        self.assertIn("sections", report)
        self.assertIn("volatile", report["sections"][0])
        self.assertIn("fingerprint", report["sections"][0])


if __name__ == "__main__":
    unittest.main()
