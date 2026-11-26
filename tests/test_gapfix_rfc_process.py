"""RFC process (lane 128, census §3B.5): the RFC-0000 decision process
mined from storytold/craft-contrib (no licence — pattern only, no
upstream file or text vendored or copied).

The mined rules under test, each with its evidence in the store:

* Open needs title, kind, and at least one affected module — each
  missing field is refused by name.
* Comment period: 7 days for app add/remove and process changes,
  3 days for everything else; started by a maintainer, from draft.
* Acceptance needs the period over, an approval from a maintainer
  other than the author, and no unresolved objection.
* An objection needs a reason AND what would resolve it.
* The owner can accept or close early, or decide an objection — with
  a reason, recorded on the decision.
* A closed RFC never moves again; it returns only as a NEW RFC that
  supersedes it.
* An accepted RFC writes a decision-journal entry.

The tool count stays 774 — the secprobe wire-up added 25 tools; 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 806 after gapfix119 added matching. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 806 after gapfix119 added matching. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 806 after gapfix119 added matching. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 806 after gapfix119 added matching.
the RFC verbs ride the
existing ``nm decisions`` / ``/decisions`` surface.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from nomorals.decisions.journal import DecisionJournal
from nomorals.decisions.rfc import (
    PERIOD_DAYS,
    Rfc,
    RfcError,
    RfcStore,
    RfcUsageError,
    handle_rfc,
    slugify,
)

T0 = 1_800_000_000.0
DAY = 86400.0


def _store(tmp: str, **kw) -> RfcStore:
    journal = DecisionJournal(os.path.join(tmp, "decisions.json"))
    return RfcStore(os.path.join(tmp, "rfcs.json"),
                    maintainers=["owner", "bob", "carol"],
                    owner="owner", journal=journal, **kw)


def _open(store: RfcStore, **kw) -> Rfc:
    kw.setdefault("title", "Merge the caches")
    kw.setdefault("kind", "other")
    kw.setdefault("modules", ["core", "memory"])
    kw.setdefault("author", "alice")
    kw.setdefault("ts", T0)
    return store.open_rfc(**kw)


class SlugTests(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Merge the Caches!"), "merge-the-caches")
        self.assertEqual(slugify("  "), "rfc")


class OpenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _store(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_open_assigns_number_and_filename(self):
        rfc = _open(self.store)
        self.assertEqual(rfc.number, 1)
        self.assertEqual(rfc.filename, "rfcs/0001-merge-the-caches.md")
        self.assertEqual(rfc.status, "draft")

    def test_open_numbers_sequence(self):
        a = _open(self.store)
        b = _open(self.store, title="Second RFC")
        self.assertEqual((a.number, b.number), (1, 2))

    def test_open_missing_title_refused_by_name(self):
        with self.assertRaisesRegex(RfcError, "title"):
            self.store.open_rfc(title=" ", kind="other",
                                modules=["core"], author="alice")

    def test_open_bad_kind_refused_by_name(self):
        with self.assertRaisesRegex(RfcError, "kind"):
            self.store.open_rfc(title="T", kind="urgent",
                                modules=["core"], author="alice")

    def test_open_missing_modules_refused_by_name(self):
        with self.assertRaisesRegex(RfcError, "module"):
            self.store.open_rfc(title="T", kind="other",
                                modules=[" ", ""], author="alice")

    def test_open_missing_author_refused_by_name(self):
        with self.assertRaisesRegex(RfcError, "author"):
            self.store.open_rfc(title="T", kind="other",
                                modules=["core"], author=" ")

    def test_period_days_by_kind(self):
        self.assertEqual(PERIOD_DAYS, {"app_add": 7, "app_remove": 7,
                                       "process": 7, "other": 3})


class PeriodTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _store(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_start_period_sets_end_from_kind(self):
        rfc = _open(self.store, kind="process")
        out = self.store.start_period(rfc.number, by="bob", ts=T0)
        self.assertEqual(out.status, "in_comment")
        self.assertEqual(out.period_ends_at, T0 + 7 * DAY)
        self.assertEqual(out.period_started_by, "bob")

    def test_start_period_three_days_for_other(self):
        rfc = _open(self.store, kind="other")
        out = self.store.start_period(rfc.number, by="bob", ts=T0)
        self.assertEqual(out.period_ends_at, T0 + 3 * DAY)

    def test_start_period_non_maintainer_refused(self):
        rfc = _open(self.store)
        with self.assertRaisesRegex(RfcError, "not a maintainer"):
            self.store.start_period(rfc.number, by="mallory", ts=T0)

    def test_start_period_twice_refused(self):
        rfc = _open(self.store)
        self.store.start_period(rfc.number, by="bob", ts=T0)
        with self.assertRaisesRegex(RfcError, "in_comment"):
            self.store.start_period(rfc.number, by="bob", ts=T0)


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _store(self.tmp.name)
        self.rfc = _open(self.store)

    def tearDown(self):
        self.tmp.cleanup()

    def test_approve_in_draft_refused(self):
        with self.assertRaisesRegex(RfcError, "comment period is running"):
            self.store.approve(self.rfc.number, by="bob", ts=T0)

    def test_approve_by_author_refused(self):
        self.store.start_period(self.rfc.number, by="bob", ts=T0)
        with self.assertRaisesRegex(RfcError, "cannot approve their own"):
            self.store.approve(self.rfc.number, by="alice", ts=T0)

    def test_approve_by_non_maintainer_refused(self):
        self.store.start_period(self.rfc.number, by="bob", ts=T0)
        with self.assertRaisesRegex(RfcError, "not a maintainer"):
            self.store.approve(self.rfc.number, by="mallory", ts=T0)

    def test_approve_records_once(self):
        self.store.start_period(self.rfc.number, by="bob", ts=T0)
        self.store.approve(self.rfc.number, by="bob", ts=T0)
        self.store.approve(self.rfc.number, by="bob", ts=T0 + 1)
        self.assertEqual(len(self.rfc.approvals), 1)
        self.assertEqual(len(self.rfc.non_author_approvals), 1)


class ObjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _store(self.tmp.name)
        self.rfc = _open(self.store)
        self.store.start_period(self.rfc.number, by="bob", ts=T0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_object_needs_reason(self):
        with self.assertRaisesRegex(RfcError, "reason"):
            self.store.object(self.rfc.number, by="bob", reason=" ",
                              resolves="fix it")

    def test_object_needs_what_resolves_it(self):
        with self.assertRaisesRegex(RfcError, "what would resolve"):
            self.store.object(self.rfc.number, by="bob",
                              reason="breaks the cache", resolves=" ")

    def test_object_in_draft_refused(self):
        draft = _open(self.store, title="Draft only")
        with self.assertRaisesRegex(RfcError, "comment period is running"):
            self.store.object(draft.number, by="bob", reason="no",
                              resolves="yes")

    def test_resolve_needs_note(self):
        ob = self.store.object(self.rfc.number, by="carol",
                               reason="breaks the cache",
                               resolves="add a migration")
        with self.assertRaisesRegex(RfcError, "note"):
            self.store.resolve_objection(self.rfc.number,
                                         objection_id=ob.id, by="carol",
                                         note=" ")

    def test_resolve_by_non_maintainer_refused(self):
        ob = self.store.object(self.rfc.number, by="carol",
                               reason="breaks the cache",
                               resolves="add a migration")
        with self.assertRaisesRegex(RfcError, "cannot resolve"):
            self.store.resolve_objection(self.rfc.number,
                                         objection_id=ob.id, by="mallory",
                                         note="fixed")

    def test_resolve_twice_refused(self):
        ob = self.store.object(self.rfc.number, by="carol",
                               reason="breaks the cache",
                               resolves="add a migration")
        self.store.resolve_objection(self.rfc.number, objection_id=ob.id,
                                     by="carol", note="migration added")
        with self.assertRaisesRegex(RfcError, "already resolved"):
            self.store.resolve_objection(self.rfc.number,
                                         objection_id=ob.id, by="carol",
                                         note="again")

    def test_unresolved_objection_blocks_accept(self):
        self.store.object(self.rfc.number, by="carol",
                          reason="breaks the cache",
                          resolves="add a migration")
        self.store.approve(self.rfc.number, by="bob", ts=T0)
        with self.assertRaisesRegex(RfcError, "unresolved objections"):
            self.store.accept(self.rfc.number, by="bob",
                              ts=T0 + 4 * DAY)


class AcceptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _store(self.tmp.name)
        self.rfc = _open(self.store)
        self.store.start_period(self.rfc.number, by="bob", ts=T0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_accept_before_period_end_refused_naming_end(self):
        self.store.approve(self.rfc.number, by="bob", ts=T0)
        with self.assertRaisesRegex(RfcError, "comment period"):
            self.store.accept(self.rfc.number, by="bob", ts=T0 + DAY)

    def test_accept_without_approval_refused(self):
        with self.assertRaisesRegex(RfcError, "no approval"):
            self.store.accept(self.rfc.number, by="bob", ts=T0 + 4 * DAY)

    def test_accept_after_period_with_approval(self):
        self.store.approve(self.rfc.number, by="bob", ts=T0)
        out = self.store.accept(self.rfc.number, by="bob",
                                ts=T0 + 4 * DAY)
        self.assertEqual(out.status, "accepted")
        self.assertEqual(out.decision["method"], "period")
        self.assertEqual(out.decision["modules"], ["core", "memory"])

    def test_accept_writes_journal_entry(self):
        self.store.approve(self.rfc.number, by="bob", ts=T0)
        self.store.accept(self.rfc.number, by="bob", ts=T0 + 4 * DAY)
        rows = self.store.journal.open_decisions()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].decision, "RFC 0001: Merge the caches")
        self.assertEqual(rows[0].chosen, "accepted")
        self.assertIn("rfc", rows[0].tags)
        self.assertIn("rfc-0001", rows[0].tags)
        self.assertIn("core", rows[0].context)

    def test_accept_twice_refused(self):
        self.store.approve(self.rfc.number, by="bob", ts=T0)
        self.store.accept(self.rfc.number, by="bob", ts=T0 + 4 * DAY)
        with self.assertRaisesRegex(RfcError, "already accepted"):
            self.store.accept(self.rfc.number, by="bob",
                              ts=T0 + 5 * DAY)

    def test_owner_early_accept_from_draft_with_reason(self):
        draft = _open(self.store, title="Early one")
        out = self.store.accept(draft.number, by="owner",
                                reason="blocks the release", ts=T0)
        self.assertEqual(out.status, "accepted")
        self.assertEqual(out.decision["method"], "owner-early")
        self.assertEqual(out.decision["reason"], "blocks the release")

    def test_owner_early_accept_without_reason_refused(self):
        draft = _open(self.store, title="Early one")
        with self.assertRaisesRegex(RfcError, "comment period"):
            self.store.accept(draft.number, by="owner", ts=T0)

    def test_non_owner_early_accept_refused(self):
        draft = _open(self.store, title="Early one")
        with self.assertRaisesRegex(RfcError, "comment period"):
            self.store.accept(draft.number, by="bob", reason="please",
                              ts=T0)


class CloseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _store(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_close_after_period_by_maintainer(self):
        rfc = _open(self.store)
        self.store.start_period(rfc.number, by="bob", ts=T0)
        out = self.store.close(rfc.number, by="carol",
                               reason="out of scope", ts=T0 + 4 * DAY)
        self.assertEqual(out.status, "closed")
        self.assertEqual(out.decision["reason"], "out of scope")

    def test_close_needs_reason(self):
        rfc = _open(self.store)
        self.store.start_period(rfc.number, by="bob", ts=T0)
        with self.assertRaisesRegex(RfcError, "reason"):
            self.store.close(rfc.number, by="carol", reason=" ",
                             ts=T0 + 4 * DAY)

    def test_close_in_period_by_non_owner_refused(self):
        rfc = _open(self.store)
        self.store.start_period(rfc.number, by="bob", ts=T0)
        with self.assertRaisesRegex(RfcError, "comment period"):
            self.store.close(rfc.number, by="carol",
                             reason="no", ts=T0 + DAY)

    def test_owner_early_close_from_draft(self):
        rfc = _open(self.store)
        out = self.store.close(rfc.number, by="owner",
                               reason="duplicate of RFC 0000", ts=T0)
        self.assertEqual(out.status, "closed")

    def test_comment_on_terminal_refused(self):
        rfc = _open(self.store)
        self.store.close(rfc.number, by="owner", reason="no", ts=T0)
        with self.assertRaisesRegex(RfcError, "no more comments"):
            self.store.comment(rfc.number, by="alice", text="wait")


class SupersedeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _store(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_supersede_closed_only(self):
        rfc = _open(self.store)
        with self.assertRaisesRegex(RfcError, "not closed"):
            self.store.supersede(rfc.number, title="Try again",
                                 kind="other", modules=["core"],
                                 author="alice")

    def test_supersede_links_both_ways(self):
        old = _open(self.store)
        self.store.close(old.number, by="owner", reason="needs rework",
                         ts=T0)
        new = self.store.supersede(old.number, title="Merge the caches v2",
                                   kind="other", modules=["core", "memory"],
                                   author="alice", ts=T0)
        self.assertEqual(new.number, 2)
        self.assertEqual(new.supersedes, 1)
        self.assertEqual(old.superseded_by, 2)
        self.assertEqual(old.status, "closed")

    def test_supersede_unknown_refused(self):
        with self.assertRaisesRegex(RfcError, "no RFC"):
            self.store.supersede(99, title="Ghost", kind="other",
                                 modules=["core"], author="alice")


class CrossModuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _store(self.tmp.name)
        _open(self.store, title="Cache", modules=["core", "memory"])
        _open(self.store, title="Voice", modules=["voice"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_by_module(self):
        rows = self.store.by_module("memory")
        self.assertEqual([r.number for r in rows], [1])
        rows = self.store.by_module("core")
        self.assertEqual([r.number for r in rows], [1])
        self.assertEqual(self.store.by_module("nope"), [])


class MaintainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _store(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_maintainer_owner_only(self):
        with self.assertRaisesRegex(RfcError, "not the owner"):
            self.store.add_maintainer("dave", by="bob")
        names = self.store.add_maintainer("dave", by="owner")
        self.assertIn("dave", names)

    def test_remove_owner_refused(self):
        with self.assertRaisesRegex(RfcError, "cannot be removed"):
            self.store.remove_maintainer("owner", by="owner")

    def test_remove_maintainer(self):
        self.store.add_maintainer("dave", by="owner")
        names = self.store.remove_maintainer("dave", by="owner")
        self.assertNotIn("dave", names)


class PersistenceTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            rfc = _open(store, kind="process")
            store.start_period(rfc.number, by="bob", ts=T0)
            store.object(rfc.number, by="carol", reason="too big",
                         resolves="split it")
            store.approve(rfc.number, by="bob", ts=T0)
            reloaded = RfcStore(os.path.join(tmp, "rfcs.json"),
                                journal=DecisionJournal(
                                    os.path.join(tmp, "decisions.json")))
            got = reloaded.get(1)
            self.assertEqual(got.status, "in_comment")
            self.assertEqual(got.period_ends_at, T0 + 7 * DAY)
            self.assertEqual(len(got.objections), 1)
            self.assertEqual(got.objections[0].reason, "too big")
            self.assertEqual(reloaded.next_number, 2)
            self.assertIn("bob", reloaded.maintainers)
            # A new RFC after reload takes number 2, not 1 again.
            nxt = reloaded.open_rfc(title="Next", kind="other",
                                    modules=["core"], author="alice", ts=T0)
            self.assertEqual(nxt.number, 2)


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _store(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dispatcher_end_to_end(self):
        out = handle_rfc(self.store,
                         ["open", "Merge the caches", "|", "other", "|",
                          "core; memory", "|", "one cache for all"],
                         actor="alice")
        self.assertIn("opened RFC 0001", out)
        out = handle_rfc(self.store, ["period", "1"], actor="bob")
        self.assertIn("comment period started", out)
        out = handle_rfc(self.store, ["approve", "1"], actor="bob")
        self.assertIn("approved RFC 0001", out)
        with self.assertRaises(RfcError):
            handle_rfc(self.store, ["accept", "1"], actor="bob")
        # Owner accepts early with a reason — the mined override.
        out = handle_rfc(self.store, ["accept", "1", "|", "needed now"],
                         actor="owner")
        self.assertIn("accepted", out)
        self.assertEqual(self.store.journal.open_decisions()[0].decision,
                         "RFC 0001: Merge the caches")

    def test_dispatcher_usage_errors(self):
        with self.assertRaises(RfcUsageError):
            handle_rfc(self.store, [], actor="owner")
        with self.assertRaises(RfcUsageError):
            handle_rfc(self.store, ["frobnicate"], actor="owner")
        with self.assertRaises(RfcUsageError):
            handle_rfc(self.store, ["show"], actor="owner")

    def test_dispatcher_object_and_resolve(self):
        handle_rfc(self.store,
                   ["open", "Merge the caches", "|", "other", "|", "core"],
                   actor="alice")
        handle_rfc(self.store, ["period", "1"], actor="bob")
        out = handle_rfc(self.store,
                         ["object", "1", "|", "breaks the cache", "|",
                          "add a migration"], actor="carol")
        self.assertIn("objection ob-", out)
        ob_id = out.split("objection ")[1].split(" ")[0]
        out = handle_rfc(self.store,
                         ["resolve", "1", ob_id, "|", "migration added"],
                         actor="carol")
        self.assertIn("resolved", out)

    def test_dispatcher_list_and_module(self):
        self.assertIn("no open RFCs",
                      handle_rfc(self.store, ["list"], actor="owner"))
        handle_rfc(self.store,
                   ["open", "Cache", "|", "other", "|", "core"],
                   actor="alice")
        self.assertIn("RFC 0001",
                      handle_rfc(self.store, ["list"], actor="owner"))
        self.assertIn("RFC 0001",
                      handle_rfc(self.store, ["module", "core"],
                                 actor="owner"))
        self.assertIn("no RFCs affect",
                      handle_rfc(self.store, ["module", "voice"],
                                 actor="owner"))

    def test_tool_count_stays_806(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        # Lane 128 adds no tool: the RFC verbs ride the existing
        # ``nm decisions`` / ``/decisions`` surface.
        # 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 806 after gapfix119 added matching. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 806 after gapfix119 added matching. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 806 after gapfix119 added matching. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 806 after gapfix119 added matching.
        self.assertEqual(len(registry._tools), 806)


if __name__ == "__main__":
    unittest.main()
