"""Doc 15 P5 remainder — transfer registry + delivery/publication
receipts (storage/delivery_records.py + the steward's transfers and
artifacts adapters).

At P5 two briefed domains were named unavailable by
adapter_health(), on the record: transfers ("no transfer registry
exists — begun transfers leave no queryable record") and artifacts
("nothing records whether an artifact was ever delivered"). This
pass built the records at the real hand-off seams — /artifact
send|export (store → chat) and execute_post (draft artifacts →
platforms) — and the adapters that read them.

Pins:
(a) the store: a begun transfer completed ok mints exactly one
    delivery receipt carrying the send's own reference; a failed
    completion records the failure and mints NO receipt; a second
    completion mints nothing new;
(b) the funnel: /artifact send through the real handler journals a
    delivered transfer + receipt (send ref from the send's own
    result); a failed send journals the failure outcome, not a
    success receipt; a registry that raises leaves the delivery
    itself untouched (fail-open);
(c) publication: an artifact-carrying draft's execute_post mints
    publication receipts tying draft ↔ artifact ↔ platform outcome
    (post reference + the publish's media accounting); a failed
    platform records a failed receipt; a media-less draft mints
    none; a raising registry leaves the post untouched;
(d) the steward: a stuck begun transfer opens ONE case across two
    sweeps and, with no outcome recorded, the case closes UNKNOWN
    (unknown stays unknown); once the transfer lands delivered,
    the case resolves ``delivered`` citing the delivery receipt;
    an old failed delivery opens an abandoned artifacts case that
    resolves ``failed``; an artifact case with a delivery receipt
    resolves ``delivered`` citing it; an artifact with no records
    at all stays UNKNOWN; renders never print the chat key or ids;
(e) registration honesty: adapter_health names all nine domains
    working (the registry moved 7 working + 2 unavailable →
    9 + 0), and the Notifier producer census
    (agents/notify_producers) validates clean with its counts
    UNMOVED — this pass adds no publish-shaped call site, so the
    mechanical enforcement accepts the new code with zero
    unregistered sites rather than a fabricated producer.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.notify_producers import (
    summarize as census_summarize,
)
from nomorals.agents.notify_producers import (
    validate_inventory,
)
from nomorals.agents.outcome_steward import (
    OutcomeSteward,
    adapter_health,
    render_case_detail,
)
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.core.jobs import JobRegistry
from nomorals.social.base import PlatformAdapter, PostResult
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.drafts import DraftQueue, execute_post
from nomorals.social.manager import SocialManager
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database
from nomorals.storage.delivery_records import (
    DeliveryRecordError,
    DeliveryRecordStore,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAT_KEY = "whatsapp:2348000000000@s.whatsapp.net"
PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png-bytes"


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _age_transfer(db: Database, entry_id: str, *, begun_ago: float,
                  finished_ago: float | None = None) -> None:
    now = time.time()
    if finished_ago is None:
        db.execute(
            "UPDATE transfer_entries SET begun_at = ? WHERE id = ?",
            (now - begun_ago, entry_id))
    else:
        db.execute(
            "UPDATE transfer_entries SET begun_at = ?,"
            " finished_at = ? WHERE id = ?",
            (now - begun_ago, now - finished_ago, entry_id))


# ── (a) the store ────────────────────────────────────────────────────


class DeliveryRecordStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _db()
        self.records = DeliveryRecordStore(self.db)

    def test_complete_ok_mints_receipt_with_send_facts(self) -> None:
        entry = self.records.begin_transfer(
            artifact_id="art-1", artifact_label="Q3 report",
            kind="send", from_surface="artifact store",
            platform="telegram", chat_key="telegram:999")
        self.assertEqual(entry.state, "begun")
        landed, receipt = self.records.complete_transfer(
            entry.id, ok=True, send_ref="msg-42")
        self.assertEqual(landed.state, "delivered")
        self.assertEqual(landed.send_ref, "msg-42")
        self.assertIsNotNone(receipt)
        assert receipt is not None
        self.assertEqual(receipt.transfer_id, entry.id)
        self.assertEqual(receipt.artifact_id, "art-1")
        self.assertEqual(receipt.platform, "telegram")
        self.assertEqual(receipt.send_ref, "msg-42")
        self.assertEqual(receipt.outcome, "delivered")

    def test_complete_failure_records_failure_and_no_receipt(
            self) -> None:
        entry = self.records.begin_transfer(
            artifact_id="art-1", platform="telegram")
        landed, receipt = self.records.complete_transfer(
            entry.id, ok=False, error="gateway exploded")
        self.assertEqual(landed.state, "failed")
        self.assertEqual(landed.error, "gateway exploded")
        self.assertIsNone(receipt)
        self.assertEqual(
            self.records.delivery_receipts_for_artifact("art-1"), [])

    def test_second_completion_mints_nothing_new(self) -> None:
        entry = self.records.begin_transfer(
            artifact_id="art-1", platform="telegram")
        self.records.complete_transfer(entry.id, ok=True,
                                       send_ref="msg-1")
        again, receipt = self.records.complete_transfer(
            entry.id, ok=True, send_ref="msg-2")
        self.assertEqual(again.state, "delivered")
        self.assertEqual(again.send_ref, "msg-1")  # first outcome stands
        self.assertIsNotNone(receipt)
        self.assertEqual(
            len(self.records.delivery_receipts_for_artifact("art-1")),
            1)

    def test_complete_unknown_transfer_refused(self) -> None:
        with self.assertRaises(DeliveryRecordError):
            self.records.complete_transfer("tr-nope", ok=True)

    def test_publication_receipt_facts(self) -> None:
        posted = self.records.record_publication(
            draft_id="d-1", artifact_id="art-1", platform="mastodon",
            ok=True, external_id="ext-9", media_offered=2,
            media_carried=1)
        self.assertEqual(posted.outcome, "posted")
        failed = self.records.record_publication(
            draft_id="d-1", artifact_id="art-1", platform="x",
            ok=False, error="platform said no", media_offered=2)
        self.assertEqual(failed.outcome, "failed")
        self.assertEqual(failed.error, "platform said no")
        by_draft = self.records.publication_receipts_for_draft("d-1")
        self.assertEqual(len(by_draft), 2)
        by_artifact = \
            self.records.publication_receipts_for_artifact("art-1")
        self.assertEqual([r.platform for r in by_artifact],
                         ["mastodon", "x"])

    def test_stuck_and_abandoned_queries(self) -> None:
        stuck = self.records.begin_transfer(
            artifact_id="art-stuck", platform="telegram")
        _age_transfer(self.db, stuck.id, begun_ago=7200)
        fresh = self.records.begin_transfer(
            artifact_id="art-fresh", platform="telegram")
        self.assertEqual(
            [e.id for e in self.records.stuck_transfers(
                time.time(), older_than_s=3600)], [stuck.id])
        failed = self.records.begin_transfer(
            artifact_id="art-failed", platform="telegram")
        self.records.complete_transfer(failed.id, ok=False,
                                       error="boom")
        _age_transfer(self.db, failed.id, begun_ago=90000,
                      finished_ago=90000)
        recovered = self.records.begin_transfer(
            artifact_id="art-recovered", platform="telegram")
        self.records.complete_transfer(recovered.id, ok=False,
                                       error="boom")
        _age_transfer(self.db, recovered.id, begun_ago=90000,
                      finished_ago=90000)
        later = self.records.begin_transfer(
            artifact_id="art-recovered", platform="telegram")
        self.records.complete_transfer(later.id, ok=True,
                                       send_ref="msg-7")
        abandoned = self.records.abandoned_deliveries(
            time.time(), older_than_s=24 * 3600)
        self.assertEqual([e.artifact_id for e in abandoned],
                         ["art-failed"])
        self.assertEqual(fresh.state, "begun")


# ── (b) the /artifact funnel ─────────────────────────────────────────


def _msg(text: str, *, chat_key: str = "telegram:999") -> ChatMessage:
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform or "telegram",
                     chat_id=chat_id or "999", kind=ChatKind.DM),
        incoming=True, text=text, sender="owner",
        sender_id="5478650254", meta={"is_owner": True})


class _RecordingGateway:
    adapters: dict = {}

    def status(self) -> dict:
        return {}

    def set_rate_limit(self, *a: Any, **k: Any) -> None:
        pass

    def typing(self, *a: Any, **k: Any) -> None:
        pass

    def send(self, platform: str, chat: Any, text: str,
             **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(ok=True, error="")


class _FakeTools:
    """The file_send seam: records calls, returns a canned outcome
    carrying the send's own message reference on success."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((name, kwargs))
        if name != "file_send":
            return SimpleNamespace(
                ok=False, value=None,
                error=SimpleNamespace(
                    message=f"unexpected tool {name}"))
        if not self.ok:
            return SimpleNamespace(
                ok=False, value=None,
                error=SimpleNamespace(message="gateway exploded"))
        return SimpleNamespace(
            ok=True,
            value={"path": kwargs["path"], "message_id": "msg-42"},
            error=None)


class ArtifactFunnelRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-d15-records-")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.tools = _FakeTools()
        self.context.tools = self.tools
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())
        db = self.context.db
        blob_dir = Path(db.path).parent / "blobs"
        self.store = ArtifactStore(db, BlobStore(db, blob_dir))
        self.report = self.store.put_text(
            "# Q3 report\n\nrevenue grew.", type="report",
            creator="analyst", metadata={"title": "Q3 report"})
        self.records = DeliveryRecordStore(db)

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str) -> str:
        return self.runtime.handle_control(
            text, "telegram:999",
            message=_msg(text, chat_key="telegram:999"))

    def test_send_mints_transfer_and_delivery_receipt(self) -> None:
        self._run(f"/artifact send {self.report.id}")
        self.assertEqual(len(self.tools.calls), 1)
        transfers = self.records.transfers_for_artifact(self.report.id)
        self.assertEqual(len(transfers), 1)
        entry = transfers[0]
        self.assertEqual(entry.kind, "send")
        self.assertEqual(entry.state, "delivered")
        self.assertEqual(entry.platform, "telegram")
        self.assertEqual(entry.artifact_label, "Q3 report")
        receipts = self.records.delivery_receipts_for_artifact(
            self.report.id)
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0].transfer_id, entry.id)
        self.assertEqual(receipts[0].send_ref, "msg-42")

    def test_failed_send_records_failure_not_a_receipt(self) -> None:
        self.tools.ok = False
        out = self._run(f"/artifact send {self.report.id}")
        self.assertIn("artifact send failed", out)
        transfers = self.records.transfers_for_artifact(self.report.id)
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].state, "failed")
        self.assertIn("gateway exploded", transfers[0].error)
        self.assertEqual(
            self.records.delivery_receipts_for_artifact(
                self.report.id), [])

    def test_export_mints_a_transfer_per_artifact(self) -> None:
        self._run(f"/artifact export {self.report.id}")
        transfers = self.records.transfers_for_artifact(self.report.id)
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].kind, "export")
        self.assertEqual(transfers[0].state, "delivered")
        self.assertEqual(
            len(self.records.delivery_receipts_for_artifact(
                self.report.id)), 1)

    def test_registry_failure_leaves_the_delivery_untouched(
            self) -> None:
        with mock.patch.object(
                DeliveryRecordStore, "begin_transfer",
                side_effect=RuntimeError("registry down")):
            out = self._run(f"/artifact send {self.report.id}")
        self.assertEqual(len(self.tools.calls), 1)  # the send happened
        self.assertNotIn("failed", out)


# ── (c) publication through execute_post ─────────────────────────────


class _PubAdapter(PlatformAdapter):
    max_chars = 5000
    supports_media = True

    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.calls: list[dict] = []

    def post(self, account, content, **kwargs):
        media = list(kwargs.get("media_paths") or [])
        self.calls.append({"content": content, "media": media})
        if self.fail:
            return PostResult(platform=self.name, ok=False,
                              error="platform said no")
        return PostResult(
            platform=self.name, ok=True, external_id="ext-1",
            metrics={"media_attached": len(media)} if media else {})


class PublicationRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-d15-pub-")
        self.root = Path(self.tmp.name)
        self.db = Database(self.root / "ctx.db")
        self.db.migrate()
        self.store = ArtifactStore(
            self.db, BlobStore(self.db, self.root / "blobs"))
        self.queue = DraftQueue(db_path=self.root / "drafts.db")
        self.records = DeliveryRecordStore(self.db)

    def tearDown(self) -> None:
        self.queue.close()
        self.tmp.cleanup()

    def _manager(self, adapters: list) -> SocialManager:
        ctx = SimpleNamespace(
            db=self.db, tools=None, extras={},
            settings=SimpleNamespace(workspace_dir=str(self.root)))
        manager = SocialManager(
            ctx, adapters={a.name: a for a in adapters},
            enforce_policy=False)
        for adapter in adapters:
            manager.connect(adapter.name, f"@owner-{adapter.name}")
        return manager

    def _media_draft(self, platforms: list[str]) -> Any:
        art = self.store.put(
            PNG_BYTES, type="image", creator="test",
            mime="image/png", metadata={"filename": "desk.png"})
        draft = self.queue.create_draft(
            "the desk card for today", platforms=platforms,
            metadata={"media": [{"artifact": art.id,
                                 "mime": "image/png",
                                 "name": "desk.png"}]})
        self.queue.propose(draft.id)
        self.queue.approve(draft.id)
        return draft, art

    def test_publication_receipt_ties_post_artifact_platform(
            self) -> None:
        adapter = _PubAdapter("mastodon")
        manager = self._manager([adapter])
        draft, art = self._media_draft(["mastodon"])
        outcome = execute_post(manager, draft.content,
                               platforms=["mastodon"],
                               draft_id=draft.id, queue=self.queue,
                               artifact_store=self.store)
        self.assertEqual(len(outcome.posted), 1)
        receipts = self.records.publication_receipts_for_artifact(
            art.id)
        self.assertEqual(len(receipts), 1)
        receipt = receipts[0]
        self.assertEqual(receipt.draft_id, draft.id)
        self.assertEqual(receipt.platform, "mastodon")
        self.assertEqual(receipt.outcome, "posted")
        self.assertEqual(receipt.external_id, "ext-1")
        self.assertEqual(receipt.media_offered, 1)
        self.assertEqual(receipt.media_carried, 1)
        self.assertEqual(
            self.records.publication_receipts_for_draft(draft.id),
            receipts)

    def test_failed_platform_records_a_failed_receipt(self) -> None:
        adapter = _PubAdapter("x", fail=True)
        manager = self._manager([adapter])
        draft, art = self._media_draft(["x"])
        outcome = execute_post(manager, draft.content, platforms=["x"],
                               draft_id=draft.id, queue=self.queue,
                               artifact_store=self.store)
        self.assertEqual(len(outcome.failed), 1)
        receipts = self.records.publication_receipts_for_artifact(
            art.id)
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0].outcome, "failed")
        self.assertIn("platform said no", receipts[0].error)

    def test_media_less_draft_mints_no_receipt(self) -> None:
        adapter = _PubAdapter("mastodon")
        manager = self._manager([adapter])
        draft = self.queue.create_draft(
            "plain text post", platforms=["mastodon"])
        self.queue.propose(draft.id)
        self.queue.approve(draft.id)
        outcome = execute_post(manager, draft.content,
                               platforms=["mastodon"],
                               draft_id=draft.id, queue=self.queue,
                               artifact_store=self.store)
        self.assertEqual(len(outcome.posted), 1)
        self.assertEqual(self.records.counts()
                         ["publication_receipts"], 0)

    def test_registry_failure_leaves_the_post_untouched(self) -> None:
        adapter = _PubAdapter("mastodon")
        manager = self._manager([adapter])
        draft, art = self._media_draft(["mastodon"])
        with mock.patch.object(
                DeliveryRecordStore, "record_publication",
                side_effect=RuntimeError("registry down")):
            outcome = execute_post(manager, draft.content,
                                   platforms=["mastodon"],
                                   draft_id=draft.id, queue=self.queue,
                                   artifact_store=self.store)
        self.assertEqual(len(outcome.posted), 1)
        self.assertEqual(len(adapter.calls), 1)


# ── (d) the steward over the new records ─────────────────────────────


class StewardRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _db()
        self.records = DeliveryRecordStore(self.db)
        self.steward = OutcomeSteward(self.db, registry=JobRegistry())

    def test_health_names_all_nine_domains_working(self) -> None:
        health = adapter_health()
        self.assertEqual(len(health), 9)
        self.assertTrue(all(r["status"] == "working"
                            for r in health))
        by_domain = {r["domain"]: r for r in health}
        self.assertIn("transfer_entries",
                      by_domain["transfers"]["note"])
        self.assertIn("delivery_receipts",
                      by_domain["artifacts"]["note"])

    def test_stuck_transfer_case_stays_unknown(self) -> None:
        entry = self.records.begin_transfer(
            artifact_id="art-1", artifact_label="Q3 report",
            platform="whatsapp", chat_key=CHAT_KEY)
        _age_transfer(self.db, entry.id, begun_ago=7200)
        self.steward.sweep()
        self.steward.sweep()  # dedupe: still exactly one case
        cases = self.steward.cases.list_cases(domain="transfers")
        self.assertEqual(len(cases), 1)
        result = self.steward.run_case(cases[0].id)
        self.assertEqual(result["outcome"], "closed-unknown")
        closed = self.steward.cases.get(cases[0].id)
        self.assertEqual(closed.verdict, "outcome_unknown")
        self.assertEqual(closed.evidence_receipt_id, "")

    def test_transfer_case_resolves_delivered_citing_receipt(
            self) -> None:
        entry = self.records.begin_transfer(
            artifact_id="art-1", artifact_label="Q3 report",
            platform="whatsapp", chat_key=CHAT_KEY)
        _age_transfer(self.db, entry.id, begun_ago=7200)
        self.steward.sweep()
        case = self.steward.cases.list_cases(domain="transfers")[0]
        _entry, receipt = self.records.complete_transfer(
            entry.id, ok=True, send_ref="msg-9")
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "resolved")
        self.assertEqual(result["verdict"], "delivered")
        closed = self.steward.cases.get(case.id)
        cited = self.steward.cases.get_receipt(
            closed.evidence_receipt_id)
        self.assertIsNotNone(cited)
        assert cited is not None and receipt is not None
        self.assertEqual(cited.evidence.get("ref"), receipt.id)
        # owner renders: no chat key, no internal ids anywhere
        text = render_case_detail(
            closed, self.steward.cases.receipts(case.id))
        self.assertNotIn(CHAT_KEY, text)
        self.assertNotIn("2348000000000", text)
        self.assertNotIn(case.id, text)
        self.assertNotIn(receipt.id, text)

    def test_abandoned_failed_delivery_resolves_failed(self) -> None:
        entry = self.records.begin_transfer(
            artifact_id="art-2", artifact_label="chart.png",
            platform="telegram", chat_key="telegram:999")
        self.records.complete_transfer(entry.id, ok=False,
                                       error="gateway exploded")
        _age_transfer(self.db, entry.id, begun_ago=90000,
                      finished_ago=90000)
        self.steward.sweep()
        cases = self.steward.cases.list_cases(domain="artifacts")
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].reason, "abandoned")
        result = self.steward.run_case(cases[0].id)
        self.assertEqual(result["outcome"], "resolved")
        self.assertEqual(result["verdict"], "failed")

    def test_artifact_case_resolves_on_delivery_receipt(self) -> None:
        entry = self.records.begin_transfer(
            artifact_id="art-3", artifact_label="report",
            platform="telegram")
        _entry, receipt = self.records.complete_transfer(
            entry.id, ok=True, send_ref="msg-3")
        case, _created = self.steward.cases.open_case(
            domain="artifacts", kind="artifact delivery",
            subject="artifact delivery: report",
            subject_key="artifact:art-3", reason="unknown-outcome",
            detail={"artifact_id": "art-3"}, actor="test")
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "resolved")
        self.assertEqual(result["verdict"], "delivered")
        closed = self.steward.cases.get(case.id)
        cited = self.steward.cases.get_receipt(
            closed.evidence_receipt_id)
        self.assertIsNotNone(cited)
        assert cited is not None and receipt is not None
        self.assertEqual(cited.evidence.get("ref"), receipt.id)

    def test_artifact_case_without_records_stays_unknown(
            self) -> None:
        case, _created = self.steward.cases.open_case(
            domain="artifacts", kind="artifact delivery",
            subject="artifact delivery: ghost",
            subject_key="artifact:art-ghost",
            reason="unknown-outcome",
            detail={"artifact_id": "art-ghost"}, actor="test")
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "closed-unknown")
        closed = self.steward.cases.get(case.id)
        self.assertEqual(closed.verdict, "outcome_unknown")

    def test_collect_writes_nothing_to_the_registry(self) -> None:
        entry = self.records.begin_transfer(
            artifact_id="art-4", platform="telegram")
        self.records.complete_transfer(entry.id, ok=True,
                                       send_ref="msg-4")
        before = self.records.counts()
        case, _created = self.steward.cases.open_case(
            domain="artifacts", kind="artifact delivery",
            subject="artifact delivery: thing",
            subject_key="artifact:art-4", reason="unknown-outcome",
            detail={"artifact_id": "art-4"}, actor="test")
        self.steward.run_case(case.id)
        self.assertEqual(self.records.counts(), before)


# ── (e) producer-census honesty ──────────────────────────────────────


class ProducerCensusTests(unittest.TestCase):
    def test_census_validates_clean_and_unmoved(self) -> None:
        self.assertEqual(validate_inventory(REPO_ROOT), [])
        summary = census_summarize()
        # No Notifier producer was added by this pass — the records
        # are registry writes at the hand-off seams, not alerts —
        # so the census stands exactly where P1 pinned it, and the
        # mechanical enforcement accepts every new call site.
        self.assertEqual(summary["producers"], 65)
        self.assertEqual(summary["excluded"], 70)


if __name__ == "__main__":
    unittest.main()
