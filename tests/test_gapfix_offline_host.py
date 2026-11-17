"""Gapfix R2-42: the offline fallback host.

The offline package (OfflineQueue / SyncEngine / OFFLINE_MATRIX) was
complete but had zero production callers: a failed send lost the
action, reconnecting synced nothing, and the feature gates never
consulted the matrix. These tests pin the host that wires it:

* a transport-failed approved email is queued durably (survives a
  restart, in FIFO order) — never silently dropped;
* on reconnect the queue drains through sync() in FIFO order and
  empties; a failed action is retried on the next drain;
* a conflict (same record changed both sides) is flagged visibly
  with BOTH versions, neither silently dropped, and published once;
* a degraded matrix feature reports its degraded level honestly;
* a blocked matrix feature refuses honestly instead of failing
  mid-way;
* the gateway outbox gets a reconnect pump (chat replies already
  persist — they were never queued twice).
"""

from types import SimpleNamespace

import pytest

from nomorals.agents.email_triage import (
    DraftQueue,
    check_explicit_send,
)
from nomorals.connectors import GmailError
from nomorals.offline import OfflineQueue
from nomorals.social.chat.base import ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.outbox import Outbox
from nomorals.storage.db import Database

# ── fakes ────────────────────────────────────────────────────────────────────


class FailingGmail:
    """Gmail whose send fails the way a dead transport fails."""

    def __init__(self, *, status_code: int = 0) -> None:
        self.status_code = status_code
        self.calls: list[dict] = []

    def send_message(self, to, subject, body, **kwargs):
        self.calls.append({"to": to, "subject": subject, "body": body})
        raise GmailError(
            "gmail request failed: network is unreachable",
            status_code=self.status_code,
        )


class WorkingGmail:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def send_message(self, to, subject, body, **kwargs):
        self.calls.append({"to": to, "subject": subject, "body": body})
        return {"ok": True, "id": "msg-1"}


class FakeNotifier:
    def __init__(self) -> None:
        self.published: list[dict] = []

    def publish(self, kind, title="", body="", **kwargs):
        self.published.append({"kind": kind, "title": title, "body": body})
        return {"ok": True}


class FakeGateway:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.drain_calls = 0

    def drain_queued(self):
        self.drain_calls += 1
        if self.fail:
            raise RuntimeError("gateway down")
        return {"ok": True, "topics": {"chat_outbox:test:1": [{"ok": True}]}}


class FakeScheduler:
    def __init__(self) -> None:
        self.jobs: list[dict] = []

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, *, routine=False):
        job = {"id": f"job-{len(self.jobs) + 1}", "name": name,
               "spec": spec, "kind": kind, "payload": payload,
               "routine": routine}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


class FakeSettings:
    def __init__(self, base) -> None:
        self._base = base

    def resolve(self, rel):
        return self._base / rel


def _context(tmp_path):
    return SimpleNamespace(settings=FakeSettings(tmp_path), db=None)


def _approved_draft(tmp_path, *, subdir=""):
    path = (tmp_path / subdir / "email_triage.json" if subdir
            else tmp_path / "email_triage.json")
    queue = DraftQueue(path=path)
    draft = queue.add(thread_id="t1", to="boss@example.com",
                      subject="Status", body="all good")
    assert queue.approve(draft.id)
    return queue, draft


# ── 1. a failed send is queued durably ────────────────────────────────────────


def test_send_failure_while_offline_is_queued_durably(tmp_path):
    from nomorals.agents.offline_host import OfflineHost

    host = OfflineHost(root=tmp_path / "off")
    queue, draft = _approved_draft(tmp_path)
    result = queue.send(FailingGmail(), draft.id, confirmed=True, host=host)
    assert result.get("queued_offline") is True
    action_id = result.get("action_id")
    assert action_id
    # The draft is still approved — the send never landed, so it must
    # not be stamped "sent".
    assert queue.get(draft.id).status == "approved"
    # The action survives a restart (fresh queue over the same root),
    # in FIFO order.
    fresh = OfflineQueue(tmp_path / "off")
    pending = fresh.pending()
    assert [a.id for a in pending] == [action_id]
    assert pending[0].kind == "email_send"
    assert pending[0].payload["to"] == "boss@example.com"
    assert pending[0].payload["subject"] == "Status"
    assert pending[0].payload["body"] == "all good"
    assert pending[0].payload["draft_id"] == draft.id


def test_send_failure_queues_in_fifo_order(tmp_path):
    from nomorals.agents.offline_host import OfflineHost

    host = OfflineHost(root=tmp_path / "off")
    queue, first = _approved_draft(tmp_path)
    second = queue.add(thread_id="t2", to="two@example.com",
                       subject="Second", body="two")
    assert queue.approve(second.id)
    queue.send(FailingGmail(), first.id, confirmed=True, host=host)
    queue.send(FailingGmail(), second.id, confirmed=True, host=host)
    pending = OfflineQueue(tmp_path / "off").pending()
    assert [a.payload["draft_id"] for a in pending] == [first.id, second.id]


def test_non_transport_gmail_error_is_not_queued(tmp_path):
    from nomorals.agents.offline_host import OfflineHost

    host = OfflineHost(root=tmp_path / "off")
    queue, draft = _approved_draft(tmp_path)
    # 401 = the token is revoked. Waiting for a reconnect will never
    # fix it, so it must surface as an error, not a queued action.
    with pytest.raises(GmailError):
        queue.send(FailingGmail(status_code=401), draft.id,
                   confirmed=True, host=host)
    assert OfflineQueue(tmp_path / "off").count_pending() == 0
    assert queue.get(draft.id).status == "approved"


def test_explicit_send_reports_queued_when_offline(tmp_path, monkeypatch):
    from nomorals.agents import email_triage
    from nomorals.agents.offline_host import OfflineHost

    host = OfflineHost(root=tmp_path / "off")
    context = _context(tmp_path)
    # check_explicit_send builds its own DraftQueue from the context's
    # settings path (data/email_triage.json) — draft there.
    queue, draft = _approved_draft(tmp_path, subdir="data")
    queue.mark_pending_send(draft.id)
    monkeypatch.setattr(email_triage, "_get_gmail",
                        lambda ctx: FailingGmail())
    reply = check_explicit_send("send it", context, host=host)
    assert reply is not None
    assert "offline" in reply.lower()
    assert "queued" in reply.lower()
    assert OfflineQueue(tmp_path / "off").count_pending() == 1


# ── 2. reconnect drains FIFO and empties ─────────────────────────────────────


def test_drain_applies_fifo_and_empties(tmp_path):
    from nomorals.agents.offline_host import OfflineHost

    host = OfflineHost(root=tmp_path / "off")
    applied: list[int] = []
    host.register_applier("step", lambda a: applied.append(a.payload["n"]))
    for n in (1, 2, 3):
        host.enqueue_failed("step", {"n": n}, entity=f"doc:{n}",
                            base_version=0)
    host.note_send_failure("network is unreachable")
    assert host.is_offline
    summary = host.drain()
    assert summary["ok"] is True
    assert applied == [1, 2, 3]
    assert summary["applied"] and len(summary["applied"]) == 3
    assert host.queue.count_pending() == 0
    # A local-only drain proves nothing about the network, so the
    # offline observation stands until a network action applies or
    # a chat send succeeds.


def test_drain_retries_a_failed_action_on_the_next_drain(tmp_path):
    from nomorals.agents.offline_host import OfflineHost

    host = OfflineHost(root=tmp_path / "off")
    attempts = {"n": 0}

    def _flaky(action):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("still down")

    host.register_applier("step", _flaky)
    action = host.enqueue_failed("step", {"n": 1})
    first = host.drain()
    assert len(first["failed"]) == 1
    assert host.queue.get(action.id).status == "failed"
    second = host.drain()
    assert second["applied"] == [action.id]
    assert host.queue.count_pending() == 0


# ── 3. conflicts are flagged visibly, neither side dropped ───────────────────


def test_drain_flags_conflict_with_both_versions(tmp_path):
    from nomorals.agents.offline_host import OfflineHost

    host = OfflineHost(root=tmp_path / "off")
    applied: list[str] = []
    host.register_applier("edit", lambda a: applied.append(a.payload["text"]))
    host.set_version("doc:1", 5)
    stale = host.enqueue_failed("edit", {"text": "stale"},
                                entity="doc:1", base_version=0)
    host.enqueue_failed("edit", {"text": "fresh-ish"},
                        entity="doc:2", base_version=0)
    notifier = FakeNotifier()
    summary = host.drain(notifier=notifier)
    assert summary["ok"] is False
    assert len(summary["conflicts"]) == 1
    conflict = summary["conflicts"][0]
    assert conflict["entity"] == "doc:1"
    assert conflict["base_version"] == 0
    assert conflict["current_version"] == 5
    # The stale write was NOT applied; the good one was.
    assert applied == ["fresh-ish"]
    assert host.queue.get(stale.id).status == "conflict"
    # The conflict is published once, naming both versions.
    assert len(notifier.published) == 1
    pub = notifier.published[0]
    assert pub["kind"] == "offline-sync"
    assert "doc:1" in pub["title"] or "doc:1" in pub["body"]
    assert "0" in pub["body"] and "5" in pub["body"]
    # A second drain does NOT re-publish the same conflict.
    host.drain(notifier=notifier)
    assert len(notifier.published) == 1


# ── 4/5. matrix consults: degraded honest, blocked refused ───────────────────


def test_feature_status_reports_degraded_honestly(tmp_path):
    from nomorals.agents.offline_host import OfflineHost

    host = OfflineHost(root=tmp_path / "off")
    host.note_send_failure("network is unreachable")
    status = host.feature_status("email_send")
    assert status["offline"] is True
    assert status["level"] == "degraded"
    assert status["allowed"] is True
    assert "queued" in status["note"].lower()
    # While online the same feature is simply allowed.
    host.note_send_success()
    online = host.feature_status("email_send")
    assert online["offline"] is False
    assert online["allowed"] is True


def test_feature_status_refuses_blocked_honestly(tmp_path):
    from nomorals.agents.offline_host import OfflineHost

    host = OfflineHost(root=tmp_path / "off")
    host.note_send_failure("network is unreachable")
    status = host.feature_status("web_search")
    assert status["level"] == "unavailable"
    assert status["allowed"] is False
    assert "network" in status["note"].lower()
    # Money stays blocked too.
    assert host.feature_status("payments")["allowed"] is False


def test_offline_feature_refusal_gate(tmp_path):
    from nomorals.agents.offline_host import OfflineHost
    from nomorals.agents.partner.runtime import PartnerRuntime

    host = OfflineHost(root=tmp_path / "off")
    host.note_send_failure("network is unreachable")
    fake_self = SimpleNamespace(_offline_host=host)
    refusal = PartnerRuntime._offline_feature_refusal(fake_self, "web_search")
    assert refusal is not None
    assert "web_search" in refusal
    assert "unavailable" in refusal.lower() or "offline" in refusal.lower()
    # A degraded feature is NOT refused — it queues instead.
    assert PartnerRuntime._offline_feature_refusal(
        fake_self, "email_send") is None
    # While online nothing is refused.
    host.note_send_success()
    assert PartnerRuntime._offline_feature_refusal(
        fake_self, "web_search") is None


# ── outbox pump ───────────────────────────────────────────────────────────────


def test_gateway_drain_queued_pumps_outbox(tmp_path):
    db = Database(":memory:")
    db.migrate()
    outbox = Outbox(db)
    chat = ChatRef(platform="test", chat_id="1")
    outbox.submit(platform="test", chat=chat, text="first")
    outbox.submit(platform="test", chat=chat, text="second")

    sent: list[str] = []

    def _deliver(payload):
        sent.append(str(payload.get("text") or ""))
        return SendResult(ok=True, platform="test",
                          message_id=f"m-{len(sent)}")

    gateway = ChatGateway.__new__(ChatGateway)
    gateway.outbox = outbox
    gateway._deliver_outbox_payload = _deliver
    summary = ChatGateway.drain_queued(gateway)
    assert summary["ok"] is True
    assert sent == ["first", "second"]
    # Nothing left to pump.
    assert ChatGateway.drain_queued(gateway)["topics"] == {}


def test_host_drain_pumps_gateway_outbox(tmp_path):
    from nomorals.agents.offline_host import OfflineHost

    host = OfflineHost(root=tmp_path / "off")
    gateway = FakeGateway()
    summary = host.drain(gateway=gateway)
    assert gateway.drain_calls == 1
    assert summary["outbox"]["ok"] is True


# ── tick + scheduler ──────────────────────────────────────────────────────────


def test_tick_drains_and_never_raises(tmp_path):
    from nomorals.agents.offline_host import (
        OfflineHost,
        run_offline_sync_tick_once,
    )

    host = OfflineHost(root=tmp_path / "off")
    applied: list[str] = []
    host.register_applier("step", lambda a: applied.append(a.payload["n"]))
    host.enqueue_failed("step", {"n": "x"})
    context = _context(tmp_path)
    summary = run_offline_sync_tick_once(
        context, host=host, gateway=FakeGateway(),
        notifier=FakeNotifier())
    assert summary["ok"] is True
    assert applied == ["x"]
    # Even a dead gateway never kills the tick: the queue drain still
    # reports ok, and the pump failure is reported in the outbox slot.
    broken = run_offline_sync_tick_once(
        context, host=host, gateway=FakeGateway(fail=True),
        notifier=FakeNotifier())
    assert broken["ok"] is True
    assert broken["outbox"]["ok"] is False
    assert "error" in broken["outbox"]


def test_ensure_offline_sync_job_registers(tmp_path):
    from nomorals.agents.offline_host import (
        OFFLINE_SYNC_JOB,
        OFFLINE_SYNC_TOOL,
        ensure_offline_sync_job,
    )

    scheduler = FakeScheduler()
    job = ensure_offline_sync_job(scheduler)
    assert job is not None
    assert job["name"] == OFFLINE_SYNC_JOB
    assert job["spec"] == "every 5m"
    assert job["payload"]["tool"] == OFFLINE_SYNC_TOOL
    assert job["routine"] is True
    # Idempotent: a second registration returns the same job.
    assert ensure_offline_sync_job(scheduler) is job
    assert len(scheduler.jobs) == 1


# ── email applier ─────────────────────────────────────────────────────────────


def test_email_applier_sends_exact_payload_and_marks_sent(
        tmp_path, monkeypatch):
    from nomorals.agents import email_triage
    from nomorals.agents.offline_host import (
        OfflineHost,
        register_default_appliers,
    )

    gmail = WorkingGmail()
    monkeypatch.setattr(email_triage, "_get_gmail", lambda ctx: gmail)
    context = _context(tmp_path)
    host = OfflineHost(root=tmp_path / "off")
    register_default_appliers(host, context)
    # The applier reads drafts through the context's settings path
    # (data/email_triage.json) — draft there.
    queue, draft = _approved_draft(tmp_path, subdir="data")
    host.enqueue_failed("email_send", {
        "draft_id": draft.id, "to": draft.to,
        "subject": draft.subject, "body": draft.body,
    }, entity=f"email_draft:{draft.id}")
    assert host.is_offline
    summary = host.drain()
    assert summary["ok"] is True
    # A network action applied — the transport proved itself alive,
    # so the offline observation clears.
    assert not host.is_offline
    assert gmail.calls == [{"to": "boss@example.com", "subject": "Status",
                            "body": "all good"}]
    # The draft store now says sent — a second drain must not re-send.
    assert DraftQueue(path=tmp_path / "data" / "email_triage.json").get(
        draft.id).status == "sent"
    host.enqueue_failed("email_send", {
        "draft_id": draft.id, "to": draft.to,
        "subject": draft.subject, "body": draft.body,
    }, entity=f"email_draft:{draft.id}")
    host.drain()
    assert len(gmail.calls) == 1


def test_mark_sent(tmp_path):
    queue, draft = _approved_draft(tmp_path)
    assert queue.mark_sent(draft.id) is True
    assert queue.get(draft.id).status == "sent"
    assert queue.mark_sent(draft.id) is False
    assert queue.mark_sent("dr_missing") is False
