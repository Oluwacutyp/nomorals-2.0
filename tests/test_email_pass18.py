"""Pass #18 — email + triage fixes.

Three verified defects, each reproduced failing-first:

1. The daily triage job (``ensure_email_triage_job._daily_run``) queued
   a fresh draft for every needs-reply unread message on EVERY run,
   with none of the per-thread dedupe the chat ``/email triage`` path
   applies.  Triage is read-only (mail stays unread), so the same
   message spawned one duplicate draft per day, forever.

2. ``/email send <id>`` armed a DISCARDED draft and replied "ready to
   send … Reply `send it` and it's gone" — the owner's "send it" then
   failed inside ``DraftQueue.send`` ("was discarded").  The approve
   branch rejects discarded drafts up front; the send branch didn't.

3. Triage-queued drafts unconditionally prefixed ``Re: `` — a message
   already titled "Re: X" got a draft titled "Re: Re: X".  The
   ``/email reply`` path guards this; the triage paths didn't.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import nomorals.agents.email_triage as et
from nomorals.agents.email_triage import DraftQueue, control_email


def _msg(mid, *, subject, sender, snippet, thread="t1"):
    return {
        "id": mid,
        "threadId": thread,
        "snippet": snippet,
        "labelIds": ["INBOX", "UNREAD"],
        "payload": {"headers": [
            {"name": "Subject", "value": subject},
            {"name": "From", "value": sender},
            {"name": "Date", "value": "Mon, 06 Oct 2026 10:00:00 +0000"},
        ]},
    }


class FakeGmail:
    def __init__(self, messages):
        self._messages = {m["id"]: m for m in messages}
        self.sent: list[dict] = []

    def list_messages(self, *, query="", max_results=50, **kw):
        msgs = list(self._messages.values())
        if "is:unread" in query:
            msgs = [m for m in msgs if "UNREAD" in m.get("labelIds", [])]
        return {"messages": [{"id": m["id"], "threadId": m["threadId"]}
                             for m in msgs[:max_results]],
                "next_page_token": "", "result_size_estimate": len(msgs)}

    def get_message(self, message_id, *, format="metadata"):  # noqa: A002
        return self._messages[message_id]

    def send_message(self, to, subject, body, **kw):
        if not kw.get("confirmed"):
            raise PermissionError("not confirmed")
        self.sent.append({"to": to, "subject": subject, "body": body})
        return {"id": "sent1", "threadId": "t_sent"}


def _ctx(queue_path: Path):
    settings = SimpleNamespace(
        resolve=lambda rel: queue_path if "email_triage" in rel else Path(rel))
    return SimpleNamespace(db=None, settings=settings, persona=None)


# ── bug 1: the daily job must dedupe like the chat path ─────────────────────

class _FakeScheduler:
    def __init__(self):
        self.actions: dict = {}
        self.jobs: list = []

    async def list_cron_jobs(self):
        return list(self.jobs)

    def register_action(self, name, fn):
        self.actions[name] = fn

    async def schedule_cron(self, task_id, cron, action, parameters=None):
        self.jobs.append(SimpleNamespace(task_id=task_id))
        return True


def test_daily_job_does_not_duplicate_drafts(tmp_path, monkeypatch):
    gmail = FakeGmail([
        _msg("m1", subject="Contract review",
             sender="Adaeze Okafor <ada@example.com>",
             snippet="Could you please review the attached contract?",
             thread="t1"),
    ])
    monkeypatch.setattr(et, "_get_gmail", lambda context: gmail)
    queue_path = tmp_path / "email_triage.json"
    ctx = _ctx(queue_path)
    sched = _FakeScheduler()
    assert asyncio.run(et.ensure_email_triage_job(sched, ctx))
    run = sched.actions[et.EMAIL_DAILY_ACTION]
    asyncio.run(run())
    asyncio.run(run())  # next day: same unread mail, still needs a reply
    pending = DraftQueue(path=queue_path).pending()
    assert len(pending) == 1, (
        f"daily job queued {len(pending)} drafts for one thread — "
        "it must dedupe against pending drafts like /email triage does")
    assert gmail.sent == []  # the daily job never sends


# ── bug 2: /email send must refuse a discarded draft up front ───────────────

def test_send_refuses_discarded_draft(tmp_path, monkeypatch):
    gmail = FakeGmail([])
    monkeypatch.setattr(et, "_get_gmail", lambda context: gmail)
    queue_path = tmp_path / "email_triage.json"
    ctx = _ctx(queue_path)
    control_email("compose a@x.com | S | B", ctx)
    q = DraftQueue(path=queue_path)
    did = q.pending()[0].id
    assert q.discard(did)
    out = control_email(f"send {did}", ctx)
    assert "discarded" in out, (
        f"/email send on a discarded draft replied {out!r} — it must "
        "refuse up front, not arm the draft and promise 'ready to send'")
    assert "ready to send" not in out
    # nothing was armed: a stray "send it" has nothing to fire
    assert DraftQueue(path=queue_path).take_pending_send() == ""
    assert gmail.sent == []


# ── bug 3: triage drafts must not double the Re: prefix ─────────────────────

def test_triage_draft_subject_keeps_single_re(tmp_path, monkeypatch):
    gmail = FakeGmail([
        _msg("m9", subject="Re: Hello",
             sender="Adaeze Okafor <ada@example.com>",
             snippet="Please confirm the slot works for you.",
             thread="t9"),
    ])
    monkeypatch.setattr(et, "_get_gmail", lambda context: gmail)
    queue_path = tmp_path / "email_triage.json"
    ctx = _ctx(queue_path)
    control_email("triage", ctx)
    pending = DraftQueue(path=queue_path).pending()
    assert len(pending) == 1
    assert pending[0].subject == "Re: Hello", (
        f"triage draft subject is {pending[0].subject!r} — the Re: "
        "prefix must not double up on an already-Re: subject")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
