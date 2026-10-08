"""Tests for the AI email triage module (build-map #11). All offline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.email_triage import (
    CATEGORY_ACTION,
    CATEGORY_FYI,
    CATEGORY_URGENT,
    Draft,
    DraftQueue,
    EmailItem,
    TriageReport,
    answer_vendor_query,
    check_explicit_send,
    classify_item,
    control_email,
    draft_reply,
    followup_watch,
    is_explicit_send,
    triage,
)


# ── fake gmail ───────────────────────────────────────────────────────────────

def _msg(mid, *, subject, sender, snippet, thread="t1", date="Mon, 06 Oct 2026 10:00:00 +0000"):
    return {
        "id": mid,
        "threadId": thread,
        "snippet": snippet,
        "labelIds": ["INBOX", "UNREAD"],
        "payload": {"headers": [
            {"name": "Subject", "value": subject},
            {"name": "From", "value": sender},
            {"name": "Date", "value": date},
        ]},
    }


class FakeGmail:
    """Canned connector: no network, records sends."""

    def __init__(self, messages):
        self._messages = {m["id"]: m for m in messages}
        self.sent: list[dict] = []
        self.read_marked: list[str] = []

    def list_messages(self, *, query="", max_results=50, **kw):
        msgs = list(self._messages.values())
        if "is:unread" in query:
            msgs = [m for m in msgs if "UNREAD" in m.get("labelIds", [])]
        if query.startswith("from:"):
            rest = query[len("from:"):]
            vendor = rest.split()[0].lower()
            topic = " ".join(rest.split()[1:]).lower()
            msgs = [m for m in msgs
                    if vendor in str(m["payload"]["headers"][1]["value"]).lower()
                    and topic in (str(m["payload"]["headers"][0]["value"])
                                  + " " + m["snippet"]).lower()]
        return {"messages": [{"id": m["id"], "threadId": m["threadId"]}
                             for m in msgs[:max_results]],
                "next_page_token": "", "result_size_estimate": len(msgs)}

    def get_message(self, message_id, *, format="metadata"):  # noqa: A002
        return self._messages[message_id]

    def get_thread(self, thread_id):
        msgs = [m for m in self._messages.values()
                if m["threadId"] == thread_id]
        return {"id": thread_id, "messages": msgs}

    def send_message(self, to, subject, body, **kw):
        if not kw.get("confirmed"):
            raise PermissionError("not confirmed")
        self.sent.append({"to": to, "subject": subject, "body": body})
        return {"id": "sent1", "threadId": "t_sent"}


@pytest.fixture()
def gmail():
    return FakeGmail([
        _msg("m1", subject="URGENT: card declined",
             sender="billing@stripe.com",
             snippet="Your payment failed. Update your card immediately.",
             thread="t1"),
        _msg("m2", subject="Contract review",
             sender="Adaeze Okafor <ada@example.com>",
             snippet="Could you please review the attached contract?",
             thread="t2"),
        _msg("m3", subject="50% off everything",
             sender="noreply@shop.com",
             snippet="This week's deals are here.",
             thread="t3"),
    ])


@pytest.fixture()
def queue_path(tmp_path: Path):
    return tmp_path / "email_triage.json"


# ── classification ───────────────────────────────────────────────────────────

def test_classify_urgent():
    cat, urg, reply = classify_item("URGENT: payment failed",
                                    "billing@x.com",
                                    "your card was declined")
    assert cat == CATEGORY_URGENT
    assert urg >= 0.4
    assert reply is True


def test_classify_action():
    cat, urg, reply = classify_item("Contract review", "ada@x.com",
                                    "Could you please review this?")
    assert cat == CATEGORY_ACTION
    assert reply is True


def test_classify_fyi_bulk():
    cat, urg, reply = classify_item("Weekly deals", "noreply@shop.com",
                                    "50% off")
    assert cat == CATEGORY_FYI
    assert urg <= 0.25
    assert reply is False


def test_classify_personal_no_ask_is_fyi():
    cat, _, reply = classify_item("hello", "mom@x.com", "just saying hi")
    assert cat == CATEGORY_FYI
    assert reply is False


# ── triage ───────────────────────────────────────────────────────────────────

def test_triage_classifies_and_sorts(gmail):
    report = triage(gmail)
    assert len(report.items) == 3
    assert report.items[0].category == CATEGORY_URGENT  # sorted by urgency
    assert len(report.urgent) == 1
    assert len(report.action) == 1
    assert len(report.fyi) == 1
    assert report.drafts_wanted  # urgent + action need replies


def test_triage_never_marks_read(gmail):
    triage(gmail)
    assert gmail.read_marked == []


def test_triage_connector_down_returns_empty():
    class Down:
        def list_messages(self, **kw):
            raise ConnectionError("nope")
    report = triage(Down())
    assert report.items == []


def test_triage_respects_max(gmail):
    report = triage(gmail, max_messages=1)
    assert len(report.items) == 1


# ── drafts ───────────────────────────────────────────────────────────────────

def _persona(*, blob=""):
    prefs = [SimpleNamespace(value=blob)] if blob else []
    return SimpleNamespace(preferences=prefs, identity={})


def test_draft_reply_template_offline():
    item = EmailItem(id="m2", thread_id="t2", subject="Contract review",
                     sender="Adaeze Okafor <ada@example.com>",
                     snippet="Could you please review this?",
                     category=CATEGORY_ACTION, needs_reply=True)
    body = draft_reply(item, _persona())
    assert body.startswith("[draft]")
    assert "Adaeze" in body
    assert len(body) < 400


def test_draft_reply_formal_voice():
    item = EmailItem(id="m1", thread_id="t1", subject="Invoice #42",
                     sender="billing@corp.com", snippet="invoice attached",
                     category=CATEGORY_ACTION, needs_reply=True)
    body = draft_reply(item, _persona(blob="I prefer formal professional tone"))
    assert "Thank you" in body or "Hello" in body


def test_draft_reply_invoice_template():
    item = EmailItem(id="m1", thread_id="t1", subject="Invoice due",
                     sender="billing@x.com", snippet="pay now",
                     category=CATEGORY_URGENT, needs_reply=True)
    body = draft_reply(item, _persona()).lower()
    assert "sort this out" in body or "take care" in body


def test_queue_round_trip(queue_path):
    q = DraftQueue(path=queue_path)
    d = q.add(thread_id="t1", to="a@x.com", subject="Re: hi", body="yo")
    assert d.status == "pending"
    assert q.get(d.id).to == "a@x.com"
    # persist + reload
    q2 = DraftQueue(path=queue_path)
    assert q2.get(d.id) is not None
    assert q2.approve(d.id) is True
    assert q2.get(d.id).status == "approved"


def test_queue_send_requires_approval(queue_path):
    q = DraftQueue(path=queue_path)
    gmail = FakeGmail([])
    d = q.add(thread_id="t1", to="a@x.com", subject="s", body="b")
    with pytest.raises(PermissionError):
        q.send(gmail, d.id)
    assert gmail.sent == []
    q.approve(d.id)
    q.send(gmail, d.id)
    assert len(gmail.sent) == 1
    assert q.get(d.id).status == "sent"


def test_queue_send_confirmed_override(queue_path):
    q = DraftQueue(path=queue_path)
    gmail = FakeGmail([])
    d = q.add(thread_id="t1", to="a@x.com", subject="s", body="b")
    q.send(gmail, d.id, confirmed=True)  # explicit override
    assert gmail.sent and q.get(d.id).status == "sent"


def test_queue_discard(queue_path):
    q = DraftQueue(path=queue_path)
    d = q.add(thread_id="t1", to="a@x.com", subject="s", body="b")
    assert q.discard(d.id) is True
    with pytest.raises(ValueError):
        q.send(FakeGmail([]), d.id, confirmed=True)


def test_queue_double_send_refused(queue_path):
    q = DraftQueue(path=queue_path)
    gmail = FakeGmail([])
    d = q.add(thread_id="t1", to="a@x.com", subject="s", body="b")
    q.approve(d.id)
    q.send(gmail, d.id)
    with pytest.raises(ValueError):
        q.send(gmail, d.id)
    assert len(gmail.sent) == 1


# ── explicit send ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "send it", "Send It", "send it!", "send all", "send them",
    "send the drafts", "reply now", "send now", "yes send",
    "yes, send it", "please send it", "go ahead and send",
    "looks good. send it",
])
def test_explicit_send_matches(text):
    assert is_explicit_send(text) is True


@pytest.mark.parametrize("text", [
    "should I send it?", "maybe send it later", "don't send it",
    "send it please now and tell everyone about it in detail",  # too long
    "", "send", "ok",
])
def test_explicit_send_rejects(text):
    assert is_explicit_send(text) is False


def test_check_explicit_send_sends_armed_draft(queue_path):
    gmail = FakeGmail([])
    q = DraftQueue(path=queue_path)
    d = q.add(thread_id="t1", to="a@x.com", subject="s", body="b")
    q.approve(d.id)
    q.mark_pending_send(d.id)
    ctx = SimpleNamespace(db=None, settings=None)
    # check_explicit_send builds its own queue from context; point it at
    # our file via a settings stub
    settings = SimpleNamespace(
        resolve=lambda rel: queue_path if "email_triage" in rel else Path(rel))
    ctx = SimpleNamespace(db=None, settings=settings)
    import nomorals.agents.email_triage as et
    orig = et._get_gmail
    et._get_gmail = lambda context: gmail  # noqa: E731
    try:
        reply = check_explicit_send("send it", ctx)
    finally:
        et._get_gmail = orig
    assert reply and "sent" in reply
    assert gmail.sent
    # reload: check_explicit_send used its own queue instance on the file
    assert DraftQueue(path=queue_path).get(d.id).status == "sent"


def test_check_explicit_send_nothing_armed(queue_path):
    settings = SimpleNamespace(resolve=lambda rel: queue_path)
    ctx = SimpleNamespace(db=None, settings=settings)
    assert check_explicit_send("send it", ctx) is None
    assert check_explicit_send("hello there", ctx) is None


# ── follow-up watch ──────────────────────────────────────────────────────────

def test_followup_flags_quiet_thread(queue_path):
    gmail = FakeGmail([
        _msg("m1", subject="Proposal", sender="bob@x.com",
             snippet="thoughts?", thread="t9",
             date="Mon, 05 Oct 2026 10:00:00 +0000"),
    ])
    # internalDate in ms — set the message older than the owner action
    old = (time.time() - 49 * 3600) * 1000
    gmail._messages["m1"]["internalDate"] = str(int(old - 3600 * 1000))
    q = DraftQueue(path=queue_path)
    q.note_owner_action("t9", time.time() - 49 * 3600)
    fus = followup_watch(gmail, q, hours=48)
    assert len(fus) == 1
    assert fus[0].thread_id == "t9"
    assert fus[0].hours_quiet >= 48
    assert fus[0].nudge_draft.startswith("[draft]")


def test_followup_ignores_recent_thread(queue_path):
    gmail = FakeGmail([
        _msg("m1", subject="Proposal", sender="bob@x.com",
             snippet="thoughts?", thread="t9"),
    ])
    q = DraftQueue(path=queue_path)
    q.note_owner_action("t9", time.time() - 47 * 3600)  # under 48h
    assert followup_watch(gmail, q, hours=48) == []


def test_followup_thread_alive_after_owner(queue_path):
    now = time.time()
    gmail = FakeGmail([
        _msg("m1", subject="Proposal", sender="bob@x.com",
             snippet="replying!", thread="t9"),
    ])
    # their reply arrived AFTER our action → thread alive
    gmail._messages["m1"]["internalDate"] = str(int(now * 1000))
    q = DraftQueue(path=queue_path)
    q.note_owner_action("t9", now - 49 * 3600)
    assert followup_watch(gmail, q, hours=48, now=now) == []


# ── NL vendor query ──────────────────────────────────────────────────────────

def test_vendor_query_searches_and_synthesizes():
    gmail = FakeGmail([
        _msg("m1", subject="Acme pricing tiers",
             sender="sales@acme.com",
             snippet="Our Pro plan is $49/mo with annual billing.",
             thread="t1"),
    ])
    out = answer_vendor_query(gmail, "what did Acme say about pricing?")
    assert out is not None
    assert "$49" in out
    assert "Acme" in out


def test_vendor_query_no_results():
    gmail = FakeGmail([])
    out = answer_vendor_query(gmail, "what did Acme say about pricing?")
    assert "nothing" in out


def test_vendor_query_not_a_query():
    gmail = FakeGmail([])
    assert answer_vendor_query(gmail, "how's the weather?") is None
    assert answer_vendor_query(gmail, "what did you say about pricing?") is None


# ── chat command ─────────────────────────────────────────────────────────────

def _ctx(gmail, queue_path):
    settings = SimpleNamespace(
        resolve=lambda rel: queue_path if "email_triage" in rel else Path(rel))
    return SimpleNamespace(db=None, settings=settings, persona=None), gmail


def test_control_email_triage(gmail, queue_path, monkeypatch):
    import nomorals.agents.email_triage as et
    monkeypatch.setattr(et, "_get_gmail", lambda context: gmail)
    ctx, _ = _ctx(gmail, queue_path)
    out = control_email("triage", ctx)
    assert "urgent" in out and "1 urgent" in out
    assert "draft" in out.lower()


def test_control_email_drafts_and_send(gmail, queue_path, monkeypatch):
    import nomorals.agents.email_triage as et
    monkeypatch.setattr(et, "_get_gmail", lambda context: gmail)
    ctx, _ = _ctx(gmail, queue_path)
    control_email("triage", ctx)  # queues drafts
    out = control_email("drafts", ctx)
    assert "pending draft" in out
    q = DraftQueue(path=queue_path)
    did = q.pending()[0].id
    preview = control_email(f"send {did}", ctx)
    assert "ready to send" in preview
    assert "send it" in preview
    # the slash form never sends by itself
    assert gmail.sent == []
    assert q.get(did).status == "pending"


def test_control_email_followups(gmail, queue_path, monkeypatch):
    import nomorals.agents.email_triage as et
    monkeypatch.setattr(et, "_get_gmail", lambda context: gmail)
    ctx, _ = _ctx(gmail, queue_path)
    out = control_email("followups", ctx)
    assert "no forgotten threads" in out


def test_control_email_not_connected(queue_path, monkeypatch):
    import nomorals.agents.email_triage as et
    monkeypatch.setattr(et, "_get_gmail", lambda context: None)
    ctx = SimpleNamespace(db=None, settings=None, persona=None)
    out = control_email("triage", ctx)
    assert "isn't connected" in out


def test_control_email_usage(queue_path, monkeypatch):
    import nomorals.agents.email_triage as et
    monkeypatch.setattr(et, "_get_gmail", lambda context: FakeGmail([]))
    ctx = SimpleNamespace(db=None, settings=None, persona=None)
    assert "usage" in control_email("frobnicate", ctx)


# ── coremind intent ──────────────────────────────────────────────────────────

def test_email_intent_matches_vendor_query():
    from nomorals.agents.coremind import understand
    intents = understand("what did Acme Corp say about pricing?")
    kinds = [i.kind for i in intents]
    assert "email_query" in kinds
    eq = next(i for i in intents if i.kind == "email_query")
    assert eq.meta["vendor"] == "Acme Corp"
    assert eq.meta["topic"] == "pricing"


def test_email_intent_rejects_pronouns():
    from nomorals.agents.coremind import understand
    intents = understand("what did you say about pricing?")
    assert "email_query" not in [i.kind for i in intents]


def test_email_intent_rejects_chat():
    from nomorals.agents.coremind import understand
    intents = understand("did you see the game last night?")
    assert "email_query" not in [i.kind for i in intents]
