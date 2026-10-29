"""Tail-38 pins for nomorals/agents/email_triage.py — 8 exclusions, 0 converts.

Each test pins one site's fallback behavior plus a control half proving the
except path was taken (trigger == fallback AND control != fallback, except
the documented default-equivalence cases 233/285/378, where the control
proves the fallback EQUALS the documented no-input value). The file itself
contains no exception handlers, so it adds zero census sites.

email_triage.py calls no record function anywhere (grep-proven: only _log),
so NOT-recording is pinned with the tail-8 snapshot method:
agency recent_failures() before/after each trigger must be unchanged — a
swallowed failure leaves no trace anywhere. (The module's own _log calls
are not failure records.)

Verdict notes (full grounds in SURVEY.md):
- 233: date-parse failure -> date_ts 0.0, which IS the field default and
  the empty-date value; date_ts has zero readers outside this module
  (write-only metadata). Nothing is misled.
- 285: prefs-read failure -> default hints, byte-identical to the
  no-preferences value; the draft is [draft]-marked and review-gated,
  never auto-sent. Default tone is not a claim about the owner's voice.
- 301: identity-read failure -> hints without owner_name: an omission in
  a review-gated draft signoff, never a false name. Narrow trigger
  (raising dict-get / raising value property).
- 378: FLAGGED for parent adjudication (weakest). settings-resolve
  failure -> the documented default path (same value as no-context);
  availability-preserving, since DraftQueue is built for every /email
  subcommand. Caveat recorded: a transient resolve failure splits the
  store across runs (drafts written to default during the window become
  invisible once resolve works again). No rendered false claim.
- 404: documented "one bad draft, skip" contract (tail-30 945 family);
  the skipped entry is unparseable, so no valid draft is hidden, and
  the queue never claims it exists.
- 652: count matches bullets shown; the all-failed case gets an honest
  line; no claim of completeness. Partial-render family (tail-30).
- 809: a corrupt part contributes nothing; later good parts still win;
  when nothing decodes, "(no readable body)" is true. Attachments are
  inventoried before decode, never affected.
- 891: explicitly labelled "unparsable" with the id kept; the owner can
  /email read it in full. Self-describing fallback.
"""

import base64
import json
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents import failures as ag_fail
from nomorals.agents.email_triage import (
    DraftQueue,
    _extract_body,
    _item_from_message,
    _list_messages,
    _store_path,
    _voice_hints,
    answer_vendor_query,
)


def _snap():
    return list(ag_fail.recent_failures())


def _msg(mid, subject="Hi", sender="a@b.com", date=None, snippet="hello"):
    headers = [{"name": "Subject", "value": subject},
               {"name": "From", "value": sender}]
    if date is not None:
        headers.append({"name": "Date", "value": date})
    return {"id": mid, "threadId": "t1", "snippet": snippet,
            "payload": {"headers": headers}}


def _b64(s):
    return base64.urlsafe_b64encode(s.encode()).decode()


class _RaisingPrefs:
    @property
    def preferences(self):
        raise RuntimeError("boom")


class _BadGet(dict):
    def get(self, k, d=None):
        raise RuntimeError("boom")


class _RaisingSettings:
    def resolve(self, p):
        raise RuntimeError("boom")


class _VendorGmail:
    def __init__(self, good_ids):
        self.good_ids = good_ids

    def list_messages(self, query="", max_results=5):
        return {"messages": [{"id": "a"}, {"id": "b"}]}

    def get_message(self, mid, format="metadata"):
        if mid not in self.good_ids:
            raise RuntimeError("fetch boom")
        return _msg(mid, subject=f"Subj {mid}", snippet="we ship widgets")


class _ListGmail:
    def list_messages(self, query="", max_results=10):
        return {"messages": [{"id": "good"}, {"id": "bad"}]}

    def get_message(self, mid, format="metadata"):
        if mid == "bad":
            return {"id": "bad", "snippet": "x",
                    "payload": {"headers": "not-a-list"}}
        return _msg(mid, subject="Good", sender="g@x.com", snippet="snip")


def test_233_bad_date_keeps_zero_and_records_nothing():
    before = _snap()
    bad = _item_from_message(_msg("m1", date="not-a-date"))
    assert bad.date_ts == 0.0
    assert _snap() == before
    good = _item_from_message(
        _msg("m2", date="Mon, 01 Jan 2024 00:00:00 +0000"))
    assert good.date_ts == 1704067200.0
    assert _item_from_message(_msg("m3")).date_ts == 0.0


def test_285_prefs_failure_returns_defaults_and_records_nothing():
    before = _snap()
    h = _voice_hints(_RaisingPrefs())
    assert h == {"brevity": "short", "formality": "casual", "signoff": ""}
    assert _snap() == before
    h2 = _voice_hints(SimpleNamespace(
        preferences=[SimpleNamespace(value="formal"),
                     SimpleNamespace(value="brief")], identity={}))
    assert h2["formality"] == "formal" and h2["brevity"] == "terse"
    assert _voice_hints(None) == {"brevity": "short", "formality": "casual",
                                  "signoff": ""}


def test_301_identity_failure_drops_owner_name_only():
    before = _snap()
    h = _voice_hints(SimpleNamespace(preferences=[],
                                     identity=_BadGet({"name": "x"})))
    assert "owner_name" not in h
    assert _snap() == before
    h2 = _voice_hints(SimpleNamespace(
        preferences=[], identity={"name": SimpleNamespace(value="Ada")}))
    assert h2.get("owner_name") == "Ada"


def test_378_resolve_failure_falls_back_to_default_path():
    before = _snap()
    p = _store_path(SimpleNamespace(settings=_RaisingSettings()))
    assert str(p).endswith(".devon/email_triage.json")
    assert _snap() == before
    p2 = _store_path(SimpleNamespace(settings=SimpleNamespace(
        resolve=lambda q: Path("/tmp/ctx/email_triage.json"))))
    assert str(p2) == "/tmp/ctx/email_triage.json"
    assert str(_store_path(None)).endswith(".devon/email_triage.json")


def test_404_bad_draft_skipped_good_draft_loads(tmp_path):
    store = tmp_path / "q.json"
    store.write_text(json.dumps({
        "drafts": [
            {"id": "good1", "thread_id": "t", "to": "a@b.com",
             "subject": "S", "body": "B", "created_ts": 1.0,
             "status": "pending"},
            {"id": "bad1", "thread_id": "t", "to": "a@b.com",
             "subject": "S", "body": "B", "created_ts": "not-a-number",
             "status": "pending"},
        ],
        "thread_activity": {}, "pending_send": ""}), encoding="utf-8")
    before = _snap()
    q = DraftQueue(path=store)
    assert q.get("good1") is not None and q.get("bad1") is None
    assert _snap() == before
    store2 = tmp_path / "q2.json"
    store2.write_text(json.dumps({
        "drafts": [
            {"id": "g1", "thread_id": "t", "to": "a@b.com", "subject": "S",
             "body": "B", "created_ts": 1.0, "status": "pending"},
            {"id": "g2", "thread_id": "t", "to": "a@b.com", "subject": "S",
             "body": "B", "created_ts": 2.0, "status": "pending"},
        ],
        "thread_activity": {}, "pending_send": ""}), encoding="utf-8")
    q2 = DraftQueue(path=store2)
    assert q2.get("g1") is not None and q2.get("g2") is not None


def test_652_vendor_skip_keeps_count_honest():
    before = _snap()
    r = answer_vendor_query(_VendorGmail({"a"}),
                            "what did acme say about widgets?")
    assert "(1 thread)" in r and "Subj a" in r and "Subj b" not in r
    assert _snap() == before
    r2 = answer_vendor_query(_VendorGmail({"a", "b"}),
                             "what did acme say about widgets?")
    assert "(2 threads)" in r2
    r3 = answer_vendor_query(_VendorGmail(set()),
                             "what did acme say about widgets?")
    assert r3 == ("found threads from acme about widgets but couldn't "
                  "read them.")


def test_809_bad_part_skipped_good_part_wins():
    before = _snap()
    bad_plain = {"mimeType": "text/plain", "filename": "",
                 "body": {"data": "!!!"}, "parts": []}
    good_html = {"mimeType": "text/html", "filename": "",
                 "body": {"data": _b64("<p>Hello</p>")}, "parts": []}
    text, _ = _extract_body({"parts": [bad_plain, good_html]})
    assert text == "Hello"
    assert _snap() == before
    good_plain = {"mimeType": "text/plain", "filename": "",
                  "body": {"data": _b64("Plain wins")}, "parts": []}
    text2, _ = _extract_body({"parts": [good_plain, good_html]})
    assert text2 == "Plain wins"
    text3, _ = _extract_body({"parts": [bad_plain]})
    assert text3 == ""


def test_891_unparsable_labelled_with_id_kept():
    before = _snap()
    out = _list_messages(_ListGmail(), "in:inbox", 10)
    assert "• bad — unparsable" in out
    assert "• `good` — g@x.com — Good" in out
    assert _snap() == before
