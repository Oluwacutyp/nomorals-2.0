"""Tail-65 pins for nomorals/agents/monitor.py —
0 converts / 6 exclusions. Each test pins one site with a
control half proving the guarded operation was actually
ATTEMPTED via the double's own call log. The file contains
no exception handlers, so it adds zero census sites.

monitor.py calls no failures-ledger record function
anywhere (grep-proven), so NOT-recording is pinned against
the agents package ledger (nomorals/agents/failures.py):
recent_failures() snapshot before/after every trigger.

Verdict notes (full grounds in the tail-65 SURVEY.md):
- 122: session.close() in fetch_page_bytes' finally is
  best-effort teardown; the fetch result is already in hand.
- 426: the decode guard degrades only the stored diff base;
  hash change detection and the change record are unaffected.
- 460 / 598 / 614: notifier.publish is best-effort delivery
  (Notifier.publish itself is contracted "Never raises");
  the protected state — change entry, webhook result,
  auto-disable, alert stamp — is persisted independently
  and no return value claims delivery.
- 506: _decode_feed's docstring IS the contract — None
  means "found nothing / could not run"; enrichment is
  fail-soft and the consumer only attaches a truthy feed.
"""

from types import SimpleNamespace
from unittest import mock

from nomorals.agents import failures as agent_fail
from nomorals.agents.monitor import MonitorAgent, fetch_page_bytes


def _snap():
    return list(agent_fail.recent_failures(200))


class FakeDB:
    def __init__(self):
        self.calls = []
        self.executes = []

    def execute(self, sql, params=()):
        self.calls.append("execute")
        self.executes.append((sql, params))

    def query_one(self, *a):
        self.calls.append("query_one")
        return None

    def query(self, *a):
        self.calls.append("query")
        return []


class BadNotifier:
    def __init__(self):
        self.calls = []

    def publish(self, **kw):
        self.calls.append(kw)
        raise RuntimeError("publish dead")


def _row(**over):
    row = {"id": "m1", "target": "https://example.com/x", "kind": "url",
           "watch": "content", "last_hash": "", "last_size": 0,
           "last_content": "", "last_change_ts": 0.0, "error_streak": 0,
           "webhook_url": "", "webhook_secret": "", "min_alert_gap_s": 0.0,
           "last_alert_ts": 0.0, "auto_decode": 0, "volatile": ""}
    row.update(over)
    return row


def _agent(db=None, notifier=None):
    return MonitorAgent(SimpleNamespace(db=db or FakeDB()),
                        notifier=notifier)


def test_122_page_fetch_close_failure_keeps_bytes():
    before = _snap()

    class FakeSession:
        instances = []

        def __init__(self):
            self.closed = 0
            FakeSession.instances.append(self)

        def _fetch(self, target):
            return {"status": 200,
                    "text": "<html><body><p>hello page</p></body></html>"}

        def close(self):
            self.closed += 1
            raise RuntimeError("close dead")

    with mock.patch("nomorals.tools.browser.BrowserSession", FakeSession):
        out = fetch_page_bytes("https://example.com/")
    assert FakeSession.instances[0].closed == 1
    assert b"hello page" in out
    assert _snap() == before


def test_426_note_check_decode_failure_stores_empty():
    before = _snap()

    class BadBytes(bytes):
        decode_calls = 0

        def decode(self, *a, **k):
            BadBytes.decode_calls += 1
            raise RuntimeError("decode dead")

    db = FakeDB()
    agent = _agent(db)
    row = _row(last_hash="0" * 64, last_content="old text", last_size=8)
    changed = []
    agent._note_check(row, BadBytes(b"hello world"), 100.0, changed)
    assert BadBytes.decode_calls == 1
    upd = [p for s, p in db.executes if "last_ts" in s]
    assert upd and upd[0][3] == ""  # stored content param
    assert len(changed) == 1 and "diff" not in changed[0]
    assert _snap() == before


def test_460_change_alert_publish_failure_keeps_entry():
    before = _snap()
    db = FakeDB()
    notifier = BadNotifier()
    agent = _agent(db, notifier=notifier)
    row = _row(last_hash="0" * 64, last_content="old text", last_size=8)
    changed = []
    agent._note_check(row, b"new text here", 100.0, changed)
    assert len(notifier.calls) == 1
    assert len(changed) == 1
    assert changed[0]["webhook"] == {"fired": False,
                                     "reason": "no webhook configured"}
    assert _snap() == before


def test_506_decode_feed_failure_returns_none():
    before = _snap()
    calls = []

    def bad_analyze(*a, **k):
        calls.append(1)
        raise RuntimeError("decoder dead")

    agent = _agent()
    with mock.patch("nomorals.core.decoder.analyze", bad_analyze):
        out = agent._decode_feed(_row(), b"some content", 123.0)
    assert out is None
    assert calls == [1]
    assert _snap() == before


def test_598_auto_disable_publish_failure_keeps_disable():
    before = _snap()
    db = FakeDB()
    notifier = BadNotifier()
    agent = _agent(db, notifier=notifier)
    agent._note_error(_row(error_streak=9), "boom", now=1000.0)
    assert len(notifier.calls) == 1
    assert any("enabled=0" in s for s, _p in db.executes)
    assert _snap() == before


def test_614_watch_down_publish_failure_keeps_stamp():
    before = _snap()
    db = FakeDB()
    notifier = BadNotifier()
    agent = _agent(db, notifier=notifier)
    agent._note_error(_row(error_streak=2), "boom", now=1000.0)
    assert len(notifier.calls) == 1
    assert any("last_alert_ts" in s for s, _p in db.executes)
    assert _snap() == before
