"""Tail-85 pins for nomorals/agents/watchers.py — 5 exclusions, 0 converts.

Each test pins one census site's fallback behavior plus a control half
proving the except path was taken. The file itself contains no
exception handlers, so it adds zero census sites, and no sys.path
manipulation — tests bind via rootdir from the tree root.

NOT-recording is pinned with the snapshot method: watchers.py calls
no failures-ledger record function anywhere (only _log), so both the
agency and agents ledgers must be unchanged across each trigger.

Verdict notes (full grounds in SURVEY.md):
- 493: _semantic_hash extraction failure -> ("", False). The False
  flag is the honest signal: UrlKind.check falls back to the raw
  body hash and records "semantic": False in the watcher's value
  and evidence. Designed ladder, truthfully consumed.
- 723: _check_market payload failure -> WatchResult(changed=False,
  error="no usd quote for BTC"). An explicit error result feeding
  the sweep's error streak — refusal-shaped, never a price claim.
- 1860 / 1894: _value_hash failure -> "" in the sweep / in the
  helper. "" is the no-hash value (the rate-limit-deferred sweep
  path writes "" too) and the hash is persisted only into
  check-history audit rows — read back for display, never compared
  for a decision.
- 1919: ensure_sweeper_job list_jobs failure -> have = [] and the
  add path runs. The documented trigger is the scheduler table not
  existing yet (first boot), where "no jobs" is literally true; a
  genuinely broken DB still surfaces at the UNGUARDED add() call.
  The batch's weakest exclusion — see SURVEY.md.
"""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.agents import watchers as W
from nomorals.storage.db import Database


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def test_493_semantic_hash_failure_reports_not_semantic():
    before = _snap()
    body = b"<html><body><p>hello watched world</p></body></html>"
    digest, ok = W._semantic_hash(body)
    assert ok is True and len(digest) == 64
    with mock.patch("nomorals.tools.browser.node_to_markdown",
                    side_effect=RuntimeError("render boom")):
        digest2, ok2 = W._semantic_hash(body)
    assert (digest2, ok2) == ("", False)
    assert _snap() == before


class _FakeResp:
    def __init__(self, status, body):
        self.status = status
        self.body = body


def _http_client(body, status=200):
    class _Client:
        def __init__(self, *a, **k):
            pass

        def get(self, url, headers=None, timeout=None):
            return _FakeResp(status, body)
    return _Client


def _market_watcher():
    return W.Watcher(
        id="w1", name="btc", kind="price",
        target={"source": "market", "symbol": "BTC", "market": "crypto",
                "currency": "usd"},
        condition=W.parse_condition(
            {"op": "lt", "field": "value", "value": 60000}))


def test_723_market_bad_payload_is_error_result():
    before = _snap()
    kind = W.PriceKind()
    wctx = W.WatcherContext(context=SimpleNamespace(db=None), store=None)
    with mock.patch("nomorals.core.http.HttpClient",
                    _http_client(b"not json at all")):
        res = kind.check(wctx, _market_watcher())
    # bad payload -> explicit ERROR result, never a price claim
    assert res.changed is False
    assert res.error == "no usd quote for BTC"
    assert res.new_value is None
    # control: a valid payload yields the real quote (fresh context —
    # the per-host 60s gap is per-WatcherContext state)
    body = json.dumps({"bitcoin": {"usd": 50000.0}}).encode()
    wctx = W.WatcherContext(context=SimpleNamespace(db=None), store=None)
    with mock.patch("nomorals.core.http.HttpClient", _http_client(body)):
        res = kind.check(wctx, _market_watcher())
    assert not res.error
    assert res.new_value["price"] == 50000.0
    assert _snap() == before


class _FakeNotifier:
    def __init__(self):
        self.sent = []

    def publish(self, kind, title, body="", **kw):
        self.sent.append({"kind": kind, "title": title, "body": body})
        return {"id": "n1", "delivered": True}


def _watcher_agent(tmp):
    db = Database(os.path.join(tmp, "w.db"))
    db.migrate()
    context = SimpleNamespace(
        db=db,
        settings=SimpleNamespace(workspace_dir=tmp, timezone="UTC"),
        gateway=None, tools=None)
    return W.WatcherAgent(context, notifier=_FakeNotifier(),
                          registry=None, now=lambda: 1_700_000_000.0)


def test_1860_sweep_survives_value_hash_failure():
    before = _snap()
    agent = _watcher_agent(tempfile.mkdtemp())
    res = agent.add("tell me if BTC drops below $60000")
    assert res["ok"], res
    wid = res["watcher"]["id"]
    body = json.dumps({"bitcoin": {"usd": 59000.0}}).encode()

    def _boom_hash(value):
        raise RuntimeError("hash boom")

    with mock.patch("nomorals.core.http.HttpClient", _http_client(body)):
        with mock.patch.object(W.WatcherAgent, "_value_hash",
                               staticmethod(_boom_hash)):
            out = agent.sweep(now=1_700_000_000.0)
    # the sweep completes; the history row carries the no-hash value
    assert out["checked"] == 1
    hist = agent.store.check_history(wid)
    assert hist and hist[0]["value_hash"] == ""
    # control: unpatched, the same sweep records a real 16-char hash
    agent2 = _watcher_agent(tempfile.mkdtemp())
    res2 = agent2.add("tell me if BTC drops below $60000")
    assert res2["ok"], res2
    wid2 = res2["watcher"]["id"]
    with mock.patch("nomorals.core.http.HttpClient", _http_client(body)):
        out2 = agent2.sweep(now=1_700_000_000.0)
    assert out2["checked"] == 1
    hist2 = agent2.store.check_history(wid2)
    assert hist2 and len(hist2[0]["value_hash"]) == 16
    assert _snap() == before


def test_1894_value_hash_unserializable_is_empty():
    before = _snap()
    h = W.WatcherAgent._value_hash({"a": 1})
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)
    circ = {}
    circ["self"] = circ
    assert W.WatcherAgent._value_hash(circ) == ""
    assert _snap() == before


def test_1919_ensure_sweeper_job_list_failure_adds():
    before = _snap()
    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "s.db"))
    db.migrate()
    context = SimpleNamespace(
        db=db, settings=SimpleNamespace(workspace_dir=tmp,
                                        timezone="UTC"),
        gateway=None, tools=None)
    # control: first call schedules, second is idempotent by name
    first = W.ensure_sweeper_job(context)
    assert first.get("scheduled") is True
    second = W.ensure_sweeper_job(context)
    assert second.get("already_scheduled") is True
    assert second.get("job_id") == first.get("job_id")
    # trigger: list_jobs fails -> treated as "no job yet", add runs
    added = []

    def _fake_add(self, name, schedule, kind, payload, routine=False):
        added.append(name)
        return {"id": "job-x"}

    with mock.patch("nomorals.agents.scheduler.Scheduler.list_jobs",
                    side_effect=RuntimeError("list boom")):
        with mock.patch("nomorals.agents.scheduler.Scheduler.add",
                        _fake_add):
            out = W.ensure_sweeper_job(context)
    assert added == [W.SWEEPER_JOB_NAME]
    assert out == {"name": W.SWEEPER_JOB_NAME, "scheduled": True,
                   "job_id": "job-x"}
    assert _snap() == before
