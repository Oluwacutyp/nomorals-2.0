"""Formatting Pass F3 — the job-report renderer.

Pins:

* the owner's finance-alert screenshot dict becomes a REPORT through
  :mod:`nomorals.agents.job_reports` — summary first, human labels,
  empty collections and noise fields gone, the two counters the job
  exists to report always visible;
* ``finance_digest``'s prose ``text`` leads with its first
  figure-bearing line (its own header line folds into the title);
* the generic fallback derives an honest summary (the tool's own
  headline field → the first visible facts → "finished — nothing to
  report") and never falls back to raw key flattening;
* the ``Field.list_limit`` option F3 added to the presentation layer
  caps scalar and dict lists with an explicit ``… (+N more)`` tail,
  and the default (0) keeps F2's whole-list behaviour;
* end-to-end through the real Scheduler: a tool job publishes a Doc
  (failures publish an error Doc naming the failure) while the stored
  ``last_result`` keeps its legacy string form byte-for-byte — the
  Pass #71/#75 contract (``startswith("finance_alert:")``,
  ``"fired: 0" in last_result``, the ``"job failed"`` prefix) is
  untouched; command jobs keep the legacy string publish.
"""

from __future__ import annotations

from types import SimpleNamespace

from nomorals.agents.job_reports import failure_doc, job_report_doc
from nomorals.agents.scheduler import Scheduler
from nomorals.social.chat.present import (Doc, Field, body_text,
                                          from_dict, render)
from nomorals.storage.db import Database

#: The owner's screenshot, verbatim (Pass #75's recorded shape of a
#: quiet finance_alert check run).
SCREENSHOT = {"ok": True, "delivered": True, "messages": [],
              "checked": 0, "fired": 0, "details": [],
              "skipped_snoozed": 0}


# ── finance_alert: the screenshot case ───────────────────────────────

def test_finance_alert_screenshot_becomes_a_report():
    doc = job_report_doc("finance_alert", dict(SCREENSHOT),
                         job_name="finance alerts")
    assert isinstance(doc, Doc)
    assert doc.tone == "ok"
    assert doc.summary == "No alerts checked — none fired."
    body = body_text(doc)
    # the counters the job exists to report are visible…
    assert "Checked" in body and "Fired" in body
    # …and none of the dump's noise survives: no empty collections,
    # no raw internal keys, no classified-noise fields.
    assert "messages" not in body
    assert "details" not in body
    assert "skipped_snoozed" not in body
    assert "[]" not in body
    assert "Delivered" not in body
    # the summary is the first thing after the title on Telegram too
    text = render(doc, "telegram-bot").text
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[1] == "No alerts checked — none fired."
    assert "<b>Checked</b>: 0" in text


def test_finance_alert_fired_report_carries_the_alert_texts():
    value = {"ok": True, "delivered": True,
             "messages": ["🔔 BTC price above 100.00\n"
                          "BTC is now 150.00 (≥ 100.00)"],
             "checked": 3, "fired": 1,
             "details": [{"id": "a1", "kind": "price_above",
                          "target": "BTC", "message": "BTC is now 150"}],
             "skipped_snoozed": 0}
    doc = job_report_doc("finance_alert", value, job_name="finance alerts")
    assert doc.summary == "1 alert fired of 3 alerts checked."
    body = body_text(doc)
    assert "Fired alerts" in body
    assert "BTC is now 150.00" in body
    # the detail section is expandable on Telegram
    text = render(doc, "telegram-bot").text
    assert "<blockquote expandable>" in text


def test_finance_alert_all_snoozed_summary():
    value = dict(SCREENSHOT, skipped_snoozed=2)
    doc = job_report_doc("finance_alert", value, job_name="finance alerts")
    assert doc.summary == "Nothing checked — 2 alerts snoozed, none fired."
    assert "Skipped (snoozed)" in body_text(doc)


def test_finance_alert_checked_none_fired_summary():
    value = dict(SCREENSHOT, checked=5)
    doc = job_report_doc("finance_alert", value, job_name="finance alerts")
    assert doc.summary == "5 alerts checked — none fired."


# ── finance_digest: prose payload ────────────────────────────────────

def test_finance_digest_report_leads_with_the_figures():
    text = ("💰 weekly money digest\n"
            "spent this week: ₦12,000\n"
            "vs last week (₦10,000): up 20%\n"
            "\n"
            "💸 safe to spend: ₦5,000 (expected ₦20,000/mo)")
    doc = job_report_doc("finance_digest",
                         {"ok": True, "delivered": True, "text": text},
                         job_name="finance digest")
    # the tool's own header line folds into the title; the first
    # figure-bearing line is the summary
    assert doc.summary == "spent this week: ₦12,000"
    body = body_text(doc)
    assert "weekly money digest" not in body
    assert "safe to spend: ₦5,000" in body
    assert "Delivered" in body  # the run's delivery fact is kept


# ── the generic fallback ─────────────────────────────────────────────

def test_generic_report_labels_and_facts_summary():
    doc = job_report_doc("pulse", {"ok": True, "sent": 2, "stories": 5,
                                   "errors": []},
                         job_name="morning pulse")
    body = body_text(doc)
    assert "Sent" in body and "Stories" in body
    assert "errors" not in body          # empty list hidden
    assert "[]" not in body
    assert "Sent: 2" in doc.summary       # summary from visible facts
    assert "Stories: 5" in doc.summary


def test_generic_report_quiet_result_says_so():
    doc = job_report_doc("watch", {"ok": True, "checked": 0, "sent": 0,
                                   "items": []},
                         job_name="watcher sweep")
    # zero counters are hidden, so nothing renders — and the summary
    # says exactly that instead of dumping zeros
    assert doc.summary == "Watch finished — nothing to report."
    assert "Checked" not in body_text(doc)


def test_generic_report_prefers_the_tools_own_headline():
    doc = job_report_doc("news", {"ok": True, "summary": "3 new posts",
                                  "found": 3},
                         job_name="news poll")
    assert doc.summary == "3 new posts"
    # the headline is not repeated as a row
    assert "Summary" not in body_text(doc)
    assert "Found" in body_text(doc)


def test_generic_report_long_prose_becomes_detail():
    prose = "\n".join(f"ledger line {i}" for i in range(50))
    doc = job_report_doc("books", {"ok": True, "count": 50,
                                   "text": prose},
                         job_name="bookkeeping")
    body = body_text(doc)
    assert "ledger line 0" in body
    assert "… (+10 more lines)" in body   # capped at 40, tail counted
    assert "ledger line 49" not in body


def test_string_result_leads_with_its_first_line():
    doc = job_report_doc("__immune_sampler__",
                         "sampled: no live process sample\n"
                         "baseline kept from the previous tick",
                         job_name="immune sampler")
    assert doc.summary == "sampled: no live process sample"
    assert "baseline kept" in body_text(doc)


def test_list_and_scalar_results():
    doc = job_report_doc("scan", ["alpha", "beta"], job_name="scan job")
    assert doc.summary == "2 items returned."
    assert "alpha" in body_text(doc)
    doc = job_report_doc("scan", None, job_name="scan job")
    assert doc.summary == "No output."


def test_failure_doc_names_the_failure():
    doc = failure_doc("finance alerts",
                      "job failed: tool finance_alert failed: boom")
    assert doc.tone == "error"
    assert doc.title == "scheduled: finance alerts"
    assert "boom" in doc.summary
    text = render(doc, "telegram-bot").text
    assert "❌" in text


# ── present.Field.list_limit (the F2 residue F3 inherited) ──────────

def test_list_limit_caps_scalar_lists_with_a_counted_tail():
    data = {"items": [f"item {i}" for i in range(15)]}
    doc = from_dict(data, [Field("items", list_limit=3)])
    body = body_text(doc)
    assert "item 0" in body and "item 2" in body
    assert "… (+12 more)" in body
    assert "item 14" not in body
    # default (no limit) keeps F2's whole-list behaviour
    doc = from_dict(data, [Field("items")])
    assert "item 14" in body_text(doc)


def test_list_limit_caps_dict_lists_with_a_counted_tail():
    data = {"rows": [{"n": i} for i in range(12)]}
    doc = from_dict(data, [Field("rows", list_limit=5)])
    body = body_text(doc)
    assert "… (+7 more)" in body


def test_generic_report_caps_long_lists():
    doc = job_report_doc("scan", {"ok": True,
                                  "items": [f"item {i}"
                                            for i in range(15)]},
                         job_name="scan job")
    body = body_text(doc)
    assert "item 9" in body
    assert "… (+5 more)" in body
    assert "item 14" not in body


# ── end-to-end through the real Scheduler ────────────────────────────

class _RecordingNotifier:
    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title, "body": body,
                           "kw": kw})
        return {"delivered": True, "channel": "test"}


class _Tools:
    """A registry double: one canned outcome for every call."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def call(self, tool, capabilities=None, **args):
        self.calls.append((tool, args))
        return self.outcome


def _ctx(tmp_path, tools):
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(
        db=db,
        settings=SimpleNamespace(home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={}, tools=tools)


def _sched(tmp_path, tools):
    sched = Scheduler(_ctx(tmp_path, tools))
    sched.notifier = _RecordingNotifier()
    return sched


def _ok(value):
    return SimpleNamespace(ok=True, value=value, error=None)


def _last_result(ctx, job_id):
    row = ctx.db.query_one(
        "SELECT last_result FROM schedule_jobs WHERE id = ?", (job_id,))
    return row["last_result"] or ""


def test_scheduler_publishes_doc_and_stores_legacy_string(tmp_path):
    tools = _Tools(_ok(dict(SCREENSHOT)))
    sched = _sched(tmp_path, tools)
    job = sched.add("finance alerts", "every 900s", "tool",
                    {"tool": "finance_alert",
                     "args": {"action": "check", "deliver": True}},
                    routine=False)
    out = sched.run_now(job["id"])
    assert out["ok"] is True, out
    # published: exactly one schedule alert, a Doc, the report content
    calls = [c for c in sched.notifier.calls if c["kind"] == "schedule"]
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["title"] == "✅ scheduled: finance alerts"
    doc = call["kw"].get("doc")
    assert isinstance(doc, Doc), call
    assert doc.summary == "No alerts checked — none fired."
    assert call["kw"].get("output_kind") == "scheduled"
    assert call["kw"].get("output_meta") == {"job": "finance alerts",
                                              "ok": True}
    # stored: the legacy string form, byte-for-byte the old contract
    last = _last_result(sched.context, job["id"])
    assert last.startswith("finance_alert:"), last
    assert "fired: 0" in last, last
    assert "skipped_snoozed: 0" in last, last


def test_scheduler_failed_job_publishes_error_doc(tmp_path):
    tools = _Tools(SimpleNamespace(
        ok=False, value=None,
        error=SimpleNamespace(message="kaboom: signature mismatch")))
    sched = _sched(tmp_path, tools)
    job = sched.add("job one", "every 1h", "tool",
                    {"tool": "wobble", "args": {}}, routine=False)
    out = sched.run_now(job["id"])
    assert out["ok"] is False, out
    calls = [c for c in sched.notifier.calls if c["kind"] == "schedule"]
    assert len(calls) == 1, calls
    assert calls[0]["title"] == "❌ scheduled: job one"
    doc = calls[0]["kw"].get("doc")
    assert isinstance(doc, Doc), calls[0]
    assert doc.tone == "error"
    assert doc.summary.startswith("job failed:"), doc.summary
    assert "kaboom: signature mismatch" in doc.summary
    # the stored record still leads with the failure prefix
    last = _last_result(sched.context, job["id"])
    assert last.startswith("job failed"), last
    assert "kaboom: signature mismatch" in last


def test_scheduler_declared_failure_publishes_error_doc(tmp_path):
    tools = _Tools(_ok({"ok": False, "error": "declared boom"}))
    sched = _sched(tmp_path, tools)
    job = sched.add("decl job", "every 1h", "tool",
                    {"tool": "wobble", "args": {}}, routine=False)
    out = sched.run_now(job["id"])
    assert out["ok"] is False, out
    calls = [c for c in sched.notifier.calls if c["kind"] == "schedule"]
    assert len(calls) == 1, calls
    doc = calls[0]["kw"].get("doc")
    assert isinstance(doc, Doc) and doc.tone == "error", calls[0]
    assert "declared boom" in doc.summary
    assert "declared boom" in _last_result(sched.context, job["id"])


def test_scheduler_command_success_keeps_string_publish(tmp_path):
    tools = _Tools(_ok({"exit_code": 0, "stdout": "hello out"}))
    sched = _sched(tmp_path, tools)
    job = sched.add("cmd job", "every 1h", "command",
                    {"command": "echo hi"}, routine=False)
    out = sched.run_now(job["id"])
    assert out["ok"] is True, out
    calls = [c for c in sched.notifier.calls if c["kind"] == "schedule"]
    assert len(calls) == 1, calls
    assert calls[0]["kw"].get("doc") is None
    assert "hello out" in calls[0]["body"]


def test_scheduler_string_tool_result_publishes_doc(tmp_path):
    tools = _Tools(_ok("sampled: no live process sample\n"
                       "baseline kept"))
    sched = _sched(tmp_path, tools)
    job = sched.add("immune sampler", "every 5m", "tool",
                    {"tool": "__immune_sampler__", "args": {}},
                    routine=False)
    out = sched.run_now(job["id"])
    assert out["ok"] is True, out
    calls = [c for c in sched.notifier.calls if c["kind"] == "schedule"]
    assert len(calls) == 1, calls
    doc = calls[0]["kw"].get("doc")
    assert isinstance(doc, Doc), calls[0]
    assert doc.summary == "sampled: no live process sample"
    # the stored record is the tool's string, untouched
    last = _last_result(sched.context, job["id"])
    assert "sampled: no live process sample" in last, last
