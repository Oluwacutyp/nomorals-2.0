"""Formatting Pass F4 — the autonomy digest renderer.

Pins the agency-shift digest (:mod:`nomorals.agents.job_reports`,
the ``__agency_shift__`` hand builder):

* the owner's screenshot case — a ``work_shift()`` result whose raw
  dump printed ``knowledge_items: 121`` repeatedly (shift + nested
  cycle + expansion) and both cycles' counters as bare keys —
  becomes a digest Doc: one summary sentence the data supports,
  per-system ran/stood-down lines, each fact rendered exactly once;
* the knowledge-base size is the FRESHEST copy (the expansion's,
  read last) and appears exactly once in every rendering;
* stood-down systems are named only when the names reconcile with
  the reported count (a with-work system can miss its turn on a
  rotation cooldown — the digest never guesses a name);
* a fully stood-down shift reads as a quiet shift, not an error;
  a missing expansion (the tool swallows its failure into None)
  is named plainly and carries the warn tone;
* end-to-end through the real Scheduler: the published alert is
  the digest Doc while the stored ``last_result`` keeps its legacy
  string form — the contract the Pass #71/#75 suites pin.
"""

from __future__ import annotations

from types import SimpleNamespace

from nomorals.agents.job_reports import job_report_doc
from nomorals.agents.scheduler import Scheduler
from nomorals.social.chat.present import Doc, body_text, render
from nomorals.storage.db import Database

#: sorted(EXPANSION_SYSTEMS) — what ExpansionSystems.run_cycle
#: returns in "ran" when every expansion system took a turn.
EXPANSION_RAN = [
    "books", "characters", "code", "commerce", "community",
    "datasci", "documents", "finance", "games", "health", "learn",
    "legal", "marketing", "media", "property", "sourcehealth",
    "travel",
]

#: The owner's screenshot case, in the __agency_shift__ tool's real
#: result shape ({ok, shift: work_shift(), expansion: run_cycle()}):
#: the core cycle ran digestion/news/weather, 1 useful turn, 8 stood
#: down, knowledge base at 121; the expansion ran all 17.
OWNER_CASE = {
    "ok": True,
    "shift": {
        "triggers_fired": 2,
        "triggers": ["news:new_cluster", "weather:forecast_change"],
        "cycle": {"systems_run": 3,
                  "ran": ["digestion", "news", "weather"],
                  "useful_turns": 1, "stood_down": 8,
                  "knowledge_items": 121, "preemptions": 0},
        "knowledge_items": 121,
    },
    "expansion": {"systems_run": 17, "ran": EXPANSION_RAN,
                  "useful_turns": 9, "stood_down": 0,
                  "knowledge_items": 121, "preemptions": 0},
}

OWNER_SUMMARY = ("2 triggers fired; 3 of 11 systems ran "
                 "(1 useful turn), 8 stood down; "
                 "expansion ran 17 of 17.")


def _doc(value, job_name="agency-shift"):
    return job_report_doc("__agency_shift__", value, job_name=job_name)


# ── the owner's case: dedupe + summary + per-system lines ────────────

def test_owner_case_becomes_a_digest():
    doc = _doc(dict(OWNER_CASE))
    assert isinstance(doc, Doc)
    assert doc.tone == "ok"
    assert doc.summary == OWNER_SUMMARY
    body = body_text(doc)
    # each cycle's parallel facts render once per section…
    assert body.count("Systems run") == 2  # Cycle + Expansion
    assert body.count("Useful turns") == 2
    # …and the triplicated knowledge count renders exactly once
    assert body.count("Knowledge base") == 1
    assert body.count("121") == 1
    assert "121 items" in body
    # no raw internal keys survive anywhere in the digest
    for raw in ("systems_run", "useful_turns", "triggers_fired",
                "stood_down", "knowledge_items", "preemptions"):
        assert raw not in body, raw
    # per-system lines: what ran, and the stood-down names (they
    # reconcile: 11 − 3 ran = 8 = the reported stood_down)
    assert "Ran: digestion · news · weather" in body
    assert ("Stood down: wisdom · research · impossible · planner "
            "· deals · social · leads · gigs") in body
    # the fired triggers are named, once each
    assert "news:new_cluster" in body
    assert "weather:forecast_change" in body
    assert body.count("news:new_cluster") == 1


def test_owner_case_dedupes_on_telegram_too():
    doc = _doc(dict(OWNER_CASE))
    text = render(doc, "telegram-bot").text
    assert text.count("121") == 1
    assert "<b>Systems run</b>: 3 of 11" in text
    assert "<b>Systems run</b>: 17 of 17" in text
    assert "<b>Knowledge base</b>: 121 items" in text
    assert "knowledge_items" not in text


def test_knowledge_count_is_the_freshest_copy():
    # the expansion deposited after the shift read its count: the
    # digest carries the expansion's 124, and 121 never appears
    value = {
        "ok": True,
        "shift": {"triggers_fired": 0, "triggers": [],
                  "cycle": {"systems_run": 1, "ran": ["digestion"],
                            "useful_turns": 1, "stood_down": 10,
                            "knowledge_items": 121, "preemptions": 0},
                  "knowledge_items": 121},
        "expansion": {"systems_run": 2, "ran": ["books", "media"],
                      "useful_turns": 1, "stood_down": 15,
                      "knowledge_items": 124, "preemptions": 0},
    }
    body = body_text(_doc(value))
    assert "124 items" in body
    assert body.count("124") == 1
    assert "121" not in body


# ── stood-down naming: reconciliation or silence ─────────────────────

def test_stood_down_names_require_reconciliation():
    # ran 2 + stood_down 8 leaves 9 unran of 11: one system had work
    # but missed its turn (cooldown) — the digest shows the count
    # and does NOT pin names it cannot prove stood down.
    value = {
        "ok": True,
        "shift": {"triggers_fired": 0, "triggers": [],
                  "cycle": {"systems_run": 2,
                            "ran": ["digestion", "news"],
                            "useful_turns": 1, "stood_down": 8,
                            "knowledge_items": 50, "preemptions": 0},
                  "knowledge_items": 50},
        "expansion": {"systems_run": 0, "ran": [], "useful_turns": 0,
                      "stood_down": 17, "knowledge_items": 50,
                      "preemptions": 0},
    }
    body = body_text(_doc(value))
    assert "wisdom · research" not in body  # no guessed name list
    assert "Ran: digestion · news" in body


def test_preemptions_row_only_when_nonzero():
    assert "Preemptions" not in body_text(_doc(dict(OWNER_CASE)))
    value = {
        "ok": True,
        "shift": {"triggers_fired": 0, "triggers": [],
                  "cycle": {"systems_run": 4,
                            "ran": ["digestion", "news", "planner",
                                    "weather"],
                            "useful_turns": 2, "stood_down": 7,
                            "knowledge_items": 50, "preemptions": 2},
                  "knowledge_items": 50},
        "expansion": None,
    }
    body = body_text(_doc(value))
    assert body.count("Preemptions") == 1
    assert "Preemptions**: 2" in body


# ── quiet + degraded shapes ──────────────────────────────────────────

QUIET = {
    "ok": True,
    "shift": {"triggers_fired": 0, "triggers": [],
              "cycle": {"systems_run": 0, "ran": [], "useful_turns": 0,
                        "stood_down": 11, "knowledge_items": 121,
                        "preemptions": 0},
              "knowledge_items": 121},
    "expansion": {"systems_run": 0, "ran": [], "useful_turns": 0,
                  "stood_down": 17, "knowledge_items": 121,
                  "preemptions": 0},
}


def test_quiet_shift_reads_quiet_not_failed():
    doc = _doc(dict(QUIET))
    assert doc.tone == "ok"
    assert doc.summary == ("Quiet shift — no triggers fired and "
                           "every system stood down.")
    body = body_text(doc)
    assert "Ran:" not in body
    assert body.count("121") == 1
    # a quiet cycle still names who stood down (11 of 11 reconciles)
    assert "Stood down: digestion · wisdom · research" in body


def test_missing_expansion_is_named_and_warns():
    value = dict(OWNER_CASE, expansion=None)
    doc = _doc(value)
    assert doc.tone == "warn"
    assert doc.resolved_icon == "⚠️"
    assert doc.summary == ("2 triggers fired; 3 of 11 systems ran "
                           "(1 useful turn), 8 stood down; "
                           "expansion did not run.")
    body = body_text(doc)
    assert "Did not run" in body
    # the knowledge count falls back to the shift's copy, still once
    assert body.count("121") == 1


def test_quiet_shift_with_missing_expansion():
    value = dict(QUIET, expansion=None)
    doc = _doc(value)
    assert doc.tone == "warn"
    assert doc.summary == ("Quiet shift — no triggers fired and "
                           "every system stood down; the expansion "
                           "cycle did not run.")


def test_ran_but_nothing_useful_is_stated_plainly():
    value = {
        "ok": True,
        "shift": {"triggers_fired": 0, "triggers": [],
                  "cycle": {"systems_run": 3,
                            "ran": ["digestion", "news", "weather"],
                            "useful_turns": 0, "stood_down": 8,
                            "knowledge_items": 7, "preemptions": 0},
                  "knowledge_items": 7},
        "expansion": {"systems_run": 0, "ran": [], "useful_turns": 0,
                      "stood_down": 17, "knowledge_items": 7,
                      "preemptions": 0},
    }
    doc = _doc(value)
    assert doc.tone == "ok"  # ran-but-useless is not a failure
    assert "(no useful turns)" in doc.summary


def test_fired_count_without_names_still_reports():
    value = {
        "ok": True,
        "shift": {"triggers_fired": 3, "triggers": [],
                  "cycle": {"systems_run": 1, "ran": ["news"],
                            "useful_turns": 1, "stood_down": 10,
                            "knowledge_items": 7, "preemptions": 0},
                  "knowledge_items": 7},
        "expansion": None,
    }
    doc = _doc(value)
    assert doc.summary.startswith("3 triggers fired;")
    assert "Fired**: 3" in body_text(doc)


def test_malformed_values_degrade_to_generic_doc():
    for value in ({"ok": True},
                  {"ok": True, "shift": "nope"},
                  {"ok": True, "shift": {"cycle": "nope"}}):
        doc = _doc(value)
        assert isinstance(doc, Doc)  # never raises, never the digest
        assert "Quiet shift" not in doc.summary
        assert "systems ran" not in doc.summary


# ── the real producer, end to end ────────────────────────────────────

def test_real_work_shift_output_digests():
    from nomorals.agency.systems import ActiveSystems

    systems = ActiveSystems(db=Database(":memory:"))
    shift = systems.work_shift()  # no adapters, empty store: quiet
    assert shift["cycle"]["systems_run"] == 0
    doc = _doc({"ok": True, "shift": shift, "expansion": None})
    assert doc.tone == "warn"  # the expansion leg did not run
    assert doc.summary == ("Quiet shift — no triggers fired and "
                           "every system stood down; the expansion "
                           "cycle did not run.")
    body = body_text(doc)
    assert body.count("Knowledge base") == 1
    assert f"{shift['knowledge_items']} items" in body


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


def _ok(value):
    return SimpleNamespace(ok=True, value=value, error=None)


def test_scheduler_publishes_digest_and_stores_legacy_string(tmp_path):
    tools = _Tools(_ok(dict(OWNER_CASE)))
    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(
        db=db,
        settings=SimpleNamespace(home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={}, tools=tools)
    sched = Scheduler(ctx)
    sched.notifier = _RecordingNotifier()
    job = sched.add("agency-shift", "every 1800s", "tool",
                    {"tool": "__agency_shift__"}, routine=False)
    out = sched.run_now(job["id"])
    assert out["ok"] is True, out
    calls = [c for c in sched.notifier.calls if c["kind"] == "schedule"]
    assert len(calls) == 1, calls
    assert calls[0]["title"] == "✅ scheduled: agency-shift"
    doc = calls[0]["kw"].get("doc")
    assert isinstance(doc, Doc), calls[0]
    assert doc.summary == OWNER_SUMMARY
    published = body_text(doc)
    assert "knowledge_items" not in published
    assert published.count("121") == 1
    # stored: the legacy string form, byte-for-byte the old contract —
    # the dump still exists in the record; it is just not what the
    # owner is sent any more.
    row = ctx.db.query_one(
        "SELECT last_result FROM schedule_jobs WHERE id = ?",
        (job["id"],))
    last = row["last_result"] or ""
    assert last.startswith("__agency_shift__:"), last
    assert "knowledge_items" in last, last
    assert "systems_run" in last, last
