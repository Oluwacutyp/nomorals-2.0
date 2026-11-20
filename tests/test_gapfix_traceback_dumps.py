"""Lane 134 — raw traceback dumps must never reach the owner's DM.

The 2026-11-19 screenshots show two scheduled-job failures arriving
as walls of Windows paths joined by arrows — once as the report
itself, once as the ``alert policy:`` line riding it.  The stored
failure text is the journal (it keeps the watchdog's sampled stack
chains, the histogram and the db tail); the chat form is one short
line.  These tests pin both halves: the published Doc / policy
reason is short and frame-free, and the run record keeps the whole
text.

The wall texts below are verbatim from the screenshots (``wisdom
organ tick`` and ``beddesk study``, 2026-11-19, 9:45 AM) so the
sanitizer is proven against the real shape, not a paraphrase.
"""

from types import SimpleNamespace

from nomorals.agents.alert_policy import AlertDecision
from nomorals.agents.job_reports import (
    chat_failure_line,
    command_failure_doc,
    failure_doc,
    summarize_last_result,
)
from nomorals.agents.scheduler import Scheduler
from nomorals.storage.db import Database

# ── the verbatim walls ───────────────────────────────────────────────

WISDOM_WALL = (
    "job failed: timed out after 300s in "
    "C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\"
    "Python311\\Lib\\pathlib.py:1044:Path.open → "
    "C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\"
    "Python311\\Lib\\pathlib.py:1058:Path.read_text → "
    "C:\\devon\\nomorals-2.0\\nomorals\\wisdom\\corpus.py:578:"
    "CanonCorpus._chapter_text (worker abandoned; 1 abandoned "
    "worker(s) outstanding for 'wisdom organ tick'; the stuck call "
    "may still be running) — where the time went: "
    "C:\\devon\\nomorals-2.0\\nomorals\\books\\library.py:1477:"
    "Library._search_fs → C:\\devon\\nomorals-2.0\\nomorals\\books\\"
    "library.py:1356:Library.search → C:\\devon\\nomorals-2.0\\"
    "nomorals\\wisdom\\corpus.py:810:CanonCorpus.ask ~75s; "
    "C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\"
    "Python311\\Lib\\pathlib.py:1044:Path.open → "
    "C:\\devon\\nomorals-2.0\\nomorals\\wisdom\\corpus.py:578:"
    "CanonCorpus._chapter_text ~40s — db: write-lock free; journal "
    "wal; 64.0MB; open txns 0 — db interrupt sent to the stuck worker"
)

BEDDESK_WALL = (
    "job failed: timed out after 300s in "
    "C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\"
    "Python311\\Lib\\ssl.py:1072:SSLContext.load_default_certs → "
    "C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\"
    "Python311\\Lib\\ssl.py:771:create_default_context → "
    "C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\"
    "Python311\\Lib\\urllib\\request.py:1318:urlopen (worker "
    "abandoned; 1 abandoned worker(s) outstanding for 'beddesk "
    "study'; the stuck call may still be running) — where the time "
    "went: C:\\devon\\nomorals-2.0\\nomorals\\storage\\__init__.py:"
    "10:Database.execute ~290s — db: write-lock free — db interrupt "
    "sent to the stuck worker"
)

TRACEBACK_TEXT = (
    "Traceback (most recent call last):\n"
    "  File \"x.py\", line 3, in f\n"
    "    raise ValueError(\"bad flag\")\n"
    "ValueError: bad flag"
)


# ── (a) the chat-short line ──────────────────────────────────────────

def test_wisdom_wall_becomes_one_short_line():
    line = chat_failure_line(WISDOM_WALL)
    assert line == (
        "job failed: timed out after 300s in "
        "pathlib.py:1044:Path.open (worker abandoned; "
        "1 abandoned worker(s) outstanding for 'wisdom organ tick'; "
        "the stuck call may still be running)")
    assert "\n" not in line
    assert "→" not in line
    assert "C:\\" not in line
    assert "where the time went" not in line
    assert "db:" not in line
    assert len(line) <= 200


def test_beddesk_wall_names_the_stuck_call_only():
    line = chat_failure_line(BEDDESK_WALL)
    assert "timed out after 300s" in line
    assert "ssl.py:1072:SSLContext.load_default_certs" in line
    assert "urlopen" not in line  # the caller frames are gone
    assert "→" not in line and "C:\\" not in line
    assert "\n" not in line
    assert len(line) <= 200


def test_short_clean_lines_pass_through_unchanged():
    # the F3 pins: an already-short failure is the owner's own words
    assert chat_failure_line(
        "job failed: tool finance_alert failed: boom") == (
        "job failed: tool finance_alert failed: boom")
    assert chat_failure_line("command failed: exit=3") == (
        "command failed: exit=3")
    assert chat_failure_line("exit=3: bad flag") == "exit=3: bad flag"


def test_pasted_traceback_reduces_to_its_exception_line():
    line = chat_failure_line(TRACEBACK_TEXT)
    assert line == "ValueError: bad flag"
    assert "File " not in line and "Traceback" not in line


def test_command_stdout_traceback_reduces_to_its_exception_line():
    summary = "command failed: exit=3: " + TRACEBACK_TEXT
    line = chat_failure_line(summary)
    assert line == "command failed: exit=3: ValueError: bad flag"


def test_empty_and_frame_only_summaries_fall_back():
    assert chat_failure_line("") == "job failed"
    assert chat_failure_line("   \n  ") == "job failed"
    assert chat_failure_line(
        "  File \"x.py\", line 1, in f\n    pass") == "job failed"


def test_long_unknown_line_caps_on_a_word_boundary():
    blob = "job failed: " + "x" * 300
    line = chat_failure_line(blob)
    assert len(line) <= 201
    assert line.endswith("…")
    assert "\n" not in line


# ── the published Docs ───────────────────────────────────────────────

def test_failure_doc_carries_the_short_line():
    doc = failure_doc("wisdom organ tick", WISDOM_WALL)
    assert doc.tone == "error"
    assert doc.title == "scheduled: wisdom organ tick"
    assert doc.summary == chat_failure_line(WISDOM_WALL)
    assert "→" not in doc.summary and "C:\\" not in doc.summary


def test_command_failure_doc_carries_the_short_line():
    doc = command_failure_doc("cmd job", "make it so",
                              "command failed: exit=3: " + TRACEBACK_TEXT)
    assert doc.summary == "command failed: exit=3: ValueError: bad flag"
    assert "File " not in doc.summary


def test_policy_line_caps_a_long_reason():
    long_reason = "timed out after 300s in " + "C:\\x\\y.py:1:f " * 40
    line = AlertDecision(tier="page", reason=long_reason,
                         subject="wisdom organ tick").body_line()
    assert "\n" not in line
    assert "wisdom organ tick" in line
    assert line.endswith("… (subject: wisdom organ tick)")
    short = AlertDecision(tier="page", reason="recurrence",
                          subject="gate").body_line()
    assert short == "alert policy: page — recurrence (subject: gate)"


def test_summarize_last_result_shortens_failed_rows_only():
    assert summarize_last_result({"last_result": WISDOM_WALL}) == (
        "❌ " + chat_failure_line(WISDOM_WALL))
    # a success row keeps its old shape
    assert summarize_last_result(
        {"last_result": "echo:\nok: yes\nsaid: hi"}) == "✅ said: hi"


# ── (b/c) end-to-end: short in chat, whole in the journal ────────────

class _Tools:
    """Duck-typed tool registry: per-tool canned Outcomes."""

    def __init__(self, outcomes):
        self._outcomes = outcomes

    def call(self, tool, capabilities=None, **args):
        outcome = self._outcomes[tool]
        if callable(outcome):
            return outcome(**args)
        return outcome


def _ok(value):
    return SimpleNamespace(ok=True, value=value, error=None)


def _err(message):
    return SimpleNamespace(ok=False, value=None,
                           error=SimpleNamespace(message=message))


def _ctx(tools=None):
    db = Database(":memory:")
    db.migrate()
    if tools is None:
        tools = _Tools({"echo": _ok({"ok": True, "said": "hi"})})
    return SimpleNamespace(db=db, tools=tools,
                           settings=SimpleNamespace(), extras={})


class _Notifier:
    """Recording notifier stub: remembers the Doc and the hints too."""

    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title, "body": body,
                           "doc": kw.get("doc"),
                           "hints": kw.get("alert_hints") or {}})
        return {"delivered": True, "channel": "test"}


def _sched(ctx):
    sched = Scheduler(ctx)
    sched.notifier = _Notifier()
    return sched


def _job(sched, name):
    rows = [j for j in sched.list_jobs() if j.get("name") == name]
    assert len(rows) == 1
    return rows[0]


def _schedule_alerts(sched):
    return [c for c in sched.notifier.calls if c["kind"] == "schedule"]


def test_declared_failure_wall_short_in_chat_whole_in_journal():
    # the tool declares failure with the verbatim wall as its error
    tools = _Tools({"echo": _ok({"ok": False, "error": WISDOM_WALL})})
    sched = _sched(_ctx(tools))
    job = sched.add("wisdom organ tick", "every 1h", "tool",
                    {"tool": "echo"})
    out = sched.run_now(job["id"])
    assert out["ok"] is False
    runs = sched.recent_runs(job["id"])
    assert runs[0]["ok"] is False
    # (b) the journal keeps the WHOLE wall: the stored columns
    # themselves.  list_jobs / recent_runs truncate their display
    # copies (300 chars), so the journal is read from the tables.
    stored = sched.db.query(
        "SELECT last_result FROM schedule_jobs WHERE id = ?",
        (job["id"],))
    assert len(stored) == 1
    assert WISDOM_WALL in (stored[0].get("last_result") or "")
    rows = sched.db.query(
        "SELECT result FROM schedule_runs WHERE job_id = ?",
        (job["id"],))
    assert len(rows) == 1
    assert WISDOM_WALL in (rows[0].get("result") or "")
    # (a) the published Doc and the policy reason are the short line
    alerts = _schedule_alerts(sched)
    assert len(alerts) == 1
    assert alerts[0]["title"].startswith("❌")
    doc = alerts[0]["doc"]
    assert doc is not None
    assert doc.summary == chat_failure_line(
        "job failed: tool echo failed: " + WISDOM_WALL)
    assert "→" not in doc.summary and "C:\\" not in doc.summary
    assert "\n" not in doc.summary
    assert "pathlib.py:1044:Path.open" in doc.summary
    assert "timed out after 300s" in doc.summary
    reason = alerts[0]["hints"].get("reason") or ""
    assert reason == doc.summary
    assert "→" not in reason and "C:\\" not in reason


def test_command_traceback_short_in_chat_whole_in_journal():
    tools = _Tools({"shell_run": _ok({"exit_code": 3,
                                      "stdout": TRACEBACK_TEXT,
                                      "stderr": ""})})
    sched = _sched(_ctx(tools))
    job = sched.add("cmd job", "every 1h", "command",
                    {"command": "make it so"})
    out = sched.run_now(job["id"])
    assert out["ok"] is False
    runs = sched.recent_runs(job["id"])
    assert "ValueError: bad flag" in runs[0]["result"]
    assert "Traceback (most recent call last)" in runs[0]["result"]
    alerts = _schedule_alerts(sched)
    assert len(alerts) == 1
    doc = alerts[0]["doc"]
    assert doc is not None
    assert doc.summary == (
        "job failed: command failed: exit=3: ValueError: bad flag")
    assert "Traceback" not in doc.summary


def test_unknown_exception_shape_falls_back_to_one_capped_line():
    # (c) a failure whose text has no recognizable shape at all:
    # three lines of garbage, no traceback header, no chain.
    weird = ("weird failure\nwith more detail on the next line\n"
             "and even more after that")
    tools = _Tools({"echo": _err(weird)})
    sched = _sched(_ctx(tools))
    job = sched.add("job one", "every 1h", "tool", {"tool": "echo"})
    out = sched.run_now(job["id"])
    assert out["ok"] is False
    alerts = _schedule_alerts(sched)
    assert len(alerts) == 1
    doc = alerts[0]["doc"]
    assert doc is not None
    assert doc.summary == "job failed: tool echo failed: weird failure"
    assert "\n" not in doc.summary
    # the journal still keeps all three lines
    assert "and even more after that" in sched.recent_runs(
        job["id"])[0]["result"]
