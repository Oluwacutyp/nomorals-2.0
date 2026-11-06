"""Presentation & message-flow overhaul gates (owner rulings
2026-11-05 evening, executable form).

The live bot's /play answered with THREE pasted messages (a queue
narration carrying an internal job id, the audio file, then a
"sources tried … sent X to this chat" receipt) and a scheduled
finance job posted an all-zero no-op report.  The laws these tests
pin:

* scheduled silence — a no-op scheduled run posts NOTHING (findings
  and failures still post);
* the stamp — "Powered by peacethefirst" rides presented output,
  lightest form on plain replies, never duplicated;
* wide buttons — one button per row through the shared layer;
* right artifact, honestly labeled — for a plain "title by artist"
  request the canonical recording outranks a Nightcore/edit copy;
  an edit that wins by default is LABELED as an edit in the line;
* the Telegram userbot's buttons are real end to end (AA-77): the
  adapter builds native inline markup and a tap routes back as
  the command, judged as the tapper.

On pristine c91e4c5 the silence, stamp, ranking and userbot tests
fail (the no-op post, the missing stamp helper, the first-listed
Nightcore win and the dropped buttons are the shipped behavior).
"""

from __future__ import annotations

import asyncio
import sys
import types as _types
from types import SimpleNamespace
from unittest import mock

# NOTE: helpers that only exist after the overhaul (noop_reason,
# apply_stamp, wide_layout) are imported inside their tests, so on
# the pristine tree the scheduler end-to-end test below still runs
# and fails on BEHAVIOR (the no-op post) rather than at import.
from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
from nomorals.agents.scheduler import Scheduler
from nomorals.storage.db import Database

# ── scheduled silence: the classifier ────────────────────────────────


def test_noop_reason_finance_zero_zero_is_silent():
    from nomorals.agents.job_reports import noop_reason

    assert noop_reason("finance_alert", {
        "ok": True, "delivered": True, "checked": 0, "fired": 0,
        "alerts": [], "snoozed": []})


def test_noop_reason_finance_checked_but_quiet_posts():
    from nomorals.agents.job_reports import noop_reason

    # Sources were actually checked — that IS a report.
    assert noop_reason("finance_alert", {
        "checked": 3, "fired": 0, "alerts": []}) is None


def test_noop_reason_finance_fired_posts():
    from nomorals.agents.job_reports import noop_reason

    assert noop_reason("finance_alert", {
        "checked": 3, "fired": 1,
        "alerts": [{"symbol": "XAUUSD"}]}) is None


def test_noop_reason_agency_quiet_only_when_expansion_ran():
    from nomorals.agents.job_reports import noop_reason

    quiet = {"shift": {"cycle": {"systems_run": 0, "useful_turns": 0},
                       "triggers_fired": 0, "triggers": []},
             "expansion": {"systems_run": 0}}
    assert noop_reason("__agency_shift__", quiet)
    # Expansion missing entirely = a finding, never silence.
    missing = {"shift": {"cycle": {"systems_run": 0,
                                   "useful_turns": 0},
                         "triggers_fired": 0, "triggers": []}}
    assert noop_reason("__agency_shift__", missing) is None


def test_noop_reason_generic_shapes():
    from nomorals.agents.job_reports import noop_reason

    assert noop_reason("any_tool", None)
    assert noop_reason("any_tool", "")
    assert noop_reason("any_tool", {"ok": True, "delivered": True})
    assert noop_reason("any_tool", {"checked": 0, "items": []})
    assert noop_reason("any_tool", {"items": [{"id": 1}]}) is None
    assert noop_reason("any_tool", "real output") is None
    assert noop_reason("any_tool", 0)


# ── scheduled silence: end to end through the scheduler ─────────────


class _SendResult:
    def __init__(self, ok=True, error=""):
        self.ok = ok
        self.error = error


class _FakeGateway:
    def __init__(self, live=("telegram",)):
        self.live = set(live)
        self.sent = []

    def status(self):
        return {p: {"running_in_session": True} for p in self.live}

    def send(self, platform, chat_ref, text, **kw):
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult(ok=True)


def _sched(value, *, ok=True):
    db = Database(":memory:")
    db.migrate()
    partner = SimpleNamespace(owner_chats="telegram:111")
    settings = SimpleNamespace(partner=partner)
    tools = SimpleNamespace(
        call=lambda tool, capabilities=None, **kw: SimpleNamespace(
            ok=ok, value=value, error=None if ok else "boom"))
    gateway = _FakeGateway()
    ctx = SimpleNamespace(db=db, settings=settings, tools=tools,
                          extras={"gateway": gateway})
    return Scheduler(ctx, gateway=gateway), gateway, db


def _run_finance_job(sched, db):
    job = sched.add("finance alerts", "every 1h", "tool",
                    {"tool": "finance_alert"})
    row = db.query_one("SELECT * FROM schedule_jobs WHERE id = ?",
                       (job["id"] if isinstance(job, dict)
                        else job,))
    return sched._execute(row)


def test_noop_scheduled_run_sends_nothing():
    sched, gateway, db = _sched({"ok": True, "delivered": True,
                                 "checked": 0, "fired": 0,
                                 "alerts": [], "snoozed": []})
    outcome = _run_finance_job(sched, db)
    assert outcome["ok"] is True
    assert outcome["silent"]
    # The whole point: zero sends crossed the gateway.
    assert gateway.sent == [], gateway.sent


def test_checked_scheduled_run_still_posts():
    sched, gateway, db = _sched({"ok": True, "delivered": True,
                                 "checked": 3, "fired": 0,
                                 "alerts": []})
    outcome = _run_finance_job(sched, db)
    assert not outcome["silent"]
    assert gateway.sent, "a run with real checks must still report"


def test_failed_scheduled_run_still_posts():
    sched, gateway, db = _sched(None, ok=False)
    outcome = _run_finance_job(sched, db)
    assert outcome["ok"] is False
    assert gateway.sent, "failures are owner-actionable — they post"


# ── the stamp ────────────────────────────────────────────────────────


def test_apply_stamp_full_and_light_forms():
    from nomorals.social.chat.typography import DEFAULT_STAMP_TEXT, apply_stamp

    full = apply_stamp("body")
    assert full.startswith("body\n\n")
    assert DEFAULT_STAMP_TEXT in full
    light = apply_stamp("body", light=True)
    assert light.startswith("body\n")
    assert "\n\n" not in light
    assert DEFAULT_STAMP_TEXT in light


def test_apply_stamp_idempotent_and_empty_safe():
    from nomorals.social.chat.typography import apply_stamp

    once = apply_stamp("body")
    assert apply_stamp(once) == once
    assert apply_stamp("") == ""


def test_job_report_docs_carry_the_stamp():
    from nomorals.agents.job_reports import (
        command_report_doc,
        job_report_doc,
        tool_result_doc,
    )
    from nomorals.social.chat.typography import DEFAULT_STAMP_TEXT
    doc = job_report_doc("greeter", "hello")
    assert (doc.footer or "") == DEFAULT_STAMP_TEXT
    doc = tool_result_doc("w", {"count": 2})
    assert (doc.footer or "") == DEFAULT_STAMP_TEXT
    doc = command_report_doc(
        "w", "ls", {"exit_code": 0, "stdout": "x", "stderr": ""})
    assert (doc.footer or "") == DEFAULT_STAMP_TEXT


# ── wide buttons ─────────────────────────────────────────────────────


def test_wide_layout_one_button_per_row():
    from nomorals.social.chat.tgbot_buttons import wide_layout

    kb = [[("A", "a"), ("B", "b")], [("C", "c")]]
    assert wide_layout(kb) == [[("A", "a")], [("B", "b")], [("C", "c")]]
    # Empty/absent stays absent (adapters read None as "no buttons").
    assert wide_layout([]) is None
    assert wide_layout(None) is None


# ── right artifact, honestly labeled ─────────────────────────────────


class _FakeGatewayMedia:
    """The mixin's gateway seam: sends, files and edits recorded."""

    def __init__(self):
        self.sent = []
        self.files = []
        self.edits = []
        self._n = 0

    def _adapter_for(self, platform):
        return self

    def send(self, platform, target, text, **kwargs):
        self._n += 1
        self.sent.append(text)
        return SimpleNamespace(ok=True, message_id=f"m{self._n}")

    def send_file(self, platform, target, path, **kwargs):
        self.files.append((path, kwargs.get("caption", "")))
        return SimpleNamespace(ok=True)

    def edit_message(self, chat, message_id, text, **kwargs):
        self.edits.append((message_id, text))
        return {"ok": True}


def _media_handler():
    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.gateway = _FakeGatewayMedia()
    rt.context = SimpleNamespace()
    return rt


def _chat(chat_id="ord-1"):
    return SimpleNamespace(platform="telegram", chat_id=chat_id,
                           key=f"telegram:{chat_id}")


_NC = {"kind": "youtube", "path": "https://youtu.be/nc",
       "title": "FEAR (Nightcore)", "artist": "NF", "duration": 213.0}
_CLEAN = {"kind": "youtube", "path": "https://youtu.be/clean",
          "title": "FEAR", "artist": "NF", "duration": 213.0}
_ATTEMPTS = ["youtube-search — ok (2)"]


def _engine():
    engine = mock.Mock()
    engine.queue.return_value = []
    return engine


def test_edit_marker_detection():
    marker = RuntimeMediaMixin._edit_marker_of
    assert marker("FEAR (Nightcore)") == "nightcore"
    assert marker("FEAR (sped up)") == "sped up"
    assert marker("FEAR (Remix)") == "remix"
    assert marker("FEAR") == ""
    assert marker("FEAR (Official Audio)") == ""
    # Substring lookalikes are not markers.
    assert marker("Creditors' Editorial") == ""


def test_original_outranks_the_nightcore_copy():
    rt = _media_handler()
    chat = _chat()
    engine = _engine()
    engine.download.return_value = {
        "ok": True, "path": "/tmp/clean.mp3", "title": "FEAR"}
    out, sent = rt._play_send_with_fallthrough(
        engine, dict(_NC), [dict(_CLEAN)], _ATTEMPTS, chat,
        match=("FEAR", "NF"), query="fear by nf")
    assert sent is True
    attempted = [c[0][0]["path"] for c in engine.download.call_args_list]
    # The canonical recording was attempted FIRST (and won) — the
    # Nightcore copy was never downloaded behind the plain request.
    assert attempted == ["https://youtu.be/clean"], attempted
    assert out == "🎵 FEAR — NF"


def test_edit_win_is_labeled_as_an_edit():
    rt = _media_handler()
    chat = _chat()
    engine = _engine()
    engine.download.return_value = {
        "ok": True, "path": "/tmp/nc.mp3", "title": "FEAR (Nightcore)"}
    out, sent = rt._play_send_with_fallthrough(
        engine, dict(_NC), [], _ATTEMPTS, chat,
        match=("FEAR", "NF"), query="fear by nf")
    assert sent is True
    # Delivered under the request's identity, labeled as the edit
    # it is — never passed off as the original.
    assert out == ("🎵 FEAR — NF · nightcore edit — "
                   "the original wasn't obtainable")
    caption = rt.gateway.files[0][1]
    assert "nightcore edit" in caption


# ── AA-77: the userbot's buttons are real ────────────────────────────


class _FakeButton:
    def __init__(self, kind, label, payload):
        self.kind = kind
        self.label = label
        self.payload = payload

    @staticmethod
    def inline(label, data):
        return _FakeButton("inline", label, data)

    @staticmethod
    def url(label, url):
        return _FakeButton("url", label, url)


def _userbot_adapter(tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    return TelegramAdapter(api_id=1, api_hash="hash",
                           media_dir=str(tmp_path))


def test_userbot_inline_markup_builds_native_buttons(tmp_path,
                                                      monkeypatch):
    fake = _types.ModuleType("telethon")
    fake.Button = _FakeButton
    monkeypatch.setitem(sys.modules, "telethon", fake)
    adapter = _userbot_adapter(tmp_path)
    markup = adapter._inline_markup(
        [[("▶ Play", "play fear"), ("Site", "https://example.com")]])
    assert markup is not None and len(markup) == 1
    inline_btn, url_btn = markup[0]
    assert inline_btn.kind == "inline"
    assert inline_btn.payload == b"play fear"
    assert url_btn.kind == "url"
    assert url_btn.payload == "https://example.com"


def test_userbot_inline_markup_drops_oversized_data(tmp_path,
                                                    monkeypatch):
    fake = _types.ModuleType("telethon")
    fake.Button = _FakeButton
    monkeypatch.setitem(sys.modules, "telethon", fake)
    adapter = _userbot_adapter(tmp_path)
    markup = adapter._inline_markup([[("Big", "x" * 65)]])
    assert markup is None  # nothing attachable remains


def test_userbot_callback_tap_routes_as_the_tapper(tmp_path):
    adapter = _userbot_adapter(tmp_path)
    delivered = []
    adapter._deliver = lambda handler, msg: delivered.append(msg)

    class _Event:
        data = b"undonotice n42"
        chat_id = "4242"
        sender_id = "777"
        message_id = "9"

        async def answer(self):
            return None

        async def get_chat(self):
            return SimpleNamespace(id=4242, title="")

        async def get_sender(self):
            return SimpleNamespace(id=777, username="ownerhandle",
                                   first_name="Owner")

    asyncio.run(adapter._handle_callback_query(_Event(),
                                               handler=None))
    assert len(delivered) == 1
    msg = delivered[0]
    # A tap re-enters as the command, sent BY THE TAPPER — downstream
    # gates judge it exactly like the tapper typing.
    assert msg.text == "/undonotice n42"
    assert msg.sender == "ownerhandle"
    assert msg.meta.get("callback_query") is True
