"""Dynamic buttons for what's actually rendering — lane 25.

Owner live report 2026-11-17 (Telegram DM): three scheduled-job
messages each carried the SAME static Jobs / Status / Re-run strip
under content it had nothing to do with — a proposals list with no
decision buttons, a trendops proposal with no Apply button, and a
day-derivation "nothing was told" message with no action at all.

The tests pin the content-derived rule:

* a scheduled proposals message carries decision buttons for the
  proposals it lists (Approve/Deny through the real /approve and
  /deny verbs);
* a scheduled trendops message carries an Apply button per proposal;
* a scheduled message with no action in its content carries NO
  buttons — the generic strip survives only on a FAILED run, where
  Re-run is the content's action;
* a tool run that already delivered its own message (pulse slots)
  does not ALSO post a duplicate scheduled alert;
* settled proposal cards clear their buttons on WhatsApp too
  (revoke + resend when the bridge refuses the clearing edit);
* adaptive keyboards pack two per row on Telegram.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from nomorals.agents import morning_pulse as mp
from nomorals.agents.job_reports import noop_reason
from nomorals.social.chat import adaptive_buttons
from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

INIT_A = "init_01M6G69CRTGS55QFCXZTDNTM05"
INIT_B = "init_01M6GP7S0MHYR87WHKTCQYZPS8"


# ── 1. the duplicate self-delivered alert ───────────────────────────


class TestSelfDeliveredSilence:
    def test_pulse_run_that_delivered_is_silent(self):
        value = {"ok": True, "slot": "evening", "stages": ["compose"],
                 "text": "wrap", "delivered_text": True,
                 "delivered_audio": False}
        assert noop_reason("pulse", value) == "already delivered by the tool run"

    def test_pulse_run_that_failed_delivery_still_posts(self):
        value = {"ok": True, "slot": "evening", "stages": ["compose"],
                 "text": "wrap", "delivered_text": False}
        assert noop_reason("pulse", value) is None

    def test_tool_without_the_flag_posts_as_before(self):
        value = {"ok": True, "slot": "evening", "text": "wrap"}
        assert noop_reason("pulse", value) is None


# ── 2. proposals on the direct pulse delivery ───────────────────────


class TestPendingProposals:
    def _ctx(self, tmp_path, rows):
        class _Db:
            def query(self, sql, *args, **kwargs):
                if "proactive_log" in sql:
                    return rows
                return []

        class _Store:
            def __init__(self, *args, **kwargs):
                pass

            def list_enabled(self):
                return []

        ctx = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=str(tmp_path / "ws")))
        ctx.db = _Db()
        return ctx, _Store

    def test_pending_ids_list_the_wrap_proposals(self, tmp_path, monkeypatch):
        rows = [{"id": INIT_A, "kind": "initiative", "platform": "telegram",
                 "chat_id": "123", "content": "post the GTA edit"},
                {"id": INIT_B, "kind": "message", "platform": "telegram",
                 "chat_id": "123", "content": "remind me at 9"}]
        ctx, store = self._ctx(tmp_path, rows)
        monkeypatch.setattr("nomorals.finance.alerts.AlertStore", store)
        fn = getattr(mp, "_pending_proposal_ids", None)
        assert fn is not None
        assert fn(ctx) == [INIT_A, INIT_B]

    def test_pending_line_names_what_is_being_decided(
            self, tmp_path, monkeypatch):
        rows = [{"id": INIT_A, "kind": "initiative", "platform": "telegram",
                 "chat_id": "123", "content": "post the GTA edit tonight"}]
        ctx, store = self._ctx(tmp_path, rows)
        monkeypatch.setattr("nomorals.finance.alerts.AlertStore", store)
        text = mp._pending_items(ctx)
        assert INIT_A in text
        # The line names the proposal's content, not just its id —
        # a decision button is only meaningful when the owner can
        # see what she would be deciding.
        assert "post the GTA edit tonight" in text


class TestPulseButtons:
    def test_each_listed_proposal_gets_its_decision_row(self):
        kb = adaptive_buttons_for(
            output_kind="pulse", payload={"proposals": [INIT_A, INIT_B]})
        assert kb is not None
        assert len(kb) == 2
        flat = [data for row in kb for _label, data in row]
        assert f"approve {INIT_A}" in flat
        assert f"deny {INIT_A}" in flat
        assert f"proposals show {INIT_A}" in flat
        assert f"approve {INIT_B}" in flat
        assert f"deny {INIT_B}" in flat

    def test_pulse_buttons_stay_inside_the_callback_budget(self):
        kb = adaptive_buttons_for(
            output_kind="pulse", payload={"proposals": [INIT_A]})
        assert kb is not None
        for _label, data in kb[0]:
            assert len(data.encode("utf-8")) <= 47

    def test_no_proposals_means_no_buttons(self):
        assert adaptive_buttons_for(
            output_kind="pulse", payload={"proposals": []}) is None
        assert adaptive_buttons_for(output_kind="pulse", payload={}) is None
        assert adaptive_buttons_for(output_kind="pulse") is None

    def test_pulse_is_a_registered_output_kind(self):
        assert "pulse" in adaptive_buttons.OUTPUT_KINDS


# ── 3. the generic strip survives only on failure ───────────────────


class TestScheduledStrip:
    def test_success_content_gets_no_filler_strip(self):
        assert adaptive_buttons_for(
            output_kind="scheduled",
            payload={"job": "day derivation", "ok": True,
                     "attempt": "run:1.0"}) is None
        assert adaptive_buttons_for(
            output_kind="scheduled",
            payload={"job": "trendops proposals", "ok": True,
                     "attempt": "run:1.0"}) is None

    def test_failed_run_keeps_the_job_strip(self):
        kb = adaptive_buttons_for(
            output_kind="scheduled",
            payload={"job": "day derivation", "ok": False,
                     "attempt": "run:1.0"})
        assert kb is not None
        flat = [data for row in kb for _label, data in row]
        assert "schedule run day derivation" in flat
        assert "schedule list" in flat
        assert "schedule status" in flat

    def test_memory_consolidation_keeps_its_content_actions(self):
        kb = adaptive_buttons_for(
            output_kind="scheduled",
            payload={"job": "memory-consolidation", "ok": True,
                     "attempt": "run:1.0"})
        assert kb is not None
        flat = [data for row in kb for _label, data in row]
        assert "recall" in flat
        assert "memories" in flat

    def test_no_job_means_no_buttons(self):
        assert adaptive_buttons_for(
            output_kind="scheduled", payload={"ok": True}) is None
        assert adaptive_buttons_for(output_kind="scheduled") is None


# ── 4. trendops Apply ───────────────────────────────────────────────


class TestTrendopsDocActions:
    def test_one_apply_button_per_proposal(self):
        from nomorals.agents.job_reports import _trendops_doc

        value = {"proposals": [
            {"name": "Roadster GTA edit", "slug": "roadster",
             "why": "Roadster GTA edit peaks now — 29d left, used 0x",
             "apply": "/trendops apply roadster <footage>",
             "days_left": 28.9, "uses": 0},
        ]}
        doc = _trendops_doc("trendops", value,
                            title="scheduled: trendops proposals")
        from nomorals.social.chat.present import actions_rows

        rows = actions_rows(doc)
        flat = [data for row in rows for _label, data in row]
        assert "trendops apply roadster" in flat

    def test_apply_buttons_cap_at_three(self):
        from nomorals.agents.job_reports import _trendops_doc
        from nomorals.social.chat.present import actions_rows

        value = {"proposals": [
            {"name": f"Format {i}", "slug": f"fmt{i}",
             "why": f"Format {i} peaks", "apply": f"/trendops apply fmt{i} <footage>",
             "days_left": 5.0, "uses": 0}
            for i in range(5)
        ]}
        doc = _trendops_doc("trendops", value,
                            title="scheduled: trendops proposals")
        rows = actions_rows(doc)
        flat = [data for row in rows for _label, data in row]
        assert len(flat) == 3
        assert "trendops apply fmt0" in flat
        assert "trendops apply fmt4" not in flat


class TestTrendopsFootageAsk:
    def _fake_self(self):
        calls = {"apply": [], "ask": []}
        store: dict[str, dict] = {}

        self_ns = SimpleNamespace(context=SimpleNamespace())
        self_ns._trendops_calls = calls
        self_ns._trendops_store = store
        self_ns._PROPOSAL_ASK_TTL_S = 900.0

        def _store():
            return self_ns._trendops_store

        def _note(chat_key, fmt):
            calls["ask"].append((chat_key, fmt))
            store[chat_key] = {"action": "apply", "format": fmt,
                               "at": time.time()}

        def _apply(fmt_name, footage, **kwargs):
            calls["apply"].append((fmt_name, footage))
            return f"cut {fmt_name} from {footage[0]}"

        self_ns._trendops_ask_store = _store
        self_ns._note_trendops_ask = _note
        self_ns._trendops_apply = _apply
        # The arming helper is the REAL mixin method, bound to this
        # fake so the test exercises it (it only needs the stubbed
        # _note_trendops_ask above).
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        self_ns._trendops_ask_for_footage = (
            lambda chat_key, fmt: RuntimeMediaMixin._trendops_ask_for_footage(
                self_ns, chat_key, fmt))
        return self_ns, calls, store

    def test_apply_without_footage_arms_the_ask(self, monkeypatch):
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        fake, calls, _store = self._fake_self()
        monkeypatch.setattr(
            "nomorals.media.trendops.TrendOps",
            lambda context: SimpleNamespace())
        out = RuntimeMediaMixin._control_trendops(
            fake, "apply roadster", chat_key="telegram:123")
        assert "footage" in str(out).lower()
        assert calls["ask"] == [("telegram:123", "roadster")]

    def test_apply_without_footage_and_without_chat_fails_closed(self, monkeypatch):
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        fake, calls, _store = self._fake_self()
        monkeypatch.setattr(
            "nomorals.media.trendops.TrendOps",
            lambda context: SimpleNamespace())
        out = RuntimeMediaMixin._control_trendops(fake, "apply roadster")
        assert "usage" in str(out).lower()
        assert calls["ask"] == []

    def test_ask_answer_runs_the_apply(self):
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        fake, calls, store = self._fake_self()
        store["telegram:123"] = {"action": "apply", "format": "roadster",
                                 "at": time.time()}
        message = SimpleNamespace(
            chat=SimpleNamespace(key="telegram:123"),
            text="https://example.com/clip.mp4", media=[])
        fn = getattr(RuntimeMediaMixin, "_trendops_ask_answer", None)
        assert fn is not None
        out = fn(fake, message)
        assert calls["apply"] == [
            ("roadster", ["https://example.com/clip.mp4"])]
        assert "roadster" in str(out)
        assert "telegram:123" not in store

    def test_ask_answer_takes_a_sent_video(self):
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        fake, calls, store = self._fake_self()
        store["telegram:123"] = {"action": "apply", "format": "roadster",
                                 "at": time.time()}
        message = SimpleNamespace(
            chat=SimpleNamespace(key="telegram:123"),
            text="",
            media=[SimpleNamespace(path="/tmp/incoming/clip.mp4",
                                   kind="video")])
        out = RuntimeMediaMixin._trendops_ask_answer(fake, message)
        assert calls["apply"] == [
            ("roadster", ["/tmp/incoming/clip.mp4"])]
        assert "roadster" in str(out)

    def test_ask_answer_cancel_drops_the_ask(self):
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        fake, calls, store = self._fake_self()
        store["telegram:123"] = {"action": "apply", "format": "roadster",
                                 "at": time.time()}
        message = SimpleNamespace(
            chat=SimpleNamespace(key="telegram:123"),
            text="cancel", media=[])
        out = RuntimeMediaMixin._trendops_ask_answer(fake, message)
        assert "dropped" in str(out).lower()
        assert calls["apply"] == []
        assert "telegram:123" not in store

    def test_apply_execution_renders_and_queues_the_draft(self, monkeypatch):
        """The extracted _trendops_apply keeps the command path whole:
        real render line, real draft queue, real post buttons."""
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        class _Ops:
            def __init__(self, context):
                pass

            def apply_format(self, footage, fmt_name, music=None,
                             caption=None, beats=None):
                assert footage == "/tmp/clip.mp4"
                assert fmt_name == "roadster"
                return {"format_name": "Roadster GTA edit",
                        "output": "/tmp/out/roadster.mp4",
                        "duration": 12.5, "width": 1080, "height": 1920,
                        "segments": 4, "beats_used": 3, "review": {}}

        class _Queue:
            def __init__(self):
                self.created = []

            def create_draft(self, content, platforms=None, metadata=None):
                draft = SimpleNamespace(id="draft-1")
                self.created.append(draft)
                return draft

            def propose(self, draft_id):
                pass

            def close(self):
                pass

        queue = _Queue()
        fake = SimpleNamespace(
            context=SimpleNamespace(),
            _draft_queue=lambda: queue)
        monkeypatch.setattr("nomorals.media.trendops.TrendOps", _Ops)
        monkeypatch.setattr(
            "nomorals.media.trendops.editor.draft_for_render",
            lambda result, footage=None, music=None, caption=None: (
                {"text": "cut"}, {"kind": "trendops"}))
        monkeypatch.setattr(
            "nomorals.media.trendops.editor.stamp_render_draft",
            lambda path, draft_id: None)
        monkeypatch.setattr(
            "nomorals.social.drafts.propose_post",
            lambda draft: ("post this cut?",
                           [("📤 Post now", "draft post draft-1")]))
        out = RuntimeMediaMixin._trendops_apply(fake, "roadster",
                                                ["/tmp/clip.mp4"])
        assert "Roadster GTA edit" in out.text
        assert "/tmp/out/roadster.mp4" in out.text
        assert "post this cut?" in out.text
        assert queue.created[0].id == "draft-1"


# ── 5. settled cards clear on WhatsApp ──────────────────────────────


class TestSettleClearsButtons:
    def test_settle_passes_empty_keyboard_to_the_edit(self):
        from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

        seen: dict[str, object] = {}

        class _Adapter:
            def edit_message(self, chat, message_id, text, *,
                             buttons=None, parse_mode=""):
                seen["buttons"] = buttons
                seen["message_id"] = message_id
                return {"ok": True}

        class _Surface:
            def __init__(self):
                self.store = SimpleNamespace(save=lambda: None)

            def _extras(self, key):
                return {"proposal_msgs": {INIT_A: "mid-99"}}

        rendered = SimpleNamespace(text="decided: approved")
        fake_self = SimpleNamespace(
            _card=lambda message, card: rendered,
            _group_adapter=lambda platform: _Adapter())
        message = SimpleNamespace(chat=SimpleNamespace(platform="whatsapp"))
        ok = RuntimeGroupMixin._settle_proposal_announcement(
            fake_self, _Surface(), "key", message, INIT_A, "approved")
        assert ok is True
        assert seen["buttons"] == []
        assert seen["message_id"] == "mid-99"

    def test_whatsapp_clearing_edit_carries_the_empty_list(self):
        from nomorals.social.chat.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
        calls: list[dict] = []

        def _admin_cmd(verb, payload, timeout=None):
            calls.append({"verb": verb, "payload": payload})
            return {"ok": True}

        adapter._admin_cmd = _admin_cmd  # type: ignore[attr-defined]
        adapter._jid = lambda chat: "123@s.whatsapp.net"  # type: ignore[attr-defined]
        adapter._buttons_list_form = lambda buttons: False  # type: ignore[attr-defined]
        adapter._journal_buttons = lambda **kwargs: None  # type: ignore[attr-defined]
        chat = SimpleNamespace(chat_id="123", platform="whatsapp")
        verdict = adapter.edit_message(chat, "mid-99", "decided", buttons=[])
        assert verdict.get("ok") is True
        assert calls[0]["verb"] == "edit"
        assert calls[0]["payload"]["buttons"] == []

    def test_whatsapp_refused_clearing_revokes_and_resends(self):
        from nomorals.social.chat.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
        calls: list[dict] = []

        def _admin_cmd(verb, payload, timeout=None):
            calls.append({"verb": verb, "payload": payload})
            if verb == "edit":
                return {"ok": False, "error": "edit window expired"}
            return {"ok": True}

        sent: dict[str, object] = {}

        def _send(chat, text, **kwargs):
            sent["text"] = text
            sent["buttons"] = kwargs.get("buttons")
            return SimpleNamespace(ok=True, message_id="mid-100")

        adapter._admin_cmd = _admin_cmd  # type: ignore[attr-defined]
        adapter._jid = lambda chat: "123@s.whatsapp.net"  # type: ignore[attr-defined]
        adapter._group_jid = lambda chat: "123@s.whatsapp.net"  # type: ignore[attr-defined]
        adapter._buttons_list_form = lambda buttons: False  # type: ignore[attr-defined]
        adapter._journal_buttons = lambda **kwargs: None  # type: ignore[attr-defined]
        adapter.send = _send  # type: ignore[method-assign]
        chat = SimpleNamespace(chat_id="123", platform="whatsapp")
        verdict = adapter.edit_message(chat, "mid-99", "decided", buttons=[])
        assert verdict.get("via") == "revoke_resend"
        assert verdict.get("message_id") == "mid-100"
        assert [c["verb"] for c in calls] == ["edit", "message_delete"]
        assert sent["buttons"] == []


# ── 6. grids ────────────────────────────────────────────────────────


class TestGridPacking:
    def test_failed_scheduled_strip_packs_two_per_row(self):
        kb = adaptive_buttons_for(
            output_kind="scheduled",
            payload={"job": "day derivation", "ok": False,
                     "attempt": "run:1.0"})
        assert kb is not None
        assert [len(row) for row in kb] == [2, 1]

    def test_search_keyboard_packs_two_per_row(self):
        kb = adaptive_buttons_for(
            output_kind="search", payload={"query": "burna boy"})
        assert kb is not None
        assert all(len(row) == 2 or len(row) == 1 for row in kb)
        assert len(kb[0]) == 2

    def test_hub_keyboards_pack_two_per_row(self):
        from nomorals.social.chat.hub import hub_keyboard

        kb = hub_keyboard()
        assert [len(row) for row in kb] == [2, 1]


# ── 7. manifest ─────────────────────────────────────────────────────


class TestManifestCoversPulse:
    def test_adaptive_output_surface_lists_pulse(self):
        from nomorals.social.chat import surface_manifest as sm

        surface = sm.surface_by_id("adaptive-output")
        assert "pulse" in surface.members
