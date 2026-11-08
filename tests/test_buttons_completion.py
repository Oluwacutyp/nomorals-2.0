"""Buttons completion (2026-11-08) — buttons as a SYSTEM.

The owner's ground truth: not one inline button had ever rendered
on his WhatsApp client, the grid law contradicted itself in code
(one-per-row "house standard" vs the menu's packed grid), the
WhatsApp menu refused the session render, edits dropped keyboards,
and resolved asks left their buttons live. These pins cover the
repairs end-to-end through the real seams:

* the WhatsApp adapter: media sends carry buttons to both bridge
  verbs' contract (the payload the bridges consume), a degraded
  buttons_form is ledgered AND flips the account to the list form
  on that evidence, offline media keeps its buttons, plain sends
  derive keyboards like the Telegram bot adapter, and edits carry
  buttons — a refused keyboard-clearing edit falls back to
  revoke + a fresh settled send;
* the grid: the reflow packs two per row (the retired one-per-row
  pin lives in test_presentation_overhaul), hub + adaptive +
  anime pick keyboards pack;
* the WhatsApp /menu takes the real session render;
* resolved mute/flag asks settle their card (buttons=[]) and a
  late tap on the settled card answers as settled, never a fresh
  action; drop cards settle on close;
* /pair buttons: the render probe journals its sent form and a
  returning tap closes the loop; /pair journal shows the
  degradation count.

Bridge-side shapes are pinned in bridge/gobridge/messages_test.go
(Go) and against the Baileys source below (the .mjs ships as
source; its shapes were verified against the vendored Baileys
6.7.24 library it runs on).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind, ChatRef, MediaRef, SendResult
from nomorals.social.chat.whatsapp import WhatsAppAdapter

CHAT = ChatRef(platform="whatsapp", chat_id="234801@s.whatsapp.net",
               kind=ChatKind.DM)
KEY = "telegram-bot:-1001234567"


def _adapter(tmp_path, responder):
    """A connected WhatsAppAdapter whose bridge is a script: the
    responder sees each command dict and answers for it."""
    adapter = WhatsAppAdapter(
        outbox_dir=str(tmp_path / "outbox"))
    adapter.connected.set()
    calls: list = []

    def fake_send_cmd(cmd, timeout=25.0):
        calls.append(dict(cmd))
        return responder(cmd)

    adapter._send_cmd = fake_send_cmd  # type: ignore[method-assign]
    return adapter, calls


def _ok(cmd):
    return {"ok": True, "message_id": "M1"}


# ── 1 · WhatsApp media buttons ──────────────────────────────────


def test_media_send_carries_buttons_to_the_bridge(tmp_path):
    adapter, calls = _adapter(tmp_path, _ok)
    out = adapter.send_media(
        CHAT, MediaRef(path="/tmp/pic.jpg", kind="photo"),
        caption="a cat",
        buttons=[[("🔁 Another one", "waifu")]])
    assert out.ok
    cmd = calls[0]
    assert cmd["cmd"] == "send_media"
    assert cmd["buttons"] == [[["🔁 Another one", "waifu"]]], cmd


def test_media_buttons_form_is_tracked(tmp_path):
    adapter, _calls = _adapter(
        tmp_path,
        lambda cmd: {"ok": True, "message_id": "M1",
                     "buttons_form": "interactive"})
    adapter.send_media(
        CHAT, MediaRef(path="/tmp/pic.jpg", kind="photo"),
        buttons=[[("🔁 Another one", "waifu")]])
    assert adapter.buttons_form_state()["last_form"] == "interactive"


def test_media_degradation_ledgers_and_flips_the_form(tmp_path):
    from nomorals.social.chat import failures

    failures.clear_failures()
    adapter, calls = _adapter(
        tmp_path,
        lambda cmd: {"ok": True, "message_id": "M1",
                     "buttons_form": "text"})
    adapter.send_media(
        CHAT, MediaRef(path="/tmp/pic.jpg", kind="photo"),
        buttons=[[("🔁 Another one", "waifu")]])
    sites = [f.site for f in failures.recent_failures(50)]
    assert "whatsapp.WhatsAppAdapter.send.buttons" in sites, sites
    # The evidence flips the account: the next TEXT send asks the
    # bridge for the list form.
    adapter.send(CHAT, "pick one",
                 buttons=[[("A", "a"), ("B", "b")]])
    send_cmd = [c for c in calls if c["cmd"] == "send"][0]
    assert send_cmd.get("buttons_style") == "list", send_cmd
    assert adapter.buttons_form_state()["preference"] == "list"


def test_media_offline_enqueue_keeps_buttons_and_replays(tmp_path):
    adapter = WhatsAppAdapter(outbox_dir=str(tmp_path / "outbox"))
    out = adapter.send_media(
        CHAT, MediaRef(path="/tmp/pic.jpg", kind="photo"),
        caption="cap", buttons=[[("🔁 Another one", "waifu")]])
    assert not out.ok and "queued" in (out.error or "")
    entries = adapter.outbox.pending()
    assert entries and entries[0]["buttons"] == [
        [["🔁 Another one", "waifu"]]], entries
    # The reconnect flush replays the media WITH its buttons.
    calls: list = []
    adapter.connected.set()
    adapter._send_cmd = (  # type: ignore[method-assign]
        lambda cmd, timeout=25.0: (calls.append(dict(cmd)),
                                   {"ok": True,
                                    "message_id": "M9"})[1])
    assert adapter._deliver_entry(dict(entries[0])) is True
    media_cmd = [c for c in calls if c["cmd"] == "send_media"][0]
    assert media_cmd["buttons"] == [[["🔁 Another one", "waifu"]]]


# ── 6 · derivation parity ───────────────────────────────────────


def _inbound(adapter, text, chat_id="234801@s.whatsapp.net"):
    # _track_inbound duck-types on chat.chat_id + text.
    msg = SimpleNamespace(
        chat=SimpleNamespace(chat_id=chat_id), text=text)
    adapter._track_inbound(msg)
    return msg


def test_plain_send_derives_buttons_from_the_fresh_command(tmp_path):
    adapter, calls = _adapter(tmp_path, _ok)
    _inbound(adapter, "/search cats")
    adapter.send(CHAT, "here are the results")
    cmd = calls[0]
    flat = [pair for row in cmd.get("buttons") or [] for pair in row]
    # The personal set for a /search origin (the same derivation
    # the Telegram bot adapter performs for this exact turn).
    assert ["🔁 Search again", "search cats"] in flat, cmd
    assert ["🔬 Deep research", "searchdeep cats"] in flat, cmd


def test_explicit_buttons_win_and_empty_stays_empty(tmp_path):
    adapter, calls = _adapter(tmp_path, _ok)
    _inbound(adapter, "/search cats")
    adapter.send(CHAT, "here are the results",
                 buttons=[[("Mine", "mine")]])
    assert calls[0]["buttons"] == [[["Mine", "mine"]]]
    adapter.send(CHAT, "plain", buttons=[])
    assert "buttons" not in calls[1], calls[1]


def test_stale_origin_lends_no_buttons(tmp_path):
    adapter, calls = _adapter(tmp_path, _ok)
    _inbound(adapter, "/search cats")
    _inbound(adapter, "just chatting now")
    adapter.send(CHAT, "here are the results")
    assert "buttons" not in calls[0], calls[0]


# ── 4 · keyboard-carrying edits ─────────────────────────────────


def test_edit_carries_buttons_and_the_clear(tmp_path):
    adapter, calls = _adapter(tmp_path, _ok)
    out = adapter.edit_message(CHAT, "M1", "settled",
                               buttons=[[("A", "a")]])
    assert out["ok"]
    assert calls[0]["buttons"] == [[["A", "a"]]]
    out = adapter.edit_message(CHAT, "M1", "settled", buttons=[])
    assert out["ok"]
    # Presence-significant: the empty keyboard is IN the payload.
    assert calls[1]["buttons"] == [], calls[1]
    out = adapter.edit_message(CHAT, "M1", "settled")
    assert "buttons" not in calls[2], calls[2]


def test_refused_clearing_edit_falls_back_to_revoke_resend(tmp_path):
    def responder(cmd):
        if cmd["cmd"] == "edit":
            return {"ok": False, "error": "edit window expired"}
        return {"ok": True, "message_id": "NEW1"}

    adapter, calls = _adapter(tmp_path, responder)
    out = adapter.edit_message(CHAT, "M1", "✅ settled", buttons=[])
    assert out["ok"] and out.get("via") == "revoke_resend", out
    assert out["message_id"] == "NEW1"
    assert [c["cmd"] for c in calls] == [
        "edit", "message_delete", "send"], calls
    # The re-send is the settled render with NO keyboard.
    assert "buttons" not in calls[2] or calls[2].get(
        "buttons") in (None, []), calls[2]


def test_refused_keyboarded_edit_stays_a_refusal(tmp_path):
    def responder(cmd):
        if cmd["cmd"] == "edit":
            return {"ok": False, "error": "edit window expired"}
        return {"ok": True, "message_id": "NEW1"}

    adapter, calls = _adapter(tmp_path, responder)
    out = adapter.edit_message(CHAT, "M1", "new",
                               buttons=[[("A", "a")]])
    assert not out["ok"]
    assert [c["cmd"] for c in calls] == ["edit"], calls


# ── 2 · the grid ────────────────────────────────────────────────


def test_hub_keyboards_are_packed():
    from nomorals.social.chat import hub

    rows = hub.hub_keyboard()
    assert [len(r) for r in rows] == [2, 1], rows
    assert [len(r) for r in hub.proxy_keyboard()] == [2]
    assert [len(r) for r in hub.proxy_result_keyboard()] == [2]
    assert [len(r) for r in hub.proxy_cooldown_keyboard()] == [2]


def test_adaptive_keyboards_pack_authored_pairs():
    from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

    kb = adaptive_buttons_for(
        output_kind="search", payload={"query": "cats"})
    assert kb, "the search reply should derive buttons"
    assert all(len(row) <= 2 for row in kb), kb
    assert any(len(row) == 2 for row in kb), kb


def test_anime_picks_pack_two_per_row():
    from types import SimpleNamespace as NS

    from nomorals.anime.present import present_outcome

    outcome = NS(ok=True, title="Waifu", icon="🎌", error="",
                 lines=["one"], stats=None, note="",
                 picks=[("A", "waifu"), ("B", "waifu"),
                        ("C", "waifu")])
    out = present_outcome(outcome)
    assert [len(r) for r in out.buttons] == [2, 1], out.buttons


# ── 3 · the WhatsApp menu takes the session render ──────────────


def test_menu_session_reply_renders_for_whatsapp():
    from nomorals.agents.partner_runtime import PartnerRuntime

    rt = PartnerRuntime.__new__(PartnerRuntime)
    chat = ChatRef(platform="whatsapp",
                   chat_id="234801@s.whatsapp.net", kind=ChatKind.DM)
    msg = SimpleNamespace(chat=chat, meta={}, text="/menu")
    command = SimpleNamespace(tail="", arg="", kind="menu")
    render = rt._menu_session_reply(msg, chat.key, command)
    assert render is not None, "WhatsApp /menu must session-render"
    assert render.session == "menu"
    assert render.buttons, "the WhatsApp menu carries its keyboard"


# ── 5 · resolved asks settle their cards ────────────────────────


class _GroupAdapter:
    """The menugrid rig's recording adapter shape."""

    def __init__(self, admins):
        self.admins = set(admins)
        self.edits: list = []
        self.restricts: list = []

    def admin_member(self, chat, user_id):
        if str(user_id) in self.admins:
            return {"status": "administrator"}
        return {"status": "member"}

    def admin_restrict(self, chat, user_id, minutes=None,
                       until=None):
        self.restricts.append((str(user_id), minutes))
        return {"ok": True}

    def edit_message(self, chat, message_id, text, *, buttons=None,
                     parse_mode=""):
        self.edits.append((message_id, text, buttons))
        return SendResult(ok=True, platform="whatsapp",
                          message_id=message_id)


def _group_rig(tmp_path, platform="telegram-bot"):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
    from nomorals.social.chat.group_surface import GroupStore, GroupSurface
    from nomorals.social.chat.live_session import LiveSessionRegistry

    adapter = _GroupAdapter(("admin2",))
    registry = LiveSessionRegistry()

    class _RT(RuntimeGroupMixin):
        def _group_adapter(self, platform):  # type: ignore[no-untyped-def]
            return adapter

        def _is_operator(self, message):  # type: ignore[no-untyped-def]
            return False

        def _live_session_registry(self):  # type: ignore[no-untyped-def]
            return registry

    rt = _RT()
    surface = GroupSurface(
        GroupStore(tmp_path / "g.json").load())
    rt._group_surface_obj = surface
    rt.gateway = SimpleNamespace(adapters={platform: adapter})
    return rt, surface, adapter, registry


def _gmsg(*, sender_id="admin2", sender="Admin", tapped="",
          platform="telegram-bot"):
    meta = {"tapped_message_id": tapped} if tapped else {}
    return SimpleNamespace(
        chat=SimpleNamespace(key=KEY, platform=platform,
                             kind=ChatKind.GROUP, title="Builders"),
        sender_id=sender_id, sender=sender,
        sender_username="boss", text="", meta=meta,
        reply_to="", message_id="1")


def _seed_member(surface, uid="777", username="xander",
                 name="Xander"):
    surface.handle_message(KEY, uid, username=username, name=name,
                           text="hello group",
                           message_id=f"seed-{uid}", ts=1_700_000_000.0)


def test_mute_ask_declares_a_session_and_settles_on_tap(tmp_path):
    rt, surface, adapter, registry = _group_rig(tmp_path)
    _seed_member(surface)
    ask = rt._control_mute("@xander", _gmsg())
    assert ask.session == "modask", ask
    # The funnel records the ask's live message under its session.
    registry.record(KEY, "modask", "555")
    out = rt._control_mute("@xander 2h", _gmsg(tapped="555"))
    assert "muted for 2h" in out, out
    assert adapter.restricts == [("777", 120)]
    # The card settled: one edit, settled text, NO buttons.
    assert len(adapter.edits) == 1, adapter.edits
    mid, text, buttons = adapter.edits[0]
    assert mid == "555" and "settled" in text and buttons == []
    tomb = (surface._extras(KEY).get("settled_ask_msgs") or {})
    assert "555" in tomb, tomb
    assert "mute @xander 1d" in tomb["555"]["payloads"]


def test_late_tap_on_the_settled_mute_card_never_executes(tmp_path):
    rt, surface, adapter, registry = _group_rig(tmp_path)
    _seed_member(surface)
    rt._control_mute("@xander", _gmsg())
    registry.record(KEY, "modask", "555")
    rt._control_mute("@xander 2h", _gmsg(tapped="555"))
    assert adapter.restricts == [("777", 120)]
    # A second tap on the (now keyboard-less) card, e.g. a stale
    # client re-delivering it, answers as settled.
    out = rt._control_mute("@xander 1d", _gmsg(tapped="555"))
    assert "already settled" in out, out
    assert adapter.restricts == [("777", 120)], adapter.restricts


def test_flag_ask_settles_on_stage(tmp_path):
    rt, surface, adapter, registry = _group_rig(tmp_path)
    _seed_member(surface)
    ask = rt._control_flag("@xander", _gmsg())
    assert not isinstance(ask, str) and ask.session == "modask"
    registry.record(KEY, "modask", "556")
    out = rt._control_flag("@xander scam", _gmsg(tapped="556"))
    assert surface.flag_proposals(KEY), "the flag staged"
    assert len(adapter.edits) == 1, adapter.edits
    mid, text, buttons = adapter.edits[0]
    assert mid == "556" and buttons == []
    out = rt._control_flag("@xander fraud", _gmsg(tapped="556"))
    assert "already settled" in out, out
    assert len(surface.flag_proposals(KEY)) == 1


def test_drop_card_settles_on_close(tmp_path):
    rt, surface, adapter, registry = _group_rig(tmp_path)
    # Funded the wiring suite's way: verify + an under-cap award.
    surface.economy.verify_member(KEY, "admin2")
    surface.economy.award(KEY, "admin2", 150, "seed",
                          now=1_800_000_000.0)
    card = rt._control_drop("100 50", _gmsg())
    assert not isinstance(card, str), card
    # The /drop card declared its session.
    assert card.session and card.session.startswith(
        "dropcard:"), card.session
    drop_id = card.session.split(":", 1)[1]
    registry.record(KEY, card.session, "557")
    _seed_member(surface, uid="88", username="ada", name="Ada")
    _seed_member(surface, uid="99", username="bola", name="Bola")
    msg88 = _gmsg(sender_id="88", sender="Ada")
    msg99 = _gmsg(sender_id="99", sender="Bola")
    rt._control_claim(msg88)
    out = rt._control_claim(msg99)
    assert isinstance(out, str) or hasattr(out, "text")
    settled = [e for e in adapter.edits if e[0] == "557"]
    assert settled and settled[0][2] == [], adapter.edits
    assert drop_id


# ── 7 · the render probe ────────────────────────────────────────


class _ProbeAdapter:
    def __init__(self):
        self.user_jid = "234801@s.whatsapp.net"
        self._pref = ""
        self._last = ""
        self.sends: list = []

    def buttons_form_state(self):
        return {"preference": self._pref or "quick",
                "last_form": self._last}

    def set_buttons_form_preference(self, style):
        self._pref = "list" if style == "list" else ""
        return self._pref or "quick"

    def send(self, chat, text, *, buttons=None, **kw):
        self.sends.append((chat, text, buttons))
        self._last = "list" if self._pref == "list" \
            else "interactive"
        return SendResult(ok=True, platform="whatsapp",
                          message_id="P1")


def _pair_rig(tmp_path):
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin
    from nomorals.social.pairing import AccountRecord, PairingRegistry
    from nomorals.social.personal import PersonalSurfaces, PersonalVault

    registry = PairingRegistry(tmp_path / "paired.json")
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    personal = PersonalSurfaces(
        registry, PersonalVault(tmp_path / "personal"))
    adapter = _ProbeAdapter()
    rt = object.__new__(RuntimeProvisionMixin)
    rt._pair_supervisor = SimpleNamespace(registry=registry)
    rt._pair_personal_svc = {"registry": registry,
                             "personal": personal}
    rt.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    rt.gateway = SimpleNamespace(adapters={"whatsapp": adapter})
    return rt, personal, adapter


def _journal_rows(vault):
    return [json.loads(line)
            for line in Path(vault.journal_path).read_text(
                encoding="utf-8").splitlines() if line.strip()]


def test_probe_send_journals_the_form_and_the_tap_closes(tmp_path):
    rt, personal, adapter = _pair_rig(tmp_path)
    out = rt._pair_buttons([])
    assert "probe sent" in out, out
    assert len(adapter.sends) == 1
    _chat, _text, buttons = adapter.sends[0]
    flat = [p for row in buttons for p in row]
    assert [p[1] for p in flat] == [
        "pair buttons tap quick 1", "pair buttons tap quick 2",
        "pair buttons tap quick 3"], flat
    rows = _journal_rows(personal.vault)
    sent = [r for r in rows if r["kind"] == "button_probe"]
    assert sent and sent[-1]["phase"] == "sent"
    assert sent[-1]["form"] == "interactive", sent
    out = rt._pair_buttons(["tap", "quick", "2"])
    assert "loop closed" in out, out
    rows = _journal_rows(personal.vault)
    taps = [r for r in rows
            if r["kind"] == "button_probe" and r["phase"] == "tap"]
    assert taps and taps[-1]["form"] == "quick", taps


def test_probe_list_form_uses_the_list_preference(tmp_path):
    rt, personal, adapter = _pair_rig(tmp_path)
    out = rt._pair_buttons(["list"])
    assert "probe sent" in out, out
    _chat, _text, buttons = adapter.sends[0]
    flat = [p for row in buttons for p in row]
    assert flat[0][1] == "pair buttons tap list 1", flat
    rows = _journal_rows(personal.vault)
    sent = [r for r in rows if r["kind"] == "button_probe"]
    assert sent[-1]["form"] == "list", sent


def test_probe_refuses_honestly_without_a_session(tmp_path):
    rt, personal, adapter = _pair_rig(tmp_path)
    adapter.user_jid = ""
    out = rt._pair_buttons([])
    assert "isn't connected" in out, out
    rows = _journal_rows(personal.vault)
    assert rows and rows[-1]["phase"] == "refused", rows


def test_pair_journal_renders_probe_rows_and_the_count(tmp_path):
    from nomorals.social.chat import failures

    rt, personal, adapter = _pair_rig(tmp_path)
    rt._pair_buttons([])
    failures.clear_failures()
    failures.record_failure(
        "whatsapp.WhatsAppAdapter.send.buttons",
        RuntimeError("buttons degraded to text action lines"))
    out = rt._control_pair("journal")
    assert "button probe" in out, out
    assert "degraded to text in the failure ledger: 1" in out, out
    out = rt._control_pair("journal button_probe")
    assert "form interactive" in out, out


# ── 8 · stale claims + the Baileys contract ─────────────────────


def test_platform_limits_names_the_real_edit_scope():
    from nomorals.social.chat.platform_limits import for_platform

    scope = for_platform("whatsapp").text_edit_scope
    assert "has no edit command" not in scope
    assert "Baileys" in scope and "keyboard" in scope


def test_baileys_bridge_carries_the_full_contract():
    src = (Path(__file__).resolve().parent.parent
           / "bridge" / "whatsapp-bridge.mjs").read_text(
               encoding="utf-8")
    # list form + style switch on sends
    assert "buildListMessage" in src and "buttons_style" in src
    # media sends take buttons (interactive header) + report form
    assert "prepareWAMessageMedia" in src
    assert "buttonsForm" in src
    # keyboard-carrying edits wrap the replacement in MESSAGE_EDIT
    assert "MESSAGE_EDIT" in src
    # the send_media reply surfaces buttons_form like send's does
    assert "out.buttons_form" in src
