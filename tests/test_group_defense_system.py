"""The DEFENSE system (DESIGN §17 + defense brief §§1–2):
envelope producers on every converter, the anticrash organ
(structural classes, max-led subscores, signature learning),
the automatic-admin sensors (antichannel / antitag /
antidelete / antiedit), ONE escalation spine shared with manual
warns, the autonomy master switch with its platform-split
authority, and platform honesty (Bot API antidelete is
impossible; degradation audits instead of faking).

Rigs follow the house shapes: GroupSurface over a tmp_path
GroupStore; RuntimeGroupMixin stubs as the surface/economy
tests; converter units built the way the surface tests build
them.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.security.payload_inspection import (
    SignatureStore,
    fingerprint,
    inspect_payload,
    normalize_envelope,
)
from nomorals.social.chat.group_defense import (
    DefenseConfig,
    check_antichannel,
    defense_availability,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-1001234567"
WA_KEY = "whatsapp:120363012345@g.us"
NOW = 1_700_000_000.0


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _env(text="", **kw):
    return normalize_envelope(kw.pop("raw", None) or kw or None,
                              text=text)


# ── envelope: normalization + producers ─────────────────────────


def test_envelope_defaults_safely_on_garbage():
    env = normalize_envelope({"entities": "junk", "media": 42,
                              "mentions": [None, ""]})
    assert env.entities == [] and env.media is None
    assert env.mentions == [] and env.quote_depth == 0
    env = normalize_envelope(None, text="hi")
    assert env.text == "hi" and env.forward is None


def test_bot_api_envelope_from_raw_message():
    from nomorals.social.chat.telegram import _bot_message_envelope

    msg = {
        "entities": [
            {"type": "text_mention", "offset": 0, "length": 3,
             "user": {"id": 42}},
            {"type": "custom_emoji", "offset": 4, "length": 2,
             "custom_emoji_id": "99"},
            {"type": "url", "offset": 7, "length": 20}],
        "photo": [{"width": 90, "height": 90, "file_size": 100},
                  {"width": 1280, "height": 960,
                   "file_size": 5000}],
        "forward_origin": {"type": "channel",
                           "chat": {"id": -100777,
                                    "title": "Signals"}},
        "reply_to_message": {
            "message_id": 3, "text": "earlier words",
            "reply_to_message": {"message_id": 2, "text": "deep"}},
    }
    env = _bot_message_envelope(msg, "hey you there")
    assert env["mentions"] == ["42"]
    assert env["media"]["width"] == 1280
    assert env["forward"]["origin_kind"] == "channel"
    assert env["forward"]["sender_chat_id"] == "-100777"
    assert env["quote_depth"] == 2
    assert env["quoted_chars"] == len("earlier wordsdeep")


def test_bot_api_convert_attaches_envelope_and_edit_flag(tmp_path):
    from nomorals.social.chat.telegram import TelegramBotAdapter

    adapter = object.__new__(TelegramBotAdapter)
    adapter.name = "telegram-bot"
    adapter.owner_only = False
    adapter.chat_allow = set()
    adapter._bot_username = "DevonBot"
    adapter._last_inbound_was_command = {}
    adapter._last_command = {}
    adapter.partner_context = ""
    adapter.media_dir = str(tmp_path)
    base = {
        "message_id": 5, "date": int(time.time()),
        "chat": {"id": -100123, "type": "supergroup",
                 "title": "Builders"},
        "from": {"id": 7, "first_name": "Ada", "username": "ada"},
        "text": "hello",
        "entities": [{"type": "text_mention", "offset": 0,
                      "length": 5, "user": {"id": 42}}],
    }
    msg = adapter._convert({"update_id": 1, "message": base})
    assert msg.meta["envelope"]["mentions"] == ["42"]
    assert "edited" not in msg.meta
    msg = adapter._convert({"update_id": 2,
                            "edited_message": base})
    assert msg.meta.get("edited") is True
    assert msg.meta["envelope"]["mentions"] == ["42"]


def test_mtproto_envelope_from_telethon_shapes():
    from nomorals.social.chat.telegram import _mtproto_envelope

    class MessageEntityTextMention:
        offset, length, user_id = 0, 3, 42

    class PeerChannel:
        channel_id = 777

    fwd = SimpleNamespace(
        from_id=PeerChannel(), from_name="Signals",
        post_author="")
    message = SimpleNamespace(
        entities=[MessageEntityTextMention()],
        media=None, fwd_from=fwd, sender_id=7,
        reply_to=SimpleNamespace(reply_to_msg_id=9),
        reply_to_msg_id=9)
    env = _mtproto_envelope(message, "hey")
    assert env["mentions"] == ["42"]
    assert env["forward"]["origin_kind"] == "channel"
    assert env["forward"]["sender_chat_id"] == "777"
    assert env["quote_depth"] == 1


def test_whatsapp_envelope_is_pass_through_only():
    from nomorals.social.chat.whatsapp import _wa_envelope

    env = _wa_envelope({"mentions": ["111@s.whatsapp.net"],
                        "media": [{"kind": "image",
                                   "mime": "image/jpeg",
                                   "path": "/x/p.jpg"}],
                        "reply_to": "m1"}, "look")
    assert env["mentions"] == ["111@s.whatsapp.net"]
    assert env["media"]["ext"] == "jpg"
    assert env["quote_depth"] == 1
    bare = _wa_envelope({}, "plain")
    assert bare["mentions"] == [] and bare["media"] is None
    assert bare["forward"] is None


def test_discord_envelope_from_message_object():
    from nomorals.social.chat.discord import _discord_envelope

    message = SimpleNamespace(
        mentions=[SimpleNamespace(id=11), SimpleNamespace(id=22)],
        mention_everyone=False,
        attachments=[SimpleNamespace(
            content_type="image/png", filename="a.png", width=800,
            height=600, size=1234)],
        stickers=[], reference=object())
    env = _discord_envelope(message, "hi all")
    assert env["mentions"] == ["11", "22"]
    assert env["media"]["width"] == 800
    assert env["quote_depth"] == 1


# ── anticrash: each class fires, lookalikes pass ────────────────


def test_length_bomb_quarantines_and_long_post_passes():
    verdict = inspect_payload(_env("a" * 13000))
    assert verdict.action == "quarantine"
    assert verdict.classes["length"] == 1.0
    verdict = inspect_payload(_env("a thoughtful post " * 160))
    assert verdict.action == "allow"


def test_zalgo_quarantines_and_legit_accents_pass():
    zalgo = ("e" + "\u0301" * 45) * 20
    verdict = inspect_payload(_env(zalgo))
    assert verdict.action == "quarantine"
    assert verdict.classes["combining"] == 1.0
    verdict = inspect_payload(_env("cafe\u0301 re\u0301sume\u0301 nai\u0308ve — decomposed accents are normal"))
    assert verdict.action == "allow", verdict


def test_invisible_flood_quarantines_and_emoji_pass():
    verdict = inspect_payload(_env("hi" + "\u200b" * 1600))
    assert verdict.action == "quarantine"
    assert verdict.classes["invisible"] == 1.0
    family = "👨‍👩‍👧 family time 🎉"
    verdict = inspect_payload(_env(family))
    assert verdict.action == "allow", verdict


def test_mention_flood_quarantines_and_small_tag_passes():
    ids = [str(i) for i in range(45)]
    raw = {"mentions": ids,
           "entities": [{"type": "text_mention", "offset": 0,
                         "length": 1, "user_id": i} for i in ids]}
    verdict = inspect_payload(normalize_envelope(raw, text="x"))
    assert verdict.action == "quarantine"
    assert verdict.classes["entity"] == 1.0
    raw = {"mentions": ["1", "2", "3"],
           "entities": [{"type": "text_mention", "offset": 0,
                         "length": 1, "user_id": "1"}]}
    verdict = inspect_payload(normalize_envelope(raw, text="x"))
    assert verdict.action == "allow", verdict


def test_quote_and_media_bombs_and_the_mismatch_flag():
    verdict = inspect_payload(normalize_envelope(
        {"quote_depth": 25}, text="quoted"))
    assert verdict.action == "quarantine"
    assert verdict.classes["quote"] == 1.0
    verdict = inspect_payload(normalize_envelope(
        {"media": {"kind": "photo", "width": 200000,
                   "height": 10}}, text=""))
    assert verdict.action == "quarantine"
    assert verdict.classes["media"] == 1.0
    # A benign photo passes; a lying extension FLAGS (structural
    # anomaly, short of quarantine on its own).
    verdict = inspect_payload(normalize_envelope(
        {"media": {"kind": "photo", "width": 1920, "height": 1080,
                   "mime": "image/jpeg", "ext": "jpg"}}, text=""))
    assert verdict.action == "allow"
    verdict = inspect_payload(normalize_envelope(
        {"media": {"kind": "photo", "mime": "image/png",
                   "ext": "exe"}}, text=""))
    assert verdict.action == "flag"
    assert verdict.classes["media"] >= 0.8


def test_subscores_cannot_be_averaged_away():
    # One hard class at 1.0 quarantines even amid benign bulk.
    raw = {"quote_depth": 30}
    verdict = inspect_payload(normalize_envelope(
        raw, text="perfectly ordinary words here"))
    assert verdict.action == "quarantine"
    assert verdict.score == 1.0


def test_sticker_velocity_quarantine_through_the_surface(tmp_path):
    surface = _surface(tmp_path)
    env = {"sticker": {"key": "stk1", "animated": False},
           "media": {"kind": "sticker", "count": 1}}
    # The 12th identical sticker inside a minute crosses the
    # repeats hard line (12) — exactly one quarantine, one strike.
    last = None
    for i in range(12):
        last = surface.handle_message(
            KEY, "7", name="Ada", text="", message_id=str(i),
            ts=NOW + i * 2, envelope=env)
    assert surface.strikes_for(KEY, "7", now=NOW + 100)["count"] \
        == 1
    kinds = [a["kind"] for a in last["actions"]]
    assert "delete" in kinds
    audits = [a for a in surface.audit(KEY, limit=100)
              if a["kind"] == "anticrash"]
    assert audits and audits[-1]["rule"] == "anticrash"


# ── signature learning ──────────────────────────────────────────


def test_signature_promotes_at_three_distinct_senders(tmp_path):
    store = SignatureStore(tmp_path / "sig.json")
    env = _env("a" * 13000)
    sig = inspect_payload(env).signature
    assert sig == fingerprint(inspect_payload(env).features)
    store.observe(sig, "a")
    store.observe(sig, "b")
    assert not store.is_active(sig)
    store.observe(sig, "c")
    assert store.is_active(sig)
    verdict = inspect_payload(env, signatures=store)
    assert verdict.action == "quarantine"
    assert verdict.classes == {"signature": 1.0}


def test_signature_decay_and_instant_demote(tmp_path):
    store = SignatureStore(tmp_path / "sig.json")
    sig = "deadbeefdeadbeef"
    store.promote(sig, by="owner")
    entry = store.data[sig]
    entry["last_seen"] = NOW - 91 * 86400
    assert not store.is_active(sig, now=NOW)  # 90-day decay
    store.promote(sig, by="owner")
    assert store.is_active(sig)
    store.demote(sig, reason="false positive")
    assert not store.is_active(sig)
    assert store.data[sig]["false_positives"] == 1
    # a demoted shape must re-earn promotion from clean senders
    store.observe(sig, "x", now=NOW)
    store.observe(sig, "y", now=NOW)
    assert not store.is_active(sig, now=NOW)


def test_signature_store_corruption_quarantines(tmp_path):
    path = tmp_path / "sig.json"
    path.write_text("{torn json", encoding="utf-8")
    store = SignatureStore(path)
    assert store.quarantined is True
    assert store.data == {}
    assert list(tmp_path.glob("sig.corrupt-*"))


# ── pipeline: quarantine earns nothing, strikes land ────────────


def test_quarantined_message_earns_no_economy(tmp_path):
    surface = _surface(tmp_path)
    ok = surface.handle_message(KEY, "8", name="Bo", text="hello",
                                message_id="1", ts=NOW)
    assert surface.economy.is_verified(KEY, "8")
    bad = surface.handle_message(KEY, "9", name="Cy",
                                 text="a" * 13000, message_id="2",
                                 ts=NOW)
    assert [a["kind"] for a in bad["actions"]] == ["delete"]
    assert not surface.economy.is_verified(KEY, "9")
    assert surface.strikes_for(KEY, "9", now=NOW)["count"] == 1
    warning = bad["actions"][0].get("warning") or ""
    # §31 shape: mention + [numeric id] leads every action
    # notice (the owner's captured reference format) — this pin
    # previously asserted the pre-notice-layer prefix.
    assert warning.startswith("⛔ Cy [9] — ")
    assert "strike 1" in warning


def test_admin_payload_still_deleted_but_never_struck(tmp_path):
    surface = _surface(tmp_path)
    out = surface.handle_message(KEY, "7", name="Ada",
                                 text="a" * 13000, message_id="1",
                                 ts=NOW, sender_is_admin=True)
    assert [a["kind"] for a in out["actions"]] == ["delete"]
    assert surface.strikes_for(KEY, "7", now=NOW)["count"] == 0


# ── sensors: antichannel / antitag ──────────────────────────────


def test_antichannel_fires_and_linked_channel_is_exempt(tmp_path):
    surface = _surface(tmp_path)
    env = {"forward": {"is_forwarded": True,
                       "origin_kind": "channel",
                       "origin_name": "Signals",
                       "sender_chat_id": "-100777"}}
    out = surface.handle_message(KEY, "7", name="Ada", text="fwd",
                                 message_id="1", ts=NOW,
                                 envelope=env)
    assert [a["kind"] for a in out["actions"]] == ["delete"]
    assert surface.strikes_for(KEY, "7", now=NOW)["count"] == 1
    # the group's OWN linked channel: automatic forwards are the
    # feature, not the attack
    own = {"forward": {"is_forwarded": True, "is_automatic": True,
                       "origin_kind": "channel",
                       "sender_chat_id": "-100777"}}
    out = surface.handle_message(KEY, "8", name="Bo", text="post",
                                 message_id="2", ts=NOW,
                                 envelope=own)
    assert out["actions"] == []
    # and the sensor itself honours an explicit linked id
    cfg = DefenseConfig()
    hit = check_antichannel(normalize_envelope(env), cfg,
                            linked_chat_id="-100777")
    assert hit is None


def test_antichannel_newsletter_and_membership_link(tmp_path):
    surface = _surface(tmp_path)
    out = surface.handle_message(
        WA_KEY, "1234@newsletter", name="Chan", text="post",
        message_id="1", ts=NOW, envelope={})
    assert [a["kind"] for a in out["actions"]] == ["delete"]
    link = {"entities": [{"type": "url", "offset": 5, "length": 33}]}
    out = surface.handle_message(
        KEY, "7", name="Ada",
        text="join https://whatsapp.com/channel/0029AbCdEf now",
        message_id="2", ts=NOW, envelope=link)
    assert [a["kind"] for a in out["actions"]] == ["delete"]


def test_antitag_bounds_and_admin_exemption(tmp_path):
    surface = _surface(tmp_path)
    for i in range(40):  # a 40-member roster → pct bound is 8
        surface.handle_message(KEY, str(100 + i), name=f"M{i}",
                               text="hi", message_id=f"s{i}",
                               ts=NOW)
    nine = {"mentions": [str(i) for i in range(9)],
            "entities": [{"type": "text_mention", "offset": 0,
                          "length": 1, "user_id": str(i)}
                         for i in range(9)]}
    out = surface.handle_message(KEY, "7", name="Ada", text="x",
                                 message_id="m1", ts=NOW,
                                 envelope=nine)
    assert [a["kind"] for a in out["actions"]] == ["delete"]
    eight = {"mentions": [str(i) for i in range(8)],
             "entities": [{"type": "text_mention", "offset": 0,
                           "length": 1, "user_id": str(i)}
                          for i in range(8)]}
    out = surface.handle_message(KEY, "8", name="Bo", text="x",
                                 message_id="m2", ts=NOW,
                                 envelope=eight)
    assert out["actions"] == []
    out = surface.handle_message(KEY, "9", name="Cy", text="x",
                                 message_id="m3", ts=NOW,
                                 envelope=nine, sender_is_admin=True)
    assert out["actions"] == []


# ── the escalation spine ────────────────────────────────────────


def _offense(surface, user, i, ts):
    return surface.handle_message(
        KEY, user, name="Ada", text=f"badword post {i}",
        message_id=f"o{i}", ts=ts)


def test_ladder_boundaries_across_sensors(tmp_path):
    surface = _surface(tmp_path)
    surface.update_settings(KEY, moderation={
        "flood_max": 6, "flood_window_s": 10.0,
        "repeat_threshold": 3, "link_allowlist": [],
        "word_filters": ["badword"], "filter_action": "delete"})
    kinds_at = {}
    for i in range(1, 8):
        out = _offense(surface, "7", i, NOW + i * 30)
        kinds_at[i] = [a["kind"] for a in out["actions"]]
        assert surface.strikes_for(KEY, "7",
                                   now=NOW + i * 30)["count"] == i
    assert kinds_at[1] == ["delete"]
    assert "mute" in kinds_at[3]           # warn_threshold rung
    assert "kick" in kinds_at[5]
    assert "ban" in kinds_at[7]
    # exactly ONE public warning rides the batch at the rung
    warnings = [a for a in out["actions"] if a.get("warning")]
    assert len(warnings) == 1
    assert "Banned from the group" in warnings[0]["warning"]


def test_strike_window_expiry_resets_the_ladder(tmp_path):
    surface = _surface(tmp_path)
    surface.update_settings(KEY, moderation={
        "flood_max": 6, "flood_window_s": 10.0,
        "repeat_threshold": 3, "link_allowlist": [],
        "word_filters": ["badword"], "filter_action": "delete"})
    _offense(surface, "7", 1, NOW)
    _offense(surface, "7", 2, NOW + 30)
    later = NOW + 8 * 86400
    out = _offense(surface, "7", 3, later)
    assert surface.strikes_for(KEY, "7", now=later)["count"] == 1
    assert [a["kind"] for a in out["actions"]] == ["delete"]


def test_manual_warns_share_the_spine(tmp_path):
    surface = _surface(tmp_path)
    assert surface.record_warn(KEY, "7", by="9") == 1
    assert surface.record_warn(KEY, "7", by="9") == 2
    info = surface.strikes_for(KEY, "7")
    assert info["count"] == 2
    assert info["strikes"][0]["rule"] == "manual_warn"
    surface.clear_warns(KEY, "7")
    assert surface.strikes_for(KEY, "7")["count"] == 0


# ── autonomy: suppression + authority split ─────────────────────


def test_autonomy_off_suppresses_actions_but_keeps_the_audit(
        tmp_path):
    surface = _surface(tmp_path)
    surface.update_settings(KEY, moderation={
        "flood_max": 6, "flood_window_s": 10.0,
        "repeat_threshold": 3, "link_allowlist": [],
        "word_filters": ["badword"], "filter_action": "delete"})
    surface.update_settings(KEY, defense={"autonomy": False})
    out = surface.handle_message(KEY, "7", name="Ada",
                                 text="badword here", message_id="1",
                                 ts=NOW)
    assert out["actions"] == []
    assert out["suppressed"] >= 1
    audits = surface.audit(KEY, limit=20)
    mod = [a for a in audits if a["kind"] == "moderation"]
    assert mod and mod[-1].get("suppressed") == "autonomy_off"
    # the message itself was never removed, so it still earns
    assert surface.economy.is_verified(KEY, "7")
    # manual commands are untouched by the switch
    assert surface.record_warn(KEY, "7", by="9") == 1


# ── evidentiary sensors ─────────────────────────────────────────


def test_antidelete_and_antiedit_are_evidentiary(tmp_path):
    surface = _surface(tmp_path)
    surface.handle_message(KEY, "7", name="Ada",
                           text="the scam link was here",
                           message_id="m1", ts=NOW)
    out = surface.handle_delete(KEY, message_id="m1")
    assert out["cached"] is True and out["callouts"] == []
    audits = [a for a in surface.audit(KEY, limit=20)
              if a["kind"] == "antidelete"]
    assert audits and audits[-1]["excerpt"] == \
        "the scam link was here"
    surface.handle_message(KEY, "7", name="Ada",
                           text="original claim",
                           message_id="m2", ts=NOW + 10)
    out = surface.handle_edit(KEY, message_id="m2",
                              new_text="edited to look innocent")
    edits = [a for a in surface.audit(KEY, limit=20)
             if a["kind"] == "antiedit"]
    assert edits[-1]["old"] == "original claim"
    assert edits[-1]["new"] == "edited to look innocent"
    # callout mode + group target composes the public callout
    surface.update_settings(KEY, defense={
        "antidelete": {"mode": "callout", "callout_to": "group"}})
    surface.handle_message(KEY, "7", name="Ada", text="vanishing",
                           message_id="m3", ts=NOW + 20)
    out = surface.handle_delete(KEY, message_id="m3")
    assert out["callouts"] and out["callouts"][0]["to"] == "group"
    assert "vanishing" in out["callouts"][0]["text"]


# ── platform honesty ────────────────────────────────────────────


def test_platform_availability_is_honest():
    bot = defense_availability("telegram-bot", "botapi")
    assert bot["antidelete"] is False
    assert "MTProto" in bot["antidelete_note"]
    assert defense_availability("telegram", "mtproto")[
        "antidelete"] is True
    assert defense_availability("whatsapp", "baileys")[
        "antidelete"] is True
    unknown = defense_availability("smoke-signals", "drum")
    assert unknown["antidelete"] is False  # claims nothing


# ── runtime rig: warnings, degradation, autonomy authority ─────

from nomorals.social.chat.base import ChatKind
from nomorals.agents.partner.runtime_group import RuntimeGroupMixin


class _Chat:
    def __init__(self, platform="telegram-bot",
                 key=KEY):
        self.platform = platform
        self.chat_id = key.split(":", 1)[1]
        self.title = "Builders"
        self.key = key
        self.kind = ChatKind.GROUP


class _Msg:
    def __init__(self, chat=None, **kw):
        self.chat = chat or _Chat()
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = ""


class _Adapter:
    """admin_member answers with a per-id status map; admin_*
    calls are recorded.  `bare` drops the execution surface."""

    def __init__(self, statuses=None, bare=False):
        self.calls = []
        self.statuses = statuses or {}
        self.bare = bare

    def admin_member(self, chat, user_id):
        return {"status": self.statuses.get(user_id, "member")}

    def admin_delete(self, chat, message_id):
        if self.bare:
            raise AttributeError("no admin_delete")
        self.calls.append(("delete", message_id))
        return {"ok": True}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True}

    def admin_unban(self, chat, user_id):
        self.calls.append(("unban", user_id))
        return {"ok": True}

    def admin_restrict(self, chat, user_id, *, minutes=0, **kw):
        self.calls.append(("restrict", user_id, minutes))
        return {"ok": True}


class _BareAdapter:
    def admin_member(self, chat, user_id):
        return {"status": "member"}


class _Runtime(RuntimeGroupMixin):
    def __init__(self, tmp_path, adapter, platform="telegram-bot",
                 operator=False):
        self.sent = []
        self.gateway = SimpleNamespace(
            adapters={platform: adapter})
        self._operator = operator
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))
        from nomorals.social.chat.group_surface import \
            default_surface

        self._group_surface_obj = default_surface(str(tmp_path))

    def _send(self, chat, text, **kw):
        self.sent.append(str(text))

    def _is_operator(self, message):
        return self._operator


def _seed_link_rule(rt):
    rt._group_surface_obj.update_settings(
        KEY, moderation={"flood_max": 6, "flood_window_s": 10.0,
                         "repeat_threshold": 3,
                         "link_allowlist": ["example.com"],
                         "word_filters": [],
                         "filter_action": "delete"})


def test_runtime_executes_delete_and_posts_one_warning(tmp_path):
    adapter = _Adapter()
    rt = _Runtime(tmp_path, adapter)
    _seed_link_rule(rt)
    msg = _Msg(text="https://evil.example.net/x", message_id="99")
    assert rt._group_surface_turn(msg) is True
    assert ("delete", "99") in adapter.calls
    warnings = [s for s in rt.sent if s.startswith("⛔")]
    assert len(warnings) == 1
    assert "Ada" in warnings[0] and "strike 1" in warnings[0]


def test_runtime_degraded_adapter_never_claims_execution(tmp_path):
    rt = _Runtime(tmp_path, _BareAdapter())
    _seed_link_rule(rt)
    msg = _Msg(text="https://evil.example.net/x", message_id="99")
    assert rt._group_surface_turn(msg) is False
    assert not [s for s in rt.sent if s.startswith("⛔")]
    kinds = [a["kind"] for a in rt._group_surface_obj.audit(KEY)]
    assert "moderation_unexecuted" in kinds


def test_runtime_delete_service_event_feeds_antidelete(tmp_path):
    rt = _Runtime(tmp_path, _Adapter())
    rt._group_surface_turn(_Msg(text="soon gone", message_id="5"))
    out = rt._group_surface_turn(_Msg(meta={
        "service": "delete", "deleted_ids": ["5"]}))
    assert out is True
    audits = [a for a in rt._group_surface_obj.audit(KEY)
              if a["kind"] == "antidelete"]
    assert audits and audits[-1]["excerpt"] == "soon gone"


def test_autonomy_authority_split_telegram_owner_only(tmp_path):
    adapter = _Adapter(statuses={"7": "administrator",
                                 "9": "creator"})
    rt = _Runtime(tmp_path, adapter)
    out = rt._control_gsettings("autonomy off",
                                _Msg(sender_id="7"))
    assert "owner" in str(out)
    settings = rt._group_surface_obj.store.settings(KEY)
    assert settings["defense"]["autonomy"] is True
    out = rt._control_gsettings("autonomy off",
                                _Msg(sender_id="9", sender="Boss",
                                     sender_username="boss"))
    assert "autonomy off" in str(out)
    settings = rt._group_surface_obj.store.settings(KEY)
    assert settings["defense"]["autonomy"] is False


def test_autonomy_authority_split_whatsapp_any_admin(tmp_path):
    adapter = _Adapter(statuses={"7": "administrator"})
    rt = _Runtime(tmp_path, adapter, platform="whatsapp")
    chat = _Chat(platform="whatsapp", key=WA_KEY)
    out = rt._control_gsettings(
        "autonomy off", _Msg(chat=chat, sender_id="7"))
    assert "autonomy off" in str(out)
    settings = rt._group_surface_obj.store.settings(WA_KEY)
    assert settings["defense"]["autonomy"] is False
    # a plain member is refused at the ordinary admin gate
    out = rt._control_gsettings(
        "autonomy on", _Msg(chat=chat, sender_id="8",
                            sender="Bo", sender_username="bo"))
    assert "admins" in str(out)


def test_autonomy_operator_may_flip_anywhere(tmp_path):
    rt = _Runtime(tmp_path, _Adapter(), operator=True)
    out = rt._control_gsettings("autonomy off",
                                _Msg(sender_id="7"))
    assert "autonomy off" in str(out)


def test_gsettings_summary_states_bot_api_honesty(tmp_path):
    adapter = _Adapter(statuses={"7": "administrator"})
    rt = _Runtime(tmp_path, adapter)
    out = rt._control_gsettings("", _Msg(sender_id="7"))
    text = getattr(out, "text", str(out))
    assert "antidelete unavailable on the bot surface" in text
    assert "autonomy" in text and "anticrash" in text


def test_gsettings_sensor_toggles_persist(tmp_path):
    adapter = _Adapter(statuses={"7": "administrator"})
    rt = _Runtime(tmp_path, adapter)
    rt._control_gsettings("antitag off", _Msg(sender_id="7"))
    rt._control_gsettings("antitag 4", _Msg(sender_id="7"))
    rt._control_gsettings("anticrash flag", _Msg(sender_id="7"))
    defense = rt._group_surface_obj.store.settings(KEY)["defense"]
    assert defense["antitag"]["enabled"] is False
    assert defense["antitag"]["max_mentions"] == 4
    assert defense["anticrash"]["action"] == "flag"


def test_strikes_command_reads_the_spine(tmp_path):
    adapter = _Adapter(statuses={"9": "administrator"})
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    surface.handle_message(KEY, "7", username="ada", name="Ada",
                           text="hello", message_id="1", ts=NOW)
    surface.record_warn(KEY, "7", by="9", reason="spam")
    surface.record_warn(KEY, "7", by="9", reason="again")
    admin = _Msg(sender_id="9", sender="Boss",
                 sender_username="boss")
    out = rt._control_strikes("", admin)
    assert "2" in str(out)
    out = rt._control_strikes("@ada", admin)
    assert "manual_warn" in str(out)
    member = _Msg(sender_id="8", sender="Bo",
                  sender_username="bo")
    assert "admin" in str(rt._control_strikes("", member))


def test_strikes_command_registered():
    from nomorals.social.chat.control import (
        CONTROL_COMMANDS,
        LIST_GROUPS,
        caller_may_run,
        parse_control,
    )

    assert "strikes" in CONTROL_COMMANDS
    assert caller_may_run("strikes", is_owner=False)
    assert parse_control("/strikes @ada").kind == "strikes"
    groups = [items for _name, items in LIST_GROUPS
              if _name == "groups & communities"]
    assert groups and groups[0].count("strikes") == 1
