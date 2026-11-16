"""Presence watch, wired into production.

Census §6 superhuman #4: alert the owner when a named contact posts a
status, comes online, or joins a channel she follows, across both
platforms. The scout verified it OPEN 2026-11-16: zero hits for any
watch/notify-on-presence organ.

Every test here drives the PRODUCTION wiring: the durable watch store
under the home's data/ directory and the /presence control — never the
organ in isolation where a seam exists. Teeth: on the pristine base
each test fails for the missing organ (ModuleNotFoundError) and the
missing wiring (no _control_presence, no "presence" command kind).

Signal matrix (verified from the real repo before building):
* WhatsApp status — live: the status_post service branch in
  runtime_provision.py calls observe_status_post per post.
* Telegram status — the story sweep: the housekeeping beat drives
  cycle_stories("telegram"), which returns the newly saved stories;
  a story watch fires when status saving is on (the toggle that
  preserves stories at all).
* Telegram online — live: Telethon UserUpdate events carry the
  transition; the adapter forwards it in the service message meta.
* WhatsApp online — NOT observable: neither bridge forwards contact
  presence (both only assert the account's own presence, §15). A
  WhatsApp online watch is refused at add time with the wall named,
  never silently stored.
* Join — live: the service == "join" handler in runtime_group.py
  records every admitted arrival (inert record_join path and the
  acceptance_on_join path); the watch fires only for admitted
  arrivals, mirroring the reputation hook's gate.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    detailed_help,
    help_text,
)
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.personal_commands import PERSONAL_OWNER_COMMANDS
from nomorals.social.presence_watch import (
    PresenceWatchStore,
    supports,
)


def _store(tmp_path):
    return PresenceWatchStore(Path(tmp_path) / "data" / "presence_watch.json")


def _reach_host(tmp_path):
    """Production control host on a throwaway home (family harness)."""
    from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

    host = RuntimeReachMixin()
    host.context = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    return host


def _make_runtime(tmp_path):
    """Real PartnerRuntime on a throwaway home (registry harness)."""
    settings = load_settings(overrides={
        "home": str(tmp_path),
        "partner.platforms": "telegram",
        "partner.owner_chats": "5478650254",
        "partner.owner_sender_ids": "5478650254",
    })
    context = build_context(settings, with_executor=False, with_tools=False)

    class _FakeAdapter(ChatAdapter):
        def __init__(self, name: str) -> None:
            super().__init__(media_dir="/tmp/nm-test-media")
            self.name = name
            self.sent: list[str] = []

        def run(self, handler) -> None:
            pass

        def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
                 buttons=None, **kwargs) -> SendResult:
            self.sent.append(text)
            return SendResult(ok=True, platform=self.name,
                              message_id=f"m{len(self.sent)}")

    adapter = _FakeAdapter("telegram")
    gateway = ChatGateway({"telegram": adapter}, db=context.db)
    return PartnerRuntime(context, gateway=gateway)


# ── durable store ───────────────────────────────────────────────────


def test_watch_persists_across_instances(tmp_path):
    store = _store(tmp_path)
    watch, refused = store.add_watch("whatsapp", "2348065574087",
                                     ["status", "join"])
    assert refused == []
    assert watch["contact"] == "2348065574087"

    fresh = _store(tmp_path)
    listed = fresh.list_watches()
    assert len(listed) == 1
    assert listed[0]["platform"] == "whatsapp"
    assert listed[0]["signals"] == ["status", "join"]


def test_unknown_signal_rejected(tmp_path):
    store = _store(tmp_path)
    watch, refused = store.add_watch("telegram", "@zoey", ["bogus"])
    assert watch is None
    assert refused and "bogus" in refused[0][0]
    assert "status" in refused[0][1] and "online" in refused[0][1]
    assert store.list_watches() == []


def test_whatsapp_online_unsupported(tmp_path):
    assert supports("whatsapp", "online") is False
    assert supports("whatsapp", "status") is True
    assert supports("telegram", "online") is True
    store = _store(tmp_path)
    watch, refused = store.add_watch("whatsapp", "2348065574087",
                                     ["online"])
    assert watch is None
    assert refused and "bridge" in refused[0][1].lower()
    assert store.list_watches() == []


def test_match_by_digits(tmp_path):
    store = _store(tmp_path)
    store.add_watch("whatsapp", "2348065574087", ["status"])
    hits = store.match_watches(
        "whatsapp", "status", "2348065574087@s.whatsapp.net", "", "Adaeze")
    assert len(hits) == 1
    # A LID form of the same number matches too.
    hits = store.match_watches(
        "whatsapp", "status", "2348065574087@lid", "", "Adaeze")
    assert len(hits) == 1


def test_match_by_username_and_name(tmp_path):
    store = _store(tmp_path)
    store.add_watch("telegram", "@Zoey", ["online"])
    assert len(store.match_watches(
        "telegram", "online", "987654", "zoey", "Zoey")) == 1
    store.add_watch("telegram", "Adaeze", ["join"])
    assert len(store.match_watches(
        "telegram", "join", "123", "", "adaeze")) == 1


def test_no_cross_platform_match(tmp_path):
    store = _store(tmp_path)
    store.add_watch("whatsapp", "2348065574087", ["status"])
    assert store.match_watches(
        "telegram", "status", "2348065574087", "", "Adaeze") == []


def test_status_episode_dedupes(tmp_path):
    store = _store(tmp_path)
    store.add_watch("whatsapp", "2348065574087", ["status"])
    first = store.check(
        "whatsapp", "status", user_id="2348065574087@s.whatsapp.net",
        username="", name="Adaeze", episode_id="msg-1")
    assert first is not None
    assert first["signal"] == "status"
    assert first["label"] == "Adaeze"
    # Same post replayed (arrival ledger replay / sweep) never re-alerts.
    assert store.check(
        "whatsapp", "status", user_id="2348065574087@s.whatsapp.net",
        username="", name="Adaeze", episode_id="msg-1") is None
    # A new post alerts again.
    second = store.check(
        "whatsapp", "status", user_id="2348065574087@s.whatsapp.net",
        username="", name="Adaeze", episode_id="msg-2")
    assert second is not None


def test_online_transition_alerts_once(tmp_path):
    store = _store(tmp_path)
    store.add_watch("telegram", "@zoey", ["online"])
    # First-seen online alerts: Devon just learned she is online.
    first = store.check("telegram", "online", user_id="987654",
                        username="zoey", name="Zoey", episode_id="")
    assert first is not None
    assert first["signal"] == "online"
    # Still online (re-fired update) stays silent.
    assert store.check("telegram", "online", user_id="987654",
                       username="zoey", name="Zoey", episode_id="") is None
    # Went offline: no alert, but the state resets.
    assert store.note_offline("telegram", user_id="987654",
                              username="zoey", name="Zoey") is True
    assert store.check("telegram", "online", user_id="987654",
                       username="zoey", name="Zoey", episode_id="") is not None


def test_join_episode_dedupes(tmp_path):
    store = _store(tmp_path)
    store.add_watch("telegram", "@zoey", ["join"])
    first = store.check("telegram", "join", user_id="987654",
                        username="zoey", name="Zoey",
                        episode_id="telegram:42:1700000000",
                        group="Gold Signals")
    assert first is not None
    assert first["group"] == "Gold Signals"
    assert store.check("telegram", "join", user_id="987654",
                       username="zoey", name="Zoey",
                       episode_id="telegram:42:1700000000",
                       group="Gold Signals") is None
    second = store.check("telegram", "join", user_id="987654",
                         username="zoey", name="Zoey",
                         episode_id="telegram:42:1700003600",
                         group="Gold Signals")
    assert second is not None


def test_corrupt_store_quarantined(tmp_path):
    path = Path(tmp_path) / "data" / "presence_watch.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    store = PresenceWatchStore(path)
    assert store.list_watches() == []
    assert store.corrupted is not None
    assert Path(store.corrupted).exists()
    assert path.exists() is False or path.read_text() == "{not json"


def test_unwatch_removes_signal_then_watch(tmp_path):
    store = _store(tmp_path)
    store.add_watch("telegram", "@zoey", ["status", "online", "join"])
    assert store.remove_watch("telegram", "@zoey", ["online"]) is True
    assert store.list_watches()[0]["signals"] == ["status", "join"]
    assert store.remove_watch("telegram", "@zoey") is True
    assert store.list_watches() == []
    assert store.remove_watch("telegram", "@zoey") is False


# ── chat surface ────────────────────────────────────────────────────


def test_presence_list_empty(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_presence("list", chat_key="telegram:5478650254")
    assert "no presence watches" in out.lower()


def test_watch_add_and_list(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_presence(
        "watch @zoey online join", chat_key="telegram:5478650254")
    assert "watching" in out.lower()
    assert "zoey" in out.lower()
    listed = host._control_presence("list", chat_key="telegram:5478650254")
    assert "@zoey" in listed or "zoey" in listed
    assert "online" in listed and "join" in listed


def test_watch_whatsapp_online_refused(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_presence(
        "watch 2348065574087 online", chat_key="whatsapp:2348065574087")
    assert "can't" in out.lower() or "cannot" in out.lower()
    assert "bridge" in out.lower()
    assert host._presence_watch_store().list_watches() == []


def test_watch_bad_signal(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_presence(
        "watch @zoey bogus", chat_key="telegram:5478650254")
    assert "bogus" in out
    assert "status" in out and "online" in out and "join" in out


def test_unwatch_missing(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_presence(
        "unwatch @nobody", chat_key="telegram:5478650254")
    assert "no watch" in out.lower()


def test_bare_presence_returns_usage(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_presence("", chat_key="telegram:5478650254")
    assert "/presence watch" in out
    assert out.strip() != ""


def test_presence_registered_owner_only(tmp_path):
    assert "presence" in CONTROL_COMMANDS
    assert "presence" not in PUBLIC_COMMANDS
    assert "presence" not in PERSONAL_OWNER_COMMANDS
    page = COMMAND_DETAILS["presence"]
    for field in ("what", "usage", "example"):
        assert str(page.get(field, "")).strip(), f"presence DETAILS.{field}"
    assert LIST_ONELINERS.get("presence", "").strip()
    slots = [kinds for _title, kinds in LIST_GROUPS if "presence" in kinds]
    assert len(slots) == 1
    assert "/presence" in help_text()
    assert "watch" in detailed_help("presence").lower()


# ── live hooks ──────────────────────────────────────────────────────


def test_status_hook_alerts_once_per_post(tmp_path):
    host = _reach_host(tmp_path)
    host._control_presence(
        "watch 2348065574087 status", chat_key="whatsapp:2348065574087")
    delivered = []
    with mock.patch("nomorals.agents.notifier.notify",
                    side_effect=lambda ctx, kind, title, body, **kw: (
                        delivered.append((kind, title, body)), {})) as _n:
        first = host._presence_watch_event(
            "whatsapp", "status",
            user_id="2348065574087@s.whatsapp.net",
            username="", name="Adaeze", episode_id="msg-1")
        second = host._presence_watch_event(
            "whatsapp", "status",
            user_id="2348065574087@s.whatsapp.net",
            username="", name="Adaeze", episode_id="msg-1")
    assert len(first) == 1
    assert len(second) == 0
    assert len(delivered) == 1
    assert delivered[0][0] == "presence_watch"
    assert "Adaeze" in delivered[0][1]
    assert "status" in delivered[0][1].lower()


def test_online_hook_via_user_update(tmp_path):
    runtime = _make_runtime(tmp_path)
    runtime.handle_control(
        "/presence watch @zoey online", "telegram:5478650254")
    delivered = []

    def _fake_notify(ctx, kind, title, body, **kw):
        delivered.append((kind, title, body))
        return {}

    msg = ChatMessage(
        chat=ChatRef(platform="telegram", chat_id="987654",
                     kind=ChatKind.DM, title="", peer="987654"),
        incoming=True, text="", sender="Zoey", sender_id="987654",
        meta={"service": "user_update", "user_id": "987654",
              "username": "zoey", "name": "Zoey", "online": "online",
              "envelope": None})
    with mock.patch("nomorals.agents.notifier.notify",
                    side_effect=_fake_notify):
        runtime._handle_user_update(msg)
        runtime._handle_user_update(msg)
    assert len(delivered) == 1
    assert delivered[0][0] == "presence_watch"
    assert "Zoey" in delivered[0][1]
    assert "online" in delivered[0][1].lower()


def test_join_hook_alert_names_group(tmp_path):
    host = _reach_host(tmp_path)
    host._control_presence(
        "watch @zoey join", chat_key="telegram:5478650254")
    delivered = []
    with mock.patch("nomorals.agents.notifier.notify",
                    side_effect=lambda ctx, kind, title, body, **kw: (
                        delivered.append((kind, title, body)), {})):
        alerts = host._presence_watch_event(
            "telegram", "join", user_id="987654", username="zoey",
            name="Zoey", episode_id="telegram:42:1700000000",
            group="Gold Signals")
    assert len(alerts) == 1
    assert len(delivered) == 1
    assert "Zoey" in delivered[0][1]
    assert "Gold Signals" in delivered[0][1]


def test_no_watch_no_notify(tmp_path):
    host = _reach_host(tmp_path)
    delivered = []
    with mock.patch("nomorals.agents.notifier.notify",
                    side_effect=lambda ctx, kind, title, body, **kw: (
                        delivered.append((kind, title, body)), {})):
        alerts = host._presence_watch_event(
            "whatsapp", "status",
            user_id="2348065574087@s.whatsapp.net",
            username="", name="Adaeze", episode_id="msg-1")
    assert alerts == []
    assert delivered == []


def test_hook_never_raises_on_corrupt_store(tmp_path):
    host = _reach_host(tmp_path)
    path = Path(tmp_path) / "data" / "presence_watch.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    delivered = []
    with mock.patch("nomorals.agents.notifier.notify",
                    side_effect=lambda ctx, kind, title, body, **kw: (
                        delivered.append((kind, title, body)), {})):
        alerts = host._presence_watch_event(
            "whatsapp", "status",
            user_id="2348065574087@s.whatsapp.net",
            username="", name="Adaeze", episode_id="msg-1")
    assert alerts == []
    assert delivered == []


def test_platform_prefix_overrides_arrival(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_presence(
        "watch wa:2348065574087 status", chat_key="telegram:5478650254")
    assert "watching" in out.lower()
    watches = host._presence_watch_store().list_watches()
    assert len(watches) == 1
    assert watches[0]["platform"] == "whatsapp"


def test_user_update_facts_carry_presence_transition():
    from nomorals.social.chat.telegram import TelegramAdapter

    online = SimpleNamespace(
        user=SimpleNamespace(id=7, username="ada_ng", first_name="Ada",
                             last_name="N."),
        online=True, offline=False)
    assert TelegramAdapter._user_update_facts(online)["online"] == "online"
    offline = SimpleNamespace(
        user=SimpleNamespace(id=7, username="ada_ng", first_name="Ada",
                             last_name="N."),
        online=False, offline=True)
    assert TelegramAdapter._user_update_facts(offline)["online"] == "offline"
    typing = SimpleNamespace(
        user=SimpleNamespace(id=7, username="ada_ng", first_name="Ada",
                             last_name="N."))
    assert TelegramAdapter._user_update_facts(typing)["online"] == ""


# ── slim-host regression ──────────────────────────────────────────
# The join hook lives on RuntimeGroupMixin; the watch helper lives
# on RuntimeReachMixin.  A host carrying only the group mixin (the
# test_group_surface.py shape) must still consume a join — the
# sensor is a no-op there, never an AttributeError.  This test
# failed before the getattr guard was added (the battery caught
# it: test_runtime_join_consumed_with_welcome raised
# AttributeError: '_Runtime' object has no attribute
# '_presence_watch_event').


class _SlimChat:
    platform = "telegram-bot"
    chat_id = "-100123"
    title = "Builders"
    key = "telegram-bot:-100123"

    from nomorals.social.chat.base import ChatKind as _CK
    kind = _CK.GROUP


class _SlimMsg:
    def __init__(self, **kw):
        self.chat = _SlimChat()
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")


class _SlimAdapter:
    def __init__(self):
        self.calls = []

    def admin_member(self, chat, user_id):
        return {"ok": True, "result": {"status": "member"}}


class _SlimGateway:
    def __init__(self, adapter):
        self.adapters = {"telegram-bot": adapter}


def test_join_on_slim_host_without_reach_mixin(tmp_path):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    class _SlimRuntime(RuntimeGroupMixin):
        """RuntimeGroupMixin only — no reach mixin, no watch helper."""

        def __init__(self, tmp_path):
            self.sent = []
            self.gateway = _SlimGateway(_SlimAdapter())
            self.context = SimpleNamespace(settings=SimpleNamespace(
                workspace_dir=str(tmp_path)))

        def _send(self, chat, text, **kw):
            self.sent.append(text)

        def _is_operator(self, message):
            return False

        def turn(self, msg):
            return self._group_surface_turn(msg)

    rt = _SlimRuntime(tmp_path)
    msg = _SlimMsg(meta={"service": "join", "new_members": [
        {"id": "7", "username": "ada", "name": "Ada"}]})
    assert rt.turn(msg) is True
    assert rt.sent and "Ada" in rt.sent[0]
