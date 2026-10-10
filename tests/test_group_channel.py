"""Group/channel admin: WhatsApp bridge commands + Telegram Bot API admin.

WhatsApp side is tested against a fake bridge (command shape verification —
no network, no Baileys). Telegram side is tested against a stubbed _api
(method-name verification). Tool registration is verified end to end.
"""

import pytest

from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.whatsapp import WhatsAppAdapter
from nomorals.social.chat.telegram import TelegramBotAdapter


class FakeBridge:
    """Records commands the adapter would send over the JSON-lines socket."""

    def __init__(self):
        self.commands = []

    def __call__(self, cmd, timeout=25.0):
        self.commands.append(cmd)
        c = cmd.get("cmd")
        # canned honest replies per command
        if c == "group_create":
            return {"ok": True, "group": {"id": "123@g.us", "subject": cmd["subject"]}}
        if c == "group_members_update":
            return {"ok": True, "action": cmd["action"],
                    "results": [{"jid": p, "status": "200"} for p in cmd["participants"]]}
        if c == "community_create":
            return {"ok": True, "community": {"id": "456@g.us", "subject": cmd["subject"]}}
        if c == "channel_create":
            return {"ok": True, "channel": {"id": "789@newsletter", "name": cmd["name"]}}
        if c == "group_list":
            return {"ok": True, "groups": []}
        return {"ok": True}


@pytest.fixture
def wa():
    a = WhatsAppAdapter.__new__(WhatsAppAdapter)
    bridge = FakeBridge()
    a._send_cmd = bridge
    a.connected = type("E", (), {"is_set": lambda s: True})()
    a._fake_bridge = bridge
    return a


def _chat(jid, kind=ChatKind.GROUP):
    return ChatRef(platform="whatsapp", chat_id=jid, kind=kind, title="")


class TestWhatsAppGroupAdmin:
    def test_group_create_shape(self, wa):
        out = wa.group_create("Lagos foodies", ["111@s.whatsapp.net"])
        assert out["ok"] and out["group"]["id"] == "123@g.us"
        cmd = wa._fake_bridge.commands[-1]
        assert cmd["cmd"] == "group_create"
        assert cmd["subject"] == "Lagos foodies"
        assert cmd["participants"] == ["111@s.whatsapp.net"]

    def test_group_create_empty_subject(self, wa):
        out = wa.group_create("  ")
        assert out["ok"] is False

    def test_members_update_actions(self, wa):
        for action in ("add", "remove", "promote", "demote"):
            out = wa.group_members_update(_chat("1@g.us"), action,
                                          ["111@s.whatsapp.net"])
            assert out["ok"] and out["action"] == action
            assert wa._fake_bridge.commands[-1]["chat"] == "1@g.us"

    def test_members_update_bad_action(self, wa):
        out = wa.group_members_update(_chat("1@g.us"), "explode",
                                      ["111@s.whatsapp.net"])
        assert out["ok"] is False

    def test_group_settings_partial(self, wa):
        out = wa.group_set_settings(_chat("1@g.us"), announce=True)
        assert out["ok"]
        cmd = wa._fake_bridge.commands[-1]
        assert cmd["announce"] is True and "restrict" not in cmd

    def test_group_settings_nothing(self, wa):
        out = wa.group_set_settings(_chat("1@g.us"))
        assert out["ok"] is False

    def test_community_create(self, wa):
        out = wa.community_create("My Community", "desc here")
        assert out["ok"] and out["community"]["id"] == "456@g.us"

    def test_channel_create(self, wa):
        out = wa.channel_create("My Channel")
        assert out["ok"] and out["channel"]["id"] == "789@newsletter"

    def test_admin_cmd_disconnected(self):
        a = WhatsAppAdapter.__new__(WhatsAppAdapter)
        a.connected = type("E", (), {"is_set": lambda s: False})()
        out = a.group_create("x")
        assert out["ok"] is False and "not connected" in out["error"]


class StubBot(TelegramBotAdapter):
    """TelegramBotAdapter with a stubbed _api — records calls."""

    def __init__(self):
        self.calls = []
        self._bot_id = 12345

    def _api(self, method, **params):
        self.calls.append((method, params))
        if method == "getChatMember":
            return {"status": "administrator", "can_delete_messages": True,
                    "can_restrict_members": True, "can_pin_messages": True}
        if method == "getChatAdministrators":
            return [{"status": "creator",
                     "user": {"id": 1, "first_name": "A"}}]
        if method == "getChatMemberCount":
            return 42
        if method == "exportChatInviteLink":
            return "https://t.me/+abc"
        return True


@pytest.fixture
def bot():
    return StubBot()


def _tchat(cid):
    return ChatRef(platform="telegram-bot", chat_id=cid,
                   kind=ChatKind.GROUP, title="")


class TestTelegramBotAdmin:
    def test_admin_rights(self, bot):
        out = bot.admin_rights(_tchat("-1001"))
        assert out["ok"] and out["status"] == "administrator"
        assert out["can_delete_messages"] is True
        method, params = bot.calls[-1]
        assert method == "getChatMember" and params["user_id"] == 12345

    def test_pin(self, bot):
        out = bot.admin_pin(_tchat("-1001"), "77", silent=True)
        assert out["ok"]
        method, params = bot.calls[-1]
        assert method == "pinChatMessage"
        assert params["message_id"] == 77 and params["disable_notification"] is True

    def test_ban_unban(self, bot):
        assert bot.admin_ban(_tchat("-1001"), "999")["ok"]
        assert bot.calls[-1][0] == "banChatMember"
        assert bot.admin_unban(_tchat("-1001"), "999")["ok"]
        assert bot.calls[-1][0] == "unbanChatMember"

    def test_restrict_denies_by_default(self, bot):
        out = bot.admin_restrict(_tchat("-1001"), "999")
        assert out["ok"]
        method, params = bot.calls[-1]
        assert method == "restrictChatMember"
        assert params["permissions"]["can_send_messages"] is False

    def test_promote_demote(self, bot):
        out = bot.admin_promote(_tchat("-1001"), "999", title="Mod")
        assert out["ok"]
        assert bot.calls[-2][0] == "promoteChatMember"
        assert bot.calls[-1][0] == "setChatAdministratorCustomTitle"
        out = bot.admin_demote(_tchat("-1001"), "999")
        assert out["ok"]
        _, params = bot.calls[-1]
        assert params["can_delete_messages"] is False

    def test_admins_list(self, bot):
        out = bot.admin_administrators(_tchat("-1001"))
        assert out["ok"] and out["administrators"][0]["status"] == "creator"

    def test_invite_link(self, bot):
        out = bot.admin_invite_link(_tchat("-1001"))
        assert out["ok"] and out["invite"].startswith("https://t.me/")

    def test_channel_in_supported_kinds(self):
        assert ChatKind.CHANNEL in TelegramBotAdapter.supported_kinds

    def test_api_error_is_honest(self):
        b = StubBot()

        def boom(method, **params):
            raise ValueError("bot was blocked by the user")
        b._api = boom
        out = b.admin_ban(_tchat("-1001"), "999")
        assert out["ok"] is False and "blocked" in out["error"]


class TestToolRegistration:
    def test_all_tools_register(self):
        from nomorals.tools.registry import ToolRegistry
        from nomorals.tools import social as soc_mod
        import nomorals.tools.whatsapp as wa_mod

        reg = ToolRegistry()
        soc_mod.register(reg)
        wa_mod.register(reg)
        names = set(reg._tools.keys())
        for n in ("whatsapp_group_create", "whatsapp_group_members",
                  "whatsapp_group_rename", "whatsapp_group_describe",
                  "whatsapp_group_settings", "whatsapp_group_leave",
                  "whatsapp_group_invite", "whatsapp_community_create",
                  "whatsapp_community_broadcast", "whatsapp_community_link",
                  "whatsapp_channel_create", "whatsapp_channel_post",
                  "tgbot_admin_rights", "tgbot_pin", "tgbot_unpin",
                  "tgbot_ban", "tgbot_unban", "tgbot_restrict",
                  "tgbot_promote", "tgbot_demote", "tgbot_delete",
                  "tgbot_admins", "tgbot_invite_link"):
            assert n in names, f"tool not registered: {n}"
