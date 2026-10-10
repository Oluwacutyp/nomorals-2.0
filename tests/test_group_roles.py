"""Role-based group permissions: group admins get admin tools in their groups."""

from unittest.mock import MagicMock

from nomorals.partner.group_roles import (
    ROLE_ADMIN, ROLE_MEMBER, ROLE_UNKNOWN,
    resolve_group_role, clear_role_cache,
)
from nomorals.partner.social_gate import (
    grant_for, check_tool_call,
    ACTOR_OWNER, ACTOR_GROUP_ADMIN, ACTOR_OUTSIDER,
    ADMIN_GROUP_TOOLS, PUBLIC_GROUP_TOOLS,
)


class TestRoleResolution:
    def setup_method(self):
        clear_role_cache()

    def test_telegram_bot_admin(self):
        adapter = MagicMock()
        adapter.admin_member.return_value = {
            "ok": True, "status": "administrator", "user": "123"}
        role = resolve_group_role("telegram", adapter, "chat1", "123")
        assert role == ROLE_ADMIN

    def test_telegram_bot_creator(self):
        adapter = MagicMock()
        adapter.admin_member.return_value = {
            "ok": True, "status": "creator", "user": "123"}
        role = resolve_group_role("telegram", adapter, "chat1", "123")
        assert role == ROLE_ADMIN

    def test_telegram_bot_member(self):
        adapter = MagicMock()
        adapter.admin_member.return_value = {
            "ok": True, "status": "member", "user": "456"}
        role = resolve_group_role("telegram", adapter, "chat1", "456")
        assert role == ROLE_MEMBER

    def test_whatsapp_admin(self):
        adapter = MagicMock()
        adapter.group_info.return_value = {
            "admins": ["123@s.whatsapp.net", "789@s.whatsapp.net"],
            "owner": "123@s.whatsapp.net",
        }
        role = resolve_group_role("whatsapp", adapter, "group1@g.us", "123")
        assert role == ROLE_ADMIN

    def test_whatsapp_member(self):
        adapter = MagicMock()
        adapter.group_info.return_value = {
            "admins": ["123@s.whatsapp.net"],
            "owner": "123@s.whatsapp.net",
        }
        role = resolve_group_role("whatsapp", adapter, "group1@g.us", "999")
        assert role == ROLE_MEMBER

    def test_lookup_failure_fails_closed(self):
        adapter = MagicMock()
        adapter.admin_member.side_effect = RuntimeError("nope")
        role = resolve_group_role("telegram", adapter, "chat1", "123")
        assert role == ROLE_UNKNOWN

    def test_cache_hit(self):
        adapter = MagicMock()
        adapter.admin_member.return_value = {
            "ok": True, "status": "administrator", "user": "123"}
        resolve_group_role("telegram", adapter, "chat1", "123")
        resolve_group_role("telegram", adapter, "chat1", "123")
        # Second call served from cache — adapter called once
        assert adapter.admin_member.call_count == 1


class TestGroupAdminGrant:
    def test_owner_unchanged(self):
        grant = grant_for(is_owner=True, chat_kind="group")
        assert grant.actor == ACTOR_OWNER
        assert grant.may_admin is True
        allowed, _ = check_tool_call("tgbot_ban", grant=grant)
        assert allowed is True

    def test_group_admin_gets_admin_tools(self):
        grant = grant_for(is_owner=False, chat_kind="group",
                          group_role="admin")
        assert grant.actor == ACTOR_GROUP_ADMIN
        assert grant.may_admin is True
        allowed, _ = check_tool_call("tgbot_ban", grant=grant)
        assert allowed is True
        allowed, _ = check_tool_call("whatsapp_group_settings", grant=grant)
        assert allowed is True

    def test_group_admin_gets_public_tools(self):
        grant = grant_for(is_owner=False, chat_kind="group",
                          group_role="admin")
        allowed, _ = check_tool_call("games", grant=grant)
        assert allowed is True

    def test_group_admin_denied_private(self):
        grant = grant_for(is_owner=False, chat_kind="group",
                          group_role="admin")
        # Memory access stays owner-only even for group admins
        assert grant.may_access_memory is False
        assert grant.may_dm_others is False

    def test_regular_member_denied_admin_tools(self):
        grant = grant_for(is_owner=False, chat_kind="group",
                          group_role="member")
        assert grant.actor == ACTOR_OUTSIDER
        assert grant.may_admin is False
        allowed, reason = check_tool_call("tgbot_ban", grant=grant)
        assert allowed is False
        assert "denied" in reason

    def test_regular_member_gets_public_tools(self):
        grant = grant_for(is_owner=False, chat_kind="group",
                          group_role="member")
        allowed, _ = check_tool_call("games", grant=grant)
        assert allowed is True

    def test_unknown_role_fails_closed(self):
        grant = grant_for(is_owner=False, chat_kind="group",
                          group_role="unknown")
        allowed, _ = check_tool_call("tgbot_ban", grant=grant)
        assert allowed is False
