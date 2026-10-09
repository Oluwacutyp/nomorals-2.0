"""Tests for the public/private capability matrix (social_gate)."""

from nomorals.partner.social_gate import (
    ACTOR_OWNER,
    ACTOR_OUTSIDER,
    CAPABILITY_MATRIX,
    PUBLIC_CATEGORIES,
    PRIVATE_CATEGORIES,
    PUBLIC_GROUP_TOOLS,
    audit_matrix,
    check_tool_call,
    grant_for,
)


def test_matrix_covers_all_categories():
    for cat in PUBLIC_CATEGORIES | PRIVATE_CATEGORIES:
        assert cat in CAPABILITY_MATRIX, f"{cat} missing from matrix"


def test_public_categories_open_in_groups():
    for cat in PUBLIC_CATEGORIES:
        vis = CAPABILITY_MATRIX[cat]
        assert vis["public_in_group"] is True
        assert vis["public_in_dm"] is False


def test_private_categories_closed_everywhere():
    for cat in PRIVATE_CATEGORIES:
        vis = CAPABILITY_MATRIX[cat]
        assert vis["public_in_group"] is False
        assert vis["public_in_dm"] is False


def test_owner_gets_everything():
    g = grant_for(is_owner=True, chat_kind="group")
    assert g.actor == ACTOR_OWNER
    ok, _ = check_tool_call("telegram_ban", grant=g)
    assert ok
    ok, _ = check_tool_call("memory_recall", grant=g)
    assert ok


def test_outsider_group_public_only():
    g = grant_for(is_owner=False, chat_kind="group")
    assert g.actor == ACTOR_OUTSIDER
    assert not g.may_admin
    assert not g.may_access_memory
    assert not g.may_dm_others
    # Public tools allowed
    for tool in PUBLIC_GROUP_TOOLS:
        ok, _ = check_tool_call(tool, grant=g)
        assert ok, f"{tool} should be public in groups"
    # Private tools denied
    for tool in ("telegram_ban", "memory_recall", "research_run"):
        ok, reason = check_tool_call(tool, grant=g)
        assert not ok, f"{tool} should be denied: {reason}"


def test_outsider_dm_no_tools():
    g = grant_for(is_owner=False, chat_kind="dm")
    assert g.actor == ACTOR_OUTSIDER
    ok, _ = check_tool_call("game_move", grant=g)
    assert not ok  # even games are gated in DMs


def test_audit_matrix_complete():
    a = audit_matrix()
    assert set(a["public_categories"]) == set(PUBLIC_CATEGORIES)
    assert set(a["private_categories"]) == set(PRIVATE_CATEGORIES)
    assert len(a["matrix"]) == len(PUBLIC_CATEGORIES) + len(PRIVATE_CATEGORIES)
