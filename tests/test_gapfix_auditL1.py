"""L1 — social-chat patch hunks 1–8 (audit V1, V2, V4, V17, V18).

Failing-first: every behaviour test below fails on pristine bab23991
and passes on the fixed tree — verified by the stash run in prove.py.

- V1 (HIGH): a group admin could act through the owner's WhatsApp
  account — the five account-level tools sat in ADMIN_GROUP_TOOLS and
  check_tool_call allowed anything in `allowed`. The fix splits them
  into OWNER_ONLY_ACCOUNT_TOOLS and denies them for non-owners with
  audit recording.
- V2 (HIGH): Baileys warning regexes matched routine per-chat errors
  ("contact blocked us", "515 restart required", ...) and the latch
  then permanently silenced ALL WhatsApp sends. The fix narrows the
  patterns to account-level signals only.
- V4 (HIGH): relative-import depth bugs in nomorals/agents/partner/*
  (`..memory` / `..visuals` / `..social_gate` resolving under
  nomorals/agents/ instead of nomorals/) — swallowed by try/except
  (proactive recall, idle stamp, tool-loop social gate) except the
  /scene surface, which was dead outright.
- V17: same depth bug in group_roster_diff (`..agents.notifier`) —
  fixed in LAYERED form, not by the patch's `...agents.notifier`
  import (that adds a new L4→L5 layering violation): the gateway
  lookup is inlined, since resolve_gateway() is a plain attribute
  read on the context.
- V18: dead `raise ValueError` after a return in
  TelegramAdapter.send_channel_story (chat_id is not even a parameter).
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

from nomorals.agents.partner.runtime_meta import RuntimeMetaMixin
from nomorals.agents.partner.tool_loop import ToolCall, ToolCallingLoop
from nomorals.partner import social_gate as sg
from nomorals.partner.social_gate import (
    ADMIN_GROUP_TOOLS,
    ChatKind,
    DenialLog,
    check_tool_call,
    grant_for,
)
from nomorals.social.chat.baileys_guard import is_warning

REPO_ROOT = Path(__file__).resolve().parents[1]

ACCOUNT_TOOLS = (
    "whatsapp_group_create",
    "whatsapp_group_leave",
    "whatsapp_community_create",
    "whatsapp_channel_create",
    "whatsapp_channel_post",
)


# ── V1: group admin must never hold the owner's account-level tools ──

def _admin_grant():
    return grant_for(is_owner=False, chat_kind=ChatKind.GROUP,
                     group_role="admin")


def test_group_admin_denied_all_five_account_tools():
    # On pristine bab23991 the five tools are in ADMIN_GROUP_TOOLS, so
    # the grant allows them and this test fails.
    grant = _admin_grant()
    log = DenialLog()
    for tool in ACCOUNT_TOOLS:
        allowed, reason = check_tool_call(tool, grant=grant,
                                          chat_kind="group", audit=log)
        assert allowed is False, tool
        assert "owner-only" in reason, reason
    recorded = {r.tool_name for r in log.recent(limit=10)}
    assert set(ACCOUNT_TOOLS) <= recorded


def test_owner_still_allowed_all_five_account_tools():
    grant = grant_for(is_owner=True, chat_kind=ChatKind.GROUP,
                      group_role="admin")
    for tool in ACCOUNT_TOOLS:
        allowed, reason = check_tool_call(tool, grant=grant,
                                          chat_kind="group")
        assert allowed is True, tool
        assert reason == "owner"


def test_group_admin_keeps_in_room_tools():
    grant = _admin_grant()
    for tool in ("whatsapp_group_members", "whatsapp_group_rename",
                 "whatsapp_group_settings", "whatsapp_group_invite",
                 "whatsapp_community_broadcast", "tgbot_pin",
                 "telegram_ban", "games"):
        allowed, _reason = check_tool_call(tool, grant=grant,
                                           chat_kind="group")
        assert allowed is True, tool


def test_owner_only_set_is_disjoint_from_admin_set():
    owner_only = getattr(sg, "OWNER_ONLY_ACCOUNT_TOOLS", None)
    assert owner_only is not None, (
        "OWNER_ONLY_ACCOUNT_TOOLS does not exist (pristine)")
    assert set(ACCOUNT_TOOLS) == set(owner_only)
    assert not (set(owner_only) & set(ADMIN_GROUP_TOOLS))


def test_outsider_group_denied_with_owner_only_reason():
    grant = grant_for(is_owner=False, chat_kind=ChatKind.GROUP,
                      group_role="member")
    allowed, reason = check_tool_call("whatsapp_channel_post",
                                      grant=grant, chat_kind="group")
    assert allowed is False
    assert "owner-only" in reason


# ── V2: warning latch must not fire on routine per-chat errors ──

ROUTINE_ERRORS = (
    # The six strings the audit live-tested at bab23991 — all matched
    # the old patterns and would latch the warning permanently.
    "contact blocked us",
    "group is in restricted mode (admins only)",
    "warning: low battery on bridge host",
    "bulk send complete: 5 messages sent",
    "515 restart required, session closed",
    "401 unauthorized: session expired",
    "user was blocked by contact",
    "group is restricted",
)


def test_routine_per_chat_errors_do_not_warn():
    for text in ROUTINE_ERRORS:
        assert is_warning(text) is False, text


ACCOUNT_SIGNALS = (
    "Your WhatsApp account has been banned",
    "temporarily banned",
    "your account was suspended",
    "403 forbidden: auth failed",
    "warning: your whatsapp account may be restricted",
    "bulk messaging detected on this number",
    "state: banned",
    "spam reports on your account",
    "you have been suspended",
    "phone number has been blocked",
)


def test_account_level_signals_still_warn():
    for text in ACCOUNT_SIGNALS:
        assert is_warning(text) is True, text


# ── V4/V17: the fixed relative imports resolve ──

def _fixed_imports(path: Path, func_name: str, package: str,
                   want: tuple[tuple[int, str], ...]):
    """Read the real file, find the fixed ImportFrom nodes inside the
    named function, assert their depth/module, and execute them in the
    function's package context — proving the depth resolves to a real
    module. On pristine the level/module assertions fail (level 2,
    resolving under nomorals/agents/ or nomorals/social/)."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom) and sub.level >= 2:
                    found.append(ast.get_source_segment(src, sub))
    assert found, f"no relative imports found in {func_name}"
    for level, module in want:
        assert any(
            (("." * level) + module) in stmt for stmt in found
        ), f"missing from {level} import of {module} in {func_name}: {found}"
    ns: dict = {"__package__": package}
    for stmt in found:
        if any((("." * level) + module) in stmt
               for level, module in want):
            exec(stmt, ns)  # noqa: S102 - the point of the test
    return found


def test_brain_proactive_import_resolves():
    _fixed_imports(
        REPO_ROOT / "nomorals/agents/partner/brain.py",
        "_generate_and_persist", "nomorals.agents.partner",
        ((3, "memory.proactive"),))


def test_runtime_idle_import_resolves():
    _fixed_imports(
        REPO_ROOT / "nomorals/agents/partner/runtime.py",
        "on_message", "nomorals.agents.partner",
        ((3, "visuals.idle"),))


def test_tool_loop_gate_imports_resolve():
    _fixed_imports(
        REPO_ROOT / "nomorals/agents/partner/tool_loop.py",
        "_invoke", "nomorals.agents.partner",
        ((3, "partner.social_gate"), (3, "partner.group_roles")))


def test_roster_diff_resolves_gateway_from_context(tmp_path, monkeypatch):
    """H7 functional proof (layered form): the tick resolves the live
    gateway from the context — context.gateway, or
    context.extras["gateway"] — without importing upward into
    nomorals/agents. On pristine the broken `..agents.notifier`
    import raises inside the try, records a group_roster_diff.gateway
    failure, and silently skips the live fetch."""
    import nomorals.social.chat.group_roster_diff as grd

    recorded: list[str] = []
    monkeypatch.setattr(
        grd, "record_failure",
        lambda site, exc: recorded.append(site))

    class _Gateway:
        adapters = {}

    ctx = types.SimpleNamespace(gateway=_Gateway())
    out = grd.run_roster_diff_tick_once(
        context=ctx, workspace_dir=str(tmp_path))
    assert out["ok"] is True
    assert "group_roster_diff.gateway" not in recorded

    # Same path via context.extras["gateway"] (the older runtime shape).
    recorded.clear()
    ctx2 = types.SimpleNamespace(extras={"gateway": _Gateway()})
    out2 = grd.run_roster_diff_tick_once(
        context=ctx2, workspace_dir=str(tmp_path))
    assert out2["ok"] is True
    assert "group_roster_diff.gateway" not in recorded


def test_scene_status_runs():
    """H3 functional proof: /scene status exercises all seven fixed
    visuals imports (six at the top, bridge at the status branch).
    On pristine it raises ModuleNotFoundError before returning."""
    import tempfile

    mixin = RuntimeMetaMixin.__new__(RuntimeMetaMixin)
    with tempfile.TemporaryDirectory() as home:
        mixin.context = types.SimpleNamespace(
            settings=types.SimpleNamespace(home=home))
        out = mixin._control_scene("status")
    assert out.startswith("scene board:"), out
    assert "idle trigger:" in out


def test_tool_loop_social_gate_runs():
    """H4 functional proof: a non-owner social tool call reaches the
    gate instead of dying on the broken import. On pristine the loop
    returns '[denied] social gate unavailable'."""
    class _Registry:
        context = None

        def get(self, name):
            return object()

        def names(self):
            return ["whatsapp_group_create"]

    loop = ToolCallingLoop(None, _Registry())
    out = loop._invoke(
        ToolCall(name="whatsapp_group_create", args={}),
        actor="outsider", loop_ctx={"chat_kind": "group"})
    assert out.startswith("[denied]"), out
    assert "social gate unavailable" not in out
    # The denial is the V1 owner-only denial, not a generic miss.
    assert "owner-only" in out


# ── V18: dead raise after return is gone ──

def test_send_channel_story_has_no_dead_raise():
    from nomorals.social.chat.telegram import TelegramAdapter

    assert hasattr(TelegramAdapter, "send_channel_story")
    src = (REPO_ROOT / "nomorals/social/chat/telegram.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) \
                and node.name == "send_channel_story":
            segment = ast.get_source_segment(src, node)
            assert segment is not None
            assert "raise ValueError" not in segment
            return
    pytest.fail("send_channel_story not found")
