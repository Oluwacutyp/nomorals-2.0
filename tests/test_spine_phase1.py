"""Phase 1 spine upgrade tests: parsing, ranking, gating."""
import json

from nomorals.agents.partner.tool_loop import (
    _scan_balanced,
    parse_tool_calls,
    _strip_tool_blocks,
)
from nomorals.partner.social_gate import (
    check_tool_call,
    grant_for,
    PUBLIC_GROUP_TOOLS,
)


def test_scan_balanced_nested():
    text = 'call {"tool": "x", "args": {"a": {"b": 1}}} then {"tool": "y"}'
    start = text.index('{"tool": "x"')
    end = _scan_balanced(text, start)
    assert end != -1
    data = json.loads(text[start:end])
    assert data["args"] == {"a": {"b": 1}}


def test_scan_balanced_string_braces():
    text = '{"tool": "x", "args": {"s": "a}b{c"}}'
    end = _scan_balanced(text, 0)
    assert end == len(text)
    assert json.loads(text[:end])["args"] == {"s": "a}b{c"}


def test_parse_bare_nested_args():
    text = 'let me check {"tool": "memory_recall", "args": {"query": "x", "filters": {"a": 1}}}'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "memory_recall"
    assert calls[0].args["filters"] == {"a": 1}


def test_parse_fenced_still_works():
    text = '```tool\n{"tool": "help", "args": {}}\n```'
    calls = parse_tool_calls(text)
    assert len(calls) == 1 and calls[0].name == "help"


def test_strip_nested_bare():
    text = 'answer {"tool": "x", "args": {"n": {"m": 2}}} done'
    stripped = _strip_tool_blocks(text)
    assert stripped == "answer  done"


def test_memory_denied_for_outsider():
    grant = grant_for(is_owner=False, chat_kind="group", group_role="member")
    allowed, reason = check_tool_call("memory_recall", grant=grant)
    assert not allowed, f"memory_recall should be denied: {reason}"


def test_memory_denied_for_group_admin():
    grant = grant_for(is_owner=False, chat_kind="group", group_role="admin")
    allowed, _ = check_tool_call("memory_recall", grant=grant)
    assert not allowed


def test_games_allowed_for_member():
    assert "games" in PUBLIC_GROUP_TOOLS
    grant = grant_for(is_owner=False, chat_kind="group", group_role="member")
    allowed, _ = check_tool_call("games", grant=grant)
    assert allowed


def test_games_tool_has_capability():
    from nomorals.tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.register_builtins()
    spec = reg.get("games")
    assert spec is not None
    assert spec.capability == "social.play", f"got {spec.capability!r}"


def test_deals_has_capability_and_is_sync():
    import inspect
    from nomorals.tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.register_builtins()
    spec = reg.get("deals")
    assert spec is not None
    assert spec.capability == "net.out", f"got {spec.capability!r}"
    assert not inspect.iscoroutinefunction(spec.fn), "deals must be sync"


def test_ranked_listing_relevant():
    from nomorals.tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.register_builtins()
    listing = reg.ranked_listing("play a song", limit=40)
    # music/play tools should rank above unrelated ones
    lines = listing.split("\n")
    assert len(lines) <= 41  # 40 + maybe the "more tools" note
    assert any("play" in line.lower() or "music" in line.lower()
               for line in lines[:10]), f"top 10: {lines[:10]}"


def test_ranked_listing_empty_query_falls_back():
    from nomorals.tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.register_builtins()
    full = reg.prompt_listing()
    ranked = reg.ranked_listing("")
    assert ranked == full
