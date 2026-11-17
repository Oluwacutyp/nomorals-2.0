"""Personal-bot (USER-ONLY) BotFather command list + button categories.

The personal bot is owner-only. Its BotFather ``/setcommands`` list is
capped by BotFather at 100 commands, while the public catalog
(``PUBLIC_COMMANDS`` in :mod:`nomorals.social.chat.control`) has long
outgrown that cap — so the personal list is a curated subset by design:
all 48 owner-only commands (``PERSONAL_OWNER_COMMANDS``) plus the
highest-priority public commands filling the remaining slots (see
:func:`personal_botfather_commands` for the priority order). Public
commands that do not fit are not gone: they stay in the public catalog,
the menus and /list — the BotFather list is only the client's command
picker. The gated bot's own lists are untouched by this module.

The list is generated from the registry at call time (descriptions from
``COMMAND_DETAILS``), never pasted by hand, so it cannot drift from the
commands the bot actually runs. :func:`render_botfather_text` is the exact
text to paste into BotFather; ``docs/PERSONAL_BOT_COMMANDS.md`` embeds it
and a test asserts the doc and the generator agree.

BotFather format: one ``command - description`` per line, command WITHOUT
the leading slash, description at most 256 chars.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "PERSONAL_OWNER_COMMANDS",
    "personal_botfather_commands",
    "render_botfather_text",
    "validate_botfather_commands",
    "PERSONAL_HELP_CATEGORIES",
    "gated_botfather_commands",
    "render_gated_botfather_text",
    "validate_gated_botfather_commands",
    "sync_bot_commands",
    "push_with_too_much_retry",
    "menu_scope_state",
]

#: Public commands in BotFather priority order: the 19 named public
#: commands + /game (same order as ``public_menu_keyboard``). This is
#: the top of the fill priority for the personal list — see
#: :func:`personal_botfather_commands`.
_PUBLIC_ORDER: tuple[str, ...] = (
    "help", "search", "news", "book", "music", "play", "video", "podcast",
    "tts", "stt", "look", "image", "lens", "decode", "dns", "whois",
    "cipher", "wisdom", "think", "game",
)

#: Curated owner-only commands on the personal bot (48). Every entry is in
#: ``CONTROL_COMMANDS`` and NOT in ``PUBLIC_COMMANDS``; no aliases
#: (``books``/``wis``/``commands``/``menu`` are covered by their primaries;
#: ``gmail``/``mail``/``inbox`` are covered by ``email``).
#: Ordered by group: control → updates → search → build → memory/money →
#: catalog. All 48 always fit: the public side fills only the slots this
#: list leaves under the BotFather maximum of 100.
#: ``searchleads`` is NOT on the menu (email took its slot — the mailbox
#: surface matters more in the menu); it stays registered and runs when
#: typed, like every other off-menu owner command.
PERSONAL_OWNER_COMMANDS: tuple[str, ...] = (
    # control — state, model, power, approvals, lifecycle
    "status", "platforms", "mood", "stage", "model", "providers",
    "say", "power", "proposals", "approve", "deny", "start", "stop",
    "quit", "mode", "profile", "mind", "features", "email",
    # platform control — updates & restarts
    "update", "pull", "deploy", "restart",
    # search & research (owner-only depth)
    "searchdeep", "money", "searchhist", "research",
    # building for real — code, missions, solving
    "code", "py", "devon", "task", "mission", "schedule", "monitor",
    "immune", "impossible", "arena", "trial", "exec", "db",
    # memory & money
    "remember", "recall", "forget", "memories",
    "spend", "budget", "spending",
    # catalog
    "list",
)

#: Max BotFather commands / description chars (Bot API limits).
BOTFATHER_MAX_COMMANDS = 100
BOTFATHER_MAX_DESCRIPTION = 256

#: Telegram's command-name rule (Bot API): 1-32 chars, lowercase
#: letters, digits, underscores. The validators enforce it so a bad
#: name is a refused push naming the name, never an API round-trip.
_COMMAND_NAME_RE = re.compile(r"^[a-z0-9_]{1,32}$")

#: Retry floor for a BOT_COMMANDS_TOO_MUCH refusal: halve the list
#: until Telegram accepts or the list reaches this size. Below it a
#: refusal means something other than the count is wrong.
_MENU_SYNC_RETRY_FLOOR = 25


def _is_too_much(error: str) -> bool:
    """True when a push failure is Telegram's TOO_MUCH refusal."""
    return "BOT_COMMANDS_TOO_MUCH" in (error or "")


def _retry_order(entries: list[tuple[str, str]],
                 *, owner_only: bool) -> list[tuple[str, str]]:
    """The order a truncated retry pushes in.

    The gated list is already priority order. The personal list is
    public-fill first, owner after — but the 48 owner commands are
    non-negotiable ("all 48 always fit"), so a retry keeps every
    owner command first, then the public fill in priority order.
    A retry therefore never drops the owner's own surface to make
    room for public commands.
    """
    if not owner_only:
        return list(entries)
    by_kind = dict(entries)
    owner = [(kind, by_kind[kind]) for kind in PERSONAL_OWNER_COMMANDS
             if kind in by_kind]
    public = [(kind, desc) for kind, desc in entries
              if kind not in PERSONAL_OWNER_COMMANDS]
    return owner + public


def push_with_too_much_retry(
        push: Any, entries: list[tuple[str, str]],
        *, first: list[tuple[str, str]] | None = None,
        floor: int = _MENU_SYNC_RETRY_FLOOR) -> dict[str, Any]:
    """Push a command list, halving on BOT_COMMANDS_TOO_MUCH.

    ``push(entries_slice)`` is the adapter call — it returns an
    ``{"ok": bool, "error": str}`` dict or raises; either shape is
    read the same way. The first attempt pushes ``first`` (the full
    list in display order); each retry pushes the priority head of
    ``entries`` halved toward ``floor`` (100 -> 50 -> 25). Any other
    failure refuses at once — a retry can never fix it. Returns
    ``{"ok", "pushed", "attempts", "error"}`` where ``attempts`` is
    the list of sizes tried, so the caller can report exactly what
    landed and what did not.
    """
    first_list = list(first) if first is not None else list(entries)
    retry_entries = list(entries)
    attempts: list[int] = []
    attempt_list = first_list
    n = len(attempt_list)
    while True:
        attempts.append(n)
        try:
            out = push(attempt_list)
        except Exception as exc:  # noqa: BLE001 - an API failure is a refusal
            out = {"ok": False, "error": str(exc)}
        if isinstance(out, dict) and out.get("ok"):
            return {"ok": True, "pushed": n, "attempts": attempts,
                    "error": ""}
        error = ""
        if isinstance(out, dict):
            error = str(out.get("error") or "")
        if not _is_too_much(error) or n <= floor:
            return {"ok": False, "pushed": 0, "attempts": attempts,
                    "error": error}
        n = max(floor, n // 2)
        attempt_list = retry_entries[:n]


def _description(kind: str) -> str:
    """First sentence of the command's ``what``, capped at 256 chars."""
    from .control import COMMAND_DETAILS

    what = (COMMAND_DETAILS.get(kind, {}) or {}).get("what", "") or kind
    # First sentence: up to the first ". " or a trailing ".".
    text = what.strip().replace("\n", " ")
    for sep in (". ", ".\n"):
        if sep in text:
            text = text.split(sep, 1)[0] + "."
            break
    else:
        text = text.rstrip(".") + "." if text else kind + "."
    if len(text) > BOTFATHER_MAX_DESCRIPTION:
        text = text[: BOTFATHER_MAX_DESCRIPTION - 3].rstrip() + "..."
    return text


def _public_priority() -> list[str]:
    """Public commands in BotFather fill-priority order.

    The curated surfaces first — ``_PUBLIC_ORDER`` (the named public
    commands in menu order), then ``PUBLIC_GAME_COMMANDS`` (the public
    games in their curated order) — then every remaining public command
    in registry order (``CONTROL_COMMANDS``), with any public kind the
    registry does not carry appended alphabetically. Deterministic:
    the same catalog always yields the same order.
    """
    from .control import CONTROL_COMMANDS, PUBLIC_COMMANDS, PUBLIC_GAME_COMMANDS

    priority: list[str] = []
    for kind in _PUBLIC_ORDER:
        if kind in PUBLIC_COMMANDS and kind not in priority:
            priority.append(kind)
    for kind in PUBLIC_GAME_COMMANDS:
        if kind in PUBLIC_COMMANDS and kind not in priority:
            priority.append(kind)
    for kind in CONTROL_COMMANDS:
        if kind in PUBLIC_COMMANDS and kind not in priority:
            priority.append(kind)
    for kind in sorted(PUBLIC_COMMANDS):
        if kind not in priority:
            priority.append(kind)
    return priority


def personal_botfather_commands() -> list[tuple[str, str]]:
    """``[(command, description)]`` for the personal bot — ≤100 entries.

    BotFather caps a bot's list at 100 commands and the public catalog
    is far larger than that, so the personal list is a curated subset:
    public commands in :func:`_public_priority` order fill the slots
    the owner list leaves, then every owner-only command follows.
    Every entry is an owner command or a public command; the result
    is deterministic and can never exceed the cap, however large the
    public catalog grows.
    """
    slots = BOTFATHER_MAX_COMMANDS - len(PERSONAL_OWNER_COMMANDS)
    ordered: list[str] = []
    for kind in _public_priority():
        if len(ordered) >= slots:
            break
        if kind not in ordered:
            ordered.append(kind)
    for kind in PERSONAL_OWNER_COMMANDS:
        if kind not in ordered:
            ordered.append(kind)
    return [(kind, _description(kind)) for kind in ordered]


def gated_botfather_commands() -> list[tuple[str, str]]:
    """``[(command, description)]`` for the gated (public) bot — ≤100 entries.

    The gated bot has no owner-only commands to carry, so its "/" picker is
    the first 100 public commands in :func:`_public_priority` order — the
    same priority order the personal list fills from, so the two bots agree
    on which public commands rank highest. The 163 public commands past
    the cap stay fully available (typed, in the in-chat ``/menu``, in
    ``/list``); the BotFather list is only the client's command picker.
    Deterministic and can never exceed the cap, however large the public
    catalog grows.
    """
    return [(kind, _description(kind))
            for kind in _public_priority()[:BOTFATHER_MAX_COMMANDS]]


def render_gated_botfather_text() -> str:
    """Exact text to paste into BotFather ``/setcommands`` (gated bot)."""
    return "\n".join(
        f"{kind} - {desc}" for kind, desc in gated_botfather_commands()
    )


def render_botfather_text() -> str:
    """Exact text to paste into BotFather ``/setcommands``."""
    return "\n".join(
        f"{kind} - {desc}" for kind, desc in personal_botfather_commands()
    )


def validate_botfather_commands() -> list[str]:
    """Return a list of problems (empty = valid BotFather list)."""
    from .control import CONTROL_COMMANDS, PUBLIC_COMMANDS

    problems: list[str] = []
    entries = personal_botfather_commands()
    if len(entries) > BOTFATHER_MAX_COMMANDS:
        problems.append(
            f"{len(entries)} commands, over BotFather max {BOTFATHER_MAX_COMMANDS}"
        )
    seen: set[str] = set()
    for kind, desc in entries:
        if kind in seen:
            problems.append(f"duplicate command: {kind}")
        seen.add(kind)
        if kind not in CONTROL_COMMANDS and kind not in PUBLIC_COMMANDS:
            problems.append(f"unknown command (not in registry): {kind}")
        if not desc or len(desc) > BOTFATHER_MAX_DESCRIPTION:
            problems.append(
                f"bad description for {kind}: {len(desc)} chars (max {BOTFATHER_MAX_DESCRIPTION})"
            )
        if " - " in kind or " " in kind:
            problems.append(f"command contains spaces: {kind!r}")
        if not _COMMAND_NAME_RE.fullmatch(kind):
            problems.append(
                f"bad command name: {kind!r} (must match "
                "^[a-z0-9_]{1,32}$)")
    # Owner commands are all small enough to fit by design and every
    # one must be listed; public commands beyond the cap are excluded
    # by design (they stay in the public catalog, menus and /list), so
    # only the owner side is checked for completeness.
    missing_owner = sorted(set(PERSONAL_OWNER_COMMANDS) - seen)
    if missing_owner:
        problems.append(f"missing owner commands: {missing_owner}")
    return problems


def validate_gated_botfather_commands() -> list[str]:
    """Return a list of problems with the gated list (empty = valid).

    Same structural checks as :func:`validate_botfather_commands`, plus the
    coverage rule for the gated side: the list must be exactly the first
    ``len(entries)`` public commands in priority order — a higher-priority
    public command must never be missing while a lower-priority one is
    listed. Public commands past the cap are excluded by design (they stay
    in the public catalog, menus and /list), so only the priority head is
    checked for completeness.
    """
    from .control import CONTROL_COMMANDS, PUBLIC_COMMANDS

    problems: list[str] = []
    entries = gated_botfather_commands()
    if len(entries) > BOTFATHER_MAX_COMMANDS:
        problems.append(
            f"{len(entries)} commands, over BotFather max {BOTFATHER_MAX_COMMANDS}"
        )
    seen: set[str] = set()
    for kind, desc in entries:
        if kind in seen:
            problems.append(f"duplicate command: {kind}")
        seen.add(kind)
        if kind not in PUBLIC_COMMANDS:
            problems.append(f"non-public command on the gated list: {kind}")
        if kind not in CONTROL_COMMANDS:
            problems.append(f"unknown command (not in registry): {kind}")
        if not desc or len(desc) > BOTFATHER_MAX_DESCRIPTION:
            problems.append(
                f"bad description for {kind}: {len(desc)} chars (max {BOTFATHER_MAX_DESCRIPTION})"
            )
        if " - " in kind or " " in kind:
            problems.append(f"command contains spaces: {kind!r}")
        if not _COMMAND_NAME_RE.fullmatch(kind):
            problems.append(
                f"bad command name: {kind!r} (must match "
                "^[a-z0-9_]{1,32}$)")
    head = _public_priority()[: len(entries)]
    missing_head = [kind for kind in head if kind not in seen]
    if missing_head:
        problems.append(
            f"missing higher-priority public commands: {missing_head}")
    return problems


def _skipped_for(pushed_kinds: list[str], *, owner_only: bool) -> int:
    """Commands past the cap that stay available via /list: every
    catalog command not on the picker. For the full 100-command push
    this is the old count (gated 163, personal 211); after a TOO_MUCH
    retry it grows by exactly the commands the smaller list dropped.
    """
    from .control import PUBLIC_COMMANDS

    public_pushed = sum(1 for kind in pushed_kinds
                        if kind in PUBLIC_COMMANDS)
    owner_pushed = len(pushed_kinds) - public_pushed
    skipped = len(_public_priority()) - public_pushed
    if owner_only:
        skipped += len(PERSONAL_OWNER_COMMANDS) - owner_pushed
    return max(0, skipped)


def menu_scope_state(adapter: Any, *, dry_run: bool = False,
                     scope_size: int | None = None) -> dict[str, Any]:
    """Scope-state diagnostic for ``nm menu sync`` (dry-run-safe).

    Reports what the builders produce (default-scope sizes for both
    bots, §16 scope size when the caller passes it — this module is
    L4 and must not import the L5 runtime that owns the §16 kinds)
    and, when the API is reachable, the live default-scope count for
    this bot — so the next EC2 run can show whether a scoped list is
    eating the budget. Never raises: offline / unreachable / a
    malformed reply is reported, not thrown. In dry-run mode the live
    query is skipped entirely — a dry run never calls the API.
    """
    state: dict[str, Any] = {
        "bot": str(getattr(adapter, "name", "") or "telegram-bot"),
        "builders": {
            "gated_default": len(gated_botfather_commands()),
            "personal_default": len(personal_botfather_commands()),
            "scope": scope_size,
        },
        "live": {"default": None, "note": ""},
    }
    if dry_run:
        state["live"]["note"] = "not queried (dry run)"
        return state
    fetch = getattr(adapter, "get_default_commands", None)
    if not callable(fetch):
        state["live"]["note"] = "adapter cannot read live commands"
        return state
    try:
        live = fetch()
    except Exception as exc:  # noqa: BLE001 - offline is a report
        state["live"]["note"] = f"unreachable: {str(exc)[:120]}"
        return state
    try:
        state["live"]["default"] = len(list(live or []))
        state["live"]["note"] = "live default scope"
    except Exception:  # noqa: BLE001 - a malformed reply is a report
        state["live"]["note"] = "unreadable live reply"
    return state


def sync_bot_commands(adapter: Any, *, dry_run: bool = False) -> dict[str, Any]:
    """Push this bot's registry-derived "/" list via setMyCommands.

    The list is picked by ``adapter.owner_only`` — the personal bot gets
    :func:`personal_botfather_commands`, the gated bot gets
    :func:`gated_botfather_commands`. Before any push, the drift gate runs:
    :func:`telegram_menu.validate_menu` must pass (owner menu covers every
    registered command, public menu every public command), and the list's
    own validator must pass. A failing gate refuses the push and names the
    problems — a half-synced or drifted menu never reaches Telegram.

    ``dry_run`` builds and validates the list without calling the API.
    Never raises for a bad adapter shape — a missing
    ``set_default_commands`` is a refusal, not a crash. Returns
    ``{"ok", "bot", "pushed", "skipped", "problems", "retried",
    "attempts", ...}`` where ``skipped`` counts the public commands
    past the BotFather cap that stay available everywhere except the
    "/" picker, and ``retried``/``attempts`` record a TOO_MUCH retry
    (the sizes tried, so the caller can report exactly what landed).
    """
    from .telegram_menu import validate_menu

    bot = str(getattr(adapter, "name", "") or "telegram-bot")
    owner_only = bool(getattr(adapter, "owner_only", False))
    result: dict[str, Any] = {
        "ok": False,
        "bot": bot,
        "pushed": 0,
        "skipped": 0,
        "problems": [],
        "retried": False,
        "attempts": [],
    }
    ok, problems = validate_menu()
    if not ok:
        result["problems"] = list(problems)
        return result
    if owner_only:
        entries = personal_botfather_commands()
        list_problems = validate_botfather_commands()
    else:
        entries = gated_botfather_commands()
        list_problems = validate_gated_botfather_commands()
    if list_problems:
        result["problems"] = list(list_problems)
        return result
    if dry_run:
        result["ok"] = True
        result["commands"] = [kind for kind, _ in entries]
        result["skipped"] = _skipped_for(result["commands"],
                                         owner_only=owner_only)
        return result
    push = getattr(adapter, "set_default_commands", None)
    if not callable(push):
        result["problems"] = [
            f"{bot} cannot push commands (no set_default_commands)"]
        return result
    # A 100-command push can be refused server-side as
    # BOT_COMMANDS_TOO_MUCH (seen live at the 2026-11-17 boot on both
    # bots) even though the payload is exactly 100 valid commands.
    # Retry with the priority head halved until Telegram accepts, so
    # the "/" picker lands instead of failing outright.
    retry_entries = _retry_order(entries, owner_only=owner_only)
    pushed = push_with_too_much_retry(push, retry_entries, first=entries)
    result["attempts"] = pushed["attempts"]
    result["retried"] = len(pushed["attempts"]) > 1
    if not pushed["ok"]:
        result["problems"] = [
            f"setMyCommands failed: "
            f"{pushed['error'] or 'unknown error'} "
            f"(attempted sizes {pushed['attempts']})"]
        return result
    n = pushed["pushed"]
    landed = (retry_entries if result["retried"] else entries)[:n]
    result["ok"] = True
    result["pushed"] = n
    result["commands"] = [kind for kind, _ in landed]
    result["skipped"] = _skipped_for(result["commands"],
                                     owner_only=owner_only)
    return result


#: /help category buttons on the personal bot: (label, callback).
#: Callbacks are real routable commands (``/list <group>`` shows that
#: category; ``power status`` shows power state). No dead buttons.
PERSONAL_HELP_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("🔍 Search", "list search"),
    ("🎬 Media", "list media"),
    ("🎮 Games", "list games"),
    ("⚡ Power", "power status"),
    ("🏗️ Build", "list build"),
    ("🧠 Memory", "list memory"),
    ("🎙️ Voice", "list voice"),
    ("📿 Wisdom", "list wisdom"),
    ("⚙️ Tools", "list tools"),
    ("🎛️ Control", "list control"),
    ("📋 All commands", "list"),
)
