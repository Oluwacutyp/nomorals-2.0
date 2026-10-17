"""Personal-bot (USER-ONLY) BotFather command list + button categories.

The personal bot is owner-only, so its BotFather ``/setcommands`` list is
the public 52 commands PLUS a curated 48 owner-only commands — 100 total,
the BotFather maximum. The gated bot's public list stays exactly as-is
(``PUBLIC_COMMANDS`` in :mod:`nomorals.social.chat.control`); this module
never touches it.

The list is generated from the registry at call time (descriptions from
``COMMAND_DETAILS``), never pasted by hand, so it cannot drift from the
commands the bot actually runs. :func:`render_botfather_text` is the exact
text to paste into BotFather; ``docs/PERSONAL_BOT_COMMANDS.md`` embeds it
and a test asserts the doc and the generator agree.

BotFather format: one ``command - description`` per line, command WITHOUT
the leading slash, description at most 256 chars.
"""

from __future__ import annotations

__all__ = [
    "PERSONAL_OWNER_COMMANDS",
    "personal_botfather_commands",
    "render_botfather_text",
    "validate_botfather_commands",
    "PERSONAL_HELP_CATEGORIES",
]

#: Public 52 in BotFather order: the 19 named public commands + /game +
#: the 32 public game commands (same order as ``public_menu_keyboard``).
_PUBLIC_ORDER: tuple[str, ...] = (
    "help", "search", "news", "book", "music", "play", "video", "podcast",
    "tts", "stt", "look", "image", "lens", "decode", "dns", "whois",
    "cipher", "wisdom", "think", "game",
)

#: Curated owner-only commands on the personal bot (48). Every entry is in
#: ``CONTROL_COMMANDS`` and NOT in ``PUBLIC_COMMANDS``; no aliases
#: (``books``/``wis``/``commands``/``menu`` are covered by their primaries).
#: Ordered by group: control → updates → search → build → memory/money →
#: catalog. 52 public + 48 here = 100, the BotFather maximum.
PERSONAL_OWNER_COMMANDS: tuple[str, ...] = (
    # control — state, model, power, approvals, lifecycle
    "status", "platforms", "mood", "stage", "model", "providers",
    "say", "power", "proposals", "approve", "deny", "start", "stop",
    "quit", "mode", "profile", "mind", "features",
    # platform control — updates & restarts
    "update", "pull", "deploy", "restart",
    # search & research (owner-only depth)
    "searchdeep", "searchleads", "money", "searchhist", "research",
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


def personal_botfather_commands() -> list[tuple[str, str]]:
    """``[(command, description)]`` for the personal bot — 100 entries.

    Public 52 first (BotFather order), then the 48 owner-only commands.
    """
    from .control import PUBLIC_COMMANDS, PUBLIC_GAME_COMMANDS

    ordered: list[str] = []
    for kind in _PUBLIC_ORDER:
        if kind not in ordered:
            ordered.append(kind)
    for kind in PUBLIC_GAME_COMMANDS:
        if kind not in ordered:
            ordered.append(kind)
    for kind in PERSONAL_OWNER_COMMANDS:
        if kind not in ordered:
            ordered.append(kind)
    # Guard: the public set must be exactly covered, nothing dropped.
    assert set(ordered) >= set(PUBLIC_COMMANDS), "personal list drops public commands"
    return [(kind, _description(kind)) for kind in ordered]


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
    missing_public = sorted(set(PUBLIC_COMMANDS) - seen)
    if missing_public:
        problems.append(f"missing public commands: {missing_public}")
    missing_owner = sorted(set(PERSONAL_OWNER_COMMANDS) - seen)
    if missing_owner:
        problems.append(f"missing owner commands: {missing_owner}")
    return problems


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
