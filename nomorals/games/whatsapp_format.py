"""WhatsApp-channel-friendly game rendering.

Game output is authored for Telegram (HTML sections, inline buttons) but
WhatsApp channels render plain text — no HTML, no inline buttons, and
long posts get visually truncated. This module renders the same game
state (stats, battles, achievements, leaderboards, menus) in
WhatsApp-native form:

* ``*bold*`` / ``_italic_`` / `` ```mono``` `` — WhatsApp markdown
* emoji headers and medal rows instead of inline buttons
* progress bars from block characters (render everywhere)
* chunked to channel-safe sizes (see :func:`wa_chunks`)

Platform detection: :func:`format_game_text` takes a ``chat_key``
(``platform:chat_id``) or a platform name and converts any text —
Telegram HTML included — to the right rendering. Telegram output is
returned untouched.

Every function never raises.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..social.chat.platforms import (
    WHATSAPP, chunk_text, detect_platform, format_for_platform,
)

__all__ = [
    "WA_CHUNK_LIMIT",
    "is_whatsapp",
    "platform_of",
    "format_game_text",
    "wa_chunks",
    "wa_bar",
    "render_stats_wa",
    "render_battle_wa",
    "render_leaderboard_wa",
    "render_achievements_wa",
    "render_game_status_wa",
    "render_game_menu_wa",
    "render_coins_wa",
]

#: Channel-safe chunk budget (WhatsApp renders very long posts poorly).
WA_CHUNK_LIMIT = 3000

_MEDALS = ("🥇", "🥈", "🥉")
_STAT_EMOJI = {
    "strength": "⚔️", "stamina": "❤️", "mana": "🔮", "intelligence": "🧠",
    "level": "⭐", "xp": "✨", "coins": "🪙",
}
_STAT_DESC = {
    "strength": "attack", "stamina": "HP + defense",
    "mana": "skill fuel", "intelligence": "skill power + combo luck",
}


def is_whatsapp(chat_key_or_platform: Any) -> bool:
    """True when the target is WhatsApp. Never raises."""
    try:
        s = str(chat_key_or_platform or "")
        if ":" in s:
            return detect_platform(s) == WHATSAPP
        return s.strip().lower() in ("whatsapp", "wa")
    except Exception:  # noqa: BLE001
        return False


def platform_of(chat_key_or_platform: Any) -> str:
    """Canonical platform name for a chat key or platform string. Never raises."""
    try:
        s = str(chat_key_or_platform or "")
        if ":" in s:
            return detect_platform(s)
        from ..social.chat.platforms import _PREFIX_MAP, TELEGRAM
        return _PREFIX_MAP.get(s.strip().lower(), TELEGRAM)
    except Exception:  # noqa: BLE001
        from ..social.chat.platforms import TELEGRAM
        return TELEGRAM


def format_game_text(text: str, chat_key_or_platform: Any) -> str:
    """Convert game text to the platform's rendering.

    Telegram → untouched. WhatsApp → WhatsApp markdown. Anything else →
    the platform adapter. Never raises.
    """
    try:
        if is_whatsapp(chat_key_or_platform):
            return format_for_platform(text, WHATSAPP)
        return str(text or "")
    except Exception:  # noqa: BLE001
        return str(text or "")


def wa_chunks(text: str, limit: int = WA_CHUNK_LIMIT) -> list[str]:
    """Split WhatsApp game output into channel-safe chunks. Never raises."""
    try:
        return chunk_text(format_for_platform(text, WHATSAPP), limit)
    except Exception:  # noqa: BLE001
        return [str(text or "")]


def wa_bar(value: int, maximum: int, width: int = 10) -> str:
    """Block-character progress bar — renders on every client. Never raises."""
    try:
        maximum = max(1, int(maximum))
        value = max(0, min(int(value), maximum))
        width = max(4, min(int(width), 20))
        filled = round(value / maximum * width)
        return "█" * filled + "░" * (width - filled)
    except Exception:  # noqa: BLE001
        return ""


def render_stats_wa(name: str, stats: dict[str, Any] | Any,
                    gear_bonus: dict[str, int] | None = None) -> str:
    """Player stat sheet for WhatsApp. Never raises."""
    try:
        gear_bonus = gear_bonus or {}
        lines = [f"*📊 {name}'s attributes*"]
        def _get(s: Any, k: str) -> int:
            try:
                return int(s.get(k, 0) if isinstance(s, dict) else getattr(s, k, 0))
            except Exception:  # noqa: BLE001
                return 0
        for key in ("strength", "stamina", "mana", "intelligence"):
            base = _get(stats, key)
            bonus = int(gear_bonus.get(key, 0) or 0)
            total = base + bonus
            extra = f" (+{bonus} gear)" if bonus else ""
            emoji = _STAT_EMOJI.get(key, "•")
            lines.append(f"{emoji} *{key.capitalize()}* {total} — {_STAT_DESC.get(key, '')}{extra}")
        unspent = _get(stats, "unspent")
        lines.append(f"✨ Unspent points: *{unspent}*")
        lines.append("")
        lines.append("_spend: /stats <strength|stamina|mana|intelligence> [points]_")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return f"*{name}* — stats unavailable"


def render_battle_wa(title: str, rounds: Sequence[str],
                     winner: str = "", loser: str = "") -> str:
    """Battle narration for WhatsApp — scenes, then the result. Never raises."""
    try:
        lines = [f"*⚔️ {title}*", ""]
        for i, r in enumerate(rounds or [], 1):
            lines.append(f"_{i}._ {r}")
        if winner:
            lines.append("")
            lines.append(f"🏆 *{winner} wins!*")
            if loser:
                lines.append(f"_{loser} fought well — rematch? /duel_")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return f"*⚔️ {title}*"


def render_leaderboard_wa(title: str,
                          entries: Sequence[tuple[str, Any]]) -> str:
    """Ranked leaderboard with medals. ``entries`` = (name, score). Never raises."""
    try:
        lines = [f"*🏆 {title}*", ""]
        if not entries:
            lines.append("_no rankings yet — play a game to get on the board_")
            return "\n".join(lines)
        for i, (name, score) in enumerate(entries, 1):
            medal = _MEDALS[i - 1] if i <= 3 else f"{i}."
            lines.append(f"{medal} *{name}* — {score}")
        lines.append("")
        lines.append("_/game leaderboard — full table_")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return f"*🏆 {title}*"


def render_achievements_wa(name: str,
                           unlocks: Sequence[dict[str, Any] | str]) -> str:
    """Achievement unlock announcement. Never raises."""
    try:
        lines = [f"*🏅 {name} unlocked!*", ""]
        for u in unlocks or []:
            if isinstance(u, dict):
                title = u.get("title", u.get("name", "achievement"))
                desc = u.get("description", u.get("desc", ""))
                line = f"🎖️ *{title}*"
                if desc:
                    line += f" — {desc}"
                lines.append(line)
            else:
                lines.append(f"🎖️ *{u}*")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return f"*🏅 {name}*"


def render_game_status_wa(game_name: str, table_lines: Sequence[str],
                          turn_hint: str = "") -> str:
    """Current table status for WhatsApp. Never raises."""
    try:
        lines = [f"*🎮 {game_name}*", ""]
        lines.extend(str(l) for l in (table_lines or []))
        if turn_hint:
            lines.append("")
            lines.append(f"_👉 {turn_hint}_")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return f"*🎮 {game_name}*"


def render_game_menu_wa(games: Sequence[tuple[str, str]],
                       header: str = "Games") -> str:
    """Scannable game list — numbered, no buttons needed. Never raises."""
    try:
        lines = [f"*🎮 {header}*", ""]
        for i, (slug, desc) in enumerate(games or [], 1):
            lines.append(f"*{i}.* `/game {slug}` — {desc}")
        lines.append("")
        lines.append("_reply with the command to play_")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return f"*🎮 {header}*"


def render_coins_wa(name: str, coins: Any, items: Sequence[str] | None = None) -> str:
    """Wallet / inventory snapshot. Never raises."""
    try:
        lines = [f"*🪙 {name}'s wallet*", "", f"Coins: *{coins}*"]
        if items:
            lines.append("")
            lines.append("*🎒 Items*")
            for it in items:
                lines.append(f"• {it}")
        lines.append("")
        lines.append("_/shop — spend them_")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return f"*🪙 {name}*"
