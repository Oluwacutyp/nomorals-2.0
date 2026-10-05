"""Inline keyboards and inline-query results for the Telegram BotFather bot.

This module is pure logic — no network, no Telegram imports — so it is
fully unit-testable. :class:`TelegramBotAdapter` calls into it:

* :func:`buttons_for_text` — contextual inline-keyboard buttons derived
  from an outgoing reply's text. The adapter attaches these when the
  caller did not pass explicit ``buttons``. Callback data is the command
  without the leading slash (``"game join"``); the adapter's
  ``_handle_callback`` re-adds the slash and routes it through the normal
  command pipeline (including the public-bot gate).
* :func:`inline_results_for` — ``answerInlineQuery`` result articles for
  ``@BotName <query>``. Tapping a result inserts the command text into
  the current chat; if the bot is present there it processes it normally.

Callback data budget: Telegram caps ``callback_data`` at 64 bytes and the
adapter's HMAC signature adds 17 (``|`` + 16 hex chars), so payloads must
stay at or under 47 bytes. :func:`check_callback_data` enforces this.
"""

from __future__ import annotations

#: Max callback_data bytes Telegram accepts.
CALLBACK_DATA_LIMIT = 64
#: Bytes reserved for the "|" + 16-char HMAC signature.
_SIGNATURE_OVERHEAD = 17
#: Max payload bytes before signing.
CALLBACK_PAYLOAD_LIMIT = CALLBACK_DATA_LIMIT - _SIGNATURE_OVERHEAD


def check_callback_data(data: str) -> None:
    """Raise ValueError if *data* would exceed Telegram's callback budget."""
    if len(data.encode("utf-8")) > CALLBACK_PAYLOAD_LIMIT:
        raise ValueError(
            f"callback data too long ({len(data.encode('utf-8'))} bytes, "
            f"limit {CALLBACK_PAYLOAD_LIMIT}): {data!r}"
        )


# Button = (label, callback_data). Rows are lists of buttons.
Button = tuple[str, str]
Keyboard = list[list[Button]]


def buttons_for_text(text: str) -> Keyboard | None:
    """Return contextual inline-keyboard buttons for an outgoing reply.

    Returns None when no pattern matches — the message goes out plain.
    Patterns are ordered most-specific first.
    """
    t = text or ""

    # ── game menu (/game, /game list, /game help) ──
    if "modes & mastery" in t and "/game <name>" in t:
        return [
            [("🎮 Join Game", "game join"), ("📊 Stats", "game stats")],
            [("🏆 Leaderboard", "game leaderboard")],
        ]

    # ── multiplayer game start blurb (mafia/connect4/…) ──
    if "is live — /game join to sit at the table" in t:
        return [[("🎮 Join", "game join")]]

    # ── live game indicator ──
    if "live game:" in t:
        return [[("🚪 Leave", "game quit"), ("📊 Stats", "game stats")]]

    # ── game results ──
    if "The House wins" in t or "points and coins credited" in t:
        return [[("🔁 Rematch", "game rematch"), ("📊 Stats", "game stats")]]

    # ── gift flow ──
    if t.startswith("🎁 gifting") or "\n🎁 gifting" in t:
        return [[("📜 History", "gift history")]]

    # ── generic help ──
    if "try /game list to see what's on" in t:
        return [[("🎮 Games", "game list"), ("❓ Help", "help")]]

    return None


# ── inline queries ────────────────────────────────────────────────────

def _article(article_id: str, title: str, description: str,
             message_text: str) -> dict:
    """One InlineQueryResultArticle dict."""
    return {
        "type": "article",
        "id": article_id,
        "title": title,
        "description": description,
        "input_message_content": {"message_text": message_text},
    }


#: Games offered through inline mode. (label, /command)
_INLINE_GAMES: tuple[tuple[str, str], ...] = (
    ("🎮 Mafia", "/game mafia"),
    ("🎮 Wordchain", "/game wordchain"),
    ("🎮 Hangman", "/game hangman"),
    ("🎮 Trivia", "/game trivia"),
    ("🎮 Connect4", "/game connect4"),
    ("⚔️ PvP Duel", "/pvp"),
    ("🏟️ Arena", "/arena"),
)


def inline_results_for(query: str) -> list[dict]:
    """Build ``answerInlineQuery`` results for an inline query string.

    Tapping a result inserts its command text into the current chat, so
    the bot can be used in groups where it was never added.
    """
    q = (query or "").strip().lower()

    # Empty query → starter pack.
    if not q:
        return [
            _article("iq_game", "🎮 Start a game",
                     "Mafia, wordchain, hangman, trivia, connect4…",
                     "/game"),
            _article("iq_stats", "📊 My game stats",
                     "Posts /game stats so the bot answers with your profile",
                     "/game stats"),
            _article("iq_help", "❓ What can you do?",
                     "Game list and commands", "/game list"),
        ]

    # "game <name>" or a bare game name → matching game starters.
    results: list[dict] = []
    needle = q[5:].strip() if q.startswith("game ") else q
    for label, cmd in _INLINE_GAMES:
        if needle and needle in label.lower().replace("🎮 ", "").replace("⚔️ ", "").replace("🏟️ ", ""):
            game = cmd.split()[-1] if cmd.startswith("/") else cmd
            results.append(_article(
                f"iq_{game}", f"{label} — start it here",
                f"Posts {cmd} into this chat", cmd))
    if results:
        return results

    if q in ("stats", "stat", "profile"):
        return [_article("iq_stats", "📊 My game stats",
                         "Posts /game stats so the bot answers with your profile",
                         "/game stats")]
    if q in ("help", "commands", "games"):
        return [_article("iq_help", "❓ Game commands",
                         "Posts the game list into this chat", "/game list")]
    if q in ("pvp", "duel", "fight"):
        return [_article("iq_pvp", "⚔️ PvP Duel — challenge someone",
                         "Posts /pvp into this chat", "/pvp")]

    # Fallback: offer to drop the raw query as a game command.
    return [_article("iq_fallback", f"🎮 /game {q}",
                     "Try it as a game command", f"/game {q}")]
