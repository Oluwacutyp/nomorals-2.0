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


class ActionableMessage:
    """A text reply bundled with tappable action buttons.

    Handlers build one via :func:`actionable` and pass
    ``buttons=msg.buttons`` to ``gateway.send(...)``. Platforms without
    a button concept (WhatsApp, SMS, …) ignore the keyboard — the raw
    command text stays in the message body, tappable as plain text.
    """

    __slots__ = ("text", "buttons")

    def __init__(self, text: str, buttons: Keyboard) -> None:
        self.text = text
        self.buttons = buttons

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ActionableMessage(text={self.text[:40]!r}, buttons={self.buttons!r})"


def actionable(text: str, buttons: list[tuple[str, str]]) -> ActionableMessage:
    """Build a message with tappable inline buttons.

    ``buttons`` is ``[(label, command), …]`` — ``command`` is the chat
    command run on tap, with or without the leading slash
    (``"/proxy refresh"`` or ``"proxy refresh"``; the adapter normalizes
    it). All buttons land on a single row; pass nested lists via the
    raw ``Keyboard`` type for multi-row layouts.

    Raises :class:`ValueError` if any callback payload exceeds Telegram's
    47-byte budget (after the adapter's HMAC signature is added).
    """
    keyboard: Keyboard = []
    row: list[Button] = []
    for label, command in buttons:
        data = (command or "").strip()
        if data.startswith("/"):
            data = data[1:]
        check_callback_data(data)
        row.append((label, data))
    if row:
        keyboard.append(row)
    return ActionableMessage(text, keyboard)


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

    # ── proxy pool empty — one tap to rescrape ──
    if "pool empty — /proxy refresh to scrape+test" in t:
        return [[("🔄 Refresh pool", "proxy refresh")]]

    # ── proxy health empty — one tap to rebuild ──
    if "Tap `/proxy refresh`" in t:
        return [[("🔄 Refresh pool", "proxy refresh")]]

    # ── research proposals — approve / deny the latest ──
    if "approve: /research approve" in t and "deny: /research deny" in t:
        return [[("✅ Approve latest", "research approve latest"),
                 ("❌ Deny latest", "research deny latest")]]

    # ── build failure — jump to checkpoints to rewind/retry ──
    if "coding bot gave up after" in t or "the builder gave up after" in t:
        return [[("📋 Checkpoints", "checkpoints")]]

    # ── /play pick-list — tappable track numbers ──
    # Checked BEFORE the music-track pattern: pick-lists also start with
    # "🎵 "...".  The message ends with a "pick:<token>" marker; each
    # button fires "/play pick <token> <n>" which downloads+sends that
    # track.  Buttons are numbered to match the message lines.
    if "— pick one" in t and "pick:" in t:
        import re as _re
        m = _re.search(r"pick:([0-9a-f]{8})", t)
        if m:
            token = m.group(1)
            nums = _re.findall(r"^(\d{1,2})\.\s", t, _re.M)
            nums = nums[:8]
            if nums:
                row = [(n, f"play pick {token} {n}") for n in nums]
                # two rows of four keeps the keyboard compact
                return [row[:4], row[4:]] if len(row) > 4 else [row]

    # ── music track info — transport controls ──
    # (matches the "🎵 "Title"" track header, not error lines)
    if t.startswith("🎵 \u201c") and "couldn't make" not in t:
        return [[("⏸️ Pause", "play pause"), ("⏭️ Next", "play next")]]

    # ── tour results — one tap to re-run ──
    if "🧪 tour —" in t and "commands probed" in t:
        return [[("🔁 Re-run tour", "tour")]]

    return None


# ── generic keyboard builders ─────────────────────────────────────────────
# Button rules (verified against Bot API 10.2 practice): labels are
# verb + object and unique within a keyboard; 1-2 buttons per row for
# sentence-length labels; destructive actions last and isolated;
# callback_data is namespaced screen:action:id with IDs, never labels —
# existing messages in users' chats must keep working after a redesign.


def nav_row(*items: tuple[str, str]) -> list[Button]:
    """One navigation row, identical on every screen (rule: nav last)."""
    row: list[Button] = []
    for label, data in items:
        check_callback_data(data)
        row.append((label, data))
    return row


def paginate(page: int,
             total_pages: int,
             prefix: str,
             *,
             labels: tuple[str, str] = ("‹ Prev", "Next ›")) -> Keyboard:
    """Prev/next pagination keyboard with index-embedded callbacks.

    ``prefix`` namespaces the payload: ``f"{prefix}:page:{n}"`` — an index,
    not a label, so it stays under the 64-byte budget and survives label
    redesigns (the Aura index-embedding pattern). No buttons when there's
    only one page.
    """
    if total_pages <= 1:
        return []
    page = max(0, min(total_pages - 1, page))
    row: list[Button] = []
    if page > 0:
        data = f"{prefix}:page:{page - 1}"
        check_callback_data(data)
        row.append((labels[0], data))
    indicator = f"{page + 1}/{total_pages}"
    if page < total_pages - 1:
        data = f"{prefix}:page:{page + 1}"
        check_callback_data(data)
        row.append((indicator, f"{prefix}:noop"))
        row.append((labels[1], data))
    else:
        row.append((indicator, f"{prefix}:noop"))
    return [row]


def confirm_keyboard(action: str,
                     item_id: str = "",
                     *,
                     confirm_label: str = "✅ Confirm",
                     cancel_label: str = "❌ Cancel") -> Keyboard:
    """Destructive-action confirm/cancel. Destructive goes last, isolated —
    and the callback carries the item ID, never the item label."""
    confirm_data = f"{action}:confirm:{item_id}".rstrip(":")
    cancel_data = f"{action}:cancel:{item_id}".rstrip(":")
    check_callback_data(confirm_data)
    check_callback_data(cancel_data)
    return [[(cancel_label, cancel_data), (confirm_label, confirm_data)]]


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
