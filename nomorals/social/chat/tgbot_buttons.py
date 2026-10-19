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

    # ── confirm-flow prompt (CONFIRMABLE commands: quit/forget) ──
    # handle_control's confirm gate writes "⚠️ confirm: /<kind> …"; the
    # buttons deliver "/<kind> confirm" / "/<kind> cancel", which the
    # gate resolves against the pending confirmation for this chat.
    if "⚠️ confirm: /" in t:
        import re as _re
        m = _re.search(r"⚠️ confirm: /([a-z0-9_]+)", t)
        if m:
            kind = m.group(1)
            return [[("❌ Cancel", f"{kind} cancel"),
                     ("✅ Confirm", f"{kind} confirm")]]

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
        # Rematch runs it back; Next opens the game list; Arena goes to
        # the self-improvement arena (public bot: the public-bot gate
        # redirects it with a friendly message; owner chat: it runs);
        # Expert rematches at expert difficulty.
        return [[("🔁 Rematch", "game rematch"),
                 ("⏭️ Next", "game list")],
                [("🏟️ Arena", "arena"),
                 ("💀 Expert", "game rematch expert")]]

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


# ── personal bot (USER-ONLY) — contextual keyboards ─────────────────────
# The personal bot is owner-only, so its replies carry richer keyboards
# than the gated bot's: category jumps for /help, power controls, and
# next-actions for search/task/mission/… The gated bot NEVER sees these —
# the adapter only calls :func:`personal_buttons_for` when
# ``self.owner_only`` is true (see ``TelegramBotAdapter.send``).
#
# Rules (same as the rest of the button system):
# * every callback is a real routable command — never a dead button;
# * a button whose payload would exceed the 47-byte budget is OMITTED,
#   never sent truncated (a truncated query would run the wrong query);
# * "Power On" carries NO key — the owner key must never travel in a
#   callback (visible in Telegram + logs). Tapping it routes to the
#   /power handler, which replies with usage ("provide the key").
# * replies with no sensible next action get NO keyboard (None).


def _personal_btn(label: str, data: str) -> Button | None:
    """One personal button, or None when its payload exceeds budget."""
    try:
        check_callback_data(data)
    except ValueError:
        return None
    return (label, data)


def _personal_row(*pairs: tuple[str, str]) -> list[Button]:
    """One row of personal buttons, dropping over-budget entries."""
    row: list[Button] = []
    for label, data in pairs:
        btn = _personal_btn(label, data)
        if btn is not None:
            row.append(btn)
    return row


def personal_buttons_for(text: str, last_command: str = "") -> Keyboard | None:
    """Inline keyboard for a personal-bot reply, or None.

    Keyed off the origin command (``last_command`` — the last inbound
    ``/command`` in this chat), not fragile text matching: the same reply
    text can follow different commands, and only the origin knows the
    next action. Returns None when the origin has no sensible buttons —
    the adapter then falls through to the shared ``buttons_for_text`` /
    footer patterns (game results, confirm prompts, …).
    """
    origin = (last_command or "").strip()
    kind = origin.lstrip("/").split(None, 1)[0].lower() if origin else ""
    # Callbacks are stored WITHOUT the leading slash (the adapter re-adds
    # it on tap); _last_command keeps the slash, so strip it once here.
    origin_cmd = origin.lstrip("/")
    rest = origin_cmd.split(None, 1)[1] if len(origin_cmd.split(None, 1)) > 1 else ""

    # ── /help /list — category jumps ──
    if kind in ("help", "list", "menu", "commands"):
        from .personal_commands import PERSONAL_HELP_CATEGORIES

        rows: Keyboard = []
        row: list[Button] = []
        for label, data in PERSONAL_HELP_CATEGORIES:
            btn = _personal_btn(label, data)
            if btn is None:
                continue
            row.append(btn)
            if len(row) == 2:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return rows or None

    # ── /power — controls (Power On carries NO key by design) ──
    if kind == "power":
        rows = [
            _personal_row(("⚡ Power On", "power on"), ("🔌 Power Off", "power off")),
            _personal_row(("📊 Power Status", "power status")),
        ]
        return [r for r in rows if r] or None

    # ── /search /searchdeep — retry / deepen / save query / catalog ──
    if kind in ("search", "searchdeep"):
        rows: Keyboard = []
        retry = _personal_row(("🔁 Search again", origin_cmd))
        if kind == "search" and rest:
            deep = _personal_row(("🔬 Deep research", f"searchdeep {rest}"))
            if retry and deep:
                rows.append(retry + deep)
            elif retry:
                rows.append(retry)
            elif deep:
                rows.append(deep)
        elif retry:
            rows.append(retry)
        tail_pairs: list[tuple[str, str]] = []
        if rest:
            tail_pairs.append(("💾 Save query", f"remember {rest}"))
        tail_pairs.append(("📋 Commands", "list"))
        tail = _personal_row(*tail_pairs)
        if tail:
            rows.append(tail)
        return rows or None

    # ── task / mission — status + list ──
    if kind == "task":
        row = _personal_row(("📋 Task list", "task list"), ("📋 Commands", "list"))
        return [row] if row else None
    if kind == "mission":
        row = _personal_row(
            ("📊 Mission status", "mission status"),
            ("📋 Mission list", "mission list"),
        )
        return [row] if row else None

    # ── status / providers / model — cross-links ──
    if kind == "status":
        row = _personal_row(
            ("⚡ Power status", "power status"), ("📊 Providers", "providers")
        )
        return [row] if row else None
    if kind == "providers":
        row = _personal_row(("📊 Status", "status"), ("📋 Commands", "list"))
        return [row] if row else None
    if kind == "model":
        row = _personal_row(("📊 Providers", "providers"), ("📋 Commands", "list"))
        return [row] if row else None

    # ── impossible / book — queue + examples / list + status ──
    if kind == "impossible":
        row = _personal_row(
            ("📋 Queue", "impossible queue"),
            ("📖 Examples", "impossible examples"),
        )
        return [row] if row else None
    if kind == "book":
        row = _personal_row(
            ("📋 Book list", "book list"), ("📊 Book status", "book status")
        )
        return [row] if row else None

    # ── /wisdom read — book-style paging ──
    if kind == "wisdom" and rest.startswith("read"):
        # The read reply prints its own navigation lines:
        #   ← prev: /wisdom read <work> [chapter] <page>
        #   next → /wisdom read <work> [chapter] <page>
        # Buttons re-run exactly those commands (callback data is the
        # command without the slash — the adapter re-adds it on tap and
        # routes it through the normal command pipeline). Page 1 has no
        # prev line, so it gets no Previous button; the last page has
        # no next line, so it gets no Next button. An over-budget
        # command is dropped by _personal_row, never truncated — the
        # printed command in the message body stays as the fallback.
        import re as _re

        prev = _re.search(r"← prev: (/wisdom read [^\n\r]+)", text or "")
        nxt = _re.search(r"next → (/wisdom read [^\n\r]+)", text or "")
        pairs: list[tuple[str, str]] = []
        if prev:
            pairs.append(("⬅️ Previous", prev.group(1).strip()[1:]))
        if nxt:
            pairs.append(("Next ➡️", nxt.group(1).strip()[1:]))
        if pairs:
            row = _personal_row(*pairs)
            return [row] if row else None
        return None

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


# ── public bottom menu (BotFather list) ─────────────────────────────────

def public_menu_keyboard() -> list[list[str]]:
    """Persistent bottom reply keyboard: the PUBLIC command list only.

    Button text is the command itself (``"/help"``) — tapping one sends
    it as a message. 13 rows: 5 rows of named commands + 8 rows of the
    32 public games (4 per row). Owner-only commands are never on the
    menu. Built from ``PUBLIC_COMMANDS`` so the menu can never drift
    from the grant table (the 32 games come from PUBLIC_GAME_COMMANDS;
    the 19 named rows are the fixed BotFather order).
    """
    from .control import PUBLIC_GAME_COMMANDS

    rows = [
        ["/help", "/search", "/news", "/book"],
        ["/music", "/play", "/video", "/podcast"],
        ["/tts", "/stt", "/look", "/image"],
        ["/lens", "/decode", "/dns", "/whois"],
        ["/cipher", "/wisdom", "/think", "/game"],
    ]
    games = [f"/{name}" for name in PUBLIC_GAME_COMMANDS]
    for i in range(0, len(games), 4):
        rows.append(games[i:i + 4])
    return rows

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
