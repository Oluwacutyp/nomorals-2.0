"""The checked-in surface manifest (Living Chat P1, 2026-11-08).

Every interactive chat surface — menus, cards, boards, pickers,
games, pairing, downloads — is recorded here with its producer,
its actions and their real handlers, the platforms it reaches,
its fallback form, and its delivery receipt. Two consumers:

* **The drift test** (tests/test_living_chat_p1.py) compares the
  manifest SET-BASED against the live registries — hub screens
  (:data:`nomorals.social.chat.hub.SCREEN_NAMES`), adaptive
  output kinds (:data:`...adaptive_buttons.OUTPUT_KINDS`), menu
  groups (:data:`...control.LIST_GROUPS`) — so a surface added
  without a manifest update fails by naming the difference,
  never by a count. It also resolves every declared action and
  every payload on every probed real keyboard through
  ``parse_control``: an action that renders but reaches no
  handler fails the build.
* **The capability chain** (doc 8 P3): for each surface and
  platform, :func:`resolve_chain` returns the full chain
  producer → action → handler → operation → adapter → bridge →
  receipt, with the WhatsApp bridge resolved per engine (Go /
  Node) — so "why didn't this button work on WhatsApp" is
  answerable from the manifest plus the journal, not archaeology.

Action FIELD support per platform (style, custom-emoji icons,
URL/copy forms) lives in :mod:`.action_schema` — this manifest
records which FORM each surface takes per platform family, and
the schema records what the form honors.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ActionRef",
    "AdapterRoute",
    "Chain",
    "Surface",
    "SURFACES",
    "surface_by_id",
    "resolve_chain",
]


@dataclass(frozen=True)
class ActionRef:
    """One action a surface offers, and the handler it reaches."""

    action_id: str
    intent: str                 # action_schema intent
    handler: str                # "control:<kind>" — the real route
    data: str = ""              # representative callback payload


@dataclass(frozen=True)
class AdapterRoute:
    """How a surface reaches one platform: adapter + bridge."""

    platform: str
    adapter: str
    bridge: str
    engine: str = ""            # whatsapp only: "go" / "node" / ""


@dataclass(frozen=True)
class Chain:
    """One surface on one platform, end to end (doc 8 P3)."""

    producer: str
    action: str
    handler: str
    operation: str
    adapter: str
    bridge: str
    receipt: str


@dataclass(frozen=True)
class Surface:
    """One interactive surface, as recorded in the manifest."""

    id: str
    title: str
    producer: str               # "module:function" that builds it
    card_type: str              # semantic card / output type
    actions: tuple[ActionRef, ...]
    platforms: tuple[str, ...]
    routes: tuple[AdapterRoute, ...]
    operation: str              # the domain operation handlers run
    receipt: str                # how delivery is evidenced
    fallback: str               # the working fallback form
    members: tuple[str, ...] = ()   # registry-derived sub-units
    tests: tuple[str, ...] = ()


# ── shared route fragments ──────────────────────────────────────

def _tg_bot(adapter_note: str = "") -> AdapterRoute:
    return AdapterRoute(
        "telegram-bot",
        "TelegramBotAdapter (social/chat/telegram.py)",
        "Telegram Bot API (direct HTTPS)")


def _tg_personal() -> AdapterRoute:
    return AdapterRoute(
        "telegram-personal",
        "TelegramBotAdapter (social/chat/telegram.py, owner-only bot)",
        "Telegram Bot API (direct HTTPS)")


def _tg_userbot() -> AdapterRoute:
    return AdapterRoute(
        "telegram",
        "TelegramAdapter (social/chat/telegram.py, Telethon)",
        "MTProto via Telethon")


def _wa() -> tuple[AdapterRoute, AdapterRoute]:
    return (
        AdapterRoute(
            "whatsapp",
            "WhatsAppAdapter (social/chat/whatsapp.py)",
            "Go bridge — bridge/gobridge/messages.go cmdSend "
            "(whatsmeow InteractiveMessage/ListMessage)",
            engine="go"),
        AdapterRoute(
            "whatsapp",
            "WhatsAppAdapter (social/chat/whatsapp.py)",
            "Node bridge — bridge/whatsapp-bridge.mjs cmdSend "
            "(Baileys InteractiveMessage/ListMessage)",
            engine="node"),
    )


def _plain(platform: str, adapter: str) -> AdapterRoute:
    return AdapterRoute(platform, adapter, "in-process / direct API")


_RECEIPT = (
    "SendResult(ok, message_id) returned by the gateway send; "
    "failures recorded via social.chat.failures.record_failure "
    "and journaled by the adapter")

_NUMBERED = (
    "numbered action list over the operation store "
    "(agents/partner/button_picks.py): the same commands execute "
    "from a bare-number reply, in the sender's (chat, owner) slot")


def _routes(*groups) -> tuple[AdapterRoute, ...]:
    out: list[AdapterRoute] = []
    for g in groups:
        if isinstance(g, AdapterRoute):
            out.append(g)
        else:
            out.extend(g)
    return tuple(out)


_ALL_CHAT = ("telegram-bot", "telegram-personal", "telegram",
             "whatsapp", "discord", "local")
_ALL_CHAT_ROUTES = _routes(_tg_bot(), _tg_personal(), _tg_userbot(),
                           _wa(), _plain("discord",
                                         "DiscordAdapter (social/chat/discord.py)"),
                           _plain("local", "LocalAdapter (in-process)"))


# ── the manifest ────────────────────────────────────────────────

SURFACES: tuple[Surface, ...] = (
    Surface(
        id="menu",
        title="Command menu (live session card)",
        producer="nomorals.social.chat.control:menu_session_render",
        card_type="menu card (live session 'menu')",
        actions=(
            ActionRef("menu.open-section", "navigate", "control:list",
                      "list games"),
            ActionRef("menu.run-command", "callback", "control:game",
                      "game list"),
            ActionRef("menu.power", "callback", "control:power",
                      "power status"),
        ),
        platforms=_ALL_CHAT,
        routes=_ALL_CHAT_ROUTES,
        operation="control dispatch (PartnerRuntime.handle_control) "
                  "renders the section page into the one live menu "
                  "message (live_session registry)",
        receipt=_RECEIPT,
        fallback=_NUMBERED,
        # The LIST_GROUPS section titles, declared as a copy: the
        # drift test compares this set against control.LIST_GROUPS
        # and fails by name when they diverge.
        members=(
            "her day — state & control",
            "search & research",
            "wisdom keeper — corpus · history · practice",
            "building for real — code & missions",
            "media system — music · playback · video · podcast",
            "anime — lookup · schedules · pictures",
            "fun one-shots — jokes · facts · pictures · space",
            "makers & tools — cards · qr · links · bible",
            "chat & privacy — your chats · your accounts",
            "memory & thinking",
            "games — 41, DM + group, start them directly",
            "voice & vision",
            "tools & automation",
            "platform control — updates & restarts",
            "groups & communities",
            "discovery",
        ),
        tests=("tests/test_living_chat_p1.py",),
    ),
    Surface(
        id="hub",
        title="Devon Hub tool screens",
        producer="nomorals.social.chat.hub:keyboard_for_screen",
        card_type="hub screen card (live session 'hub')",
        actions=(
            ActionRef("hub.open-proxy", "navigate", "control:toolhub",
                      "toolhub proxy"),
            ActionRef("hub.open-bin", "navigate", "control:toolhub",
                      "toolhub bin"),
            ActionRef("hub.proxy-run", "callback", "control:toolhub",
                      "toolhub proxy run"),
            ActionRef("hub.back", "navigate", "control:toolhub",
                      "toolhub"),
        ),
        platforms=_ALL_CHAT,
        routes=_ALL_CHAT_ROUTES,
        operation="control dispatch runs the toolhub screen flow; "
                  "navigation edits the one live screen message",
        receipt=_RECEIPT,
        fallback=_NUMBERED,
        members=("hub", "proxy", "proxy_result", "proxy_cooldown",
                 "bin", "bin_result", "help"),
        tests=("tests/test_living_chat_p1.py",),
    ),
    Surface(
        id="personal-keyboards",
        title="Personal-bot contextual keyboards",
        producer="nomorals.social.chat.tgbot_buttons:personal_buttons_for",
        card_type="contextual keyboard over a control reply",
        actions=(
            ActionRef("personal.help-category", "navigate",
                      "control:list", "list"),
            ActionRef("personal.power-on", "callback", "control:power",
                      "power on"),
            ActionRef("personal.search-again", "refresh",
                      "control:search", "search burna boy"),
        ),
        platforms=("telegram-personal",),
        routes=_routes(_tg_personal()),
        operation="control dispatch (the personal bot is owner-only; "
                  "keyboards key off the origin command)",
        receipt=_RECEIPT,
        fallback="the shared buttons_for_text patterns still apply "
                 "when no personal keyboard matches",
        tests=("tests/test_inline_buttons.py",
               "tests/test_living_chat_p1.py"),
    ),
    Surface(
        id="adaptive-output",
        title="Adaptive output actions",
        producer="nomorals.social.chat.adaptive_buttons:adaptive_buttons_for",
        card_type="actions derived from the output itself",
        actions=(
            ActionRef("adaptive.search-again", "refresh",
                      "control:search", "search burna boy"),
            ActionRef("adaptive.schedule-rerun", "callback",
                      "control:schedule", "schedule list"),
            ActionRef("adaptive.game-rematch", "callback",
                      "control:game", "game rematch"),
        ),
        platforms=_ALL_CHAT,
        routes=_ALL_CHAT_ROUTES,
        operation="control dispatch; buttons derive from the "
                  "output's kind + payload, never the last command",
        receipt=_RECEIPT,
        fallback=_NUMBERED,
        members=("pulse_stories", "scheduled", "search", "game",
                 "memory", "music-ask", "command", "tool"),
        tests=("tests/test_living_chat_p1.py",),
    ),
    Surface(
        id="text-pattern-keyboards",
        title="Text-pattern keyboards (gated bot derivation)",
        producer="nomorals.social.chat.tgbot_buttons:buttons_for_text",
        card_type="keyboard derived from a reply's content patterns",
        actions=(
            ActionRef("pattern.confirm", "callback", "control:quit",
                      "quit confirm"),
            ActionRef("pattern.game-join", "callback", "control:game",
                      "game join"),
            ActionRef("pattern.play-pick", "choose", "control:play",
                      "play pick 0123abcd 1"),
            ActionRef("pattern.play-pause", "callback", "control:play",
                      "play pause"),
        ),
        platforms=("telegram-bot", "telegram-personal"),
        routes=_routes(_tg_bot(), _tg_personal()),
        operation="control dispatch; the Bot API adapter derives "
                  "the keyboard from the reply text at send time",
        receipt=_RECEIPT,
        fallback="plain text with the commands named in the copy",
        tests=("tests/test_inline_buttons.py",
               "tests/test_living_chat_p1.py"),
    ),
    Surface(
        id="confirm",
        title="Confirm / cancel prompts",
        producer="nomorals.social.chat.control:handle_control "
                 "(confirm gate)",
        card_type="confirmation card",
        actions=(
            ActionRef("confirm.yes", "callback", "control:quit",
                      "quit confirm"),
            ActionRef("confirm.no", "callback", "control:quit",
                      "quit cancel"),
        ),
        platforms=_ALL_CHAT,
        routes=_ALL_CHAT_ROUTES,
        operation="the confirm gate resolves the tap against the "
                  "pending confirmation for the chat, then runs or "
                  "drops the guarded command",
        receipt=_RECEIPT,
        fallback=_NUMBERED,
        tests=("tests/test_living_chat_p1.py",),
    ),
    Surface(
        id="play-picklist",
        title="/play pick-list (numbered + token buttons)",
        producer="nomorals.media.picklist:format_picklist",
        card_type="numbered pick-list with a pick:<token> marker",
        actions=(
            ActionRef("play.pick", "choose", "control:play",
                      "play pick 0123abcd 1"),
        ),
        platforms=_ALL_CHAT,
        routes=_ALL_CHAT_ROUTES,
        operation="PickCache operation: token-addressed picks "
                  "download exactly the chosen candidate; a "
                  "superseded list answers 'replaced'; bare numbers "
                  "resolve in the sender's (chat, owner) slot",
        receipt="the downloaded file is sent into the chat; the "
                "pick-list is consumed only when the send lands",
        fallback="the numbered list IS the primary form on "
                 "buttonless surfaces (reply with the number)",
        tests=("tests/test_play_picklist.py",
               "tests/test_living_chat_p1.py"),
    ),
    Surface(
        id="button-fallback",
        title="Numbered button fallback",
        producer="nomorals.agents.partner.runtime:_numbered_button_fallback",
        card_type="numbered action list (register #27)",
        actions=(
            ActionRef("fallback.pick", "choose", "control:toolhub",
                      "toolhub proxy"),
        ),
        platforms=("discord", "sms", "web", "local", "webhook",
                   "whatsapp"),
        routes=_routes(
            _plain("discord", "DiscordAdapter (social/chat/discord.py)"),
            _plain("sms", "SmsAdapter (social/chat/sms.py)"),
            _plain("web", "WebAdapter (in-process)"),
            _plain("local", "LocalAdapter (in-process)"),
            _plain("webhook", "WebhookAdapter (in-process)"),
            _wa()),
        operation="operation store (core.operations) via "
                  "ButtonPickStore: the sender's bare number "
                  "executes the same command a tap would have run",
        receipt=_RECEIPT,
        fallback="this surface IS the fallback; below it, plain "
                 "text naming the commands",
        tests=("tests/test_surface_laws.py",
               "tests/test_living_chat_p1.py"),
    ),
    Surface(
        id="games",
        title="Game cards and boards in chat",
        producer="nomorals.agents.partner.runtime_games (game handlers)",
        card_type="game card / live game board",
        actions=(
            ActionRef("game.join", "callback", "control:game",
                      "game join"),
            ActionRef("game.rematch", "callback", "control:game",
                      "game rematch"),
            ActionRef("game.stats", "callback", "control:game",
                      "game stats"),
            ActionRef("game.leaderboard", "callback", "control:game",
                      "game leaderboard"),
            ActionRef("game.quit", "callback", "control:game",
                      "game quit"),
        ),
        platforms=_ALL_CHAT,
        routes=_ALL_CHAT_ROUTES,
        operation="the games spine runs the move/match the tap "
                  "names; identity keys off the sender id",
        receipt=_RECEIPT,
        fallback=_NUMBERED,
        tests=("tests/test_game_buttons.py",
               "tests/test_living_chat_p1.py"),
    ),
    Surface(
        id="live-session",
        title="Live sessions (one evolving message)",
        producer="nomorals.social.chat.live_session:LiveSessionRegistry",
        card_type="live session render (session= declaration)",
        actions=(
            ActionRef("session.act", "callback", "control:riddle",
                      "riddle answer"),
        ),
        platforms=_ALL_CHAT,
        routes=_ALL_CHAT_ROUTES,
        operation="the flow's handler renders its next state; the "
                  "control-reply channel edits the ONE recorded "
                  "live message instead of posting anew",
        receipt="the edit verb's verdict (adapter dict / None) — a "
                "failed edit falls back to a fresh send and is "
                "never reported as an edit",
        fallback="platforms without text edit get a fresh send "
                 "per render (the pre-session behavior)",
        tests=("tests/test_living_chat_p1.py",),
    ),
    Surface(
        id="pairing",
        title="Pairing and account surfaces (/pair)",
        producer="nomorals.agents.partner.runtime_provision "
                 "(pair handlers)",
        card_type="pairing card + buttons probe",
        actions=(
            ActionRef("pair.status", "callback", "control:pair",
                      "pair"),
            ActionRef("pair.buttons-probe", "callback", "control:pair",
                      "pair buttons"),
            ActionRef("pair.saved", "callback", "control:pair",
                      "pair saved"),
        ),
        platforms=("whatsapp", "telegram-bot", "telegram-personal",
                   "telegram"),
        routes=_routes(_wa(), _tg_bot(), _tg_personal(), _tg_userbot()),
        operation="the pairing registry runs the named account "
                  "operation; the buttons probe sends REAL buttons "
                  "whose taps re-enter as pair commands, and the "
                  "bridge-reported form is journaled as evidence",
        receipt="the probe's verdict names the form the bridge "
                "actually used (interactive / list / text), "
                "recorded by the adapter",
        fallback="list form, then text action lines — the probe "
                 "exists to prove which form the account renders",
        tests=("tests/test_buttons_completion.py",
               "tests/test_living_chat_p1.py"),
    ),
    Surface(
        id="downloads",
        title="Download jobs (measured progress cards)",
        producer="nomorals.core.progress_events:ProgressTracker",
        card_type="living card (family: progress) over a download job",
        actions=(
            ActionRef("download.cancel", "callback", "control:download",
                      "download cancel a title"),
        ),
        platforms=_ALL_CHAT,
        routes=_ALL_CHAT_ROUTES,
        operation="the download job runs the measured progress "
                  "stream (searching→…→delivered); the progress "
                  "card (social.chat.living_cards, family "
                  "'progress') is the job's durable record — its "
                  "snapshots revise the ack; cancel resolves the "
                  "job by title and stops it cooperatively",
        receipt="the ack message id is the job's handle AND the "
                "card's recorded message; terminal state edits "
                "land on it and the delivered file is the receipt",
        fallback="living-card update forms "
                 "(social.chat.living_cards.CARD_UPDATE_FORMS): "
                 "edit in place where platform_limits records "
                 "text-edit support (Telegram family, WhatsApp), "
                 "an honest superseding send where it does not; "
                 "the numbered action list (" + _NUMBERED + ") "
                 "remains the buttonless form",
        tests=("tests/test_progress_events.py",
               "tests/test_living_chat_p1.py",
               "tests/test_living_chat_p2.py"),
    ),
    Surface(
        id="startup",
        title="Startup readiness card (doc 4 startup journey)",
        producer="nomorals.agents.partner.runtime:"
                 "PartnerRuntime._publish_startup_card",
        card_type="living card (family: readiness)",
        actions=(
            ActionRef("startup.status", "callback", "control:status",
                      "status"),
            ActionRef("startup.diagnose", "callback",
                      "control:doctor", "doctor"),
        ),
        platforms=_ALL_CHAT,
        routes=_ALL_CHAT_ROUTES,
        operation="boot produces ONE readiness card per owner "
                  "chat from real facts (storage, update "
                  "recovery, platforms, brain gate, scheduler, "
                  "the Batch D composition readiness map as the "
                  "agency tenant, mission resumption); verdict "
                  "ready / degraded / recovery-required; the "
                  "card is persisted before delivery and delivered "
                  "on the first owner interaction when no "
                  "transport was up at boot",
        receipt="the durable living_cards row (delivered flag + "
                "message id) is the retained startup record; "
                "delivery edits land on the recorded message",
        fallback="living-card update forms: edit in place on "
                 "text-edit surfaces, superseding send elsewhere; "
                 "an undelivered card stays persisted and is "
                 "delivered when the transport connects",
        tests=("tests/test_living_chat_p2.py",),
    ),
    Surface(
        id="restart",
        title="Restart journey card (doc 4 restart journey)",
        producer="nomorals.agents.partner.runtime:"
                 "PartnerRuntime (update branch + boot finalize)",
        card_type="living card (family: restart)",
        actions=(
            ActionRef("restart.status", "refresh", "control:status",
                      "status"),
        ),
        platforms=_ALL_CHAT,
        routes=_ALL_CHAT_ROUTES,
        operation="an update/restart opens a journey card in the "
                  "requesting chat (preflight → swap → restart); "
                  "the dying process advances it only to the "
                  "handoff; the NEW process finalizes it from "
                  "the restart-truth receipt "
                  "(os.update.consume_restart_intent: "
                  "restart_confirmed / restart_unconfirmed) — "
                  "the final state is never declared by the "
                  "process going down",
        receipt="the card's terminal revision, written by the "
                "new process and edited onto the original "
                "message where the surface supports edits",
        fallback="living-card update forms: edit in place on "
                 "text-edit surfaces, superseding send elsewhere; "
                 "the durable Notifier recovery report remains "
                 "the cross-channel record beside the card",
        tests=("tests/test_living_chat_p2.py",),
    ),
    Surface(
        id="group-boards",
        title="Group leaderboards (§14 boards)",
        producer="nomorals.social.chat.group_boards (board organ)",
        card_type="leaderboard board (living snapshot surface)",
        actions=(
            ActionRef("boards.show", "refresh", "control:boards",
                      "boards"),
        ),
        platforms=_ALL_CHAT,
        routes=_ALL_CHAT_ROUTES,
        operation="the board organ renders standings over the "
                  "living source; publishing stores a snapshot so "
                  "later renders show movement against it",
        receipt=_RECEIPT,
        fallback="plain-text board (boards are text-native)",
        tests=("tests/test_group_surface.py",
               "tests/test_living_chat_p1.py"),
    ),
    Surface(
        id="inline-query",
        title="Telegram inline-query results",
        producer="nomorals.social.chat.tgbot_buttons:inline_results_for",
        card_type="inline result article",
        actions=(
            ActionRef("inline.insert-command", "choose",
                      "control:game", "game"),
        ),
        platforms=("telegram-bot",),
        routes=_routes(_tg_bot()),
        operation="tapping a result inserts the command text into "
                  "the current chat; the bot processes it through "
                  "the normal command pipeline when present there",
        receipt="answerInlineQuery result accepted by the Bot API",
        fallback="none — inline mode is a bot-only surface",
        tests=("tests/test_inline_buttons.py",
               "tests/test_living_chat_p1.py"),
    ),
)


_BY_ID = {s.id: s for s in SURFACES}


def surface_by_id(surface_id: str) -> Surface:
    """The manifest entry for ``surface_id`` (KeyError if absent)."""
    return _BY_ID[surface_id]


def resolve_chain(surface_id: str, platform: str, *,
                  engine: str = "") -> Chain:
    """The capability chain for one surface on one platform.

    ``engine`` selects the WhatsApp bridge ("go" / "node"); with
    no engine named, the Go route is the default record and the
    node route is returned for engine="node". Raises KeyError
    for a surface or platform the manifest does not record —
    an unanswerable chain is a manifest gap, not a guess.
    """
    surface = surface_by_id(surface_id)
    plat = str(platform or "")
    route: AdapterRoute | None = None
    for r in surface.routes:
        if r.platform != plat:
            continue
        if r.engine and engine and r.engine != engine:
            continue
        route = r
        break
    if route is None:
        raise KeyError(
            f"no route for surface {surface_id!r} on {plat!r} "
            f"(engine {engine!r})")
    handlers = ", ".join(sorted({a.handler for a in surface.actions}))
    return Chain(
        producer=surface.producer,
        action=f"{len(surface.actions)} declared action(s) over "
               f"the rich action schema (social.chat.action_schema)",
        handler=handlers,
        operation=surface.operation,
        adapter=route.adapter,
        bridge=route.bridge,
        receipt=surface.receipt)
